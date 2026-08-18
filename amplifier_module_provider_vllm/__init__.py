"""
vLLM provider module for Amplifier.
Integrates with vLLM's OpenAI-compatible Responses API.
"""

__all__ = ["mount", "VLLMProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

import openai
from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.events import PROVIDER_RETRY
from amplifier_core.utils import redact_secrets
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import ToolCall
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from openai import AsyncOpenAI

from ._constants import DEFAULT_CONTEXT_WINDOW
from ._constants import DEFAULT_MAX_OUTPUT_TOKENS
from ._constants import DEFAULT_MAX_TOKENS
from ._constants import DEFAULT_MODEL
from ._constants import DEFAULT_REASONING_SUMMARY
from ._constants import DEFAULT_STREAM_IDLE_TIMEOUT
from ._constants import DEFAULT_TIMEOUT
from ._constants import DEFAULT_TRUNCATION
from ._constants import GATEWAY_WARMUP_MARKERS
from ._constants import GATEWAY_WARMUP_STATUS_CODES
from ._constants import MAX_CONTINUATION_ATTEMPTS
from ._constants import METADATA_INCOMPLETE_REASON
from ._constants import METADATA_RESPONSE_ID
from ._constants import METADATA_STATUS
from ._cost import compute_cost
from ._response_handling import convert_response_with_accumulated_output
from ._token_accounting import apply_token_accounting
from ._token_accounting import should_apply_token_accounting
from ._truncation import check_silent_input_truncation

logger = logging.getLogger(__name__)


def _is_remote_host(base_url: str | None) -> bool:
    """True when base_url points at a non-localhost vLLM endpoint.

    A "remote" vLLM is anything that is NOT loopback (localhost / 127.0.0.1
    / ::1 / etc.). Used as the single source of truth for capability
    tagging — local vs remote affects how routing matrices and downstream
    consumers reason about the deployment shape.

    Unlike Ollama Cloud (a single canonical hostname), vLLM has no
    first-party SaaS — "remote" simply means the inference engine isn't
    running on this machine. Could be a self-hosted box on the LAN, a
    cloud VM (RunPod / Modal / Anyscale / Lambda Labs), or a vLLM-backed
    API proxy. From the provider's perspective they're all the same:
    a non-loopback URL where Bearer auth might apply if api_key is set.
    """
    if not base_url:
        return False
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    # Use .hostname (not .netloc) — it handles IPv6 brackets and userinfo
    # correctly. e.g. urlparse("http://[::1]:8000/v1").hostname == "::1".
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    loopback = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    return host not in loopback


def _deep_unstringify(obj: Any) -> Any:
    """Recursively parse stringified JSON values in tool call arguments.

    Some models (e.g., Qwen3-Coder-Next) emit nested JSON arrays/objects as
    strings within tool call arguments, like:
        {"action": "create", "todos": "[{\\"content\\": \\"...\\"}, ...]"}
    instead of:
        {"action": "create", "todos": [{"content": "..."}, ...]}

    This walks the parsed dict and tries to json.loads any string values
    that look like JSON (start with '[' or '{').
    """
    if isinstance(obj, dict):
        return {k: _deep_unstringify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_unstringify(item) for item in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped and stripped[0] in ("[", "{"):
            try:
                parsed = json.loads(stripped)
                return _deep_unstringify(parsed)
            except (json.JSONDecodeError, RecursionError):
                pass
    return obj


class VLLMChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the vLLM provider."""
    config = config or {}

    # vLLM server URL from config or environment
    base_url = config.get("base_url") or os.environ.get(
        "VLLM_BASE_URL", "http://localhost:8000/v1"
    )

    # API key from config or environment (for auth proxies)
    api_key = config.get("api_key") or os.environ.get("VLLM_API_KEY", "EMPTY")

    _totals: dict[str, Any] = {"cost_usd": Decimal(0), "has_data": False}

    def _add_cost(cost: Decimal | None) -> None:
        if cost is not None:
            _totals["cost_usd"] += cost
            _totals["has_data"] = True

    provider = VLLMProvider(
        base_url=base_url,
        api_key=api_key,
        config=config,
        coordinator=coordinator,
        add_cost=_add_cost,
    )
    await coordinator.mount("providers", provider, name="vllm")
    coordinator.register_contributor(
        "session.cost",
        "provider-vllm",
        lambda: (
            {
                "cost_usd": str(_totals["cost_usd"])
                if _totals["cost_usd"] is not None
                else None
            }
            if _totals["has_data"]
            else None
        ),
    )
    logger.info(f"Mounted VLLMProvider (Responses API) at {base_url}")

    # Return cleanup function
    async def cleanup():
        await provider.close()

    return cleanup


def _build_assistant_message_item(
    content_parts: list[dict[str, Any]],
    message_id: str | None = None,
) -> dict[str, Any]:
    """Serialize assistant content as a spec-compliant Responses API message item.

    Emits the canonical ``ResponseOutputMessage`` shape ("Form 2+") used when
    assistant history is replayed in the Responses API ``input`` array. This is the
    single form every tested backend accepts -- verified on the wire against the
    OpenAI Responses API, llama.cpp's llama-server, and vLLM 0.19:

    - ``type: "message"`` is REQUIRED by llama-server: its input-item dispatch keys
      on the literal ``type`` field, so a role-only item 400s with
      "Cannot determine type of 'item'".
    - ``id`` and ``status`` are REQUIRED by vLLM: input items validate against the
      openai SDK's ``ResponseOutputMessageParam``, which marks both required, so a
      role-only (or bare ``type: message``) item raises ``pydantic.ValidationError``.
    - ``annotations: []`` on each ``output_text`` part mirrors OpenAI's own output
      items and is accepted by every backend.

    Real OpenAI is permissive and accepts looser forms, but this canonical form is
    the intersection all backends accept. Ref: OpenAI Responses API -- a "message"
    is a discriminated Item type alongside function_call / function_call_output /
    reasoning.

    Args:
        content_parts: Assistant output parts, each ``{"type": "output_text",
            "text": ...}``. ``annotations`` is filled in if absent.
        message_id: Preserved message id when available; a fresh ``msg_<hex>`` is
            synthesized otherwise (replayed-history ids need only be valid strings,
            not server-issued references).

    Returns:
        One Responses API assistant message item.
    """
    normalized: list[dict[str, Any]] = []
    for part in content_parts:
        if isinstance(part, dict):
            normalized.append(
                {
                    "type": part.get("type", "output_text"),
                    "text": part.get("text", ""),
                    "annotations": part.get("annotations", []),
                }
            )
        else:
            normalized.append(
                {"type": "output_text", "text": str(part), "annotations": []}
            )
    return {
        "type": "message",
        "id": message_id or f"msg_{uuid.uuid4().hex}",
        "role": "assistant",
        "status": "completed",
        "content": normalized,
    }


class VLLMProvider:
    """vLLM Responses API integration (OpenAI-compatible)."""

    name = "vllm"
    api_label = "vLLM"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str = "EMPTY",
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        client: AsyncOpenAI | None = None,
        add_cost=None,
    ):
        """Initialize vLLM provider with Responses API client.

        The SDK client is created lazily on first use, allowing get_info()
        to work without a running vLLM server.

        Args:
            base_url: vLLM server URL (e.g., http://192.168.128.5:8000/v1)
            api_key: API key for auth proxies (default "EMPTY" for local)
            config: Provider configuration
            coordinator: Module coordinator
            client: Pre-configured AsyncOpenAI client (for testing)
        """
        self._client: AsyncOpenAI | None = client  # Lazy init if None
        self.config = config or {}
        self.coordinator = coordinator
        self.base_url = base_url
        self.api_key = api_key
        self._add_cost = add_cost or (lambda cost: None)

        # Cache is_remote at construction so we don't re-parse the URL on
        # every property access (used in capabilities tagging, default
        # behavior decisions). The URL is the SSOT for local-vs-remote.
        self._is_remote_cached: bool = _is_remote_host(base_url)

        # Fail fast: require either base_url or a pre-built client.
        # Without this guard, instantiation silently succeeds with base_url=None and
        # the error only surfaces later when list_models() or complete() accesses self.client.
        # Raising here causes the CLI's _try_instantiate_provider() Approach 1 (no base_url)
        # to fall through to Approach 3, which correctly passes the resolved base_url.
        if self.base_url is None and self._client is None:
            raise ValueError("base_url or client must be provided for API calls")

        # Configuration with sensible defaults (from _constants.py - single source of truth)
        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        self.max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.temperature = self.config.get(
            "temperature", None
        )  # None = not sent (some models don't support it)
        self.reasoning = self.config.get(
            "reasoning", None
        )  # None = not sent (minimal|low|medium|high)
        self.reasoning_summary = self.config.get(
            "reasoning_summary", DEFAULT_REASONING_SUMMARY
        )
        self.truncation = self.config.get(
            "truncation", DEFAULT_TRUNCATION
        )  # Automatic context management
        self.enable_state = self.config.get("enable_state", False)
        self.raw = self.config.get("raw", False)  # Include raw API I/O in base events
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)

        # Inter-chunk idle timeout for streaming (seconds).
        # Resolution: config -> VLLM_STREAM_IDLE_TIMEOUT env var -> default
        # constant (same pattern as base_url/api_key). See _constants.py for
        # the default's rationale (long prefill TTFT vs never-hang-forever).
        self.stream_idle_timeout = float(
            self.config.get("stream_idle_timeout")
            or os.environ.get("VLLM_STREAM_IDLE_TIMEOUT")
            or DEFAULT_STREAM_IDLE_TIMEOUT
        )

        # Advertised model limits for downstream context managers (token budgeting).
        # vLLM does not expose context length via /v1/models, so these are
        # config-overridable per provider instance, with env-var fallbacks
        # (same config-then-env pattern as base_url/api_key in mount()).
        # NOTE: max_output_tokens is the advertised model maximum output —
        # a different concept from self.max_tokens (per-request completion cap).
        self.context_window = int(
            self.config.get("context_window")
            or os.environ.get("VLLM_CONTEXT_WINDOW")
            or DEFAULT_CONTEXT_WINDOW
        )
        self.max_output_tokens = int(
            self.config.get("max_output_tokens")
            or os.environ.get("VLLM_MAX_OUTPUT_TOKENS")
            or DEFAULT_MAX_OUTPUT_TOKENS
        )

        # Whether the operator EXPLICITLY set context_window (config key or
        # env var) rather than inheriting the DEFAULT_* fallback above.
        # _resolve_limits() needs this: an explicit value is a preference to
        # weigh against the server-reported ceiling, whereas an unset value
        # means "auto-detect from the model" -- the same semantics as
        # provider-ollama's num_ctx=0.
        self._context_window_explicit = bool(
            self.config.get("context_window") or os.environ.get("VLLM_CONTEXT_WINDOW")
        )

        # Per-model context windows read from /v1/models model cards
        # (max_model_len, or max_input_tokens behind a proxy that strips
        # it), keyed by model id. See _resolve_limits().
        self._discovered_limits: dict[str, int] = {}

        # Per-model output ceilings, keyed by model id. Direct vLLM cards
        # carry no output limit; LiteLLM-style proxies report one as
        # max_output_tokens. Only ever tightens the resolved value.
        self._discovered_output_limits: dict[str, int] = {}

        # Streaming flag — when True (default), emits llm:stream_* contract events
        # via chunked HTTP transport. Set to False to use the blocking create() path
        # (useful for background tasks like session-namer that must NOT stream).
        self.use_streaming = self.config.get("use_streaming", True)

        # Provider priority for selection (lower = higher priority)
        self.priority = self.config.get("priority", 100)

        # Retry configuration — delegates to shared retry_with_backoff() from amplifier-core.
        self._retry_config = RetryConfig(
            max_retries=int(self.config.get("max_retries", 5)),
            initial_delay=float(self.config.get("min_retry_delay", 1.0)),
            max_delay=float(self.config.get("max_retry_delay", 60.0)),
            jitter=bool(self.config.get("retry_jitter", True)),
        )

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

    @property
    def client(self) -> AsyncOpenAI:
        """Lazily initialize the vLLM client on first access."""
        if self._client is None:
            if self.base_url is None:
                raise ValueError("base_url or client must be provided for API calls")
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,  # From config, env, or "EMPTY" for local
                max_retries=0,  # Phase 2: Disable SDK retries — we handle retry ourselves
            )
        return self._client

    @staticmethod
    def _is_cloudflare_challenge(error: openai.APIStatusError) -> bool:
        """Detect Cloudflare bot-management challenge responses.

        Cloudflare interposes HTML challenge pages (HTTP 403) that look nothing
        like real API errors.  Signals:

        1. The body did not parse as a JSON object/array. (When the SDK
           cannot parse the body as JSON it stores the RAW TEXT in
           ``error.body`` -- a str, NOT None; a parsed error is a dict/list.)
        2. The Content-Type is text/html (not application/json).
        3. The raw response text contains Cloudflare markers.

        Any combination of (1 + 2) or (1 + 3) is sufficient.  If the SDK
        successfully parsed a JSON body, this is a real API error regardless
        of other signals.
        """
        # Only a PARSED JSON body (dict/list) means a genuine, structured
        # API error. When the SDK cannot parse the body as JSON it stores the
        # RAW TEXT in ``error.body`` -- a str, NOT None -- so a "body is not
        # None" guard bails on exactly the HTML challenge pages this exists to
        # catch. Fall through for a str (or absent) body; bail only on parsed
        # JSON.
        body = getattr(error, "body", None)
        if isinstance(body, (dict, list)):
            return False

        # Inspect the raw HTTP response for HTML / Cloudflare signals
        response = getattr(error, "response", None)
        if response is None:
            return False

        content_type = getattr(response, "headers", {}).get("content-type", "").lower()
        if "text/html" in content_type:
            return True

        # Fallback: scan response text for Cloudflare markers
        text = (getattr(response, "text", "") or "").lower()
        cf_markers = (
            "just a moment",
            "cf-browser-verification",
            "cloudflare",
            "checking if the site connection is secure",
        )
        return any(marker in text for marker in cf_markers)

    @staticmethod
    def _is_gateway_warmup_page(error: openai.APIStatusError) -> bool:
        """Detect a hosted front door serving a holding page while it warms up.

        A vLLM server behind a hosted-GPU gateway can answer with an HTML
        "waiting for service to respond" page instead of the API while the
        backend is still starting -- and, while the model group is loading,
        that page can arrive as a 404. That is transient and resolves once
        the backend is up, so it must not be reported as a permanently
        missing model.

        Three conditions, ALL required:

        1. The body did not parse as a JSON object/array. (The SDK stores an
           unparsed body as the raw text str in ``error.body`` -- never None;
           a parsed JSON object/array is a real API error, always.)
        2. The status is one whose default classification would otherwise be
           permanently fatal (``GATEWAY_WARMUP_STATUS_CODES``). 5xx is
           already retryable and needs no help; 400/401/403/413 are real,
           operator-fixable errors that must keep their own diagnostics.
        3. The body carries an explicit warm-up phrase
           (``GATEWAY_WARMUP_MARKERS``).

        Condition 3 is what keeps this honest. Treating "the body is HTML"
        as sufficient -- the way ``_is_cloudflare_challenge`` can, because it
        is scoped to 403 -- would swallow every permanent HTML error page a
        proxy emits: a typo'd model name returns a 404 page too, and would
        then be retried for a minute and reported as a warm-up, sending the
        operator after a fix that will never work.
        """
        # Only a parsed JSON body (dict/list) is a genuine API error. An
        # unparsed HTML holding page arrives as a str in error.body (never
        # None), so bail only on parsed JSON -- see _is_cloudflare_challenge.
        body = getattr(error, "body", None)
        if isinstance(body, (dict, list)):
            return False

        if getattr(error, "status_code", None) not in GATEWAY_WARMUP_STATUS_CODES:
            return False

        response = getattr(error, "response", None)
        if response is None:
            return False

        text = (getattr(response, "text", "") or "").lower()
        return any(marker in text for marker in GATEWAY_WARMUP_MARKERS)

    @property
    def is_remote(self) -> bool:
        """True when configured against a non-localhost vLLM endpoint.

        Returns the value cached in ``__init__`` — see :func:`_is_remote_host`
        for the URL-parse logic.
        """
        return self._is_remote_cached

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="vllm",
            display_name="vLLM",
            credential_env_vars=["VLLM_API_KEY"],
            capabilities=[
                "streaming",
                "tools",
                "reasoning",
                "remote" if self._is_remote_cached else "local",
            ],
            defaults={
                "model": self.default_model,
                "max_tokens": 16384,
                "temperature": None,
                "timeout": 600.0,
                "context_window": self.context_window,
                "max_output_tokens": self.max_output_tokens,
            },
            config_fields=[
                # base_url is the single source of truth for local-vs-remote.
                # Localhost URLs are treated as local; any other URL is treated
                # as remote (capability-tagged accordingly). To run BOTH a local
                # and a remote vLLM instance simultaneously, configure two
                # provider instances with different ``instance_id`` values
                # (see README "Mixed local + remote (multi-instance)").
                ConfigField(
                    id="base_url",
                    display_name="Server URL",
                    prompt="vLLM server URL (localhost for local; any remote URL for hosted)",
                    field_type="text",
                    env_var="VLLM_BASE_URL",
                    default="http://localhost:8000/v1",
                    required=True,
                ),
                # api_key is OPTIONAL — required only when the upstream vLLM
                # endpoint enforces Bearer auth (most hosted providers and any
                # custom auth-proxy deployment). For an unauthenticated local
                # vLLM, leave it empty (the OpenAI client tolerates "EMPTY").
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    prompt="API key (required for auth-protected endpoints; leave empty for local)",
                    field_type="secret",
                    env_var="VLLM_API_KEY",
                    default="EMPTY",
                    required=False,
                ),
                # Inter-chunk idle timeout for streaming. Bounds the wait for
                # every chunk (including the first) so a connection dropped
                # without close by a hosted-GPU proxy surfaces as a retryable
                # timeout instead of hanging the session forever (#339).
                ConfigField(
                    id="stream_idle_timeout",
                    display_name="Stream Idle Timeout",
                    prompt=(
                        "Max seconds to wait between streamed chunks before "
                        "aborting (generous: long prefill on 60-90k-token "
                        "prompts legitimately takes minutes)"
                    ),
                    field_type="text",
                    env_var="VLLM_STREAM_IDLE_TIMEOUT",
                    default=str(DEFAULT_STREAM_IDLE_TIMEOUT),
                    required=False,
                ),
                # vLLM's /v1/models model cards expose max_model_len, so
                # context_window is normally auto-discovered per model (see
                # _resolve_limits()) and this field acts as an override/cap
                # rather than the only source of truth. It still matters for
                # proxies that strip that field, and for max_output_tokens,
                # which no /v1/models response carries.
                ConfigField(
                    id="context_window",
                    display_name="Context Window",
                    prompt=(
                        "Context window in tokens advertised to the context "
                        "manager (auto-discovered from the server's model "
                        "card when available; this value acts as an "
                        "override/cap)"
                    ),
                    field_type="text",
                    env_var="VLLM_CONTEXT_WINDOW",
                    default=str(DEFAULT_CONTEXT_WINDOW),
                    required=False,
                ),
                ConfigField(
                    id="max_output_tokens",
                    display_name="Max Output Tokens",
                    prompt=(
                        "Maximum output tokens advertised to the context "
                        "manager (model maximum, not the per-request "
                        "max_tokens cap; vLLM's model cards do not expose "
                        "this, so it is never auto-discovered)"
                    ),
                    field_type="text",
                    env_var="VLLM_MAX_OUTPUT_TOKENS",
                    default=str(DEFAULT_MAX_OUTPUT_TOKENS),
                    required=False,
                ),
            ],
        )

    def get_model_info(self) -> ModelInfo:
        """Model info for the configured default model.

        Context managers (e.g. context-simple) prefer ``get_model_info()``
        over ``get_info().defaults`` when computing their token budget, so
        this must report the same resolved limits as list_models(),
        including any context window discovered by a prior list_models()
        call (see _resolve_limits()).
        """
        context_window, max_output_tokens = self._resolve_limits(self.default_model)
        return ModelInfo(
            id=self.default_model,
            display_name=self.default_model,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            capabilities=[
                "tools",
                "streaming",
                "reasoning",
                "remote" if self._is_remote_cached else "local",
            ],
            defaults={"temperature": None, "max_tokens": 16384},
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available models from vLLM server.

        vLLM serves a single model per instance, so we query
        the models endpoint to get info about the loaded model.
        Raises exception if server is unreachable (no fallback - caller handles errors).

        vLLM's model cards include ``max_model_len`` (the server's real
        context length); it is cached per model before resolving that
        model's advertised limits, so an endpoint serving several models
        with different context lengths reports each one accurately instead
        of one flat instance-level number. Servers that do not report the
        field fall back to the configured value or DEFAULT_CONTEXT_WINDOW.
        """
        # vLLM supports OpenAI-compatible /v1/models endpoint
        # Let exceptions propagate - connection errors should be shown to user
        models_response = await self.client.models.list()
        models = []
        for model in models_response.data:
            discovered, discovered_out, source = self._discover_card_limits(model)
            if discovered is not None:
                # Name the field the number came from. The bug this
                # discovery path exists to fix was a *silent* fallback to
                # DEFAULT_CONTEXT_WINDOW; an unlabelled number from a proxy
                # would be the same failure one layer up.
                if self._discovered_limits.get(model.id) != discovered:
                    logger.info(
                        "[PROVIDER] Discovered context_window=%s for %s via %s",
                        discovered,
                        model.id,
                        source,
                    )
                self._discovered_limits[model.id] = discovered
            if discovered_out is not None:
                self._discovered_output_limits[model.id] = discovered_out
            context_window, max_output_tokens = self._resolve_limits(model.id)
            models.append(
                ModelInfo(
                    id=model.id,
                    display_name=model.id,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    capabilities=[
                        "tools",
                        "streaming",
                        "reasoning",
                        "remote" if self._is_remote_cached else "local",
                    ],
                    defaults={"temperature": None, "max_tokens": 16384},
                )
            )
        return models

    async def _iter_with_idle_timeout(self, stream):
        """Yield stream events, bounding the wait between chunks.

        Hosted-GPU HTTPS proxies (RunPod et al.) drop quiet connections
        without FIN, leaving an established stream silently hung: no chunk,
        no exception, forever. This wrapper bounds the wait for EVERY chunk
        — including the FIRST (a hang before the first chunk is the same
        failure mode) — and aborts with a retryable LLMTimeoutError so that
        upstream retry logic (retry_with_backoff) engages instead of the
        session hanging indefinitely. See microsoft/amplifier#339.
        """
        iterator = stream.__aiter__()
        while True:
            try:
                event = await asyncio.wait_for(
                    iterator.__anext__(), timeout=self.stream_idle_timeout
                )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as e:
                raise kernel_errors.LLMTimeoutError(
                    f"Streaming response stalled: no chunk received for "
                    f"{self.stream_idle_timeout}s (stream_idle_timeout). "
                    f"Likely a dropped connection (proxy killed the stream "
                    f"without close). Aborting so the attempt can be retried.",
                    provider=self.name,
                    retryable=True,
                ) from e
            yield event

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        """Best-effort coercion to a positive int; None for anything else.

        Model cards are untrusted input: a missing, null, non-numeric, or
        non-positive ``max_model_len`` must degrade to "not discovered"
        rather than raise or poison the cache.
        """
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return None
        return coerced if coerced > 0 else None

    @staticmethod
    def _extract_card_int(card: Any, field: str) -> int | None:
        """Probe one extra field off a /v1/models model card.

        ``openai.types.Model`` preserves unknown fields (Pydantic
        ``extra="allow"``), reachable either as a plain attribute or via
        ``model_extra`` depending on SDK version, so both are checked. A
        missing field is a miss -- that means "the server didn't tell us",
        never an error.
        """
        try:
            value = getattr(card, field, None)
            if value is None:
                extra = getattr(card, "model_extra", None) or {}
                value = extra.get(field)
        except Exception:
            return None
        return VLLMProvider._coerce_positive_int(value)

    @staticmethod
    def _discover_card_limits(card: Any) -> tuple[int | None, int | None, str | None]:
        """Recover a model's real limits from its /v1/models card.

        Returns ``(context_window, max_output_tokens, source)`` where
        ``source`` names which field the context window came from, so the
        discovery path is visible in logs rather than being an unmarked
        number. ``None`` for a limit means "not discovered".

        Direct vLLM reports the server's real total context length as
        ``max_model_len`` and carries no output limit. OpenAI-compatible
        proxies/gateways in front of vLLM (LiteLLM-style, e.g. RunPod
        endpoints) strip ``max_model_len`` and instead report
        ``max_input_tokens`` / ``max_output_tokens``, so those are probed as
        a fallback. ``max_model_len`` wins whenever it is present.

        Note the vocabularies differ: ``max_model_len`` is total context
        (input + output) while ``max_input_tokens`` is input-only. Treating
        the latter as a total is deliberately conservative -- it can only
        under-report real capacity, never over-commit it.
        """
        context = VLLMProvider._extract_card_int(card, "max_model_len")
        source = "max_model_len" if context is not None else None
        if context is None:
            context = VLLMProvider._extract_card_int(card, "max_input_tokens")
            source = "max_input_tokens" if context is not None else None
        output = VLLMProvider._extract_card_int(card, "max_output_tokens")
        return context, output, source

    @staticmethod
    def _extract_max_model_len(card: Any) -> int | None:
        """Context window discovered from a model card, or None on a miss.

        Thin accessor over :meth:`_discover_card_limits` for callers that
        only need the context window.
        """
        context, _output, _source = VLLMProvider._discover_card_limits(card)
        return context

    def _resolve_limits(self, model_id: str) -> tuple[int, int]:
        """Resolve the (context_window, max_output_tokens) to advertise for one model.

        An explicitly configured ``context_window`` is a preference; the
        server-reported ``max_model_len`` is the ceiling of what will
        actually be accepted. Taking ``min(preference, ceiling)`` keeps a
        stale config from guaranteeing 400s on a smaller endpoint, while
        leaving the config unset lets a large endpoint be used to its full
        discovered limit -- the same "unset means auto-detect" shape as
        provider-ollama's ``num_ctx``.
        """
        ceiling = self._discovered_limits.get(model_id)

        if not self._context_window_explicit:
            cw = ceiling or self.context_window
        else:
            cw = self.context_window
            if ceiling and cw > ceiling:
                # The clamp actually bites: surface it rather than silently
                # capping context_window below what the operator configured
                # (see amplifier-module-provider-anthropic's equivalent
                # "Clamping max_tokens" warning).
                logger.warning(
                    "[PROVIDER] Clamping context_window from %s to %s for %s",
                    cw,
                    ceiling,
                    model_id,
                )
                cw = ceiling

        # Direct vLLM model cards carry no output limit, so the only
        # ceiling is the resolved context window. Cap output at HALF that
        # window rather than the whole thing: context managers budget input
        # as roughly `context_window - max_output_tokens // 2 -
        # safety_margin`, so advertising an output limit equal to the window
        # leaves no room for input at all once a small window is discovered
        # (an 8k endpoint would budget zero usable input tokens). Half
        # leaves headroom while keeping every configured value at or above a
        # 64k window untouched.
        mo = min(self.max_output_tokens, max(1, cw // 2))

        # A proxy that reports max_output_tokens gives a real server ceiling
        # the half-window heuristic can't know about. Apply it as an
        # additional clamp only -- it can tighten the advertised limit but
        # never raise it above the headroom guard above.
        out_ceiling = self._discovered_output_limits.get(model_id)
        if out_ceiling is not None and out_ceiling < mo:
            logger.info(
                "[PROVIDER] Clamping max_output_tokens from %s to server-reported %s for %s",
                mo,
                out_ceiling,
                model_id,
            )
            mo = out_ceiling

        return cw, mo

    def _maybe_warn_truncated_input(
        self,
        *,
        model: str,
        truncation: Any,
        input_tokens: int | None,
        requested_max_output_tokens: int | None,
    ) -> None:
        """Log once when truncation="auto" likely discarded part of the input.

        truncation="auto" (opt-in only -- the default is "disabled", see
        DEFAULT_TRUNCATION in _constants.py) lets vLLM silently drop input
        content that overflows the context window: HTTP 200, no error, no
        warning. This surfaces that after the fact so operators who chose
        "auto" still have visibility.

        Resolves the model's context window via the SAME per-model
        _resolve_limits() path used everywhere else in this provider
        (server-reported max_model_len, clamped by an explicitly configured
        context_window) -- no separate ceiling is introduced here. Delegates the
        actual detection rule (and its evidence-based margin) to
        check_silent_input_truncation(); this method is just the
        provider-side wiring: resolve the ceiling, check, log at most once.

        Never raises. Never modifies the response -- this is observability
        only, called after the returned ChatResponse has already been built.
        """
        context_window, _ = self._resolve_limits(model)
        message = check_silent_input_truncation(
            truncation=truncation,
            input_tokens=input_tokens,
            context_window=context_window,
            requested_max_output_tokens=requested_max_output_tokens,
        )
        if message:
            logger.warning(
                "[PROVIDER] Possible silent input truncation for model %s: %s",
                model,
                message,
            )

    def _build_continuation_input(
        self, original_input: list, accumulated_output: list
    ) -> list:
        """Build input for continuation call in stateless mode.

        Instead of using previous_response_id (requires store:true), we include
        the accumulated output in the next request's input to preserve context.
        This allows continuation to work in stateless mode.

        Per OpenAI Responses API docs: "context += response.output" - the API
        accepts output items (reasoning, message, tool_call) directly in the
        input array for continuation.

        Args:
            original_input: The original input messages from the first call
            accumulated_output: Output items accumulated from incomplete response(s)

        Returns:
            New input array with accumulated output included for continuation
        """
        # Start with original input (the conversation so far)
        continuation_input = list(original_input)

        # Convert accumulated output to assistant messages for input
        # Extract text from message blocks and reasoning summaries
        assistant_content = []

        for item in accumulated_output:
            if hasattr(item, "type"):
                item_type = item.type
                if item_type == "message":
                    # Extract text from message content
                    content = getattr(item, "content", [])
                    for content_item in content:
                        if (
                            hasattr(content_item, "type")
                            and content_item.type == "output_text"
                        ):
                            text = getattr(content_item, "text", "")
                            if text:
                                assistant_content.append(
                                    {"type": "output_text", "text": text}
                                )
                elif item_type == "reasoning":
                    # For reasoning, we can't really include it in input as text
                    # The reasoning trace is internal and not meant for reinsertion
                    # Skip for now - continuation will lose reasoning context
                    pass
                elif item_type in {"tool_call", "function_call"}:
                    # Tool calls - we'd need to include these but this is complex
                    # For now, skip - incomplete with tool calls is edge case
                    pass
            else:
                # Dictionary format
                item_type = item.get("type")
                if item_type == "message":
                    content = item.get("content", [])
                    for content_item in content:
                        if content_item.get("type") == "output_text":
                            text = content_item.get("text", "")
                            if text:
                                assistant_content.append(
                                    {"type": "output_text", "text": text}
                                )

        # If we extracted any assistant content, add as a spec-compliant message item
        if assistant_content:
            continuation_input.append(
                _build_assistant_message_item(assistant_content)
            )

        return continuation_input

    def _find_missing_tool_results(
        self, messages: list
    ) -> list[tuple[int, str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs, including
        the index of the source assistant message so callers can insert
        synthetic results at the correct position.

        Filters out tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops across LLM iterations.

        Returns:
            List of (msg_index, call_id, tool_name, tool_arguments) tuples for
            unpaired calls, where msg_index is the position of the assistant
            message that issued the tool call.
        """
        tool_calls: dict[
            str, tuple[int, str, dict]
        ] = {}  # {call_id: (msg_idx, name, args)}
        tool_results: set[str] = set()  # {call_id}

        for msg_idx, msg in enumerate(messages):
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (msg_idx, block.name, block.input)

            # Check tool messages for tool_call_id
            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        # Exclude IDs that have already been repaired to prevent infinite loops
        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_assistant_response(self):
        """Create a synthetic assistant turn to bridge tool results and a following user message.

        This is part of FM3 repair: when synthetic tool results are injected but the very
        next message is a real user message (not an assistant turn), the conversation
        violates the expected alternating structure expected by the LLM API.  Inserting
        this synthetic assistant acknowledgment restores a valid turn sequence.
        """
        from amplifier_core.message_models import Message

        return Message(
            role="assistant",
            content=(
                "[SYSTEM REPAIR: Synthetic assistant turn inserted to maintain valid "
                "conversation structure after missing tool result recovery.]"
            ),
        )

    def _create_synthetic_result(self, call_id: str, tool_name: str):
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.
        """
        from amplifier_core.message_models import Message

        return Message(
            role="tool",
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Generate completion using Responses API.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        # VALIDATE AND REPAIR: Check for missing tool results (backup safety net)
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] vLLM: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            # Group by source assistant message index so we can insert each synthetic
            # result directly after the message that issued the call, rather than
            # appending them all at the end of the list (which violates API ordering).
            by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for msg_idx, call_id, tool_name, _ in missing:
                by_msg_idx[msg_idx].append((call_id, tool_name))

            # Process in reverse index order so inserting into earlier positions
            # does not shift the indices we will use for later groups.
            for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
                synthetics = []
                for call_id, tool_name in by_msg_idx[msg_idx]:
                    synthetics.append(self._create_synthetic_result(call_id, tool_name))
                    # Track this ID so we don't detect it as missing again
                    self._repaired_tool_ids.add(call_id)

                insert_pos = msg_idx + 1
                for i, synthetic in enumerate(synthetics):
                    request.messages.insert(insert_pos + i, synthetic)

                # FM3: if the message immediately following the injected tool results
                # is a real user message, insert a synthetic assistant acknowledgment
                # to maintain valid alternating turn structure.
                next_pos = insert_pos + len(synthetics)
                if (
                    next_pos < len(request.messages)
                    and request.messages[next_pos].role == "user"
                ):
                    request.messages.insert(
                        next_pos, self._create_synthetic_assistant_response()
                    )

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        return await self._complete_chat_request(request, **kwargs)

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from ChatResponse.

        Args:
            response: Typed chat response

        Returns:
            List of tool calls from the response
        """
        if not response.tool_calls:
            return []
        return response.tool_calls

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs
    ) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.info(
            f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages"
        )
        logger.info(f"[PROVIDER] Message roles: {[m.role for m in request.messages]}")

        message_list = list(request.messages)

        # Separate messages by role
        system_msgs = [m for m in message_list if m.role == "system"]
        developer_msgs = [m for m in message_list if m.role == "developer"]
        conversation = [
            m for m in message_list if m.role in ("user", "assistant", "tool")
        ]

        logger.info(
            f"[PROVIDER] Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Combine system messages as instructions
        instructions = (
            "\n\n".join(
                m.content if isinstance(m.content, str) else "" for m in system_msgs
            )
            if system_msgs
            else None
        )

        # Convert all messages (developer + conversation) to Responses API format
        # Developer messages become XML-wrapped user messages, tools are batched
        all_messages_for_conversion = []

        # Add developer messages first
        for dev_msg in developer_msgs:
            all_messages_for_conversion.append(dev_msg.model_dump())

        # Add conversation messages
        for conv_msg in conversation:
            all_messages_for_conversion.append(conv_msg.model_dump())

        # Check for previous response metadata to preserve reasoning state across turns
        previous_response_id = None
        if message_list:
            # Look at the last assistant message for metadata
            for msg in reversed(message_list):
                if msg.role == "assistant":
                    # Check if message has our metadata
                    msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else msg
                    if isinstance(msg_dict, dict) and msg_dict.get("metadata"):
                        metadata = msg_dict["metadata"]
                        prev_id = metadata.get(METADATA_RESPONSE_ID)
                        if prev_id:
                            previous_response_id = prev_id
                            logger.info(
                                f"[PROVIDER] Found previous_response_id={prev_id} "
                                f"from last assistant message - will preserve reasoning state"
                            )
                            break

        # Convert to vLLM Responses API message format (array of message objects)
        input_messages = self._convert_messages(all_messages_for_conversion)
        logger.info(
            f"[PROVIDER] Converted {len(all_messages_for_conversion)} messages to {len(input_messages)} API messages"
        )

        # Prepare request parameters per vLLM Responses API spec
        params = {
            "model": kwargs.get("model", self.default_model),
            "input": input_messages,  # Array of message objects, same as OpenAI
        }

        # Determine store parameter early (needed for previous_response_id logic)
        store_enabled = kwargs.get("store", self.enable_state)
        params["store"] = store_enabled

        # Add previous_response_id ONLY if store is enabled (server-side state)
        # With store=False, we rely on explicit reasoning re-insertion instead
        if previous_response_id and store_enabled:
            params["previous_response_id"] = previous_response_id
            logger.debug("[PROVIDER] Using previous_response_id (store=True)")
        elif previous_response_id and not store_enabled:
            logger.debug(
                "[PROVIDER] Skipping previous_response_id (store=False). "
                "Relying on explicit reasoning re-insertion from metadata/content."
            )

        # Add instructions if provided
        if instructions:
            params["instructions"] = instructions

        if request.max_output_tokens:
            params["max_output_tokens"] = request.max_output_tokens
        elif max_tokens := kwargs.get("max_tokens", self.max_tokens):
            params["max_output_tokens"] = max_tokens

        if request.temperature is not None:
            params["temperature"] = request.temperature
        elif temperature := kwargs.get("temperature", self.temperature):
            params["temperature"] = temperature

        # Phase 2: Reasoning parameter precedence chain
        # kwargs["reasoning"] > request.reasoning_effort > config default > None
        reasoning_param = kwargs.get("reasoning", getattr(request, "reasoning", None))
        if reasoning_param is None and request.reasoning_effort:
            reasoning_param = {
                "effort": request.reasoning_effort,
                "summary": self.reasoning_summary,
            }
        if reasoning_param is None:
            reasoning_param = self.reasoning
        if reasoning_param is not None:
            # Handle both dict format ({"effort": "low", "summary": "auto"}) and string format ("low")
            if isinstance(reasoning_param, dict):
                params["reasoning"] = {
                    "effort": reasoning_param.get("effort", "medium"),
                    "summary": reasoning_param.get("summary", self.reasoning_summary),
                }
            else:
                params["reasoning"] = {
                    "effort": reasoning_param,
                    "summary": self.reasoning_summary,  # Verbosity: auto|concise|detailed
                }
            logger.info(f"[PROVIDER] Setting reasoning: {params['reasoning']}")

        # CRITICAL: Always request encrypted_content with store=False for stateless reasoning preservation
        # This is separate from reasoning effort - we need encrypted content even if effort not explicitly set
        if not store_enabled:
            params["include"] = kwargs.get("include", ["reasoning.encrypted_content"])
            logger.debug(
                "[PROVIDER] Requesting encrypted_content (store=False, enables stateless reasoning)"
            )

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)
            # Add tool-related parameters per Responses API spec
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
            params["parallel_tool_calls"] = kwargs.get("parallel_tool_calls", True)

        # Add truncation parameter for automatic context management
        if self.truncation:
            params["truncation"] = kwargs.get("truncation", self.truncation)

        logger.info(
            f"[PROVIDER] {self.api_label} API call - model: {params['model']}, has_instructions: {bool(instructions)}, tools: {len(request.tools) if request.tools else 0}"
        )

        thinking_enabled = bool(kwargs.get("extended_thinking"))
        thinking_budget = None
        if thinking_enabled:
            if "reasoning" not in params:
                params["reasoning"] = {
                    "effort": kwargs.get("reasoning_effort")
                    or self.config.get("reasoning_effort", "high"),
                    "summary": self.reasoning_summary,  # Verbosity: auto|concise|detailed
                }

            budget_tokens = (
                kwargs.get("thinking_budget_tokens")
                or self.config.get("thinking_budget_tokens")
                or 0
            )
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get(
                "thinking_budget_buffer", 1024
            )

            if budget_tokens:
                thinking_budget = budget_tokens
                target_tokens = budget_tokens + buffer_tokens
                if params.get("max_output_tokens"):
                    params["max_output_tokens"] = max(
                        params["max_output_tokens"], target_tokens
                    )
                else:
                    params["max_output_tokens"] = target_tokens

            logger.info(
                "[PROVIDER] Extended thinking enabled (effort=%s, budget=%s, buffer=%s)",
                params["reasoning"]["effort"],
                thinking_budget or "default",
                buffer_tokens,
            )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            request_payload: dict[str, Any] = {
                "provider": self.name,
                "model": params["model"],
                "message_count": len(message_list),
                "has_instructions": bool(instructions),
                "reasoning_enabled": params.get("reasoning") is not None,
                "thinking_enabled": thinking_enabled,
                "thinking_budget": thinking_budget,
            }
            if self.raw:
                request_payload["raw"] = redact_secrets(params)
            await self.coordinator.hooks.emit("llm:request", request_payload)

        start_time = time.time()

        # Per-request streaming override (does NOT mutate self.use_streaming).
        # Callers like session-namer pass metadata={"stream": False} to force
        # the blocking create() path and suppress llm:stream_* events.
        _meta = getattr(request, "metadata", None)
        _use_streaming = self.use_streaming
        if isinstance(_meta, dict) and _meta.get("stream") is False:
            _use_streaming = False

        # Hot-loop optimisation: evaluate the coordinator guard once.
        hooks_available = bool(self.coordinator and hasattr(self.coordinator, "hooks"))

        # Mutable dict shared between _do_complete (first streaming round) and the
        # continuation loop (subsequent rounds).  Keys populated on first success:
        #   request_id        – stable uuid4 for the whole logical call
        #   seq               – {block_index: next_sequence_number}
        #   block_types       – {block_index: "text"|"thinking"|"tool_use"}
        #   partial_emitted   – True once any delta was sent
        #   block_index_offset – advances after each round
        _stream_state: dict = {}

        # Call provider API with shared retry_with_backoff from amplifier-core.
        # Error translation happens inside _do_complete() so that retry_with_backoff
        # sees LLMError (and checks retryable) rather than raw SDK exceptions.

        async def _do_complete():
            """Single API call attempt with SDK → kernel error translation."""
            try:
                if _use_streaming:
                    # -------------------------------------------------------
                    # Streaming path — first round
                    # Emits llm:stream_block_start/delta/end per the provider
                    # streaming contract. ONE delta event for all content;
                    # block_type ("text"|"thinking") carried on every delta.
                    # State is persisted to _stream_state for the continuation
                    # loop to reuse the same request_id and advance block_index.
                    # -------------------------------------------------------
                    request_id = str(uuid.uuid4())
                    seq: dict[int, int] = {}
                    block_types_local: dict[int, str] = {}
                    partial_emitted = False
                    offset = _stream_state.get("block_index_offset", 0)

                    try:
                        async with asyncio.timeout(self.timeout):
                            async with self.client.responses.stream(**params) as stream:
                                async for event in self._iter_with_idle_timeout(stream):
                                    if hooks_available:
                                        et = event.type

                                        if et == "response.output_item.added":
                                            idx = event.output_index + offset
                                            item_type = getattr(
                                                event.item, "type", None
                                            )
                                            block_type = {
                                                "message": "text",
                                                "reasoning": "thinking",
                                                "function_call": "tool_use",
                                            }.get(item_type, "text")
                                            block_types_local[idx] = block_type
                                            seq[idx] = 0
                                            payload: dict[str, Any] = {
                                                "request_id": request_id,
                                                "block_index": idx,
                                                "block_type": block_type,
                                            }
                                            if block_type == "tool_use":
                                                name = getattr(event.item, "name", None)
                                                if name:
                                                    payload["name"] = name
                                            await self.coordinator.hooks.emit(
                                                "llm:stream_block_start", payload
                                            )

                                        elif et == "response.output_text.delta":
                                            text = event.delta
                                            if text:
                                                idx = event.output_index + offset
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_delta",
                                                    {
                                                        "request_id": request_id,
                                                        "block_index": idx,
                                                        "block_type": block_types_local.get(
                                                            idx, "text"
                                                        ),
                                                        "sequence": seq.get(idx, 0),
                                                        "text": text,
                                                    },
                                                )
                                                seq[idx] = seq.get(idx, 0) + 1
                                                partial_emitted = True

                                        elif et in (
                                            "response.reasoning_summary_text.delta",
                                            "response.reasoning_text.delta",
                                        ):
                                            text = event.delta
                                            if text:
                                                idx = event.output_index + offset
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_delta",
                                                    {
                                                        "request_id": request_id,
                                                        "block_index": idx,
                                                        "block_type": "thinking",
                                                        "sequence": seq.get(idx, 0),
                                                        "text": text,
                                                    },
                                                )
                                                seq[idx] = seq.get(idx, 0) + 1
                                                partial_emitted = True

                                        elif et == "response.output_item.done":
                                            idx = event.output_index + offset
                                            if idx in block_types_local:
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_end",
                                                    {
                                                        "request_id": request_id,
                                                        "block_index": idx,
                                                        "block_type": block_types_local[
                                                            idx
                                                        ],
                                                    },
                                                )

                                round_response = await stream.get_final_response()

                        # Persist state for outer continuation access
                        _stream_state["request_id"] = request_id
                        _stream_state["partial_emitted"] = partial_emitted
                        _stream_state["seq"] = seq
                        _stream_state["block_types"] = block_types_local
                        # Advance offset: next round blocks start above current max
                        if block_types_local:
                            _stream_state["block_index_offset"] = (
                                max(block_types_local.keys()) + 1
                            )
                        else:
                            _stream_state.setdefault("block_index_offset", 0)

                        return round_response

                    except Exception as e:
                        if partial_emitted and hooks_available:
                            await self.coordinator.hooks.emit(
                                "llm:stream_aborted",
                                {
                                    "request_id": request_id,
                                    "error": {
                                        "type": type(e).__name__,
                                        "msg": str(e),
                                    },
                                },
                            )
                        raise

                else:
                    # Non-streaming fallback — preserved for backward compat and
                    # for callers that pass metadata={"stream": False}.
                    return await asyncio.wait_for(
                        self.client.responses.create(**params), timeout=self.timeout
                    )
            except openai.RateLimitError as e:
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    ra_header = e.response.headers.get("retry-after")
                    if ra_header:
                        try:
                            retry_after = float(ra_header)
                        except (ValueError, TypeError):
                            pass
                # Fail-fast: if retry_after exceeds max_delay, mark non-retryable
                # so retry_with_backoff raises immediately instead of sleeping.
                retryable = True
                if (
                    retry_after is not None
                    and retry_after > self._retry_config.max_delay
                ):
                    retryable = False
                body = getattr(e, "body", None)
                msg = json.dumps(body) if body is not None else str(e)
                raise kernel_errors.RateLimitError(
                    msg,
                    provider=self.name,
                    status_code=429,
                    retryable=retryable,
                    retry_after=retry_after,
                ) from e
            except openai.AuthenticationError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body) if body is not None else str(e)
                raise kernel_errors.AuthenticationError(
                    msg,
                    provider=self.name,
                    status_code=getattr(e, "status_code", 401),
                ) from e
            except openai.BadRequestError as e:
                raw_msg = str(e).lower()
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                if (
                    "context length" in raw_msg
                    or "too many tokens" in raw_msg
                    or "maximum context" in raw_msg
                ):
                    raise kernel_errors.ContextLengthError(
                        error_msg,
                        provider=self.name,
                        status_code=400,
                    ) from e
                elif (
                    "content filter" in raw_msg
                    or "safety" in raw_msg
                    or "blocked" in raw_msg
                ):
                    raise kernel_errors.ContentFilterError(
                        error_msg,
                        provider=self.name,
                        status_code=400,
                    ) from e
                else:
                    raise kernel_errors.InvalidRequestError(
                        error_msg,
                        provider=self.name,
                        status_code=400,
                    ) from e
            except openai.APIStatusError as e:
                status = getattr(e, "status_code", 500)
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                if status == 403:
                    if self._is_cloudflare_challenge(e):
                        logger.warning(
                            "[PROVIDER] Cloudflare challenge detected (HTTP 403 "
                            "with HTML body). Treating as transient — will retry."
                        )
                        raise kernel_errors.ProviderUnavailableError(
                            "Cloudflare bot challenge (transient 403 with HTML body). "
                            "This typically resolves on retry.",
                            provider=self.name,
                            status_code=403,
                            retryable=True,
                        ) from e
                    raise kernel_errors.AccessDeniedError(
                        error_msg,
                        provider=self.name,
                        status_code=403,
                    ) from e
                if self._is_gateway_warmup_page(e):
                    logger.warning(
                        "[PROVIDER] Gateway warm-up holding page detected "
                        "(HTTP %s, non-JSON body carrying a warm-up notice). "
                        "Treating as transient — will retry while the backend "
                        "finishes starting.",
                        status,
                    )
                    raise kernel_errors.ProviderUnavailableError(
                        "Endpoint gateway is still starting the backend "
                        "(transient holding page). This typically resolves on retry.",
                        provider=self.name,
                        status_code=status,
                        retryable=True,
                    ) from e
                if status == 408:
                    if "time taken=0.0" in error_msg:
                        raise kernel_errors.LLMTimeoutError(
                            "Gateway reported an instant timeout (408 with "
                            "'time taken=0.0'): the request never reached the "
                            "backend. Retrying.",
                            provider=self.name,
                            retryable=True,
                        ) from e
                    raise kernel_errors.LLMError(
                        error_msg,
                        provider=self.name,
                        status_code=status,
                        retryable=False,
                    ) from e
                if status == 404:
                    raise kernel_errors.NotFoundError(
                        error_msg,
                        provider=self.name,
                        status_code=404,
                    ) from e
                if status >= 500:
                    raise kernel_errors.ProviderUnavailableError(
                        error_msg,
                        provider=self.name,
                        status_code=status,
                        retryable=True,
                    ) from e
                raise kernel_errors.LLMError(
                    error_msg,
                    provider=self.name,
                    status_code=status,
                    retryable=False,
                ) from e
            except asyncio.TimeoutError as e:
                raise kernel_errors.LLMTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider=self.name,
                    retryable=True,
                ) from e
            except kernel_errors.LLMError:
                raise  # Already translated, don't double-wrap
            except Exception as e:
                body = getattr(e, "body", None)
                error_msg = (
                    json.dumps(body)
                    if body is not None
                    else (str(e) or f"{type(e).__name__}: (no message)")
                )
                raise kernel_errors.LLMError(
                    error_msg,
                    provider=self.name,
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: kernel_errors.LLMError):
            """Callback invoked before each retry sleep."""
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    PROVIDER_RETRY,
                    {
                        "provider": self.name,
                        "attempt": attempt,
                        "max_retries": self._retry_config.max_retries,
                        "delay": delay,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        try:
            response = await retry_with_backoff(
                _do_complete,
                self._retry_config,
                on_retry=_on_retry,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Apply token accounting for GPT-OSS models (vLLM bug returns zeros)
            if should_apply_token_accounting(params["model"]):
                response = apply_token_accounting(params, response)

            logger.info("[PROVIDER] Received response from %s API", self.api_label)

            # Handle incomplete responses via auto-continuation
            # OpenAI Responses API may return status="incomplete" with reason like "max_output_tokens"
            # We automatically continue until complete to provide seamless experience
            accumulated_output = (
                list(response.output)
                if hasattr(response, "output") and response.output is not None
                else []
            )
            final_response = response
            continuation_count = 0

            while (
                hasattr(final_response, "status")
                and final_response.status == "incomplete"
                and continuation_count < MAX_CONTINUATION_ATTEMPTS
            ):
                continuation_count += 1

                # Extract incomplete reason for logging
                incomplete_reason = "unknown"
                if hasattr(final_response, "incomplete_details"):
                    details = final_response.incomplete_details
                    if isinstance(details, dict):
                        incomplete_reason = details.get("reason", "unknown")
                    elif hasattr(details, "reason"):
                        incomplete_reason = details.reason

                logger.info(
                    f"[PROVIDER] Response incomplete (reason: {incomplete_reason}), "
                    f"auto-continuing with previous_response_id={final_response.id} "
                    f"(continuation {continuation_count}/{MAX_CONTINUATION_ATTEMPTS})"
                )

                # Emit continuation event for observability
                if self.coordinator and hasattr(self.coordinator, "hooks"):
                    await self.coordinator.hooks.emit(
                        "provider:incomplete_continuation",
                        {
                            "provider": self.name,
                            "response_id": final_response.id,
                            "reason": incomplete_reason,
                            "continuation_number": continuation_count,
                            "max_attempts": MAX_CONTINUATION_ATTEMPTS,
                        },
                    )

                # Build continuation params using input-based pattern (stateless-compatible)
                continuation_input = self._build_continuation_input(
                    all_messages_for_conversion, accumulated_output
                )

                continue_params = {
                    "model": params["model"],
                    "input": continuation_input,
                }

                # Inherit important params if they were set
                if "instructions" in params:
                    continue_params["instructions"] = params["instructions"]
                if "max_output_tokens" in params:
                    continue_params["max_output_tokens"] = params["max_output_tokens"]
                if "temperature" in params:
                    continue_params["temperature"] = params["temperature"]
                if "reasoning" in params:
                    continue_params["reasoning"] = params["reasoning"]
                if "include" in params:
                    continue_params["include"] = params["include"]
                if "tools" in params:
                    continue_params["tools"] = params["tools"]
                    continue_params["tool_choice"] = params.get("tool_choice", "auto")
                    continue_params["parallel_tool_calls"] = params.get(
                        "parallel_tool_calls", True
                    )
                if "store" in params:
                    continue_params["store"] = params["store"]

                # Make continuation call (streaming or blocking)
                try:
                    continue_start = time.time()
                    if _use_streaming:
                        # -------------------------------------------------------
                        # Streaming continuation round
                        # Reuses request_id from _stream_state (same logical call).
                        # block_index_offset advances so renderer sees one sequence.
                        # -------------------------------------------------------
                        cont_request_id = _stream_state.get(
                            "request_id", str(uuid.uuid4())
                        )
                        cont_seq = _stream_state.get("seq", {})
                        cont_block_types = _stream_state.get("block_types", {})
                        cont_offset = _stream_state.get("block_index_offset", 0)
                        cont_partial = _stream_state.get("partial_emitted", False)

                        async with asyncio.timeout(self.timeout):
                            async with self.client.responses.stream(
                                **continue_params
                            ) as cont_stream:
                                async for event in self._iter_with_idle_timeout(
                                    cont_stream
                                ):
                                    if hooks_available:
                                        et = event.type

                                        if et == "response.output_item.added":
                                            idx = event.output_index + cont_offset
                                            item_type = getattr(
                                                event.item, "type", None
                                            )
                                            block_type = {
                                                "message": "text",
                                                "reasoning": "thinking",
                                                "function_call": "tool_use",
                                            }.get(item_type, "text")
                                            cont_block_types[idx] = block_type
                                            cont_seq[idx] = 0
                                            cont_payload: dict[str, Any] = {
                                                "request_id": cont_request_id,
                                                "block_index": idx,
                                                "block_type": block_type,
                                            }
                                            if block_type == "tool_use":
                                                cont_name = getattr(
                                                    event.item, "name", None
                                                )
                                                if cont_name:
                                                    cont_payload["name"] = cont_name
                                            await self.coordinator.hooks.emit(
                                                "llm:stream_block_start", cont_payload
                                            )

                                        elif et == "response.output_text.delta":
                                            text = event.delta
                                            if text:
                                                idx = event.output_index + cont_offset
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_delta",
                                                    {
                                                        "request_id": cont_request_id,
                                                        "block_index": idx,
                                                        "block_type": cont_block_types.get(
                                                            idx, "text"
                                                        ),
                                                        "sequence": cont_seq.get(
                                                            idx, 0
                                                        ),
                                                        "text": text,
                                                    },
                                                )
                                                cont_seq[idx] = cont_seq.get(idx, 0) + 1
                                                cont_partial = True

                                        elif et in (
                                            "response.reasoning_summary_text.delta",
                                            "response.reasoning_text.delta",
                                        ):
                                            text = event.delta
                                            if text:
                                                idx = event.output_index + cont_offset
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_delta",
                                                    {
                                                        "request_id": cont_request_id,
                                                        "block_index": idx,
                                                        "block_type": "thinking",
                                                        "sequence": cont_seq.get(
                                                            idx, 0
                                                        ),
                                                        "text": text,
                                                    },
                                                )
                                                cont_seq[idx] = cont_seq.get(idx, 0) + 1
                                                cont_partial = True

                                        elif et == "response.output_item.done":
                                            idx = event.output_index + cont_offset
                                            if idx in cont_block_types:
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_end",
                                                    {
                                                        "request_id": cont_request_id,
                                                        "block_index": idx,
                                                        "block_type": cont_block_types[
                                                            idx
                                                        ],
                                                    },
                                                )

                                final_response = await cont_stream.get_final_response()

                        # Update shared streaming state for next continuation round
                        _stream_state["partial_emitted"] = cont_partial
                        if cont_block_types:
                            _stream_state["block_index_offset"] = (
                                max(cont_block_types.keys()) + 1
                            )
                    else:
                        # Non-streaming continuation fallback
                        final_response = await asyncio.wait_for(
                            self.client.responses.create(**continue_params),
                            timeout=self.timeout,
                        )

                    continue_elapsed = int((time.time() - continue_start) * 1000)
                    elapsed_ms += continue_elapsed

                    # Accumulate output from continuation
                    if (
                        hasattr(final_response, "output")
                        and final_response.output is not None
                    ):
                        accumulated_output.extend(final_response.output)

                except Exception as e:
                    logger.error(
                        f"[PROVIDER] Continuation call {continuation_count} failed: {e}. "
                        f"Returning partial response from {continuation_count} continuation(s)"
                    )
                    break  # Return what we have so far

            # Log completion summary
            if continuation_count > 0:
                final_status = getattr(final_response, "status", "unknown")
                logger.info(
                    f"[PROVIDER] Completed after {continuation_count} continuation(s), "
                    f"final status: {final_status}, total time: {elapsed_ms}ms"
                )

            # Use the final response and accumulated output for conversion
            response = final_response

            # Convert to ChatResponse FIRST (before emitting llm:response)
            # so event usage fields come from the canonical ChatResponse
            if continuation_count > 0:
                # Use new helper for accumulated output
                chat_response = convert_response_with_accumulated_output(
                    response, accumulated_output, continuation_count, VLLMChatResponse
                )
            else:
                # Use existing conversion for normal (non-continued) responses
                chat_response = self._convert_to_chat_response(response)

            # Surface silent input truncation (truncation="auto" only -- the
            # default is "disabled", which fails loud instead). Reuses the
            # SAME canonical chat_response.usage.input_tokens the llm:response
            # event below reports -- no separate/parallel usage parsing. Runs
            # exactly once here regardless of how many continuation rounds
            # preceded this point, so at most one warning is logged per
            # response. See _truncation.py for the detection rule and its
            # evidence-based margin.
            self._maybe_warn_truncated_input(
                model=params["model"],
                truncation=params.get("truncation"),
                input_tokens=(
                    chat_response.usage.input_tokens if chat_response.usage else None
                ),
                requested_max_output_tokens=params.get("max_output_tokens"),
            )

            # Emit llm:response event using canonical usage fields from chat_response
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                event_usage: dict[str, Any] = {}
                if chat_response.usage:
                    event_usage["input_tokens"] = chat_response.usage.input_tokens
                    event_usage["output_tokens"] = chat_response.usage.output_tokens
                    if chat_response.usage.cache_read_tokens is not None:
                        event_usage["cache_read_tokens"] = (
                            chat_response.usage.cache_read_tokens
                        )
                    _cost_usd = getattr(chat_response.usage, "cost_usd", None)
                    event_usage["cost_usd"] = (
                        str(_cost_usd) if _cost_usd is not None else None
                    )
                response_event: dict[str, Any] = {
                    "provider": self.name,
                    "model": params["model"],
                    "usage": event_usage,
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                    "continuation_count": continuation_count
                    if continuation_count > 0
                    else None,
                }
                if self.raw:
                    response_event["raw"] = redact_secrets(response.model_dump())
                await self.coordinator.hooks.emit("llm:response", response_event)

            return chat_response

        except kernel_errors.LLMError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error("[PROVIDER] %s API error: %s", self.api_label, str(e))

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                        "provider": self.name,
                        "model": params["model"],
                    },
                )
            raise

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            # Ensure error message is never empty
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] %s API error: %s", self.api_label, error_msg)

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                        "provider": self.name,
                        "model": params["model"],
                    },
                )
            raise

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract plain text from content (handles both string and structured formats).

        Args:
            content: Content value (string or list of content blocks)

        Returns:
            Plain text string
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    # Extract text from various block types
                    if block.get("type") in ("text", "input_text", "output_text"):
                        text_parts.append(block.get("text", ""))
                elif (
                    hasattr(block, "type")
                    and hasattr(block, "text")
                    and block.type in ("text", "input_text", "output_text")
                ):
                    # Handle ContentBlock objects
                    text_parts.append(block.text)
            return "\n".join(text_parts) if text_parts else ""

        return str(content) if content else ""

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to vLLM Responses API format (OpenAI-compatible).

        Handles:
        - User messages: Simple string content
        - Assistant messages: Simple string content with tool calls as function_call items
        - Tool messages: Native function_call_output format for explicit correlation

        Args:
            messages: List of message dicts from ChatRequest

        Returns:
            List of vLLM-formatted message objects with proper function_call handling
        """
        openai_messages = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled via instructions parameter)
            if role == "system":
                i += 1
                continue

            # Handle tool result messages - use native function_call_output format
            if role == "tool":
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_call_id = tool_msg.get("tool_call_id")
                    tool_content = tool_msg.get("content", "")
                    tool_name = tool_msg.get("tool_name", "unknown")

                    if tool_call_id:
                        # Native format: function_call_output with call_id for explicit correlation
                        # vLLM Responses API supports this format for precise tool result matching
                        output_str = (
                            tool_content
                            if isinstance(tool_content, str)
                            else json.dumps(tool_content)
                        )
                        openai_messages.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call_id,
                                "output": output_str,
                            }
                        )
                    else:
                        # Fallback for messages without tool_call_id (legacy/compacted messages)
                        logger.warning(
                            f"Tool result missing tool_call_id for '{tool_name}', using text fallback. "
                            "This may reduce model accuracy for multi-tool scenarios."
                        )
                        openai_messages.append(
                            {
                                "role": "user",
                                "content": f"[Tool: {tool_name}]\n{tool_content}",
                            }
                        )
                    i += 1
                continue

            # Handle assistant messages
            if role == "assistant":
                reasoning_items_to_add = []  # Top-level reasoning items (before assistant message)
                function_call_items = []  # Top-level function_call items (for tool calls)
                text_parts = []  # Accumulate text for simple string content

                # Handle tool_calls field (from context storage)
                tool_calls_field = msg.get("tool_calls", [])
                for tc in tool_calls_field:
                    tc_id = tc.get("id") or tc.get("tool_call_id", "")
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("arguments") or tc.get("input", {})
                    if isinstance(tc_args, str):
                        tc_args_str = tc_args
                    else:
                        tc_args_str = json.dumps(tc_args) if tc_args else "{}"
                    if tc_id and tc_name:
                        # vLLM requires complete function_call object including status field
                        function_call_items.append(
                            {
                                "type": "function_call",
                                "call_id": tc_id,
                                "name": tc_name,
                                "arguments": tc_args_str,
                                "id": f"ft_{tc_id.replace('call_', '')}",  # Generate matching id
                                "status": None,  # vLLM requires this field
                            }
                        )

                # Handle structured content (list of blocks)
                if isinstance(content, list):
                    for block in content:
                        # Handle dict blocks (from context storage)
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                text_parts.append(block.get("text", ""))
                            elif block_type == "tool_call":
                                # Convert tool_call block to function_call item
                                tc_id = block.get("id", "")
                                tc_name = block.get("name", "")
                                tc_input = block.get("input", {})
                                if isinstance(tc_input, str):
                                    tc_args_str = tc_input
                                else:
                                    tc_args_str = (
                                        json.dumps(tc_input) if tc_input else "{}"
                                    )
                                if tc_id and tc_name:
                                    function_call_items.append(
                                        {
                                            "type": "function_call",
                                            "call_id": tc_id,
                                            "name": tc_name,
                                            "arguments": tc_args_str,
                                            "id": f"ft_{tc_id.replace('call_', '')}",
                                            "status": None,
                                        }
                                    )
                            elif block_type == "thinking":
                                # Extract reasoning text for re-insertion
                                thinking_text = block.get("thinking", "")
                                if thinking_text:
                                    # Create reasoning item with both summary and content (vLLM requires both)
                                    reasoning_item = {
                                        "type": "reasoning",  # Required: must be "reasoning"
                                        "id": f"local_{uuid.uuid4().hex[:16]}",  # Required: unique ID
                                        "summary": [  # Required: summary array
                                            {
                                                "type": "summary_text",
                                                "text": thinking_text,
                                            }
                                        ],
                                        "content": [  # Required: reasoning text content
                                            {
                                                "type": "reasoning_text",
                                                "text": thinking_text,
                                            }
                                        ],
                                    }
                                    reasoning_items_to_add.append(reasoning_item)
                        elif hasattr(block, "type"):
                            # Handle ContentBlock objects (TextBlock, ThinkingBlock, ToolCallBlock, etc.)
                            if block.type == "text":
                                text_parts.append(block.text)
                            elif block.type == "tool_call":
                                # Convert ToolCallBlock to function_call item
                                tc_id = getattr(block, "id", "")
                                tc_name = getattr(block, "name", "")
                                tc_input = getattr(block, "input", {})
                                if isinstance(tc_input, str):
                                    tc_args_str = tc_input
                                else:
                                    tc_args_str = (
                                        json.dumps(tc_input) if tc_input else "{}"
                                    )
                                if tc_id and tc_name:
                                    function_call_items.append(
                                        {
                                            "type": "function_call",
                                            "call_id": tc_id,
                                            "name": tc_name,
                                            "arguments": tc_args_str,
                                            "id": f"ft_{tc_id.replace('call_', '')}",
                                            "status": None,
                                        }
                                    )
                            elif block.type == "thinking" and hasattr(
                                block, "thinking"
                            ):
                                # Extract reasoning text for re-insertion
                                thinking_text = block.thinking
                                if thinking_text:
                                    # Create reasoning item with both summary and content (vLLM requires both)
                                    reasoning_item = {
                                        "type": "reasoning",  # Required: must be "reasoning"
                                        "id": f"local_{uuid.uuid4().hex[:16]}",  # Required: unique ID
                                        "summary": [  # Required: summary array
                                            {
                                                "type": "summary_text",
                                                "text": thinking_text,
                                            }
                                        ],
                                        "content": [  # Required: reasoning text content
                                            {
                                                "type": "reasoning_text",
                                                "text": thinking_text,
                                            }
                                        ],
                                    }
                                    reasoning_items_to_add.append(reasoning_item)
                # Handle simple string content
                elif isinstance(content, str) and content:
                    text_parts.append(content)

                # Add reasoning items as TOP-LEVEL entries (before function_calls and text)
                for reasoning_item in reasoning_items_to_add:
                    openai_messages.append(reasoning_item)
                    logger.debug(
                        f"[PROVIDER] Added reasoning item: id={reasoning_item['id']}, "
                        f"summary_len={len(reasoning_item['summary'][0]['text'])} chars, "
                        f"content_len={len(reasoning_item['content'][0]['text'])} chars"
                    )

                # Add function_call items as TOP-LEVEL entries (after reasoning, before text)
                for fc_item in function_call_items:
                    openai_messages.append(fc_item)
                    logger.debug(
                        f"[PROVIDER] Added function_call item: call_id={fc_item['call_id']}, name={fc_item['name']}"
                    )

                # Only add assistant message if there's text content
                if text_parts:
                    combined_text = "\n".join(text_parts)
                    openai_messages.append(
                        _build_assistant_message_item(
                            [{"type": "output_text", "text": combined_text}]
                        )
                    )

                i += 1

            # Handle developer messages as XML-wrapped user messages
            elif role == "developer":
                text_content = self._extract_text_from_content(content)
                wrapped = f"<context_file>\n{text_content}\n</context_file>"
                openai_messages.append({"role": "user", "content": wrapped})
                i += 1

            # Handle user messages
            elif role == "user":
                # Extract text content as simple string
                text_content = self._extract_text_from_content(content)
                if text_content:
                    openai_messages.append({"role": "user", "content": text_content})
                i += 1
            else:
                # Unknown role - skip
                logger.warning(f"Unknown message role: {role}")
                i += 1

        return openai_messages

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to vLLM Responses API format.

        vLLM Responses API uses flat tool structure (not nested under 'function'):
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of vLLM-formatted tool definitions (flat structure)
        """
        vllm_tools = []
        for tool in tools:
            vllm_tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters,
                }
            )
        return vllm_tools

    def _convert_to_chat_response(self, response: Any) -> ChatResponse:
        """Convert vLLM response to ChatResponse format.

        Args:
            response: OpenAI API response

        Returns:
            ChatResponse with content blocks
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []
        tool_calls = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        text_accumulator: list[str] = []

        # Parse output blocks
        for block in response.output or []:
            # Handle both SDK objects and dictionaries
            if hasattr(block, "type"):
                block_type = block.type

                if block_type == "message":
                    # Extract text from message content
                    block_content = getattr(block, "content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if (
                                hasattr(content_item, "type")
                                and content_item.type == "output_text"
                            ):
                                text = getattr(content_item, "text", "")
                                content_blocks.append(TextBlock(text=text))
                                text_accumulator.append(text)
                                event_blocks.append(
                                    TextContent(
                                        text=text,
                                        raw=getattr(content_item, "raw", None),
                                    )
                                )
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))
                        text_accumulator.append(block_content)
                        event_blocks.append(TextContent(text=block_content))

                elif block_type == "reasoning":
                    # Extract reasoning text from content field
                    # vLLM provides actual reasoning text here (not just summary like OpenAI!)
                    reasoning_text = None
                    block_content = getattr(block, "content", None)

                    if block_content and isinstance(block_content, list):
                        # Extract reasoning_text items from content array
                        texts = []
                        for item in block_content:
                            if (
                                isinstance(item, dict)
                                and item.get("type") == "reasoning_text"
                            ):
                                texts.append(item.get("text", ""))
                            elif (
                                hasattr(item, "type") and item.type == "reasoning_text"
                            ):  # type: ignore[union-attr]
                                texts.append(getattr(item, "text", ""))
                        if texts:
                            reasoning_text = "\n".join(texts)

                    # Create thinking block with reasoning text (displays to user)
                    if reasoning_text:
                        thinking_block = ThinkingBlock(
                            thinking=reasoning_text,
                            signature=None,
                            visibility="internal",
                            content=None,  # Simple: no encryption state needed
                        )
                        logger.info(
                            f"[PROVIDER] Created ThinkingBlock from content field: text_len={len(reasoning_text)}"
                        )
                        content_blocks.append(thinking_block)
                        event_blocks.append(ThinkingContent(text=reasoning_text))
                        # NOTE: Do NOT add reasoning to text_accumulator - it's internal process, not response content

                elif block_type in {"tool_call", "function_call"}:
                    tool_id = getattr(block, "id", "") or getattr(block, "call_id", "")
                    tool_name = getattr(block, "name", "")
                    tool_input = getattr(block, "input", None)
                    if tool_input is None and hasattr(block, "arguments"):
                        tool_input = block.arguments
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except json.JSONDecodeError:
                            logger.debug(
                                "Failed to decode tool call arguments: %s", tool_input
                            )
                    if tool_input is None:
                        tool_input = {}
                    # Ensure tool_input is dict after json.loads or default
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    # Repair stringified nested JSON (common with Qwen3-Coder-Next)
                    tool_input = _deep_unstringify(tool_input)
                    content_blocks.append(
                        ToolCallBlock(id=tool_id, name=tool_name, input=tool_input)
                    )
                    tool_calls.append(
                        ToolCall(id=tool_id, name=tool_name, arguments=tool_input)
                    )
            else:
                # Dictionary format
                block_type = block.get("type")

                if block_type == "message":
                    block_content = block.get("content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                content_blocks.append(TextBlock(text=text))
                                text_accumulator.append(text)
                                event_blocks.append(
                                    TextContent(text=text, raw=content_item)
                                )
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))
                        text_accumulator.append(block_content)
                        event_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    # Extract reasoning text from content field
                    # vLLM provides actual reasoning text here (not just summary like OpenAI!)
                    reasoning_text = None
                    block_content = block.get("content")

                    if block_content and isinstance(block_content, list):
                        # Extract reasoning_text items from content array
                        texts = []
                        for item in block_content:
                            if (
                                isinstance(item, dict)
                                and item.get("type") == "reasoning_text"
                            ):
                                texts.append(item.get("text", ""))
                        if texts:
                            reasoning_text = "\n".join(texts)

                    # Create thinking block with reasoning text (displays to user)
                    if reasoning_text:
                        thinking_block = ThinkingBlock(
                            thinking=reasoning_text,
                            signature=None,
                            visibility="internal",
                            content=None,  # Simple: no encryption state needed
                        )
                        logger.info(
                            f"[PROVIDER] Created ThinkingBlock from content field: text_len={len(reasoning_text)}"
                        )
                        content_blocks.append(thinking_block)
                        event_blocks.append(ThinkingContent(text=reasoning_text))
                        # NOTE: Do NOT add reasoning to text_accumulator - it's internal process, not response content

                elif block_type in {"tool_call", "function_call"}:
                    tool_id = block.get("id") or block.get("call_id", "")
                    tool_name = block.get("name", "")
                    tool_input = block.get("input")
                    if tool_input is None:
                        tool_input = block.get("arguments", {})
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except json.JSONDecodeError:
                            logger.debug(
                                "Failed to decode tool call arguments: %s", tool_input
                            )
                    if tool_input is None:
                        tool_input = {}
                    # Ensure tool_input is dict after json.loads or default
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    content_blocks.append(
                        ToolCallBlock(id=tool_id, name=tool_name, input=tool_input)
                    )
                    tool_calls.append(
                        ToolCall(id=tool_id, name=tool_name, arguments=tool_input)
                    )
                    event_blocks.append(
                        ToolCallContent(
                            id=tool_id, name=tool_name, arguments=tool_input, raw=block
                        )
                    )

        # Extract usage counts
        usage_obj = response.usage if hasattr(response, "usage") else None
        usage_counts = {"input": 0, "output": 0, "total": 0}
        if usage_obj:
            if hasattr(usage_obj, "input_tokens"):
                usage_counts["input"] = usage_obj.input_tokens
            if hasattr(usage_obj, "output_tokens"):
                usage_counts["output"] = usage_obj.output_tokens
            usage_counts["total"] = usage_counts["input"] + usage_counts["output"]

        # Phase 2: Extract reasoning_tokens from output_tokens_details
        reasoning_tokens = None
        if usage_obj and hasattr(usage_obj, "output_tokens_details"):
            details = usage_obj.output_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                reasoning_tokens = details.reasoning_tokens

        # Extract cache_read_tokens from input_tokens_details
        cache_read_tokens = None
        if usage_obj and hasattr(usage_obj, "input_tokens_details"):
            details = usage_obj.input_tokens_details
            if details and hasattr(details, "cached_tokens"):
                cache_read_tokens = details.cached_tokens  # 0 is a valid measurement

        usage = Usage(
            input_tokens=usage_counts["input"],
            output_tokens=usage_counts["output"],
            total_tokens=usage_counts["total"],
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
        )

        # Stamp cost_usd — vLLM is self-hosted so cost is always indeterminate (None).
        model_id = getattr(response, "model", "")
        cost = compute_cost(
            model_id,
            input_tokens=usage_counts["input"],
            output_tokens=usage_counts["output"],
        )
        usage = usage.model_copy(update={"cost_usd": cost})
        self._add_cost(cost)

        combined_text = "\n\n".join(text_accumulator).strip()

        # Build metadata with provider-specific state
        metadata = {}

        # Response ID (for next turn's previous_response_id)
        if hasattr(response, "id"):
            metadata[METADATA_RESPONSE_ID] = response.id

        # Status (completed/incomplete)
        if hasattr(response, "status"):
            metadata[METADATA_STATUS] = response.status

            # If incomplete, record the reason
            if response.status == "incomplete":
                incomplete_details = getattr(response, "incomplete_details", None)
                if incomplete_details:
                    if isinstance(incomplete_details, dict):
                        metadata[METADATA_INCOMPLETE_REASON] = incomplete_details.get(
                            "reason"
                        )
                    elif hasattr(incomplete_details, "reason"):
                        metadata[METADATA_INCOMPLETE_REASON] = incomplete_details.reason

        # DEBUG: Log what we're returning
        logger.info(
            f"[PROVIDER] Returning ChatResponse with {len(content_blocks)} content blocks"
        )
        for i, block in enumerate(content_blocks):
            block_type = block.type if hasattr(block, "type") else "unknown"
            has_content = hasattr(block, "content") and block.content is not None
            logger.info(
                f"[PROVIDER]   Block {i}: type={block_type}, has_content_field={has_content}"
            )

        chat_response = VLLMChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=getattr(response, "finish_reason", None),
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
            metadata=metadata if metadata else None,
        )

        return chat_response

    async def close(self) -> None:
        """Close the underlying OpenAI-compatible client to prevent resource leaks."""
        if self._client is not None:
            await self._client.close()
            self._client = None
