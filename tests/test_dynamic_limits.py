"""Tests for per-model context/output limit discovery.

vLLM's /v1/models model cards expose ``max_model_len`` (the server's real,
per-model context length) via extra fields the OpenAI SDK's Model schema
doesn't declare but preserves. These tests verify that discovery and the
config-vs-discovery precedence in ``VLLMProvider._resolve_limits()``
combine to advertise accurate per-model limits -- instead of stamping one
flat instance-level number on every model -- while still degrading cleanly
to the static-config behavior covered by test_context_limits.py when
nothing is discovered.

Regression context: list_models() previously stamped the same
self.context_window / self.max_output_tokens onto every model returned by
/v1/models, with a comment claiming "vLLM doesn't expose this" -- wrong for
direct vLLM, whose model cards DO include max_model_len. An endpoint
serving multiple models with different real limits (e.g. glm-5.2=131072,
qwen3-coder-30b=262144) was forced to report a single flat number for all
of them.
"""

import logging
from openai.types import Model
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_module_provider_vllm import VLLMProvider
from amplifier_module_provider_vllm._constants import DEFAULT_CONTEXT_WINDOW

BASE_URL = "http://localhost:8000/v1"


def _model_card(model_id: str, **extra) -> SimpleNamespace:
    """Build a fake /v1/models card.

    Extra kwargs simulate vendor fields (e.g. max_model_len) that
    openai.types.Model preserves via Pydantic's extra="allow" on a real
    response; SimpleNamespace exposes them as plain attributes, which is
    all _extract_max_model_len()'s getattr probe needs.
    """
    return SimpleNamespace(id=model_id, **extra)


def _mock_client(*cards: SimpleNamespace) -> MagicMock:
    """Build a mock AsyncOpenAI client whose models.list() is awaitable."""
    client = MagicMock()
    client.models.list = AsyncMock(return_value=SimpleNamespace(data=list(cards)))
    return client


# ---------------------------------------------------------------------------
# 1, 2, 3, 4, 5, 9: list_models() discovery, precedence, and clamping
# ---------------------------------------------------------------------------


class TestRealSDKModelObject:
    """Discovery must survive the REAL openai.types.Model, not just a stub.

    Every other test here builds cards as SimpleNamespace, where getattr
    trivially succeeds for any kwarg. That proves nothing about the object
    the SDK actually hands us: openai.types.Model declares a fixed schema,
    and max_model_len is a vLLM vendor extension outside it. If the SDK ever
    stopped preserving unknown fields, discovery would silently never fire
    and every SimpleNamespace-based test here would still pass. This pins
    the premise against the real type.

    vLLM has declared max_model_len on ModelCard continuously (verified on
    v0.6.0, v0.9.0, v0.11.0), so the server side of the contract is stable;
    this guards the client side.
    """

    # Realistic direct-vLLM /v1/models card, vendor fields included.
    RAW = {
        "id": "zai-org/GLM-4.6",
        "object": "model",
        "created": 1753900000,
        "owned_by": "vllm",
        "root": "zai-org/GLM-4.6",
        "parent": None,
        "max_model_len": 131072,
    }

    def test_construct_path_preserves_max_model_len(self):
        """The SDK builds responses via construct() -- extras must survive it."""
        card = Model.construct(**self.RAW)
        assert VLLMProvider._extract_max_model_len(card) == 131072

    def test_validate_path_preserves_max_model_len(self):
        """Strict validation must preserve extras too (pydantic extra="allow")."""
        card = Model.model_validate(self.RAW)
        assert VLLMProvider._extract_max_model_len(card) == 131072

    def test_card_without_max_model_len_is_a_miss_not_an_error(self):
        """A proxy that strips vendor fields degrades to "not discovered"."""
        raw = {k: v for k, v in self.RAW.items() if k != "max_model_len"}
        assert VLLMProvider._extract_max_model_len(Model.construct(**raw)) is None

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("131072", 131072),
            (131072.9, 131072),
            (0, None),
            (-5, None),
            (None, None),
            ("abc", None),
        ],
    )
    def test_malformed_values_never_raise(self, value, expected):
        """Non-positive/garbage max_model_len means "unknown", never an exception."""
        card = Model.construct(**{**self.RAW, "max_model_len": value})
        assert VLLMProvider._extract_max_model_len(card) == expected


class TestProxyMaxInputTokensFallback:
    """Proxies (LiteLLM-style, e.g. RunPod) strip max_model_len and report
    max_input_tokens / max_output_tokens instead; discovery must fall back
    to max_input_tokens rather than silently defaulting to
    DEFAULT_CONTEXT_WINDOW (observed: Qwen3-Coder-30B's real 262144 window
    halved to 128000 behind such a proxy)."""

    # Realistic proxied /v1/models card: no max_model_len, proxy limit
    # fields instead.
    RAW = {
        "id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "object": "model",
        "created": 1753900000,
        "owned_by": "vllm",
        "max_input_tokens": 262144,
        "max_output_tokens": 262144,
    }

    def test_construct_path_falls_back_to_max_input_tokens(self):
        card = Model.construct(**self.RAW)
        assert VLLMProvider._extract_max_model_len(card) == 262144

    def test_validate_path_falls_back_to_max_input_tokens(self):
        card = Model.model_validate(self.RAW)
        assert VLLMProvider._extract_max_model_len(card) == 262144

    def test_max_model_len_wins_over_max_input_tokens(self):
        """When both are present, the server's own max_model_len is
        authoritative; the proxy field is only a fallback."""
        card = Model.construct(**{**self.RAW, "max_model_len": 131072})
        assert VLLMProvider._extract_max_model_len(card) == 131072

    def test_plain_attribute_fallback(self):
        """SimpleNamespace path: attribute probe without model_extra."""
        card = SimpleNamespace(id="glm-5.2", max_input_tokens=131072)
        assert VLLMProvider._extract_max_model_len(card) == 131072

    def test_model_extra_fallback(self):
        """model_extra path: SDK versions exposing extras only via dict."""
        card = SimpleNamespace(id="glm-5.2", model_extra={"max_input_tokens": 131072})
        assert VLLMProvider._extract_max_model_len(card) == 131072

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("262144", 262144),
            (0, None),
            (-5, None),
            (None, None),
            ("abc", None),
        ],
    )
    def test_malformed_max_input_tokens_never_raise(self, value, expected):
        """The fallback field goes through the same coercion/positivity
        validation as max_model_len: garbage means "unknown", never an
        exception."""
        card = Model.construct(**{**self.RAW, "max_input_tokens": value})
        assert VLLMProvider._extract_max_model_len(card) == expected

    @pytest.mark.asyncio
    async def test_list_models_discovers_via_max_input_tokens(self):
        """End-to-end: a proxied card with only max_input_tokens must not
        fall back to DEFAULT_CONTEXT_WINDOW."""
        provider = VLLMProvider(
            client=_mock_client(
                _model_card("qwen3-coder-30b", max_input_tokens=262144)
            ),
            config={},
        )
        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].context_window == 262144
        assert models[0].context_window != DEFAULT_CONTEXT_WINDOW


class TestListModelsDiscovery:
    """list_models() feeds each card's max_model_len into discovery and
    resolves per-model limits instead of stamping the instance-level flat
    self.context_window / self.max_output_tokens on every model."""

    @pytest.mark.asyncio
    async def test_discovers_context_window_from_model_card(self):
        """1. Model card exposes max_model_len, no config -> discovered value wins."""
        provider = VLLMProvider(
            client=_mock_client(_model_card("glm-5.2", max_model_len=131072)),
            config={},
        )
        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].context_window == 131072

    @pytest.mark.asyncio
    async def test_two_models_different_limits_reported_independently(self):
        """2. One endpoint, two models, two different max_model_len -> each
        ModelInfo reports its own value (not one flat instance-level number)."""
        provider = VLLMProvider(
            client=_mock_client(
                _model_card("glm-5.2", max_model_len=131072),
                _model_card("qwen3-coder-30b", max_model_len=262144),
            ),
            config={},
        )
        models = await provider.list_models()
        by_id = {m.id: m for m in models}

        assert by_id["glm-5.2"].context_window == 131072
        assert by_id["qwen3-coder-30b"].context_window == 262144

    @pytest.mark.asyncio
    async def test_config_clamped_to_discovered_ceiling(self):
        """3. Card reports 131072, config asks for 200000 -> clamped to
        131072 (discovery is the ceiling; config can't guarantee a 400)."""
        provider = VLLMProvider(
            client=_mock_client(_model_card("glm-5.2", max_model_len=131072)),
            config={"context_window": 200000},
        )
        models = await provider.list_models()

        assert models[0].context_window == 131072

    @pytest.mark.asyncio
    async def test_config_preference_below_ceiling_is_honored(self):
        """4. Card reports 262144, config asks for 100000 -> operator
        preference wins (config may always ask for less than the ceiling)."""
        provider = VLLMProvider(
            client=_mock_client(_model_card("qwen3-coder-30b", max_model_len=262144)),
            config={"context_window": 100000},
        )
        models = await provider.list_models()

        assert models[0].context_window == 100000

    @pytest.mark.asyncio
    async def test_no_card_field_no_config_falls_back_to_default(self):
        """5. Regression guard: nothing discovered, nothing configured ->
        DEFAULT_CONTEXT_WINDOW (128000), same as pre-discovery behavior."""
        provider = VLLMProvider(
            client=_mock_client(_model_card("plain-model")),
            config={},
        )
        models = await provider.list_models()

        assert models[0].context_window == DEFAULT_CONTEXT_WINDOW == 128000

    @pytest.mark.asyncio
    async def test_max_output_tokens_leaves_input_headroom_on_small_window(self):
        """9. A small discovered context_window caps max_output_tokens to a
        quarter of it. Context managers budget input as roughly
        `context_window - max_output_tokens // 2 - safety_margin`, so
        advertising an output limit equal to the window would leave no room
        for input at all."""
        provider = VLLMProvider(
            client=_mock_client(_model_card("small-model", max_model_len=8192)),
            config={},
        )
        models = await provider.list_models()

        assert models[0].context_window == 8192
        assert models[0].max_output_tokens == 4096  # capped from the DEFAULT (32768)
        # context-simple's budget formula must still leave usable input room.
        budget = 8192 - models[0].max_output_tokens // 2 - 4096
        assert budget > 0

    @pytest.mark.asyncio
    async def test_max_output_tokens_uncapped_on_large_window(self):
        """A window large enough that half of it exceeds the configured
        max_output_tokens leaves that value untouched."""
        provider = VLLMProvider(
            client=_mock_client(_model_card("glm-5.2", max_model_len=131072)),
            config={},
        )
        models = await provider.list_models()

        assert models[0].max_output_tokens == 32768  # the DEFAULT, unchanged


# ---------------------------------------------------------------------------
# 10: malformed values never raise and degrade cleanly
# ---------------------------------------------------------------------------


class TestMalformedValuesDegradeCleanly:
    """None / non-numeric / non-positive values on a model card must never
    raise and must fall back to the configured value or the default
    instead of corrupting state."""

    @pytest.mark.parametrize("bad_value", [None, "abc", -1, 0])
    def test_extract_max_model_len_ignores_malformed_values(self, bad_value):
        """10. Malformed max_model_len values resolve to None, not an error."""
        card = SimpleNamespace(id="weird-model", max_model_len=bad_value)

        assert VLLMProvider._extract_max_model_len(card) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_value", [None, "abc", -1, 0])
    async def test_list_models_falls_back_cleanly_for_malformed_card(self, bad_value):
        """10. list_models() end-to-end: a malformed card value never raises
        and the model still gets a sane (default) context_window."""
        provider = VLLMProvider(
            client=_mock_client(_model_card("weird-model", max_model_len=bad_value)),
            config={},
        )
        models = await provider.list_models()

        assert models[0].context_window == DEFAULT_CONTEXT_WINDOW

    def test_extract_max_model_len_handles_missing_attributes(self):
        """Card objects with neither max_model_len nor model_extra degrade
        to None instead of raising AttributeError."""
        card = SimpleNamespace(id="plain-model")

        assert VLLMProvider._extract_max_model_len(card) is None

    def test_extract_max_model_len_reads_model_extra_fallback(self):
        """Some SDK versions only surface unknown vendor fields via
        model_extra rather than as a direct attribute."""
        card = SimpleNamespace(id="glm-5.2", model_extra={"max_model_len": 131072})

        assert VLLMProvider._extract_max_model_len(card) == 131072


# ---------------------------------------------------------------------------
# 11: clamp visibility -- an operator-configured preference that gets
# clamped to a lower discovered ceiling must be logged, not silent.
# ---------------------------------------------------------------------------


class TestClampWarning:
    """A clamped preference (pref > discovered ceiling) must be visible via
    a logger.warning, mirroring amplifier-module-provider-anthropic's
    "Clamping max_tokens from %s to %s for %s" warning for max_tokens."""

    @pytest.mark.asyncio
    async def test_clamp_emits_warning_naming_model_and_both_values(self, caplog):
        """11a. context_window=200000 configured, card reports 131072 ->
        exactly one warning, naming the model, the configured (pref) value,
        and the effective (clamped) value."""
        provider = VLLMProvider(
            client=_mock_client(_model_card("glm-5.2", max_model_len=131072)),
            config={"context_window": 200000},
        )

        with caplog.at_level(logging.WARNING):
            models = await provider.list_models()

        assert models[0].context_window == 131072
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "glm-5.2" in message
        assert "200000" in message
        assert "131072" in message

    def test_no_warning_when_preference_is_below_ceiling(self, caplog):
        """11b. Preference already below the discovered ceiling -> no clamp,
        no warning."""
        provider = VLLMProvider(base_url=BASE_URL, config={"context_window": 100000})
        provider._discovered_limits["glm-5.2"] = 131072

        with caplog.at_level(logging.WARNING):
            context_window, _ = provider._resolve_limits("glm-5.2")

        assert context_window == 100000
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_no_warning_when_nothing_discovered(self, caplog):
        """11c. No server ceiling discovered at all -> the configured
        preference is used as-is, never a "clamp" (nothing to clamp
        against), so no warning fires."""
        provider = VLLMProvider(base_url=BASE_URL, config={"context_window": 200000})

        with caplog.at_level(logging.WARNING):
            context_window, _ = provider._resolve_limits("undiscovered-model")

        assert context_window == 200000
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
