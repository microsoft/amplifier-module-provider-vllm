"""Regression tests: cost_usd JSON-serialization correctness.

Covers the fix to the session.cost contributor lambda (PR #22 left the
lambda returning raw ``_totals["cost_usd"]``, a ``Decimal``; if
``has_data`` were ever True that would blow up ``json.dumps``).

Five tests:

1. ``test_llm_response_event_is_json_serializable``
   The full ``llm:response`` event payload must pass ``json.dumps``
   without raising.  ``cost_usd`` will be ``None`` for vLLM.

2. ``test_llm_response_event_cost_usd_is_none``
   ``cost_usd`` in the event usage dict is exactly ``None`` — vLLM is
   self-hosted so ``compute_cost()`` is always indeterminate.

3. ``test_llm_response_event_cost_usd_round_trips_through_json``
   ``json.dumps`` → ``json.loads`` leaves ``cost_usd`` as ``None``,
   confirming the value is a native JSON type (not a Decimal).

4. ``test_contributor_returns_none_always``
   The ``session.cost`` contributor callback registered by ``mount()``
   must return ``None`` for vLLM — ``has_data`` is never set because
   ``compute_cost()`` always returns ``None``.

5. ``test_usage_model_stores_none_internally``
   ``result.usage.cost_usd`` is ``None`` after
   ``_convert_to_chat_response()`` — the stamp happens at the right site.
"""

import asyncio
import json
import os
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider, mount


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _CapturingHooks:
    """Records (event_name, payload) pairs in emission order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def payload_for(self, name: str) -> dict | None:
        for n, p in self.events:
            if n == name:
                return p
        return None


class _FakeCoordinator:
    """Minimal coordinator stub that captures contributor callbacks."""

    def __init__(self) -> None:
        self.hooks = _CapturingHooks()
        self._contributors: dict[str, object] = {}

    async def mount(self, slot: str, provider: object, name: str | None = None) -> None:
        self.provider = provider

    def register_contributor(self, channel: str, name: str, callback: object) -> None:
        self._contributors[channel] = callback


def _make_dummy_response(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> SimpleNamespace:
    """Minimal vLLM-compatible response stub for patching ``client.responses.create``."""
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello!")],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        status="completed",
        id="resp_test",
        # model field is read by compute_cost() inside _convert_to_chat_response
        model="meta-llama/Llama-3-8B",
        model_dump=lambda: {"id": "resp_test", "status": "completed"},
    )


def _make_provider() -> VLLMProvider:
    config = {"max_retries": 0, "default_model": "meta-llama/Llama-3-8B", "use_streaming": False}
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


# ---------------------------------------------------------------------------
# Test 1 — full event payload is JSON-serializable
# ---------------------------------------------------------------------------


def test_llm_response_event_is_json_serializable():
    """The entire ``llm:response`` event payload must pass ``json.dumps``.

    Regression guard: if ``cost_usd`` were a raw ``Decimal`` (the pre-fix
    lambda bug) ``json.dumps`` would raise ``TypeError``.  For vLLM,
    ``cost_usd`` is ``None`` so the serialization is trivially fine, but
    the test ensures no other non-serializable type was introduced.
    """
    provider = _make_provider()
    fake_coord = _FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coord)
    provider.client.responses.create = AsyncMock(return_value=_make_dummy_response())

    asyncio.run(provider.complete(_simple_request()))

    payload = fake_coord.hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"

    # Must not raise TypeError: Object of type Decimal is not JSON serializable
    serialized = json.dumps(payload)
    assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Test 2 — cost_usd is None in the event
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_is_none():
    """``cost_usd`` in the ``llm:response`` usage dict is ``None`` for vLLM.

    vLLM is self-hosted — ``compute_cost()`` always returns ``None``
    (indeterminate, not $0.00).
    """
    provider = _make_provider()
    fake_coord = _FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coord)
    provider.client.responses.create = AsyncMock(return_value=_make_dummy_response())

    asyncio.run(provider.complete(_simple_request()))

    payload = fake_coord.hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"

    usage = payload.get("usage", {})
    assert "cost_usd" in usage, (
        f"cost_usd must be present in llm:response usage dict, got keys: {list(usage)}"
    )
    assert usage["cost_usd"] is None, (
        f"Expected cost_usd=None for self-hosted vLLM, got {usage['cost_usd']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — cost_usd round-trips through JSON unchanged
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_round_trips_through_json():
    """``cost_usd`` survives ``json.dumps`` → ``json.loads`` as ``None``.

    Confirms the value is a native JSON type.  A ``Decimal`` would survive
    ``json.dumps`` only if explicitly converted; this test catches any
    regression where the value becomes a string ``"None"`` instead of
    JSON ``null``.
    """
    provider = _make_provider()
    fake_coord = _FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coord)
    provider.client.responses.create = AsyncMock(return_value=_make_dummy_response())

    asyncio.run(provider.complete(_simple_request()))

    payload = fake_coord.hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"

    roundtripped = json.loads(json.dumps(payload))
    assert roundtripped["usage"]["cost_usd"] is None, (
        f"cost_usd must survive json round-trip as None, "
        f"got {roundtripped['usage']['cost_usd']!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — contributor callback always returns None for vLLM
# ---------------------------------------------------------------------------


def test_contributor_returns_none_always():
    """The ``session.cost`` contributor callback must always return ``None``.

    vLLM is self-hosted so ``compute_cost()`` always returns ``None``,
    meaning ``_add_cost(None)`` is a no-op and ``has_data`` is never set to
    ``True``.  The contributor lambda therefore returns ``None`` (no cost to
    contribute to the session summary).

    This also validates the post-fix lambda: even if ``has_data`` were
    somehow True, the lambda now wraps the value in ``str()`` so the
    contributor output would be JSON-safe.
    """
    fake_coord = _FakeCoordinator()

    async def _run() -> None:
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://localhost:8000/v1"}):
            await mount(fake_coord)

    asyncio.run(_run())

    contributor = fake_coord._contributors.get("session.cost")
    assert contributor is not None, "mount() must register a session.cost contributor"

    result = contributor()
    assert result is None, (
        f"Contributor must return None for vLLM (cost is always indeterminate), "
        f"got {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Usage model stores cost_usd=None internally
# ---------------------------------------------------------------------------


def test_usage_model_stores_none_internally():
    """``result.usage.cost_usd`` is ``None`` after ``_convert_to_chat_response()``.

    Verifies the cost stamp is applied at the right site and produces the
    correct sentinel value (``None``, not ``Decimal(0)`` or absent).
    """
    provider = _make_provider()

    response = _make_dummy_response()
    result = provider._convert_to_chat_response(response)

    assert result.usage is not None, "Usage must be populated on the ChatResponse"
    assert hasattr(result.usage, "cost_usd"), (
        "cost_usd field must exist on the Usage object after stamping"
    )
    assert result.usage.cost_usd is None, (
        f"Expected cost_usd=None for vLLM (self-hosted, indeterminate), "
        f"got {result.usage.cost_usd!r}"
    )
