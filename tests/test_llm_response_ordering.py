"""Tests for llm:response event canonical usage keys.

Verifies that the llm:response event uses canonical usage keys:
  input_tokens, output_tokens, cache_read_tokens  (not input/output)

The fix ensures ChatResponse is built FIRST (via _convert_to_chat_response),
then the event is emitted from canonical chat_response.usage fields.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OrderRecordingHooks:
    """Hooks that record event names and payloads in emission order."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def payload_for(self, event_name: str) -> dict | None:
        for name, payload in self.events:
            if name == event_name:
                return payload
        return None


class FakeCoordinator:
    def __init__(self):
        self.hooks = OrderRecordingHooks()


def _make_dummy_response(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached_tokens: int | None = None,
) -> SimpleNamespace:
    """Create a minimal vLLM/OpenAI-compatible response stub with configurable usage."""
    usage_attrs: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cached_tokens is not None:
        usage_attrs["input_tokens_details"] = SimpleNamespace(
            cached_tokens=cached_tokens
        )

    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello!")],
            )
        ],
        usage=SimpleNamespace(**usage_attrs),
        status="completed",
        id="resp_test",
        model_dump=lambda: {"id": "resp_test", "status": "completed"},
    )


def _make_provider() -> VLLMProvider:
    config = {"max_retries": 0, "default_model": "meta-llama/Llama-3-8B"}
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_event_usage_uses_input_tokens_key():
    """llm:response event usage must have 'input_tokens' key (not 'input')."""
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    provider.client.responses.create = AsyncMock(return_value=_make_dummy_response())
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"
    usage = payload.get("usage", {})
    assert "input_tokens" in usage, f"Expected 'input_tokens', got keys: {list(usage.keys())}"


def test_event_usage_uses_output_tokens_key():
    """llm:response event usage must have 'output_tokens' key (not 'output')."""
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    provider.client.responses.create = AsyncMock(return_value=_make_dummy_response())
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"
    usage = payload.get("usage", {})
    assert "output_tokens" in usage, f"Expected 'output_tokens', got keys: {list(usage.keys())}"


def test_event_usage_does_not_have_old_input_key():
    """llm:response event usage must NOT have old 'input' key."""
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    provider.client.responses.create = AsyncMock(return_value=_make_dummy_response())
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"
    usage = payload.get("usage", {})
    assert "input" not in usage, f"Old 'input' key must be removed, found: {usage}"


def test_event_usage_does_not_have_old_output_key():
    """llm:response event usage must NOT have old 'output' key."""
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    provider.client.responses.create = AsyncMock(return_value=_make_dummy_response())
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"
    usage = payload.get("usage", {})
    assert "output" not in usage, f"Old 'output' key must be removed, found: {usage}"


def test_event_usage_includes_cache_read_tokens_when_present():
    """llm:response event usage must include 'cache_read_tokens' when cache data is present."""
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    provider.client.responses.create = AsyncMock(
        return_value=_make_dummy_response(cached_tokens=800)
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"
    usage = payload.get("usage", {})
    assert "cache_read_tokens" in usage, f"Expected 'cache_read_tokens', got: {list(usage.keys())}"
    assert usage["cache_read_tokens"] == 800


def test_event_usage_input_tokens_value_correct():
    """llm:response event usage.input_tokens reflects actual API token count."""
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    provider.client.responses.create = AsyncMock(
        return_value=_make_dummy_response(input_tokens=150, output_tokens=75)
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None, "llm:response event must be emitted"
    usage = payload.get("usage", {})
    assert usage.get("input_tokens") == 150
    assert usage.get("output_tokens") == 75
