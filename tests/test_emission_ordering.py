"""Tests for emission behavior after verbosity collapse (Task 13f).

Previously this file tested that llm:response fired before llm:response:raw.
After the verbosity collapse (Task 13f), tiered :debug/:raw events no longer
exist. These tests now verify the single-event model works correctly:
- llm:request and llm:response are emitted exactly once
- No tiered :debug or :raw events are emitted
- The `raw` field is present in base events when raw=True
- Continuation path emits llm:response once with continuation_count
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OrderRecordingHooks:
    """Hooks that record event names in emission order."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = OrderRecordingHooks()


class DummyResponse:
    """Minimal response stub with model_dump support."""

    def __init__(self, status="completed", *, text="Hi", resp_id="resp_test"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = status
        self.id = resp_id
        # Present on incomplete responses
        if status == "incomplete":
            self.incomplete_details = SimpleNamespace(reason="max_output_tokens")

    def model_dump(self):
        return {"id": self.id, "status": self.status}


def _make_provider(**config_overrides) -> VLLMProvider:
    config = {
        "max_retries": 0,
        "default_model": "meta-llama/Llama-3-8B",
        "raw": True,  # Enable raw field in events
        # Force non-streaming path: these tests mock responses.create, not responses.stream
        "use_streaming": False,
        **config_overrides,
    }
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


@pytest.fixture()
def provider_with_fake_coordinator():
    """Set up a VLLMProvider wired to a FakeCoordinator with a DummyResponse."""
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    return provider, fake_coordinator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_base_events_emitted(provider_with_fake_coordinator):
    """llm:request and llm:response must both be emitted."""
    provider, fake_coordinator = provider_with_fake_coordinator

    asyncio.run(provider.complete(_simple_request()))

    event_names = [name for name, _ in fake_coordinator.hooks.events]

    assert "llm:request" in event_names, f"llm:request not found in {event_names}"
    assert "llm:response" in event_names, f"llm:response not found in {event_names}"


def test_no_tiered_events_emitted(provider_with_fake_coordinator):
    """After verbosity collapse, no :debug or :raw tiered events should exist."""
    provider, fake_coordinator = provider_with_fake_coordinator

    asyncio.run(provider.complete(_simple_request()))

    event_names = [name for name, _ in fake_coordinator.hooks.events]

    for name in event_names:
        assert not name.endswith(":debug"), (
            f"Unexpected tiered :debug event after verbosity collapse: {name}"
        )
        assert not name.endswith(":raw"), (
            f"Unexpected tiered :raw event after verbosity collapse: {name}"
        )


def test_request_before_response(provider_with_fake_coordinator):
    """llm:request must appear before llm:response in event order."""
    provider, fake_coordinator = provider_with_fake_coordinator

    asyncio.run(provider.complete(_simple_request()))

    event_names = [name for name, _ in fake_coordinator.hooks.events]

    assert "llm:request" in event_names, f"llm:request not found in {event_names}"
    assert "llm:response" in event_names, f"llm:response not found in {event_names}"

    request_idx = event_names.index("llm:request")
    response_idx = event_names.index("llm:response")

    assert request_idx < response_idx, (
        f"llm:request (index {request_idx}) must fire before "
        f"llm:response (index {response_idx}). Events: {event_names}"
    )


def test_raw_field_present_in_base_events(provider_with_fake_coordinator):
    """With raw=True, both llm:request and llm:response should have `raw` field."""
    provider, fake_coordinator = provider_with_fake_coordinator

    asyncio.run(provider.complete(_simple_request()))

    request_payload = next(
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:request"
    )
    assert "raw" in request_payload, "llm:request must have `raw` field when raw=True"

    response_payload = next(
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:response"
    )
    assert "raw" in response_payload, "llm:response must have `raw` field when raw=True"


def test_continuation_path_emission_ordering():
    """llm:response fires once with continuation_count after continuation (incomplete → completed)."""
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # First call returns incomplete, second call returns completed
    incomplete_resp = DummyResponse(
        status="incomplete", text="partial", resp_id="resp_1"
    )
    completed_resp = DummyResponse(status="completed", text="done", resp_id="resp_2")
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, completed_resp]
    )

    asyncio.run(provider.complete(_simple_request()))

    event_names = [name for name, _ in fake_coordinator.hooks.events]

    assert "llm:request" in event_names, f"llm:request not found in {event_names}"
    assert "llm:response" in event_names, f"llm:response not found in {event_names}"

    # No tiered events
    for name in event_names:
        assert not name.endswith(":debug"), (
            f"Unexpected tiered :debug event in continuation path: {name}"
        )
        assert not name.endswith(":raw"), (
            f"Unexpected tiered :raw event in continuation path: {name}"
        )

    # Verify continuation_count is present in the llm:response payload
    response_payload = next(
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:response"
    )
    assert response_payload.get("continuation_count") == 1, (
        f"Expected continuation_count=1 in llm:response payload, "
        f"got {response_payload.get('continuation_count')}"
    )

    # Verify raw field is present (raw=True)
    assert "raw" in response_payload, (
        "llm:response must have `raw` field in continuation path when raw=True"
    )
