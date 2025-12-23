"""Tests for tool result repair and infinite loop prevention."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCallBlock
from amplifier_module_provider_vllm import VLLMProvider


class DummyResponse:
    """Minimal response stub for provider tests."""

    def __init__(self, output=None):
        self.output = output or []
        self.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self.stop_reason = "stop"


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def test_tool_call_sequence_missing_tool_message_is_repaired():
    """Missing tool results should be repaired with synthetic results and emit event."""
    provider = VLLMProvider(base_url="http://localhost:8000/v1")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_1", name="do_something", input={"value": 1})],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Should succeed (not raise validation error)
    provider.client.responses.create.assert_awaited_once()

    # Should not emit validation error
    assert all(event_name != "provider:validation_error" for event_name, _ in fake_coordinator.hooks.events)

    # Should emit repair event
    repair_events = [e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "vllm"
    assert repair_events[0][1]["repair_count"] == 1
    assert repair_events[0][1]["repairs"][0]["tool_name"] == "do_something"


def test_repaired_tool_ids_are_not_detected_again():
    """Repaired tool IDs should be tracked and not trigger infinite detection loops.

    This test verifies the fix for the infinite loop bug where:
    1. Missing tool results are detected and synthetic results are injected
    2. Synthetic results are NOT persisted to message store
    3. On next iteration, same missing tool results are detected again
    4. This creates an infinite loop of detection -> injection -> detection

    The fix tracks repaired tool IDs to skip re-detection.
    """
    provider = VLLMProvider(base_url="http://localhost:8000/v1")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create a request with missing tool result
    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    # First call - should detect and repair
    asyncio.run(provider.complete(request))

    # Verify repair happened
    assert "call_abc123" in provider._repaired_tool_ids  # pyright: ignore[reportAttributeAccessIssue]
    repair_events_1 = [e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"]
    assert len(repair_events_1) == 1

    # Clear events for second call
    fake_coordinator.hooks.events.clear()

    # Second call with SAME messages (simulating message store not persisting synthetic results)
    # This would previously cause infinite loop detection
    messages_2 = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request_2 = ChatRequest(messages=messages_2)

    asyncio.run(provider.complete(request_2))

    # Should NOT emit another repair event for the same tool ID
    repair_events_2 = [e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"]
    assert len(repair_events_2) == 0, "Should not re-detect already-repaired tool IDs"


def test_multiple_missing_tool_results_all_tracked():
    """Multiple missing tool results should all be tracked to prevent infinite loops."""
    provider = VLLMProvider(base_url="http://localhost:8000/v1")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create request with 3 parallel tool calls, none with results
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="grep", input={"pattern": "a"}),
                ToolCallBlock(id="call_2", name="grep", input={"pattern": "b"}),
                ToolCallBlock(id="call_3", name="grep", input={"pattern": "c"}),
            ],
        ),
        Message(role="user", content="No tool results"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # All 3 should be tracked
    assert provider._repaired_tool_ids == {"call_1", "call_2", "call_3"}  # pyright: ignore[reportAttributeAccessIssue]

    # Verify repair event has all 3
    repair_events = [e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3
