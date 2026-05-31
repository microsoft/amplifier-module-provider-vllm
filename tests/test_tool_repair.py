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
        self.usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
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
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config={"max_retries": 0, "use_streaming": False})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="do_something", input={"value": 1})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Should succeed (not raise validation error)
    provider.client.responses.create.assert_awaited_once()

    # Should not emit validation error
    assert all(
        event_name != "provider:validation_error"
        for event_name, _ in fake_coordinator.hooks.events
    )

    # Should emit repair event
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
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
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config={"max_retries": 0, "use_streaming": False})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create a request with missing tool result
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    # First call - should detect and repair
    asyncio.run(provider.complete(request))

    # Verify repair happened
    assert "call_abc123" in provider._repaired_tool_ids  # pyright: ignore[reportAttributeAccessIssue]
    repair_events_1 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_1) == 1

    # Clear events for second call
    fake_coordinator.hooks.events.clear()

    # Second call with SAME messages (simulating message store not persisting synthetic results)
    # This would previously cause infinite loop detection
    messages_2 = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request_2 = ChatRequest(messages=messages_2)

    asyncio.run(provider.complete(request_2))

    # Should NOT emit another repair event for the same tool ID
    repair_events_2 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_2) == 0, "Should not re-detect already-repaired tool IDs"


def test_multiple_missing_tool_results_all_tracked():
    """Multiple missing tool results should all be tracked to prevent infinite loops."""
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config={"max_retries": 0, "use_streaming": False})
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
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3


# ---------------------------------------------------------------------------
# Insertion-position correctness tests (the append-vs-insert bug)
# ---------------------------------------------------------------------------


def test_synthetic_result_inserted_at_correct_position_not_appended():
    """Synthetic result must be inserted right after the assistant message, not at the end.

    Bug: the old code called request.messages.append(synthetic), placing the
    synthetic tool result AFTER any subsequent user messages.  The fix uses
    insert(msg_idx + 1, ...) to keep the result in the correct position.
    """
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config={"max_retries": 0, "use_streaming": False})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_pos", name="read_file", input={})],
        ),
        Message(role="user", content="User message that comes after the tool call"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    roles = [m.role for m in request.messages]
    tool_idx = roles.index("tool")
    user_idx = roles.index("user")
    assert tool_idx < user_idx, (
        f"Synthetic tool result (index {tool_idx}) must appear BEFORE the user "
        f"message (index {user_idx}). Got role sequence: {roles}"
    )
    assert roles[0] == "assistant", "Source assistant message must stay at index 0"
    assert roles[1] == "tool", (
        "Synthetic tool result must be at index 1, directly after the assistant"
    )


def test_fm3_synthetic_assistant_inserted_between_tool_result_and_user_message():
    """FM3: A synthetic assistant turn must bridge the tool result and the next user message.

    When a synthetic tool result is injected, the next real message is often a user
    message.  Without an intervening assistant turn the conversation structure is
    invalid (tool → user with no assistant in between).  FM3 inserts a synthetic
    assistant acknowledgment to restore the required turn order.
    """
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config={"max_retries": 0, "use_streaming": False})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_fm3", name="search", input={"q": "test"})],
        ),
        Message(role="user", content="Continue please"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Expected: [assistant(call), tool(synthetic), assistant(FM3), user]
    roles = [m.role for m in request.messages]
    assert roles == ["assistant", "tool", "assistant", "user"], (
        f"Expected [assistant, tool, assistant, user] after FM3 repair, got {roles}"
    )


def test_no_fm3_when_no_user_message_follows_tool_result():
    """FM3 synthetic assistant must NOT be inserted when no user message follows.

    If the tool call is the final assistant turn (no subsequent user message),
    inserting a synthetic assistant turn would be incorrect and wasteful.
    """
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config={"max_retries": 0, "use_streaming": False})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    # Tool call is the last meaningful message — no user message follows
    messages = [
        Message(role="user", content="Do a thing"),
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_last", name="end_tool", input={})],
        ),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Expected: [user, assistant(call), tool(synthetic)] — NO extra assistant
    roles = [m.role for m in request.messages]
    assert roles == ["user", "assistant", "tool"], (
        f"Expected [user, assistant, tool], got {roles}. "
        "FM3 must NOT insert a synthetic assistant when no user message follows."
    )


def test_multiple_groups_inserted_at_correct_positions():
    """Missing results from different assistant turns are each inserted at the right place.

    Given two separate assistant turns that both have missing tool results, the repair
    must insert each synthetic result directly after its own assistant message, preserving
    the full conversation order.
    """
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config={"max_retries": 0, "use_streaming": False})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    messages = [
        Message(role="user", content="Start"),
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_a", name="tool_a", input={})],
        ),
        Message(role="user", content="Middle user message"),
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_b", name="tool_b", input={})],
        ),
        Message(role="user", content="End user message"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Expected after repair (FM3 inserts synthetic assistant before each user message):
    # [user, assistant(a), tool(syn_a), assistant(FM3), user(Middle),
    #  assistant(b), tool(syn_b), assistant(FM3), user(End)]
    roles = [m.role for m in request.messages]
    assert roles == [
        "user",  # Start
        "assistant",  # call_a
        "tool",  # synthetic for call_a
        "assistant",  # FM3 bridge for call_a group
        "user",  # Middle
        "assistant",  # call_b
        "tool",  # synthetic for call_b
        "assistant",  # FM3 bridge for call_b group
        "user",  # End
    ], f"Unexpected role sequence after multi-group repair: {roles}"
