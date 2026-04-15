"""Tests for multi-turn assistant message serialization.

Verifies that _convert_messages produces proper Responses API format
for assistant messages (type: message with structured content array),
not Chat Completions format (plain string content).
"""

from typing import Any, cast
from unittest.mock import MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_module_provider_vllm import VLLMProvider


def _make_provider() -> VLLMProvider:
    """Create a minimal provider instance for unit testing."""
    provider = VLLMProvider(base_url="http://localhost:8000/v1")
    fake_coordinator = MagicMock()
    fake_coordinator.hooks = MagicMock()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    return provider


class TestAssistantMessageFormat:
    """Assistant messages must use Responses API format, not Chat Completions."""

    def test_assistant_text_uses_responses_api_format(self) -> None:
        """Assistant text content should serialize as type:message with structured content array."""
        provider = _make_provider()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Say hello"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Say goodbye"},
        ]
        result = provider._convert_messages(messages)
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        msg = assistant_msgs[0]
        assert msg.get("type") == "message"
        content = msg.get("content")
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0] == {"type": "output_text", "text": "Hello!"}

    def test_assistant_with_structured_text_blocks(self) -> None:
        """Assistant messages with structured text blocks should also get type:message."""
        provider = _make_provider()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Tell me something"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Here is something."}],
            },
            {"role": "user", "content": "Tell me more"},
        ]
        result = provider._convert_messages(messages)
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        msg = assistant_msgs[0]
        assert msg.get("type") == "message"
        content = msg.get("content")
        assert isinstance(content, list)
        assert content[0] == {"type": "output_text", "text": "Here is something."}

    def test_user_messages_unchanged(self) -> None:
        """User messages should NOT get type:message."""
        provider = _make_provider()
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
        result = provider._convert_messages(messages)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}
        assert "type" not in result[0]


class TestContinuationInputFormat:
    """_build_continuation_input must also produce type:message for assistant items."""

    def test_continuation_assistant_has_type_message(self) -> None:
        """Assistant content in continuation input should include type:message."""
        provider = _make_provider()
        original_input = [{"role": "user", "content": "Start a story"}]
        accumulated_output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Once upon a time..."}],
            }
        ]
        result = provider._build_continuation_input(original_input, accumulated_output)
        assert len(result) == 2
        assistant_msg = result[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg.get("type") == "message"
        assert assistant_msg["content"] == [
            {"type": "output_text", "text": "Once upon a time..."}
        ]
