"""Wire-format regression tests for multi-turn assistant message serialization.

Guards the fix for the "Cannot determine type of 'item'" 400 that llama-server
raises -- and the ``pydantic.ValidationError`` that vLLM raises -- when assistant
history is replayed into the Responses API ``input`` array.

The provider must emit the canonical ``ResponseOutputMessage`` shape ("Form 2+"):
``{"type": "message", "id": ..., "role": "assistant", "status": "completed",
   "content": [{"type": "output_text", "text": ..., "annotations": []}]}``.

Verified on the wire against the OpenAI Responses API, llama.cpp's llama-server,
and vLLM 0.19 as the single form every backend accepts:
- ``type`` is required by llama-server (item dispatch keys on it),
- ``id`` and ``status`` are required by vLLM (openai SDK ``ResponseOutputMessageParam``),
- ``annotations`` mirrors OpenAI's own output items and is accepted everywhere.

A bare ``type: message`` (no ``id``/``status``) is NOT sufficient: vLLM's strict
Pydantic validation rejects it. User/input messages must stay untyped.
"""

from typing import Any, cast
from unittest.mock import MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_module_provider_vllm import VLLMProvider, _build_assistant_message_item


def _make_provider() -> VLLMProvider:
    """Create a minimal provider instance for unit testing."""
    provider = VLLMProvider(base_url="http://localhost:8000/v1")
    fake_coordinator = MagicMock()
    fake_coordinator.hooks = MagicMock()
    # cast() satisfies pyright — ModuleCoordinator is only needed for type checking,
    # not at runtime (MagicMock handles all attribute access dynamically).
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    return provider


class TestAssistantMessageHelper:
    """The shared serializer must produce the canonical Form 2+ shape."""

    def test_single_text_part(self) -> None:
        item = _build_assistant_message_item(
            [{"type": "output_text", "text": "Hello!"}]
        )
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert item["status"] == "completed"
        assert item["id"].startswith("msg_")
        assert item["content"] == [
            {"type": "output_text", "text": "Hello!", "annotations": []}
        ]

    def test_preserved_id_used_when_given(self) -> None:
        item = _build_assistant_message_item(
            [{"type": "output_text", "text": "x"}], message_id="msg_keep_me"
        )
        assert item["id"] == "msg_keep_me"

    def test_empty_content(self) -> None:
        item = _build_assistant_message_item([])
        assert item["type"] == "message"
        assert item["content"] == []


class TestAssistantMessageFormat:
    """Assistant messages must use the canonical Form 2+ Responses API shape."""

    def test_assistant_text_uses_responses_api_format(self) -> None:
        """Assistant text serializes as a typed message with id, status, annotations."""
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
        assert msg.get("status") == "completed"
        assert msg.get("id", "").startswith("msg_")
        content = msg.get("content")
        assert isinstance(content, list)
        assert content[0] == {
            "type": "output_text",
            "text": "Hello!",
            "annotations": [],
        }

    def test_assistant_with_structured_text_blocks(self) -> None:
        """Structured text blocks also get the full Form 2+ shape."""
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
        assert msg.get("status") == "completed"
        assert msg.get("id", "").startswith("msg_")
        content = msg.get("content")
        assert isinstance(content, list)
        assert content[0] == {
            "type": "output_text",
            "text": "Here is something.",
            "annotations": [],
        }

    def test_user_messages_unchanged(self) -> None:
        """User messages should NOT get type:message."""
        provider = _make_provider()
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
        result = provider._convert_messages(messages)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}
        assert "type" not in result[0]


class TestContinuationInputFormat:
    """_build_continuation_input must also produce Form 2+ for assistant items."""

    def test_continuation_assistant_is_form_2plus(self) -> None:
        """Assistant content in continuation input includes type/id/status/annotations."""
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
        assert assistant_msg.get("status") == "completed"
        assert assistant_msg.get("id", "").startswith("msg_")
        assert assistant_msg["content"] == [
            {"type": "output_text", "text": "Once upon a time...", "annotations": []}
        ]

    def test_empty_assistant_content_not_appended(self) -> None:
        """When accumulated output has no extractable text, no assistant message is added."""
        provider = _make_provider()
        original_input = [{"role": "user", "content": "Start a story"}]
        # Output item has no content entries — nothing to extract
        accumulated_output: list[dict[str, Any]] = [{"type": "message", "content": []}]
        result = provider._build_continuation_input(original_input, accumulated_output)
        # Only the original user message — no assistant message appended
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Start a story"}
