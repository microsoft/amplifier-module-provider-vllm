"""Response handling for vLLM Responses API.

This module handles conversion of vLLM API responses to Amplifier's ChatResponse format,
including reasoning extraction, incomplete response continuation, and reasoning state preservation.

Following the "bricks and studs" philosophy, this is a self-contained module that can be
regenerated independently of the main provider code.
"""

import json
import logging
from typing import Any

from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core.message_models import TextBlock
from amplifier_core.message_models import ThinkingBlock
from amplifier_core.message_models import ToolCall
from amplifier_core.message_models import ToolCallBlock
from amplifier_core.message_models import Usage

from ._constants import METADATA_CONTINUATION_COUNT
from ._constants import METADATA_INCOMPLETE_REASON
from ._constants import METADATA_RESPONSE_ID
from ._constants import METADATA_STATUS

logger = logging.getLogger(__name__)


def convert_response_with_accumulated_output(
    final_response: Any,
    accumulated_output: list[Any],
    continuation_count: int,
    chat_response_class: type,
) -> Any:
    """Convert OpenAI response with accumulated output to ChatResponse.

    This handles responses that may have been continued multiple times due to
    incomplete status. All output from all continuations is accumulated and
    merged into a single ChatResponse.

    Args:
        final_response: The final (completed) response object from OpenAI
        accumulated_output: All output items from all continuation calls
        continuation_count: Number of continuations made (0 if no continuations)
        chat_response_class: The ChatResponse class to instantiate (allows OpenAIChatResponse)

    Returns:
        ChatResponse with all accumulated content and metadata
    """
    content_blocks = []
    tool_calls = []
    event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
    text_accumulator: list[str] = []

    # Process ALL accumulated output items
    for block in accumulated_output:
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
                                    text=text, raw=getattr(content_item, "raw", None)
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
                        elif hasattr(item, "type") and item.type == "reasoning_text":  # type: ignore[union-attr]
                            texts.append(getattr(item, "text", ""))
                    if texts:
                        reasoning_text = "\n".join(texts)

                # Create thinking block with reasoning text (displays to user)
                if reasoning_text:
                    content_blocks.append(
                        ThinkingBlock(
                            thinking=reasoning_text,
                            signature=None,
                            visibility="internal",
                            content=None,  # Simple: no encryption state needed
                        )
                    )
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
                content_blocks.append(
                    ToolCallBlock(id=tool_id, name=tool_name, input=tool_input)
                )
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, arguments=tool_input)
                )
                event_blocks.append(
                    ToolCallContent(id=tool_id, name=tool_name, arguments=tool_input)
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
                    content_blocks.append(
                        ThinkingBlock(
                            thinking=reasoning_text,
                            signature=None,
                            visibility="internal",
                            content=None,  # Simple: no encryption state needed
                        )
                    )
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

    # Extract usage from final response
    usage_obj = final_response.usage if hasattr(final_response, "usage") else None
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
            cache_read_tokens = details.cached_tokens or None

    usage = Usage(
        input_tokens=usage_counts["input"],
        output_tokens=usage_counts["output"],
        total_tokens=usage_counts["total"],
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
    )

    # Build metadata with provider-specific state
    metadata = {}

    # Response ID (for next turn's previous_response_id)
    if hasattr(final_response, "id"):
        metadata[METADATA_RESPONSE_ID] = final_response.id

    # Status (should be "completed" after continuations, or "incomplete" if we gave up)
    if hasattr(final_response, "status"):
        metadata[METADATA_STATUS] = final_response.status

        # If still incomplete after all attempts, record the reason
        if final_response.status == "incomplete":
            incomplete_details = getattr(final_response, "incomplete_details", None)
            if incomplete_details:
                if isinstance(incomplete_details, dict):
                    metadata[METADATA_INCOMPLETE_REASON] = incomplete_details.get(
                        "reason"
                    )
                elif hasattr(incomplete_details, "reason"):
                    metadata[METADATA_INCOMPLETE_REASON] = incomplete_details.reason

    # Continuation count (for debugging/metrics)
    if continuation_count > 0:
        metadata[METADATA_CONTINUATION_COUNT] = continuation_count

    combined_text = "\n\n".join(text_accumulator).strip()

    chat_response = chat_response_class(
        content=content_blocks,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
        finish_reason=getattr(final_response, "finish_reason", None),
        content_blocks=event_blocks if event_blocks else None,
        text=combined_text or None,
        metadata=metadata if metadata else None,
    )

    return chat_response
