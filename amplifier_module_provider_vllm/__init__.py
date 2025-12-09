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
from typing import Any
from typing import cast

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import ToolCall
from openai import AsyncOpenAI

from ._constants import DEFAULT_DEBUG_TRUNCATE_LENGTH
from ._constants import DEFAULT_MAX_TOKENS
from ._constants import DEFAULT_MODEL
from ._constants import DEFAULT_REASONING_SUMMARY
from ._constants import DEFAULT_TIMEOUT
from ._constants import DEFAULT_TRUNCATION
from ._constants import MAX_CONTINUATION_ATTEMPTS
from ._constants import METADATA_CONTINUATION_COUNT
from ._constants import METADATA_INCOMPLETE_REASON
from ._constants import METADATA_RESPONSE_ID
from ._constants import METADATA_STATUS
from ._response_handling import convert_response_with_accumulated_output
from ._token_accounting import apply_token_accounting
from ._token_accounting import should_apply_token_accounting

logger = logging.getLogger(__name__)


class VLLMChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the vLLM provider."""
    config = config or {}

    # vLLM server URL from config or environment
    base_url = config.get("base_url") or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    provider = VLLMProvider(base_url=base_url, config=config, coordinator=coordinator)
    await coordinator.mount("providers", provider, name="vllm")
    logger.info(f"Mounted VLLMProvider (Responses API) at {base_url}")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class VLLMProvider:
    """vLLM Responses API integration (OpenAI-compatible)."""

    name = "vllm"
    api_label = "vLLM"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """Initialize vLLM provider with Responses API client.

        Args:
            base_url: vLLM server URL (e.g., http://192.168.128.5:8000/v1)
            config: Provider configuration
            coordinator: Module coordinator
            client: Pre-configured AsyncOpenAI client (for testing)
        """
        if client is None:
            if base_url is None:
                raise ValueError("base_url or client must be provided")
            # vLLM uses OpenAI-compatible API but with custom base_url
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key="EMPTY",  # vLLM doesn't require API key
            )
        else:
            self.client = client
        self.config = config or {}
        self.coordinator = coordinator
        self.base_url = base_url

        # Configuration with sensible defaults (from _constants.py - single source of truth)
        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        self.max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.temperature = self.config.get("temperature", None)  # None = not sent (some models don't support it)
        self.reasoning = self.config.get("reasoning", None)  # None = not sent (minimal|low|medium|high)
        self.reasoning_summary = self.config.get("reasoning_summary", DEFAULT_REASONING_SUMMARY)
        self.truncation = self.config.get("truncation", DEFAULT_TRUNCATION)  # Automatic context management
        self.enable_state = self.config.get("enable_state", False)
        self.debug = self.config.get("debug", False)  # Enable full request/response logging
        self.raw_debug = self.config.get("raw_debug", False)  # Enable ultra-verbose raw API I/O logging
        self.debug_truncate_length = self.config.get("debug_truncate_length", DEFAULT_DEBUG_TRUNCATE_LENGTH)
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)

        # Provider priority for selection (lower = higher priority)
        self.priority = self.config.get("priority", 100)

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="vllm",
            display_name="vLLM",
            credential_env_vars=[],  # No API key needed for vLLM
            capabilities=["streaming", "tools", "reasoning", "local"],
            defaults={
                "model": self.default_model,
                "max_tokens": 16384,
                "temperature": None,
                "timeout": 300.0,
            },
            config_fields=[
                ConfigField(
                    id="base_url",
                    display_name="Server URL",
                    prompt="vLLM server URL",
                    field_type="text",
                    env_var="VLLM_BASE_URL",
                    default="http://localhost:8000/v1",
                    required=True,
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available models from vLLM server.

        vLLM serves a single model per instance, so we query
        the models endpoint to get info about the loaded model.
        """
        try:
            # vLLM supports OpenAI-compatible /v1/models endpoint
            models_response = await self.client.models.list()
            models = []
            for model in models_response.data:
                models.append(
                    ModelInfo(
                        id=model.id,
                        display_name=model.id,
                        context_window=128000,  # Default, vLLM doesn't expose this
                        max_output_tokens=32768,  # Default
                        capabilities=["tools", "streaming", "reasoning", "local"],
                        defaults={"temperature": None, "max_tokens": 16384},
                    )
                )
            return models
        except Exception as e:
            logger.warning(f"Failed to list vLLM models: {e}")
            return []

    def _build_continuation_input(self, original_input: list, accumulated_output: list) -> list:
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
                        if hasattr(content_item, "type") and content_item.type == "output_text":
                            text = getattr(content_item, "text", "")
                            if text:
                                assistant_content.append({"type": "output_text", "text": text})
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
                                assistant_content.append({"type": "output_text", "text": text})

        # If we extracted any assistant content, add as assistant message
        if assistant_content:
            continuation_input.append({"role": "assistant", "content": assistant_content})

        return continuation_input

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """Recursively truncate string values in nested structures.

        Preserves structure, only truncates leaf string values longer than max_length.
        Uses self.debug_truncate_length if max_length not specified.

        Args:
            obj: Any JSON-serializable structure (dict, list, primitives)
            max_length: Maximum string length (defaults to self.debug_truncate_length)

        Returns:
            Structure with truncated string values
        """
        if max_length is None:
            max_length = self.debug_truncate_length

        # Type guard: max_length is guaranteed to be int after this point
        assert max_length is not None, "max_length should never be None after initialization"

        if isinstance(obj, str):
            if len(obj) > max_length:
                return obj[:max_length] + f"... (truncated {len(obj) - max_length} chars)"
            return obj
        if isinstance(obj, dict):
            return {k: self._truncate_values(v, max_length) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_values(item, max_length) for item in obj]
        return obj  # Numbers, booleans, None pass through unchanged

    def _find_missing_tool_results(self, messages: list) -> list[tuple[str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs.

        Returns:
            List of (call_id, tool_name, tool_arguments) tuples for unpaired calls
        """
        from amplifier_core.message_models import Message

        tool_calls = {}  # {call_id: (name, args)}
        tool_results = set()  # {call_id}

        for msg in messages:
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (block.name, block.input)

            # Check tool messages for tool_call_id
            elif msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                tool_results.add(msg.tool_call_id)

        return [(call_id, name, args) for call_id, (name, args) in tool_calls.items() if call_id not in tool_results]

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
                f"Tool IDs: {[call_id for call_id, _, _ in missing]}"
            )

            # Inject synthetic results
            for call_id, tool_name, _ in missing:
                synthetic = self._create_synthetic_result(call_id, tool_name)
                request.messages.append(synthetic)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name} for call_id, tool_name, _ in missing
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

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.info(f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages")
        logger.info(f"[PROVIDER] Message roles: {[m.role for m in request.messages]}")

        message_list = list(request.messages)

        # Separate messages by role
        system_msgs = [m for m in message_list if m.role == "system"]
        developer_msgs = [m for m in message_list if m.role == "developer"]
        conversation = [m for m in message_list if m.role in ("user", "assistant", "tool")]

        logger.info(
            f"[PROVIDER] Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Combine system messages as instructions
        instructions = (
            "\n\n".join(m.content if isinstance(m.content, str) else "" for m in system_msgs) if system_msgs else None
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

        reasoning_effort = kwargs.get("reasoning", getattr(request, "reasoning", None)) or self.reasoning
        if reasoning_effort:
            # DEBUG: Log reasoning configuration
            logger.info(f"[PROVIDER] Setting reasoning: effort={reasoning_effort}, summary={self.reasoning_summary}")
            params["reasoning"] = {
                "effort": reasoning_effort,
                "summary": self.reasoning_summary,  # Verbosity: auto|concise|detailed
            }

        # CRITICAL: Always request encrypted_content with store=False for stateless reasoning preservation
        # This is separate from reasoning effort - we need encrypted content even if effort not explicitly set
        if not store_enabled:
            params["include"] = kwargs.get("include", ["reasoning.encrypted_content"])
            logger.debug("[PROVIDER] Requesting encrypted_content (store=False, enables stateless reasoning)")

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
                    "effort": kwargs.get("reasoning_effort") or self.config.get("reasoning_effort", "high"),
                    "summary": self.reasoning_summary,  # Verbosity: auto|concise|detailed
                }

            budget_tokens = kwargs.get("thinking_budget_tokens") or self.config.get("thinking_budget_tokens") or 0
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get("thinking_budget_buffer", 1024)

            if budget_tokens:
                thinking_budget = budget_tokens
                target_tokens = budget_tokens + buffer_tokens
                if params.get("max_output_tokens"):
                    params["max_output_tokens"] = max(params["max_output_tokens"], target_tokens)
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
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": self.name,
                    "model": params["model"],
                    "message_count": len(message_list),
                    "has_instructions": bool(instructions),
                    "reasoning_enabled": params.get("reasoning") is not None,
                    "thinking_enabled": thinking_enabled,
                    "thinking_budget": thinking_budget,
                },
            )

            # DEBUG level: Full request payload with truncated values (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": self.name,
                        "request": self._truncate_values(params),
                    },
                )

            # RAW level: Complete params dict as sent to OpenAI API (if debug AND raw_debug enabled)
            if self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": self.name,
                        "params": params,  # Complete untruncated params
                    },
                )

        start_time = time.time()

        # Call provider API
        try:
            response = await asyncio.wait_for(self.client.responses.create(**params), timeout=self.timeout)
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Apply token accounting for GPT-OSS models (vLLM bug returns zeros)
            if should_apply_token_accounting(params["model"]):
                response = apply_token_accounting(params, response)

            logger.info("[PROVIDER] Received response from %s API", self.api_label)

            # RAW level: Complete response object from OpenAI API (if debug AND raw_debug enabled)
            if self.coordinator and hasattr(self.coordinator, "hooks") and self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:response:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": self.name,
                        "response": response.model_dump(),  # Pydantic model → dict (complete untruncated)
                    },
                )

            # Handle incomplete responses via auto-continuation
            # OpenAI Responses API may return status="incomplete" with reason like "max_output_tokens"
            # We automatically continue until complete to provide seamless experience
            accumulated_output = list(response.output) if hasattr(response, "output") else []
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
                # Instead of previous_response_id (requires store:true), we include the
                # accumulated output in the input to preserve context
                continuation_input = self._build_continuation_input(all_messages_for_conversion, accumulated_output)

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
                    continue_params["parallel_tool_calls"] = params.get("parallel_tool_calls", True)
                if "store" in params:
                    continue_params["store"] = params["store"]

                # Make continuation call
                try:
                    continue_start = time.time()
                    final_response = await asyncio.wait_for(
                        self.client.responses.create(**continue_params),
                        timeout=self.timeout,
                    )
                    continue_elapsed = int((time.time() - continue_start) * 1000)
                    elapsed_ms += continue_elapsed

                    # Accumulate output from continuation
                    if hasattr(final_response, "output"):
                        accumulated_output.extend(final_response.output)

                    # Emit raw debug for continuation if enabled
                    if self.coordinator and hasattr(self.coordinator, "hooks") and self.debug and self.raw_debug:
                        await self.coordinator.hooks.emit(
                            "llm:response:raw",
                            {
                                "lvl": "DEBUG",
                                "provider": self.name,
                                "response": final_response.model_dump(),
                                "continuation": continuation_count,
                            },
                        )

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

            # Extract usage counts
            usage_obj = response.usage if hasattr(response, "usage") else None
            usage_counts = {"input": 0, "output": 0, "total": 0}
            if usage_obj:
                if hasattr(usage_obj, "input_tokens"):
                    usage_counts["input"] = usage_obj.input_tokens
                if hasattr(usage_obj, "output_tokens"):
                    usage_counts["output"] = usage_obj.output_tokens
                usage_counts["total"] = usage_counts["input"] + usage_counts["output"]

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": params["model"],
                        "usage": {"input": usage_counts["input"], "output": usage_counts["output"]},
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                        "continuation_count": continuation_count if continuation_count > 0 else None,
                    },
                )

                # DEBUG level: Full response with truncated values (if debug enabled)
                if self.debug:
                    response_dict = response.model_dump()  # Pydantic model → dict
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": self.name,
                            "response": self._truncate_values(response_dict),
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                            "continuation_count": continuation_count if continuation_count > 0 else None,
                        },
                    )

            # Convert to ChatResponse with accumulated output
            # If there were continuations, use the accumulated output; otherwise use response.output directly
            if continuation_count > 0:
                # Use new helper for accumulated output
                return convert_response_with_accumulated_output(
                    response, accumulated_output, continuation_count, VLLMChatResponse
                )
            # Use existing conversion for normal (non-continued) responses
            return self._convert_to_chat_response(response)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error("[PROVIDER] %s API error: %s", self.api_label, e)

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
        - Assistant messages: Simple string content with tool calls if present
        - Tool messages: Converts to text format

        Args:
            messages: List of message dicts from ChatRequest

        Returns:
            List of vLLM-formatted message objects with simple string content
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

            # Handle tool result messages
            if role == "tool":
                # For OpenAI Responses API, convert tool results to text
                tool_results_parts = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_name = tool_msg.get("tool_name", "unknown")
                    tool_content = tool_msg.get("content", "")

                    # Format as text for API
                    tool_results_parts.append(f"[Tool: {tool_name}]\n{tool_content}")
                    i += 1

                # Add as user message with combined tool results as simple string
                if tool_results_parts:
                    combined_text = "\n\n".join(tool_results_parts)
                    openai_messages.append({"role": "user", "content": combined_text})
                continue

            # Handle assistant messages
            if role == "assistant":
                reasoning_items_to_add = []  # Top-level reasoning items (before assistant message)
                text_parts = []  # Accumulate text for simple string content

                # Handle structured content (list of blocks)
                if isinstance(content, list):
                    for block in content:
                        # Handle dict blocks (from context storage)
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                text_parts.append(block.get("text", ""))
                            elif block_type == "thinking":
                                # Extract reasoning text for re-insertion
                                thinking_text = block.get("thinking", "")
                                if thinking_text:
                                    # Create reasoning item with both summary and content (vLLM requires both)
                                    reasoning_item = {
                                        "type": "reasoning",  # Required: must be "reasoning"
                                        "id": f"local_{uuid.uuid4().hex[:16]}",  # Required: unique ID
                                        "summary": [  # Required: summary array
                                            {"type": "summary_text", "text": thinking_text}
                                        ],
                                        "content": [  # Required: reasoning text content
                                            {"type": "reasoning_text", "text": thinking_text}
                                        ],
                                    }
                                    reasoning_items_to_add.append(reasoning_item)
                        elif hasattr(block, "type"):
                            # Handle ContentBlock objects (TextBlock, ThinkingBlock, etc.)
                            if block.type == "text":
                                text_parts.append(block.text)
                            elif block.type == "thinking" and hasattr(block, "thinking"):
                                # Extract reasoning text for re-insertion
                                thinking_text = block.thinking
                                if thinking_text:
                                    # Create reasoning item with both summary and content (vLLM requires both)
                                    reasoning_item = {
                                        "type": "reasoning",  # Required: must be "reasoning"
                                        "id": f"local_{uuid.uuid4().hex[:16]}",  # Required: unique ID
                                        "summary": [  # Required: summary array
                                            {"type": "summary_text", "text": thinking_text}
                                        ],
                                        "content": [  # Required: reasoning text content
                                            {"type": "reasoning_text", "text": thinking_text}
                                        ],
                                    }
                                    reasoning_items_to_add.append(reasoning_item)
                # Handle simple string content
                elif isinstance(content, str) and content:
                    text_parts.append(content)

                # Add reasoning items as TOP-LEVEL entries (before assistant message)
                for reasoning_item in reasoning_items_to_add:
                    openai_messages.append(reasoning_item)
                    logger.debug(
                        f"[PROVIDER] Added reasoning item: id={reasoning_item['id']}, "
                        f"summary_len={len(reasoning_item['summary'][0]['text'])} chars, "
                        f"content_len={len(reasoning_item['content'][0]['text'])} chars"
                    )

                # Only add assistant message if there's text content
                if text_parts:
                    combined_text = "\n".join(text_parts)
                    openai_messages.append({"role": "assistant", "content": combined_text})

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
        from amplifier_core.message_models import ReasoningBlock as ResponseReasoningBlock
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
        for block in response.output:
            # Handle both SDK objects and dictionaries
            if hasattr(block, "type"):
                block_type = block.type

                if block_type == "message":
                    # Extract text from message content
                    block_content = getattr(block, "content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if hasattr(content_item, "type") and content_item.type == "output_text":
                                text = getattr(content_item, "text", "")
                                content_blocks.append(TextBlock(text=text))
                                text_accumulator.append(text)
                                event_blocks.append(TextContent(text=text, raw=getattr(content_item, "raw", None)))
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
                            if isinstance(item, dict) and item.get("type") == "reasoning_text":
                                texts.append(item.get("text", ""))
                            elif hasattr(item, "type") and item.type == "reasoning_text":  # type: ignore[union-attr]
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
                            logger.debug("Failed to decode tool call arguments: %s", tool_input)
                    if tool_input is None:
                        tool_input = {}
                    # Ensure tool_input is dict after json.loads or default
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    content_blocks.append(ToolCallBlock(id=tool_id, name=tool_name, input=tool_input))
                    tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_input))
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
                                event_blocks.append(TextContent(text=text, raw=content_item))
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
                            if isinstance(item, dict) and item.get("type") == "reasoning_text":
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
                            logger.debug("Failed to decode tool call arguments: %s", tool_input)
                    if tool_input is None:
                        tool_input = {}
                    # Ensure tool_input is dict after json.loads or default
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    content_blocks.append(ToolCallBlock(id=tool_id, name=tool_name, input=tool_input))
                    tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_input))
                    event_blocks.append(ToolCallContent(id=tool_id, name=tool_name, arguments=tool_input, raw=block))

        # Extract usage counts
        usage_obj = response.usage if hasattr(response, "usage") else None
        usage_counts = {"input": 0, "output": 0, "total": 0}
        if usage_obj:
            if hasattr(usage_obj, "input_tokens"):
                usage_counts["input"] = usage_obj.input_tokens
            if hasattr(usage_obj, "output_tokens"):
                usage_counts["output"] = usage_obj.output_tokens
            usage_counts["total"] = usage_counts["input"] + usage_counts["output"]

        usage = Usage(
            input_tokens=usage_counts["input"],
            output_tokens=usage_counts["output"],
            total_tokens=usage_counts["total"],
        )

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
                        metadata[METADATA_INCOMPLETE_REASON] = incomplete_details.get("reason")
                    elif hasattr(incomplete_details, "reason"):
                        metadata[METADATA_INCOMPLETE_REASON] = incomplete_details.reason

        # DEBUG: Log what we're returning
        logger.info(f"[PROVIDER] Returning ChatResponse with {len(content_blocks)} content blocks")
        for i, block in enumerate(content_blocks):
            block_type = block.type if hasattr(block, "type") else "unknown"
            has_content = hasattr(block, "content") and block.content is not None
            logger.info(f"[PROVIDER]   Block {i}: type={block_type}, has_content_field={has_content}")

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
