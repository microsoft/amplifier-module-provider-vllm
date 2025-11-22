"""vLLM provider module for Amplifier.

Integrates with vLLM's OpenAI-compatible API for open-weight models.
Supports completions endpoint with streaming and logprobs.
"""

__all__ = ["mount", "VLLMProvider"]

import asyncio
import logging
import os
import time
from typing import Any

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import TextBlock
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the vLLM provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including base_url and optional api_key

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get base URL from config (required for vLLM)
    base_url = config.get("base_url")
    if not base_url:
        logger.error("No base_url found for vLLM provider - this is required")
        return None

    # API key is optional for vLLM (many instances don't require it)
    api_key = config.get("api_key") or os.environ.get("VLLM_API_KEY", "EMPTY")

    provider = VLLMProvider(base_url, api_key, config, coordinator)
    await coordinator.mount("providers", provider, name="vllm")
    logger.info(f"Mounted VLLMProvider (base_url={base_url})")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class VLLMProvider:
    """vLLM API integration via OpenAI-compatible API.

    Provides open-weight models (e.g., gpt-oss-20b) with support for:
    - Text completion via /v1/completions endpoint
    - Streaming responses
    - Log probabilities
    """

    name = "vllm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        """
        Initialize vLLM provider.

        Args:
            base_url: vLLM server base URL (e.g., http://192.168.128.5:8000/v1)
            api_key: API key (often "EMPTY" for local vLLM instances)
            config: Additional configuration
            coordinator: Module coordinator for event emission
        """
        # Ensure base_url ends with /v1
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.config = config or {}
        self.coordinator = coordinator
        self.base_url = base_url
        self.default_model = self.config.get("default_model", "openai/gpt-oss-20b")
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.7)
        self.priority = self.config.get("priority", 100)
        self.debug = self.config.get("debug", False)
        self.raw_debug = self.config.get("raw_debug", False)
        self.debug_truncate_length = self.config.get("debug_truncate_length", 180)
        self.timeout = self.config.get("timeout", 300.0)
        self.use_completions = self.config.get("use_completions", True)  # Default to completions API

        # Debug: Log which API will be used
        logger.info(
            f"[PROVIDER] VLLMProvider initialized: use_completions={self.use_completions}, "
            f"model={self.default_model}, base_url={self.base_url}"
        )

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """Recursively truncate string values in nested structures.

        Args:
            obj: Any JSON-serializable structure
            max_length: Maximum string length

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
        return obj

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert messages to a single prompt string for completions API.

        Args:
            messages: List of Message objects

        Returns:
            Single prompt string
        """
        # Simple concatenation strategy - customize for specific models
        parts = []
        for msg in messages:
            role = msg.role.upper()

            # Handle content as string or list of content blocks
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Extract text from content blocks (only TextBlock has .text)
                text_parts = []
                for block in msg.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text" and hasattr(block, "text"):
                        text_parts.append(block.text)  # type: ignore[union-attr]
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                content = " ".join(text_parts)
            else:
                content = str(msg.content)

            parts.append(f"[{role}]\n{content}\n")

        return "\n".join(parts)

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content
        """
        use_completions = self.use_completions or kwargs.get("use_completions", True)

        if use_completions:
            return await self._complete_with_completions_api(request, **kwargs)
        return await self._complete_with_chat_api(request, **kwargs)

    def parse_tool_calls(self, response: ChatResponse) -> list:
        """Parse tool calls from ChatResponse.

        Args:
            response: Typed chat response

        Returns:
            List of tool calls from the response
        """
        if not response.tool_calls:
            return []
        return response.tool_calls

    def _parse_response_with_thinking(
        self, text: str, tool_calls: Any = None, reasoning_content: str | None = None
    ) -> ChatResponse:
        """Parse response to create proper content blocks.

        vLLM's chat completions API provides:
        - text: The actual response content
        - reasoning_content: The model's internal thinking/reasoning (separate field)
        - tool_calls: Tool calls made by the model

        This method properly separates these into ThinkingBlock, TextBlock, and ToolCallBlock.

        Args:
            text: Response content from the model
            tool_calls: Optional tool calls from the API response
            reasoning_content: Optional reasoning/thinking from the API response

        Returns:
            ChatResponse with separated thinking, response blocks, and tool calls
        """
        import json
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock

        content_blocks = []
        parsed_tool_calls = []

        # Add thinking block if reasoning_content is present (proper API field)
        if reasoning_content:
            content_blocks.append(
                ThinkingBlock(
                    thinking=reasoning_content,
                    signature=None,
                    visibility="internal",
                )
            )
            logger.debug(f"[PROVIDER] Added thinking block from reasoning_content ({len(reasoning_content)} chars)")

        # Check for fallback marker-based thinking (for models that don't use reasoning_content)
        # This handles cases where the model embeds thinking in the text with markers
        elif text and "assistantfinal" in text:
            parts = text.split("assistantfinal", 1)
            thinking_text = parts[0].strip()
            response_text = parts[1].strip() if len(parts) > 1 else ""

            # Add thinking block if there's thinking content
            if thinking_text:
                content_blocks.append(
                    ThinkingBlock(
                        thinking=thinking_text,
                        signature=None,
                        visibility="internal",
                    )
                )
                logger.debug(f"[PROVIDER] Parsed thinking block from marker ({len(thinking_text)} chars)")

            # Override text with just the response part
            text = response_text

        # Add text response block if present
        if text:
            content_blocks.append(TextBlock(text=text))
            logger.debug(f"[PROVIDER] Added response block ({len(text)} chars)")

        # Parse tool calls if present
        if tool_calls:
            for tc in tool_calls:
                tool_id = tc.id if hasattr(tc, "id") else ""
                tool_name = tc.function.name if hasattr(tc, "function") else ""
                tool_args_str = tc.function.arguments if hasattr(tc, "function") else "{}"

                # Parse arguments JSON string to dict
                try:
                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments: {tool_args_str}")
                    tool_args = {}

                content_blocks.append(ToolCallBlock(id=tool_id, name=tool_name, input=tool_args))
                parsed_tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_args))
                logger.debug(f"[PROVIDER] Parsed tool call: {tool_name}")

        return ChatResponse(
            content=content_blocks,
            tool_calls=parsed_tool_calls if parsed_tool_calls else None,
        )

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to OpenAI format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of OpenAI-formatted tool definitions (vLLM uses OpenAI format)
        """
        vllm_tools = []
        for tool in tools:
            vllm_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    },
                }
            )
        return vllm_tools

    async def _complete_with_completions_api(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle completion using /v1/completions endpoint.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content
        """
        logger.debug(f"Received ChatRequest with {len(request.messages)} messages (debug={self.debug})")

        # Convert messages to prompt
        prompt = self._messages_to_prompt(request.messages)

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "prompt": prompt,
            "max_tokens": request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature or kwargs.get("temperature", self.temperature),
            "stream": False,  # Start with non-streaming
        }

        # Add optional parameters
        if logprobs := kwargs.get("logprobs"):
            params["logprobs"] = logprobs
        if echo := kwargs.get("echo"):
            params["echo"] = echo
        if stop := kwargs.get("stop"):
            params["stop"] = stop

        logger.info(
            f"[PROVIDER] vLLM completions API call - model: {params['model']}, prompt_length: {len(prompt)}, max_tokens: {params['max_tokens']}"
        )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "vllm",
                    "model": params["model"],
                    "endpoint": "completions",
                    "prompt_length": len(prompt),
                },
            )

            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "vllm",
                        "request": self._truncate_values(params),
                    },
                )

            if self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": "vllm",
                        "params": params,
                    },
                )

        start_time = time.time()

        # Call vLLM API
        try:
            response = await asyncio.wait_for(self.client.completions.create(**params), timeout=self.timeout)
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from vLLM completions API")

            # Extract text from response
            text = response.choices[0].text if response.choices else ""

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                usage_data = {
                    "input": response.usage.prompt_tokens if response.usage else 0,
                    "output": response.usage.completion_tokens if response.usage else 0,
                }

                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "vllm",
                        "model": params["model"],
                        "usage": usage_data,
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                if self.debug:
                    response_dict = response.model_dump()
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "vllm",
                            "response": self._truncate_values(response_dict),
                        },
                    )

                if self.debug and self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": "vllm",
                            "response": response.model_dump(),
                        },
                    )

            # Create ChatResponse (completions API has no tool or reasoning support)
            return self._parse_response_with_thinking(text, None, None)

        except TimeoutError:
            logger.error(f"[PROVIDER] vLLM API timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"[PROVIDER] vLLM API error: {e}")
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "vllm",
                        "model": params["model"],
                        "status": "error",
                        "error": str(e),
                    },
                )
            raise

    async def _complete_with_chat_api(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle completion using /v1/chat/completions endpoint.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content
        """
        logger.debug(f"Received ChatRequest with {len(request.messages)} messages for chat API")

        # Convert messages to chat format
        messages = []
        for msg in request.messages:
            # Handle content as string or list of content blocks
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Extract text from content blocks (only TextBlock has .text)
                text_parts = []
                for block in msg.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text" and hasattr(block, "text"):
                        text_parts.append(block.text)  # type: ignore[union-attr]
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                content = " ".join(text_parts)
            else:
                content = str(msg.content)

            messages.append({"role": msg.role, "content": content})

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": messages,
            "max_tokens": request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature or kwargs.get("temperature", self.temperature),
            "stream": False,
        }

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)
            # Add tool_choice if specified (defaults to auto)
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
            logger.info(f"[PROVIDER] Added {len(request.tools)} tools to vLLM request")

        logger.info(
            f"[PROVIDER] vLLM chat API call - model: {params['model']}, messages: {len(messages)}, max_tokens: {params['max_tokens']}, tools: {len(request.tools) if request.tools else 0}"
        )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "vllm",
                    "model": params["model"],
                    "endpoint": "chat",
                    "message_count": len(messages),
                },
            )

            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "vllm",
                        "request": self._truncate_values(params),
                    },
                )

        start_time = time.time()

        # Call vLLM API
        try:
            response = await asyncio.wait_for(self.client.chat.completions.create(**params), timeout=self.timeout)
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from vLLM chat API")

            # Extract text, tool calls, and reasoning from response
            message = response.choices[0].message if response.choices else None
            text = message.content if message and message.content else ""
            tool_calls = message.tool_calls if message and hasattr(message, "tool_calls") else None
            reasoning_content = message.reasoning_content if message and hasattr(message, "reasoning_content") else None

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                usage_data = {
                    "input": response.usage.prompt_tokens if response.usage else 0,
                    "output": response.usage.completion_tokens if response.usage else 0,
                }

                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "vllm",
                        "model": params["model"],
                        "usage": usage_data,
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                if self.debug:
                    response_dict = response.model_dump()
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "vllm",
                            "response": self._truncate_values(response_dict),
                        },
                    )

            # Create ChatResponse (parse thinking blocks and tool calls if present)
            return self._parse_response_with_thinking(text, tool_calls, reasoning_content)

        except TimeoutError:
            logger.error(f"[PROVIDER] vLLM API timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"[PROVIDER] vLLM API error: {e}")
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "vllm",
                        "model": params["model"],
                        "status": "error",
                        "error": str(e),
                    },
                )
            raise
