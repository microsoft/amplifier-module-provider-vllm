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

            # Create ChatResponse
            return ChatResponse(
                content=[TextBlock(text=text)],
            )

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

        logger.info(
            f"[PROVIDER] vLLM chat API call - model: {params['model']}, messages: {len(messages)}, max_tokens: {params['max_tokens']}"
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

            # Extract text from response
            text = response.choices[0].message.content if response.choices else ""

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

            # Create ChatResponse
            return ChatResponse(
                content=[TextBlock(text=text)],
            )

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
