"""Phase 2: Error translation tests.

Verifies that native OpenAI SDK errors (used by vLLM) are translated to kernel
error types with correct attributes (provider, status_code, retryable, __cause__).
"""

import asyncio
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> VLLMProvider:
    """Create a provider with retries disabled so errors propagate immediately."""
    config = {"max_retries": 0, **config_overrides}
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config=config)
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _mock_httpx_response(
    status_code: int = 429, headers: dict | None = None
) -> httpx.Response:
    """Build a minimal httpx.Response for OpenAI SDK error constructors."""
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request("POST", "http://localhost:8000/v1/responses"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rate_limit_error_translated():
    """openai.RateLimitError -> kernel RateLimitError with retryable=True."""
    provider = _make_provider()
    native = openai.RateLimitError(
        "Rate limit exceeded",
        response=_mock_httpx_response(429),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.status_code == 429
    assert err.retryable is True
    assert err.__cause__ is native


def test_rate_limit_error_parses_retry_after_header():
    """Retry-After header value is parsed into retry_after attribute."""
    provider = _make_provider()
    native = openai.RateLimitError(
        "Rate limit exceeded",
        response=_mock_httpx_response(429, headers={"retry-after": "30"}),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retry_after == 30.0


def test_authentication_error_translated():
    """openai.AuthenticationError -> kernel AuthenticationError (retryable=False)."""
    provider = _make_provider()
    native = openai.AuthenticationError(
        "Invalid API key",
        response=_mock_httpx_response(401),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.retryable is False
    assert err.__cause__ is native


def test_bad_request_context_length():
    """openai.BadRequestError with 'context length' -> kernel ContextLengthError."""
    provider = _make_provider()
    native = openai.BadRequestError(
        "This model's maximum context length is 128000 tokens",
        response=_mock_httpx_response(400),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContextLengthError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.status_code == 400
    assert err.__cause__ is native


def test_bad_request_content_filter():
    """openai.BadRequestError with 'content filter' -> kernel ContentFilterError."""
    provider = _make_provider()
    native = openai.BadRequestError(
        "Your request was rejected as a result of our safety system. Content filter triggered.",
        response=_mock_httpx_response(400),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.provider == "vllm"
    assert exc_info.value.__cause__ is native


def test_bad_request_invalid_request():
    """openai.BadRequestError with generic message -> kernel InvalidRequestError."""
    provider = _make_provider()
    native = openai.BadRequestError(
        "Invalid parameter: temperature must be between 0 and 2",
        response=_mock_httpx_response(400),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.status_code == 400
    assert err.__cause__ is native


def test_api_status_error_5xx_translated():
    """openai.APIStatusError with status >= 500 -> kernel ProviderUnavailableError."""
    provider = _make_provider()
    native = openai.APIStatusError(
        "Internal server error",
        response=_mock_httpx_response(500),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.status_code == 500
    assert err.retryable is True
    assert err.__cause__ is native


def test_timeout_error_translated():
    """asyncio.TimeoutError -> kernel LLMTimeoutError (retryable=True)."""
    provider = _make_provider()
    native = asyncio.TimeoutError()
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMTimeoutError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.retryable is True
    assert err.__cause__ is native


def test_generic_exception_translated():
    """Unknown Exception -> kernel LLMError (retryable=True, unknown defaults to retryable)."""
    provider = _make_provider()
    native = RuntimeError("Something unexpected")
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.retryable is True
    assert err.__cause__ is native
    assert "Something unexpected" in str(err)


def test_llm_response_error_event_emitted_on_kernel_error():
    """llm:response error event is still emitted when a kernel error is raised."""
    from typing import cast

    from amplifier_core import ModuleCoordinator

    class FakeHooks:
        def __init__(self):
            self.events: list[tuple[str, dict]] = []

        async def emit(self, name: str, payload: dict) -> None:
            self.events.append((name, payload))

    class FakeCoordinator:
        def __init__(self):
            self.hooks = FakeHooks()

    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    native = openai.AuthenticationError(
        "Bad key",
        response=_mock_httpx_response(401),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.AuthenticationError):
        asyncio.run(provider.complete(_simple_request()))

    # Verify llm:response error event was emitted
    error_events = [
        (name, payload)
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "error"
    ]
    assert len(error_events) >= 1
    assert error_events[0][1]["provider"] == "vllm"
