"""Tests for json.dumps(e.body) error message pattern.

Verifies that when OpenAI SDK exceptions carry a `.body` dict, the kernel
error message contains the JSON-serialised body rather than str(e).
"""

import asyncio
import json
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
    config = {"max_retries": 0, **config_overrides}
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _mock_httpx_response(
    status_code: int = 400, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request("POST", "http://localhost:8000/v1/responses"),
    )


# ---------------------------------------------------------------------------
# Block 2: RateLimitError — body is JSON-serialised into message
# ---------------------------------------------------------------------------


def test_rate_limit_error_uses_json_body():
    """RateLimitError with a body dict should JSON-serialise the body into the message."""
    provider = _make_provider()
    body = {"error": {"message": "Rate limit exceeded", "type": "rate_limit"}}
    native = openai.RateLimitError(
        "Rate limit exceeded",
        response=_mock_httpx_response(429),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


def test_rate_limit_error_no_body_uses_str():
    """RateLimitError with body=None should fall back to str(e)."""
    provider = _make_provider()
    native = openai.RateLimitError(
        "Rate limit exceeded",
        response=_mock_httpx_response(429),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    # Should contain string representation, not 'null'
    assert "Rate limit" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Block 3: AuthenticationError — body is JSON-serialised into message
# ---------------------------------------------------------------------------


def test_authentication_error_uses_json_body():
    """AuthenticationError with a body dict should JSON-serialise the body."""
    provider = _make_provider()
    body = {"error": {"message": "Invalid API key", "type": "auth_error"}}
    native = openai.AuthenticationError(
        "Invalid API key",
        response=_mock_httpx_response(401),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


# ---------------------------------------------------------------------------
# Block 4: BadRequestError — body is JSON-serialised; keyword matching on raw str
# ---------------------------------------------------------------------------


def test_bad_request_context_length_uses_json_body():
    """BadRequestError with 'context length' and body uses JSON body in message."""
    provider = _make_provider()
    body = {"error": {"message": "context length exceeded", "type": "invalid_request"}}
    native = openai.BadRequestError(
        "This model's maximum context length is 128000 tokens",
        response=_mock_httpx_response(400),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContextLengthError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


def test_bad_request_content_filter_uses_json_body():
    """BadRequestError with 'content filter' and body uses JSON body in message."""
    provider = _make_provider()
    body = {"error": {"message": "content filter triggered", "type": "invalid_request"}}
    native = openai.BadRequestError(
        "Your request was rejected: content filter triggered.",
        response=_mock_httpx_response(400),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


def test_bad_request_invalid_uses_json_body():
    """BadRequestError generic with body uses JSON body in message."""
    provider = _make_provider()
    body = {"error": {"message": "invalid temperature", "type": "invalid_request"}}
    native = openai.BadRequestError(
        "Invalid parameter: temperature must be between 0 and 2",
        response=_mock_httpx_response(400),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


def test_bad_request_maximum_context_keyword():
    """BadRequestError with 'maximum context' keyword triggers ContextLengthError."""
    provider = _make_provider()
    native = openai.BadRequestError(
        "maximum context length exceeded",
        response=_mock_httpx_response(400),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(provider.complete(_simple_request()))


# ---------------------------------------------------------------------------
# Block 5: APIStatusError — body is JSON-serialised into message
# ---------------------------------------------------------------------------


def test_api_status_error_403_uses_json_body():
    """APIStatusError 403 with body uses JSON body in message."""
    provider = _make_provider()
    body = {"error": {"message": "Access denied", "type": "permission_error"}}
    native = openai.APIStatusError(
        "Access denied",
        response=_mock_httpx_response(403),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.AccessDeniedError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


def test_api_status_error_404_uses_json_body():
    """APIStatusError 404 with body uses JSON body in message."""
    provider = _make_provider()
    body = {"error": {"message": "Model not found", "type": "not_found"}}
    native = openai.APIStatusError(
        "Model not found",
        response=_mock_httpx_response(404),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.NotFoundError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


def test_api_status_error_5xx_uses_json_body():
    """APIStatusError 500 with body uses JSON body in message."""
    provider = _make_provider()
    body = {"error": {"message": "Internal error", "type": "server_error"}}
    native = openai.APIStatusError(
        "Internal server error",
        response=_mock_httpx_response(500),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


def test_api_status_error_other_uses_json_body():
    """APIStatusError with other status code and body uses JSON body in message."""
    provider = _make_provider()
    body = {"error": {"message": "Payment required", "type": "billing_error"}}
    native = openai.APIStatusError(
        "Payment required",
        response=_mock_httpx_response(402),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


# ---------------------------------------------------------------------------
# Block 8: Exception catch-all — body is JSON-serialised with fallback
# ---------------------------------------------------------------------------


def test_generic_exception_with_body_uses_json():
    """Exception with .body attr uses json.dumps(body)."""
    provider = _make_provider()

    class CustomError(Exception):
        def __init__(self, msg, body=None):
            super().__init__(msg)
            self.body = body

    body = {"error": "unexpected"}
    native = CustomError("Something broke", body=body)
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert json.dumps(body) in str(err)


def test_generic_exception_without_body_uses_str():
    """Exception without .body falls back to str(e)."""
    provider = _make_provider()
    native = RuntimeError("Something unexpected")
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert "Something unexpected" in str(err)


def test_generic_exception_empty_message_uses_type_name():
    """Exception with empty str(e) uses type(e).__name__ fallback."""
    provider = _make_provider()
    native = RuntimeError()
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert "RuntimeError" in str(err)
