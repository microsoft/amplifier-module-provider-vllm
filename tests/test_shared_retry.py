"""Tests for shared retry_with_backoff integration (vLLM).

Verifies that the vLLM provider uses the shared RetryConfig and
retry_with_backoff from amplifier-core instead of its own retry loop,
and adopts new error types (AccessDeniedError for 403, NotFoundError for 404).
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
import openai
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import AccessDeniedError, NotFoundError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.utils.retry import RetryConfig

from amplifier_module_provider_vllm import VLLMProvider


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


class DummyResponse:
    """Minimal response stub that satisfies _convert_to_chat_response."""

    def __init__(self):
        self.output = []
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = "completed"
        self.id = "resp_test"


def _make_provider(**config_overrides) -> VLLMProvider:
    config = {
        "max_retries": 3,
        "min_retry_delay": 0.01,
        "max_retry_delay": 1.0,
        **config_overrides,
    }
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _mock_httpx_response(
    status_code: int = 429, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request("POST", "http://localhost:8000/v1/responses"),
    )


# --- Structural: uses shared RetryConfig ---


def test_provider_has_retry_config():
    """Provider should store a RetryConfig instance (not separate vars)."""
    provider = _make_provider()
    assert hasattr(provider, "_retry_config")
    assert isinstance(provider._retry_config, RetryConfig)


def test_retry_config_respects_config_values():
    """RetryConfig should be populated from provider config dict."""
    provider = _make_provider(
        max_retries=7,
        min_retry_delay=2.0,
        max_retry_delay=120.0,
        retry_jitter=False,
    )
    assert provider._retry_config.max_retries == 7
    assert provider._retry_config.min_delay == 2.0
    assert provider._retry_config.max_delay == 120.0
    assert provider._retry_config.jitter == 0.0  # False -> 0.0


def test_no_calculate_retry_delay_method():
    """_calculate_retry_delay should be removed (replaced by shared utility)."""
    provider = _make_provider()
    assert not hasattr(provider, "_calculate_retry_delay")


def test_jitter_backward_compat_bool_true():
    """retry_jitter=True (old bool format) should map to jitter=0.2."""
    provider = _make_provider(retry_jitter=True)
    assert provider._retry_config.jitter == 0.2


def test_jitter_backward_compat_bool_false():
    """retry_jitter=False (old bool format) should map to jitter=0.0."""
    provider = _make_provider(retry_jitter=False)
    assert provider._retry_config.jitter == 0.0


# --- Error type: 403 -> AccessDeniedError, 404 -> NotFoundError ---


def test_api_status_error_403_becomes_access_denied_error():
    """openai.APIStatusError with status 403 -> AccessDeniedError."""
    provider = _make_provider(max_retries=0)
    native = openai.APIStatusError(
        "Forbidden",
        response=_mock_httpx_response(403),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import pytest

    with pytest.raises(AccessDeniedError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.status_code == 403
    assert err.__cause__ is native


def test_api_status_error_404_becomes_not_found_error():
    """openai.APIStatusError with status 404 -> NotFoundError."""
    provider = _make_provider(max_retries=0)
    native = openai.APIStatusError(
        "Not found",
        response=_mock_httpx_response(404),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import pytest

    with pytest.raises(NotFoundError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.status_code == 404
    assert err.__cause__ is native


# --- Retry behavior through shared utility ---


def test_retry_with_shared_utility_succeeds():
    """Shared retry_with_backoff should retry transient errors and return on success."""
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())

    native_error = openai.RateLimitError(
        "Rate limit",
        response=_mock_httpx_response(429),
        body=None,
    )

    # Fail twice, succeed on third
    provider.client.responses.create = AsyncMock(
        side_effect=[native_error, native_error, DummyResponse()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(provider.complete(_simple_request()))

    assert result is not None
    assert provider.client.responses.create.await_count == 3

    # Should have emitted provider:retry events
    retry_events = [
        e for e in provider.coordinator.hooks.events if e[0] == "provider:retry"
    ]
    assert len(retry_events) == 2
    assert retry_events[0][1]["provider"] == "vllm"
