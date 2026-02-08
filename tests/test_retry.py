"""Phase 2: Retry pattern tests.

Verifies exponential backoff, jitter, retry-after > max_delay fast-fail,
non-retryable errors propagating immediately, provider:retry event emission,
and final failure raising the kernel error type (not RuntimeError).

vLLM is typically a local server, so retry primarily targets connection/timeout errors.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
import openai
from amplifier_core import ModuleCoordinator
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config=config)
    return provider


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retryable_error_retried_then_succeeds():
    """Retryable errors are retried and success on later attempt is returned."""
    provider = _make_provider()
    native_error = openai.RateLimitError(
        "Rate limit",
        response=_mock_httpx_response(429),
        body=None,
    )

    # Fail twice, succeed on third
    provider.client.responses.create = AsyncMock(
        side_effect=[native_error, native_error, DummyResponse()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = asyncio.run(provider.complete(_simple_request()))

    assert result is not None
    assert provider.client.responses.create.await_count == 3
    # Two sleeps (one per retry)
    assert mock_sleep.await_count == 2


def test_non_retryable_error_not_retried():
    """Non-retryable errors (e.g., AuthenticationError) raise immediately."""
    provider = _make_provider()
    native = openai.AuthenticationError(
        "Invalid key",
        response=_mock_httpx_response(401),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import pytest

    with pytest.raises(kernel_errors.AuthenticationError):
        asyncio.run(provider.complete(_simple_request()))

    # Only one attempt — no retries for non-retryable
    assert provider.client.responses.create.await_count == 1


def test_retry_after_exceeds_max_delay_raises_immediately():
    """If retry_after > max_retry_delay, raise immediately (fail fast)."""
    provider = _make_provider(max_retry_delay=10.0)
    native = openai.RateLimitError(
        "Rate limit",
        response=_mock_httpx_response(429, headers={"retry-after": "120"}),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import pytest

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    # retry_after=120 > max_retry_delay=10 → only 1 attempt
    assert provider.client.responses.create.await_count == 1
    assert exc_info.value.retry_after == 120.0


def test_provider_retry_event_emitted():
    """provider:retry event is emitted on each retry with correct fields."""
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())

    native = openai.RateLimitError(
        "Rate limit",
        response=_mock_httpx_response(429),
        body=None,
    )
    # Fail twice, succeed on third
    provider.client.responses.create = AsyncMock(
        side_effect=[native, native, DummyResponse()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(provider.complete(_simple_request()))

    retry_events = [
        (name, payload)
        for name, payload in provider.coordinator.hooks.events
        if name == "provider:retry"
    ]
    assert len(retry_events) == 2

    # First retry
    assert retry_events[0][1]["provider"] == "vllm"
    assert retry_events[0][1]["attempt"] == 1
    assert retry_events[0][1]["error_type"] == "RateLimitError"
    assert "delay" in retry_events[0][1]

    # Second retry
    assert retry_events[1][1]["attempt"] == 2


def test_exponential_backoff_delays():
    """Retry delays follow exponential backoff pattern."""
    provider = _make_provider(
        min_retry_delay=1.0,
        max_retry_delay=60.0,
        retry_jitter=False,
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())

    native = openai.RateLimitError(
        "Rate limit",
        response=_mock_httpx_response(429),
        body=None,
    )
    # Fail 3 times, succeed on 4th
    provider.client.responses.create = AsyncMock(
        side_effect=[native, native, native, DummyResponse()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(provider.complete(_simple_request()))

    retry_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "provider:retry"
    ]
    assert len(retry_events) == 3

    # Without jitter, delays should be exactly: 1s, 2s, 4s
    assert retry_events[0]["delay"] == 1.0
    assert retry_events[1]["delay"] == 2.0
    assert retry_events[2]["delay"] == 4.0


def test_max_retries_exhausted_raises_kernel_error():
    """After exhausting retries, the kernel error type is raised (not RuntimeError)."""
    provider = _make_provider(max_retries=2)
    native = openai.RateLimitError(
        "Rate limit",
        response=_mock_httpx_response(429),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import pytest

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(kernel_errors.RateLimitError):
            asyncio.run(provider.complete(_simple_request()))

    # 1 initial + 2 retries = 3 total attempts
    assert provider.client.responses.create.await_count == 3


def test_timeout_error_retried():
    """LLMTimeoutError (from asyncio.TimeoutError) is retried."""
    provider = _make_provider(max_retries=2)

    # Timeout once, then succeed
    provider.client.responses.create = AsyncMock(
        side_effect=[asyncio.TimeoutError(), DummyResponse()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(provider.complete(_simple_request()))

    assert result is not None
    assert provider.client.responses.create.await_count == 2


def test_provider_unavailable_retried():
    """ProviderUnavailableError (from 5xx) is retried."""
    provider = _make_provider(max_retries=2)

    native_500 = openai.APIStatusError(
        "Server error",
        response=_mock_httpx_response(500),
        body=None,
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[native_500, DummyResponse()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(provider.complete(_simple_request()))

    assert result is not None
    assert provider.client.responses.create.await_count == 2


def test_sdk_retries_disabled():
    """OpenAI SDK max_retries is set to 0 (we handle retries ourselves)."""
    provider = _make_provider()
    assert provider.client.max_retries == 0
