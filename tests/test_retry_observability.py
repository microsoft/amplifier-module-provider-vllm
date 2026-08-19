"""A call that retried internally must say so on the way out.

From the forensic report on session `eec9ae98`:

> **A 30-minute failure was logged as success.** The resume-2 turn spent
> 14:19:26 -> 14:49:44 on three sequential 600 s timeouts. The eventual response
> recorded `duration_ms: 1818942` (30.3 min) with `status: ok` -- one
> "successful" call swallowing three timeouts, hiding the failure from any
> latency metric.

`elapsed_ms` already spans the whole retry loop, so a long duration was
indistinguishable from a slow model. `provider:retry` events exist, but they are
a separate stream that has to be joined by hand -- and `retry_with_backoff` only
invokes `on_retry` *before a retry sleep*, so a terminal failure (retries
exhausted) and a non-retryable error emit nothing at all. The one event every
consumer already reads, `llm:response`, said nothing.

It now carries `retries`, on success and on both error paths.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


class FakeHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self) -> None:
        self.hooks = FakeHooks()


class DummyResponse:
    def __init__(self) -> None:
        self.output: list[Any] = []
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = "completed"
        self.id = "resp_test"


def _provider() -> tuple[VLLMProvider, FakeCoordinator]:
    provider = VLLMProvider(
        base_url="http://localhost:8000/v1",
        config={
            "max_retries": 3,
            "min_retry_delay": 0.01,
            "max_retry_delay": 1.0,
            "use_streaming": False,
        },
    )
    coordinator = FakeCoordinator()
    provider.coordinator = coordinator  # type: ignore[assignment]
    return provider, coordinator


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(
        request=httpx.Request("POST", "http://localhost:8000/v1/responses")
    )


def _responses(coordinator: FakeCoordinator) -> list[dict[str, Any]]:
    return [
        payload for name, payload in coordinator.hooks.events if name == "llm:response"
    ]


def test_a_success_that_swallowed_retries_reports_them() -> None:
    """The incident's shape: one `status: ok` hiding three timeouts."""
    provider, coordinator = _provider()
    provider.client.responses.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=[_timeout_error(), _timeout_error(), DummyResponse()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(provider.complete(_request()))

    responses = _responses(coordinator)
    assert len(responses) == 1
    assert responses[0]["status"] == "ok"
    assert responses[0]["retries"] == 2, (
        "a success that cost three attempts reported as a clean single call; "
        "duration alone cannot distinguish it from a slow model"
    )


def test_a_clean_success_reports_no_retries() -> None:
    """The field must be present and zero, not absent.

    An absent key is indistinguishable from an old producer; consumers should
    never have to guess whether zero means "none" or "not reported".
    """
    provider, coordinator = _provider()
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())  # type: ignore[method-assign]

    asyncio.run(provider.complete(_request()))

    responses = _responses(coordinator)
    assert responses[0]["status"] == "ok"
    assert responses[0]["retries"] == 0


def test_a_terminal_failure_reports_what_it_cost() -> None:
    """`on_retry` never fires for the final attempt, so this is the only record.

    `retry_with_backoff` raises without notifying once retries are exhausted --
    the most important failure in the sequence is the one the retry stream never
    sees.
    """
    provider, coordinator = _provider()
    provider.client.responses.create = AsyncMock(side_effect=_timeout_error())  # type: ignore[method-assign]

    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(kernel_errors.LLMError),
    ):
        asyncio.run(provider.complete(_request()))

    responses = _responses(coordinator)
    assert responses[-1]["status"] == "error"
    assert responses[-1]["retries"] == 3, (
        "a failure after four attempts must record the three retries it burned"
    )


def test_a_non_retryable_failure_reports_zero() -> None:
    """Raised immediately, never retried -- and `on_retry` never fires here either."""
    provider, coordinator = _provider()
    provider.client.responses.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=openai.AuthenticationError(
            "Invalid key",
            response=httpx.Response(
                status_code=401,
                request=httpx.Request("POST", "http://localhost:8000/v1/responses"),
            ),
            body=None,
        )
    )

    with pytest.raises(kernel_errors.AuthenticationError):
        asyncio.run(provider.complete(_request()))

    responses = _responses(coordinator)
    assert responses[-1]["status"] == "error"
    assert responses[-1]["retries"] == 0


def test_duration_and_retries_are_reported_together() -> None:
    """Neither number is actionable alone.

    30 minutes with `retries: 0` is a slow model; 30 minutes with `retries: 3`
    is three dead connections. The incident could not tell those apart.
    """
    provider, coordinator = _provider()
    provider.client.responses.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=[_timeout_error(), DummyResponse()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(provider.complete(_request()))

    payload = _responses(coordinator)[0]
    assert {"status", "duration_ms", "retries"} <= payload.keys()
    assert payload["retries"] == 1
