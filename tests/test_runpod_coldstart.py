"""RunPod cold-start (pod resume) tolerance tests.

A RunPod serverless / persistent pod that has cooled off returns an HTML
"Waiting for service to respond" holding page (and, while the model is still
loading, plain 404s for the model group) instead of a normal API response.

These must be treated as TRANSIENT / retryable so the shared
retry_with_backoff() engages and the request survives the warm-up window,
instead of hard-failing the whole (sub-)session with a non-retryable
NotFoundError.

Guard: a genuine 404 that carries a real JSON error body (model truly absent)
must STAY non-retryable so real misconfiguration still fails fast.
"""

import asyncio
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider

RUNPOD_HOLDING_PAGE = (
    b"<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"/>"
    b"<title>Waiting for service to respond \xe2\x80\x94 RunPod</title></head>"
    b"<body>Waiting for service to respond</body></html>"
)


def _make_provider(**config_overrides) -> VLLMProvider:
    config = {
        "max_retries": 2,
        "min_retry_delay": 0.01,
        "max_retry_delay": 0.05,
        "use_streaming": False,
        **config_overrides,
    }
    return VLLMProvider(base_url="https://pod-4000.proxy.runpod.net/v1", config=config)


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _html_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "text/html"},
        content=RUNPOD_HOLDING_PAGE,
        request=httpx.Request("POST", "https://pod-4000.proxy.runpod.net/v1/responses"),
    )


def _json_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://pod-4000.proxy.runpod.net/v1/responses"),
    )


def test_coldstart_404_holding_page_is_retryable():
    """A 404 whose body is the RunPod HTML holding page must be retried, not fatal."""
    provider = _make_provider(max_retries=2)
    native = openai.NotFoundError(
        "Not Found", response=_html_response(404), body=None
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_request()))

    # It exhausted retries (initial + 2) rather than failing on the first attempt.
    assert provider.client.responses.create.await_count == 3
    # And the terminal error is flagged retryable (transient), not a hard NotFound.
    assert getattr(exc_info.value, "retryable", False) is True


def test_coldstart_503_holding_page_is_retryable():
    """A 5xx holding page is likewise transient and retried."""
    provider = _make_provider(max_retries=2)
    native = openai.APIStatusError(
        "Service Unavailable", response=_html_response(503), body=None
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError):
        asyncio.run(provider.complete(_request()))

    assert provider.client.responses.create.await_count == 3


def test_real_404_json_body_stays_fatal():
    """Guard: a genuine 404 with a JSON error body must fail fast (non-retryable)."""
    provider = _make_provider(max_retries=2)
    native = openai.NotFoundError(
        "model not found",
        response=_json_response(404),
        body={"error": {"message": "The model `nope` does not exist", "code": 404}},
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.NotFoundError):
        asyncio.run(provider.complete(_request()))

    # No retries: real misconfiguration fails on the first attempt.
    assert provider.client.responses.create.await_count == 1
