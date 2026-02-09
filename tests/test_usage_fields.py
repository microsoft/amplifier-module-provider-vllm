"""Phase 2: Usage fields tests.

Verifies that reasoning_tokens is extracted from output_tokens_details
and that existing usage fields remain populated correctly.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider() -> VLLMProvider:
    # Use a non-GPT-OSS model to bypass Harmony token accounting
    # (which overrides usage values for gpt-oss models)
    config = {"max_retries": 0, "default_model": "meta-llama/Llama-3-8B"}
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    """Response stub with configurable usage."""

    def __init__(self, usage=None):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi there")],
            )
        ]
        self.usage = usage
        self.status = "completed"
        self.id = "resp_test"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reasoning_tokens_extracted():
    """reasoning_tokens is extracted from usage.output_tokens_details.reasoning_tokens."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        output_tokens_details=SimpleNamespace(reasoning_tokens=500),
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.reasoning_tokens == 500
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.total_tokens == 150


def test_reasoning_tokens_none_when_no_details():
    """reasoning_tokens is None when output_tokens_details is absent."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=80,
        output_tokens=20,
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.reasoning_tokens is None
    assert result.usage.input_tokens == 80
    assert result.usage.output_tokens == 20
    assert result.usage.total_tokens == 100


def test_reasoning_tokens_none_when_details_has_no_reasoning():
    """reasoning_tokens is None when output_tokens_details exists but lacks reasoning_tokens."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=50,
        output_tokens=30,
        output_tokens_details=SimpleNamespace(),  # No reasoning_tokens attr
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.reasoning_tokens is None


def test_reasoning_tokens_none_when_details_is_none():
    """reasoning_tokens is None when output_tokens_details is None."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=50,
        output_tokens=30,
        output_tokens_details=None,
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.reasoning_tokens is None


def test_existing_usage_fields_preserved():
    """Standard usage fields (input_tokens, output_tokens, total_tokens) still work."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=200,
        output_tokens=100,
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 100
    assert result.usage.total_tokens == 300


def test_reasoning_tokens_zero_is_preserved():
    """reasoning_tokens=0 is preserved (not coerced to None)."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=50,
        output_tokens=30,
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.reasoning_tokens == 0


# ---------------------------------------------------------------------------
# cache_read_tokens extraction tests
# ---------------------------------------------------------------------------


def test_cache_read_tokens_extracted():
    """cache_read_tokens is extracted from usage.input_tokens_details.cached_tokens."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        input_tokens_details=SimpleNamespace(cached_tokens=800),
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.cache_read_tokens == 800


def test_cache_read_tokens_none_when_no_input_details():
    """cache_read_tokens is None when input_tokens_details is absent."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=80,
        output_tokens=20,
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.cache_read_tokens is None


def test_cache_read_tokens_zero_is_preserved():
    """cache_read_tokens preserves 0 (0 means 'measured, value is zero')."""
    provider = _make_provider()

    usage_obj = SimpleNamespace(
        input_tokens=50,
        output_tokens=30,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(usage=usage_obj)
    )

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    # 0 is preserved — it means "measured, value is zero" (not "field absent")
    assert result.usage.cache_read_tokens == 0
