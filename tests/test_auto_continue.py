"""Caller-controlled output bounds must not cause hidden paid continuations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


def response(status):
    return SimpleNamespace(
        id="resp_fake",
        status=status,
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="partial")],
            )
        ],
        usage=SimpleNamespace(input_tokens=100, output_tokens=10, total_tokens=110),
        model_dump=lambda: {"status": status},
    )


def provider(**config):
    config = {
        "use_streaming": False,
        "max_retries": 0,
        "default_model": "test-model",
        **config,
    }
    p = VLLMProvider(base_url="http://localhost:8000/v1", config=config)
    p.client.responses.create = AsyncMock(
        side_effect=[response("incomplete"), response("completed")]
    )
    # All transport is mocked, including exact request preflight when present.
    if hasattr(p, "_guard_assembled_params_with_provider_count"):
        p._guard_assembled_params_with_provider_count = AsyncMock(return_value=100)
    return p


@pytest.mark.asyncio
async def test_bounded_call_does_not_continue_or_mutate_provider():
    p = provider()
    req = ChatRequest(
        messages=[Message(role="user", content="Summarize")], max_output_tokens=100
    )
    result = await p.complete(req, request_options={"auto_continue": False})
    assert p.client.responses.create.await_count == 1
    assert result.finish_reason == "length"
    assert result.usage.input_tokens == 100
    assert p.auto_continue is True
    params = p.client.responses.create.call_args.kwargs
    assert "auto_continue" not in params and "request_options" not in params
    assert params["max_output_tokens"] == 100
    assert "completion:auto_continue:v1" in p.get_info().capabilities


@pytest.mark.asyncio
async def test_normal_call_still_continues_and_accounts_for_both_requests():
    p = provider()
    result = await p.complete(
        ChatRequest(messages=[Message(role="user", content="Hello")])
    )
    assert p.client.responses.create.await_count == 2
    assert result.usage.input_tokens == 200


@pytest.mark.asyncio
async def test_config_can_disable_continuation_and_call_can_override():
    p = provider(auto_continue=False)
    await p.complete(
        ChatRequest(messages=[Message(role="user", content="Hello")]),
        request_options={"auto_continue": False},
        auto_continue=True,
    )
    assert p.client.responses.create.await_count == 2
    assert p.auto_continue is False


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["false", 0, None])
async def test_invalid_per_call_option_fails_before_transport(value):
    p = provider()
    with pytest.raises(ValueError, match="auto_continue"):
        await p.complete(
            ChatRequest(messages=[Message(role="user", content="Hello")]),
            auto_continue=value,
        )
    p.client.responses.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_bounded_partial_function_is_not_executable():
    from amplifier_core.llm_errors import LLMError

    p = provider()
    partial = response("incomplete")
    partial.output = [
        SimpleNamespace(
            type="function_call",
            status="incomplete",
            call_id="call-1",
            id="item-1",
            name="write",
            arguments='{"path":',
        )
    ]
    p.client.responses.create = AsyncMock(return_value=partial)
    with pytest.raises(LLMError, match="incomplete function"):
        await p.complete(
            ChatRequest(messages=[Message(role="user", content="Hello")]),
            auto_continue=False,
        )
    p.client.responses.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_and_reasoning_counters_are_not_double_counted():
    p = provider()
    responses = [response("incomplete"), response("completed")]
    for item in responses:
        item.usage.input_tokens_details = SimpleNamespace(cached_tokens=60)
        item.usage.output_tokens_details = SimpleNamespace(reasoning_tokens=4)
    p.client.responses.create = AsyncMock(side_effect=responses)
    result = await p.complete(
        ChatRequest(messages=[Message(role="user", content="Hello")])
    )
    assert result.usage.total_tokens == 220
    assert result.usage.cache_read_tokens == 120 and result.usage.reasoning_tokens == 8
