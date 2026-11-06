"""Actual OpenAI SDK transport and canonical vLLM model IDs."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core.llm_errors import (
    AuthenticationError,
    LLMError,
    ProviderUnavailableError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_vllm import VLLMProvider
from amplifier_module_provider_vllm._model_listing import ModelFilters

IDS = ["production-chat", "Team/Custom:7b-Q4", "Team/Custom:70b-Q8"]


def make(handler, **config):
    client = openai.AsyncOpenAI(
        base_url="https://vllm.invalid/v1",
        api_key="test-key",
        default_headers={"X-Tenant": "a"},
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return VLLMProvider(client=client, config={"max_retries": 0, **config})


@pytest.mark.asyncio
async def test_exact_ids_filters_binding_and_inference():
    calls = []
    cfg = {"include": ["Team/*"]}

    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["x-tenant"] == "a"
        if request.method == "GET":
            return httpx.Response(
                200, json={"data": [{"id": mid, "object": "model"} for mid in IDS]}
            )
        body = json.loads(request.content)
        assert body["model"] == "hidden-exact-alias"
        assert body["max_output_tokens"] == 100
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body["model"],
                "output": [],
            },
        )

    p = make(handler, model_filters=cfg, use_streaming=False)
    cfg["include"].clear()
    try:
        assert [m.id for m in await p.list_models()] == sorted(IDS[1:])
        assert [m.id for m in await p.list_models(filters=None)] == sorted(IDS[1:])
        assert [m.id for m in await p.list_models(filters={})] == sorted(IDS)
        assert [
            m.id
            for m in await p.list_models(
                filters={"include": ["TEAM/*"], "exclude": ["*:70B-*"]}
            )
        ] == [IDS[1]]
        assert await p.list_models(filters={"include": []}) == []
        info = p.get_model_list_info()
        assert info["source"] == "provider-api"
        info["default_filters"]["include"].clear()
        assert p.get_model_list_info()["default_filters"]["include"] == ["Team/*"]
        await p.complete(
            ChatRequest(
                model="hidden-exact-alias",
                max_output_tokens=100,
                messages=[Message(role="user", content="test")],
            ),
            model="hidden-exact-alias",
        )
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_current_card_limits_not_cache_or_defaults():
    responses = [
        {"data": [{"id": "custom", "max_model_len": 8192, "max_output_tokens": 500}]},
        {"data": [{"id": "custom"}]},
        {"data": []},
    ]
    p = make(lambda r: httpx.Response(200, json=responses.pop(0)))
    try:
        m = (await p.list_models())[0]
        assert (m.context_window, m.max_output_tokens) == (8192, 500)
        m = (await p.list_models())[0]
        assert (m.context_window, m.max_output_tokens, m.capabilities) == (0, 0, [])
        assert await p.list_models() == []
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [{}, {"data": None}, {"data": [{}]}, {"data": [{"id": ""}]}, {"data": [{"id": 5}]}],
)
async def test_malformed_not_empty(payload):
    p = make(lambda r: httpx.Response(200, json=payload))
    try:
        with pytest.raises(LLMError):
            await p.list_models()
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_errors_sanitized_cancel_propagates():
    p = make(lambda r: httpx.Response(401, json={"error": {"message": "SECRET"}}))
    try:
        with pytest.raises(AuthenticationError) as caught:
            await p.list_models()
        assert "SECRET" not in str(caught.value)
    finally:
        await p.close()

    async def cancel(r):
        raise asyncio.CancelledError()

    p = make(cancel)
    try:
        with pytest.raises(asyncio.CancelledError):
            await p.list_models()
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_all_pages_and_later_failure():
    second = SimpleNamespace(
        data=[SimpleNamespace(id="two")], has_next_page=lambda: False
    )
    first = SimpleNamespace(
        data=[SimpleNamespace(id="one")],
        has_next_page=lambda: True,
        get_next_page=AsyncMock(return_value=second),
    )
    p = VLLMProvider(
        client=SimpleNamespace(
            models=SimpleNamespace(list=AsyncMock(return_value=first))
        )
    )
    assert [m.id for m in await p.list_models()] == ["one", "two"]
    first.get_next_page.side_effect = openai.APIConnectionError(
        request=httpx.Request("GET", "https://vllm.invalid/v1/models")
    )
    with pytest.raises(ProviderUnavailableError):
        await p.list_models()


@pytest.mark.parametrize(
    "value",
    [None, [], {"include": "*"}, {"exclude": [1]}, {"hide_dated": 1}, {"unknown": []}],
)
def test_invalid_filters(value):
    with pytest.raises((TypeError, ValueError)):
        VLLMProvider(base_url="https://vllm.invalid", config={"model_filters": value})


def test_common_vectors():
    fixture = ["Custom:Q4", "Custom:Q8", "future", "future-2099-01-01"]
    assert ModelFilters.parse({}).select(fixture) == sorted(fixture)
    assert ModelFilters.parse({"include": ["CUSTOM:*"], "exclude": ["*:q8"]}).select(
        fixture
    ) == ["Custom:Q4"]
    assert ModelFilters.parse({"hide_dated": True}).select(fixture) == [
        "Custom:Q4",
        "Custom:Q8",
        "future",
    ]
    assert ModelFilters.parse({"include": []}).select(fixture) == []
