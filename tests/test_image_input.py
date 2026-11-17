"""Offline image regressions; SDK MockTransport is not live vLLM qualification."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import openai
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
)

from amplifier_module_provider_vllm import VLLMProvider
from amplifier_module_provider_vllm import _token_accounting as accounting
from amplifier_module_provider_vllm._images import image_diagnostic_copy
from tests.sdk_transport import httpx
from tests.test_sdk_compatibility import payload

# Tiny PNG fixture, passed unchanged (no decoder, resize, or model call).
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)
DATA_URL = f"data:image/png;base64,{PNG}"


def image(source=None):
    return ImageBlock(source=source or {
        "type": "base64", "media_type": "image/png", "data": PNG,
    })


def provider(**config):
    return VLLMProvider(base_url="http://vllm.invalid/v1", config={
        "use_streaming": False, "max_retries": 0, **config,
    })


@pytest.mark.parametrize("objects", [False, True])
def test_direct_mixed_images_preserve_order_and_history(objects):
    blocks = [
        TextBlock(text="before"), image(),
        TextBlock(text="between"),
        image({"type": "url", "url": "https://images.invalid/x.png?sig=private"}),
        TextBlock(text="after"),
    ]
    if not objects:
        blocks = [b.model_dump() for b in blocks]
    messages = [{"role": "user", "content": blocks}]
    original = copy.deepcopy(messages)
    converted = provider()._convert_messages(messages)
    assert converted == [{"role": "user", "content": [
        {"type": "input_text", "text": "before"},
        {"type": "input_image", "image_url": DATA_URL, "detail": "auto"},
        {"type": "input_text", "text": "between"},
        {"type": "input_image", "image_url": "https://images.invalid/x.png?sig=private",
         "detail": "auto"},
        {"type": "input_text", "text": "after"},
    ]}]
    assert messages == original


def test_image_only_not_dropped():
    assert provider()._convert_messages([{
        "role": "user", "content": [image().model_dump()],
    }])[0]["content"][0]["image_url"] == DATA_URL


@pytest.mark.parametrize("source", [
    {}, {"type": "base64", "data": PNG},
    {"type": "base64", "media_type": "image/png", "data": "not base64!"},
    {"type": "url", "url": "file:///private/image.png"},
    {"type": "url", "url": "https://images.invalid/\nsecret"},
    {"type": "url", "url": "data:text/plain;base64,c2VjcmV0"},
])
def test_malformed_image_is_explicit_and_does_not_swallow_following_text(source):
    result = provider()._convert_messages([{"role": "user", "content": [
        {"type": "image", "source": source}, {"type": "text", "text": "still here"},
    ]}])
    content = result[0]["content"]
    assert content[0]["type"] == "input_text"
    assert "image unavailable" in content[0]["text"].lower()
    assert content[1] == {"type": "input_text", "text": "still here"}
    assert "not base64!" not in json.dumps(result)
    assert provider()._convert_messages([{"role": "user", "content": "plain"}]) == [
        {"role": "user", "content": "plain"},
    ]


@pytest.mark.parametrize("call_id", ["call_read", None])
def test_tool_image_in_original_slot_without_stringified_pixels_or_fake_user(call_id):
    messages = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Inspect it"},
        {"type": "tool_call", "id": "call_read", "name": "read_file", "input": {"path": "x"}},
    ]}, {"role": "tool", "tool_call_id": call_id, "content": [
        {"type": "text", "text": "caption before"}, image().model_dump(),
        {"type": "text", "text": "caption after"},
    ]}]
    original = copy.deepcopy(messages)
    if call_id is None:
        with pytest.raises(InvalidRequestError, match="tool_call_id"):
            provider()._convert_messages(messages)
        assert messages == original
        return
    converted = provider()._convert_messages(messages)
    tool = converted[-1]
    assert tool["type"] == "function_call_output"
    assert tool["call_id"] == call_id
    assert tool["output"] == [
        {"type": "input_text", "text": "caption before"},
        {"type": "input_image", "image_url": DATA_URL, "detail": "auto"},
        {"type": "input_text", "text": "caption after"},
    ]
    assert not any(item.get("role") == "user" for item in converted)
    assert converted[0]["type"] == "reasoning"
    assert converted[0]["content"][0]["text"] == "Inspect it"
    assert converted[1]["call_id"] == "call_read"
    assert converted[1]["id"] == "ft_read"
    assert messages == original


def test_harmony_accounting_does_not_claim_image_count(monkeypatch, caplog):
    params = {"input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": DATA_URL, "detail": "auto"},
    ]}]}
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=321))
    compute = MagicMock(side_effect=AssertionError("text-only accounting invoked"))
    monkeypatch.setattr(accounting, "_get_harmony_encoding", compute)
    assert accounting.compute_input_tokens(params) == 0
    assert accounting.apply_token_accounting(params, response) is response
    assert response.usage.input_tokens == 321
    compute.assert_not_called()
    assert "image" in caplog.text.lower() and "unavailable" in caplog.text.lower()
    assert PNG not in caplog.text


def test_diagnostic_copy_omits_sources_and_signed_urls_without_mutating():
    value = {"input": [
        {"role": "user", "content": [
            image().model_dump(),
            {"type": "input_image", "image_url": "https://images.invalid/x?secret=private"},
        ]},
        {"type": "function_call_output", "call_id": "call_read", "output": [
            {"type": "input_image", "image_url": DATA_URL},
        ]},
    ]}
    original = copy.deepcopy(value)
    diagnostic = image_diagnostic_copy(value)
    assert value == original
    assert PNG not in json.dumps(diagnostic)
    assert "secret=private" not in json.dumps(diagnostic)
    diagnostic["input"].clear()
    assert value == original


def test_wrapped_core_tool_result_preserves_native_parts_and_checks_call_id():
    result = ToolResultBlock(tool_call_id="call_read", output=[
        TextBlock(text="retain caption"), image(),
    ])
    converted = provider()._convert_messages([{
        "role": "tool", "tool_call_id": "call_read", "content": [result],
    }])
    assert converted[0]["output"] == [
        {"type": "input_text", "text": "retain caption"},
        {"type": "input_image", "image_url": DATA_URL, "detail": "auto"},
    ]
    with pytest.raises(InvalidRequestError, match="mismatched"):
        provider()._convert_messages([{
            "role": "tool", "tool_call_id": "call_wrong", "content": [result],
        }])


@pytest.mark.asyncio
async def test_structured_server_error_redacts_images_before_log_and_emit(caplog):
    events = []

    async def emit(event, data):
        events.append((event, copy.deepcopy(data)))

    async def handler(request):
        return httpx.Response(400, json={"error": {
            "type": "invalid_request_error", "message": "Bad image",
            "input": json.loads(request.content)["input"],
        }})

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport,
        openai.AsyncOpenAI(api_key="fixture", base_url="http://vllm.invalid/v1",
                           http_client=transport, max_retries=0) as client,
    ):
        instance = VLLMProvider(client=client, config={
            "use_streaming": False, "max_retries": 0, "raw": False,
        })
        instance.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
        with pytest.raises(InvalidRequestError) as caught:
            await instance.complete(ChatRequest(messages=[
                Message(role="user", content=[image()]),
            ]))
    assert "Bad image" in str(caught.value)
    assert PNG not in str(caught.value)
    assert PNG not in caplog.text and PNG not in json.dumps(events)
    assert caught.value.retryable is False


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_sdk_wire_payload_and_diagnostics_are_separate_copies(streaming):
    requests = []
    events = []

    async def emit(event, data):
        events.append((event, copy.deepcopy(data)))
        # A hook must not be able to strip the actual wire image.
        if event == "llm:request" and "raw" in data:
            data["raw"]["input"].clear()

    async def handler(request):
        assert str(request.url) == "http://vllm.invalid/custom/v1/responses"
        requests.append(json.loads(request.content))
        body = payload("private-served-alias")
        if not streaming:
            return httpx.Response(200, json=body)
        events = [
            {"type": "response.created", "response": {**body, "output": [],
                                                     "status": "in_progress"}},
            {"type": "response.completed", "response": body},
        ]
        wire = "".join("event: " + event["type"] + "\ndata: " +
                       json.dumps({**event, "sequence_number": number}) + "\n\n"
                       for number, event in enumerate(events))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=wire.encode())

    messages = [Message(role="user", content=[
        TextBlock(text="inspect"), image(), TextBlock(text="then answer"),
    ])]
    original = copy.deepcopy(messages)
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport,
        openai.AsyncOpenAI(
            api_key="fixture", base_url="http://vllm.invalid/custom/v1",
            http_client=transport, max_retries=0,
        ) as client,
    ):
        instance = VLLMProvider(client=client, config={
            "default_model": "private-served-alias", "raw": True,
            "use_streaming": streaming, "max_retries": 0,
        })
        instance.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
        await instance.complete(ChatRequest(messages=messages))
    assert len(requests) == 1
    assert requests[0]["input"][0]["content"][1]["image_url"] == DATA_URL
    assert PNG not in json.dumps(events)
    assert messages == original


@pytest.mark.asyncio
async def test_tool_image_parts_reach_sdk_original_call_slot():
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload("private-served-alias"))

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport,
        openai.AsyncOpenAI(
            api_key="fixture", base_url="http://vllm.invalid/v1",
            http_client=transport, max_retries=0,
        ) as client,
    ):
        instance = VLLMProvider(client=client, config={
            "default_model": "private-served-alias", "use_streaming": False,
            "max_retries": 0,
        })
        await instance.complete(ChatRequest(messages=[
            Message(role="assistant", content=[{
                "type": "tool_call", "id": "call_read", "name": "read_file", "input": {},
            }]),
            Message(role="tool", tool_call_id="call_read", content=[
                TextBlock(text="original caption"), image(),
            ]),
        ]))
    tool = requests[0]["input"][-1]
    assert tool["type"] == "function_call_output" and tool["call_id"] == "call_read"
    assert tool["output"] == [
        {"type": "input_text", "text": "original caption"},
        {"type": "input_image", "image_url": DATA_URL, "detail": "auto"},
    ]


@pytest.mark.parametrize("continuation_error", [False, True])
@pytest.mark.asyncio
async def test_native_images_and_call_pair_survive_actual_sdk_continuation(
    continuation_error, caplog,
):
    requests = []

    async def handler(request):
        sent = json.loads(request.content)
        requests.append(sent)
        if len(requests) == 2 and continuation_error:
            return httpx.Response(400, json={"error": {
                "message": "Bad continuation image", "input": sent["input"],
            }})
        body = payload("private-served-alias")
        if len(requests) == 1:
            body["status"] = "incomplete"
            body["incomplete_details"] = {"reason": "max_output_tokens"}
        return httpx.Response(200, json=body)

    messages = [
        Message(role="user", content=[TextBlock(text="inspect"), image()]),
        Message(role="assistant", content=[
            {"type": "tool_call", "id": "call_read", "name": "read_file", "input": {}},
        ]),
        Message(role="tool", tool_call_id="call_read", content=[
            TextBlock(text="tool caption"), image(),
        ]),
    ]
    original = copy.deepcopy(messages)
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport,
        openai.AsyncOpenAI(api_key="fixture", base_url="http://vllm.invalid/v1",
                           http_client=transport, max_retries=0) as client,
    ):
        instance = VLLMProvider(client=client, config={
            "default_model": "private-served-alias", "use_streaming": False,
            "max_retries": 0,
        })
        result = await instance.complete(ChatRequest(messages=messages))
    assert len(requests) == 2
    assert requests[1]["input"][:3] == requests[0]["input"]
    assert requests[1]["input"][0]["content"][1]["image_url"] == DATA_URL
    assert requests[1]["input"][1]["call_id"] == "call_read"
    assert requests[1]["input"][2]["output"][1]["image_url"] == DATA_URL
    assert result.usage.input_tokens == (10 if continuation_error else 20)
    if continuation_error:
        assert result.finish_reason == "length"
    assert PNG not in caplog.text
    assert messages == original