"""Residual non-user image loss: offline conversion and actual SDK transport."""

import copy
import json
from types import SimpleNamespace

import openai
import pytest
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)

from amplifier_module_provider_vllm import VLLMProvider
from tests.sdk_transport import httpx
from tests.test_image_input import PNG
from tests.test_sdk_compatibility import payload

SOURCES = [
    {"type": "base64", "media_type": "image/png", "data": PNG},
    {"type": "url", "url": "https://images.invalid/secret.png?token=private"},
    {"type": "base64", "media_type": "image/png", "data": "MALFORMED_PRIVATE_PIXELS!"},
    {"type": "url", "url": "file:///PRIVATE_IMAGE_PATH.png"},
    {},
]


def make_provider(**config):
    return VLLMProvider(base_url="http://vllm.invalid/v1", config={
        "default_model": "private-served-alias", "use_streaming": False,
        "max_retries": 0, **config,
    })


def assert_marker(text, role):
    assert "image unavailable" in text.lower()
    assert role in text.lower()
    assert "before" in text and "after" in text
    assert text.index("before") < text.lower().index("image unavailable") < text.index("after")


def assert_no_pixels(value):
    text = json.dumps(value)
    for private in (PNG, "images.invalid/secret", "MALFORMED_PRIVATE", "PRIVATE_IMAGE_PATH"):
        assert private not in text


@pytest.mark.parametrize("role", ["assistant", "developer"])
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("objects", [False, True])
def test_nonuser_converter_marker_preserves_role_text_and_history(role, source, objects):
    blocks = [TextBlock(text="before"), ImageBlock(source=source), TextBlock(text="after")]
    if not objects:
        blocks = [block.model_dump() for block in blocks]
    messages = [{"role": role, "content": blocks}, {"role": "user", "content": "followup"}]
    original = copy.deepcopy(messages)
    converted = make_provider()._convert_messages(messages)
    message = next(item for item in converted if item.get("role") == role)
    if role == "assistant":
        text = message["content"][0]["text"]
    else:
        text = message["content"]
    assert_marker(text, role)
    assert [item for item in converted if item.get("role") == "user"] == [
        {"role": "user", "content": "followup"},
    ]
    assert_no_pixels(converted)
    assert messages == original


@pytest.mark.parametrize("objects", [False, True])
def test_assistant_marker_does_not_change_tool_or_reasoning_policy(objects):
    blocks = [
        ThinkingBlock(thinking="private reasoning", signature="private signature"),
        TextBlock(text="before"), ImageBlock(source=SOURCES[0]),
        ToolCallBlock(id="call_read", name="read_file", input={"path": "kept"}),
        TextBlock(text="after"),
    ]
    if not objects:
        blocks = [block.model_dump() for block in blocks]
    messages = [
        {"role": "assistant", "content": blocks},
        {"role": "tool", "tool_call_id": "call_read", "content": "result"},
    ]
    original = copy.deepcopy(messages)
    output = make_provider()._convert_messages(messages)
    assert [item.get("type") for item in output] == [
        "reasoning", "function_call", "message", "function_call_output",
    ]
    assert output[0]["content"][0]["text"] == "private reasoning"
    assert output[0]["summary"][0]["text"] == "private reasoning"
    assert "private signature" not in json.dumps(output)
    assert output[1] == {
        "type": "function_call", "call_id": "call_read", "name": "read_file",
        "arguments": '{"path": "kept"}', "id": "ft_read", "status": None,
    }
    assert_marker(output[2]["content"][0]["text"], "assistant")
    assert output[3] == {
        "type": "function_call_output", "call_id": "call_read", "output": "result",
    }
    assert_no_pixels(output)
    assert messages == original


@pytest.mark.parametrize("role", ["assistant", "developer"])
def test_nonuser_image_only_is_not_dropped_and_legacy_text_policy_is_unchanged(role):
    instance = make_provider()
    converted = instance._convert_messages([{
        "role": role, "content": [ImageBlock(source=SOURCES[0])],
    }])
    assert len(converted) == 1 and converted[0]["role"] == role
    assert "image unavailable" in json.dumps(converted).lower()
    assert_no_pixels(converted)
    plain = instance._convert_messages([{"role": role, "content": "plain"}])
    if role == "developer":
        assert plain == [{"role": "user", "content": "<context_file>\nplain\n</context_file>"}]
    else:
        assert plain[0]["role"] == "assistant"
        assert plain[0]["content"] == [
            {"type": "output_text", "text": "plain", "annotations": []},
        ]


@pytest.mark.asyncio
async def test_system_image_only_and_multiple_instruction_slots():
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload("private-served-alias"))

    messages = [
        Message(role="system", content="first"),
        Message(role="system", content=[ImageBlock(source=SOURCES[1])]),
        Message(role="system", content="last"),
        Message(role="user", content="followup"),
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
        await instance.complete(ChatRequest(messages=messages))
    text = requests[0]["instructions"]
    assert text.startswith("first\n\n") and text.endswith("\n\nlast")
    assert "image unavailable" in text.lower() and "system" in text
    assert requests[0]["input"] == [{"role": "user", "content": "followup"}]
    assert_no_pixels(requests)
    assert messages == original


@pytest.mark.parametrize("role", ["assistant", "system", "developer"])
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.asyncio
async def test_nonuser_images_actual_sdk_payload_and_plaintext_reuse(role, source, caplog):
    requests = []
    events = []

    async def emit(name, data):
        events.append((name, copy.deepcopy(data)))

    async def handler(request):
        assert str(request.url) == "http://vllm.invalid/custom/v1/responses"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload("private-served-alias"))

    messages = [
        Message(role=role, content=[
            TextBlock(text="before"), ImageBlock(source=source), TextBlock(text="after"),
        ]),
        Message(role="user", content="followup"),
    ]
    original = copy.deepcopy(messages)
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport,
        openai.AsyncOpenAI(api_key="fixture", base_url="http://vllm.invalid/custom/v1",
                           http_client=transport, max_retries=0) as client,
    ):
        instance = VLLMProvider(client=client, config={
            "default_model": "private-served-alias", "use_streaming": False,
            "max_retries": 0, "raw": True,
        })
        instance.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
        await instance.complete(ChatRequest(messages=messages))
        await instance.complete(ChatRequest(messages=[Message(role="user", content="plain")]))
    assert len(requests) == 2
    sent = requests[0]
    if role == "system":
        text = sent["instructions"]
        assert len(sent["input"]) == 1
    else:
        converted = next(item for item in sent["input"] if item.get("role") == role)
        text = (converted["content"][0]["text"] if role == "assistant"
                else converted["content"])
        assert len(sent["input"]) == 2
    assert_marker(text, role)
    assert [item for item in sent["input"] if item.get("role") == "user"] == [
        {"role": "user", "content": "followup"},
    ]
    assert requests[1]["input"] == [{"role": "user", "content": "plain"}]
    assert "instructions" not in requests[1]
    assert "image unavailable" not in json.dumps(requests[1]).lower()
    assert_no_pixels(requests)
    assert_no_pixels(events)
    assert_no_pixels(caplog.text)
    assert messages == original