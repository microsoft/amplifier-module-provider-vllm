"""Real SDK parsing and validated token accounting across 2.x and 3.x clients."""
import json
from types import SimpleNamespace

import openai
import pytest
from amplifier_core.message_models import ChatRequest, Message
from openai.types.responses.response_usage import ResponseUsage

from amplifier_module_provider_vllm import VLLMProvider
from amplifier_module_provider_vllm import _token_accounting as accounting
from amplifier_module_provider_vllm._constants import METADATA_RESPONSE_ID
from tests.sdk_transport import httpx


def payload(model):
    # A real older vLLM server need not supply the SDK's new cache-write field.
    return {
        "id": "resp_compat", "object": "response", "created_at": 1,
        "status": "completed", "model": model, "error": None,
        "incomplete_details": None, "instructions": None, "metadata": {},
        "parallel_tool_calls": True, "temperature": 1, "top_p": 1,
        "tool_choice": "auto", "tools": [],
        "output": [{"id": "msg_compat", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text",
                    "text": "SDK-compatible answer", "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 2},
                  "output_tokens_details": {"reasoning_tokens": 1}},
    }


@pytest.mark.parametrize("immutable", [False, True])
def test_inject_usage_constructs_validated_sdk_models(immutable):
    class FrozenResponse:
        @property
        def usage(self):
            return None

        def model_dump(self):
            return {}

        def __init__(self, usage=None):
            self.saved_usage = usage

    response = FrozenResponse() if immutable else SimpleNamespace(usage=None)
    changed = accounting.inject_usage(response, 10, 5)
    usage = changed.saved_usage if immutable else changed.usage
    assert isinstance(usage, ResponseUsage)
    # Round-trip validation, not model_construct or a mocked usage object.
    ResponseUsage.model_validate(usage.model_dump())
    assert usage.input_tokens == 10 and usage.output_tokens == 5
    assert usage.total_tokens == 15
    assert usage.input_tokens_details.cached_tokens == 0
    assert usage.input_tokens_details.cache_write_tokens == 0


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_usage_schema_failure_keeps_completed_response_and_warns(monkeypatch, caplog, error_type):
    from openai.types.responses import response_usage

    def incompatible_schema(**kwargs):
        raise error_type("Future SDK requires another usage field")

    monkeypatch.setattr(response_usage, "InputTokensDetails", incompatible_schema)
    original_usage = SimpleNamespace(input_tokens=0, output_tokens=0)
    response = SimpleNamespace(usage=original_usage, output="completed answer")
    assert accounting.inject_usage(response, 10, 5) is response
    assert response.usage is original_usage
    assert response.output == "completed answer"
    assert "Corrected token accounting is unavailable" in caplog.text


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("model", ["meta-llama/Llama-3-8B", "openai/gpt-oss-20b"])
@pytest.mark.asyncio
async def test_real_sdk_parses_old_vllm_payload_and_completes(monkeypatch, streaming, model):
    # Leave actual injection active so GPT-OSS exercises the production bug.
    monkeypatch.setattr(accounting, "compute_input_tokens", lambda params: 10)
    monkeypatch.setattr(accounting, "compute_output_tokens", lambda text: 5)
    body = payload(model)
    requests = []

    async def handler(request):
        assert request.url.path == "/v1/responses"
        sent = json.loads(request.content)
        assert sent["model"] == model
        requests.append(sent)
        if not streaming:
            return httpx.Response(200, json=body)
        item = body["output"][0]
        events = [
            {"type": "response.created", "response": {**body, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**item, "status": "in_progress", "content": []}},
            {"type": "response.content_part.added", "item_id": item["id"],
             "output_index": 0, "content_index": 0,
             "part": {"type": "output_text", "text": "", "annotations": []}},
            {"type": "response.output_text.delta", "item_id": item["id"],
             "output_index": 0, "content_index": 0, "delta": "SDK-compatible answer"},
            {"type": "response.output_text.done", "item_id": item["id"],
             "output_index": 0, "content_index": 0, "text": "SDK-compatible answer"},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": body},
        ]
        wire = "".join("event: " + event["type"] + "\ndata: " +
                       json.dumps({**event, "sequence_number": number}) + "\n\n"
                       for number, event in enumerate(events))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=wire.encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        async with openai.AsyncOpenAI(api_key="fixture", base_url="http://vllm.invalid/v1",
                                     http_client=transport, max_retries=0) as client:
            provider = VLLMProvider(client=client, config={"default_model": model,
                                    "use_streaming": streaming, "max_retries": 0})
            result = await provider.complete(ChatRequest(messages=[Message(role="user", content="Hello")]))
    assert len(requests) == 1
    assert any(getattr(block, "text", None) == "SDK-compatible answer" for block in result.content)
    assert result.usage.input_tokens == 10 and result.usage.output_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.usage.cache_read_tokens == (0 if "gpt-oss" in model else 2)
    assert result.metadata[METADATA_RESPONSE_ID] == "resp_compat"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_real_sdk_parses_tool_call_without_losing_arguments(streaming):
    model = "meta-llama/Llama-3-8B"
    body = payload(model)
    body["output"] = [{"id": "fc_compat", "type": "function_call",
                       "call_id": "call_compat", "name": "lookup",
                       "arguments": '{"key":"kept"}', "status": "completed"}]

    async def handler(request):
        assert request.url.path == "/v1/responses"
        if not streaming:
            return httpx.Response(200, json=body)
        events = [
            {"type": "response.created", "response": {**body, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**body["output"][0], "arguments": "", "status": "in_progress"}},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_compat",
             "output_index": 0, "delta": '{"key":"kept"}'},
            {"type": "response.function_call_arguments.done", "item_id": "fc_compat",
             "output_index": 0, "arguments": '{"key":"kept"}'},
            {"type": "response.output_item.done", "output_index": 0, "item": body["output"][0]},
            {"type": "response.completed", "response": body},
        ]
        wire = "".join("event: " + event["type"] + "\ndata: " +
                       json.dumps({**event, "sequence_number": number}) + "\n\n"
                       for number, event in enumerate(events))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=wire.encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        async with openai.AsyncOpenAI(api_key="fixture", base_url="http://vllm.invalid/v1",
                                     http_client=transport, max_retries=0) as client:
            provider = VLLMProvider(client=client, config={"default_model": model,
                                    "use_streaming": streaming, "max_retries": 0})
            result = await provider.complete(ChatRequest(messages=[Message(role="user", content="Hello")]))
    tool = next(block for block in result.content if getattr(block, "type", None) == "tool_call")
    assert tool.id == "call_compat" and tool.name == "lookup"
    assert tool.input == {"key": "kept"}
    assert result.tool_calls[0].id == "call_compat"
    assert result.metadata[METADATA_RESPONSE_ID] == "resp_compat"
    followup = provider._convert_messages([
        {"role": "assistant", "content": [tool.model_dump()]},
        {"role": "tool", "content": "found", "tool_call_id": tool.id},
    ])
    assert next(item["call_id"] for item in followup if item["type"] == "function_call") == "call_compat"
    assert next(item["call_id"] for item in followup if item["type"] == "function_call_output") == "call_compat"


@pytest.mark.asyncio
async def test_real_sdk_sends_second_tool_result_request_with_invocation_id():
    model = "meta-llama/Llama-3-8B"
    tool_body = payload(model)
    tool_body["output"] = [{"id": "fc_compat", "type": "function_call",
                           "call_id": "call_compat", "name": "lookup",
                           "arguments": '{"key":"kept"}', "status": "completed"}]
    requests = []

    async def handler(request):
        sent = json.loads(request.content)
        requests.append(sent)
        if len(requests) == 1:
            return httpx.Response(200, json=tool_body)
        calls = [item for item in sent["input"] if item.get("type") == "function_call"]
        results = [item for item in sent["input"] if item.get("type") == "function_call_output"]
        assert calls[0]["call_id"] == results[0]["call_id"] == "call_compat"
        assert json.loads(calls[0]["arguments"]) == {"key": "kept"}
        assert results[0]["output"] == "found"
        return httpx.Response(200, json=payload(model))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        async with openai.AsyncOpenAI(api_key="fixture", base_url="http://vllm.invalid/v1",
                                     http_client=transport, max_retries=0) as client:
            provider = VLLMProvider(client=client, config={"default_model": model,
                                    "use_streaming": False, "max_retries": 0})
            user = Message(role="user", content="Hello")
            first = await provider.complete(ChatRequest(messages=[user]))
            tool = next(block for block in first.content if getattr(block, "type", None) == "tool_call")
            second = await provider.complete(ChatRequest(messages=[
                user, Message(role="assistant", content=first.content),
                Message(role="tool", content="found", tool_call_id=tool.id),
            ]))
    assert len(requests) == 2
    assert any(getattr(block, "text", None) == "SDK-compatible answer" for block in second.content)
