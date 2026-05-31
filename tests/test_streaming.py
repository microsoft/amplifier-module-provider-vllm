"""
TDD tests for proper token streaming in the vLLM provider.

Contract source: docs/provider-streaming-contract.md
Reference impl:  amplifier-module-provider-openai (feat/proper-streaming, c0d5367)

These tests assert the exact event names, payload shapes, and sequencing
required by the streaming contract.  They must be RED before the streaming
implementation exists and GREEN afterwards.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class RecordingHooks:
    """Capture (event_name, payload) in emission order."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, dict(payload)))

    def names(self) -> list[str]:
        return [n for n, _ in self.events]

    def payloads_for(self, name: str) -> list[dict]:
        return [p for n, p in self.events if n == name]

    def first(self, name: str) -> dict:
        return self.payloads_for(name)[0]


class FakeCoordinator:
    def __init__(self):
        self.hooks = RecordingHooks()


def _make_provider(**config_overrides) -> VLLMProvider:
    config = {"max_retries": 0, "default_model": "meta-llama/Llama-3-8B", **config_overrides}
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config=config)
    provider.coordinator = FakeCoordinator()  # type: ignore[assignment]
    return provider


def _simple_request(**kwargs) -> ChatRequest:
    metadata = kwargs.pop("metadata", None)
    req = ChatRequest(messages=[Message(role="user", content="Hello")])
    if metadata is not None:
        object.__setattr__(req, "metadata", metadata)
    return req


# ---------------------------------------------------------------------------
# Fake stream infrastructure
# ---------------------------------------------------------------------------


def _fake_item(item_type: str, name: str | None = None) -> SimpleNamespace:
    item = SimpleNamespace(type=item_type)
    if name is not None:
        item.name = name
    return item


def _evt(event_type: str, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **kwargs)


def _make_usage(input_tokens: int = 10, output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        input_tokens_details=SimpleNamespace(cached_tokens=0),
    )


def _make_response(
    status: str = "completed",
    text: str = "Hello!",
    resp_id: str = "resp_001",
    output: list | None = None,
) -> SimpleNamespace:
    if output is None:
        output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ]
    r = SimpleNamespace(
        id=resp_id,
        status=status,
        output=output,
        usage=_make_usage(),
        model="meta-llama/Llama-3-8B",
    )
    if status == "incomplete":
        r.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    return r


class FakeStream:
    """Async context manager mimicking openai.AsyncOpenAI.responses.stream()."""

    def __init__(self, events: list[SimpleNamespace], final_response: Any):
        self._events = events
        self._final = final_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._async_iter()

    async def _async_iter(self):
        for event in self._events:
            yield event

    async def get_final_response(self):
        return self._final


def _install_fake_stream(provider: VLLMProvider, stream: "FakeStream") -> None:
    def _stream_factory(**kwargs):
        return stream
    provider.client.responses.stream = _stream_factory  # type: ignore[attr-defined]


def _install_fake_stream_sequence(provider: VLLMProvider, streams: list["FakeStream"]) -> None:
    call_idx = [0]

    def _stream_factory(**kwargs):
        s = streams[call_idx[0]]
        call_idx[0] += 1
        return s

    provider.client.responses.stream = _stream_factory  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 1. Plain text stream
# ---------------------------------------------------------------------------


def test_text_stream_emits_block_start_deltas_block_end():
    """Single text block: block_start + 2 block_delta (ascending seq) + block_end."""
    provider = _make_provider()
    item = _fake_item("message")
    events = [
        _evt("response.output_item.added", output_index=0, item=item),
        _evt("response.output_text.delta", output_index=0, delta="Hello "),
        _evt("response.output_text.delta", output_index=0, delta="world!"),
        _evt("response.output_item.done", output_index=0, item=item),
    ]
    _install_fake_stream(provider, FakeStream(events, _make_response()))
    asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    starts = h.payloads_for("llm:stream_block_start")
    deltas = h.payloads_for("llm:stream_block_delta")
    ends = h.payloads_for("llm:stream_block_end")

    assert len(starts) == 1
    assert len(deltas) == 2
    assert len(ends) == 1

    rids = {p["request_id"] for p in starts + deltas + ends}
    assert len(rids) == 1, f"All events must share one request_id; got {rids}"

    assert starts[0]["block_type"] == "text"
    assert starts[0]["block_index"] == 0
    assert deltas[0]["sequence"] == 0
    assert deltas[1]["sequence"] == 1
    assert deltas[0]["block_type"] == "text"
    assert deltas[1]["block_type"] == "text"
    assert ends[0]["block_type"] == "text"
    assert ends[0]["block_index"] == 0


def test_block_index_consistent_across_event_types():
    """block_index must be identical in block_start, block_delta, block_end."""
    provider = _make_provider()
    item = _fake_item("message")
    events = [
        _evt("response.output_item.added", output_index=0, item=item),
        _evt("response.output_text.delta", output_index=0, delta="x"),
        _evt("response.output_item.done", output_index=0, item=item),
    ]
    _install_fake_stream(provider, FakeStream(events, _make_response()))
    asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    assert h.first("llm:stream_block_start")["block_index"] == h.first("llm:stream_block_delta")["block_index"]
    assert h.first("llm:stream_block_delta")["block_index"] == h.first("llm:stream_block_end")["block_index"]


# ---------------------------------------------------------------------------
# 2. Reasoning / thinking deltas
# ---------------------------------------------------------------------------


def test_reasoning_summary_delta_emits_block_delta_with_thinking():
    """response.reasoning_summary_text.delta -> llm:stream_block_delta with block_type='thinking'."""
    provider = _make_provider()
    reasoning_item = _fake_item("reasoning")
    events = [
        _evt("response.output_item.added", output_index=0, item=reasoning_item),
        _evt("response.reasoning_summary_text.delta", output_index=0, delta="I am thinking..."),
        _evt("response.output_item.done", output_index=0, item=reasoning_item),
    ]
    _install_fake_stream(provider, FakeStream(events, _make_response()))
    asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    td = [p for p in h.payloads_for("llm:stream_block_delta") if p.get("block_type") == "thinking"]
    assert len(td) == 1
    assert td[0]["text"] == "I am thinking..."
    assert "request_id" in td[0]
    assert "block_index" in td[0]
    assert "sequence" in td[0]


def test_reasoning_text_delta_also_emits_block_delta():
    """response.reasoning_text.delta (vLLM full-reasoning variant) -> block_delta with block_type='thinking'."""
    provider = _make_provider()
    item = _fake_item("reasoning")
    events = [
        _evt("response.output_item.added", output_index=0, item=item),
        _evt("response.reasoning_text.delta", output_index=0, delta="deep thought"),
        _evt("response.output_item.done", output_index=0, item=item),
    ]
    _install_fake_stream(provider, FakeStream(events, _make_response()))
    asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    td = [p for p in h.payloads_for("llm:stream_block_delta") if p.get("block_type") == "thinking"]
    assert len(td) == 1
    assert td[0]["text"] == "deep thought"


def test_thinking_delta_sequence_per_block():
    """Sequence counter for thinking block_delta is per-block and starts at 0."""
    provider = _make_provider()
    item = _fake_item("reasoning")
    events = [
        _evt("response.output_item.added", output_index=0, item=item),
        _evt("response.reasoning_summary_text.delta", output_index=0, delta="step 1"),
        _evt("response.reasoning_summary_text.delta", output_index=0, delta="step 2"),
        _evt("response.output_item.done", output_index=0, item=item),
    ]
    _install_fake_stream(provider, FakeStream(events, _make_response()))
    asyncio.run(provider.complete(_simple_request()))

    seqs = [
        d["sequence"]
        for d in provider.coordinator.hooks.payloads_for("llm:stream_block_delta")  # type: ignore[union-attr]
        if d.get("block_type") == "thinking"
    ]
    assert seqs == [0, 1]


# ---------------------------------------------------------------------------
# 3. Two-round continuation with streaming
# ---------------------------------------------------------------------------


def _incomplete_resp(resp_id: str = "r1") -> SimpleNamespace:
    return _make_response(status="incomplete", text="partial", resp_id=resp_id)


def _round_events(delta_text: str) -> list[SimpleNamespace]:
    item = _fake_item("message")
    return [
        _evt("response.output_item.added", output_index=0, item=item),
        _evt("response.output_text.delta", output_index=0, delta=delta_text),
        _evt("response.output_item.done", output_index=0, item=item),
    ]


def test_two_round_continuation_advances_block_index():
    """Second round's block_index must be higher than first round's."""
    provider = _make_provider()
    _install_fake_stream_sequence(provider, [
        FakeStream(_round_events("R1"), _incomplete_resp("r1")),
        FakeStream(_round_events("R2"), _make_response(resp_id="r2")),
    ])
    asyncio.run(provider.complete(_simple_request()))

    starts = provider.coordinator.hooks.payloads_for("llm:stream_block_start")  # type: ignore[union-attr]
    assert len(starts) == 2
    assert starts[1]["block_index"] > starts[0]["block_index"], (
        f"Round-2 block_index ({starts[1]['block_index']}) must exceed "
        f"round-1 block_index ({starts[0]['block_index']})"
    )


def test_two_round_continuation_one_request_id():
    """All stream events across both rounds share ONE request_id."""
    provider = _make_provider()
    _install_fake_stream_sequence(provider, [
        FakeStream(_round_events("R1"), _incomplete_resp("r1")),
        FakeStream(_round_events("R2"), _make_response(resp_id="r2")),
    ])
    asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    stream_payloads = [p for n, p in h.events if n.startswith("llm:stream_")]
    assert stream_payloads, "No stream events emitted"
    rids = {p["request_id"] for p in stream_payloads}
    assert len(rids) == 1, f"Expected one request_id; got {rids}"


def test_two_round_continuation_seq_restarts_per_block():
    """Each block's seq starts at 0; new block in round 2 starts at 0."""
    provider = _make_provider()
    _install_fake_stream_sequence(provider, [
        FakeStream(_round_events("R1"), _incomplete_resp("r1")),
        FakeStream(_round_events("R2"), _make_response(resp_id="r2")),
    ])
    asyncio.run(provider.complete(_simple_request()))

    deltas = provider.coordinator.hooks.payloads_for("llm:stream_block_delta")  # type: ignore[union-attr]
    assert len(deltas) == 2
    assert deltas[0]["sequence"] == 0
    assert deltas[1]["sequence"] == 0  # new block -> new counter


# ---------------------------------------------------------------------------
# 4. metadata stream=False -> blocking path
# ---------------------------------------------------------------------------


def test_stream_false_metadata_no_stream_events():
    """metadata={'stream': False} must suppress all llm:stream_* events."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=_make_response())
    asyncio.run(provider.complete(_simple_request(metadata={"stream": False})))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    assert [n for n in h.names() if n.startswith("llm:stream_")] == []


def test_stream_false_still_emits_base_events():
    """Non-streaming path still emits llm:request and llm:response."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=_make_response())
    asyncio.run(provider.complete(_simple_request(metadata={"stream": False})))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    assert "llm:request" in h.names()
    assert "llm:response" in h.names()


# ---------------------------------------------------------------------------
# 5. Error after partial emit -> stream_aborted
# ---------------------------------------------------------------------------


class _BoomStream:
    """Emits one delta then raises."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        item = _fake_item("message")
        yield _evt("response.output_item.added", output_index=0, item=item)
        yield _evt("response.output_text.delta", output_index=0, delta="partial...")
        raise RuntimeError("mid-stream failure")

    async def get_final_response(self):
        pass  # pragma: no cover


def test_error_after_partial_emits_stream_aborted():
    """Exception after partial emit must fire llm:stream_aborted."""
    provider = _make_provider()
    provider.client.responses.stream = lambda **kw: _BoomStream()  # type: ignore[attr-defined]

    with pytest.raises(Exception):
        asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    aborted = h.payloads_for("llm:stream_aborted")
    assert len(aborted) == 1
    assert "request_id" in aborted[0]
    assert "error" in aborted[0]
    assert "type" in aborted[0]["error"]
    assert "msg" in aborted[0]["error"]


def test_error_before_any_delta_no_aborted():
    """Exception with no delta emitted must NOT fire stream_aborted."""

    class _ImmediateErrorStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            return self._iter()

        async def _iter(self):
            raise RuntimeError("immediate failure")
            yield  # make it an async generator

        async def get_final_response(self):
            pass  # pragma: no cover

    provider = _make_provider()
    provider.client.responses.stream = lambda **kw: _ImmediateErrorStream()  # type: ignore[attr-defined]

    with pytest.raises(Exception):
        asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    assert h.payloads_for("llm:stream_aborted") == []


# ---------------------------------------------------------------------------
# 6. use_streaming config flag
# ---------------------------------------------------------------------------


def test_use_streaming_config_false_no_stream_events():
    """use_streaming=False in config must disable streaming path."""
    provider = _make_provider(use_streaming=False)
    provider.client.responses.create = AsyncMock(return_value=_make_response())
    asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    assert [n for n in h.names() if n.startswith("llm:stream_")] == []


def test_use_streaming_defaults_to_true():
    """VLLMProvider.use_streaming must default to True."""
    provider = VLLMProvider(base_url="http://localhost:8000/v1")
    assert provider.use_streaming is True


# ---------------------------------------------------------------------------
# 7. tool_use block
# ---------------------------------------------------------------------------


def test_tool_use_block_start_includes_name():
    """function_call item -> block_start with block_type='tool_use' and name."""
    provider = _make_provider()
    fc_item = _fake_item("function_call", name="bash")
    final_resp = _make_response(
        output=[SimpleNamespace(type="function_call", name="bash", call_id="c1", arguments="{}")]
    )
    events = [
        _evt("response.output_item.added", output_index=0, item=fc_item),
        _evt("response.output_item.done", output_index=0, item=fc_item),
    ]
    _install_fake_stream(provider, FakeStream(events, final_resp))
    asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    starts = h.payloads_for("llm:stream_block_start")
    ends = h.payloads_for("llm:stream_block_end")
    assert len(starts) == 1
    assert starts[0]["block_type"] == "tool_use"
    assert starts[0].get("name") == "bash"
    assert len(ends) == 1
    assert ends[0]["block_type"] == "tool_use"


# ---------------------------------------------------------------------------
# 8. Empty deltas skipped
# ---------------------------------------------------------------------------


def test_empty_text_delta_skipped():
    """Empty string text deltas must never be emitted."""
    provider = _make_provider()
    item = _fake_item("message")
    events = [
        _evt("response.output_item.added", output_index=0, item=item),
        _evt("response.output_text.delta", output_index=0, delta=""),  # skip
        _evt("response.output_text.delta", output_index=0, delta="real"),
        _evt("response.output_item.done", output_index=0, item=item),
    ]
    _install_fake_stream(provider, FakeStream(events, _make_response()))
    asyncio.run(provider.complete(_simple_request()))

    deltas = provider.coordinator.hooks.payloads_for("llm:stream_block_delta")  # type: ignore[union-attr]
    assert len(deltas) == 1
    assert deltas[0]["text"] == "real"
    assert deltas[0]["block_type"] == "text"


def test_empty_thinking_delta_skipped():
    """Empty reasoning deltas must never be emitted."""
    provider = _make_provider()
    item = _fake_item("reasoning")
    events = [
        _evt("response.output_item.added", output_index=0, item=item),
        _evt("response.reasoning_summary_text.delta", output_index=0, delta=""),  # skip
        _evt("response.reasoning_summary_text.delta", output_index=0, delta="thought"),
        _evt("response.output_item.done", output_index=0, item=item),
    ]
    _install_fake_stream(provider, FakeStream(events, _make_response()))
    asyncio.run(provider.complete(_simple_request()))

    thinking = [
        p for p in provider.coordinator.hooks.payloads_for("llm:stream_block_delta")  # type: ignore[union-attr]
        if p.get("block_type") == "thinking"
    ]
    assert len(thinking) == 1
    assert thinking[0]["text"] == "thought"


# ---------------------------------------------------------------------------
# 9. Multi-block (reasoning + text)
# ---------------------------------------------------------------------------


def test_reasoning_then_text_separate_indices():
    """Reasoning block (idx=0) precedes text block (idx=1); both start at 0."""
    provider = _make_provider()
    ri = _fake_item("reasoning")
    ti = _fake_item("message")
    events = [
        _evt("response.output_item.added", output_index=0, item=ri),
        _evt("response.reasoning_summary_text.delta", output_index=0, delta="thinking"),
        _evt("response.output_item.done", output_index=0, item=ri),
        _evt("response.output_item.added", output_index=1, item=ti),
        _evt("response.output_text.delta", output_index=1, delta="answer"),
        _evt("response.output_item.done", output_index=1, item=ti),
    ]
    _install_fake_stream(provider, FakeStream(events, _make_response()))
    asyncio.run(provider.complete(_simple_request()))

    h = provider.coordinator.hooks  # type: ignore[union-attr]
    starts = h.payloads_for("llm:stream_block_start")
    assert len(starts) == 2
    assert starts[0]["block_type"] == "thinking"
    assert starts[1]["block_type"] == "text"
    assert starts[1]["block_index"] > starts[0]["block_index"]
