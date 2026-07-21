"""
TDD tests for the inter-chunk stream idle timeout.

Upstream issue: microsoft/amplifier#339 — "Hung streaming response stalls
session indefinitely — no client-side read/idle timeout".

Against remote vLLM endpoints reached through hosted-GPU HTTPS proxies
(e.g. RunPod), connections are routinely dropped without FIN mid-stream.
Nothing bounded the time-between-chunks on an established stream, so a
silent drop left the session hanging indefinitely (observed: ~8.7 hours).

These tests assert:
1. A stalled stream (mid-stream or before the FIRST chunk) aborts after
   ``stream_idle_timeout`` seconds with a retryable LLMTimeoutError.
2. The stream context is closed (cleaned up) when the timeout fires.
3. A healthy slow-but-under-threshold stream is NOT interrupted.
4. Config resolution: config -> VLLM_STREAM_IDLE_TIMEOUT env var -> default
   constant, with float coercion, and config beating env.
5. get_info() exposes the field in config_fields.
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider
from amplifier_module_provider_vllm._constants import DEFAULT_STREAM_IDLE_TIMEOUT

# ---------------------------------------------------------------------------
# Test helpers (mirrors tests/test_streaming.py conventions)
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


class FakeCoordinator:
    def __init__(self):
        self.hooks = RecordingHooks()


def _make_provider(**config_overrides) -> VLLMProvider:
    config = {
        "max_retries": 0,
        "default_model": "meta-llama/Llama-3-8B",
        **config_overrides,
    }
    provider = VLLMProvider(base_url="http://localhost:8000/v1", config=config)
    provider.coordinator = FakeCoordinator()  # type: ignore[assignment]
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _fake_item(item_type: str) -> SimpleNamespace:
    return SimpleNamespace(type=item_type)


def _evt(event_type: str, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **kwargs)


def _make_usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        input_tokens_details=SimpleNamespace(cached_tokens=0),
    )


def _make_response(text: str = "Hello!") -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_001",
        status="completed",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=_make_usage(),
        model="meta-llama/Llama-3-8B",
    )


class StallingStream:
    """Async stream that yields ``events`` then hangs forever (no close).

    Simulates a hosted-GPU HTTPS proxy dropping the connection without FIN:
    the socket stays "open" but no further chunk ever arrives.
    """

    def __init__(self, events: list[SimpleNamespace]):
        self._events = events
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event
        await asyncio.Event().wait()  # hang forever — never yields, never raises

    async def get_final_response(self):  # pragma: no cover - stream never finishes
        raise AssertionError(
            "get_final_response must not be reached on a stalled stream"
        )


class SlowStream:
    """Healthy-but-slow stream: pauses ``delay`` seconds before each event."""

    def __init__(
        self, events: list[SimpleNamespace], final_response: Any, delay: float
    ):
        self._events = events
        self._final = final_response
        self._delay = delay
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            await asyncio.sleep(self._delay)
            yield event

    async def get_final_response(self):
        return self._final


def _text_round_events(deltas: list[str]) -> list[SimpleNamespace]:
    item = _fake_item("message")
    events: list[SimpleNamespace] = [
        _evt("response.output_item.added", output_index=0, item=item)
    ]
    for d in deltas:
        events.append(_evt("response.output_text.delta", output_index=0, delta=d))
    events.append(_evt("response.output_item.done", output_index=0, item=item))
    return events


def _install_stream(provider: VLLMProvider, stream: Any) -> None:
    provider.client.responses.stream = lambda **kw: stream  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 1. Stalled stream aborts with a retryable timeout error
# ---------------------------------------------------------------------------


def test_stall_after_chunks_raises_retryable_timeout():
    """Stream yields chunks then hangs -> LLMTimeoutError(retryable=True)."""
    provider = _make_provider(stream_idle_timeout=0.05)
    stream = StallingStream(_text_round_events(["Hello ", "wor"]))
    _install_stream(provider, stream)

    start = time.monotonic()
    with pytest.raises(kernel_errors.LLMTimeoutError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))
    elapsed = time.monotonic() - start

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.retryable is True
    assert elapsed < 5.0, f"Timeout should fire in ~0.05s, took {elapsed:.1f}s"


def test_stall_before_first_chunk_raises_retryable_timeout():
    """A hang BEFORE the first chunk is the same failure mode — must abort too."""
    provider = _make_provider(stream_idle_timeout=0.05)
    stream = StallingStream([])  # never yields anything
    _install_stream(provider, stream)

    with pytest.raises(kernel_errors.LLMTimeoutError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "vllm"
    assert err.retryable is True


def test_stall_error_message_mentions_idle_timeout():
    """The error should be diagnosable: name the knob and the window."""
    provider = _make_provider(stream_idle_timeout=0.05)
    _install_stream(provider, StallingStream([]))

    with pytest.raises(kernel_errors.LLMTimeoutError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert "stream_idle_timeout" in str(exc_info.value)


def test_stream_closed_on_idle_timeout():
    """The stream context must be exited (cleaned up) when the timeout fires."""
    provider = _make_provider(stream_idle_timeout=0.05)
    stream = StallingStream(_text_round_events(["partial"]))
    _install_stream(provider, stream)

    with pytest.raises(kernel_errors.LLMTimeoutError):
        asyncio.run(provider.complete(_simple_request()))

    assert stream.closed is True


def test_stall_after_partial_emit_fires_stream_aborted():
    """Existing convention: partial deltas + failure -> llm:stream_aborted."""
    provider = _make_provider(stream_idle_timeout=0.05)
    _install_stream(provider, StallingStream(_text_round_events(["partial"])))

    with pytest.raises(kernel_errors.LLMTimeoutError):
        asyncio.run(provider.complete(_simple_request()))

    aborted = provider.coordinator.hooks.payloads_for("llm:stream_aborted")  # type: ignore[union-attr]
    assert len(aborted) == 1
    assert aborted[0]["error"]["type"] == "LLMTimeoutError"


def test_idle_timeout_error_engages_retry():
    """Retryable convention must engage upstream retry logic (retry_with_backoff)."""
    calls = {"count": 0}

    def _stream_factory(**kw):
        calls["count"] += 1
        return StallingStream([])

    provider = _make_provider(
        stream_idle_timeout=0.05,
        max_retries=1,
        min_retry_delay=0.01,
        max_retry_delay=0.02,
        retry_jitter=False,
    )
    provider.client.responses.stream = _stream_factory  # type: ignore[attr-defined]

    with pytest.raises(kernel_errors.LLMTimeoutError):
        asyncio.run(provider.complete(_simple_request()))

    assert calls["count"] == 2, "retryable timeout must trigger one retry attempt"


# ---------------------------------------------------------------------------
# 2. Healthy slow stream is NOT interrupted
# ---------------------------------------------------------------------------


def test_slow_but_under_threshold_stream_completes():
    """Chunks arriving slower than instant but under the idle window must flow."""
    provider = _make_provider(stream_idle_timeout=0.5)
    stream = SlowStream(
        _text_round_events(["Hello ", "world!"]), _make_response(), delay=0.02
    )
    _install_stream(provider, stream)

    response = asyncio.run(provider.complete(_simple_request()))

    assert response is not None
    deltas = provider.coordinator.hooks.payloads_for("llm:stream_block_delta")  # type: ignore[union-attr]
    assert [d["text"] for d in deltas] == ["Hello ", "world!"]
    assert provider.coordinator.hooks.payloads_for("llm:stream_aborted") == []  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 3. Config resolution: config -> env -> default (float-coerced)
# ---------------------------------------------------------------------------


def _bare_provider(config: dict | None = None) -> VLLMProvider:
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config or {})


def test_default_comes_from_constants(monkeypatch):
    monkeypatch.delenv("VLLM_STREAM_IDLE_TIMEOUT", raising=False)
    provider = _bare_provider()
    assert provider.stream_idle_timeout == DEFAULT_STREAM_IDLE_TIMEOUT


def test_default_is_generous_for_long_prefill():
    """Long time-to-first-token on 60-90k-token prompts is legitimate (minutes).

    The default must be minutes-scale — big enough to never false-positive on
    a healthy long prefill, small enough to bound a silent hang.
    """
    assert 120.0 <= DEFAULT_STREAM_IDLE_TIMEOUT <= 600.0


def test_config_overrides_default(monkeypatch):
    monkeypatch.delenv("VLLM_STREAM_IDLE_TIMEOUT", raising=False)
    provider = _bare_provider({"stream_idle_timeout": 45})
    assert provider.stream_idle_timeout == 45.0


def test_env_var_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_STREAM_IDLE_TIMEOUT", "120")
    provider = _bare_provider()
    assert provider.stream_idle_timeout == 120.0


def test_config_beats_env(monkeypatch):
    monkeypatch.setenv("VLLM_STREAM_IDLE_TIMEOUT", "120")
    provider = _bare_provider({"stream_idle_timeout": 45})
    assert provider.stream_idle_timeout == 45.0


def test_string_config_value_coerced_to_float(monkeypatch):
    monkeypatch.delenv("VLLM_STREAM_IDLE_TIMEOUT", raising=False)
    provider = _bare_provider({"stream_idle_timeout": "7.5"})
    assert provider.stream_idle_timeout == 7.5
    assert isinstance(provider.stream_idle_timeout, float)


# ---------------------------------------------------------------------------
# 4. get_info() exposes the field
# ---------------------------------------------------------------------------


def test_config_fields_include_stream_idle_timeout():
    provider = _bare_provider()
    fields = {f.id: f for f in provider.get_info().config_fields}
    assert "stream_idle_timeout" in fields
    field = fields["stream_idle_timeout"]
    assert field.env_var == "VLLM_STREAM_IDLE_TIMEOUT"
    assert float(field.default) == DEFAULT_STREAM_IDLE_TIMEOUT
    assert field.required is False
