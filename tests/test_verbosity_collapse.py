"""Tests for CP-V Provider Verbosity Collapse (Task 13f).

Verifies:
- `raw: true` config adds `raw` field to base llm:request and llm:response events
- `raw: false` (default) produces no `raw` field in those events
- :debug and :raw tiered events are never emitted
- Config flag `raw` replaces deprecated debug/raw_debug/debug_truncate_length flags
- `_truncate_values` method is removed
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def emitted_names(self) -> list[str]:
        return [name for name, _ in self.events]

    def payload_for(self, event_name: str) -> dict | None:
        for name, payload in self.events:
            if name == event_name:
                return payload
        return None


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_provider(*, raw: bool = False) -> VLLMProvider:
    config: dict = {"max_retries": 0, "use_streaming": False}
    if raw:
        config["raw"] = True
    provider = VLLMProvider(
        base_url="http://localhost:8000/v1",
        config=config,
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _make_dummy_response(status: str = "completed") -> SimpleNamespace:
    """Create a minimal mock vLLM Responses API response."""
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello!")],
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        status=status,
        id="resp_test",
    )
    response.model_dump = lambda: {"id": "resp_test", "status": status}
    return response


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestRawConfigFlag:
    def test_raw_flag_defaults_to_false(self):
        """Without config, self.raw should be False."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1", config={})
        assert provider.raw is False  # type: ignore[attr-defined]

    def test_raw_flag_true_when_configured(self):
        """With raw=True in config, self.raw should be True."""
        provider = VLLMProvider(
            base_url="http://localhost:8000/v1", config={"raw": True}
        )
        assert provider.raw is True  # type: ignore[attr-defined]

    def test_raw_flag_false_when_explicitly_set(self):
        """With raw=False in config, self.raw should be False."""
        provider = VLLMProvider(
            base_url="http://localhost:8000/v1", config={"raw": False}
        )
        assert provider.raw is False  # type: ignore[attr-defined]

    def test_debug_flag_removed(self):
        """The old `debug` flag should not exist on the provider."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1", config={})
        assert not hasattr(provider, "debug"), (
            "Old `debug` config flag must be removed; use `raw` instead"
        )

    def test_raw_debug_flag_removed(self):
        """The old `raw_debug` flag should not exist on the provider."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1", config={})
        assert not hasattr(provider, "raw_debug"), (
            "Old `raw_debug` config flag must be removed; use `raw` instead"
        )

    def test_debug_truncate_length_removed(self):
        """The old `debug_truncate_length` flag should not exist on the provider."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1", config={})
        assert not hasattr(provider, "debug_truncate_length"), (
            "Old `debug_truncate_length` config flag must be removed"
        )

    def test_truncate_values_method_removed(self):
        """The `_truncate_values` method should not exist on the provider."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1", config={})
        assert not hasattr(provider, "_truncate_values"), (
            "`_truncate_values` method must be removed"
        )


# ---------------------------------------------------------------------------
# llm:request event tests
# ---------------------------------------------------------------------------


class TestLLMRequestEvent:
    def test_base_request_event_emitted_without_raw(self):
        """llm:request is always emitted, even when raw=False."""
        provider = _make_provider(raw=False)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        assert "llm:request" in hooks.emitted_names()

    def test_request_event_has_no_raw_field_by_default(self):
        """llm:request payload should NOT have `raw` field when raw=False."""
        provider = _make_provider(raw=False)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:request")
        assert payload is not None
        assert "raw" not in payload, (
            "llm:request payload must not contain `raw` field when raw=False"
        )

    def test_request_event_has_raw_field_when_raw_true(self):
        """llm:request payload should have `raw` field when raw=True."""
        provider = _make_provider(raw=True)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:request")
        assert payload is not None
        assert "raw" in payload, (
            "llm:request payload must contain `raw` field when raw=True"
        )

    def test_request_raw_field_is_dict(self):
        """`raw` field in llm:request should be a dict (the full params)."""
        provider = _make_provider(raw=True)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:request")
        assert payload is not None
        assert isinstance(payload["raw"], dict)

    def test_no_debug_request_event_emitted(self):
        """llm:request:debug must never be emitted."""
        provider = _make_provider(raw=True)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        assert "llm:request:debug" not in hooks.emitted_names(), (
            "llm:request:debug must never be emitted after verbosity collapse"
        )

    def test_no_raw_request_event_emitted(self):
        """llm:request:raw must never be emitted."""
        provider = _make_provider(raw=True)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        assert "llm:request:raw" not in hooks.emitted_names(), (
            "llm:request:raw must never be emitted after verbosity collapse"
        )

    def test_request_event_base_fields_present(self):
        """llm:request should always have provider, model, message_count fields."""
        provider = _make_provider(raw=False)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:request")
        assert payload is not None
        assert payload["provider"] == "vllm"
        assert "model" in payload
        assert "message_count" in payload


# ---------------------------------------------------------------------------
# llm:response event tests
# ---------------------------------------------------------------------------


class TestLLMResponseEvent:
    def test_base_response_event_emitted_without_raw(self):
        """llm:response is always emitted, even when raw=False."""
        provider = _make_provider(raw=False)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        assert "llm:response" in hooks.emitted_names()

    def test_response_event_has_no_raw_field_by_default(self):
        """llm:response payload should NOT have `raw` field when raw=False."""
        provider = _make_provider(raw=False)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert "raw" not in payload, (
            "llm:response payload must not contain `raw` field when raw=False"
        )

    def test_response_event_has_raw_field_when_raw_true(self):
        """llm:response payload should have `raw` field when raw=True."""
        provider = _make_provider(raw=True)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert "raw" in payload, (
            "llm:response payload must contain `raw` field when raw=True"
        )

    def test_response_raw_field_is_dict(self):
        """`raw` field in llm:response should be a dict."""
        provider = _make_provider(raw=True)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert isinstance(payload["raw"], dict)

    def test_no_debug_response_event_emitted(self):
        """llm:response:debug must never be emitted."""
        provider = _make_provider(raw=True)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        assert "llm:response:debug" not in hooks.emitted_names(), (
            "llm:response:debug must never be emitted after verbosity collapse"
        )

    def test_no_raw_response_event_emitted(self):
        """llm:response:raw must never be emitted."""
        provider = _make_provider(raw=True)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        assert "llm:response:raw" not in hooks.emitted_names(), (
            "llm:response:raw must never be emitted after verbosity collapse"
        )

    def test_response_event_base_fields_present(self):
        """llm:response should always have provider, model, status fields."""
        provider = _make_provider(raw=False)
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert payload["provider"] == "vllm"
        assert "model" in payload
        assert payload["status"] == "ok"


# ---------------------------------------------------------------------------
# No :debug/:raw events in any path
# ---------------------------------------------------------------------------


class TestNoTieredEvents:
    def test_no_tiered_events_regardless_of_raw_flag(self):
        """No :debug or :raw tiered events should ever be emitted."""
        for raw_flag in [True, False]:
            provider = _make_provider(raw=raw_flag)
            provider.client.responses.create = AsyncMock(
                return_value=_make_dummy_response()
            )

            asyncio.run(provider.complete(_simple_request()))

            hooks = cast(FakeCoordinator, provider.coordinator).hooks
            for event_name in hooks.emitted_names():
                assert not event_name.endswith(":debug"), (
                    f"Found tiered :debug event: {event_name}"
                )
                assert not event_name.endswith(":raw"), (
                    f"Found tiered :raw event: {event_name}"
                )

    def test_no_tiered_events_in_continuation_path(self):
        """No :debug or :raw tiered events when continuation fires (incomplete → completed)."""
        provider = _make_provider(raw=True)

        incomplete = _make_dummy_response(status="incomplete")
        incomplete.incomplete_details = SimpleNamespace(reason="max_output_tokens")
        completed = _make_dummy_response(status="completed")

        provider.client.responses.create = AsyncMock(
            side_effect=[incomplete, completed]
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        for event_name in hooks.emitted_names():
            assert not event_name.endswith(":debug"), (
                f"Found tiered :debug event in continuation path: {event_name}"
            )
            assert not event_name.endswith(":raw"), (
                f"Found tiered :raw event in continuation path: {event_name}"
            )

    def test_continuation_path_has_raw_field_in_response_when_raw_true(self):
        """llm:response base event gets `raw` field even after continuation."""
        provider = _make_provider(raw=True)

        incomplete = _make_dummy_response(status="incomplete")
        incomplete.incomplete_details = SimpleNamespace(reason="max_output_tokens")
        completed = _make_dummy_response(status="completed")

        provider.client.responses.create = AsyncMock(
            side_effect=[incomplete, completed]
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert "raw" in payload, (
            "llm:response must have `raw` field in continuation path when raw=True"
        )
