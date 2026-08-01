"""Tests for surfacing silent input truncation (fix/surface-silent-input-truncation).

vLLM's Responses API `truncation` parameter has two values:

- "disabled" (the default as of this change): an oversized prompt returns a
  clear HTTP 400 naming the exact context limit.
- "auto" (opt-in only): an oversized prompt is silently truncated
  server-side -- HTTP 200, no error, no warning, content dropped, and the
  model answers from whatever survived.

Verified live against a direct vLLM endpoint (glm-5.2, real
max_model_len=131072): a ~150,000-token prompt with truncation="auto"
returned HTTP 200 with usage.input_tokens=131056 (~19,000 tokens silently
discarded); the identical prompt with truncation="disabled" returned a
clear HTTP 400 naming the limit.

These tests cover two changes, stacked on PR #29's per-model context-limit
discovery (VLLMProvider._resolve_limits()):

1. DEFAULT_TRUNCATION flipped from "auto" to "disabled" (_constants.py).
2. A warning is logged (never raised, never alters the response) when an
   operator has explicitly opted into truncation="auto" AND the reported
   usage.input_tokens lands at/near the resolved context window for that
   model -- see amplifier_module_provider_vllm._truncation for the
   detection rule and the evidence behind its margin.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_vllm import VLLMProvider
from amplifier_module_provider_vllm._constants import DEFAULT_TRUNCATION
from amplifier_module_provider_vllm._truncation import check_silent_input_truncation

BASE_URL = "http://localhost:8000/v1"

# Live-probe numbers (see module docstring): a real 131072-token ceiling,
# a request that asked for a 16-token completion, and the input_tokens
# vLLM actually reported after auto-truncating an oversized prompt.
REAL_CONTEXT_WINDOW = 131072
REAL_MAX_OUTPUT_TOKENS = 16
REAL_TRUNCATED_INPUT_TOKENS = 131056  # exactly ceiling - max_output_tokens


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_usage_fields.py's pattern)
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> VLLMProvider:
    """Build a provider for the non-streaming request/response round trip.

    use_streaming=False forces the blocking client.responses.create() path
    (rather than client.responses.stream()), matching the mocking pattern
    already used by tests/test_usage_fields.py and tests/test_dynamic_limits.py.
    """
    config = {
        "default_model": "glm-5.2",
        "use_streaming": False,
        "max_retries": 0,
        **config_overrides,
    }
    return VLLMProvider(base_url=BASE_URL, config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    """Response stub with configurable usage and status (see test_usage_fields.py)."""

    def __init__(self, usage=None, status="completed"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi there")],
            )
        ]
        self.usage = usage
        self.status = status
        self.id = "resp_test"


def _warning_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# 1. DEFAULT_TRUNCATION is "disabled"
# ---------------------------------------------------------------------------


class TestDefaultTruncation:
    def test_default_truncation_constant_is_disabled(self):
        """The module constant itself flipped from 'auto' to 'disabled'."""
        assert DEFAULT_TRUNCATION == "disabled"

    def test_provider_defaults_to_disabled_truncation(self):
        """A provider constructed with no explicit `truncation` config key
        picks up the new "disabled" default (not "auto")."""
        provider = _make_provider()

        assert provider.truncation == "disabled"


# ---------------------------------------------------------------------------
# 2. Explicit config truncation="auto" is still honored and reaches the
#    actual request parameters sent to the vLLM client.
# ---------------------------------------------------------------------------


class TestExplicitAutoStillHonored:
    def test_explicit_auto_config_sets_provider_attribute(self):
        provider = _make_provider(truncation="auto")

        assert provider.truncation == "auto"

    def test_explicit_auto_passed_through_to_request_params(self):
        """Operators who opt back into truncation="auto" must still see it
        sent on the wire -- flipping the default must not silently drop the
        override."""
        provider = _make_provider(truncation="auto")
        usage_obj = SimpleNamespace(input_tokens=10, output_tokens=5)
        provider.client.responses.create = AsyncMock(
            return_value=DummyResponse(usage=usage_obj)
        )

        asyncio.run(provider.complete(_simple_request()))

        call_kwargs = provider.client.responses.create.call_args.kwargs
        assert call_kwargs["truncation"] == "auto"

    def test_default_disabled_passed_through_to_request_params(self):
        """Symmetry check: the new default also reaches the wire (so a
        misconfigured/omitted key doesn't silently fall back to "auto")."""
        provider = _make_provider()
        usage_obj = SimpleNamespace(input_tokens=10, output_tokens=5)
        provider.client.responses.create = AsyncMock(
            return_value=DummyResponse(usage=usage_obj)
        )

        asyncio.run(provider.complete(_simple_request()))

        call_kwargs = provider.client.responses.create.call_args.kwargs
        assert call_kwargs["truncation"] == "disabled"


# ---------------------------------------------------------------------------
# 3, 4, 5: the warning fires only for truncation="auto" AND input_tokens
# at/near the resolved ceiling -- not for "disabled", and not when input is
# comfortably below the ceiling.
# ---------------------------------------------------------------------------


class TestWarningConditions:
    @staticmethod
    def _provider_at_real_ceiling(**overrides) -> VLLMProvider:
        """Provider configured with the live-probe's real numbers: a
        131072-token context window and a 16-token requested completion
        (so the margin/threshold math matches the verified evidence
        exactly: 131072 - 16 = 131056)."""
        return _make_provider(
            context_window=REAL_CONTEXT_WINDOW,
            max_tokens=REAL_MAX_OUTPUT_TOKENS,
            **overrides,
        )

    def test_warns_when_auto_and_input_tokens_at_ceiling(self, caplog):
        """3. truncation="auto" + usage.input_tokens at the ceiling -> warning."""
        provider = self._provider_at_real_ceiling(truncation="auto")
        usage_obj = SimpleNamespace(
            input_tokens=REAL_TRUNCATED_INPUT_TOKENS, output_tokens=16
        )
        provider.client.responses.create = AsyncMock(
            return_value=DummyResponse(usage=usage_obj)
        )

        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(_simple_request()))

        warnings = _warning_records(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "glm-5.2" in message
        assert str(REAL_TRUNCATED_INPUT_TOKENS) in message
        assert str(REAL_CONTEXT_WINDOW) in message

    def test_no_warning_when_auto_and_input_tokens_well_below_ceiling(self, caplog):
        """4. truncation="auto" + usage.input_tokens well below the ceiling -> no warning."""
        provider = self._provider_at_real_ceiling(truncation="auto")
        usage_obj = SimpleNamespace(input_tokens=1_000, output_tokens=16)
        provider.client.responses.create = AsyncMock(
            return_value=DummyResponse(usage=usage_obj)
        )

        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(_simple_request()))

        assert _warning_records(caplog) == []

    def test_no_warning_when_disabled_even_at_ceiling(self, caplog):
        """5. truncation="disabled" + input_tokens at the ceiling -> no warning
        (not applicable: "disabled" fails loud via HTTP 400 instead, so
        reaching this code path with input_tokens at the ceiling under
        "disabled" would mean the request legitimately used the whole
        window, not that anything was silently dropped)."""
        provider = self._provider_at_real_ceiling(truncation="disabled")
        usage_obj = SimpleNamespace(
            input_tokens=REAL_TRUNCATED_INPUT_TOKENS, output_tokens=16
        )
        provider.client.responses.create = AsyncMock(
            return_value=DummyResponse(usage=usage_obj)
        )

        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(_simple_request()))

        assert _warning_records(caplog) == []


# ---------------------------------------------------------------------------
# 6: missing/None usage never raises and never warns.
# ---------------------------------------------------------------------------


class TestMissingUsageIsSafe:
    def test_pure_check_returns_none_for_none_input_tokens(self):
        """The detection rule itself, called directly with input_tokens=None
        (usage unavailable), returns None -- not an exception."""
        result = check_silent_input_truncation(
            truncation="auto",
            input_tokens=None,
            context_window=REAL_CONTEXT_WINDOW,
            requested_max_output_tokens=REAL_MAX_OUTPUT_TOKENS,
        )

        assert result is None

    def test_provider_hook_does_not_raise_or_warn_for_none_input_tokens(self, caplog):
        """The provider-side wiring (_maybe_warn_truncated_input) is equally
        defensive: called directly with input_tokens=None, it must not raise
        and must not log anything."""
        provider = self._provider()

        with caplog.at_level(logging.WARNING):
            provider._maybe_warn_truncated_input(
                model="glm-5.2",
                truncation="auto",
                input_tokens=None,
                requested_max_output_tokens=REAL_MAX_OUTPUT_TOKENS,
            )

        assert _warning_records(caplog) == []

    @staticmethod
    def _provider() -> VLLMProvider:
        return _make_provider(context_window=REAL_CONTEXT_WINDOW, truncation="auto")


# ---------------------------------------------------------------------------
# 7: the warning fires at most once per response -- even across a
# continuation round (two raw API calls merged into one final ChatResponse).
# ---------------------------------------------------------------------------


class TestWarnsAtMostOnce:
    def test_single_qualifying_response_warns_exactly_once(self, caplog):
        """Baseline: one qualifying response, one warning (not zero, not two)."""
        provider = TestWarningConditions._provider_at_real_ceiling(truncation="auto")
        usage_obj = SimpleNamespace(
            input_tokens=REAL_TRUNCATED_INPUT_TOKENS, output_tokens=16
        )
        provider.client.responses.create = AsyncMock(
            return_value=DummyResponse(usage=usage_obj)
        )

        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(_simple_request()))

        assert len(_warning_records(caplog)) == 1

    def test_continuation_round_still_warns_only_once(self, caplog):
        """A response that required one auto-continuation (status="incomplete"
        then "completed") makes TWO raw calls to responses.create() internally,
        but the truncation check runs once -- against the final merged
        ChatResponse's usage -- so exactly one warning is logged, not one per
        raw API round-trip."""
        provider = TestWarningConditions._provider_at_real_ceiling(truncation="auto")

        first_usage = SimpleNamespace(input_tokens=500, output_tokens=16)
        first_response = DummyResponse(usage=first_usage, status="incomplete")

        final_usage = SimpleNamespace(
            input_tokens=REAL_TRUNCATED_INPUT_TOKENS, output_tokens=16
        )
        final_response = DummyResponse(usage=final_usage, status="completed")

        provider.client.responses.create = AsyncMock(
            side_effect=[first_response, final_response]
        )

        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(_simple_request()))

        assert provider.client.responses.create.await_count == 2  # sanity: 2 raw calls
        warnings = _warning_records(caplog)
        assert len(warnings) == 1
        assert str(REAL_TRUNCATED_INPUT_TOKENS) in warnings[0].getMessage()
