"""Discovery of model limits from proxied /v1/models model cards.

Covers the two gaps left open when the ``max_input_tokens`` fallback was
first proposed (microsoft/amplifier#354, PR #32 by @danshapiro):

1. the same proxied card carries ``max_output_tokens``, which was not read;
2. the fallback emitted no log, so a proxy-derived context window was
   indistinguishable from a natively-reported one at runtime -- the same
   silent-number failure the fallback exists to fix, one layer up.
"""

import logging

import openai
import pytest

from amplifier_module_provider_vllm import VLLMProvider


def _card(model_id: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct", **extra):
    """A /v1/models model card carrying arbitrary extra fields."""
    return openai.types.Model.construct(
        id=model_id, created=0, object="model", owned_by="vllm", **extra
    )


class _FakeModels:
    def __init__(self, cards):
        self._cards = cards

    async def list(self):
        cards = self._cards

        class _Response:
            data = cards

        return _Response()


class _FakeClient:
    def __init__(self, cards):
        self.models = _FakeModels(cards)


def _provider(**config):
    return VLLMProvider(base_url="http://localhost:8000/v1", config=config)


def _provider_serving(cards, **config):
    return VLLMProvider(client=_FakeClient(cards), config=config)


class TestCardLimitDiscovery:
    """_discover_card_limits reports both limits and names its source."""

    def test_native_card_reports_max_model_len_as_source(self):
        assert VLLMProvider._discover_card_limits(_card(max_model_len=262144)) == (
            262144,
            None,
            "max_model_len",
        )

    def test_proxied_card_falls_back_to_max_input_tokens(self):
        assert VLLMProvider._discover_card_limits(
            _card(max_input_tokens=262144, max_output_tokens=262144)
        ) == (262144, 262144, "max_input_tokens")

    def test_max_model_len_wins_over_max_input_tokens(self):
        ctx, _out, source = VLLMProvider._discover_card_limits(
            _card(max_model_len=131072, max_input_tokens=262144)
        )
        assert (ctx, source) == (131072, "max_model_len")

    def test_output_limit_is_read_independently_of_context_source(self):
        assert VLLMProvider._discover_card_limits(
            _card(max_model_len=262144, max_output_tokens=8192)
        ) == (262144, 8192, "max_model_len")

    def test_bare_card_discovers_nothing_without_raising(self):
        assert VLLMProvider._discover_card_limits(_card()) == (None, None, None)

    @pytest.mark.parametrize("bad", [0, -1, "not-a-number", None, [262144]])
    def test_malformed_values_degrade_to_not_discovered(self, bad):
        ctx, _out, source = VLLMProvider._discover_card_limits(_card(max_model_len=bad))
        assert (ctx, source) == (None, None)

    def test_malformed_max_model_len_still_falls_through_to_input_tokens(self):
        ctx, _out, source = VLLMProvider._discover_card_limits(
            _card(max_model_len="garbage", max_input_tokens=131072)
        )
        assert (ctx, source) == (131072, "max_input_tokens")


class TestOutputCeilingClamp:
    """A server-reported output limit tightens, never loosens."""

    def test_server_output_ceiling_tightens_the_advertised_limit(self):
        p = _provider(max_output_tokens=32768)
        p._discovered_limits["m"] = 262144
        p._discovered_output_limits["m"] = 4096
        assert p._resolve_limits("m")[1] == 4096

    def test_server_output_ceiling_never_raises_above_half_window_guard(self):
        """An 8k window still budgets input, even if the card claims 8k output."""
        p = _provider(max_output_tokens=32768)
        p._discovered_limits["m"] = 8192
        p._discovered_output_limits["m"] = 8192
        assert p._resolve_limits("m") == (8192, 4096)

    def test_absent_output_ceiling_preserves_existing_behaviour(self):
        p = _provider(max_output_tokens=32768)
        p._discovered_limits["m"] = 262144
        assert p._resolve_limits("m")[1] == 32768

    def test_configured_value_still_wins_when_lower_than_server_ceiling(self):
        p = _provider(max_output_tokens=1024)
        p._discovered_limits["m"] = 262144
        p._discovered_output_limits["m"] = 65536
        assert p._resolve_limits("m")[1] == 1024


class TestDiscoveryProvenanceIsLogged:
    """The discovery path must be identifiable in logs, not an unmarked number."""

    @pytest.mark.asyncio
    async def test_proxy_derived_window_names_its_source(self, caplog):
        p = _provider_serving(
            [_card(max_input_tokens=262144, max_output_tokens=262144)]
        )
        with caplog.at_level(logging.INFO):
            models = await p.list_models()
        assert models[0].context_window == 262144
        assert "via max_input_tokens" in caplog.text

    @pytest.mark.asyncio
    async def test_native_window_names_its_source(self, caplog):
        p = _provider_serving([_card(max_model_len=262144)])
        with caplog.at_level(logging.INFO):
            await p.list_models()
        assert "via max_model_len" in caplog.text

    @pytest.mark.asyncio
    async def test_unchanged_limit_is_not_re_logged_on_every_call(self, caplog):
        p = _provider_serving([_card(max_model_len=262144)])
        await p.list_models()
        with caplog.at_level(logging.INFO):
            await p.list_models()
        assert "Discovered context_window" not in caplog.text
