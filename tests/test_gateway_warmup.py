"""Gateway warm-up holding pages must retry; permanent errors must not.

A vLLM server behind a hosted-GPU front door can answer with an HTML holding
page while the backend starts, sometimes as a 404 -- reported by @ramparte in
PR #28, which established the failure mode and the retry-instead-of-fatal fix.

These tests pin BOTH directions. The true-positive cases are the ones PR #28
set out to fix. The false-positive cases are the reason the predicate requires
an explicit warm-up phrase and a narrow status gate rather than treating "the
body is HTML" as sufficient: a typo'd model name also returns a 404 HTML page
from a proxy, and retrying it for a minute before reporting "warming up" sends
the operator after a fix that will never work.
"""

import httpx
import openai
import pytest

from amplifier_module_provider_vllm import VLLMProvider

_SDK_CLASSES = {
    400: openai.BadRequestError,
    401: openai.AuthenticationError,
    403: openai.PermissionDeniedError,
    404: openai.NotFoundError,
    429: openai.RateLimitError,
}

_HOLDING_PAGE = "<html><body><h1>Waiting for service to respond</h1></body></html>"


def _error(status: int, body: str, content_type: str = "text/html"):
    """Build the SDK exception the OpenAI client would raise for a response."""
    request = httpx.Request(
        "POST", "https://abc123-8000.proxy.example.net/v1/responses"
    )
    response = httpx.Response(
        status,
        headers={"content-type": content_type},
        content=body.encode(),
        request=request,
    )
    parsed = None
    if "json" in content_type:
        try:
            parsed = response.json()
        except Exception:
            parsed = None
    return _SDK_CLASSES.get(status, openai.APIStatusError)(
        "upstream", response=response, body=parsed
    )


class TestWarmupIsDetected:
    """The condition PR #28 set out to fix."""

    def test_404_holding_page_is_transient(self):
        assert VLLMProvider._is_gateway_warmup_page(_error(404, _HOLDING_PAGE))

    def test_425_too_early_holding_page_is_transient(self):
        assert VLLMProvider._is_gateway_warmup_page(_error(425, _HOLDING_PAGE))

    @pytest.mark.parametrize(
        "phrase",
        [
            "service is starting",
            "Starting up, please wait",
            "WARMING UP",
            "Initializing model",
        ],
    )
    def test_warmup_phrasing_variants_are_detected_case_insensitively(self, phrase):
        assert VLLMProvider._is_gateway_warmup_page(
            _error(404, f"<html>{phrase}</html>")
        )

    def test_detection_does_not_depend_on_a_vendor_name(self):
        """Any hosted front door, not one named provider."""
        page = "<html>Waiting for service to respond</html>"
        assert VLLMProvider._is_gateway_warmup_page(_error(404, page))


class TestPermanentErrorsStayFatal:
    """Regression cases: these were misclassified as warm-ups before the gate."""

    def test_permanent_404_html_page_is_not_a_warmup(self):
        """A typo'd model name behind a proxy returns an HTML 404 too."""
        page = "<html><h1>404 Not Found</h1><hr>nginx/1.24.0</html>"
        assert not VLLMProvider._is_gateway_warmup_page(_error(404, page))

    def test_permanent_404_naming_the_host_is_not_a_warmup(self):
        """The hostname appearing in the body proves nothing about transience."""
        page = "<html><h1>404 Not Found</h1><p>abc123-8000.proxy.example.net</p></html>"
        assert not VLLMProvider._is_gateway_warmup_page(_error(404, page))

    def test_413_payload_too_large_is_not_a_warmup(self):
        page = "<html><h1>413 Request Entity Too Large</h1><hr>nginx</html>"
        assert not VLLMProvider._is_gateway_warmup_page(_error(413, page))

    def test_real_json_404_stays_fatal(self):
        err = _error(404, '{"error":{"message":"model not found"}}', "application/json")
        assert not VLLMProvider._is_gateway_warmup_page(err)

    def test_json_body_wins_even_when_it_mentions_starting_up(self):
        """A parsed JSON body is a real API error, always."""
        err = _error(404, '{"error":{"message":"starting up"}}', "application/json")
        assert not VLLMProvider._is_gateway_warmup_page(err)

    @pytest.mark.parametrize("status", [400, 401, 403, 413, 429, 500, 502, 503, 504])
    def test_statuses_outside_the_gate_keep_their_own_handling(self, status):
        """5xx is already retryable; 4xx here is operator-fixable. Neither needs us."""
        assert not VLLMProvider._is_gateway_warmup_page(_error(status, _HOLDING_PAGE))

    def test_missing_response_object_is_not_a_warmup(self):
        err = _error(404, _HOLDING_PAGE)
        err.response = None  # type: ignore[assignment]
        assert not VLLMProvider._is_gateway_warmup_page(err)


class TestStatusGateMirrorsCloudflarePrecedent:
    """The cloudflare predicate is scoped to 403; this one is scoped too."""

    def test_cloudflare_page_on_403_is_not_claimed_by_the_warmup_path(self):
        page = "<html>Just a moment... cf-browser-verification</html>"
        assert not VLLMProvider._is_gateway_warmup_page(_error(403, page))
        assert VLLMProvider._is_cloudflare_challenge(_error(403, page))
