"""Regression: HTML-error detectors must fire on REAL SDK error objects.

The detectors (`_is_cloudflare_challenge`, `_is_gateway_warmup_page`) guard on
``error.body``. Every prior test in this suite CONSTRUCTS the error with
``body=None`` by hand -- but the real OpenAI SDK never does that for an HTML
page: when it cannot parse the body as JSON it stores the RAW TEXT in
``error.body`` (a str, not None). A "body is not None" guard therefore bailed on
exactly the pages these detectors exist to catch, and the hand-built fixtures
hid it.

These tests build the error the way the SDK actually does -- via
``client._make_status_error_from_response(response)`` over a real httpx.Response
-- so they fail if anyone reintroduces the body-is-None assumption.
"""

import httpx
import openai

from amplifier_module_provider_vllm import VLLMProvider


def _sdk_error(status: int, content_type: str, body: bytes) -> openai.APIStatusError:
    """An APIStatusError built through the SDK's own construction path."""
    client = openai.OpenAI(api_key="x", base_url="https://x/v1")
    request = httpx.Request("POST", "https://x-8000.proxy.example.net/v1/responses")
    response = httpx.Response(
        status, headers={"content-type": content_type}, content=body, request=request
    )
    return client._make_status_error_from_response(response)


class TestBodyIsRawStringNotNone:
    """Pin the SDK reality the detectors depend on."""

    def test_html_error_body_is_a_str_not_none(self):
        err = _sdk_error(403, "text/html", b"<html>Just a moment...</html>")
        assert isinstance(err.body, str)
        assert err.body is not None

    def test_json_error_body_is_a_dict(self):
        err = _sdk_error(400, "application/json", b'{"error":{"message":"bad"}}')
        assert isinstance(err.body, dict)


class TestCloudflareDetectionOnRealErrors:
    def test_real_html_challenge_is_detected(self):
        err = _sdk_error(
            403, "text/html", b"<html><title>Just a moment...</title>Cloudflare</html>"
        )
        assert VLLMProvider._is_cloudflare_challenge(err) is True

    def test_real_json_error_is_not_a_challenge(self):
        err = _sdk_error(403, "application/json", b'{"error":{"message":"forbidden"}}')
        assert VLLMProvider._is_cloudflare_challenge(err) is False


class TestGatewayWarmupDetectionOnRealErrors:
    def test_real_html_holding_page_is_detected(self):
        err = _sdk_error(
            404, "text/html", b"<html>Waiting for service to respond</html>"
        )
        assert VLLMProvider._is_gateway_warmup_page(err) is True

    def test_real_permanent_404_html_is_not_a_warmup(self):
        err = _sdk_error(404, "text/html", b"<html><h1>404 Not Found</h1>nginx</html>")
        assert VLLMProvider._is_gateway_warmup_page(err) is False

    def test_real_json_404_is_not_a_warmup(self):
        err = _sdk_error(
            404, "application/json", b'{"error":{"message":"model not found"}}'
        )
        assert VLLMProvider._is_gateway_warmup_page(err) is False

    def test_real_413_with_warmup_text_is_gated_out_by_status(self):
        err = _sdk_error(
            413, "text/html", b"<html>Waiting for service to respond</html>"
        )
        assert VLLMProvider._is_gateway_warmup_page(err) is False
