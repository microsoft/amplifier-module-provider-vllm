# amplifier-module-provider-vllm

vLLM provider module for Amplifier - Responses API integration for local/self-hosted LLMs.

## Overview

This provider module integrates vLLM's OpenAI-compatible Responses API with Amplifier, enabling the use of open-weight models like gpt-oss-20b with full reasoning and tool calling support.

**Key Features:**
- **Responses API only** - Optimized for reasoning models (gpt-oss, etc.)
- **Full reasoning support** - Automatic reasoning block separation
- **Tool calling** - Complete tool integration via Responses API
- **Local OR remote** - Works against a local vLLM with no auth, or any remote / hosted endpoint with Bearer auth
- **OpenAI-compatible** - Uses OpenAI SDK under the hood

## Installation

```bash
# Via uv (recommended)
uv pip install git+https://github.com/microsoft/amplifier-module-provider-vllm@main

# For development
git clone https://github.com/microsoft/amplifier-module-provider-vllm
cd amplifier-module-provider-vllm
uv pip install -e .
```

**Note for GPT-OSS models**: Token accounting requires vocab files that are automatically downloaded to `~/.amplifier/cache/vocab/` on first use (requires internet access). If working offline, see troubleshooting section for manual setup.

## vLLM Server Setup

This provider requires a running vLLM server. Example setup:

```bash
# Start vLLM server (basic)
vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2

# For production (recommended - full config in /etc/vllm/model.env)
sudo systemctl start vllm
```

**Server requirements:**
- vLLM version: ≥0.10.1 (tested with 0.10.1.1)
- Responses API: Automatically available (no special flags needed)
- Model: Any model compatible with vLLM (gpt-oss, Llama, Qwen, etc.)

## Configuration

### Model listing

`list_models(*, filters=None)` fetches all current served IDs from the bound
client's `/v1/models`, including custom aliases. All are recommended by default,
with no vendor prefix or family suppression. Every page is consumed; empty success
stays empty and typed sanitized failures never become cached inventory.

`config.model_filters` snapshots `include`, `exclude` and `hide_dated`.
Per-call mappings replace it: `{}` is unfiltered, `include: []` includes none,
and exclude wins. Globs are case-insensitive and OS-independent; IDs stay exact.
`hide_dated` only suppresses YYYY-MM-DD snapshots when their undated alias is
selected. `get_model_list_info()` reports the v1 schema and `source: provider-api`.

Listing reports only limits present in current model cards; missing values are
zero (unknown), and capabilities are not inferred. Existing inference limit
caches, configured budget clamping, explicit model selection and request parameters
remain separate and unchanged.

### Minimal Configuration

```yaml
providers:
  - module: provider-vllm
    source: git+https://github.com/microsoft/amplifier-module-provider-vllm@main
    config:
      base_url: "http://192.168.128.5:8000/v1"  # Your vLLM server
```

### Full Configuration

```yaml
providers:
  - module: provider-vllm
    source: git+https://github.com/microsoft/amplifier-module-provider-vllm@main
    config:
      # Connection
      base_url: "http://192.168.128.5:8000/v1"  # Required: vLLM server URL

      # Model settings
      default_model: "openai/gpt-oss-20b"  # Model name from vLLM
      max_tokens: 4096                      # Max output tokens
      temperature: 0.7                      # Sampling temperature

      # Reasoning
      reasoning: "high"                     # Reasoning effort: minimal|low|medium|high
      reasoning_summary: "detailed"         # Summary verbosity: auto|concise|detailed

      # Context limits (advertised to context managers)
      context_window: 128000                # Model context window in tokens
      max_output_tokens: 32768              # Model max output tokens

      # Advanced
      enable_state: false                   # Server-side state (requires vLLM config)
      truncation: "disabled"                # Fail loud (HTTP 400) instead of silently dropping input (see below)
      timeout: 300.0                        # API timeout (seconds)
      stream_idle_timeout: 300.0            # Max seconds between streamed chunks (see below)
      priority: 100                         # Provider selection priority

      # Debug
      raw: false                            # Attach redacted request params to llm:request
```

`debug`, `raw_debug`, and `debug_truncate_length` are ghost keys from an
older version of this README -- they were never read by this provider.
Use `raw: true` to attach request params on the `llm:request` event, with
secrets and structured image payloads/URLs omitted from the diagnostic copy.

### Image inputs (endpoint-dependent)

Core `ImageBlock` sources support `{"type": "base64", "media_type": "image/png",
"data": "..."}` and `{"type": "url", "url": "https://..."}` (including base64
`data:image/...` URLs). The provider preserves image bytes, order and adjacent
text as native Responses `input_image`/`input_text` parts. It does not resize,
fetch URLs, decode image formats or guess vision capability from model names.
The server fetches HTTP(S) URLs, subject to its network/domain configuration.
Local file paths, OpenAI file IDs and malformed sources become explicit
`[Image unavailable: ...]` text parts; later text is retained.

| Input | Provider wire behavior | Server qualification needed |
| --- | --- | --- |
| User image or mixed text/images | Native ordered message content parts | Generic multimodal Responses path, vision model and chat template |
| Tool image with `tool_call_id` | Native `function_call_output.output` parts, same call ID/slot | Server SDK supporting list outputs, multimodal parser and tool-aware vision template |
| Image tool result without call ID | Non-retryable error, no fabricated user turn | Restore original invocation identity upstream |
| Assistant, system or developer image history | Payload-free `Image unavailable` marker at its text position; no native pixels replayed | No native non-user vision-support claim |
| Text-only history | Existing serialization unchanged | Existing endpoint requirements |
| Harmony/GPT-OSS image processing | Not established by this provider | Inspected Harmony path is text-only; no image-support claim |

Non-user image sources (including malformed sources) are not silently dropped or
serialized as pixels/URLs. Assistant markers stay in assistant `output_text`;
system markers stay in the existing `instructions` parameter. Image-bearing
developer messages retain their developer role as text rather than being moved
to a user message. Neighboring text is retained, and existing assistant
reasoning/tool-call serialization is unchanged. Text-only developer context-file
wrapping remains unchanged. Canonical input/history blocks are never rewritten
by this fallback; it is an honest unavailable result, not native non-user vision.

Actual vLLM source evidence (not just OpenAI API documentation):
[v0.17.0 Responses serving](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/entrypoints/openai/responses/serving.py)
routes generic input through
[Responses conversion](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/entrypoints/openai/responses/utils.py)
and the shared
[`input_image` parser](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/entrypoints/chat_utils.py).
The same generic path exists in
[v0.12.0](https://github.com/vllm-project/vllm/blob/v0.12.0/vllm/entrypoints/openai/serving_responses.py).
This is not a minimum-version guarantee: even
[v0.17.0's requirements](https://github.com/vllm-project/vllm/blob/v0.17.0/requirements/common.txt)
permit older server-side OpenAI SDK schemas that accept **string-only** function
outputs. Client SDK acceptance alone does not qualify the server installation.
Unsupported servers return their normal errors; the provider never retries by
moving images to a fake user or silently stripping them. The bound custom
`base_url` remains the destination, not the global OpenAI service.

There is no image-aware preflight count/refit API in this provider. Its local
Harmony text estimator cannot count image tokens: image requests preserve server
usage instead of replacing it with a purported exact text-only count, with an
explicit warning when that correction is unavailable. Missing/zero server usage
does not establish a zero-cost image request. Context/overflow and continuation
behavior remain unchanged; native parts are retained during continuation.

Offline tests use Core objects and the installed OpenAI SDK's in-memory HTTP
transport. They establish serialization, diagnostic redaction and history
preservation, **not live server/model vision acceptance**. Qualify your exact
server SDK, served model, chat template and multimodal limits separately.
Native image parts are covered on JSON and completed-SSE requests; the automatic
continuation image regression is JSON-only. A supplemental OpenAI SDK 3.22.1
stream test with a terminal `response.incomplete` event failed in the SDK's
`get_final_response()` (which requires `response.completed`); this patch does
not claim to repair that separate incomplete-stream boundary.

### Stream idle timeout

Model requests and streaming reads have no elapsed-time deadline by default.
They wait for completion, cancellation, or a real provider/transport failure.
Quiet prefill or thinking alone does not establish a failed connection.

Set `timeout` to opt into a request deadline. Set `stream_idle_timeout` or
`VLLM_STREAM_IDLE_TIMEOUT` to opt into a limit between streamed chunks, including
the first chunk. Both defaults are `null`; an explicit `null` config value also
turns off an environment-provided idle limit. Explicit numeric limits retain
retryable timeout errors. Connection and pool acquisition stay bounded to
5 seconds, and `close_timeout` continues to bound cleanup.

### Context limits

vLLM's `/v1/models` model cards expose the loaded model's real context
length (`max_model_len`), so `context_window` is **auto-discovered per
model** the first time `list_models()` runs — an endpoint serving several
models with different limits reports each one accurately instead of a
single flat number. Proxy cards can also supply `max_input_tokens` and
`max_output_tokens`. Listing reports only current card values; missing values
remain unknown (zero).

Leave `context_window` unset to auto-detect from the model, the same way
`num_ctx: 0` behaves in the Ollama provider. Setting it explicitly (config
key, or the `VLLM_CONTEXT_WINDOW` environment variable) overrides
discovery, clamped to the server's reported limit so a stale config value
can never guarantee a 400. For inference budget resolution, servers that do not
report limits use the configured value or the `128000` context default, exactly
as before. Those budget defaults are not reported as authentic catalog limits.

This matters for long-context deployments: with the previous hardcoded
values (`max_output_tokens: 128000`), the effective input budget was
capped at ~59k tokens even on endpoints that comfortably handle 120k+
token prompts — causing premature compaction thrashing. `max_output_tokens`
here is the advertised model maximum used for budgeting, distinct from
`max_tokens` (the per-request completion cap).

### Truncation

**BEHAVIOR CHANGE**: `truncation` now defaults to `"disabled"` (previously
`"auto"`). With `truncation: "auto"`, vLLM's Responses API silently drops
input content that overflows the context window: HTTP 200, no error, no
warning — the model simply answers from whatever survived. Verified live
against a direct vLLM endpoint (glm-5.2, real `max_model_len=131072`): a
~150,000-token prompt with `truncation: "auto"` returned HTTP 200 with
`usage.input_tokens=131056` (~19,000 tokens silently discarded), while the
identical prompt with `truncation: "disabled"` returned a clear HTTP 400
naming the exact limit. OpenAI's own Responses API defaults truncation to
`"disabled"` for the same reason: a caller that cannot detect data loss
cannot recover from it.

If your deployment relied on the old auto-truncating behavior (e.g. because
an upstream context manager doesn't yet cap prompt size to the model's
context window), opt back in explicitly:

```yaml
config:
  truncation: "auto"
```

Even with `truncation: "auto"` explicitly configured, this provider now
warns (once per response, via `logger.warning`) whenever the reported
`usage.input_tokens` lands at or near the resolved context window for that
model — a signal that the server likely truncated the prompt server-side.
The warning names the model, the reported `input_tokens`, and the resolved
context window, but never raises and never alters the response. It reuses
the same per-model context-window resolution described above (server-reported
`max_model_len`, clamped by an explicitly configured `context_window`), so it
stays accurate on endpoints serving multiple models with different limits.

### Local vs remote — single source of truth

`base_url` is the **single source of truth** for whether this provider
instance is local or remote. The provider URL-parses it once at
construction and caches the result; everything downstream
(`is_remote` property and capability tagging in `get_info()` and
`get_model_info()`) flows from that one decision. `list_models()` does not infer
capabilities from the endpoint's location.

- **Local** — `base_url` resolves to `localhost`, `127.0.0.1`, `::1`, or
  `0.0.0.0`. Capability tag: `local`. No auth required (api_key is
  ignored if it is the placeholder `"EMPTY"`).
- **Remote** — anything else (LAN IP, public hostname, RunPod / Modal /
  Anyscale / Lambda Labs URL, or a vLLM-backed proxy like
  OpenRouter/Together/Fireworks). Capability tag: `remote`. Bearer auth
  is attached when `api_key` is set.

The `is_remote` property is purely informational — it does not change
how Bearer is attached (the OpenAI SDK does that whenever `api_key` is
non-empty regardless of host). It exists so routing matrices and other
downstream consumers can reason about the deployment shape.

## Mixed local + remote (multi-instance)

To use **both** a local vLLM and a remote / hosted vLLM in the same
session, configure two provider instances. Amplifier supports multiple
named instances of the same provider module via the `instance_id` key:

```toml
# Default LOCAL instance — keeps the natural mount name "vllm"
[[providers]]
module = "amplifier-module-provider-vllm"
[providers.config]
base_url = "http://localhost:8000/v1"
default_model = "openai/gpt-oss-20b"

# Second instance — explicit `instance_id` makes it addressable as "vllm-remote"
[[providers]]
module = "amplifier-module-provider-vllm"
instance_id = "vllm-remote"
[providers.config]
base_url = "https://api.endpoints.anyscale.com/v1"  # or your hosted vLLM URL
api_key = "${VLLM_API_KEY}"
default_model = "meta-llama/Llama-3.3-70B-Instruct"
```

A routing matrix can then target each independently:

```yaml
roles:
  reasoning:
    candidates:
      - provider: vllm-remote
        model: "meta-llama/Llama-3.3-70B-Instruct"
      - provider: vllm
        model: "openai/gpt-oss-20b"
  fast:
    candidates:
      - provider: vllm
        model: "openai/gpt-oss-20b"
```

The kernel validates that at most one entry per module omits
`instance_id` (the "default" keeps the natural mount name); any
additional entries must specify an `instance_id`. See
[`amplifier-core/_session_init.py`](https://github.com/microsoft/amplifier-core)
for the exact contract.

## Usage Examples

### Basic Chat

```python
from amplifier_core import AmplifierSession

config = {
    "session": {
        "orchestrator": "loop-basic",
        "context": "context-simple"
    },
    "providers": [{
        "module": "provider-vllm",
        "config": {
            "base_url": "http://192.168.128.5:8000/v1",
            "default_model": "openai/gpt-oss-20b"
        }
    }]
}

async with AmplifierSession(config=config) as session:
    response = await session.execute("Explain quantum computing")
    print(response)
```

### With Reasoning

```python
config = {
    "providers": [{
        "module": "provider-vllm",
        "config": {
            "base_url": "http://192.168.128.5:8000/v1",
            "default_model": "openai/gpt-oss-20b",
            "reasoning": "high",  # Enable high-effort reasoning
            "reasoning_summary": "detailed"
        }
    }],
    # ... rest of config
}

async with AmplifierSession(config=config) as session:
    # Model will show internal reasoning before answering
    response = await session.execute("Solve this complex problem...")
```

### With Tool Calling

```python
config = {
    "providers": [{
        "module": "provider-vllm",
        "config": {
            "base_url": "http://192.168.128.5:8000/v1",
            "default_model": "openai/gpt-oss-20b"
        }
    }],
    "tools": [{
        "module": "tool-bash",  # Enable bash tool
        "config": {}
    }],
    # ... rest of config
}

async with AmplifierSession(config=config) as session:
    # Model can call tools autonomously
    response = await session.execute("List the files in the current directory")
```

## Architecture

This provider uses the **OpenAI SDK** with a custom `base_url` pointing to your vLLM server. Since vLLM implements the OpenAI-compatible Responses API, the integration is clean and direct.

**Key components:**
- `VLLMProvider`: Main provider class (handles Responses API calls)
- `_constants.py`: Configuration defaults and metadata keys
- `_response_handling.py`: Response parsing and content block conversion

**Response flow:**
```
ChatRequest → VLLMProvider.complete() → AsyncOpenAI.responses.create() →
→ vLLM Server → Response → Content blocks (Thinking + Text + ToolCall) → ChatResponse
```

## Responses API Details

The vLLM provider uses the **Responses API** (`/v1/responses`) which provides:

1. **Structured reasoning**: Separate reasoning blocks from response text
2. **Tool calling**: Native function calling support
3. **Conversation state**: Built-in multi-turn conversation handling
4. **Automatic continuation**: Handles incomplete responses transparently

**Tool format** (vLLM Responses API):
```json
{
  "type": "function",
  "name": "tool_name",
  "description": "Tool description",
  "parameters": {"type": "object", "properties": {...}}
}
```

**Response structure:**
```json
{
  "output": [
    {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "..."}]},
    {"type": "function_call", "name": "tool_name", "arguments": "{...}"},
    {"type": "message", "content": [{"type": "output_text", "text": "..."}]}
  ]
}
```

## Debugging

Enable `raw` to attach redacted request params to the `llm:request` event:

```yaml
config:
  raw: true    # Diagnostic copy; secrets and image blocks omitted
```

**Check logs:**
```bash
# Find recent session
ls -lt ~/.amplifier/projects/*/sessions/*/events.jsonl | head -1

# View raw requests
grep '"event":"llm:request:raw"' <log-file> | python3 -m json.tool

# View raw responses
grep '"event":"llm:response:raw"' <log-file> | python3 -m json.tool
```

## Troubleshooting

### Connection refused

**Problem**: Cannot connect to vLLM server

**Solution**:
```bash
# Check vLLM service status
sudo systemctl status vllm

# Verify server is listening
curl http://192.168.128.5:8000/health

# Check logs
sudo journalctl -u vllm -n 50
```

### Tool calling not working

**Problem**: Model responds with text instead of calling tools

**Verification**:
- ✅ vLLM version ≥0.10.1
- ✅ Using Responses API (not Chat Completions)
- ✅ Tools defined in request

**Note**: Tool calling works via Responses API without special vLLM flags. If it's not working, check the model supports tool calling.

### No reasoning blocks

**Problem**: Responses don't include reasoning/thinking

**Check**:
- Is `reasoning` parameter set in config? (`minimal|low|medium|high`)
- Is the model a reasoning model? (gpt-oss supports reasoning)
- Check raw debug logs to see if reasoning is in API response

### Token usage shows zeros

**For GPT-OSS models**: Token accounting is automatic but requires vocab files.

**How it works**:
- First use: Automatically downloads vocab files to `~/.amplifier/cache/vocab/`
- Subsequent uses: Uses cached files
- No manual setup needed if you have internet access

**What's computed**:
- **Input tokens**: Accurate count using Harmony's tokenization (matches model training format)
- **Output tokens**: Approximate count based on visible output text
- **Limitation**: Output count doesn't include hidden reasoning channels (REST API limitation)

**If auto-download fails** (offline/air-gapped):

```bash
# Manual setup for offline environments
mkdir -p ~/.amplifier/cache/vocab

# Download vocab files (on a machine with internet)
curl -sS -o ~/.amplifier/cache/vocab/o200k_base.tiktoken \
  https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken

curl -sS -o ~/.amplifier/cache/vocab/cl100k_base.tiktoken \
  https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken

# Transfer ~/.amplifier/cache/vocab/ directory to offline machine
# Then set environment variable:
export TIKTOKEN_ENCODINGS_BASE=~/.amplifier/cache/vocab
```

**Check logs for**:
- `[TOKEN_ACCOUNTING] Downloading Harmony vocab files to ~/.amplifier/cache/vocab/...` (first use)
- `[TOKEN_ACCOUNTING] Loaded Harmony GPT-OSS encoder` (success)
- `[TOKEN_ACCOUNTING] Injected usage: input=X, output=Y` (active)

## Development

```bash
# Clone and install
git clone https://github.com/microsoft/amplifier-module-provider-vllm
cd amplifier-module-provider-vllm
uv pip install -e .

# Run tests
pytest tests/

# Check types and lint
make check
```

## Testing

Run the offline suite with `uv run pytest -q -m "not live"`. CI covers OpenAI
SDK 2.8.1, 2.9.0, 2.53.0, and 3.22.1 on Python 3.11 and 3.12. The default
lock targets the latest qualified SDK; retained 2.x environments remain covered.
Real SDK JSON and SSE parsing tests use an in-memory HTTP transport, not paid
model calls. The live model-list check remains explicitly deselected until a
reachable vLLM endpoint is supplied.

Token accounting constructs validated SDK usage records with both cached-read
and cache-write counts. Newer SDKs require the cache-write field even when vLLM
does not report it. If a future usage-schema change prevents construction, the
completed answer and original usage are retained with an explicit warning that
corrected accounting is unavailable; a successful answer is not discarded.
Error fixtures use the installed SDK's HTTP package (`httpx`
on 2.x, `httpx2` on 3.x), including when both are installed. Do not restore the
old `<2.9` cap to work around test fixtures: it conflicts with providers that
require native Responses compaction. A future major SDK requires requalification.
The supported minimum is the lowest qualified SDK, 2.8.1. CI names each SDK
case and checks its effective version before tests. Test imports explicitly
include the checkout root rather than depending on an editable installation.
Responses tool invocation IDs are kept distinct from output-item IDs, including
the second SDK request carrying a tool result. This aligns correlation with the
Responses contract; it does not establish how every live vLLM server validates IDs.

See `ai_working/vllm-investigation/` for comprehensive test scripts:

- `test_provider_simple.py` - Basic provider functionality test
- `06_test_responses_correct_format.py` - Responses API format validation
- `04_test_tool_calling.py` - Tool calling verification

## License

MIT

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

### Bounded output without a completion deadline

`auto_continue` is an optional settings-only provider configuration key, not a
setup prompt. It defaults to `true` when omitted, preserving normal continuation
of truncated responses. To disable continuation, set it explicitly in settings:

```yaml
config:
  auto_continue: false
```

For a single bounded output operation, pass
`request_options={"auto_continue": False}` to `complete()`. The per-call option
takes precedence and never changes the mounted provider. The option is consumed
locally and is not sent to the API. An incomplete
response retains its partial content and usage, reports `finish_reason="length"`,
and is not retried with a larger output budget. Consumers must not treat that
partial response as a complete summary. The provider advertises this optional
contract as `completion:auto_continue:v1`. This option does not impose a time limit.
