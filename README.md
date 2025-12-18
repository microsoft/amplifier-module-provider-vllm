# amplifier-module-provider-vllm

vLLM provider module for Amplifier - Responses API integration for local/self-hosted LLMs.

## Overview

This provider module integrates vLLM's OpenAI-compatible Responses API with Amplifier, enabling the use of open-weight models like gpt-oss-20b with full reasoning and tool calling support.

**Key Features:**
- **Responses API only** - Optimized for reasoning models (gpt-oss, etc.)
- **Full reasoning support** - Automatic reasoning block separation
- **Tool calling** - Complete tool integration via Responses API
- **No API key required** - Works with local vLLM servers
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

      # Advanced
      enable_state: false                   # Server-side state (requires vLLM config)
      truncation: "auto"                    # Automatic context management
      timeout: 300.0                        # API timeout (seconds)
      priority: 100                         # Provider selection priority

      # Debug
      debug: true                           # Enable detailed logging
      raw_debug: false                      # Enable raw API I/O logging
      debug_truncate_length: 180            # Truncate long debug strings
```

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

Enable debug logging to see full request/response details:

```yaml
config:
  debug: true        # Summary logging
  raw_debug: true    # Complete API I/O
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

See `ai_working/vllm-investigation/` for comprehensive test scripts:

- `test_provider_simple.py` - Basic provider functionality test
- `06_test_responses_correct_format.py` - Responses API format validation
- `04_test_tool_calling.py` - Tool calling verification

## License

MIT

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
