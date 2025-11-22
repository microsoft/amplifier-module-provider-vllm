# amplifier-module-provider-vllm

vLLM provider module for Amplifier - OpenAI-compatible API integration.

## Overview

This provider module integrates vLLM's OpenAI-compatible API server with Amplifier, enabling the use of open-weight models like gpt-oss-20b through both `/v1/completions` and `/v1/chat/completions` endpoints.

## Features

- **Completions API support** - Direct prompt completion via `/v1/completions`
- **Chat API support** - Conversational format via `/v1/chat/completions`
- **Flexible endpoint selection** - Configure which endpoint to use
- **Streaming responses** - Token-by-token streaming (future)
- **Log probabilities** - Access token probabilities for analysis
- **Debug logging** - Full request/response logging with truncation
- **Timeout handling** - Configurable API timeouts
- **Network instances** - Connect to remote vLLM servers

## Installation

```bash
# Via uv (recommended)
uv pip install git+https://github.com/microsoft/amplifier-module-provider-vllm@main

# For development
git clone https://github.com/microsoft/amplifier-module-provider-vllm
cd amplifier-module-provider-vllm
uv pip install -e .
```

## Configuration

### Basic Configuration

```yaml
providers:
  - module: provider-vllm
    source: git+https://github.com/microsoft/amplifier-module-provider-vllm@main
    config:
      base_url: "http://localhost:8000" # Required: vLLM server URL
      default_model: "openai/gpt-oss-20b"
      max_tokens: 1024
      temperature: 0.7
```

### Advanced Configuration

```yaml
providers:
  - module: provider-vllm
    source: git+https://github.com/microsoft/amplifier-module-provider-vllm@main
    config:
      # Required
      base_url: "http://localhost:8000"

      # Model settings
      default_model: "openai/gpt-oss-20b"
      max_tokens: 2048
      temperature: 0.5

      # Endpoint selection
      use_completions: true # true = /v1/completions, false = /v1/chat/completions

      # Authentication (optional - many vLLM instances don't require keys)
      api_key: "your-api-key" # Defaults to "EMPTY"

      # Debug settings
      debug: true # Enable detailed logging
      raw_debug: true # Enable ultra-verbose raw API I/O logging
      debug_truncate_length: 180 # Truncate long strings in debug output

      # Performance
      timeout: 300.0 # API timeout in seconds (default 5 minutes)
      priority: 100 # Provider selection priority
```

## Usage

### With Completions API (Default)

The completions API (`/v1/completions`) provides direct prompt completion:

```python
from amplifier_core import AmplifierSession

config = {
    "session": {
        "orchestrator": "loop-basic",
        "context": "context-simple"
    },
    "providers": [{
        "module": "provider-vllm",
        "source": "git+https://github.com/microsoft/amplifier-module-provider-vllm@main",
        "config": {
            "base_url": "http://localhost:8000",
            "default_model": "openai/gpt-oss-20b",
            "use_completions": True  # Use completions endpoint
        }
    }]
}

async with AmplifierSession(config=config) as session:
    response = await session.execute("What is 2+2?")
    print(response)
```

### With Chat API

The chat API (`/v1/chat/completions`) provides conversational format:

```python
config = {
    "providers": [{
        "module": "provider-vllm",
        "source": "git+https://github.com/microsoft/amplifier-module-provider-vllm@main",
        "config": {
            "base_url": "http://localhost:8000",
            "default_model": "openai/gpt-oss-20b",
            "use_completions": False  # Use chat endpoint
        }
    }]
}
```

### With Log Probabilities

```python
# Pass logprobs parameter via kwargs
response = await session.execute(
    "The capital of France is",
    logprobs=1  # Show top 1 logprob for each token
)
```

## API Endpoints

### Completions Endpoint (`/v1/completions`)

**When to use:**

- Direct prompt completion
- No conversation history needed
- Maximum control over prompt format
- Legacy compatibility

**Parameters:**

- `model` - Model name (e.g., "openai/gpt-oss-20b")
- `prompt` - Input text to complete
- `max_tokens` - Maximum tokens to generate
- `temperature` - Sampling temperature (0-2)
- `logprobs` - Number of top logprobs to return
- `echo` - Echo prompt in response
- `stop` - Stop sequences

### Chat Endpoint (`/v1/chat/completions`)

**When to use:**

- Conversational interactions
- Multi-turn dialogues
- Structured message format
- gpt-oss models (includes reasoning_content)

**Parameters:**

- `model` - Model name
- `messages` - List of message objects
- `max_tokens` - Maximum tokens to generate
- `temperature` - Sampling temperature

## vLLM Server Setup

### Starting vLLM Server

```bash
# Install vLLM with gpt-oss support
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Start server (network-accessible)
vllm serve openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key your-secret-key  # Optional
```

### Verifying Server

```bash
# Check available models
curl http://localhost:8000/v1/models

# Test completions endpoint
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "prompt": "What is 2+2?",
    "max_tokens": 50
  }'
```

## Network Configuration

### Firewall Rules

Ensure port 8000 (or your chosen port) is accessible:

```bash
# Check if port is open
nc -zv localhost 8000

# If using firewall
sudo ufw allow 8000/tcp
```

### Security

For production deployments:

1. **Use API keys**: Set `--api-key` on server, provide in config
2. **Use HTTPS**: Place behind reverse proxy (nginx, traefik)
3. **Rate limiting**: Add rate limiting at proxy level
4. **Network isolation**: Use VPN or internal network

## Debugging

### Enable Debug Logging

```yaml
config:
  debug: true # Summary logging
  raw_debug: true # Full API I/O
```

### Check Session Logs

```bash
# Find recent session logs
ls -lt ~/.amplifier/projects/*/sessions/*/events.jsonl | head -1

# View vLLM request
grep '"provider":"vllm"' <log-file> | grep request

# View vLLM response
grep '"provider":"vllm"' <log-file> | grep response
```

## Model Support

Tested models:

- ✅ **gpt-oss-20b** - 20B parameter model, MXFP4 quantized
- ✅ **gpt-oss-120b** - 120B parameter model (requires more VRAM)

Other vLLM-compatible models should work but may require prompt format adjustments.

## Troubleshooting

### "No base_url found"

**Cause**: Missing required `base_url` configuration

**Solution**: Add `base_url` to provider config:

```yaml
config:
  base_url: "http://localhost:8000"
```

### Connection Refused

**Cause**: vLLM server not running or firewall blocking

**Solution**:

1. Verify server is running: `curl http://localhost:8000/health`
2. Check firewall rules
3. Verify server started with `--host 0.0.0.0`

### API Timeout

**Cause**: Response took longer than timeout setting

**Solution**: Increase timeout in config:

```yaml
config:
  timeout: 600.0 # 10 minutes
```

## Performance

### gpt-oss-20b Requirements

- **VRAM**: ~16GB
- **GPU**: NVIDIA Blackwell/Hopper or AMD MI300x/MI355x
- **Model size**: ~14GB on disk (MXFP4 quantized)

### Expected Latency

- First token: 50-200ms
- Subsequent tokens: 20-50ms each
- Full response (100 tokens): 2-5 seconds

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
