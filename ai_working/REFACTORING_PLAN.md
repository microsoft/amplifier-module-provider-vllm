# vLLM Provider Refactoring Plan

**Status**: Planning  
**Created**: 2025-11-28  
**Priority**: Critical bugs → High complexity → Medium improvements → Low polish

---

## Overview

This plan refactors the vLLM provider to eliminate duplication, fix bugs, and improve maintainability while preserving all existing functionality.

**Key Principle**: Extract and normalize, don't rebuild. Keep what works, fix what's duplicated.

---

## Phase 1: CRITICAL - Eliminate Duplication (Sprint 1)

### Goal
Fix critical bugs and eliminate 136 lines of duplicated code

### Problem
Lines 988-1124 have parallel code paths for SDK objects vs dictionaries, causing:
- Event emission bug (tool calls missing from event_blocks)
- Maintenance burden (changes needed in two places)
- Code divergence (already happening)

### Solution: Normalize at Entry Point

**Strategy**: Convert SDK objects to dictionaries at the top of `_convert_to_chat_response`, then process once.

#### Step 1: Add Normalization Methods

```python
def _normalize_block(self, block: Any) -> dict:
    """Convert SDK object or dict to normalized dict format.
    
    Args:
        block: SDK object or dictionary from response.output
    
    Returns:
        Dictionary with normalized structure
    """
    # Already a dict - return as-is
    if isinstance(block, dict):
        return block
    
    # SDK object - convert to dict
    normalized = {}
    
    # Extract type
    if hasattr(block, "type"):
        normalized["type"] = block.type
    
    # Extract content
    if hasattr(block, "content"):
        content = block.content
        # Recursively normalize content items if it's a list
        if isinstance(content, list):
            normalized["content"] = [
                self._normalize_content_item(item) for item in content
            ]
        else:
            normalized["content"] = content
    
    return normalized


def _normalize_content_item(self, item: Any) -> dict:
    """Normalize a content item (nested in block.content).
    
    Args:
        item: SDK object or dict from block.content array
    
    Returns:
        Normalized dictionary
    """
    if isinstance(item, dict):
        return item
    
    result = {}
    
    # Common fields
    if hasattr(item, "type"):
        result["type"] = item.type
    if hasattr(item, "text"):
        result["text"] = item.text
    if hasattr(item, "thinking"):
        result["thinking"] = item.thinking
    
    # Tool call fields
    if hasattr(item, "id"):
        result["id"] = item.id
    if hasattr(item, "function"):
        function = item.function
        if hasattr(function, "name"):
            result["function"] = {
                "name": function.name,
                "arguments": getattr(function, "arguments", "")
            }
        elif isinstance(function, dict):
            result["function"] = function
    
    return result
```

#### Step 2: Refactor `_convert_to_chat_response` to Single Code Path

Replace lines 988-1124 with:

```python
# NORMALIZE: Convert all blocks to dict format once
normalized_blocks = [self._normalize_block(block) for block in response.output]

# SINGLE CODE PATH: Process normalized blocks
for block in normalized_blocks:
    block_type = block.get("type")
    
    if block_type == "message":
        # Extract text from message content
        block_content = block.get("content", [])
        
        if isinstance(block_content, list):
            for content_item in block_content:
                if content_item.get("type") == "output_text":
                    text = content_item.get("text", "")
                    content_blocks.append(TextBlock(text=text))
                    text_accumulator.append(text)
                    event_blocks.append(
                        TextContent(text=text, raw=content_item.get("raw"))
                    )
        elif isinstance(block_content, str):
            content_blocks.append(TextBlock(text=block_content))
            text_accumulator.append(block_content)
            event_blocks.append(TextContent(text=block_content))
    
    elif block_type == "reasoning":
        # Extract reasoning text
        reasoning_text = None
        block_content = block.get("content")
        
        if block_content and isinstance(block_content, list):
            texts = []
            for item in block_content:
                if item.get("type") == "reasoning_text":
                    texts.append(item.get("text", ""))
            if texts:
                reasoning_text = "\n".join(texts)
        
        if reasoning_text:
            # Create ThinkingBlock for content_blocks
            thinking_block = ThinkingBlock(thinking=reasoning_text)
            content_blocks.append(thinking_block)
            
            # Create ThinkingContent for event_blocks
            event_blocks.append(
                ThinkingContent(
                    thinking=reasoning_text,
                    summary=block.get("summary"),
                    raw=block
                )
            )
    
    elif block_type == "tool_call":
        # Extract tool call details
        tool_id = block.get("id", str(uuid.uuid4()))
        function = block.get("function", {})
        tool_name = function.get("name", "unknown")
        tool_input = function.get("arguments", "{}")
        
        # Parse arguments if string
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except:
                pass
        
        if tool_input is None:
            tool_input = {}
        
        if not isinstance(tool_input, dict):
            tool_input = {}
        
        # Add to all three lists (FIX: Previously missing from SDK path)
        content_blocks.append(
            ToolCallBlock(id=tool_id, name=tool_name, input=tool_input)
        )
        tool_calls.append(
            ToolCall(id=tool_id, name=tool_name, arguments=tool_input)
        )
        event_blocks.append(
            ToolCallContent(
                id=tool_id, name=tool_name, arguments=tool_input, raw=block
            )
        )
```

### Testing Strategy

```python
# Test normalization
def test_normalize_sdk_object():
    """Verify SDK objects convert to expected dict structure"""

def test_normalize_dict_passthrough():
    """Verify dicts pass through unchanged"""

def test_normalize_nested_content():
    """Verify nested content arrays normalize correctly"""

# Test unified parsing
def test_parse_message_block():
    """Test message block parsing with normalized input"""

def test_parse_reasoning_block():
    """Test reasoning block parsing with normalized input"""

def test_parse_tool_call_block():
    """Test tool call parsing includes all three outputs"""
    # This test should verify the bug fix: event_blocks contains tool calls
```

### Implementation Steps

1. [ ] Add `_normalize_block()` method
2. [ ] Add `_normalize_content_item()` method
3. [ ] Write unit tests for normalization (test both SDK objects and dicts)
4. [ ] Refactor `_convert_to_chat_response` to normalize at entry
5. [ ] Replace lines 988-1124 with single code path
6. [ ] Verify event_blocks includes tool calls (bug fix validation)
7. [ ] Run existing integration tests
8. [ ] Delete old dual-path code (136 lines)

### Success Criteria

- [ ] All tests pass
- [ ] Event emission bug fixed (tool calls in event_blocks)
- [ ] Code reduced by ~136 lines
- [ ] No behavior changes
- [ ] Both SDK objects and dicts handled correctly

### Risks & Mitigation

**Risk**: Normalization adds overhead  
**Mitigation**: Negligible - response processing is tiny compared to network I/O

**Risk**: New bugs in normalization  
**Mitigation**: Comprehensive unit tests for edge cases (empty content, missing fields, None values)

---

## Phase 2: HIGH - Refactor `_complete_chat_request` (Sprint 2)

### Goal
Break 400-line method into focused, testable units

### Problem
Single method doing 7+ responsibilities, impossible to test independently

### Solution: Extract Focused Methods

#### Method 1: Message Separation

```python
def _separate_messages_by_role(
    self, messages: list
) -> tuple[list, list, list]:
    """Separate messages into system, developer, and conversation.
    
    Args:
        messages: Mixed list of messages
    
    Returns:
        Tuple of (system_messages, developer_messages, conversation_messages)
    """
    system_msgs = [m for m in messages if m.role == "system"]
    developer_msgs = [m for m in messages if m.role == "developer"]
    conversation = [m for m in messages if m.role in ("user", "assistant", "tool")]
    
    return system_msgs, developer_msgs, conversation
```

#### Method 2: Instructions Building

```python
def _build_instructions(
    self, system_messages: list, developer_messages: list
) -> str:
    """Build combined instructions from system and developer messages.
    
    Args:
        system_messages: System-role messages
        developer_messages: Developer-role messages
    
    Returns:
        Combined instruction string
    """
    parts = []
    
    if system_messages:
        parts.extend(msg.content for msg in system_messages)
    
    if developer_messages:
        parts.extend(msg.content for msg in developer_messages)
    
    return "\n\n".join(parts) if parts else ""
```

#### Method 3: Parameter Building

```python
def _build_api_parameters(
    self,
    messages: list,
    instructions: str,
    request: ChatRequest,
    **kwargs
) -> dict:
    """Build parameters for vLLM API call.
    
    Args:
        messages: Converted messages for API
        instructions: Combined system/developer instructions
        request: Original ChatRequest
        kwargs: Override parameters
    
    Returns:
        Dictionary of API parameters
    """
    model = kwargs.get("model") or request.model or self.default_model
    
    params = {
        "model": model,
        "input": messages,
    }
    
    # Add instructions if present
    if instructions:
        params["instructions"] = instructions
    
    # Thinking/reasoning parameters
    if request.thinking_enabled:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": request.thinking_budget or 8000,
        }
    
    if request.reasoning_effort:
        params["reasoning_effort"] = request.reasoning_effort
    
    # Tools
    if request.tools:
        params["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters or {},
                    "strict": tool.strict,
                },
            }
            for tool in request.tools
        ]
    
    # Generation parameters
    if request.temperature is not None:
        params["temperature"] = request.temperature
    
    max_tokens = kwargs.get("max_tokens") or request.max_tokens or self.max_tokens
    if max_tokens:
        params["max_tokens"] = max_tokens
    
    if request.top_p is not None:
        params["top_p"] = request.top_p
    
    if request.stop_sequences:
        params["stop"] = request.stop_sequences
    
    # Response format
    if request.response_format:
        params["response_format"] = request.response_format
    
    # Metadata
    if request.metadata:
        params["metadata"] = request.metadata
    
    # Timeout
    params["timeout"] = self.timeout
    
    # Stateless mode
    params["store"] = False
    
    return params
```

#### Method 4: Continuation Loop

```python
async def _handle_continuation_loop(
    self, initial_response: Any, params: dict, request: ChatRequest
) -> Any:
    """Handle continuation for incomplete responses.
    
    Args:
        initial_response: First API response
        params: API parameters for continuation calls
        request: Original ChatRequest
    
    Returns:
        Final complete response
    """
    final_response = initial_response
    continuation_count = 0
    accumulated_output = []
    
    # Accumulate initial output
    if hasattr(initial_response, "output"):
        accumulated_output.extend(initial_response.output)
    
    # Continue while incomplete
    while (
        hasattr(final_response, "status")
        and final_response.status == "incomplete"
        and continuation_count < MAX_CONTINUATION_ATTEMPTS
    ):
        continuation_count += 1
        
        logger.info(
            f"[PROVIDER] {self.api_label} response incomplete, "
            f"continuing (attempt {continuation_count}/{MAX_CONTINUATION_ATTEMPTS})"
        )
        
        # Build continuation input
        continuation_input = self._build_continuation_input(
            params["input"], accumulated_output
        )
        
        # Update parameters for continuation
        continuation_params = params.copy()
        continuation_params["input"] = continuation_input
        
        # Make continuation call
        continuation_response = await self.client.responses.create(
            **continuation_params
        )
        
        # Accumulate output
        if hasattr(continuation_response, "output"):
            accumulated_output.extend(continuation_response.output)
        
        final_response = continuation_response
    
    # Store continuation metadata
    if continuation_count > 0:
        if not hasattr(final_response, "metadata"):
            final_response.metadata = {}
        final_response.metadata[METADATA_CONTINUATION_COUNT] = continuation_count
    
    return final_response
```

#### Method 5: Request Event Emission

```python
async def _emit_request_events(
    self, params: dict, request: ChatRequest
):
    """Emit observability events for request.
    
    Args:
        params: API parameters
        request: Original ChatRequest
    """
    # INFO level: Summary
    await self.coordinator.hooks.emit(
        "llm:request",
        {
            "provider": self.name,
            "model": params.get("model"),
            "message_count": len(params.get("input", [])),
            "has_tools": bool(params.get("tools")),
            "max_tokens": params.get("max_tokens"),
        },
    )
    
    # DEBUG level: Truncated payload
    if self.debug:
        truncated_params = self._truncate_for_debug(params)
        await self.coordinator.hooks.emit(
            "llm:request:debug",
            {
                "provider": self.name,
                "params": truncated_params,
            },
        )
    
    # RAW level: Complete untruncated
    if self.debug and self.raw_debug:
        await self.coordinator.hooks.emit(
            "llm:request:raw",
            {
                "provider": self.name,
                "params": params,
            },
        )
```

#### Method 6: Response Event Emission

```python
async def _emit_response_events(self, response: Any, duration_ms: int):
    """Emit observability events for response.
    
    Args:
        response: API response
        duration_ms: Request duration in milliseconds
    """
    # INFO level
    usage_dict = {}
    if hasattr(response, "usage"):
        usage = response.usage
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif isinstance(usage, dict):
            usage_dict = usage
    
    await self.coordinator.hooks.emit(
        "llm:response",
        {
            "provider": self.name,
            "model": getattr(response, "model", None),
            "status": getattr(response, "status", "unknown"),
            "usage": usage_dict,
            "duration_ms": duration_ms,
        },
    )
    
    # DEBUG level
    if self.debug:
        truncated_response = self._truncate_response_for_debug(response)
        await self.coordinator.hooks.emit(
            "llm:response:debug",
            {
                "provider": self.name,
                "response": truncated_response,
            },
        )
    
    # RAW level
    if self.debug and self.raw_debug:
        await self.coordinator.hooks.emit(
            "llm:response:raw",
            {
                "provider": self.name,
                "response": response,
            },
        )
```

#### Refactored Main Method

```python
async def _complete_chat_request(
    self, request: ChatRequest, **kwargs
) -> ChatResponse:
    """Complete a chat request using vLLM Responses API.
    
    Orchestrates: separation → conversion → parameters → emission →
    API call → continuation → response conversion
    """
    start_time = time.time()
    message_list = request.messages
    
    logger.info(
        f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages"
    )
    
    try:
        # Step 1: Separate messages by role
        system_msgs, developer_msgs, conversation = (
            self._separate_messages_by_role(message_list)
        )
        
        # Step 2: Build instructions
        instructions = self._build_instructions(system_msgs, developer_msgs)
        
        # Step 3: Convert messages
        input_messages = self._convert_messages(conversation)
        
        # Step 4: Build parameters
        params = self._build_api_parameters(
            input_messages, instructions, request, **kwargs
        )
        
        # Step 5: Emit events
        await self._emit_request_events(params, request)
        
        # Step 6: Make API call
        logger.debug(f"[PROVIDER] Calling {self.api_label} Responses API")
        response = await self.client.responses.create(**params)
        
        # Step 7: Handle continuation if needed
        final_response = await self._handle_continuation_loop(
            response, params, request
        )
        
        # Step 8: Emit response events
        duration_ms = int((time.time() - start_time) * 1000)
        await self._emit_response_events(final_response, duration_ms)
        
        # Step 9: Convert to ChatResponse
        return self._convert_to_chat_response(final_response)
    
    except Exception as e:
        logger.error(f"[PROVIDER] {self.api_label} API error: %s", e)
        # Emit error event
        await self.coordinator.hooks.emit(
            "llm:error",
            {
                "provider": self.name,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise
```

### Testing Strategy

```python
# Test each extracted method independently
def test_separate_messages_by_role():
    """Verify messages separated correctly"""

def test_build_instructions_empty():
    """Verify empty instructions handled"""

def test_build_instructions_combined():
    """Verify system + developer combined"""

def test_build_api_parameters_minimal():
    """Verify minimal parameters with defaults"""

def test_build_api_parameters_full():
    """Verify all parameters included"""

async def test_handle_continuation_loop_single():
    """Verify single response (no continuation)"""

async def test_handle_continuation_loop_multiple():
    """Verify multiple continuations"""

async def test_handle_continuation_loop_max_attempts():
    """Verify max attempts enforced"""

async def test_emit_request_events():
    """Verify events emitted at correct levels"""

async def test_emit_response_events():
    """Verify response events emitted correctly"""

# Integration test
async def test_complete_chat_request_end_to_end():
    """Full flow with mocked API (behavior unchanged)"""
```

### Implementation Steps

1. [ ] Add extracted methods (test each as you add)
2. [ ] Refactor main `_complete_chat_request` to call extracted methods
3. [ ] Run existing integration tests (should pass unchanged)
4. [ ] Add unit tests for extracted methods
5. [ ] Delete old inline code

### Success Criteria

- [ ] Main method <150 lines (down from 400)
- [ ] 6 new testable units
- [ ] All logic testable without API mocking
- [ ] No behavior changes
- [ ] Test coverage >80%

---

## Phase 3: MEDIUM - Tactical Improvements (Sprint 3)

### 3.1: Fix Production Assertion

**Location**: Line 257

**Replace**:
```python
assert max_length is not None, "max_length should never be None after initialization"
```

**With**:
```python
if max_length is None:
    raise RuntimeError(
        "max_length must be initialized before truncation. "
        "This indicates a provider initialization bug."
    )
```

### 3.2: Flatten Deep Nesting

Extract method to flatten lines 858-900:

```python
def _extract_reasoning_from_content(
    self, content: Any
) -> list[dict]:
    """Extract reasoning items from assistant message content.
    
    Returns list of reasoning items to append.
    """
    # Guard: Not a list
    if not isinstance(content, list):
        return []
    
    reasoning_items = []
    
    for block in content:
        # Guard: Not a dict
        if not isinstance(block, dict):
            continue
        
        block_type = block.get("type")
        
        # Guard: Not thinking block
        if block_type != "thinking":
            continue
        
        thinking_text = block.get("thinking", "")
        
        # Guard: Empty thinking
        if not thinking_text:
            continue
        
        # Main logic (at top level!)
        reasoning_item = {
            "type": "reasoning",
            "content": [
                {
                    "type": "reasoning_text",
                    "text": thinking_text
                }
            ]
        }
        reasoning_items.append(reasoning_item)
    
    return reasoning_items
```

### 3.3: Extract Continuation Logic

Make continuation logic independently testable by extracting helper methods.

### Success Criteria

- [ ] 0 assertions in production code
- [ ] Max nesting depth ≤3 levels
- [ ] All edge cases covered

---

## Phase 4: LOW - Code Cleanup

### 4.1: Standardize Logging

Convert all logging to f-strings:

**Before**:
```python
logger.error("[PROVIDER] %s API error: %s", self.api_label, e)
```

**After**:
```python
logger.error(f"[PROVIDER] {self.api_label} API error: {e}")
```

### 4.2: Extract Magic Strings

Add to `_constants.py`:

```python
# Block types
BLOCK_TYPE_MESSAGE = "message"
BLOCK_TYPE_REASONING = "reasoning"
BLOCK_TYPE_TOOL_CALL = "tool_call"

# Content types
CONTENT_TYPE_OUTPUT_TEXT = "output_text"
CONTENT_TYPE_REASONING_TEXT = "reasoning_text"
CONTENT_TYPE_SUMMARY_TEXT = "summary_text"
```

### 4.3: Remove Redundant Type Guards

Lines 1054-1055, 1120-1121 have redundant isinstance checks - remove after ensuring test coverage.

### Success Criteria

- [ ] Consistent code style
- [ ] 0 magic strings in logic
- [ ] Clean linter pass

---

## What NOT to Change

### Keep These Excellent Patterns ✅

1. **Constants Extraction** (`_constants.py`) - Single source of truth
2. **Helper Modules** - `_response_handling.py`, `_token_accounting.py`
3. **Defensive Tool Validation** (lines 269-318) - Critical safety net
4. **Layered Observability** - INFO/DEBUG/RAW event structure
5. **Stateless Continuation** - `_build_continuation_input()` method
6. **Direct AsyncOpenAI Usage** - No unnecessary wrappers
7. **Clear Configuration** - Explicit defaults, no magic
8. **Extension Point** - `VLLMChatResponse` for streaming UI

---

## Risk Management

### Risk: Breaking Existing Behavior

**Mitigation**:
- Run integration tests after each phase
- Use feature flags for gradual rollout if needed
- Keep old code until new code proven

### Risk: Performance Regression

**Mitigation**:
- Profile before/after if concerned
- Normalization overhead is negligible vs network I/O

### Risk: New Bugs in Refactored Code

**Mitigation**:
- Comprehensive unit tests
- Integration tests for smoke testing
- Test edge cases (empty content, missing fields, None values)

### Risk: Test Maintenance Burden

**Mitigation**:
- Focus tests on contracts, not implementation
- Test public behavior, not private methods

---

## Progress Tracking

### Phase 1: CRITICAL
- [ ] Normalization methods added
- [ ] Tests written
- [ ] Single code path implemented
- [ ] Event emission bug fixed
- [ ] Duplicate code deleted
- [ ] Integration tests pass

### Phase 2: HIGH
- [ ] 6 methods extracted
- [ ] Unit tests added
- [ ] Main method refactored
- [ ] Integration tests pass

### Phase 3: MEDIUM
- [ ] Assertion fixed
- [ ] Nesting flattened
- [ ] Edge cases tested

### Phase 4: LOW
- [ ] Logging standardized
- [ ] Magic strings extracted
- [ ] Redundant code removed

---

## Timeline Estimate

- **Phase 1**: 2-3 days (critical path)
- **Phase 2**: 3-5 days (extraction + testing)
- **Phase 3**: 1-2 days (tactical fixes)
- **Phase 4**: 1 day (cleanup)

**Total**: ~2 weeks with testing

---

## Additional Resources

- Original code review: [CODE_REVIEW_2025-11-28.md](./CODE_REVIEW_2025-11-28.md)
- Detailed issues: [ISSUES_DETAILED.md](./ISSUES_DETAILED.md)
