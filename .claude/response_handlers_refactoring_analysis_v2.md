# Response Handlers Refactoring Analysis (Updated with Harmony)

## Overview

This document analyzes **three** Responses API implementations to identify common components that can be refactored into reusable utilities.

## Three Implementations - Architecture Comparison

### 1. gRPC Standard Responses (`grpc/responses/`)

**Purpose**: Handles `/v1/responses` for non-Harmony open source models (llama, mistral, etc.)

**Data Flow**:
```
ResponsesRequest → ChatCompletionRequest → Chat Pipeline → ChatCompletionResponse → ResponsesResponse
                                                     ↓
                                             MCP Tool Loop (if tools present)
```

**Key Characteristics**:
- **Format Conversion**: ResponsesRequest ↔ ChatCompletionRequest (bidirectional)
- **Pipeline**: Uses chat completion pipeline stages
- **Event Emission**: Creates SSE events from scratch using `ResponseStreamEventEmitter`
- **Input**: Receives `ChatCompletionStreamResponse` chunks from pipeline
- **Tool Loop**: `tool_loop.rs` with local `execute_mcp_call()` and `ToolLoopState`
- **Streaming**: `streaming.rs` with `ResponseStreamEventEmitter` that builds complete SSE events

### 2. OpenAI Router (`openai/`)

**Purpose**: Proxies requests to third-party OpenAI-compatible APIs (OpenAI, xAI, etc.)

**Data Flow**:
```
ResponsesRequest → Payload Transform → Upstream API (SSE) → Transform Events → Client
                                              ↓
                                      MCP Tool Loop (if tools present)
```

**Key Characteristics**:
- **No Format Conversion**: Works directly with Responses payloads
- **HTTP Proxy**: Uses reqwest to call upstream APIs
- **Event Transformation**: Parses existing SSE events and transforms them (function_call → mcp_call)
- **Input**: Receives SSE blocks from upstream API
- **Tool Loop**: `mcp.rs` with canonical `execute_mcp_call()` and `ToolLoopState`
- **Streaming**: `streaming.rs` with SSE parsing and in-place transformation

### 3. Harmony Responses (`grpc/responses/harmony.rs`) - **NEW**

**Purpose**: Handles `/v1/responses` for gpt-oss (Harmony) models specifically

**Data Flow**:
```
ResponsesRequest → GenerateRequest → Generate Pipeline → Tokens → StreamableParser → ResponseOutputItems
                                                  ↓
                                          MCP Tool Loop (for built-in tools: browser, python)
```

**Key Characteristics**:
- **No Chat Conversion**: Uses generate pipeline directly (NOT chat completion)
- **Token-Level Processing**: Uses Harmony's `StreamableParser` for direct token processing
- **Channel-Based Routing**: Routes Harmony messages to output items by channel:
  - `"analysis"` → Reasoning items
  - `"commentary"` → Tool calls (browser.*, python, functions.*)
  - `"final"` → Message items
- **Native Responses Format**: No conversion needed - processes Responses API natively
- **Pipeline**: Uses generate pipeline (lower level than chat)
- **Tool Loop**: Has own implementation for built-in Harmony tools
- **Streaming**: Direct token streaming through `StreamableParser`

## Key Architectural Differences

| Aspect | gRPC Standard | OpenAI Router | Harmony |
|--------|--------------|---------------|---------|
| **Backend** | Chat pipeline | HTTP proxy | Generate pipeline |
| **Input Format** | Chat chunks | SSE blocks | Token stream |
| **Processing** | Creates events | Transforms events | Parses tokens to items |
| **Format Conversion** | Responses ↔ Chat | None | None |
| **Tool Detection** | From chat response | From SSE events | From Harmony channels |
| **Event Creation** | From scratch (`ResponseStreamEventEmitter`) | Transform existing SSE | From parser messages |
| **Tool Loop** | Chat-based | Responses-based | Generate-based |

## Updated Common Components Analysis

### 1. ✅ **MCP Tool Execution** (HIGH Priority - Shared by All Three)

**Current State**:
- **OpenAI**: Canonical implementation in `openai/mcp.rs:179-201`
- **gRPC Standard**: Re-exports OpenAI's version (already shared!)
- **Harmony**: Re-exports OpenAI's version (already shared!)

**Implementation**:
```rust
// openai/mcp.rs (current canonical version)
pub async fn execute_mcp_call(
    mcp_mgr: &Arc<McpClientManager>,
    tool_name: &str,
    args_json_str: &str,
) -> Result<(String, String), String> {
    // Parse arguments
    let args: serde_json::Value = serde_json::from_str(args_json_str)
        .map_err(|e| format!("Failed to parse arguments: {}", e))?;

    // Execute via MCP manager
    let result = mcp_mgr.execute_tool(tool_name, args).await
        .map_err(|e| e.to_string())?;

    // Return (server_name, output)
    Ok((result.server_name, result.output))
}
```

**Status**: ✅ **Already well-factored** - all three implementations share this

### 2. ✅ **Tool Loop State Management** (HIGH Priority - Similar Pattern)

**Current Implementations**:

| Component | gRPC Standard | OpenAI | Harmony |
|-----------|--------------|--------|---------|
| State struct | `ToolLoopState` (local) | `ToolLoopState` (in mcp.rs) | Custom per-loop tracking |
| Config | Hardcoded constants | `McpLoopConfig` | Hardcoded max iterations |
| Conversation history | Yes | Yes | Harmony format (different structure) |

**Commonalities** (gRPC Standard & OpenAI):
```rust
struct ToolLoopState {
    iteration: usize,
    total_calls: usize,
    conversation_history: Vec<Value>,
    original_input: ResponseInput,
}
```

**Harmony Differences**:
- Uses Harmony-specific message format (not generic conversation history)
- Tracks tool calls in Harmony's channel system
- Different loop termination logic (based on Harmony message channels)

**Recommendation**: ✅ **Extract for gRPC Standard + OpenAI**
- These two share identical structure
- Harmony is different enough to keep separate
- Create `common/tool_loop_state.rs` for non-Harmony implementations

### 3. ⚠️ **SSE Event Processing** (MEDIUM Priority - Partially Shared)

**Comparison**:

| Component | gRPC Standard | OpenAI | Harmony |
|-----------|--------------|--------|---------|
| Event parsing | N/A (works with chunks) | `parse_sse_block()` | N/A (works with tokens) |
| Event creation | `ResponseStreamEventEmitter` | Manual SSE formatting | Harmony message conversion |
| Sequence tracking | Internal to emitter | External `sequence_number` | N/A |
| Output index mapping | N/A | `OutputIndexMapper` | N/A |

**Shared Utilities** (OpenAI only):
- `parse_sse_block()` - parses SSE format
- `SequenceTracker` - could be extracted
- Event type constants (already shared via `event_types` module)

**Recommendation**: ⚠️ **Extract SSE parsing utilities**
- `parse_sse_block()` is OpenAI-specific but could be useful for future proxy implementations
- Event type constants already shared
- Event emission/transformation are fundamentally different operations (don't abstract)

### 4. ❌ **Streaming Response Accumulation** (Keep Separate)

**Why They're Different**:

| Accumulator | Input Type | Purpose | Output |
|-------------|-----------|---------|--------|
| gRPC Standard | `ChatCompletionStreamResponse` | Convert chat chunks to Responses | `ResponsesResponse` |
| OpenAI | SSE blocks (strings) | Extract final response from events | `Value` (JSON) |
| Harmony | Tokens + `StreamableParser` | Parse tokens to output items | `Vec<ResponseOutputItem>` |

**Recommendation**: ❌ **Keep all three separate**
- Fundamentally different input types and processing logic
- Abstracting would create leaky abstraction

### 5. ✅ **Data Storage & Persistence** (Already Shared!)

**Current State**: All three use the same persistence function:

```rust
// openai/conversations.rs (shared by all three)
pub async fn persist_conversation_items(
    conversation_storage: Arc<dyn ConversationStorage>,
    item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response_json: &Value,
    original_body: &ResponsesRequest,
) -> Result<(), String>
```

**Status**: ✅ **Already well-factored** - no changes needed

### 6. ✅ **MCP Manager Creation** (Already Shared!)

**Current State**:
```rust
// openai/mcp.rs
pub async fn mcp_manager_from_request_tools(
    tools: &[ResponseTool],
) -> Option<Arc<McpClientManager>>

// grpc/responses/tool_loop.rs
pub(super) use crate::routers::openai::mcp::mcp_manager_from_request_tools
    as create_mcp_manager_from_request;

// grpc/responses/harmony.rs
pub(super) use crate::routers::openai::mcp::mcp_manager_from_request_tools
    as create_mcp_manager_from_request;
```

**Status**: ✅ **Already well-factored** - no changes needed

### 7. ⚠️ **Tool Loop Execution** (Different Enough to Keep Separate)

**Why They're Different**:

| Implementation | Request Format | Response Format | Tool Detection | Iteration Logic |
|----------------|----------------|----------------|----------------|-----------------|
| gRPC Standard | Chat | Chat | Parse chat response | Chat-based loop |
| OpenAI | Responses | SSE events | Parse SSE events | SSE-based loop |
| Harmony | Responses | Harmony messages | Parse channel field | Channel-based loop |

**Core differences**:
1. **gRPC Standard**: Converts Responses → Chat, executes, converts back
2. **OpenAI**: Works with Responses directly, parses SSE events
3. **Harmony**: Works with Responses directly, uses Harmony channel system

**Recommendation**: ⚠️ **Keep separate**
- Loop control logic is similar but request/response handling is fundamentally different
- Attempting to unify would require exposing format conversion details
- Better to share utilities (`execute_mcp_call`, `ToolLoopState`) than full loops

## Revised Refactoring Recommendations

### Phase 1: Extract Shared Utilities (Low Risk)

Create `sgl-router/src/routers/common/` module:

```rust
// src/routers/common/mod.rs
pub mod mcp_tool_execution;  // Already shared, just move location
pub mod sse_utils;            // OpenAI-specific but useful for future proxy implementations
pub mod tool_loop_state;      // Shared by gRPC Standard + OpenAI (NOT Harmony)
pub mod event_types;          // Already shared, consolidate location
```

#### 1. MCP Tool Execution (`common/mcp_tool_execution.rs`)

**Current Location**: `openai/mcp.rs:179-201`

**Action**: Move to common module (all three already use it)

```rust
use std::sync::Arc;
use serde_json::Value;
use crate::mcp::McpClientManager;

/// Execute an MCP tool call
pub async fn execute_mcp_call(
    mcp_mgr: &Arc<McpClientManager>,
    tool_name: &str,
    args_json_str: &str,
) -> Result<(String, String), String> {
    // Current implementation from openai/mcp.rs
    // Returns (server_name, output)
}
```

#### 2. Tool Loop State (`common/tool_loop_state.rs`)

**Shared by**: gRPC Standard + OpenAI (NOT Harmony - it uses Harmony-specific format)

```rust
use serde_json::Value;
use crate::protocols::responses::ResponseInput;

/// State for tracking multi-turn tool calling loops
/// Used by non-Harmony implementations (gRPC Standard, OpenAI)
pub struct ToolLoopState {
    pub iteration: usize,
    pub total_calls: usize,
    pub conversation_history: Vec<Value>,
    pub original_input: ResponseInput,
}

impl ToolLoopState {
    pub fn new(original_input: ResponseInput) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
        }
    }

    pub fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        arguments: String,
        output: String,
    ) {
        // Add function_call item
        self.conversation_history.push(json!({
            "type": "function_call",
            "call_id": call_id,
            "name": tool_name,
            "arguments": arguments
        }));

        // Add function_call_output item
        self.conversation_history.push(json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output
        }));
    }
}

/// Configuration for MCP tool calling loops
#[derive(Debug, Clone)]
pub struct McpLoopConfig {
    pub max_iterations: usize,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self { max_iterations: 10 }
    }
}
```

#### 3. SSE Utilities (`common/sse_utils.rs`)

**Used by**: OpenAI (could be useful for future HTTP proxy implementations)

```rust
use std::borrow::Cow;

/// Parse an SSE block into event name and data
pub fn parse_sse_block(block: &str) -> (Option<&str>, Cow<'_, str>) {
    // Current implementation from openai/streaming.rs:525-544
}

/// Sequence number tracker for SSE events
pub struct SequenceTracker(u64);

impl SequenceTracker {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn next(&mut self) -> u64 {
        let current = self.0;
        self.0 += 1;
        current
    }

    pub fn current(&self) -> u64 {
        self.0
    }
}
```

#### 4. Event Type Constants (`common/event_types.rs`)

**Currently in**: `openai/utils.rs`

**Move to**: `common/event_types.rs` (all three implementations use these)

```rust
// Response lifecycle events
pub const RESPONSE_CREATED: &str = "response.created";
pub const RESPONSE_IN_PROGRESS: &str = "response.in_progress";
pub const RESPONSE_COMPLETED: &str = "response.completed";

// Output item events
pub const OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
pub const OUTPUT_ITEM_DONE: &str = "response.output_item.done";
pub const OUTPUT_ITEM_DELTA: &str = "response.output_item.delta";

// ... (rest of constants)
```

### Phase 2: Document Architecture Differences (No Code Changes)

Add architectural documentation to explain why certain components are NOT shared:

```rust
// src/routers/README.md

## Response Handling Architecture

### Three Implementations

1. **gRPC Standard** (`grpc/responses/`) - Non-Harmony models
   - Chat pipeline with format conversion
   - Creates SSE events from chat chunks

2. **OpenAI Router** (`openai/`) - Third-party API proxy
   - HTTP proxy with SSE transformation
   - Transforms existing SSE events

3. **Harmony** (`grpc/responses/harmony.rs`) - gpt-oss models
   - Generate pipeline with token-level parsing
   - No format conversion, native Responses API

### Shared Components

- MCP tool execution (`common/mcp_tool_execution.rs`)
- Tool loop state (`common/tool_loop_state.rs`) - gRPC Standard + OpenAI only
- Event type constants (`common/event_types.rs`)
- Persistence (`openai/conversations.rs`)

### Why NOT Shared

- **Tool loops**: Different request/response formats and loop control logic
- **Event emission**: Creation vs transformation vs token parsing (fundamentally different)
- **Streaming accumulators**: Different input types and state management
```

## Summary of Refactoring Opportunities

### ✅ **Recommended: Extract Common Utilities**

1. **MCP Tool Execution** - Already shared, just consolidate location
2. **Tool Loop State** - Shared by gRPC Standard + OpenAI (NOT Harmony)
3. **SSE Utilities** - OpenAI-specific but useful for future implementations
4. **Event Type Constants** - Shared by all three

### ❌ **Not Recommended: Over-abstraction**

Don't abstract:
1. **Tool loop execution** - Request/response formats differ significantly
2. **Event emission/transformation/parsing** - Three fundamentally different operations
3. **Streaming accumulators** - Different input types and state management
4. **Format conversions** - Implementation-specific (gRPC Standard only)

## Key Insights

1. **Three Distinct Architectures**:
   - gRPC Standard: Pipeline-based with format conversion
   - OpenAI: HTTP proxy with event transformation
   - Harmony: Token-level parsing with native Responses format

2. **Shared Utilities, Not Frameworks**:
   - Extract small, well-defined utilities (MCP execution, state tracking)
   - Don't force architectural unification

3. **Harmony is Unique**:
   - Uses completely different processing approach (token-level vs request-level)
   - Different tool loop mechanism (Harmony channels vs generic conversation history)
   - Should remain separate from the other two

4. **Already Well-Factored**:
   - MCP tool execution is already shared
   - Persistence is already shared
   - Most differences are legitimate architectural choices

## Implementation Principles

1. ✅ **Extract utilities, not frameworks** - Share functions, not architectures
2. ✅ **Document differences** - Make it clear why some things aren't shared
3. ✅ **Preserve existing behavior** - Zero functional changes
4. ✅ **Minimize abstraction** - Simple functions > complex traits
5. ✅ **Respect architectural differences** - gRPC vs HTTP vs Harmony are fundamentally different

## Conclusion

The three Responses API implementations represent **legitimate architectural diversity**:

- **gRPC Standard**: Format conversion + pipeline processing
- **OpenAI**: HTTP proxy + event transformation
- **Harmony**: Token-level parsing + native processing

Refactoring should focus on:
1. ✅ Consolidating already-shared utilities (MCP execution, persistence)
2. ✅ Extracting truly common state management (ToolLoopState for non-Harmony)
3. ✅ Moving event type constants to common location
4. ❌ NOT forcing architectural unification where differences are legitimate

The best approach: **Extract small, well-defined utilities** that all implementations can use, while respecting that they solve fundamentally different problems in fundamentally different ways.
