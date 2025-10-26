# Unified Harmony Pipeline Design

**Document Version**: 1.0
**Date**: 2025-10-25
**Status**: Design Proposal

## Executive Summary

This document describes the design for a unified harmony pipeline that supports both Chat Completion and Responses API for GPT OSS models. The pipeline integrates harmony-based request building, MCP tool loop orchestration, and response parsing while maintaining compatibility with existing infrastructure.

**Key Design Goals**:
- Single unified pipeline handling both Chat Completion and Responses API
- Harmony integration for request building and response parsing
- MCP loop support for tool orchestration in Responses API
- Minimal code duplication through shared components
- Support for streaming and non-streaming modes
- Support for single and dual (prefill+decode) dispatch modes

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GrpcRouter                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              HarmonyPipeline                              │  │
│  │                                                            │  │
│  │  ┌──────────────┐    ┌──────────────┐   ┌─────────────┐ │  │
│  │  │   Request    │───▶│   Harmony    │──▶│   Worker    │ │  │
│  │  │  Processor   │    │   Builder    │   │  Selector   │ │  │
│  │  └──────────────┘    └──────────────┘   └─────────────┘ │  │
│  │                                                            │  │
│  │  ┌──────────────┐    ┌──────────────┐   ┌─────────────┐ │  │
│  │  │   Dispatch   │───▶│   Response   │──▶│     MCP     │ │  │
│  │  │  Coordinator │    │   Parser     │   │   Manager   │ │  │
│  │  └──────────────┘    └──────────────┘   └─────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  gRPC Scheduler  │
                    └──────────────────┘
```

### Component Breakdown

1. **HarmonyPipeline**: Main orchestrator that routes between Chat and Responses flows
2. **HarmonyRequestBuilder**: Converts API requests to harmony format
3. **HarmonyResponseParser**: Parses gRPC responses through harmony channels
4. **RequestContext**: Tracks request state across MCP loops
5. **DispatchCoordinator**: Manages gRPC dispatch and stream collection
6. **McpLoopOrchestrator**: Orchestrates MCP tool loops for Responses API
7. **StreamingProcessor**: Handles SSE streaming for both modes
8. **ResponseProcessor**: Handles non-streaming response assembly

---

## Detailed Request Flows

### Responses API Flow (8 Steps with MCP Loop)

```
┌────────────────────────────────────────────────────────────────┐
│                    Responses API Flow                           │
└────────────────────────────────────────────────────────────────┘

1. Build Request via Harmony
   ┌──────────────────────────────────────────────┐
   │ ResponsesRequest                             │
   │   ├─ previous_response_id OR conversation    │
   │   ├─ input (user message)                    │
   │   └─ tools, tool_choice, etc.                │
   └──────────────────────────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────────┐
   │ Load History from Storage                    │
   │   ├─ response_storage.get_chain()            │
   │   └─ conversation_storage.get_history()      │
   └──────────────────────────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────────┐
   │ HarmonyRequestBuilder                        │
   │   ├─ Build harmony messages array            │
   │   ├─ Apply chat template                     │
   │   └─ Generate harmony text                   │
   └──────────────────────────────────────────────┘
                    ▼
   RequestContext { harmony_text, messages, tools, ... }

2. Select Worker
   ┌──────────────────────────────────────────────┐
   │ WorkerRegistry::select_worker()              │
   │   ├─ Input: harmony_text, model              │
   │   ├─ Apply policy constraints                │
   │   └─ Return: worker_url, dispatch_metadata   │
   └──────────────────────────────────────────────┘

3. Construct gRPC Payload
   ┌──────────────────────────────────────────────┐
   │ proto::GenerateRequest                       │
   │   ├─ text = harmony_text                     │
   │   ├─ sampling_params                         │
   │   ├─ return_logprob, logprob_start_len       │
   │   └─ stream = request.stream                 │
   └──────────────────────────────────────────────┘

4. Execute gRPC Request
   ┌──────────────────────────────────────────────┐
   │ DispatchCoordinator::execute()               │
   │   ├─ Single dispatch OR                      │
   │   ├─ Dual dispatch (prefill + decode)        │
   │   └─ Return: ExecutionResult                 │
   └──────────────────────────────────────────────┘

5. Collect Response from gRPC
   ┌──────────────────────────────────────────────┐
   │ StreamCollector::collect()                   │
   │   ├─ Single: collect from one stream         │
   │   ├─ Dual: collect prefill + decode          │
   │   └─ Merge input_logprobs if needed          │
   └──────────────────────────────────────────────┘

6. Parse via Harmony Channels
   ┌──────────────────────────────────────────────┐
   │ HarmonyResponseParser                        │
   │   ├─ Decode tokens via stop decoder          │
   │   ├─ Parse reasoning content                 │
   │   ├─ Parse tool calls                        │
   │   └─ Build ResponsesResponse                 │
   └──────────────────────────────────────────────┘

7. Handle Tools/Reasoning/MCP/Events
   ┌──────────────────────────────────────────────┐
   │ ResponseProcessor::process()                 │
   │   ├─ Extract reasoning_content               │
   │   ├─ Extract tool_calls                      │
   │   ├─ Emit SSE events if streaming            │
   │   └─ Check for MCP tool calls                │
   └──────────────────────────────────────────────┘

8. MCP Loop (if MCP detected)
   ┌──────────────────────────────────────────────┐
   │ McpLoopOrchestrator                          │
   │   ├─ Execute MCP tool calls                  │
   │   ├─ Collect tool results                    │
   │   ├─ Update RequestContext with results      │
   │   ├─ Build new harmony request (back to #1)  │
   │   └─ Loop until no MCP detected              │
   └──────────────────────────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────────┐
   │ Persist Response to Storage                  │
   │   └─ response_storage.store()                │
   └──────────────────────────────────────────────┘
```

### Chat Completion Flow (7 Steps, No MCP Loop)

```
┌────────────────────────────────────────────────────────────────┐
│                  Chat Completion Flow                           │
└────────────────────────────────────────────────────────────────┘

1. Build Request via Harmony
   ┌──────────────────────────────────────────────┐
   │ ChatCompletionRequest                        │
   │   ├─ messages[]                              │
   │   ├─ tools, tool_choice                      │
   │   └─ stream, logprobs, etc.                  │
   └──────────────────────────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────────┐
   │ HarmonyRequestBuilder                        │
   │   ├─ Build harmony messages array            │
   │   ├─ Apply chat template                     │
   │   └─ Generate harmony text                   │
   └──────────────────────────────────────────────┘
                    ▼
   RequestContext { harmony_text, messages, tools, ... }

2. Select Worker (same as Responses API)

3. Construct gRPC Payload (same as Responses API)

4. Execute gRPC Request (same as Responses API)

5. Collect Response (same as Responses API)

6. Parse via Harmony Channels (same as Responses API)

7. Handle Tools/Reasoning/Chat Response
   ┌──────────────────────────────────────────────┐
   │ ResponseProcessor::process_chat()            │
   │   ├─ Extract reasoning_content               │
   │   ├─ Extract tool_calls                      │
   │   ├─ Build ChatCompletionResponse            │
   │   └─ Return to client (NO MCP LOOP)          │
   └──────────────────────────────────────────────┘
```

---

## Core Component Specifications

### 1. HarmonyPipeline

**Purpose**: Main orchestrator that routes requests through the appropriate flow

**Structure**:
```rust
pub struct HarmonyPipeline {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    harmony_builder: Arc<HarmonyRequestBuilder>,
    harmony_parser: Arc<HarmonyResponseParser>,
    dispatch_coordinator: Arc<DispatchCoordinator>,
    mcp_orchestrator: Arc<McpLoopOrchestrator>,
    streaming_processor: StreamingProcessor,
    response_processor: ResponseProcessor,
}

impl HarmonyPipeline {
    pub async fn execute_chat(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
    ) -> Response;

    pub async fn execute_responses(
        &self,
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        response_storage: SharedResponseStorage,
        conversation_storage: SharedConversationStorage,
        conversation_item_storage: SharedConversationItemStorage,
        mcp_manager: Arc<McpManager>,
    ) -> Response;
}
```

**Responsibilities**:
- Route requests to appropriate flow (chat vs responses)
- Coordinate between harmony builder, worker selection, dispatch, and parsing
- Orchestrate MCP loops for responses API
- Handle both streaming and non-streaming modes

### 2. HarmonyRequestBuilder

**Purpose**: Convert API requests to harmony format

**Structure**:
```rust
pub struct HarmonyRequestBuilder {
    tokenizer: Arc<dyn Tokenizer>,
    tool_parser_factory: ToolParserFactory,
}

pub struct HarmonyMessages {
    pub messages: Vec<HarmonyMessage>,
    pub text: String,  // Final harmony-formatted text
}

pub struct HarmonyMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub reasoning_content: Option<String>,
}

impl HarmonyRequestBuilder {
    /// Build harmony messages from chat request
    pub async fn build_from_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<HarmonyMessages, String>;

    /// Build harmony messages from responses request + history
    pub async fn build_from_responses(
        &self,
        request: &ResponsesRequest,
        history: Vec<ResponsesResponse>,
    ) -> Result<HarmonyMessages, String>;

    /// Add MCP tool results to harmony context
    pub async fn append_mcp_results(
        &self,
        current: &HarmonyMessages,
        tool_results: Vec<McpToolResult>,
    ) -> Result<HarmonyMessages, String>;

    /// Apply chat template and generate final text
    async fn apply_chat_template(
        &self,
        messages: Vec<HarmonyMessage>,
        tools: Option<&[Tool]>,
        tool_choice: Option<&ToolChoice>,
    ) -> Result<String, String>;
}
```

**Key Operations**:
1. Convert ChatCompletionRequest → HarmonyMessages
2. Convert ResponsesRequest + history → HarmonyMessages
3. Apply chat template with tool constraints
4. Generate final harmony text for worker selection and gRPC dispatch
5. Append MCP tool results for loop continuation

### 3. HarmonyResponseParser

**Purpose**: Parse gRPC responses through harmony channels

**Structure**:
```rust
pub struct HarmonyResponseParser {
    tokenizer: Arc<dyn Tokenizer>,
    tool_parser_factory: ToolParserFactory,
    reasoning_parser_factory: ReasoningParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
}

pub struct ParsedResponse {
    pub text: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub finish_reason: String,
    pub matched_stop: Option<Value>,
    pub logprobs: Option<ChatLogProbs>,
}

impl HarmonyResponseParser {
    /// Parse a single GenerateComplete response
    pub async fn parse_complete(
        &self,
        complete: &proto::GenerateComplete,
        stop_decoder: &mut StopSequenceDecoder,
        request_tools: Option<&[Tool]>,
        request_tool_choice: Option<&ToolChoice>,
        separate_reasoning: bool,
        model: &str,
        history_tool_calls_count: usize,
    ) -> Result<ParsedResponse, String>;

    /// Parse streaming chunks
    pub async fn parse_streaming_chunk(
        &self,
        chunk: &proto::GenerateChunk,
        state: &mut StreamingState,
    ) -> Result<Option<ParsedChunk>, String>;
}
```

**Parsing Steps** (per response):
1. Decode tokens using stop sequence decoder
2. Extract reasoning content (if separate_reasoning enabled)
3. Parse tool calls (if tools enabled)
   - JSON schema constraint mode: parse structured JSON
   - Parser mode: use model-specific tool parser
4. Convert logprobs to OpenAI format
5. Determine finish reason (tool_calls override)
6. Return ParsedResponse

### 4. RequestContext

**Purpose**: Track request state across MCP loops

**Structure**:
```rust
pub struct RequestContext {
    /// Unique request ID
    pub request_id: String,

    /// Current harmony messages (grows with each MCP loop)
    pub harmony_messages: Vec<HarmonyMessage>,

    /// Current harmony text for dispatch
    pub harmony_text: String,

    /// Original request configuration
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub model: String,
    pub stream: bool,
    pub separate_reasoning: bool,

    /// Dispatch metadata
    pub worker_url: Option<String>,
    pub dispatch_metadata: Option<DispatchMetadata>,

    /// MCP loop tracking
    pub loop_iteration: usize,
    pub max_loops: usize,
    pub accumulated_responses: Vec<ResponsesResponse>,

    /// Timestamps
    pub created: u64,
    pub last_updated: u64,
}

impl RequestContext {
    /// Create new context from chat request
    pub fn from_chat(
        request: &ChatCompletionRequest,
        harmony_messages: HarmonyMessages,
    ) -> Self;

    /// Create new context from responses request
    pub fn from_responses(
        request: &ResponsesRequest,
        harmony_messages: HarmonyMessages,
    ) -> Self;

    /// Update context with MCP tool results
    pub fn append_mcp_results(
        &mut self,
        tool_results: Vec<McpToolResult>,
        new_harmony: HarmonyMessages,
    );

    /// Check if MCP loop should continue
    pub fn should_continue_loop(&self) -> bool {
        self.loop_iteration < self.max_loops
    }
}
```

### 5. DispatchCoordinator

**Purpose**: Manage gRPC dispatch and stream collection

**Structure**:
```rust
pub struct DispatchCoordinator {
    grpc_client_manager: Arc<GrpcClientManager>,
}

pub enum ExecutionResult {
    Single {
        stream: GrpcResponseStream,
    },
    Dual {
        prefill: GrpcResponseStream,
        decode: Box<GrpcResponseStream>,
    },
}

impl DispatchCoordinator {
    /// Execute request with appropriate dispatch mode
    pub async fn execute(
        &self,
        worker_url: &str,
        request: proto::GenerateRequest,
        use_dual_dispatch: bool,
    ) -> Result<ExecutionResult, Response>;

    /// Collect responses from execution result
    pub async fn collect_responses(
        execution_result: ExecutionResult,
        request_logprobs: bool,
    ) -> Result<Vec<proto::GenerateComplete>, Response>;
}
```

**Dispatch Modes**:
1. **Single**: One gRPC stream (decode only)
2. **Dual**: Two gRPC streams (prefill for input_logprobs, decode for output)

### 6. McpLoopOrchestrator

**Purpose**: Orchestrate MCP tool loop for Responses API

**Structure**:
```rust
pub struct McpLoopOrchestrator {
    mcp_manager: Arc<McpManager>,
    harmony_builder: Arc<HarmonyRequestBuilder>,
}

pub struct McpLoopResult {
    pub final_response: ResponsesResponse,
    pub all_responses: Vec<ResponsesResponse>,
    pub total_iterations: usize,
}

impl McpLoopOrchestrator {
    /// Execute MCP loop until completion
    pub async fn execute_loop(
        &self,
        initial_response: ResponsesResponse,
        context: &mut RequestContext,
        pipeline: &HarmonyPipeline,
    ) -> Result<McpLoopResult, Response>;

    /// Detect if response contains MCP tool calls
    fn has_mcp_tools(response: &ResponsesResponse) -> bool;

    /// Execute MCP tool calls and collect results
    async fn execute_mcp_tools(
        &self,
        tool_calls: &[ToolCall],
    ) -> Result<Vec<McpToolResult>, Response>;
}
```

**MCP Loop Algorithm**:
```
loop {
    // 1. Check response for MCP tool calls
    if !has_mcp_tools(response) {
        break;  // No MCP tools, exit loop
    }

    // 2. Execute MCP tools
    tool_results = execute_mcp_tools(response.tool_calls);

    // 3. Update context with results
    new_harmony = harmony_builder.append_mcp_results(
        context.harmony_messages,
        tool_results
    );
    context.append_mcp_results(tool_results, new_harmony);

    // 4. Check loop limit
    if !context.should_continue_loop() {
        break;
    }

    // 5. Build new request via harmony
    // (Back to step 2 of Responses API flow)
    response = pipeline.execute_one_iteration(context).await?;

    // 6. Accumulate response
    context.accumulated_responses.push(response);
}
```

### 7. StreamingProcessor

**Purpose**: Handle SSE streaming for both chat and responses modes

**Structure** (reuse existing):
```rust
pub struct StreamingProcessor {
    tokenizer: Arc<dyn Tokenizer>,
    tool_parser_factory: ToolParserFactory,
    reasoning_parser_factory: ReasoningParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
}

// Existing methods remain
impl StreamingProcessor {
    pub fn process_streaming_response(...) -> Response;
    pub async fn process_streaming_chunks(...) -> Result<(), String>;
    pub async fn process_dual_streaming_chunks(...) -> Result<(), String>;
    pub fn process_streaming_generate(...) -> Response;
}
```

**Enhancement for Harmony**:
- Add method to stream MCP loop iterations
- Support progressive SSE events for tool execution
- Maintain compatibility with existing chat streaming

### 8. ResponseProcessor

**Purpose**: Handle non-streaming response assembly

**Structure** (reuse existing):
```rust
pub struct ResponseProcessor {
    pub tokenizer: Arc<dyn Tokenizer>,
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    pub configured_tool_parser: Option<String>,
    pub configured_reasoning_parser: Option<String>,
}

// Existing methods remain
impl ResponseProcessor {
    pub async fn process_single_choice(...) -> Result<ChatChoice, String>;
    pub async fn process_non_streaming_chat_response(...) -> Result<ChatCompletionResponse, Response>;
    pub async fn parse_tool_calls(...) -> (Option<Vec<ToolCall>>, String);
    pub async fn process_non_streaming_generate_response(...) -> Result<Vec<GenerateResponse>, Response>;
}
```

**Enhancement for Harmony**:
- Add method to assemble multi-iteration MCP responses
- Support responses API response format
- Maintain compatibility with existing chat processing

---

## Implementation Strategy

### Phase 1: Core Components (Week 1-2)

**Deliverables**:
1. `HarmonyRequestBuilder` implementation
   - Chat request → harmony messages
   - Responses request → harmony messages
   - Chat template application
   - MCP result appending

2. `HarmonyResponseParser` implementation
   - Single response parsing
   - Streaming chunk parsing
   - Tool/reasoning extraction via harmony channels

3. `RequestContext` implementation
   - State tracking structure
   - Context update methods
   - Loop iteration management

**Testing**:
- Unit tests for harmony message building
- Unit tests for response parsing
- Integration tests with mock gRPC responses

### Phase 2: Pipeline Integration (Week 2-3)

**Deliverables**:
1. `HarmonyPipeline` implementation
   - Chat completion flow (7 steps)
   - Worker selection integration
   - Dispatch coordination
   - Response processing

2. `DispatchCoordinator` implementation
   - Single/dual dispatch routing
   - Stream collection
   - Error handling

3. Integration with existing `ResponseProcessor` and `StreamingProcessor`
   - Adapter methods for harmony responses
   - Compatibility layer

**Testing**:
- End-to-end tests for chat completion flow
- Tests for single and dual dispatch modes
- Tests for streaming and non-streaming modes

### Phase 3: MCP Loop (Week 3-4)

**Deliverables**:
1. `McpLoopOrchestrator` implementation
   - MCP tool detection
   - Tool execution via McpManager
   - Loop iteration logic
   - Result accumulation

2. Responses API flow (8 steps)
   - History loading
   - MCP loop integration
   - Response persistence

3. Storage integration
   - Response storage
   - Conversation storage
   - Conversation item storage

**Testing**:
- Tests for MCP tool detection
- Tests for single-iteration MCP loop
- Tests for multi-iteration MCP loop
- Tests for loop limits and termination

### Phase 4: Router Integration (Week 4-5)

**Deliverables**:
1. Update `GrpcRouter` to use `HarmonyPipeline`
   - Replace existing pipeline for chat
   - Add responses API route
   - Maintain backward compatibility

2. Migration path
   - Feature flag for harmony vs legacy
   - A/B testing infrastructure
   - Rollback mechanism

3. Documentation
   - API documentation
   - Architecture diagrams
   - Migration guide

**Testing**:
- Full integration tests
- Performance benchmarks
- Load testing
- Backward compatibility tests

### Phase 5: Optimization (Week 5-6)

**Deliverables**:
1. Performance optimization
   - Reduce harmony building overhead
   - Optimize response parsing
   - Cache chat templates

2. Monitoring and observability
   - Add metrics for harmony pipeline
   - Add tracing for MCP loops
   - Add logging for debugging

3. Production readiness
   - Error handling improvements
   - Retry logic for MCP calls
   - Circuit breakers for failures

**Testing**:
- Performance regression tests
- Chaos testing for MCP failures
- Long-running stability tests

---

## Key Design Decisions

### 1. Shared Components

**Decision**: Reuse existing `ResponseProcessor` and `StreamingProcessor` with adapters

**Rationale**:
- Eliminates ~1,800 lines of duplicate code
- Maintains consistency with existing chat completion behavior
- Reduces testing burden
- Simplifies maintenance

**Trade-off**: Need adapter layer for harmony-specific features

### 2. RequestContext State Management

**Decision**: Mutable RequestContext passed through MCP loop iterations

**Rationale**:
- Simplifies state tracking across iterations
- Natural fit for Rust's ownership model
- Easy to add new fields for future features

**Trade-off**: Must be careful about cloning and ownership

### 3. Harmony Integration Points

**Decision**: Harmony used for request building and response parsing, NOT for gRPC dispatch

**Rationale**:
- Clear separation of concerns
- gRPC uses standard proto::GenerateRequest
- Harmony focuses on chat template and message formatting
- Easier to debug and test

**Trade-off**: Need conversion layer between harmony and gRPC formats

### 4. MCP Loop Placement

**Decision**: MCP loop orchestrated at pipeline level, not in router

**Rationale**:
- Keeps router thin and focused on routing
- Pipeline owns the full request lifecycle
- Easier to test MCP loop in isolation
- Better separation of concerns

**Trade-off**: Pipeline becomes more complex

### 5. Streaming Mode for MCP Loops

**Decision**: Support streaming for each MCP iteration

**Rationale**:
- Better user experience (progressive feedback)
- Matches VLLM behavior
- Allows cancellation mid-iteration

**Trade-off**: More complex SSE event sequencing

---

## Migration Strategy

### Backward Compatibility

1. **Feature Flag**:
   - `ENABLE_HARMONY_PIPELINE` environment variable
   - Default: `false` (use legacy pipeline)
   - Gradual rollout per model or per request

2. **Fallback Mechanism**:
   - If harmony pipeline fails, fall back to legacy
   - Log failures for monitoring
   - Gradual confidence building

3. **A/B Testing**:
   - Shadow mode: run both pipelines, compare results
   - Metrics: latency, correctness, error rates
   - Gradual traffic shifting

### Rollout Plan

**Week 1-2**: Internal testing
- Deploy to staging environment
- Run A/B tests with synthetic traffic
- Validate correctness and performance

**Week 3-4**: Canary deployment
- 5% of production traffic
- Monitor error rates and latency
- Roll back if issues detected

**Week 5-6**: Gradual rollout
- Increase to 25%, 50%, 75%
- Monitor metrics at each stage
- Address any issues discovered

**Week 7+**: Full rollout
- 100% of traffic on harmony pipeline
- Deprecate legacy pipeline
- Remove feature flags

---

## Appendix

### A. Comparison with VLLM Implementation

**VLLM Strengths**:
- Clean separation between request building and execution
- Well-defined harmony message format
- Clear MCP loop boundaries

**VLLM Weaknesses**:
- Tightly coupled to Python backend
- Limited streaming support in MCP loops
- No dual dispatch mode

**Our Improvements**:
- Full streaming support in MCP loops
- Support for dual dispatch (prefill + decode)
- Rust performance and safety guarantees
- Better error handling and recovery

### B. Alternative Designs Considered

**Alternative 1**: Separate pipelines for chat and responses
- **Pros**: Simpler individual components
- **Cons**: Code duplication, maintenance burden
- **Decision**: Rejected in favor of unified pipeline

**Alternative 2**: Harmony at gRPC dispatch level
- **Pros**: More consistent abstraction
- **Cons**: Breaking change to gRPC protocol, complexity
- **Decision**: Rejected, keep gRPC protocol stable

**Alternative 3**: MCP loop in router
- **Pros**: Simpler pipeline
- **Cons**: Router becomes too complex, harder to test
- **Decision**: Rejected, keep router thin

### C. Open Questions

1. **Chat Template Caching**: Should we cache compiled chat templates?
   - **Impact**: Performance optimization
   - **Decision**: Phase 5 optimization

2. **MCP Loop Limits**: What are the right defaults?
   - **Current**: max_loops = 10
   - **Decision**: Make configurable per request

3. **Error Recovery**: How to handle partial MCP failures?
   - **Options**: Fail fast vs. best effort
   - **Decision**: TBD based on real-world usage

4. **Streaming Event Format**: What SSE events for MCP iterations?
   - **Options**: New event types vs. reuse existing
   - **Decision**: Design in Phase 3

---

## Conclusion

This design provides a unified harmony pipeline that:
- ✅ Supports both Chat Completion and Responses API
- ✅ Integrates harmony for request building and response parsing
- ✅ Orchestrates MCP tool loops for Responses API
- ✅ Reuses existing components to minimize duplication
- ✅ Supports streaming and non-streaming modes
- ✅ Supports single and dual dispatch modes
- ✅ Provides clear migration path from legacy implementation

The phased implementation plan allows for incremental development and testing while maintaining backward compatibility. The design learns from VLLM's approach while improving on streaming support, dual dispatch, and error handling.

**Next Steps**:
1. Review and approve design document
2. Create implementation tickets for Phase 1
3. Set up testing infrastructure
4. Begin implementation of core components
