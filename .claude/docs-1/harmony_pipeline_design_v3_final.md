# GPT-OSS Harmony Unified Pipeline Design - Final

**Document Version**: 3.0 (Final)
**Date**: 2025-10-26
**Status**: Implementation Ready

## Executive Summary

This document describes the integration of Harmony support for GPT-OSS models into the existing gRPC router pipeline. The design **extends the existing 7-stage pipeline** with conditional Harmony branches, leveraging `openai_harmony::StreamableParser` for token-level parsing.

**Key Design Principles**:
- ✅ Extend existing pipeline stages (no separate pipeline)
- ✅ Token-based approach (`input_ids` → `output_ids`)
- ✅ Harmony channels (`analysis`/`commentary`/`final`)
- ✅ Worker selection post-build using `selection_text`
- ✅ MCP loop for Responses API (existing infrastructure)
- ✅ Zero regression for non-Harmony models

---

## Objectives

### Goals
1. One pipeline supporting both Harmony (GPT-OSS) and non-Harmony models
2. Build requests via Harmony encoding (messages → token_ids) before worker selection
3. Parse outputs via Harmony channels (analysis/commentary/final), not text parsers
4. Integrate MCP tool loop for Responses API
5. Preserve OpenAI tool semantics for Chat Completions
6. Zero regression for non-Harmony models

### Non-Goals
- Replacing existing non-Harmony chat pipeline
- Introducing Harmony semantics to all models
- Changing storage schemas for conversations/responses
- Creating a separate "HarmonyPipeline"

---

## Architecture Overview

### Existing Pipeline (7 Stages)

```
┌──────────────────────────────────────────────────────────────┐
│                    RequestPipeline                            │
│                                                                │
│  Stage 1: PreparationStage                                    │
│    ├─ Filter tools, process messages, apply chat template     │
│    ├─ Tokenize → token_ids                                    │
│    ├─ Build tool constraints                                  │
│    └─ Create StopSequenceDecoder                              │
│                                                                │
│  Stage 2: WorkerSelectionStage                                │
│    └─ Select worker(s) using policy + text                    │
│                                                                │
│  Stage 3: ClientAcquisitionStage                              │
│    └─ Get gRPC clients from workers                           │
│                                                                │
│  Stage 4: RequestBuildingStage                                │
│    ├─ Build proto::GenerateRequest                            │
│    └─ Inject PD metadata if needed                            │
│                                                                │
│  Stage 5: DispatchMetadataStage                               │
│    └─ Prepare metadata (request_id, model, created, etc.)     │
│                                                                │
│  Stage 6: RequestExecutionStage                               │
│    ├─ Execute gRPC request(s)                                 │
│    └─ Mode: Single or DualDispatch                            │
│                                                                │
│  Stage 7: ResponseProcessingStage                             │
│    ├─ Streaming: StreamingProcessor → SSE response            │
│    └─ Non-streaming: ResponseProcessor → final response       │
└──────────────────────────────────────────────────────────────┘
```

### Harmony Integration (Conditional Branches)

```
Stage 1: Preparation
  if harmony_mode:
    ├─ HarmonyBuilder.build() → input_ids + selection_text
    ├─ Add Harmony stop token ids (<|return|>, <|call|>)
    └─ Skip StopSequenceDecoder (Harmony has own parser)
  else:
    └─ Existing logic (chat template + tokenization)

Stage 2: Worker Selection
  ├─ Use selection_text for Harmony (concise snippet)
  └─ Use original_text for non-Harmony

Stage 4: Request Building
  if harmony_mode:
    ├─ build_plain_generate_request(input_ids)
    └─ Inject Harmony stop token ids to sampling_params
  else:
    └─ Existing build_generate_request() logic

Stage 7: Response Processing
  if harmony_mode:
    ├─ HarmonyParserAdapter.parse(output_ids)
    ├─ Map channels → response format
    └─ Reject if logprobs requested
  else:
    └─ Existing tool/reasoning parsers
```

---

## Core Components

### 1. HarmonyDetector

**Location**: `src/routers/grpc/harmony/detector.rs`

```rust
pub struct HarmonyDetector;

impl HarmonyDetector {
    /// Detect if model uses Harmony based on model name/id
    pub fn is_harmony_model(model_name: &str) -> bool {
        // Simple rule-based matching (configurable in future)
        model_name.contains("gpt-oss")
            || model_name.contains("gpt-4o")
            || model_name.contains("gpt-4.5")
            || model_name.starts_with("gpt-5")
    }
}
```

### 2. HarmonyBuilder

**Location**: `src/routers/grpc/harmony/builder.rs`

```rust
pub struct HarmonyBuilder {
    // Harmony encoding provider (TBD: crate feature vs runtime dependency)
}

pub struct HarmonyBuildOutput {
    /// Token IDs for gRPC generation
    pub input_ids: Vec<u32>,

    /// Harmony stop token IDs for assistant actions
    pub stop_token_ids: Vec<u32>,

    /// Concise text snippet for worker selection (e.g., last user message)
    pub selection_text: String,

    /// Full Harmony messages for MCP loop continuation
    pub harmony_messages: Vec<HarmonyMessage>,
}

pub struct HarmonyMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

impl HarmonyBuilder {
    /// Build from ChatCompletionRequest
    pub fn build_from_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<HarmonyBuildOutput, String>;

    /// Build from ResponsesRequest + history
    ///
    /// Important: Handles conversation splicing:
    /// - New conversation: include system (reasoning effort), developer (tools) if custom tools present
    /// - Continuing: splice previous messages; drop intermediate analysis messages (vLLM logic)
    pub fn build_from_responses(
        &self,
        request: &ResponsesRequest,
        history: Vec<ResponsesResponse>,
    ) -> Result<HarmonyBuildOutput, String>;

    /// Append MCP tool results and rebuild
    pub fn append_tool_results(
        &self,
        messages: Vec<HarmonyMessage>,
        tool_results: Vec<McpToolResult>,
    ) -> Result<HarmonyBuildOutput, String>;

    /// Get Harmony assistant action stop tokens
    fn get_stop_token_ids(&self) -> Vec<u32> {
        // <|return|>, <|call|>, etc.
        vec![/* token IDs for Harmony control tokens */]
    }

    /// Extract selection text (last user message text)
    fn extract_selection_text(messages: &[HarmonyMessage]) -> String;
}
```

**Key Operations**:
1. Convert messages → Harmony message format
2. Handle conversation continuation vs new conversation
3. Drop intermediate analysis messages when continuing (align with vLLM)
4. Encode via Harmony encoder → `input_ids`
5. Extract concise `selection_text` (last user message)
6. Add Harmony stop tokens for assistant actions

### 3. HarmonyParserAdapter

**Location**: `src/routers/grpc/harmony/parser.rs`

```rust
use openai_harmony::StreamableParser;

pub struct HarmonyParserAdapter {
    parser: StreamableParser,
}

pub struct HarmonyChannelOutput {
    /// Analysis channel → reasoning content
    pub analysis: Option<String>,

    /// Commentary channel → tool calls
    pub commentary: Option<Vec<ToolCall>>,

    /// Final channel → content text
    pub final_text: String,

    /// Finish reason derived from Harmony state
    pub finish_reason: String,

    /// Matched stop information
    pub matched_stop: Option<serde_json::Value>,
}

impl HarmonyParserAdapter {
    /// Create new parser instance
    pub fn new() -> Self;

    /// Parse complete response (non-streaming)
    pub fn parse_complete(
        &mut self,
        output_ids: &[u32],
    ) -> Result<HarmonyChannelOutput, String>;

    /// Parse streaming chunk
    pub fn parse_chunk(
        &mut self,
        chunk_ids: &[u32],
    ) -> Result<Option<HarmonyChannelDelta>, String>;

    /// Finalize and get complete output
    pub fn finalize(&mut self) -> Result<HarmonyChannelOutput, String>;

    /// Route parsed message to ResponseOutputItem (for Responses API)
    pub fn route_to_response_item(
        parsed_message: &HarmonyParsedMessage,
    ) -> Vec<ResponseOutputItem>;
}

pub struct HarmonyChannelDelta {
    pub analysis_delta: Option<String>,
    pub commentary_delta: Option<String>,  // Partial tool call JSON
    pub final_delta: Option<String>,
}

pub struct HarmonyParsedMessage {
    pub channel: HarmonyChannel,
    pub content: String,
}

pub enum HarmonyChannel {
    Analysis,
    Commentary,
    Final,
}
```

**Channel Routing**:
- **analysis** → reasoning content
- **commentary** + `functions.*` → FunctionToolCall (arguments already JSON)
- **commentary** + `python`/`browser.*`/`container.*` → reasoning output items
- **final** → text content

### 4. Modified PreparationOutput

**Extension to** `src/routers/grpc/context.rs`:

```rust
pub struct PreparationOutput {
    pub original_text: Option<String>,
    pub token_ids: Vec<u32>,
    pub processed_messages: Option<ProcessedMessages>,
    pub tool_constraints: Option<(String, String)>,
    pub filtered_request: Option<ChatCompletionRequest>,

    // NEW: Harmony-specific fields
    pub harmony_mode: bool,
    pub selection_text: Option<String>,  // Concise text for worker selection
    pub harmony_messages: Option<Vec<HarmonyMessage>>,  // For MCP loop
    pub harmony_stop_ids: Option<Vec<u32>>,  // Harmony control token IDs
}
```

### 5. Modified ResponseState

**Extension to** `src/routers/grpc/context.rs`:

```rust
pub struct ResponseState {
    pub stop_decoder: Option<StopSequenceDecoder>,
    pub execution_result: Option<ExecutionResult>,
    pub final_response: Option<FinalResponse>,

    // NEW: Harmony-specific parser state
    pub harmony_parser: Option<HarmonyParserAdapter>,  // For non-streaming
    pub harmony_parser_per_index: Option<HashMap<usize, HarmonyParserAdapter>>,  // For streaming
}
```

---

## Detailed Stage Integration

### Stage 1: Preparation (MODIFIED)

**File**: `src/routers/grpc/pipeline.rs`

```rust
async fn prepare_chat(
    &self,
    ctx: &mut RequestContext,
    request: &ChatCompletionRequest,
) -> Result<(), Response> {
    // Detect Harmony mode
    let is_harmony = HarmonyDetector::is_harmony_model(&request.model);

    if is_harmony {
        // HARMONY PATH

        // Validate - reject logprobs
        if request.logprobs {
            return Err(utils::bad_request_error(
                "logprobs are not supported for Harmony models"
            ));
        }

        // Build via Harmony
        let harmony_builder = HarmonyBuilder::new();
        let build_output = harmony_builder
            .build_from_chat(request)
            .map_err(|e| utils::bad_request_error(format!("Harmony build failed: {}", e)))?;

        // Store results
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,
            token_ids: build_output.input_ids,
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            harmony_mode: true,
            selection_text: Some(build_output.selection_text),
            harmony_messages: Some(build_output.harmony_messages),
            harmony_stop_ids: Some(build_output.stop_token_ids),
        });

        // No StopSequenceDecoder for Harmony

    } else {
        // EXISTING NON-HARMONY PATH (unchanged)
        // ... existing implementation ...
    }

    Ok(())
}

async fn prepare_responses(
    &self,
    ctx: &mut RequestContext,
    request: &ResponsesRequest,
    history: Vec<ResponsesResponse>,
) -> Result<(), Response> {
    let is_harmony = HarmonyDetector::is_harmony_model(&request.model);

    if is_harmony {
        // HARMONY PATH

        // Validate - reject logprobs
        if request.logprobs.unwrap_or(false) {
            return Err(utils::bad_request_error(
                "logprobs are not supported for Harmony models"
            ));
        }

        // Validate - reject tool_choice != auto
        if !matches!(request.tool_choice, None | Some(ToolChoice::Value(ToolChoiceValue::Auto))) {
            return Err(utils::bad_request_error(
                "Harmony models only support tool_choice='auto' for Responses API"
            ));
        }

        // Build via Harmony with history
        let harmony_builder = HarmonyBuilder::new();
        let build_output = harmony_builder
            .build_from_responses(request, history)
            .map_err(|e| utils::bad_request_error(format!("Harmony build failed: {}", e)))?;

        // Store results
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,
            token_ids: build_output.input_ids,
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            harmony_mode: true,
            selection_text: Some(build_output.selection_text),
            harmony_messages: Some(build_output.harmony_messages),
            harmony_stop_ids: Some(build_output.stop_token_ids),
        });

    } else {
        // NON-HARMONY PATH
        // Convert to ChatCompletionRequest and use existing logic
        // ... implementation ...
    }

    Ok(())
}
```

### Stage 2: Worker Selection (MODIFIED)

**File**: `src/routers/grpc/pipeline.rs`

```rust
fn select_single_worker(
    &self,
    model_id: Option<&str>,
    prep: &PreparationOutput,
) -> Option<Arc<dyn Worker>> {
    // Get workers
    let workers = self.worker_registry.get_workers_filtered(
        model_id,
        Some(WorkerType::Regular),
        Some(ConnectionMode::Grpc { port: None }),
        false,
    );

    let available: Vec<Arc<dyn Worker>> = workers
        .iter()
        .filter(|w| w.is_available())
        .cloned()
        .collect();

    if available.is_empty() {
        return None;
    }

    let policy = match model_id {
        Some(model) => self.policy_registry.get_policy_or_default(model),
        None => self.policy_registry.get_default_policy(),
    };

    // Use selection_text for Harmony, original_text for non-Harmony
    let text_for_policy = if prep.harmony_mode {
        prep.selection_text.as_deref()
    } else {
        prep.original_text.as_deref()
    };

    let idx = policy.select_worker(&available, text_for_policy)?;
    Some(available[idx].clone())
}
```

### Stage 4: Request Building (MODIFIED)

**File**: `src/routers/grpc/pipeline.rs`

```rust
async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
    let prep = ctx.state.preparation.as_ref()
        .ok_or_else(|| utils::internal_error_static("Preparation not completed"))?;

    let clients = ctx.state.clients.as_ref()
        .ok_or_else(|| utils::internal_error_static("Client acquisition not completed"))?;

    let builder_client = match clients {
        ClientSelection::Single { client } => client,
        ClientSelection::Dual { prefill, .. } => prefill,
    };

    let mut proto_request = if prep.harmony_mode {
        // HARMONY PATH: Use build_plain_generate_request

        let request_id = match &ctx.input.request_type {
            RequestType::Chat(_) => format!("chatcmpl-{}", Uuid::new_v4()),
            RequestType::Responses(_) => format!("resp-{}", Uuid::new_v4()),
            RequestType::Generate(req) => req.rid.clone()
                .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4())),
        };

        // Build plain generate request with input_ids
        let sampling_params = match &ctx.input.request_type {
            RequestType::Chat(req) => extract_sampling_params_from_chat(req),
            RequestType::Responses(req) => extract_sampling_params_from_responses(req),
            RequestType::Generate(req) => req.sampling_params.clone(),
        };

        let mut gen_req = builder_client
            .build_plain_generate_request_from_ids(
                request_id,
                prep.token_ids.clone(),
                sampling_params,
            )
            .map_err(|e| utils::bad_request_error(format!("Invalid request: {}", e)))?;

        // Inject Harmony stop token IDs
        if let Some(harmony_stops) = &prep.harmony_stop_ids {
            if let Some(params) = gen_req.sampling_params.as_mut() {
                params.stop_token_ids.get_or_insert_with(Vec::new)
                    .extend_from_slice(harmony_stops);
            }
        }

        gen_req

    } else {
        // EXISTING NON-HARMONY PATH (unchanged)
        // ... existing implementation ...
    };

    // Inject PD metadata if needed (same for both paths)
    if self.inject_pd_metadata {
        if let WorkerSelection::Dual { prefill, .. } = ctx.state.workers.as_ref().unwrap() {
            self.inject_bootstrap_metadata(&mut proto_request, prefill);
        }
    }

    ctx.state.proto_request = Some(proto_request);
    Ok(None)
}
```

### Stage 7: Response Processing (MODIFIED)

**File**: `src/routers/grpc/processing.rs`

```rust
impl ResponseProcessor {
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
        stop_decoder: &mut Option<StopSequenceDecoder>,
        request_logprobs: bool,
        harmony_mode: bool,
    ) -> Result<ChatCompletionResponse, Response> {
        let all_responses =
            Self::collect_and_merge_responses(execution_result, request_logprobs).await?;

        if harmony_mode {
            self.process_harmony_chat(all_responses, chat_request, dispatch).await
        } else {
            self.process_legacy_chat(
                all_responses,
                chat_request,
                dispatch,
                stop_decoder.as_mut().unwrap(),
                request_logprobs,
            ).await
        }
    }

    async fn process_harmony_chat(
        &self,
        all_responses: Vec<proto::GenerateComplete>,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
    ) -> Result<ChatCompletionResponse, Response> {
        let mut choices = Vec::new();
        let mut total_reasoning_tokens = 0u32;

        for (index, complete) in all_responses.iter().enumerate() {
            // Parse via Harmony
            let mut parser = HarmonyParserAdapter::new();
            let channel_output = parser
                .parse_complete(&complete.output_ids)
                .map_err(|e| utils::internal_error_message(format!("Harmony parsing failed: {}", e)))?;

            // Count reasoning tokens (analysis + commentary channels)
            total_reasoning_tokens += channel_output.reasoning_token_count();

            // Build ChatChoice from Harmony channels
            let chat_message = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: if channel_output.final_text.is_empty() {
                    None
                } else {
                    Some(channel_output.final_text)
                },
                tool_calls: channel_output.commentary,
                reasoning_content: channel_output.analysis,
            };

            // Override finish reason if tool calls present
            let finish_reason = if chat_message.tool_calls.is_some() {
                "tool_calls".to_string()
            } else {
                channel_output.finish_reason
            };

            let choice = ChatChoice {
                index: index as u32,
                message: chat_message,
                logprobs: None,  // Not supported for Harmony
                finish_reason: Some(finish_reason),
                matched_stop: channel_output.matched_stop,
                hidden_states: None,
            };

            choices.push(choice);
        }

        // Build usage with reasoning_tokens
        let total_prompt_tokens: u32 = all_responses.iter().map(|r| r.prompt_tokens as u32).sum();
        let total_completion_tokens: u32 = all_responses.iter().map(|r| r.completion_tokens as u32).sum();

        let usage = Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: total_reasoning_tokens,
                ..Default::default()
            }),
        };

        Ok(ChatCompletionResponse {
            id: dispatch.request_id,
            object: "chat.completion".to_string(),
            created: dispatch.created,
            model: dispatch.model,
            choices,
            usage: Some(usage),
            system_fingerprint: dispatch.weight_version,
        })
    }
}
```

**Similar modifications for StreamingProcessor** in `src/routers/grpc/streaming.rs`:
- Maintain per-index `HarmonyParserAdapter`
- Parse chunks → emit SSE deltas with reasoning_tokens tracking

---

## Responses API MCP Loop Integration

### MCP Loop Flow

**File**: `src/routers/grpc/responses/mod.rs`

```rust
pub async fn route_responses(
    pipeline: &RequestPipeline,
    request: Arc<ResponsesRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: Arc<SharedComponents>,
    response_storage: SharedResponseStorage,
    conversation_storage: SharedConversationStorage,
    conversation_item_storage: SharedConversationItemStorage,
    mcp_manager: Arc<McpManager>,
    background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
) -> Response {
    let is_harmony = HarmonyDetector::is_harmony_model(&request.model);
    let max_iterations = 10;
    let mut response_history = Vec::new();

    // MCP Loop
    for iteration in 0..max_iterations {
        // Build request (Harmony or legacy)
        let chat_request = if is_harmony {
            // Use HarmonyBuilder to rebuild with history
            build_chat_from_harmony_history(&request, &response_history)?
        } else {
            // Existing non-Harmony path
            build_chat_from_legacy_history(&request, &response_history)?
        };

        // Execute via pipeline
        let chat_response = pipeline.execute_chat_for_responses(
            Arc::new(chat_request),
            headers.clone(),
            model_id.clone(),
            components.clone(),
            Some(response_id.clone()),
            Some(background_tasks.clone()),
        ).await?;

        // Check for tool calls
        let has_tool_calls = chat_response.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|t| !t.is_empty())
            .unwrap_or(false);

        // Convert to ResponsesResponse
        let responses_response = if is_harmony {
            convert_harmony_chat_to_responses(chat_response, &request)?
        } else {
            convert_legacy_chat_to_responses(chat_response, &request)?
        };

        // Persist
        response_storage.store(&response_id, &responses_response).await?;
        response_history.push(responses_response.clone());

        // Check MCP loop termination
        if !has_tool_calls {
            return Ok(axum::Json(responses_response).into_response());
        }

        // Execute MCP tools
        let tool_results = execute_mcp_tools(
            responses_response.output.as_ref().unwrap(),
            &mcp_manager,
        ).await?;

        // For Harmony: append tool results and continue
        if is_harmony {
            // Tool results will be incorporated in next build_from_harmony_history call
            append_harmony_tool_results(&mut response_history, tool_results);
        } else {
            append_legacy_tool_results(&mut response_history, tool_results);
        }
    }

    // Max iterations reached
    Err(utils::bad_request_error("Maximum MCP iterations reached"))
}

fn build_chat_from_harmony_history(
    request: &ResponsesRequest,
    history: &[ResponsesResponse],
) -> Result<ChatCompletionRequest, String> {
    // Extract all messages from history
    let mut messages = Vec::new();

    for resp in history {
        // Convert ResponsesResponse → Harmony messages
        // Important: Drop intermediate analysis messages (vLLM logic)
        messages.extend(extract_harmony_messages_from_response(resp, /* drop_analysis */ true));
    }

    // Append new input from request
    messages.push(HarmonyMessage {
        role: "user".to_string(),
        content: request.input.clone(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    // Build ChatCompletionRequest
    // Note: This will be fed through PreparationStage which will
    // detect Harmony and rebuild via HarmonyBuilder.build_from_chat()

    Ok(ChatCompletionRequest {
        model: request.model.clone(),
        messages: convert_harmony_to_openai_messages(messages),
        tools: request.tools.clone(),
        tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Auto)),
        // ... other fields ...
    })
}
```

### Harmony Message Building from History

**Key logic** (align with vLLM):

1. **New Conversation**:
   - Include system message with reasoning effort if specified
   - Include developer message with tool descriptions if custom tools present
   - Add user input

2. **Continuing Conversation**:
   - Splice previous messages from history
   - **Drop intermediate analysis messages** (vLLM behavior)
   - Keep final messages and tool results
   - Add new user input

3. **Tool Results**:
   - Append tool messages with results
   - Re-encode entire context

---

## Harmony Channel Mapping

### Chat Completions

| Harmony Channel | Chat Field | Type | Notes |
|----------------|-----------|------|-------|
| `analysis` | `message.reasoning_content` | `Option<String>` | O1-style reasoning |
| `commentary` | `message.tool_calls` | `Option<Vec<ToolCall>>` | Parse JSON → ToolCall array |
| `final` | `message.content` | `Option<String>` | Assistant content |
| (finish state) | `choice.finish_reason` | `String` | `"tool_calls"` if commentary present |

### Responses API

| Harmony Channel | ResponseOutputItem | Notes |
|----------------|-------------------|-------|
| `analysis` | `ResponseOutputItem::Reasoning` | Reasoning content |
| `commentary` + `functions.*` | `ResponseOutputItem::FunctionToolCall` | Function calls (args already JSON) |
| `commentary` + `python` | (reasoning output) | Python execution |
| `commentary` + `browser.*` | `ResponseOutputItem::WebSearch*` | Browser actions |
| `commentary` + `container.*` | (reasoning output) | Container actions |
| `final` | `response.text` | Final text content |

---

## Token Accounting & Metrics

### Harmony Token Tracking

**Track per request**:
- `prompt_tokens`: Total input tokens
- `completion_tokens`: Total output tokens (all channels)
- `reasoning_tokens`: Tokens in `analysis` + `commentary` channels
- `cached_tokens`: From backend (prefill cache hits)

**Computation**:
```rust
struct HarmonyTokenCounts {
    reasoning_tokens: u32,  // analysis + commentary
    final_tokens: u32,      // final channel
}

impl HarmonyParserAdapter {
    fn get_token_counts(&self) -> HarmonyTokenCounts {
        // Track based on channel state during parsing
    }
}
```

**Usage in responses**:
```rust
let usage = Usage {
    prompt_tokens: total_prompt_tokens,
    completion_tokens: total_completion_tokens,
    total_tokens: total_prompt_tokens + total_completion_tokens,
    completion_tokens_details: Some(CompletionTokensDetails {
        reasoning_tokens: harmony_counts.reasoning_tokens,
        ..Default::default()
    }),
};
```

### Observability

**Metrics**:
- `harmony_requests_total{model, status}` - Counter
- `harmony_reasoning_tokens_total{model}` - Counter
- `harmony_parsing_errors_total{model, error_type}` - Counter
- `harmony_mcp_iterations{model, count}` - Histogram

**Logs**:
- Model type, Harmony on/off
- Worker type (single/PD)
- Token counts per channel
- MCP iteration count
- Parsing errors

---

## Validation & Error Handling

### Request Validation

**Harmony-specific constraints**:

1. **Logprobs**: Reject at handler level
   ```rust
   if is_harmony && request.logprobs {
       return Err(bad_request_error("logprobs not supported for Harmony models"));
   }
   ```

2. **Responses tool_choice**: Only `auto` allowed
   ```rust
   if is_harmony && request.tool_choice != Some(ToolChoice::Value(ToolChoiceValue::Auto)) {
       return Err(bad_request_error("Harmony only supports tool_choice='auto' for Responses"));
   }
   ```

3. **Stop tokens**: Ensure Harmony assistant action tokens always included

### Error Handling

**Harmony message rendering failures** → 400 with clear message

**Parser failures during stream**:
- Send SSE error event
- Mark stream as completed
- Terminate gracefully

**MCP execution failures**:
- Log warning
- Add error status to Responses output
- Continue loop unless fatal

**Example**:
```rust
match harmony_builder.build_from_chat(request) {
    Ok(output) => output,
    Err(e) => {
        return Err(utils::bad_request_error(format!(
            "Harmony encoding failed: {}", e
        )));
    }
}
```

---

## Implementation Plan

### Phase 1: Core Components (Week 1)

**Deliverables**:
1. Create `src/routers/grpc/harmony/` module:
   - `detector.rs` - Model detection by id/name
   - `builder.rs` - Harmony encoding (messages → token_ids)
   - `parser.rs` - StreamableParser adapter
   - `types.rs` - Shared types

2. Harmony encoder integration:
   - Decide: crate feature vs runtime dependency
   - Implement encoding functions
   - Add stop token constants

3. Unit tests:
   - Detector logic
   - Builder encoding (new conversation vs continuing)
   - Parser channel mapping

### Phase 2: Pipeline Integration (Week 1-2)

**Deliverables**:
1. Modify `PreparationStage`:
   - Add Harmony branch for Chat
   - Add Harmony branch for Responses (with history)
   - Validate logprobs/tool_choice rejection

2. Modify `WorkerSelectionStage`:
   - Use `selection_text` for Harmony

3. Modify `RequestBuildingStage`:
   - Use `build_plain_generate_request` for Harmony
   - Inject Harmony stop token IDs

4. Extend context types:
   - `PreparationOutput` with Harmony fields
   - `ResponseState` with parser state

5. Integration tests:
   - Non-streaming chat with Harmony
   - Validate token flow

### Phase 3: Response Processing (Week 2)

**Deliverables**:
1. Modify `ResponseProcessor`:
   - Add `process_harmony_chat()`
   - Add `process_harmony_responses()`
   - Token counting

2. Modify `StreamingProcessor`:
   - Per-index parser state
   - SSE delta emission
   - Reasoning tokens tracking

3. Tests:
   - Non-streaming Harmony responses
   - Streaming Harmony responses
   - Tool call detection
   - Reasoning content extraction
   - Token counting accuracy

### Phase 4: MCP Loop Integration (Week 2-3)

**Deliverables**:
1. Modify responses module:
   - Detect Harmony mode
   - Build from history (drop analysis messages)
   - Rebuild via HarmonyBuilder in loop
   - Tool result appending

2. Storage integration:
   - Ensure ResponsesResponse format works

3. Tests:
   - Single MCP iteration
   - Multi-iteration loop
   - Loop termination
   - History splicing
   - Analysis message dropping

### Phase 5: Testing & Rollout (Week 3-4)

**Deliverables**:
1. End-to-end tests:
   - Harmony chat completion
   - Harmony responses with MCP
   - Streaming modes
   - Token accounting

2. Feature flag:
   - Per-model Harmony opt-in
   - Monitoring and metrics

3. Documentation:
   - API docs
   - Harmony semantics guide
   - Migration guide

---

## Open Questions

1. **Harmony Encoder Source**
   - **Option A**: Crate feature (`openai-harmony` crate)
   - **Option B**: Runtime dependency (optional compilation)
   - **Recommendation**: Crate feature for cleaner integration

2. **Responses SSE Format**
   - **Option A**: Direct Harmony responses SSE events (lower overhead)
   - **Option B**: Continue transforming Chat SSE (reuse existing code)
   - **Recommendation**: Phase 1 use Option B, Phase 2 add Option A

3. **Token Accounting Centralization**
   - Should we centralize Harmony token counting across streaming/non-streaming?
   - **Recommendation**: Yes - create shared token counter in HarmonyParserAdapter

4. **Harmony Detection Configuration**
   - Static list vs registry hints from workers?
   - **Recommendation**: Start with static list, add registry support Phase 2

---

## Migration & Rollout

### Feature Flag

**Environment variable**: `HARMONY_ENABLED_MODELS="gpt-oss,gpt-4o,gpt-4.5,gpt-5"`
- Default: empty (Harmony disabled)
- Per-model opt-in for gradual rollout

### Rollout Plan

**Week 1-2**: Internal testing
- Deploy to staging
- Test all Harmony paths
- Validate token counts

**Week 3**: Canary (5% traffic)
- Monitor error rates
- Track reasoning_tokens metrics
- Validate MCP loop behavior

**Week 4**: Gradual increase (25%, 50%, 75%)
- Monitor at each stage
- Address issues

**Week 5**: Full rollout (100%)
- All detected Harmony models
- Deprecation notice for legacy path

### Monitoring

**Metrics to watch**:
- Harmony parsing error rate (target: <0.1%)
- MCP loop iteration distribution
- Reasoning token usage
- Latency impact vs non-Harmony

**Alerts**:
- Parsing error rate >1%
- MCP loop max iterations >5% of requests
- Harmony requests failing validation >5%

---

## Alignment with vLLM

✅ **StreamableParser** is the only parsing mechanism
✅ **Channels** drive all routing (analysis/commentary/final)
✅ **Stop tokens** include Harmony assistant action tokens
✅ **Logprobs** disabled for Harmony
✅ **Responses tool_choice** restricted to `auto`
✅ **Worker selection** post-build using `selection_text`
✅ **History splicing** drops intermediate analysis messages
✅ **Token accounting** tracks reasoning_tokens separately

---

## Conclusion

This final design provides a comprehensive integration of Harmony support into the existing gRPC router pipeline:

✅ **Minimal Changes**: ~500 lines of modifications vs ~2000 new lines
✅ **Token-Based**: Correct use of `input_ids`/`output_ids`
✅ **Harmony Channels**: Direct StreamableParser integration
✅ **MCP Loop**: Reuses existing responses infrastructure
✅ **Zero Regression**: Non-Harmony models unaffected
✅ **Token Accounting**: Proper reasoning_tokens tracking
✅ **vLLM Alignment**: Matches vLLM behavior and constraints

**Next Steps**:
1. Review and approve final design
2. Implement Phase 1 (core components)
3. Set up Harmony encoder integration
4. Begin Stage 1 modifications
