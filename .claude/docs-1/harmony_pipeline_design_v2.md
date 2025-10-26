# Unified Harmony Pipeline Design v2 (GPT-OSS Integration)

**Document Version**: 2.0
**Date**: 2025-10-25
**Status**: Design Proposal - Revised

## Executive Summary

This document describes the integration of Harmony support for GPT-OSS models into the existing gRPC router pipeline. The design focuses on **extending the existing 7-stage pipeline** with conditional Harmony branches, rather than creating a separate pipeline.

**Key Changes from V1**:
- ✅ Integrate into existing pipeline stages (not a separate pipeline)
- ✅ Token-based approach (`input_ids` / `output_ids`, not text)
- ✅ Harmony channels semantics (`analysis`/`commentary`/`final`)
- ✅ Use `openai_harmony::StreamableParser` directly
- ✅ Conditional branching via `harmony_mode` flag
- ✅ Reuse existing ResponseProcessor/StreamingProcessor with Harmony branches
- ✅ MCP loop orchestration in responses module (existing infrastructure)

---

## Goals & Non-Goals

### Goals
- One pipeline supporting both Harmony (GPT-OSS) and non-Harmony models
- Keep existing 7-stage pipeline architecture intact
- Build requests via Harmony encoding (messages → token_ids) before worker selection
- Parse outputs via Harmony channels (analysis/commentary/final), not text parsers
- Integrate MCP tool loop for Responses API
- Zero regression for non-Harmony models

### Non-Goals
- Replacing existing non-Harmony chat pipeline
- Introducing Harmony semantics to all models
- Changing storage schemas for conversations/responses
- Creating a separate "HarmonyPipeline" (integrate into existing pipeline)

---

## Architecture Overview

### Existing Pipeline Architecture (7 Stages)

```
┌──────────────────────────────────────────────────────────────┐
│                    RequestPipeline                            │
│                                                                │
│  Stage 1: PreparationStage                                    │
│    ├─ Filter tools                                            │
│    ├─ Process messages & apply chat template                  │
│    ├─ Tokenize → token_ids                                    │
│    ├─ Build tool constraints                                  │
│    └─ Create StopSequenceDecoder                              │
│                                                                │
│  Stage 2: WorkerSelectionStage                                │
│    ├─ Select worker(s) using policy                           │
│    └─ Mode: Regular or PrefillDecode                          │
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

### Harmony Integration Points

```
┌──────────────────────────────────────────────────────────────┐
│           Harmony Integration (Conditional Branches)          │
│                                                                │
│  Stage 0 (NEW): HarmonyDetectionStage                         │
│    ├─ Detect if model uses Harmony (gpt-oss, gpt-4o, etc.)   │
│    └─ Set ctx.state.harmony_mode = true/false                 │
│                                                                │
│  Stage 1: PreparationStage (MODIFIED)                         │
│    if harmony_mode:                                           │
│      ├─ HarmonyBuilder.build() → input_ids + selection_text   │
│      ├─ Add Harmony stop token ids (<|return|>, <|call|>)     │
│      └─ Skip StopSequenceDecoder (Harmony has own parser)     │
│    else:                                                       │
│      └─ Existing logic (chat template + tokenization)         │
│                                                                │
│  Stage 2: WorkerSelectionStage (MODIFIED)                     │
│    ├─ Use selection_text for Harmony (concise snippet)        │
│    └─ Use original_text for non-Harmony                       │
│                                                                │
│  Stage 4: RequestBuildingStage (MODIFIED)                     │
│    if harmony_mode:                                           │
│      ├─ build_plain_generate_request(input_ids)               │
│      └─ Inject Harmony stop token ids to sampling_params      │
│    else:                                                       │
│      └─ Existing build_generate_request() logic               │
│                                                                │
│  Stage 7: ResponseProcessingStage (MODIFIED)                  │
│    if harmony_mode:                                           │
│      ├─ HarmonyParserAdapter.parse(output_ids)                │
│      ├─ Map channels → response format:                       │
│      │   - analysis → reasoning_content                       │
│      │   - commentary → tool_calls                            │
│      │   - final → content                                    │
│      └─ Reject if logprobs requested (not supported)          │
│    else:                                                       │
│      └─ Existing tool/reasoning parsers                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Comparison: V1 vs V2 Design

| Aspect               | V1 Design (Original)                          | V2 Design (Revised)                                |
|----------------------|-----------------------------------------------|----------------------------------------------------|
| **Architecture**     | Separate `HarmonyPipeline` alongside existing | Extend existing pipeline with conditional branches |
| **Integration**      | New pipeline with new components              | Modify existing stages with `if harmony_mode`      |
| **Input Format**     | Harmony text (string)                         | Token IDs (`input_ids: Vec<u32>`)                  |
| **Output Format**    | Text-based parsing                            | Token-based parsing (`output_ids`)                 |
| **Parsing**          | Custom `HarmonyResponseParser`                | `openai_harmony::StreamableParser` adapter         |
| **Channels**         | Generic tool/reasoning parsers                | Harmony channels: analysis/commentary/final        |
| **Worker Selection** | Full harmony_text                             | Concise `selection_text` snippet                   |
| **Stop Tokens**      | Generic stop decoder                          | Harmony-specific tokens (`<                        |return|>`, `<|call|>`) |
| **Logprobs**         | Attempted conversion                          | Explicitly rejected for Harmony                    |
| **MCP Loop**         | New `McpLoopOrchestrator`                     | Reuse existing responses module loop               |
| **Migration**        | Switchable pipeline                           | Feature flag per stage                             |
| **Code Duplication** | ~2000 lines new code                          | ~500 lines of modifications                        |

---

## Detailed Component Specifications

### 1. HarmonyDetector

**Purpose**: Determine if a model uses Harmony encoding

**Location**: `src/routers/grpc/harmony/detector.rs`

**Structure**:
```rust
pub struct HarmonyDetector;

impl HarmonyDetector {
    /// Detect if model uses Harmony based on model name/id
    pub fn is_harmony_model(model_name: &str) -> bool {
        // Simple rule-based matching
        model_name.contains("gpt-oss")
            || model_name.contains("gpt-4o")
            || model_name.contains("gpt-4.5")
            || model_name.starts_with("gpt-5")
    }
}
```

**Integration**: Called in new Stage 0 (or beginning of Stage 1)

### 2. HarmonyBuilder

**Purpose**: Convert Chat/Responses requests to Harmony token IDs

**Location**: `src/routers/grpc/harmony/builder.rs`

**Structure**:
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
}
```

**Key Operations**:
1. Convert messages → Harmony message format
2. Encode via Harmony encoder → `input_ids`
3. Extract concise `selection_text` (last user message)
4. Add Harmony stop tokens for assistant actions
5. Store full message context for MCP loops

### 3. HarmonyParserAdapter

**Purpose**: Wrap `openai_harmony::StreamableParser` for both streaming and non-streaming

**Location**: `src/routers/grpc/harmony/parser.rs`

**Structure**:
```rust
use openai_harmony::StreamableParser;

pub struct HarmonyParserAdapter {
    // Internal StreamableParser instance per choice/index
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
}

pub struct HarmonyChannelDelta {
    pub analysis_delta: Option<String>,
    pub commentary_delta: Option<String>,  // Partial tool call JSON
    pub final_delta: Option<String>,
}
```

**Parsing Flow**:
1. Feed `output_ids` (tokens) to `StreamableParser`
2. Parser emits channel-specific content
3. Map channels to protocol shapes:
   - **Chat**: `analysis` → `reasoning_content`, `commentary` → `tool_calls`, `final` → `content`
   - **Responses**: `analysis` → `ResponseOutputItem::Reasoning`, `commentary` → tool call items, `final` → text

### 4. Modified PreparationOutput

**Extension to existing struct** in `src/routers/grpc/context.rs`:

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

**Extension to existing struct** in `src/routers/grpc/context.rs`:

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

## Detailed Integration by Stage

### Stage 0: Harmony Detection (NEW)

**File**: `src/routers/grpc/harmony/detector.rs`

```rust
pub struct HarmonyDetectionStage;

#[async_trait]
impl PipelineStage for HarmonyDetectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let model_name = match &ctx.input.request_type {
            RequestType::Chat(req) => &req.model,
            RequestType::Generate(_) => ctx.input.model_id.as_deref().unwrap_or(""),
        };

        let is_harmony = HarmonyDetector::is_harmony_model(model_name);

        // Store in preparation output (will be created in next stage)
        // For now, we can add a field to RequestContext or set in Stage 1

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "HarmonyDetection"
    }
}
```

**Alternative**: Perform detection at the start of Stage 1 (PreparationStage) to avoid adding new stage

### Stage 1: Preparation (MODIFIED)

**File**: `src/routers/grpc/pipeline.rs` - `PreparationStage::prepare_chat()`

**Modifications**:
```rust
async fn prepare_chat(
    &self,
    ctx: &mut RequestContext,
    request: &ChatCompletionRequest,
) -> Result<(), Response> {
    // NEW: Detect Harmony mode
    let is_harmony = HarmonyDetector::is_harmony_model(&request.model);

    if is_harmony {
        // HARMONY PATH

        // Step 1: Validate - reject logprobs
        if request.logprobs {
            return Err(utils::bad_request_error(
                "logprobs are not supported for Harmony models"
            ));
        }

        // Step 2: Build via Harmony
        let harmony_builder = HarmonyBuilder::new();
        let build_output = harmony_builder
            .build_from_chat(request)
            .map_err(|e| utils::bad_request_error(format!("Harmony build failed: {}", e)))?;

        // Step 3: Store results
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,  // Not needed for Harmony
            token_ids: build_output.input_ids,
            processed_messages: None,
            tool_constraints: None,  // Harmony handles tools internally
            filtered_request: None,
            harmony_mode: true,
            selection_text: Some(build_output.selection_text),
            harmony_messages: Some(build_output.harmony_messages),
            harmony_stop_ids: Some(build_output.stop_token_ids),
        });

        // Step 4: NO StopSequenceDecoder for Harmony
        // Harmony uses its own parser

    } else {
        // EXISTING NON-HARMONY PATH (unchanged)

        // Step 1: Filter tools if needed
        let body_ref = utils::filter_tools_for_request(request);

        // Step 2: Process messages and apply chat template
        let processed_messages =
            match utils::process_chat_messages(&body_ref, &*ctx.components.tokenizer) {
                Ok(msgs) => msgs,
                Err(e) => return Err(utils::bad_request_error(e)),
            };

        // Step 3: Tokenize
        let encoding = match ctx.components.tokenizer.encode(&processed_messages.text) {
            Ok(encoding) => encoding,
            Err(e) => return Err(utils::internal_error_message(format!(
                "Tokenization failed: {}", e
            ))),
        };

        let token_ids = encoding.token_ids().to_vec();

        // Step 4: Build tool constraints
        let tool_call_constraint = if let Some(tools) = body_ref.tools.as_ref() {
            utils::generate_tool_constraints(tools, &request.tool_choice, &request.model)
                .map_err(|e| utils::bad_request_error(format!("Invalid tool configuration: {}", e)))?
        } else {
            None
        };

        // Step 5: Create stop sequence decoder
        let stop_decoder = utils::create_stop_decoder(
            &ctx.components.tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
        );

        // Store results
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(processed_messages.text.clone()),
            token_ids,
            processed_messages: Some(processed_messages),
            tool_constraints: tool_call_constraint,
            filtered_request: if matches!(body_ref, Cow::Owned(_)) {
                Some(body_ref.into_owned())
            } else {
                None
            },
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        // Store stop decoder
        ctx.state.response.stop_decoder = Some(stop_decoder);
    }

    Ok(())
}
```

### Stage 2: Worker Selection (MODIFIED)

**File**: `src/routers/grpc/pipeline.rs` - `WorkerSelectionStage`

**Modifications**:
```rust
fn select_single_worker(
    &self,
    model_id: Option<&str>,
    ctx: &RequestContext,  // Pass full context instead of just text
) -> Option<Arc<dyn Worker>> {
    // Get workers for the specified model
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

    // NEW: Use selection_text for Harmony, original_text for non-Harmony
    let prep = ctx.state.preparation.as_ref()?;
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

**File**: `src/routers/grpc/pipeline.rs` - `RequestBuildingStage`

**Modifications**:
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

        match &ctx.input.request_type {
            RequestType::Chat(request) => {
                let request_id = format!("chatcmpl-{}", Uuid::new_v4());

                // Build plain generate request with input_ids
                let mut gen_req = builder_client
                    .build_plain_generate_request_from_ids(
                        request_id,
                        prep.token_ids.clone(),
                        request.sampling_params(),  // Extract sampling params
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
            }
            RequestType::Generate(request) => {
                // Already has input_ids, just pass through with Harmony stops
                let request_id = request.rid.clone()
                    .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

                let mut gen_req = builder_client
                    .build_plain_generate_request(
                        request_id,
                        request,
                        None,  // No text
                        prep.token_ids.clone(),
                    )
                    .map_err(utils::bad_request_error)?;

                // Inject Harmony stop token IDs
                if let Some(harmony_stops) = &prep.harmony_stop_ids {
                    if let Some(params) = gen_req.sampling_params.as_mut() {
                        params.stop_token_ids.get_or_insert_with(Vec::new)
                            .extend_from_slice(harmony_stops);
                    }
                }

                gen_req
            }
        }

    } else {
        // EXISTING NON-HARMONY PATH (unchanged)
        match &ctx.input.request_type {
            RequestType::Chat(request) => {
                let request_id = format!("chatcmpl-{}", Uuid::new_v4());
                let body_ref = prep.filtered_request.as_ref().unwrap_or(request);

                builder_client.build_generate_request(
                    request_id,
                    body_ref,
                    prep.processed_messages.as_ref().unwrap().text.clone(),
                    prep.token_ids.clone(),
                    prep.processed_messages.as_ref().unwrap().multimodal_inputs.clone(),
                    prep.tool_constraints.clone(),
                )
                .map_err(|e| utils::bad_request_error(format!("Invalid request parameters: {}", e)))?
            }
            RequestType::Generate(request) => {
                let request_id = request.rid.clone()
                    .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

                builder_client.build_plain_generate_request(
                    request_id,
                    request,
                    prep.original_text.clone(),
                    prep.token_ids.clone(),
                )
                .map_err(utils::bad_request_error)?
            }
        }
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

**File**: `src/routers/grpc/processing.rs` - `ResponseProcessor`

**Modifications**:
```rust
impl ResponseProcessor {
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
        stop_decoder: &mut Option<StopSequenceDecoder>,  // Changed to Option
        request_logprobs: bool,
        harmony_mode: bool,  // NEW parameter
    ) -> Result<ChatCompletionResponse, Response> {
        // Collect all responses
        let all_responses =
            Self::collect_and_merge_responses(execution_result, request_logprobs).await?;

        if harmony_mode {
            // HARMONY PATH
            self.process_harmony_responses(
                all_responses,
                chat_request,
                dispatch,
            ).await
        } else {
            // EXISTING NON-HARMONY PATH (unchanged)
            self.process_legacy_responses(
                all_responses,
                chat_request,
                dispatch,
                stop_decoder.as_mut().unwrap(),
                request_logprobs,
            ).await
        }
    }

    async fn process_harmony_responses(
        &self,
        all_responses: Vec<proto::GenerateComplete>,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
    ) -> Result<ChatCompletionResponse, Response> {
        let mut choices = Vec::new();

        for (index, complete) in all_responses.iter().enumerate() {
            // Parse via Harmony
            let mut parser = HarmonyParserAdapter::new();
            let channel_output = parser
                .parse_complete(&complete.output_ids)
                .map_err(|e| utils::internal_error_message(format!("Harmony parsing failed: {}", e)))?;

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
                matched_stop: None,  // TBD: extract from Harmony parser state
                hidden_states: None,
            };

            choices.push(choice);
        }

        // Build usage
        let total_prompt_tokens: u32 = all_responses.iter().map(|r| r.prompt_tokens as u32).sum();
        let total_completion_tokens: u32 = all_responses.iter().map(|r| r.completion_tokens as u32).sum();
        let usage = Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
            completion_tokens_details: None,
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

    // Rename existing method
    async fn process_legacy_responses(
        &self,
        all_responses: Vec<proto::GenerateComplete>,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
    ) -> Result<ChatCompletionResponse, Response> {
        // EXISTING implementation from process_non_streaming_chat_response
        // (unchanged)
    }
}
```

**Similar modifications for StreamingProcessor** in `src/routers/grpc/streaming.rs`:
- Add `harmony_mode` parameter
- Branch on Harmony vs legacy
- For Harmony: maintain `HarmonyParserAdapter` per index
- Parse chunks through Harmony parser → emit SSE deltas

---

## MCP Loop Integration

### Responses API with MCP Loop

**File**: `src/routers/grpc/responses/mod.rs` (existing module)

**Key Changes**:
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
    // ... existing setup ...

    // NEW: Detect Harmony mode from model
    let is_harmony = HarmonyDetector::is_harmony_model(&request.model);

    // MCP Loop
    loop {
        // Build ChatCompletionRequest from ResponsesRequest + history
        let chat_request = if is_harmony {
            // Use HarmonyBuilder to rebuild with history
            build_chat_from_harmony_history(request.clone(), response_history)?
        } else {
            // Existing non-Harmony path
            build_chat_from_history(request.clone(), response_history)?
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

        // Parse response for tool calls
        let has_tool_calls = chat_response.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|t| !t.is_empty())
            .unwrap_or(false);

        // Convert to ResponsesResponse
        let responses_response = convert_chat_to_responses(chat_response, &request)?;

        // Persist
        response_storage.store(&response_id, &responses_response).await?;
        response_history.push(responses_response.clone());

        // Check MCP loop continuation
        if !has_tool_calls || response_history.len() >= max_iterations {
            // Final response
            return Ok(axum::Json(responses_response).into_response());
        }

        // Execute MCP tools
        let tool_results = execute_mcp_tools(
            responses_response.output.as_ref().unwrap(),
            &mcp_manager,
        ).await?;

        // Update history with tool results (will be used in next iteration)
        append_tool_results_to_history(&mut response_history, tool_results);
    }
}

fn build_chat_from_harmony_history(
    request: ResponsesRequest,
    history: Vec<ResponsesResponse>,
) -> Result<ChatCompletionRequest, String> {
    // Use HarmonyBuilder to rebuild messages with history
    let harmony_builder = HarmonyBuilder::new();

    // Extract messages from history
    let mut messages = Vec::new();
    for resp in history {
        // Convert ResponsesResponse back to messages
        // ... implementation ...
    }

    // Build new ChatCompletionRequest
    // Note: This will be fed through PreparationStage which will
    // detect Harmony and rebuild via HarmonyBuilder.build_from_chat()

    Ok(ChatCompletionRequest {
        model: request.model,
        messages,
        tools: request.tools,
        // ... other fields ...
    })
}
```

---

## Harmony Channels → Protocol Mapping

### Chat Completion

| Harmony Channel | Chat Field | Notes |
|----------------|-----------|-------|
| `analysis` | `message.reasoning_content` | O1-style reasoning |
| `commentary` | `message.tool_calls` | Parse JSON → ToolCall array |
| `final` | `message.content` | Assistant content |
| (finish) | `choice.finish_reason` | `"tool_calls"` if commentary present |

### Responses API

| Harmony Channel | ResponseOutputItem | Notes |
|----------------|-------------------|-------|
| `analysis` | `ResponseOutputItem::Reasoning` | Reasoning content |
| `commentary` → `mcp_*` | `ResponseOutputItem::McpCall`, `McpListTools` | MCP tool calls |
| `commentary` → `function_tool_call` | `ResponseOutputItem::FunctionToolCall` | Function calls |
| `commentary` → `web_search` | `ResponseOutputItem::WebSearch*` | Browser actions |
| `final` | `response.text` | Final text content |

---

## Implementation Plan

### Phase 1: Core Harmony Components (Week 1)
1. Create `src/routers/grpc/harmony/` module
   - `detector.rs` - Model detection
   - `builder.rs` - Harmony encoding (messages → token_ids)
   - `parser.rs` - StreamableParser adapter
   - `types.rs` - Shared types

2. Add Harmony encoder integration
   - Decide: crate feature vs runtime dependency
   - Implement encoding functions
   - Add stop token constants

3. Unit tests
   - Detector logic
   - Builder encoding
   - Parser channel mapping

### Phase 2: Pipeline Integration (Week 2)
1. Modify `PreparationStage`
   - Add Harmony branch
   - Validate logprobs rejection
   - Store Harmony-specific state

2. Modify `WorkerSelectionStage`
   - Use `selection_text` for Harmony

3. Modify `RequestBuildingStage`
   - Use `build_plain_generate_request` for Harmony
   - Inject Harmony stop token IDs

4. Extend `PreparationOutput` and `ResponseState`

5. Integration tests
   - Non-streaming chat with Harmony
   - Validate token flow

### Phase 3: Response Processing (Week 2-3)
1. Modify `ResponseProcessor`
   - Add Harmony branch
   - Implement `process_harmony_responses()`

2. Modify `StreamingProcessor`
   - Add Harmony branch
   - Per-index parser state
   - SSE delta emission

3. Tests
   - Non-streaming Harmony responses
   - Streaming Harmony responses
   - Tool call detection
   - Reasoning content extraction

### Phase 4: MCP Loop Integration (Week 3)
1. Modify responses module
   - Detect Harmony mode
   - Rebuild via HarmonyBuilder in loop
   - Tool result appending

2. Storage integration
   - Ensure ResponsesResponse format works with Harmony

3. Tests
   - Single MCP iteration
   - Multi-iteration loop
   - Loop termination
   - Tool result persistence

### Phase 5: Testing & Rollout (Week 4)
1. End-to-end tests
   - Harmony chat completion
   - Harmony responses with MCP
   - Streaming modes

2. Feature flag implementation
   - Per-model Harmony opt-in
   - Monitoring and metrics

3. Documentation
   - API docs
   - Migration guide
   - Harmony semantics guide

---

## Key Improvements Over V1

### 1. Architecture
- ✅ **Reuses existing 7-stage pipeline** instead of creating parallel pipeline
- ✅ **Conditional branching** via `harmony_mode` flag per stage
- ✅ **Zero code duplication** - extends existing stages

### 2. Token-Based Approach
- ✅ **Input**: `input_ids: Vec<u32>` not text
- ✅ **Output**: Feed `output_ids` to parser, not decoded text
- ✅ **Worker Selection**: Concise `selection_text` snippet, not full payload

### 3. Harmony Semantics
- ✅ **Channels**: Direct use of analysis/commentary/final
- ✅ **Parser**: `openai_harmony::StreamableParser` wrapped
- ✅ **Stop Tokens**: Harmony-specific (`<|return|>`, `<|call|>`)

### 4. Integration Points
- ✅ **Stage 1 (Prep)**: Harmony encoding vs chat template
- ✅ **Stage 4 (Build)**: `build_plain_generate_request` vs `build_generate_request`
- ✅ **Stage 7 (Response)**: Harmony parser vs tool/reasoning parsers

### 5. MCP Loop
- ✅ **Reuses existing responses module** MCP loop
- ✅ **Harmony rebuilding** in each iteration
- ✅ **Tool result appending** via HarmonyBuilder

### 6. Compatibility
- ✅ **Zero regression** for non-Harmony models
- ✅ **Feature flag** per model/request
- ✅ **Incremental rollout** with fallback

---

## Open Questions

1. **Harmony Encoder Source**
   - Option A: Crate feature (`openai-harmony` crate)
   - Option B: Runtime dependency (optional compilation)
   - **Recommendation**: Crate feature for cleaner integration

2. **Chat API MCP Loop**
   - Should Chat Completion support MCP loop?
   - **Recommendation**: No - keep MCP confined to Responses API (aligns with OpenAI semantics)

3. **Streaming Event Format**
   - Use existing SSE events or new Harmony-specific events?
   - **Recommendation**: Reuse existing `choice.delta.*` fields, map Harmony channels to them

4. **Logprobs Alternative**
   - Should we provide any alternative for debugging/analysis?
   - **Recommendation**: No - align with vLLM limitation, reject cleanly

5. **Tool Choice Constraints**
   - Harmony handles tools internally - how to validate `tool_choice`?
   - **Recommendation**: Validate that `tool_choice` is `auto` for Responses (vLLM constraint)

---

## Migration Notes

1. **Feature Flag**
   - Environment variable: `ENABLE_HARMONY_FOR_MODELS="gpt-oss,gpt-4o"`
   - Default: empty (Harmony disabled)

2. **Incremental Rollout**
   - Week 1-2: Internal testing (staging)
   - Week 3: 5% production traffic (canary)
   - Week 4-5: Gradual increase (25%, 50%, 75%)
   - Week 6: 100% for detected Harmony models

3. **Monitoring**
   - Metrics: `harmony_requests_total`, `harmony_parsing_errors`, `harmony_mcp_iterations`
   - Logs: Model type, Harmony on/off, token counts per channel
   - Alerts: Parsing error rate > 1%

4. **Fallback**
   - If Harmony encoding fails → return 400 error
   - If Harmony parsing fails mid-stream → send SSE error event and terminate
   - No automatic fallback to non-Harmony (fail explicitly)

---

## Conclusion

This revised design integrates Harmony support into the existing 7-stage pipeline through **conditional branching** rather than creating a separate pipeline. Key improvements:

- ✅ **Minimal Code Changes**: ~500 lines of modifications vs ~2000 lines new code
- ✅ **Token-Based**: Uses `input_ids`/`output_ids` correctly
- ✅ **Harmony Channels**: Direct use of analysis/commentary/final semantics
- ✅ **Parser Integration**: Wraps `openai_harmony::StreamableParser`
- ✅ **Zero Regression**: Non-Harmony models completely unaffected
- ✅ **MCP Loop**: Reuses existing responses module infrastructure
- ✅ **Incremental**: Can be landed incrementally (detector → builder → parser → MCP)

**Next Steps**:
1. Review and approve design
2. Implement Phase 1 (core components)
3. Add unit tests for Harmony builder/parser
4. Begin Stage 1 integration (PreparationStage)
