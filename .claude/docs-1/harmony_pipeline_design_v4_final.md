# GPT-OSS Harmony Unified Pipeline Design - Final v4

**Document Version**: 4.0 (Final)
**Date**: 2025-10-26
**Status**: Implementation Ready

## Executive Summary

This document describes the integration of Harmony support for GPT-OSS models through a **dual-pipeline architecture** that shares common stages. The design creates a dedicated Harmony pipeline alongside the regular pipeline, with maximum code reuse for shared stages.

**Key Architecture Principles**:
- ✅ Two pipeline configurations: Regular and Harmony
- ✅ Share stages where logic is identical (5 out of 7 stages)
- ✅ Harmony-specific stages for encoding, request building, and response parsing
- ✅ Refactored file structure: `stages/` folder for better organization
- ✅ Token-based approach (`input_ids` → `output_ids`)
- ✅ Harmony channels (`analysis`/`commentary`/`final`)
- ✅ Zero regression for non-Harmony models

---

## Architecture Decision: Why Two Pipelines?

### The Critical Stage Ordering Issue

**Regular Pipeline**:
```
1. Preparation (chat template + tokenization) → original_text
2. Worker Selection (uses original_text)
3. ... rest of stages
```

**Harmony Pipeline**:
```
1. Harmony Preparation (Harmony encoding) → input_ids + selection_text
2. Worker Selection (uses selection_text from step 1)
3. ... rest of stages
```

**Key Insight**: For Harmony, worker selection needs `selection_text` which is computed DURING Harmony encoding. This means Harmony encoding must happen BEFORE worker selection, but in a different way than regular preparation.

**Solution**: Two pipeline configurations with shared stage implementations where possible.

---

## Proposed File Structure

### Current Structure (Before)
```
src/routers/grpc/
  pipeline.rs          # 1183 lines: pipeline + all 7 stage implementations
  context.rs
  processing.rs
  streaming.rs
  router.rs
  utils.rs
  responses/
```

### New Structure (After)
```
src/routers/grpc/
  pipeline.rs          # Pipeline orchestration only (~200 lines)
  context.rs           # Unchanged
  processing.rs        # Unchanged
  streaming.rs         # Unchanged
  router.rs            # Modified: choose regular vs harmony pipeline
  utils.rs             # Unchanged

  stages/              # NEW: Regular pipeline stages
    mod.rs
    preparation.rs                # PreparationStage
    worker_selection.rs           # WorkerSelectionStage (shared)
    client_acquisition.rs         # ClientAcquisitionStage (shared)
    request_building.rs           # RequestBuildingStage
    dispatch_metadata.rs          # DispatchMetadataStage (shared)
    request_execution.rs          # RequestExecutionStage (shared)
    response_processing.rs        # ResponseProcessingStage

  harmony/             # NEW: Harmony implementation
    mod.rs
    detector.rs                   # HarmonyDetector
    builder.rs                    # HarmonyBuilder
    parser.rs                     # HarmonyParserAdapter
    types.rs                      # Harmony types

    stages/            # Harmony-specific stages
      mod.rs
      preparation.rs              # HarmonyPreparationStage
      request_building.rs         # HarmonyRequestBuildingStage
      response_processing.rs      # HarmonyResponseProcessingStage

  responses/           # Unchanged
    mod.rs
    ...
```

---

## Pipeline Architecture

### Shared vs Harmony-Specific Stages

| Stage | Regular Pipeline | Harmony Pipeline | Shared? |
|-------|-----------------|------------------|---------|
| 1. Preparation | `PreparationStage` (chat template) | `HarmonyPreparationStage` (Harmony encoding) | ❌ Different |
| 2. Worker Selection | `WorkerSelectionStage` | `WorkerSelectionStage` | ✅ Shared* |
| 3. Client Acquisition | `ClientAcquisitionStage` | `ClientAcquisitionStage` | ✅ Shared |
| 4. Request Building | `RequestBuildingStage` | `HarmonyRequestBuildingStage` | ❌ Different |
| 5. Dispatch Metadata | `DispatchMetadataStage` | `DispatchMetadataStage` | ✅ Shared |
| 6. Execution | `RequestExecutionStage` | `RequestExecutionStage` | ✅ Shared |
| 7. Response Processing | `ResponseProcessingStage` | `HarmonyResponseProcessingStage` | ❌ Different |

*WorkerSelectionStage needs minor modification to check `selection_text` vs `original_text`

### Pipeline Implementation

**File**: `src/routers/grpc/pipeline.rs`

```rust
use std::sync::Arc;
use async_trait::async_trait;
use axum::response::Response;

mod stages;
use stages::*;

mod harmony;

/// Pipeline stage trait (unchanged)
#[async_trait]
pub trait PipelineStage: Send + Sync {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response>;
    fn name(&self) -> &'static str;
}

/// Pipeline orchestrator
#[derive(Clone)]
pub struct RequestPipeline {
    stages: Arc<Vec<Box<dyn PipelineStage>>>,
}

impl RequestPipeline {
    /// Create regular (non-Harmony) pipeline
    pub fn new_regular(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tokenizer: Arc<dyn Tokenizer>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        let processor = processing::ResponseProcessor::new(
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            configured_tool_parser.clone(),
            configured_reasoning_parser.clone(),
        );

        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            tokenizer,
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
        ));

        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(stages::PreparationStage),
            Box::new(stages::WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                stages::WorkerSelectionMode::Regular,
            )),
            Box::new(stages::ClientAcquisitionStage),
            Box::new(stages::RequestBuildingStage::new(false)),
            Box::new(stages::DispatchMetadataStage),
            Box::new(stages::RequestExecutionStage::new(stages::ExecutionMode::Single)),
            Box::new(stages::ResponseProcessingStage::new(processor, streaming_processor)),
        ];

        Self {
            stages: Arc::new(stages),
        }
    }

    /// Create Harmony pipeline
    pub fn new_harmony(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tokenizer: Arc<dyn Tokenizer>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        let harmony_processor = harmony::stages::HarmonyResponseProcessor::new(
            tokenizer.clone(),
        );

        let harmony_streaming_processor = Arc::new(
            harmony::stages::HarmonyStreamingProcessor::new(tokenizer)
        );

        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(harmony::stages::HarmonyPreparationStage::new()),
            Box::new(stages::WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                stages::WorkerSelectionMode::Regular,
            )),
            Box::new(stages::ClientAcquisitionStage),
            Box::new(harmony::stages::HarmonyRequestBuildingStage::new(false)),
            Box::new(stages::DispatchMetadataStage),
            Box::new(stages::RequestExecutionStage::new(stages::ExecutionMode::Single)),
            Box::new(harmony::stages::HarmonyResponseProcessingStage::new(
                harmony_processor,
                harmony_streaming_processor,
            )),
        ];

        Self {
            stages: Arc::new(stages),
        }
    }

    /// Create PD (prefill-decode) pipeline - regular
    pub fn new_pd(...) -> Self {
        // Similar to new_regular but with PD stages
    }

    /// Create Harmony PD pipeline
    pub fn new_harmony_pd(...) -> Self {
        // Similar to new_harmony but with PD stages
    }

    /// Execute pipeline for chat request
    pub async fn execute_chat(...) -> Response {
        // Same implementation as before
        // Iterate through stages, handle Ok(Some)/Ok(None)/Err
    }

    /// Execute pipeline for generate request
    pub async fn execute_generate(...) -> Response {
        // Same implementation as before
    }

    /// Execute for responses (with background tasks)
    pub async fn execute_chat_for_responses(...) -> Result<ChatCompletionResponse, String> {
        // Same implementation as before
    }
}
```

### Router Integration

**File**: `src/routers/grpc/router.rs`

```rust
pub struct GrpcRouter {
    // Regular pipelines
    regular_pipeline: RequestPipeline,
    pd_pipeline: RequestPipeline,

    // Harmony pipelines
    harmony_pipeline: RequestPipeline,
    harmony_pd_pipeline: RequestPipeline,

    // ... other fields (unchanged)
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tokenizer: Arc<dyn Tokenizer>,
    // ...
}

impl GrpcRouter {
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Create all four pipelines
        let regular_pipeline = RequestPipeline::new_regular(...);
        let pd_pipeline = RequestPipeline::new_pd(...);
        let harmony_pipeline = RequestPipeline::new_harmony(...);
        let harmony_pd_pipeline = RequestPipeline::new_harmony_pd(...);

        Ok(GrpcRouter {
            regular_pipeline,
            pd_pipeline,
            harmony_pipeline,
            harmony_pd_pipeline,
            // ... other fields
        })
    }

    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Detect Harmony mode
        let is_harmony = harmony::HarmonyDetector::is_harmony_model(&body.model);

        // Detect PD mode (existing logic)
        let use_pd = self.should_use_pd(&body.model);

        // Select appropriate pipeline
        let pipeline = match (is_harmony, use_pd) {
            (true, true) => &self.harmony_pd_pipeline,
            (true, false) => &self.harmony_pipeline,
            (false, true) => &self.pd_pipeline,
            (false, false) => &self.regular_pipeline,
        };

        pipeline
            .execute_chat(
                Arc::new(body.clone()),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                self.shared_components.clone(),
            )
            .await
    }

    async fn route_generate_impl(...) -> Response {
        // Similar logic
    }
}
```

---

## Detailed Stage Specifications

### Shared Stages (Used by Both Pipelines)

#### 1. WorkerSelectionStage (Minor Modification)

**File**: `src/routers/grpc/stages/worker_selection.rs`

```rust
pub struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    mode: WorkerSelectionMode,
}

impl WorkerSelectionStage {
    fn select_single_worker(
        &self,
        model_id: Option<&str>,
        prep: &PreparationOutput,
    ) -> Option<Arc<dyn Worker>> {
        // Get workers
        let workers = self.worker_registry.get_workers_filtered(...);
        let available: Vec<_> = workers.iter().filter(|w| w.is_available()).cloned().collect();

        if available.is_empty() {
            return None;
        }

        let policy = self.policy_registry.get_policy_or_default(model_id.unwrap_or(""));

        // MODIFICATION: Use selection_text for Harmony, original_text for regular
        let text_for_policy = if prep.harmony_mode {
            prep.selection_text.as_deref()
        } else {
            prep.original_text.as_deref()
        };

        let idx = policy.select_worker(&available, text_for_policy)?;
        Some(available[idx].clone())
    }
}
```

#### 2. ClientAcquisitionStage (Unchanged)

**File**: `src/routers/grpc/stages/client_acquisition.rs`

```rust
pub struct ClientAcquisitionStage;

#[async_trait]
impl PipelineStage for ClientAcquisitionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Existing implementation - works for both regular and Harmony
        // (just gets gRPC clients from selected workers)
    }

    fn name(&self) -> &'static str {
        "ClientAcquisition"
    }
}
```

#### 3. DispatchMetadataStage (Unchanged)

**File**: `src/routers/grpc/stages/dispatch_metadata.rs`

```rust
pub struct DispatchMetadataStage;

#[async_trait]
impl PipelineStage for DispatchMetadataStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Existing implementation - works for both regular and Harmony
        // (just prepares request_id, model, created, weight_version)
    }

    fn name(&self) -> &'static str {
        "DispatchMetadata"
    }
}
```

#### 4. RequestExecutionStage (Unchanged)

**File**: `src/routers/grpc/stages/request_execution.rs`

```rust
pub struct RequestExecutionStage {
    mode: ExecutionMode,
}

#[async_trait]
impl PipelineStage for RequestExecutionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Existing implementation - works for both regular and Harmony
        // (executes gRPC generate request, single or dual dispatch)
    }

    fn name(&self) -> &'static str {
        "RequestExecution"
    }
}
```

### Regular Pipeline Stages

#### 5. PreparationStage

**File**: `src/routers/grpc/stages/preparation.rs`

```rust
pub struct PreparationStage;

#[async_trait]
impl PipelineStage for PreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Existing implementation from pipeline.rs
        // - Filter tools
        // - Process messages & apply chat template
        // - Tokenize → token_ids
        // - Build tool constraints
        // - Create StopSequenceDecoder
    }

    fn name(&self) -> &'static str {
        "Preparation"
    }
}
```

#### 6. RequestBuildingStage

**File**: `src/routers/grpc/stages/request_building.rs`

```rust
pub struct RequestBuildingStage {
    inject_pd_metadata: bool,
}

#[async_trait]
impl PipelineStage for RequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Existing implementation from pipeline.rs
        // - Build proto::GenerateRequest using chat template
        // - Inject PD metadata if needed
    }

    fn name(&self) -> &'static str {
        "RequestBuilding"
    }
}
```

#### 7. ResponseProcessingStage

**File**: `src/routers/grpc/stages/response_processing.rs`

```rust
pub struct ResponseProcessingStage {
    processor: processing::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
}

#[async_trait]
impl PipelineStage for ResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Existing implementation from pipeline.rs
        // - For streaming: use StreamingProcessor → SSE
        // - For non-streaming: use ResponseProcessor → final response
    }

    fn name(&self) -> &'static str {
        "ResponseProcessing"
    }
}
```

### Harmony Pipeline Stages

#### 8. HarmonyPreparationStage

**File**: `src/routers/grpc/harmony/stages/preparation.rs`

```rust
use crate::routers::grpc::harmony::{HarmonyBuilder, HarmonyDetector};

pub struct HarmonyPreparationStage {
    builder: HarmonyBuilder,
}

impl HarmonyPreparationStage {
    pub fn new() -> Self {
        Self {
            builder: HarmonyBuilder::new(),
        }
    }
}

#[async_trait]
impl PipelineStage for HarmonyPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(request) => {
                self.prepare_chat(ctx, request).await
            }
            RequestType::Generate(request) => {
                self.prepare_generate(ctx, request).await
            }
            RequestType::Responses(request) => {
                self.prepare_responses(ctx, request).await
            }
        }
    }

    fn name(&self) -> &'static str {
        "HarmonyPreparation"
    }
}

impl HarmonyPreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Validate - reject logprobs
        if request.logprobs {
            return Err(utils::bad_request_error(
                "logprobs are not supported for Harmony models"
            ));
        }

        // Build via Harmony
        let build_output = self.builder
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

        Ok(())
    }

    async fn prepare_responses(
        &self,
        ctx: &mut RequestContext,
        request: &ResponsesRequest,
    ) -> Result<(), Response> {
        // Validate
        if request.logprobs.unwrap_or(false) {
            return Err(utils::bad_request_error(
                "logprobs are not supported for Harmony models"
            ));
        }

        if !matches!(request.tool_choice, None | Some(ToolChoice::Value(ToolChoiceValue::Auto))) {
            return Err(utils::bad_request_error(
                "Harmony models only support tool_choice='auto' for Responses API"
            ));
        }

        // Load history from storage
        let history = load_response_history(request, &ctx.storage).await?;

        // Build via Harmony with history
        let build_output = self.builder
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

        Ok(())
    }
}
```

#### 9. HarmonyRequestBuildingStage

**File**: `src/routers/grpc/harmony/stages/request_building.rs`

```rust
pub struct HarmonyRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl HarmonyRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for HarmonyRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref()
            .ok_or_else(|| utils::internal_error_static("Preparation not completed"))?;

        let clients = ctx.state.clients.as_ref()
            .ok_or_else(|| utils::internal_error_static("Client acquisition not completed"))?;

        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Build plain generate request with input_ids
        let request_id = self.generate_request_id(&ctx.input.request_type);
        let sampling_params = self.extract_sampling_params(&ctx.input.request_type);

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
                params.stop_token_ids
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(harmony_stops);
            }
        }

        // Inject PD metadata if needed
        if self.inject_pd_metadata {
            if let WorkerSelection::Dual { prefill, .. } = ctx.state.workers.as_ref().unwrap() {
                self.inject_bootstrap_metadata(&mut gen_req, prefill);
            }
        }

        ctx.state.proto_request = Some(gen_req);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "HarmonyRequestBuilding"
    }
}
```

#### 10. HarmonyResponseProcessingStage

**File**: `src/routers/grpc/harmony/stages/response_processing.rs`

```rust
use super::super::{HarmonyParserAdapter, HarmonyResponseProcessor, HarmonyStreamingProcessor};

pub struct HarmonyResponseProcessingStage {
    processor: HarmonyResponseProcessor,
    streaming_processor: Arc<HarmonyStreamingProcessor>,
}

impl HarmonyResponseProcessingStage {
    pub fn new(
        processor: HarmonyResponseProcessor,
        streaming_processor: Arc<HarmonyStreamingProcessor>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
        }
    }
}

#[async_trait]
impl PipelineStage for HarmonyResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.process_chat_response(ctx).await,
            RequestType::Generate(_) => self.process_generate_response(ctx).await,
            RequestType::Responses(_) => self.process_responses_response(ctx).await,
        }
    }

    fn name(&self) -> &'static str {
        "HarmonyResponseProcessing"
    }
}

impl HarmonyResponseProcessingStage {
    async fn process_chat_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();
        let execution_result = ctx.state.response.execution_result.take()
            .ok_or_else(|| utils::internal_error_static("No execution result"))?;

        let dispatch = ctx.state.dispatch.as_ref()
            .ok_or_else(|| utils::internal_error_static("Dispatch metadata not set"))?
            .clone();

        if is_streaming {
            // Use HarmonyStreamingProcessor
            return Ok(Some(
                self.streaming_processor.clone().process_streaming_response(
                    execution_result,
                    ctx.chat_request_arc(),
                    dispatch,
                )
            ));
        }

        // Non-streaming: use HarmonyResponseProcessor
        let response = self.processor
            .process_non_streaming_chat_response(
                execution_result,
                ctx.chat_request_arc(),
                dispatch,
            )
            .await?;

        ctx.state.response.final_response = Some(FinalResponse::Chat(response));
        Ok(None)
    }
}
```

---

## Harmony Core Components

### 1. HarmonyDetector

**File**: `src/routers/grpc/harmony/detector.rs`

```rust
pub struct HarmonyDetector;

impl HarmonyDetector {
    pub fn is_harmony_model(model_name: &str) -> bool {
        model_name.contains("gpt-oss")
            || model_name.contains("gpt-4o")
            || model_name.contains("gpt-4.5")
            || model_name.starts_with("gpt-5")
    }
}
```

### 2. HarmonyBuilder

**File**: `src/routers/grpc/harmony/builder.rs`

```rust
pub struct HarmonyBuilder {
    // Harmony encoding provider
}

pub struct HarmonyBuildOutput {
    pub input_ids: Vec<u32>,
    pub stop_token_ids: Vec<u32>,
    pub selection_text: String,
    pub harmony_messages: Vec<HarmonyMessage>,
}

impl HarmonyBuilder {
    pub fn new() -> Self {
        Self { /* ... */ }
    }

    pub fn build_from_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<HarmonyBuildOutput, String> {
        // Convert Chat → Harmony messages
        // Encode → input_ids
        // Extract selection_text
        // Get stop token IDs
    }

    pub fn build_from_responses(
        &self,
        request: &ResponsesRequest,
        history: Vec<ResponsesResponse>,
    ) -> Result<HarmonyBuildOutput, String> {
        // Convert Responses + history → Harmony messages
        // Handle conversation splicing (drop intermediate analysis)
        // Encode → input_ids
        // Extract selection_text
    }

    pub fn append_tool_results(
        &self,
        messages: Vec<HarmonyMessage>,
        tool_results: Vec<McpToolResult>,
    ) -> Result<HarmonyBuildOutput, String> {
        // Append tool results to messages
        // Re-encode → new input_ids
    }
}
```

### 3. HarmonyParserAdapter

**File**: `src/routers/grpc/harmony/parser.rs`

```rust
use openai_harmony::StreamableParser;

pub struct HarmonyParserAdapter {
    parser: StreamableParser,
}

pub struct HarmonyChannelOutput {
    pub analysis: Option<String>,
    pub commentary: Option<Vec<ToolCall>>,
    pub final_text: String,
    pub finish_reason: String,
    pub matched_stop: Option<Value>,
}

impl HarmonyParserAdapter {
    pub fn new() -> Self;

    pub fn parse_complete(
        &mut self,
        output_ids: &[u32],
    ) -> Result<HarmonyChannelOutput, String>;

    pub fn parse_chunk(
        &mut self,
        chunk_ids: &[u32],
    ) -> Result<Option<HarmonyChannelDelta>, String>;

    pub fn finalize(&mut self) -> Result<HarmonyChannelOutput, String>;
}
```

### 4. HarmonyResponseProcessor

**File**: `src/routers/grpc/harmony/processor.rs`

```rust
pub struct HarmonyResponseProcessor {
    tokenizer: Arc<dyn Tokenizer>,
}

impl HarmonyResponseProcessor {
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
    ) -> Result<ChatCompletionResponse, Response> {
        // Collect responses
        // Parse via HarmonyParserAdapter
        // Map channels → ChatChoice
        // Track reasoning_tokens
        // Build ChatCompletionResponse
    }
}
```

### 5. HarmonyStreamingProcessor

**File**: `src/routers/grpc/harmony/streaming.rs`

```rust
pub struct HarmonyStreamingProcessor {
    tokenizer: Arc<dyn Tokenizer>,
}

impl HarmonyStreamingProcessor {
    pub fn process_streaming_response(
        self: Arc<Self>,
        execution_result: ExecutionResult,
        request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
    ) -> Response {
        // Spawn background task
        // Maintain per-index HarmonyParserAdapter
        // Parse chunks → SSE deltas
        // Track reasoning_tokens
    }
}
```

---

## Modified Context Types

**File**: `src/routers/grpc/context.rs`

```rust
pub struct PreparationOutput {
    pub original_text: Option<String>,
    pub token_ids: Vec<u32>,
    pub processed_messages: Option<ProcessedMessages>,
    pub tool_constraints: Option<(String, String)>,
    pub filtered_request: Option<ChatCompletionRequest>,

    // Harmony-specific fields
    pub harmony_mode: bool,
    pub selection_text: Option<String>,
    pub harmony_messages: Option<Vec<HarmonyMessage>>,
    pub harmony_stop_ids: Option<Vec<u32>>,
}

pub struct ResponseState {
    pub stop_decoder: Option<StopSequenceDecoder>,
    pub execution_result: Option<ExecutionResult>,
    pub final_response: Option<FinalResponse>,

    // Harmony-specific parser state
    pub harmony_parser: Option<HarmonyParserAdapter>,
    pub harmony_parser_per_index: Option<HashMap<usize, HarmonyParserAdapter>>,
}
```

---

## Implementation Plan

### Phase 1: File Structure Refactoring (Week 1)

**Goal**: Extract existing stages into separate files (no functionality changes)

1. Create `src/routers/grpc/stages/` folder
2. Move each stage from `pipeline.rs` into its own file:
   - `preparation.rs`
   - `worker_selection.rs`
   - `client_acquisition.rs`
   - `request_building.rs`
   - `dispatch_metadata.rs`
   - `request_execution.rs`
   - `response_processing.rs`
   - `mod.rs` (exports all stages)

3. Update `pipeline.rs`:
   - Keep only pipeline orchestration code
   - Import stages from `stages::*`
   - Ensure all tests still pass

**Testing**: Run full test suite, ensure zero regressions

### Phase 2: Harmony Core Components (Week 1-2)

**Goal**: Implement Harmony building and parsing

1. Create `src/routers/grpc/harmony/` folder
2. Implement core components:
   - `detector.rs` - Model detection
   - `builder.rs` - Harmony encoding
   - `parser.rs` - StreamableParser adapter
   - `types.rs` - Shared types
   - `mod.rs`

3. Unit tests:
   - Detector logic
   - Builder encoding (new vs continuing conversation)
   - Parser channel mapping
   - Token counting

### Phase 3: Harmony Stages (Week 2)

**Goal**: Implement Harmony-specific stages

1. Create `src/routers/grpc/harmony/stages/` folder
2. Implement Harmony stages:
   - `preparation.rs` - HarmonyPreparationStage
   - `request_building.rs` - HarmonyRequestBuildingStage
   - `response_processing.rs` - HarmonyResponseProcessingStage
   - `mod.rs`

3. Implement Harmony processors:
   - `processor.rs` - HarmonyResponseProcessor
   - `streaming.rs` - HarmonyStreamingProcessor

4. Integration tests:
   - Non-streaming chat
   - Streaming chat
   - Token flow validation

### Phase 4: Pipeline Integration (Week 2-3)

**Goal**: Create Harmony pipelines and integrate with router

1. Modify `pipeline.rs`:
   - Add `new_harmony()` method
   - Add `new_harmony_pd()` method

2. Modify `router.rs`:
   - Add `harmony_pipeline` and `harmony_pd_pipeline` fields
   - Update `route_chat_impl()` to select appropriate pipeline
   - Update `route_generate_impl()` similarly

3. Modify `WorkerSelectionStage`:
   - Check `selection_text` vs `original_text`

4. Integration tests:
   - Regular pipeline still works
   - Harmony pipeline works for detected models
   - PD mode works for both

### Phase 5: MCP Loop Integration (Week 3)

**Goal**: Integrate Harmony with Responses API MCP loop

1. Modify `src/routers/grpc/responses/mod.rs`:
   - Detect Harmony mode
   - Build from history (drop analysis messages)
   - Use appropriate pipeline

2. Tests:
   - Single MCP iteration
   - Multi-iteration loop
   - History splicing
   - Tool result appending

### Phase 6: Testing & Rollout (Week 3-4)

**Goal**: Comprehensive testing and production rollout

1. End-to-end tests:
   - All Harmony flows
   - Token accounting
   - Metrics validation

2. Feature flag:
   - `HARMONY_ENABLED_MODELS` env var
   - Per-model opt-in

3. Documentation:
   - Architecture guide
   - API documentation
   - Migration guide

---

## Benefits of This Architecture

### 1. Code Organization
✅ Each stage in its own file (~150-200 lines each)
✅ Clear separation between regular and Harmony logic
✅ Easy to find and modify specific stage logic

### 2. Reusability
✅ 5 out of 7 stages completely shared
✅ Only 3 Harmony-specific stage implementations needed
✅ Common utilities and helpers shared

### 3. Testability
✅ Each stage can be unit tested independently
✅ Pipeline orchestration tested separately
✅ Easy to mock stages for integration tests

### 4. Maintainability
✅ Changes to shared stages benefit both pipelines
✅ Harmony-specific changes isolated to harmony/ folder
✅ Clear ownership and responsibilities

### 5. Future Extensibility
✅ Easy to add new pipeline configurations (e.g., for new models)
✅ Easy to add new stages or modify existing ones
✅ Clear pattern to follow for future enhancements

---

## Migration Path

### Step 1: Refactor (No Functionality Change)
- Extract stages into separate files
- All tests pass
- Zero behavioral changes

### Step 2: Add Harmony (Feature Flagged)
- Implement Harmony components
- Create Harmony pipeline
- Disabled by default (env var)

### Step 3: Gradual Rollout
- Enable for internal testing
- Canary deployment (5%)
- Gradual increase to 100%

---

## Conclusion

This dual-pipeline architecture provides the best balance of:
- **Code reuse**: 5 out of 7 stages shared
- **Clean separation**: Harmony logic isolated
- **Maintainability**: Each stage in its own file
- **Flexibility**: Easy to modify or extend
- **Zero regression**: Non-Harmony models unaffected

The refactored file structure makes the codebase more navigable and maintainable, while the dual-pipeline approach cleanly handles the different stage ordering requirements between regular and Harmony modes.

**Next Steps**:
1. Review and approve architecture
2. Begin Phase 1: File structure refactoring
3. Implement Phase 2: Harmony core components
4. Continue through implementation plan
