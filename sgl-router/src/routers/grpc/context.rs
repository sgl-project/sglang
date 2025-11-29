//! Request context types for gRPC router pipeline
//!
//! This module provides the core context types that flow through the router pipeline,
//! eliminating deep parameter passing chains and providing a single source of truth
//! for request state.

use std::{collections::HashMap, sync::Arc};

use axum::http::HeaderMap;
use serde_json::Value;

use super::{
    client::GrpcClient,
    proto_wrapper::{ProtoGenerateComplete, ProtoGenerateRequest, ProtoStream},
};
use crate::{
    core::Worker,
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        generate::{GenerateRequest, GenerateResponse},
        responses::ResponsesRequest,
    },
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    tokenizer::{stop::StopSequenceDecoder, traits::Tokenizer},
    tool_parser::ParserFactory as ToolParserFactory,
};

/// Main request processing context
///
/// This is the single source of truth for all request state as it flows
/// through the pipeline stages. Uses Rust's type system to enforce proper
/// stage ordering at compile time.
pub struct RequestContext {
    pub input: RequestInput,
    pub components: Arc<SharedComponents>,
    pub state: ProcessingState,
}

/// Immutable request input
pub struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    pub model_id: Option<String>,
}

/// Request type variants
/// Using Arc instead of Box to enable cheap cloning for background tasks
pub enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Generate(Arc<GenerateRequest>),
    Responses(Arc<ResponsesRequest>),
}

/// Shared components (injected once at creation)
pub struct SharedComponents {
    pub tokenizer: Arc<dyn Tokenizer>,
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
}

/// Mutable processing state (evolves through pipeline stages)
#[derive(Default)]
pub struct ProcessingState {
    // Stage 1: Preparation outputs
    pub preparation: Option<PreparationOutput>,

    // Stage 2: Worker selection outputs
    pub workers: Option<WorkerSelection>,

    // Stage 3: Client acquisition outputs
    pub clients: Option<ClientSelection>,

    // Stage 4: Request building outputs
    pub proto_request: Option<ProtoGenerateRequest>,

    // Stage 5: Dispatch metadata
    pub dispatch: Option<DispatchMetadata>,

    // Stage 6: Response processing state
    pub response: ResponseState,
}

/// Output from preparation stage (Step 1)
pub struct PreparationOutput {
    /// Original text (for chat) or resolved text (for generate)
    pub original_text: Option<String>,

    /// Tokenized input
    pub token_ids: Vec<u32>,

    /// Processed messages (chat only)
    pub processed_messages: Option<super::ProcessedMessages>,

    /// Tool call constraints (if applicable)
    pub tool_constraints: Option<(String, String)>,

    /// Filtered request (if tools were filtered)
    pub filtered_request: Option<ChatCompletionRequest>,

    // Harmony-specific fields
    /// Whether this is a Harmony request (default: false)
    pub harmony_mode: bool,

    /// Selection text for worker routing (Harmony only)
    pub selection_text: Option<String>,

    /// Harmony messages for history tracking (Harmony only)
    pub harmony_messages: Option<Vec<super::harmony::HarmonyMessage>>,

    /// Stop token IDs for Harmony models
    pub harmony_stop_ids: Option<Vec<u32>>,
}

/// Worker selection (Step 2)
pub enum WorkerSelection {
    Single {
        worker: Arc<dyn Worker>,
    },
    Dual {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    },
}

/// Client selection (Step 3)
pub enum ClientSelection {
    Single {
        client: GrpcClient,
    },
    Dual {
        prefill: GrpcClient,
        decode: GrpcClient,
    },
}

/// Dispatch metadata (Step 5)
#[derive(Clone)]
pub struct DispatchMetadata {
    pub request_id: String,
    pub model: String,
    pub created: u64,
    pub weight_version: Option<String>,
    pub is_streaming: bool,
}

/// Response processing state (Step 6)
#[derive(Default)]
pub struct ResponseState {
    /// Stop sequence decoder
    pub stop_decoder: Option<StopSequenceDecoder>,

    /// Per-index streaming state (for n>1 support)
    pub streaming: StreamingState,

    /// Collected responses (non-streaming)
    pub collected: Option<Vec<ProtoGenerateComplete>>,

    /// Execution result (streams from workers)
    pub execution_result: Option<ExecutionResult>,

    /// Final processed response
    pub final_response: Option<FinalResponse>,

    /// Responses API iteration result (Harmony only, for tool loop orchestration)
    pub responses_iteration_result: Option<super::harmony::ResponsesIterationResult>,

    // Harmony-specific parser state
    /// Harmony parser for non-streaming (single parser for all indices)
    pub harmony_parser: Option<super::harmony::HarmonyParserAdapter>,

    /// Harmony parsers for streaming (one per index for n>1 support)
    pub harmony_parser_per_index: Option<HashMap<usize, super::harmony::HarmonyParserAdapter>>,
}

/// Streaming state (per-choice tracking)
#[derive(Default)]
pub struct StreamingState {
    pub is_firsts: HashMap<u32, bool>,
    pub stream_buffers: HashMap<u32, String>,
    pub finish_reasons: HashMap<u32, String>,
    pub matched_stops: HashMap<u32, Option<Value>>,
    pub prompt_tokens: HashMap<u32, u32>,
    pub completion_tokens: HashMap<u32, u32>,
    pub cached_tokens: HashMap<u32, u32>,

    // Parser state (lazy initialization per index)
    pub reasoning_parsers:
        HashMap<u32, Arc<std::sync::Mutex<Box<dyn crate::reasoning_parser::ReasoningParser>>>>,
    pub tool_parsers:
        HashMap<u32, Arc<tokio::sync::Mutex<Box<dyn crate::tool_parser::ToolParser>>>>,
    pub has_tool_calls: HashMap<u32, bool>,
}

impl RequestContext {
    /// Create context for chat completion request
    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Chat(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for generate request
    pub fn for_generate(
        request: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Generate(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for Responses API request
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Responses(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Get reference to original request (type-safe)
    pub fn request(&self) -> &RequestType {
        &self.input.request_type
    }

    /// Get chat request (panics if not chat)
    pub fn chat_request(&self) -> &ChatCompletionRequest {
        match &self.input.request_type {
            RequestType::Chat(req) => req.as_ref(),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get Arc clone of chat request (panics if not chat)
    pub fn chat_request_arc(&self) -> Arc<ChatCompletionRequest> {
        match &self.input.request_type {
            RequestType::Chat(req) => Arc::clone(req),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get generate request (panics if not generate)
    pub fn generate_request(&self) -> &GenerateRequest {
        match &self.input.request_type {
            RequestType::Generate(req) => req.as_ref(),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get Arc clone of generate request (panics if not generate)
    pub fn generate_request_arc(&self) -> Arc<GenerateRequest> {
        match &self.input.request_type {
            RequestType::Generate(req) => Arc::clone(req),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get responses request (panics if not responses)
    pub fn responses_request(&self) -> &ResponsesRequest {
        match &self.input.request_type {
            RequestType::Responses(req) => req.as_ref(),
            _ => panic!("Expected responses request"),
        }
    }

    /// Get Arc clone of responses request (panics if not responses)
    pub fn responses_request_arc(&self) -> Arc<ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Arc::clone(req),
            _ => panic!("Expected responses request"),
        }
    }

    /// Check if request is streaming
    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Generate(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
        }
    }
}

impl WorkerSelection {
    pub fn is_dual(&self) -> bool {
        matches!(self, Self::Dual { .. })
    }

    pub fn single(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Single { worker } => Some(worker),
            _ => None,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn dual(&self) -> Option<(&Arc<dyn Worker>, &Arc<dyn Worker>)> {
        match self {
            Self::Dual { prefill, decode } => Some((prefill, decode)),
            _ => None,
        }
    }

    pub fn prefill_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            _ => None,
        }
    }

    pub fn decode_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            _ => None,
        }
    }
}

impl ClientSelection {
    pub fn is_dual(&self) -> bool {
        matches!(self, Self::Dual { .. })
    }

    pub fn single(&self) -> Option<&GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            _ => None,
        }
    }

    pub fn single_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            _ => None,
        }
    }

    pub fn dual(&self) -> Option<(&GrpcClient, &GrpcClient)> {
        match self {
            Self::Dual { prefill, decode } => Some((prefill, decode)),
            _ => None,
        }
    }

    pub fn dual_mut(&mut self) -> Option<(&mut GrpcClient, &mut GrpcClient)> {
        match self {
            Self::Dual { prefill, decode } => Some((prefill, decode)),
            _ => None,
        }
    }

    pub fn prefill_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            _ => None,
        }
    }

    pub fn prefill_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            _ => None,
        }
    }

    pub fn decode_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            _ => None,
        }
    }

    pub fn decode_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            _ => None,
        }
    }
}

/// Result of request execution (streams from workers)
/// Uses ProtoStream to automatically abort on cancellation
pub enum ExecutionResult {
    Single {
        stream: ProtoStream,
    },
    Dual {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
    },
}

/// Final processed response
pub enum FinalResponse {
    Chat(ChatCompletionResponse),
    /// Generate response is a Vec of GenerateResponse (n=1 returns single item, n>1 returns multiple)
    Generate(Vec<GenerateResponse>),
}
