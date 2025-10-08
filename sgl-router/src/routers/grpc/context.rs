//! Request context types for gRPC router pipeline
//!
//! This module provides the core context types that flow through the router pipeline,
//! eliminating deep parameter passing chains and providing a single source of truth
//! for request state.

use std::collections::HashMap;
use std::sync::Arc;

use axum::http::HeaderMap;
use serde_json::Value;

use crate::core::Worker;
use crate::grpc_client::{proto, SglangSchedulerClient};
use crate::protocols::spec::{ChatCompletionRequest, ChatCompletionResponse, GenerateRequest};
use crate::reasoning_parser::ReasoningParserFactory;
use crate::tokenizer::stop::StopSequenceDecoder;
use crate::tokenizer::traits::Tokenizer;
use crate::tool_parser::ToolParserFactory;

// ============================================================================
// Core Context Types
// ============================================================================

/// Main request processing context
///
/// This is the single source of truth for all request state as it flows
/// through the pipeline stages. Uses Rust's type system to enforce proper
/// stage ordering at compile time.
pub struct RequestContext {
    // === Input (Immutable) ===
    pub input: RequestInput,

    // === Shared Components (Immutable References) ===
    pub components: Arc<SharedComponents>,

    // === Processing State (Mutable, evolves through pipeline) ===
    pub state: ProcessingState,
}

/// Immutable request input
pub struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    pub model_id: Option<String>,
}

/// Request type variants
pub enum RequestType {
    Chat(Box<ChatCompletionRequest>),
    Generate(Box<GenerateRequest>),
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
    pub proto_request: Option<proto::GenerateRequest>,

    // Stage 5: Dispatch metadata
    pub dispatch: Option<DispatchMetadata>,

    // Stage 6: Response processing state
    pub response: ResponseState,
}

// ============================================================================
// Stage-Specific Output Types
// ============================================================================

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
        client: SglangSchedulerClient,
    },
    Dual {
        prefill: SglangSchedulerClient,
        decode: SglangSchedulerClient,
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
    pub collected: Option<Vec<proto::GenerateComplete>>,

    /// Execution result (streams from workers)
    pub execution_result: Option<ExecutionResult>,

    /// Final processed response
    pub final_response: Option<FinalResponse>,
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

// ============================================================================
// Context Builders
// ============================================================================

impl RequestContext {
    /// Create context for chat completion request
    pub fn for_chat(
        request: ChatCompletionRequest,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Chat(Box::new(request)),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for generate request
    pub fn for_generate(
        request: GenerateRequest,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Generate(Box::new(request)),
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

    /// Try to get chat request
    pub fn try_chat_request(&self) -> Option<&ChatCompletionRequest> {
        match &self.input.request_type {
            RequestType::Chat(req) => Some(req.as_ref()),
            _ => None,
        }
    }

    /// Get generate request (panics if not generate)
    pub fn generate_request(&self) -> &GenerateRequest {
        match &self.input.request_type {
            RequestType::Generate(req) => req.as_ref(),
            _ => panic!("Expected generate request"),
        }
    }

    /// Try to get generate request
    pub fn try_generate_request(&self) -> Option<&GenerateRequest> {
        match &self.input.request_type {
            RequestType::Generate(req) => Some(req.as_ref()),
            _ => None,
        }
    }

    /// Check if request is streaming
    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Generate(req) => req.stream,
        }
    }

    /// Check if request is chat
    pub fn is_chat(&self) -> bool {
        matches!(&self.input.request_type, RequestType::Chat(_))
    }

    /// Check if request is generate
    pub fn is_generate(&self) -> bool {
        matches!(&self.input.request_type, RequestType::Generate(_))
    }
}

// ============================================================================
// Default Implementations
// ============================================================================

// ============================================================================
// Helper Methods
// ============================================================================

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

    pub fn single(&self) -> Option<&SglangSchedulerClient> {
        match self {
            Self::Single { client } => Some(client),
            _ => None,
        }
    }

    pub fn single_mut(&mut self) -> Option<&mut SglangSchedulerClient> {
        match self {
            Self::Single { client } => Some(client),
            _ => None,
        }
    }

    pub fn dual(&self) -> Option<(&SglangSchedulerClient, &SglangSchedulerClient)> {
        match self {
            Self::Dual { prefill, decode } => Some((prefill, decode)),
            _ => None,
        }
    }

    pub fn dual_mut(&mut self) -> Option<(&mut SglangSchedulerClient, &mut SglangSchedulerClient)> {
        match self {
            Self::Dual { prefill, decode } => Some((prefill, decode)),
            _ => None,
        }
    }

    pub fn prefill_client(&self) -> Option<&SglangSchedulerClient> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            _ => None,
        }
    }

    pub fn prefill_client_mut(&mut self) -> Option<&mut SglangSchedulerClient> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            _ => None,
        }
    }

    pub fn decode_client(&self) -> Option<&SglangSchedulerClient> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            _ => None,
        }
    }

    pub fn decode_client_mut(&mut self) -> Option<&mut SglangSchedulerClient> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            _ => None,
        }
    }
}

// ============================================================================
// Execution and Response Types
// ============================================================================

use tonic::codec::Streaming;

/// Result of request execution (streams from workers)
pub enum ExecutionResult {
    Single {
        stream: Streaming<proto::GenerateResponse>,
    },
    Dual {
        prefill: Streaming<proto::GenerateResponse>,
        decode: Box<Streaming<proto::GenerateResponse>>,
    },
}

/// Final processed response
pub enum FinalResponse {
    Chat(ChatCompletionResponse),
    Generate(Box<GenerateRequest>),
}
