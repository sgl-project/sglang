//! Request context types for OpenAI router pipeline
//!
//! This module provides the core context types that flow through the router pipeline,
//! providing a single source of truth for request state.

use std::{collections::HashMap, sync::Arc, time::Instant};

use axum::http::HeaderMap;
use dashmap::DashMap;
use reqwest::StatusCode;
use serde_json::Value;

use crate::{
    core::CircuitBreaker,
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::McpManager,
    protocols::{
        chat::ChatCompletionRequest,
        responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest},
    },
};

use super::mcp::ToolLoopState;

// ============================================================================
// Main Request Context
// ============================================================================

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
#[derive(Clone)]
pub enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Responses(Arc<ResponsesRequest>),
}

// ============================================================================
// Shared Components
// ============================================================================

/// Shared components (injected once at creation)
pub struct SharedComponents {
    pub http_client: reqwest::Client,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub model_cache: Arc<DashMap<String, CachedEndpoint>>,
    pub mcp_manager: Arc<McpManager>,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
    pub worker_urls: Vec<String>,
}

/// Cached endpoint information
#[derive(Clone, Debug)]
pub struct CachedEndpoint {
    pub url: String,
    pub cached_at: Instant,
}

// ============================================================================
// Processing State (Mutable, evolves through stages)
// ============================================================================

/// Mutable processing state (evolves through pipeline stages)
#[derive(Default)]
pub struct ProcessingState {
    // Stage 1: Validation & Auth outputs
    pub validation: Option<ValidationOutput>,

    // Stage 2: Model Discovery outputs
    pub discovery: Option<DiscoveryOutput>,

    // Stage 3: Context Loading outputs
    pub context: Option<ContextOutput>,

    // Stage 4: Request Building outputs
    pub payload: Option<PayloadOutput>,

    // Stage 5: MCP Preparation outputs
    pub mcp: Option<McpOutput>,

    // Stage 6: Request Execution outputs
    pub execution: Option<ExecutionResult>,

    // Stage 7: Response Processing outputs
    pub processed: Option<ProcessedResponse>,

    // Stage 8: Final response
    pub final_response: Option<FinalResponse>,
}

// ============================================================================
// Stage Output Types
// ============================================================================

/// Output from validation stage (Stage 1)
#[derive(Clone)]
pub struct ValidationOutput {
    pub auth_header: Option<String>,
    pub validated_at: Instant,
}

/// Output from model discovery stage (Stage 2)
#[derive(Clone)]
pub struct DiscoveryOutput {
    pub endpoint_url: String,
    pub model: String,
}

/// Output from context loading stage (Stage 3)
#[derive(Clone)]
pub struct ContextOutput {
    pub conversation_items: Option<Vec<ResponseInputOutputItem>>,
    pub conversation_id: Option<String>,
    pub previous_response_id: Option<String>,
}

/// Output from request building stage (Stage 4)
#[derive(Clone)]
pub struct PayloadOutput {
    pub json_payload: Value,
    pub is_streaming: bool,
}

/// Output from MCP preparation stage (Stage 5)
pub struct McpOutput {
    pub active: bool,
    pub tool_loop_state: ToolLoopState,
    pub max_iterations: usize,
}

/// Output from request execution stage (Stage 6)
pub struct ExecutionResult {
    pub response: reqwest::Response,
    pub status: StatusCode,
}

/// Output from response processing stage (Stage 7)
pub struct ProcessedResponse {
    pub json_response: Value,
}

/// Final response (Stage 8)
pub struct FinalResponse {
    pub json_response: Value,
}

// ============================================================================
// RequestContext Implementation
// ============================================================================

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

    /// Create context for responses request
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
            RequestType::Responses(req) => req.stream.unwrap_or(false),
        }
    }

    /// Get model name
    pub fn model(&self) -> &str {
        match &self.input.request_type {
            RequestType::Chat(req) => &req.model,
            RequestType::Responses(req) => {
                self.input.model_id.as_deref().unwrap_or(&req.model)
            }
        }
    }
}
