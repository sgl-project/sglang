//! Context types for Responses pipeline
//!
//! This module defines the context types used by the responses pipeline,
//! which handles /v1/responses with full MCP support, conversation history,
//! and persistence.

use std::{sync::Arc, time::Instant};

use axum::http::HeaderMap;
use dashmap::DashMap;
use reqwest::StatusCode;
use serde_json::Value;

use super::super::mcp::ToolLoopState;
use crate::{
    core::CircuitBreaker,
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::McpManager,
    protocols::responses::{ResponseInputOutputItem, ResponsesRequest},
    routers::openai::router::CachedEndpoint,
};

// ============================================================================
// Responses Request Context
// ============================================================================

/// Main request processing context for responses pipeline
///
/// This context flows through all responses pipeline stages (8 stages),
/// accumulating state as it progresses.
pub struct ResponsesRequestContext {
    pub input: ResponsesRequestInput,
    pub dependencies: Arc<ResponsesDependencies>,
    pub state: ResponsesProcessingState,
}

/// Immutable request input for responses pipeline
pub struct ResponsesRequestInput {
    pub request: Arc<ResponsesRequest>,
    pub headers: Option<HeaderMap>,
    pub model_id: Option<String>,
}

/// Dependencies injected into responses pipeline
///
/// Includes both basic dependencies (HTTP client, circuit breaker) and
/// responses-specific dependencies (storage, MCP manager).
pub struct ResponsesDependencies {
    // Basic dependencies (same as chat)
    pub http_client: reqwest::Client,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub model_cache: Arc<DashMap<String, CachedEndpoint>>,
    pub worker_urls: Vec<String>,

    // Responses-specific dependencies
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
    pub mcp_manager: Arc<McpManager>,
}

/// Mutable processing state for responses pipeline
///
/// State accumulates through 8 stages:
/// 1. Validation -> validation output
/// 2. ModelDiscovery -> discovery output
/// 3. ContextLoading -> context output (conversation history)
/// 4. RequestBuilding -> payload output
/// 5. McpPreparation -> mcp output (tool loop state)
/// 6. RequestExecution -> execution output
/// 7. ResponseProcessing -> processed output (or returns response for streaming)
/// 8. Persistence -> final_response output
#[derive(Default)]
pub struct ResponsesProcessingState {
    // Stage 1: Validation output
    pub validation: Option<ValidationOutput>,

    // Stage 2: Model Discovery output
    pub discovery: Option<DiscoveryOutput>,

    // Stage 3: Context Loading output (responses-specific)
    pub context: Option<ContextOutput>,

    // Stage 4: Request Building output
    pub payload: Option<PayloadOutput>,

    // Stage 5: MCP Preparation output (responses-specific)
    pub mcp: Option<McpOutput>,

    // Stage 6: Request Execution output
    pub execution: Option<ExecutionResult>,

    // Stage 7: Response Processing output
    pub processed: Option<ProcessedResponse>,

    // Stage 8: Final response (after persistence)
    pub final_response: Option<FinalResponse>,
}

// ============================================================================
// Stage Output Types
// ============================================================================

/// Output from validation stage
#[derive(Clone)]
pub struct ValidationOutput {
    pub auth_header: Option<String>,
    pub validated_at: Instant,
}

/// Output from model discovery stage
#[derive(Clone)]
pub struct DiscoveryOutput {
    pub endpoint_url: String,
    pub model: String,
}

/// Output from context loading stage (responses-specific)
#[derive(Clone)]
pub struct ContextOutput {
    pub conversation_items: Option<Vec<ResponseInputOutputItem>>,
    pub conversation_id: Option<String>,
    pub previous_response_id: Option<String>,
}

/// Output from request building stage
#[derive(Clone)]
pub struct PayloadOutput {
    pub json_payload: Value,
    pub is_streaming: bool,
}

/// Output from MCP preparation stage (responses-specific)
pub struct McpOutput {
    pub active: bool,
    pub tool_loop_state: ToolLoopState,
    pub max_iterations: usize,
}

/// Output from request execution stage
pub struct ExecutionResult {
    pub response: reqwest::Response,
    pub status: StatusCode,
}

/// Output from response processing stage
pub struct ProcessedResponse {
    pub json_response: Value,
}

/// Final response (after persistence)
pub struct FinalResponse {
    pub json_response: Value,
}

// ============================================================================
// ResponsesRequestContext Implementation
// ============================================================================

impl ResponsesRequestContext {
    /// Create a new responses request context
    pub fn new(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        dependencies: Arc<ResponsesDependencies>,
    ) -> Self {
        Self {
            input: ResponsesRequestInput {
                request,
                headers,
                model_id,
            },
            dependencies,
            state: ResponsesProcessingState::default(),
        }
    }

    /// Get reference to the responses request
    pub fn request(&self) -> &ResponsesRequest {
        self.input.request.as_ref()
    }

    /// Get Arc clone of the responses request
    pub fn request_arc(&self) -> Arc<ResponsesRequest> {
        Arc::clone(&self.input.request)
    }

    /// Check if request is streaming
    pub fn is_streaming(&self) -> bool {
        self.input.request.stream.unwrap_or(false)
    }

    /// Get model name
    pub fn model(&self) -> &str {
        self.input
            .model_id
            .as_deref()
            .unwrap_or(&self.input.request.model)
    }
}
