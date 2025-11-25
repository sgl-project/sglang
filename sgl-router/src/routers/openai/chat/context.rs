//! Context types for Chat pipeline
//!
//! This module defines the context types used by the chat pipeline,
//! which is a lightweight proxy for /v1/chat/completions.

use std::{sync::Arc, time::Instant};

use axum::http::HeaderMap;
use dashmap::DashMap;
use serde_json::Value;

use crate::{
    core::CircuitBreaker,
    protocols::chat::ChatCompletionRequest,
    routers::openai::router::CachedEndpoint,
};

// ============================================================================
// Chat Request Context
// ============================================================================

/// Main request processing context for chat pipeline
///
/// This context flows through all chat pipeline stages, accumulating
/// state as it progresses.
pub struct ChatRequestContext {
    pub input: ChatRequestInput,
    pub dependencies: Arc<ChatDependencies>,
    pub state: ChatProcessingState,
}

/// Immutable request input for chat pipeline
pub struct ChatRequestInput {
    pub request: Arc<ChatCompletionRequest>,
    pub headers: Option<HeaderMap>,
    pub model_id: Option<String>,
}

/// Dependencies injected into chat pipeline
///
/// These are the shared components that the pipeline needs to execute,
/// injected once at creation and shared across all requests.
pub struct ChatDependencies {
    pub http_client: reqwest::Client,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub model_cache: Arc<DashMap<String, CachedEndpoint>>,
    pub worker_urls: Vec<String>,
}

/// Mutable processing state for chat pipeline
///
/// State accumulates as request flows through stages:
/// 1. Validation -> validation output
/// 2. ModelDiscovery -> discovery output
/// 3. RequestBuilding -> payload output
/// 4. RequestExecution -> returns response directly (no state)
#[derive(Default)]
pub struct ChatProcessingState {
    // Stage 1: Validation output
    pub validation: Option<ValidationOutput>,

    // Stage 2: Model Discovery output
    pub discovery: Option<DiscoveryOutput>,

    // Stage 3: Request Building output
    pub payload: Option<PayloadOutput>,

    // Note: No execution or response fields
    // RequestExecution stage returns Response directly via Ok(Some(response))
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

/// Output from request building stage
#[derive(Clone)]
pub struct PayloadOutput {
    pub json_payload: Value,
    pub is_streaming: bool,
}

// ============================================================================
// ChatRequestContext Implementation
// ============================================================================

impl ChatRequestContext {
    /// Create a new chat request context
    pub fn new(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        dependencies: Arc<ChatDependencies>,
    ) -> Self {
        Self {
            input: ChatRequestInput {
                request,
                headers,
                model_id,
            },
            dependencies,
            state: ChatProcessingState::default(),
        }
    }

    /// Get reference to the chat request
    pub fn request(&self) -> &ChatCompletionRequest {
        self.input.request.as_ref()
    }

    /// Get Arc clone of the chat request
    pub fn request_arc(&self) -> Arc<ChatCompletionRequest> {
        Arc::clone(&self.input.request)
    }

    /// Check if request is streaming
    pub fn is_streaming(&self) -> bool {
        self.input.request.stream
    }

    /// Get model name
    pub fn model(&self) -> &str {
        self.input.model_id.as_deref().unwrap_or(&self.input.request.model)
    }
}
