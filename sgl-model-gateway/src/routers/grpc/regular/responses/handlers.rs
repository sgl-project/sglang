//! Handler functions for /v1/responses endpoints
//!
//! # Public API
//!
//! - `route_responses()` - POST /v1/responses (main entry point)
//!
//! # Architecture
//!
//! This module provides the entry point for the /v1/responses endpoint.
//! It supports two execution modes:
//!
//! 1. **Synchronous** - Returns complete response immediately (non_streaming.rs)
//! 2. **Streaming** - Returns SSE stream with real-time events (streaming.rs)
//!
//! Note: Background mode is no longer supported. Requests with background=true
//! will be rejected with a 400 error.
//!
//! # Request Flow
//!
//! ```text
//! route_responses()
//!   ├─► route_responses_sync()  → non_streaming::route_responses_internal()
//!   └─► route_responses_streaming()
//!       ├─► streaming::execute_tool_loop_streaming() (MCP tools)
//!       └─► streaming::convert_chat_stream_to_responses_stream() (no MCP)
//! ```

use std::sync::Arc;

use axum::{
    http,
    response::{IntoResponse, Response},
};
use tracing::debug;
use uuid::Uuid;

use super::{
    common::load_conversation_history, context::ResponsesContext, conversions, non_streaming,
    streaming,
};
use crate::{
    protocols::responses::ResponsesRequest,
    routers::{error, grpc::common::responses::ensure_mcp_connection},
};

/// Main handler for POST /v1/responses
///
/// Validates request, determines execution mode (sync/streaming), and delegates
pub(crate) async fn route_responses(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    // 1. Reject background mode (no longer supported)
    let is_background = request.background.unwrap_or(false);
    if is_background {
        return error::bad_request(
            "unsupported_parameter",
            "Background mode is not supported. Please set 'background' to false or omit it.",
        );
    }

    // 2. Route based on execution mode
    let is_streaming = request.stream.unwrap_or(false);
    if is_streaming {
        route_responses_streaming(ctx, request, headers, model_id).await
    } else {
        // Generate response ID for synchronous execution
        let response_id = Some(format!("resp_{}", Uuid::new_v4()));
        route_responses_sync(ctx, request, headers, model_id, response_id).await
    }
}

// ============================================================================
// Synchronous Entry Point
// ============================================================================

/// Execute synchronous responses request
async fn route_responses_sync(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Response {
    match non_streaming::route_responses_internal(ctx, request, headers, model_id, response_id)
        .await
    {
        Ok(responses_response) => axum::Json(responses_response).into_response(),
        Err(response) => response, // Already a Response with proper status code
    }
}

// ============================================================================
// Streaming Entry Point
// ============================================================================

/// Execute streaming responses request
async fn route_responses_streaming(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    // 1. Load conversation history
    let modified_request = match load_conversation_history(ctx, &request).await {
        Ok(req) => req,
        Err(response) => return response, // Already a Response with proper status code
    };

    // 2. Check MCP connection and get whether MCP tools are present
    let has_mcp_tools =
        match ensure_mcp_connection(&ctx.mcp_manager, request.tools.as_deref()).await {
            Ok(has_mcp) => has_mcp,
            Err(response) => return response,
        };

    if has_mcp_tools {
        debug!("MCP tools detected in streaming mode, using streaming tool loop");

        return streaming::execute_tool_loop_streaming(
            ctx,
            modified_request,
            &request,
            headers,
            model_id,
        )
        .await;
    }

    // 3. Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = match conversions::responses_to_chat(&modified_request) {
        Ok(req) => Arc::new(req),
        Err(e) => {
            return error::bad_request(
                "convert_request_failed",
                format!("Failed to convert request: {}", e),
            );
        }
    };

    // 4. Execute chat pipeline and convert streaming format (no MCP tools)
    streaming::convert_chat_stream_to_responses_stream(
        ctx,
        chat_request,
        headers,
        model_id,
        &request,
    )
    .await
}
