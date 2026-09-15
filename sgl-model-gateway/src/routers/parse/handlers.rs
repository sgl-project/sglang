//! Parser handlers for function calls and reasoning extraction

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use tracing::error;

use crate::{
    app_context::AppContext,
    protocols::parser::{ParseFunctionCallRequest, SeparateReasoningRequest},
};

/// Helper to create error responses
fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": message,
            "success": false
        })),
    )
        .into_response()
}

/// Parse function calls from model output text
pub async fn parse_function_call(
    ctx: &Arc<AppContext>,
    req: &ParseFunctionCallRequest,
) -> Response {
    let Some(factory) = &ctx.tool_parser_factory else {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Tool parser factory not initialized",
        );
    };

    let Some(pooled_parser) = factory.registry().get_pooled_parser(&req.tool_call_parser) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!("Unknown tool parser: {}", req.tool_call_parser),
        );
    };

    let parser = pooled_parser.lock().await;
    match parser.parse_complete(&req.text).await {
        Ok((remaining_text, tool_calls)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "remaining_text": remaining_text,
                "tool_calls": tool_calls,
                "success": true
            })),
        )
            .into_response(),
        Err(e) => {
            error!("Failed to parse function calls: {}", e);
            error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to parse function calls: {}", e),
            )
        }
    }
}

/// Parse and separate reasoning from normal text
pub async fn parse_reasoning(ctx: &Arc<AppContext>, req: &SeparateReasoningRequest) -> Response {
    let Some(factory) = &ctx.reasoning_parser_factory else {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Reasoning parser factory not initialized",
        );
    };

    let Some(pooled_parser) = factory.registry().get_pooled_parser(&req.reasoning_parser) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!("Unknown reasoning parser: {}", req.reasoning_parser),
        );
    };

    let mut parser = pooled_parser.lock().await;
    match parser.detect_and_parse_reasoning(&req.text) {
        Ok(result) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "normal_text": result.normal_text,
                "reasoning_text": result.reasoning_text,
                "success": true
            })),
        )
            .into_response(),
        Err(e) => {
            error!("Failed to separate reasoning: {}", e);
            error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to separate reasoning: {}", e),
            )
        }
    }
}
