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

/// Parse function calls from model output text
pub async fn parse_function_call(
    context: Option<&Arc<AppContext>>,
    req: &ParseFunctionCallRequest,
) -> Response {
    match context {
        Some(ctx) => match &ctx.tool_parser_factory {
            Some(factory) => match factory.registry().get_pooled_parser(&req.tool_call_parser) {
                Some(pooled_parser) => {
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
                            (
                                StatusCode::BAD_REQUEST,
                                Json(serde_json::json!({
                                    "error": format!("Failed to parse function calls: {}", e),
                                    "success": false
                                })),
                            )
                                .into_response()
                        }
                    }
                }
                None => (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": format!("Unknown tool parser: {}", req.tool_call_parser),
                        "success": false
                    })),
                )
                    .into_response(),
            },
            None => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Tool parser factory not initialized",
                    "success": false
                })),
            )
                .into_response(),
        },
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Context not initialized",
                "success": false
            })),
        )
            .into_response(),
    }
}

/// Parse and separate reasoning from normal text
pub async fn parse_reasoning(
    context: Option<&Arc<AppContext>>,
    req: &SeparateReasoningRequest,
) -> Response {
    match context {
        Some(ctx) => match &ctx.reasoning_parser_factory {
            Some(factory) => match factory.registry().get_pooled_parser(&req.reasoning_parser) {
                Some(pooled_parser) => {
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
                            (
                                StatusCode::BAD_REQUEST,
                                Json(serde_json::json!({
                                    "error": format!("Failed to separate reasoning: {}", e),
                                    "success": false
                                })),
                            )
                                .into_response()
                        }
                    }
                }
                None => (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": format!("Unknown reasoning parser: {}", req.reasoning_parser),
                        "success": false
                    })),
                )
                    .into_response(),
            },
            None => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "Reasoning parser factory not initialized",
                    "success": false
                })),
            )
                .into_response(),
        },
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Context not initialized",
                "success": false
            })),
        )
            .into_response(),
    }
}
