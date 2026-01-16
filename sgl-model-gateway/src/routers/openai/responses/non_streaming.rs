//! Non-streaming response handling for OpenAI-compatible responses
//!
//! This module handles non-streaming Responses API requests with MCP tool support.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use tracing::warn;

use super::{
    mcp::{execute_tool_loop, prepare_mcp_tools_as_functions},
    utils::{mask_tools_as_mcp, patch_response_with_request_metadata},
};
use crate::routers::{
    header_utils::{apply_provider_headers, extract_auth_header},
    mcp_utils::{ensure_request_mcp_client, McpLoopConfig},
    openai::context::{PayloadState, RequestContext},
    persistence_utils::persist_conversation_items,
};

/// Handle a non-streaming responses request
pub async fn handle_non_streaming_response(mut ctx: RequestContext) -> Response {
    let payload_state = match ctx.state.payload.take() {
        Some(ps) => ps,
        None => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "Payload not prepared").into_response();
        }
    };

    let PayloadState {
        json: mut payload,
        url,
        previous_response_id,
    } = payload_state;

    let original_body = ctx.responses_request();
    let worker = match ctx.worker() {
        Some(w) => w.clone(),
        None => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "Worker not selected").into_response();
        }
    };
    let mcp_manager = match ctx.components.mcp_manager() {
        Some(m) => m,
        None => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "MCP manager required").into_response();
        }
    };

    let server_keys = match original_body.tools.as_ref() {
        Some(tools) => match ensure_request_mcp_client(mcp_manager, tools.as_slice()).await {
            Some((_manager, keys)) => keys,
            None => Vec::new(),
        },
        None => Vec::new(),
    };

    let active_mcp = if mcp_manager.list_tools_for_servers(&server_keys).is_empty() {
        None
    } else {
        Some(mcp_manager)
    };

    let mut response_json: Value;

    if let Some(mcp) = active_mcp {
        let config = McpLoopConfig {
            server_keys: server_keys.clone(),
            ..McpLoopConfig::default()
        };
        prepare_mcp_tools_as_functions(
            &mut payload,
            mcp,
            &server_keys,
            original_body.tools.as_deref(),
        );

        match execute_tool_loop(
            ctx.components.client(),
            &url,
            ctx.headers(),
            payload,
            original_body,
            mcp,
            &config,
        )
        .await
        {
            Ok(resp) => response_json = resp,
            Err(err) => {
                worker.circuit_breaker().record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": {"message": err}})),
                )
                    .into_response();
            }
        }
    } else {
        let mut request_builder = ctx.components.client().post(&url).json(&payload);
        let auth_header = extract_auth_header(ctx.headers(), worker.api_key());
        request_builder = apply_provider_headers(request_builder, &url, auth_header.as_ref());

        let response = match request_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                worker.circuit_breaker().record_failure();
                tracing::error!(
                    url = %url,
                    error = %e,
                    "Failed to forward request to OpenAI"
                );
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("Failed to forward request to OpenAI: {}", e),
                )
                    .into_response();
            }
        };

        if !response.status().is_success() {
            worker.circuit_breaker().record_failure();
            let status = StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let body = response.text().await.unwrap_or_default();
            return (status, body).into_response();
        }

        response_json = match response.json::<Value>().await {
            Ok(r) => r,
            Err(e) => {
                worker.circuit_breaker().record_failure();
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to parse upstream response: {}", e),
                )
                    .into_response();
            }
        };

        worker.circuit_breaker().record_success();
    }

    mask_tools_as_mcp(&mut response_json, original_body);
    patch_response_with_request_metadata(
        &mut response_json,
        original_body,
        previous_response_id.as_deref(),
    );

    if let Err(err) = persist_conversation_items(
        ctx.components
            .conversation_storage()
            .expect("Conversation storage required")
            .clone(),
        ctx.components
            .conversation_item_storage()
            .expect("Conversation item storage required")
            .clone(),
        ctx.components
            .response_storage()
            .expect("Response storage required")
            .clone(),
        &response_json,
        original_body,
    )
    .await
    {
        warn!("Failed to persist conversation items: {}", err);
    }

    (StatusCode::OK, Json(response_json)).into_response()
}
