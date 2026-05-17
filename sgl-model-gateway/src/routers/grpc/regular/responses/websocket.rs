//! WebSocket executor for the gRPC regular Responses backend.
//!
//! Bridges the transport-neutral [`WsResponsesExecutor`] trait to the existing
//! gRPC regular Responses streaming pipeline so that WebSocket connections
//! reuse the same event-emission, storage, and persistence logic as HTTP SSE.

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{body::to_bytes, extract::ws::Message, http::HeaderMap};
use serde_json::json;
use tokio::sync::mpsc;
use tracing::debug;

use super::{
    common::{load_conversation_history_with_cache, normalize_request_input_items},
    conversions,
    streaming::{execute_non_mcp_stream_with_sink, execute_tool_loop_streaming_with_sink},
};
use crate::{
    core::WorkerRegistry,
    protocols::responses::{generate_id, ResponseStatus, ResponsesRequest, ResponsesResponse},
    routers::{
        error,
        grpc::{
            common::responses::{
                ensure_mcp_connection,
                utils::{persist_response_if_needed, validate_worker_availability},
                ResponsesContext,
            },
            harmony::HarmonyDetector,
        },
        responses_validation::normalize_and_validate_responses_request,
        ws_responses::{
            CachedWsResponse, WsClientError, WsResponseCreateOptions, WsResponsesExecutor,
        },
    },
};

#[derive(Clone)]
pub(crate) struct GrpcWsResponsesExecutor {
    worker_registry: Arc<WorkerRegistry>,
    responses_context: ResponsesContext,
}

impl GrpcWsResponsesExecutor {
    pub fn new(worker_registry: Arc<WorkerRegistry>, responses_context: ResponsesContext) -> Self {
        Self {
            worker_registry,
            responses_context,
        }
    }
}

#[async_trait]
impl WsResponsesExecutor for GrpcWsResponsesExecutor {
    async fn execute_response_create(
        &self,
        headers: HeaderMap,
        mut request: ResponsesRequest,
        options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let request_started_at = Instant::now();
        // WebSocket Responses is inherently event-streamed, so force streaming
        // on the downstream chat pipeline regardless of the client payload.
        request.stream = Some(true);
        request.background = Some(false);

        normalize_and_validate_responses_request(&mut request)
            .map_err(|err| WsClientError::new("invalid_request", err.to_string()))?;

        if request.conversation.is_some() {
            return Err(WsClientError::new(
                "unsupported_parameter",
                "The `conversation` field is not supported in WebSocket Responses V1.",
            ));
        }

        if let Some(error_response) =
            validate_worker_availability(&self.worker_registry, request.model.as_str())
        {
            return Err(response_to_ws_error(error_response).await);
        }

        if HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &request.model) {
            return Err(WsClientError::new(
                "unsupported_model",
                "Harmony-backed Responses are not supported on the WebSocket path in V1.",
            ));
        }

        let ctx = self.responses_context.clone_for_request();
        let had_cached_response = cached_response.is_some();
        let modified_request = match load_conversation_history_with_cache(
            &ctx,
            &request,
            cached_response.as_ref(),
            true,
        )
        .await
        {
            Ok(modified_request) => modified_request,
            Err(error_response) => return Err(response_to_ws_error(error_response).await),
        };
        debug!(
            model = %request.model,
            cached_response_hit = had_cached_response,
            input_items = normalize_request_input_items(&modified_request).len(),
            elapsed_ms = request_started_at.elapsed().as_secs_f64() * 1000.0,
            "loaded websocket response conversation history"
        );

        if options.generate == Some(false) {
            return warmup_response_create(&ctx, &request, &modified_request, outbound_tx).await;
        }

        let (has_mcp_tools, server_keys) =
            match ensure_mcp_connection(&ctx.mcp_manager, request.tools.as_deref()).await {
                Ok(result) => result,
                Err(error_response) => return Err(response_to_ws_error(error_response).await),
            };

        {
            let mut servers = ctx.requested_servers.write().unwrap();
            *servers = server_keys;
        }

        let final_response = if has_mcp_tools {
            execute_tool_loop_streaming_with_sink(
                &ctx,
                modified_request.clone(),
                &request,
                Some(headers),
                Some(request.model.clone()),
                outbound_tx,
            )
            .await
            .map_err(|err| WsClientError::new("stream_execution_failed", err))?
        } else {
            let chat_request = conversions::responses_to_chat(&modified_request)
                .map_err(|err| WsClientError::new("convert_request_failed", err.to_string()))?;

            execute_non_mcp_stream_with_sink(
                &ctx,
                Arc::new(chat_request),
                request.clone(),
                Some(headers),
                Some(request.model.clone()),
                &crate::routers::grpc::common::responses::streaming::WsResponseEventSink::new(
                    outbound_tx,
                ),
            )
            .await
            .map_err(|err| WsClientError::new("stream_execution_failed", err))?
        };
        debug!(
            model = %request.model,
            response_id = %final_response.id,
            status = ?final_response.status,
            elapsed_ms = request_started_at.elapsed().as_secs_f64() * 1000.0,
            "finished websocket response execution"
        );

        Ok(CachedWsResponse {
            response: final_response,
            input_items: normalize_request_input_items(&modified_request),
        })
    }
}

async fn response_to_ws_error(response: axum::response::Response) -> WsClientError {
    let status = response.status();
    let header_code = response
        .headers()
        .get(error::HEADER_X_SMG_ERROR_CODE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("responses_ws_error")
        .to_string();
    // Cap error-body reads to 1 MiB to prevent OOM on malformed upstream responses.
    let body_bytes = to_bytes(response.into_body(), 1_048_576).await.ok();
    let parsed_error = body_bytes
        .as_ref()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).ok())
        .and_then(|value| value.get("error").cloned());

    let error_code = parsed_error
        .as_ref()
        .and_then(|error| error.get("code"))
        .and_then(|value| value.as_str())
        .unwrap_or(&header_code)
        .to_string();
    let error_message = parsed_error
        .as_ref()
        .and_then(|error| error.get("message"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("WebSocket Responses request failed with status {}", status));

    WsClientError::new(error_code, error_message)
        .with_status(status.as_u16())
        .with_param_if_previous_response_not_found()
}

async fn warmup_response_create(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
    modified_request: &ResponsesRequest,
    outbound_tx: mpsc::Sender<Message>,
) -> Result<CachedWsResponse, WsClientError> {
    let response = ResponsesResponse::builder(generate_id("resp"), &request.model)
        .copy_from_request(request)
        .status(ResponseStatus::Completed)
        .output(vec![])
        .build();

    let in_progress_response = ResponsesResponse::builder(&response.id, &response.model)
        .copy_from_request(request)
        .status(ResponseStatus::InProgress)
        .output(vec![])
        .build();
    send_ws_message(
        &outbound_tx,
        json!({ "type": "response.created", "response": in_progress_response }),
    )?;
    send_ws_message(
        &outbound_tx,
        json!({ "type": "response.completed", "response": response.clone() }),
    )?;

    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &response,
        request,
    )
    .await;

    Ok(CachedWsResponse {
        response,
        input_items: normalize_request_input_items(modified_request),
    })
}

fn send_ws_message(
    outbound_tx: &mpsc::Sender<Message>,
    payload: serde_json::Value,
) -> Result<(), WsClientError> {
    outbound_tx
        .try_send(Message::Text(payload.to_string().into()))
        .map_err(|_| {
            WsClientError::new(
                "client_disconnected",
                "WebSocket client disconnected or outbound buffer full.",
            )
            .with_status(499)
        })
}

trait WsClientErrorExt {
    fn with_param_if_previous_response_not_found(self) -> Self;
}

impl WsClientErrorExt for WsClientError {
    fn with_param_if_previous_response_not_found(self) -> Self {
        if self.code == "previous_response_not_found" {
            self.with_param("previous_response_id")
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::StatusCode, response::IntoResponse};

    use super::*;

    #[tokio::test]
    async fn test_response_to_ws_error_preserves_router_error_body_message() {
        let err = response_to_ws_error(error::not_found(
            "previous_response_not_found",
            "Previous response 'resp_missing' was not found in the current session or durable storage.",
        ))
        .await;

        assert_eq!(err.code, "previous_response_not_found");
        assert_eq!(
            err.message,
            "Previous response 'resp_missing' was not found in the current session or durable storage."
        );
        assert_eq!(err.status, 404);
        assert_eq!(err.param.as_deref(), Some("previous_response_id"));
    }

    #[tokio::test]
    async fn test_response_to_ws_error_falls_back_for_non_json_error_body() {
        let response = (
            StatusCode::BAD_REQUEST,
            [(error::HEADER_X_SMG_ERROR_CODE, "responses_ws_error")],
            Body::from("plain text error"),
        )
            .into_response();

        let err = response_to_ws_error(response).await;

        assert_eq!(err.code, "responses_ws_error");
        assert_eq!(
            err.message,
            "WebSocket Responses request failed with status 400 Bad Request"
        );
        assert_eq!(err.status, 400);
        assert_eq!(err.param, None);
    }
}
