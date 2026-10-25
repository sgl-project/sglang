//! `/parse_function_call` and `/separate_reasoning`: frontend-only parsing
//! over a finished text.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use sglang_api_types::api::v1 as api;

use super::app::AppState;
use super::native_api::native_error;
use crate::api_server::handlers::{ToolSpec, parse_function_call, separate_reasoning};

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/parse_function_call", post(parse_function_call_route))
        .route("/separate_reasoning", post(separate_reasoning_route))
}

/// `POST /parse_function_call` — Python's `{"normal_text", "calls": [{"tool_index",
/// "name", "parameters"}]}`; a parser the server does not know is a 400.
async fn parse_function_call_route(
    State(_state): State<Arc<AppState>>,
    body: Result<Json<api::ParseFunctionCallRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let tools: Vec<ToolSpec> = request
        .tools
        .into_iter()
        .map(|tool| {
            let function = tool.function.unwrap_or_default();
            ToolSpec {
                name: function.name,
                parameters: function.parameters,
            }
        })
        .collect();
    match parse_function_call(&request.text, &tools, request.tool_call_parser.as_deref()).await {
        Ok((normal_text, calls)) => Json(api::ParseFunctionCallResponse {
            normal_text,
            calls: calls
                .into_iter()
                .map(|call| api::ParsedToolCall {
                    tool_index: call.tool_index,
                    name: call.name,
                    parameters: call.parameters,
                })
                .collect(),
        })
        .into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// `POST /separate_reasoning` — Python's `{"reasoning_text", "text"}`, plus
/// `reasoning_blocks` / `text_blocks` with `return_blocks`.
async fn separate_reasoning_route(
    State(_state): State<Arc<AppState>>,
    body: Result<Json<api::SeparateReasoningRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    match separate_reasoning(&request.text, &request.reasoning_parser) {
        Ok((reasoning_text, text)) => {
            let blocks = request.return_blocks;
            Json(api::SeparateReasoningResponse {
                reasoning_blocks: blocks.then(|| api::StringList {
                    items: if reasoning_text.is_empty() {
                        Vec::new()
                    } else {
                        vec![reasoning_text.clone()]
                    },
                }),
                text_blocks: blocks.then(|| api::StringList {
                    items: vec![text.clone()],
                }),
                reasoning_text,
                text,
            })
            .into_response()
        }
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request as HttpRequest, header::CONTENT_TYPE},
    };
    use tower::ServiceExt;

    use super::*;
    use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::ServerArgs;
    use crate::message::request::GenerationPolicy;

    fn state() -> Arc<AppState> {
        let server_args = Arc::new(ServerArgs::default());
        let (intake_tx, _intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        Arc::new(AppState {
            core: CoreHandle::new(
                intake_tx,
                abort_tx,
                CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    peer_endpoints: Vec::new(),
                    metadata: CoreMetadata::from(server_args.as_ref()),
                },
            ),
            generation_policy: GenerationPolicy::from(server_args.as_ref()),
            server_args,
            chat_formatter: None,
        })
    }

    async fn post(uri: &str, body: &str) -> (StatusCode, serde_json::Value) {
        let response = routes()
            .with_state(state())
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri(uri)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// Both routes answer Python's field names, and `return_blocks` adds the
    /// block lists.
    #[tokio::test]
    async fn parser_routes_answer_python_shapes() {
        let (status, body) = post(
            "/parse_function_call",
            r#"{"text": "<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":1}}</tool_call>",
                "tools": [{"function": {"name": "f"}}], "tool_call_parser": "qwen25"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["normal_text"], "");
        assert_eq!(body["calls"][0]["tool_index"], 0);
        assert_eq!(body["calls"][0]["name"], "f");
        assert_eq!(body["calls"][0]["parameters"], "{\"x\":1}");

        let (status, _) = post(
            "/parse_function_call",
            r#"{"text": "x", "tool_call_parser": "nope"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, body) = post(
            "/separate_reasoning",
            r#"{"text": "<think>why</think>four", "reasoning_parser": "deepseek-r1", "return_blocks": true}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["reasoning_text"], "why");
        assert_eq!(body["text"], "four");
        assert_eq!(body["reasoning_blocks"], serde_json::json!(["why"]));
        assert_eq!(body["text_blocks"], serde_json::json!(["four"]));

        let (_, body) = post(
            "/separate_reasoning",
            r#"{"text": "plain", "reasoning_parser": "deepseek-r1"}"#,
        )
        .await;
        assert!(body.get("reasoning_blocks").is_none());
    }
}
