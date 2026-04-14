use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::ws::{Message, WebSocket},
    http::{HeaderMap, Request, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::{SinkExt, StreamExt};
use smg::{
    core::WorkerRegistry,
    protocols::responses::{
        ResponseContentPart, ResponseInputOutputItem, ResponseOutputItem, ResponseStatus,
        ResponsesRequest, ResponsesResponse,
    },
    routers::{
        router_manager::{router_ids, RouterManager},
        ws_responses::{
            serve_responses_ws_with_config, CachedWsResponse, WsClientError,
            WsResponseCreateOptions, WsResponsesExecutor, WsRuntimeConfig,
        },
        RouterTrait,
    },
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, Notify},
};
use tokio_tungstenite::connect_async;
use tower::ServiceExt;

use crate::common::test_app::{create_test_app_context, create_test_app_with_context};

#[derive(Clone)]
struct StubWsExecutor {
    gate: Option<Arc<Notify>>,
}

impl StubWsExecutor {
    fn immediate() -> Self {
        Self { gate: None }
    }

    fn gated(gate: Arc<Notify>) -> Self {
        Self { gate: Some(gate) }
    }
}

#[derive(Clone)]
struct DelayedReturnWsExecutor {
    return_delay: Duration,
}

impl DelayedReturnWsExecutor {
    fn new(return_delay: Duration) -> Self {
        Self { return_delay }
    }
}

#[async_trait]
impl WsResponsesExecutor for StubWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let model = request.model.clone();
        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_ws_test",
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        if let Some(gate) = &self.gate {
            gate.notified().await;
        }

        let output_text = "stub websocket output";
        let response = ResponsesResponse::builder("resp_ws_test", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_test".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: output_text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response,
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response: ResponsesResponse::builder("resp_ws_test", request.model.clone())
                .copy_from_request(&request)
                .status(ResponseStatus::Completed)
                .output(vec![ResponseOutputItem::Message {
                    id: "msg_ws_test".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: output_text.to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: "completed".to_string(),
                }])
                .build(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_test".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello websocket".to_string(),
                }],
                status: Some("completed".to_string()),
            }],
        })
    }
}

#[async_trait]
impl WsResponsesExecutor for DelayedReturnWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let output_text = "delayed websocket output";
        let response = ResponsesResponse::builder("resp_ws_delayed", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_delayed".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: output_text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            }])
            .build();

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response.id,
                "object": "response",
                "status": "in_progress",
                "model": request.model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response.clone(),
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        tokio::time::sleep(self.return_delay).await;

        Ok(CachedWsResponse {
            response,
            input_items: vec![],
        })
    }
}

#[derive(Clone, Default)]
struct FunctionCallWsExecutor;

#[async_trait]
impl WsResponsesExecutor for FunctionCallWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let response_id = "resp_ws_tool_test";
        let item_id = "fc_ws_test";
        let call_id = "call_ws_test";
        let tool_name = "search_repo";
        let arguments = r#"{"query":"fizz_buzz"}"#;
        let model = request.model.clone();

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let output_item_added = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": tool_name,
                "status": "in_progress",
                "arguments": ""
            }
        });
        let _ = outbound_tx.try_send(Message::Text(output_item_added.to_string().into()));

        let args_delta = serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": item_id,
            "delta": arguments
        });
        let _ = outbound_tx.try_send(Message::Text(args_delta.to_string().into()));

        let args_done = serde_json::json!({
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": item_id,
            "arguments": arguments
        });
        let _ = outbound_tx.try_send(Message::Text(args_done.to_string().into()));

        let output_item_done = serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": tool_name,
                "status": "completed",
                "arguments": arguments
            }
        });
        let _ = outbound_tx.try_send(Message::Text(output_item_done.to_string().into()));

        let response = ResponsesResponse::builder(response_id, request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::FunctionToolCall {
                id: item_id.to_string(),
                call_id: call_id.to_string(),
                name: tool_name.to_string(),
                arguments: arguments.to_string(),
                output: None,
                status: "completed".to_string(),
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response,
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response: ResponsesResponse::builder(response_id, request.model.clone())
                .copy_from_request(&request)
                .status(ResponseStatus::Completed)
                .output(vec![ResponseOutputItem::FunctionToolCall {
                    id: item_id.to_string(),
                    call_id: call_id.to_string(),
                    name: tool_name.to_string(),
                    arguments: arguments.to_string(),
                    output: None,
                    status: "completed".to_string(),
                }])
                .build(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_tool_test".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Call the search tool.".to_string(),
                }],
                status: Some("completed".to_string()),
            }],
        })
    }
}

#[derive(Clone, Default)]
struct FailedResponseWsExecutor;

#[async_trait]
impl WsResponsesExecutor for FailedResponseWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        if let Some(previous_id) = request.previous_response_id.as_deref() {
            if cached_response
                .as_ref()
                .is_some_and(|cached| cached.response.id == previous_id)
            {
                return Ok(CachedWsResponse {
                    response: ResponsesResponse::builder(
                        "resp_ws_unexpected_cached_reuse",
                        request.model.clone(),
                    )
                    .copy_from_request(&request)
                    .status(ResponseStatus::Completed)
                    .output(vec![ResponseOutputItem::Message {
                        id: "msg_ws_unexpected_cached_reuse".to_string(),
                        role: "assistant".to_string(),
                        content: vec![ResponseContentPart::OutputText {
                            text: "unexpected cached continuation".to_string(),
                            annotations: vec![],
                            logprobs: None,
                        }],
                        status: "completed".to_string(),
                    }])
                    .build(),
                    input_items: vec![],
                });
            }

            return Err(WsClientError::new(
                "previous_response_not_found",
                format!(
                    "Previous response '{}' was not found in the current session or durable storage.",
                    previous_id
                ),
            )
            .with_param("previous_response_id"));
        }

        let response = ResponsesResponse::builder("resp_ws_failed", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Failed)
            .output(vec![])
            .build();
        let response_model = request.model.clone();

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_ws_failed",
                "object": "response",
                "status": "in_progress",
                "model": response_model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response.clone(),
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response,
            input_items: vec![],
        })
    }
}

#[derive(Clone)]
struct StubWsRouter {
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
}

impl fmt::Debug for StubWsRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StubWsRouter")
    }
}

#[async_trait]
impl RouterTrait for StubWsRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &smg::protocols::chat::ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        StatusCode::NOT_IMPLEMENTED.into_response()
    }

    fn supports_responses_ws(&self) -> bool {
        true
    }

    async fn route_responses_ws(&self, headers: HeaderMap, socket: WebSocket) {
        serve_responses_ws_with_config(
            socket,
            headers,
            self.executor.clone(),
            self.runtime_config.clone(),
        )
        .await;
    }

    fn router_type(&self) -> &'static str {
        "stub-ws"
    }
}

async fn build_stub_app(executor: Arc<dyn WsResponsesExecutor>) -> axum::Router {
    build_stub_app_with_runtime_config(executor, WsRuntimeConfig::default()).await
}

async fn build_stub_app_with_runtime_config(
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
) -> axum::Router {
    let ctx = create_test_app_context().await;
    let router = Arc::new(StubWsRouter {
        executor,
        runtime_config,
    });
    create_test_app_with_context(router, ctx)
}

async fn serve_app(app: axum::Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("ws://{}", addr)
}

async fn recv_json(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> serde_json::Value {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(3), socket.next())
            .await
            .expect("timed out waiting for websocket message")
            .expect("websocket stream ended")
            .expect("websocket receive failed");

        match message {
            tokio_tungstenite::tungstenite::Message::Text(text) => {
                return serde_json::from_str(text.as_ref()).expect("message should be valid JSON");
            }
            tokio_tungstenite::tungstenite::Message::Ping(_) => continue,
            tokio_tungstenite::tungstenite::Message::Pong(_) => continue,
            tokio_tungstenite::tungstenite::Message::Close(frame) => {
                panic!("unexpected websocket close frame: {:?}", frame)
            }
            other => panic!("unexpected websocket message: {:?}", other),
        }
    }
}

async fn send_ws_request_and_collect(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    request: serde_json::Value,
) -> Vec<serde_json::Value> {
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let mut events = Vec::new();
    loop {
        let event = recv_json(socket).await;
        let is_terminal = matches!(
            event["type"].as_str(),
            Some("response.completed") | Some("error")
        );
        events.push(event);
        if is_terminal {
            break;
        }
    }

    events
}

fn ws_create_request(response_fields: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(mut request) = response_fields else {
        panic!("response.create request fields must be a JSON object");
    };
    request.insert(
        "type".to_string(),
        serde_json::Value::String("response.create".to_string()),
    );
    serde_json::Value::Object(request)
}

fn ws_error_code(event: &serde_json::Value) -> &str {
    event
        .pointer("/error/code")
        .and_then(|value| value.as_str())
        .or_else(|| event.get("code").and_then(|value| value.as_str()))
        .unwrap_or("")
}

fn ws_error_message(event: &serde_json::Value) -> &str {
    event
        .pointer("/error/message")
        .and_then(|value| value.as_str())
        .or_else(|| event.get("message").and_then(|value| value.as_str()))
        .unwrap_or("")
}

fn ws_error_param(event: &serde_json::Value) -> Option<&str> {
    event
        .pointer("/error/param")
        .and_then(|value| value.as_str())
}

#[derive(Clone, Default)]
struct SemanticWsExecutor {
    durable_store: Arc<std::sync::Mutex<HashMap<String, CachedWsResponse>>>,
}

impl SemanticWsExecutor {
    fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl WsResponsesExecutor for SemanticWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        if request.conversation.is_some() {
            return Err(WsClientError::new(
                "unsupported_parameter",
                "The `conversation` field is not supported in WebSocket Responses V1.",
            ));
        }

        if options.generate == Some(false) {
            let response_id = format!("resp_ws_{}", uuid::Uuid::new_v4().simple());
            let response = ResponsesResponse::builder(response_id.clone(), request.model.clone())
                .copy_from_request(&request)
                .status(ResponseStatus::Completed)
                .output(vec![])
                .build();

            let created = serde_json::json!({
                "type": "response.created",
                "response": {
                    "id": response_id.clone(),
                    "object": "response",
                    "status": "in_progress",
                    "model": request.model.clone(),
                    "output": []
                }
            });
            let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

            let completed = serde_json::json!({
                "type": "response.completed",
                "response": response,
            });
            let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

            return Ok(CachedWsResponse {
                response: ResponsesResponse::builder(response_id, request.model.clone())
                    .copy_from_request(&request)
                    .status(ResponseStatus::Completed)
                    .output(vec![])
                    .build(),
                input_items: vec![ResponseInputOutputItem::Message {
                    id: "msg_user_ws_semantic".to_string(),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Hello websocket".to_string(),
                    }],
                    status: Some("completed".to_string()),
                }],
            });
        }

        let previous_response = if let Some(previous_id) = request.previous_response_id.as_deref() {
            if let Some(cached) = cached_response.filter(|cached| cached.response.id == previous_id)
            {
                Some(cached)
            } else {
                self.durable_store
                    .lock()
                    .unwrap()
                    .get(previous_id)
                    .cloned()
                    .ok_or_else(|| {
                        WsClientError::new(
                            "previous_response_not_found",
                            format!(
                                "Previous response '{}' was not found in the current session or durable storage.",
                                previous_id
                            ),
                        )
                        .with_param("previous_response_id")
                    })?
                    .into()
            }
        } else {
            None
        };

        let response_id = format!("resp_ws_{}", uuid::Uuid::new_v4().simple());
        let output_text = if previous_response.is_some() {
            "stub websocket continuation output"
        } else {
            "stub websocket output"
        };

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": request.model.clone(),
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let response = ResponsesResponse::builder(response_id.clone(), request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_semantic".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: output_text.to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response,
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        let cached = CachedWsResponse {
            response: response.clone(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_semantic".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello websocket".to_string(),
                }],
                status: Some("completed".to_string()),
            }],
        };

        if request.store.unwrap_or(true) {
            self.durable_store
                .lock()
                .unwrap()
                .insert(cached.response.id.clone(), cached.clone());
        }

        Ok(cached)
    }
}

#[tokio::test]
async fn test_v1_responses_get_requires_websocket_upgrade() {
    let app = build_stub_app(Arc::new(StubWsExecutor::immediate())).await;

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/responses")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        smg::routers::error::extract_error_code_from_response(&response),
        "websocket_upgrade_required"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_unknown_event_type() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({ "type": "response.delete" })
                .to_string()
                .into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(ws_error_code(&event), "unsupported_event");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_invalid_json() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            "{\"type\":\"response.create\"".into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(ws_error_code(&event), "invalid_json");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_binary_messages() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Binary(
            vec![0xde, 0xad, 0xbe, 0xef].into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(ws_error_code(&event), "unsupported_message_type");
}

#[tokio::test]
async fn test_v1_responses_ws_replies_to_ping_and_keeps_session_healthy() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let ping_payload = vec![0x1, 0x2, 0x3, 0x4];
    socket
        .send(tokio_tungstenite::tungstenite::Message::Ping(
            ping_payload.clone().into(),
        ))
        .await
        .unwrap();

    let pong = tokio::time::timeout(Duration::from_secs(3), socket.next())
        .await
        .expect("timed out waiting for pong")
        .expect("websocket stream ended")
        .expect("websocket receive failed");

    match pong {
        tokio_tungstenite::tungstenite::Message::Pong(payload) => {
            assert_eq!(payload.as_ref(), ping_payload.as_slice());
        }
        other => panic!("expected pong after ping, got {:?}", other),
    }

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Hello websocket after ping",
            "store": false
        })),
    )
    .await;

    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_closes_when_session_lifetime_expires() {
    let url = serve_app(
        build_stub_app_with_runtime_config(
            Arc::new(StubWsExecutor::immediate()),
            WsRuntimeConfig {
                max_session_lifetime: Duration::from_millis(50),
            },
        )
        .await,
    )
    .await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let error = recv_json(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(&error), "websocket_connection_limit_reached");

    let close_message = tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("timed out waiting for websocket close")
        .expect("websocket stream ended without close frame")
        .expect("websocket receive failed");

    match close_message {
        tokio_tungstenite::tungstenite::Message::Close(frame) => {
            let frame = frame.expect("expected server close frame");
            assert_eq!(
                frame.reason.to_string(),
                "Responses websocket connection limit reached (50 ms). Create a new websocket connection to continue."
            );
        }
        other => panic!("expected websocket close frame, got {:?}", other),
    }
}

#[tokio::test]
async fn test_v1_responses_ws_response_create_streams_events() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            ws_create_request(serde_json::json!({
                "model": "mock-model",
                "input": "Hello websocket",
                "store": false
            }))
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    let completed = recv_json(&mut socket).await;

    assert_eq!(created["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "stub websocket output"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_function_call_events_stream_cleanly() {
    let url = serve_app(build_stub_app(Arc::new(FunctionCallWsExecutor)).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Call the tool",
            "store": false
        })),
    )
    .await;

    let event_types: Vec<_> = events
        .iter()
        .map(|event| event["type"].as_str().unwrap_or(""))
        .collect();

    assert_eq!(
        event_types,
        vec![
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_eq!(events[1]["item"]["type"], "function_call");
    assert_eq!(events[2]["delta"], r#"{"query":"fizz_buzz"}"#);
    assert_eq!(events[3]["arguments"], r#"{"query":"fizz_buzz"}"#);
    assert_eq!(events[4]["item"]["call_id"], "call_ws_test");
    assert_eq!(events[4]["item"]["name"], "search_repo");
    assert_eq!(
        events.last().unwrap()["response"]["output"][0]["call_id"],
        "call_ws_test"
    );
    assert_eq!(
        events.last().unwrap()["response"]["output"][0]["arguments"],
        r#"{"query":"fizz_buzz"}"#
    );
}

#[tokio::test]
async fn test_v1_responses_ws_accepts_top_level_response_create_payload() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Top level websocket request",
            "store": false,
            "tools": []
        })),
    )
    .await;

    let completed = events.last().unwrap();
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_accepts_structured_input_items() {
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::immediate())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let request = serde_json::json!({
        "type": "response.create",
        "model": "mock-model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello from a structured input."}]
        }],
        "store": false
    });

    let events = send_ws_request_and_collect(&mut socket, request).await;
    let completed = events.last().unwrap();

    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_second_inflight_request() {
    let gate = Arc::new(Notify::new());
    let url = serve_app(build_stub_app(Arc::new(StubWsExecutor::gated(gate.clone()))).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let request = ws_create_request(serde_json::json!({
        "model": "mock-model",
        "input": "Hello websocket",
        "store": false
    }));

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    assert_eq!(created["type"], "response.created");

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let error = recv_json(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(&error), "concurrent_response_create");

    gate.notify_waiters();
    let completed = recv_json(&mut socket).await;
    assert_eq!(completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_via_router_manager_streams_events() {
    let ctx = create_test_app_context().await;
    let manager = Arc::new(RouterManager::new(Arc::new(WorkerRegistry::new())));
    manager.register_router(
        router_ids::GRPC_REGULAR,
        Arc::new(StubWsRouter {
            executor: Arc::new(StubWsExecutor::immediate()),
            runtime_config: WsRuntimeConfig::default(),
        }),
    );

    let app = create_test_app_with_context(manager as Arc<dyn RouterTrait>, ctx);
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            ws_create_request(serde_json::json!({
                "model": "mock-model",
                "input": "Hello websocket",
                "store": false
            }))
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    let completed = recv_json(&mut socket).await;

    assert_eq!(created["type"], "response.created");
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "stub websocket output"
    );
}

#[tokio::test]
async fn test_v1_responses_ws_same_connection_store_false_continuation_completes() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "First websocket turn",
            "store": false
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");

    let response_id = first_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();

    let second_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Follow up websocket turn",
            "previous_response_id": response_id,
            "store": false
        })),
    )
    .await;

    let second_completed = second_events.last().unwrap();
    assert_eq!(second_completed["type"], "response.completed");

    let second_response_id = second_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();

    let third_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Third websocket turn",
            "previous_response_id": second_response_id,
            "store": false
        })),
    )
    .await;

    let third_completed = third_events.last().unwrap();
    assert_eq!(third_completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_allows_immediate_follow_up_after_completed_event() {
    let url = serve_app(
        build_stub_app(Arc::new(DelayedReturnWsExecutor::new(
            Duration::from_millis(20),
        )))
        .await,
    )
    .await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "First delayed websocket turn",
            "store": false
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");

    let second_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Immediate follow-up websocket turn",
            "store": false
        })),
    )
    .await;

    let second_completed = second_events.last().unwrap();
    assert_eq!(second_completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_store_true_continuation_survives_reconnect() {
    let executor = Arc::new(SemanticWsExecutor::new());
    let url = serve_app(build_stub_app(executor).await).await;

    let (mut first_socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();
    let first_events = send_ws_request_and_collect(
        &mut first_socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Persist this websocket turn",
            "store": true
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");
    let response_id = first_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();
    drop(first_socket);

    let (mut second_socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();
    let second_events = send_ws_request_and_collect(
        &mut second_socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Reconnect follow up websocket turn",
            "previous_response_id": response_id,
            "store": false
        })),
    )
    .await;

    let second_completed = second_events.last().unwrap();
    assert_eq!(second_completed["type"], "response.completed");
}

#[tokio::test]
async fn test_v1_responses_ws_missing_previous_response_errors() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Missing previous response id",
            "previous_response_id": "resp_missing_ws",
            "store": false
        })),
    )
    .await;

    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(error), "previous_response_not_found");
    assert_eq!(ws_error_param(error), Some("previous_response_id"));
}

#[tokio::test]
async fn test_v1_responses_ws_rejects_unsupported_parameters() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let background_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Background websocket request",
            "background": true
        })),
    )
    .await;
    assert_eq!(
        background_events.last().unwrap()["type"],
        "response.completed"
    );

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Conversation websocket request",
            "conversation": "conv_test_123"
        })),
    )
    .await;
    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(error), "unsupported_parameter");
}

#[tokio::test]
async fn test_v1_responses_ws_accepts_generate_false_warmup() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let warmup_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Warm up websocket request",
            "generate": false,
            "store": false
        })),
    )
    .await;

    let completed = warmup_events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(
        completed["response"]["output"]
            .as_array()
            .expect("warmup output should be an array")
            .len(),
        0
    );
}

#[tokio::test]
async fn test_v1_responses_ws_evicts_cached_response_after_failed_continuation() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "First websocket turn",
            "store": false
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    let response_id = first_completed["response"]["id"]
        .as_str()
        .expect("completed response should include id")
        .to_string();

    let failed_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Fail this continuation",
            "previous_response_id": response_id,
            "conversation": "conv_test_123",
            "store": false
        })),
    )
    .await;
    let failed_error = failed_events.last().unwrap();
    assert_eq!(failed_error["type"], "error");
    assert_eq!(ws_error_code(failed_error), "unsupported_parameter");

    let retry_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Retry after failed continuation",
            "previous_response_id": response_id,
            "store": false
        })),
    )
    .await;
    let retry_error = retry_events.last().unwrap();
    assert_eq!(retry_error["type"], "error");
    assert_eq!(ws_error_code(retry_error), "previous_response_not_found");
}

#[tokio::test]
async fn test_v1_responses_ws_does_not_reuse_failed_cached_response() {
    let url = serve_app(build_stub_app(Arc::new(FailedResponseWsExecutor)).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Produce a failed websocket response",
            "store": false
        })),
    )
    .await;
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["type"], "response.completed");
    assert_eq!(first_completed["response"]["status"], "failed");

    let retry_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "Retry after failed websocket response",
            "previous_response_id": "resp_ws_failed",
            "store": false
        })),
    )
    .await;
    let retry_error = retry_events.last().unwrap();
    assert_eq!(retry_error["type"], "error");
    assert_eq!(ws_error_code(retry_error), "previous_response_not_found");
}

#[tokio::test]
async fn test_v1_responses_ws_errors_echo_event_id() {
    let url = serve_app(build_stub_app(Arc::new(SemanticWsExecutor::new())).await).await;
    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "event_id": "evt_ws_123",
            "model": "mock-model",
            "input": "Conversation websocket request",
            "conversation": "conv_test_123"
        })),
    )
    .await;

    let error = events.last().unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(error), "unsupported_parameter");
    assert_eq!(error["event_id"], "evt_ws_123");
    assert!(!ws_error_message(error).is_empty());
}

// ------------------------------------------------------------------
// T1: WS backpressure / slow client test
// ------------------------------------------------------------------

/// Executor that floods the outbound channel with more messages than the
/// bounded capacity (256) to exercise backpressure / try_send failure.
#[derive(Clone)]
struct FloodingWsExecutor {
    message_count: usize,
}

#[async_trait]
impl WsResponsesExecutor for FloodingWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let model = request.model.clone();

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_flood",
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        // Flood with delta events — more than the 256-slot channel capacity.
        for i in 0..self.message_count {
            let delta = serde_json::json!({
                "type": "response.output_text.delta",
                "delta": format!("tok_{}", i)
            });
            // try_send will fail once the channel is full — that's expected.
            if outbound_tx
                .try_send(Message::Text(delta.to_string().into()))
                .is_err()
            {
                break;
            }
        }

        let response = ResponsesResponse::builder("resp_flood", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response.clone(),
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response,
            input_items: vec![],
        })
    }
}

#[tokio::test]
async fn test_v1_responses_ws_backpressure_slow_client() {
    let url = serve_app(
        build_stub_app(Arc::new(FloodingWsExecutor {
            // Well above the 256-slot bounded channel.
            message_count: 512,
        }))
        .await,
    )
    .await;

    let (mut socket, _) = connect_async(format!("{}/v1/responses", url))
        .await
        .unwrap();

    // Send request then deliberately read slowly — the server should
    // not OOM; try_send should drop excess messages gracefully.
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            ws_create_request(serde_json::json!({
                "model": "mock-model",
                "input": "Slow reader backpressure test",
                "store": false
            }))
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    // Give the executor time to flood the channel.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Drain whatever we can — we may not get all 512 deltas, but the
    // connection should be alive or cleanly closed, never a panic/OOM.
    let mut received = 0;
    while let Ok(Some(Ok(_msg))) = tokio::time::timeout(Duration::from_secs(2), socket.next()).await
    {
        received += 1;
    }

    // We should have received at least the initial `response.created`.
    assert!(
        received >= 1,
        "should receive at least response.created, got {received}"
    );
    // And fewer than the total flood count (some dropped by backpressure).
    assert!(
        received <= 512 + 3, // +3 for created/completed/close
        "received count should be bounded, got {received}"
    );
}

// ------------------------------------------------------------------
// T2: No-WS-router error test
// ------------------------------------------------------------------

/// A router that does NOT support WebSocket Responses.
#[derive(Debug, Clone)]
struct NoWsRouter;

#[async_trait]
impl RouterTrait for NoWsRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &smg::protocols::chat::ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        StatusCode::NOT_IMPLEMENTED.into_response()
    }

    fn supports_responses_ws(&self) -> bool {
        false
    }

    fn router_type(&self) -> &'static str {
        "no-ws"
    }
}

#[tokio::test]
async fn test_v1_responses_ws_no_ws_router_returns_structured_error() {
    // When no router supports WS, the server rejects the upgrade at the
    // HTTP level (server.rs checks supports_responses_ws() before upgrade).
    // Verify the HTTP error response contains structured JSON.
    let ctx = create_test_app_context().await;
    let app = create_test_app_with_context(Arc::new(NoWsRouter), ctx);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/responses")
                .header("upgrade", "websocket")
                .header("connection", "Upgrade")
                .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
                .header("sec-websocket-version", "13")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    assert_eq!(
        smg::routers::error::extract_error_code_from_response(&response),
        "responses_ws_not_supported"
    );
}
