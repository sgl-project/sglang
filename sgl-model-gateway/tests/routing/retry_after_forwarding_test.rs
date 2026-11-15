//! Integration tests: the OpenAI gateway must forward the upstream
//! `Retry-After` header to clients (sgl-project/sglang#50516).
//!
//! `OpenAIRouter::route_chat()` reconstructs the response from the upstream
//! `reqwest` response and used to copy only `Content-Type`, so provider retry
//! guidance on 429/503 responses never reached clients.

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use axum::{
    body::Body,
    http::{header, HeaderMap, StatusCode},
    response::Response,
    routing::post,
    Router,
};
use serde_json::json;
use smg::{
    protocols::chat::ChatCompletionRequest,
    routers::{openai::OpenAIRouter, RouterTrait},
};
use tokio::net::TcpListener;

/// One scripted upstream reply.
#[derive(Clone)]
struct PlannedResponse {
    status: StatusCode,
    retry_after: Option<&'static str>,
    /// JSON body for non-streaming replies.
    json_body: serde_json::Value,
    /// Raw SSE payload; when set, the reply is served as `text/event-stream`.
    sse_body: Option<&'static str>,
}

fn error_body(status: StatusCode) -> serde_json::Value {
    json!({
        "error": {
            "message": format!("mock upstream error ({})", status.as_u16()),
            "type": "rate_limit_error",
            "code": status.as_str(),
        }
    })
}

fn success_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

const SSE_BODY: &str = "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1677652288,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello!\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";

/// Minimal mock OpenAI-compatible upstream serving a script of replies.
/// Once the script is exhausted, the last reply is repeated.
struct ScriptedUpstream {
    addr: SocketAddr,
    _handle: tokio::task::JoinHandle<()>,
    requests: Arc<AtomicUsize>,
}

impl ScriptedUpstream {
    async fn new(script: Vec<PlannedResponse>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let script = Arc::new(Mutex::new(script));
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_clone = Arc::clone(&requests);

        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let script = Arc::clone(&script);
                let requests = Arc::clone(&requests_clone);
                async move {
                    let n = requests.fetch_add(1, Ordering::SeqCst);
                    let planned = {
                        let script = script.lock().unwrap();
                        script
                            .get(n)
                            .cloned()
                            .unwrap_or_else(|| script.last().unwrap().clone())
                    };
                    let mut builder = Response::builder().status(planned.status);
                    if let Some(retry_after) = planned.retry_after {
                        builder = builder.header(header::RETRY_AFTER, retry_after);
                    }
                    if let Some(sse) = planned.sse_body {
                        builder
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .body(Body::from(sse))
                            .unwrap()
                    } else {
                        builder
                            .header(header::CONTENT_TYPE, "application/json")
                            .body(Body::from(planned.json_body.to_string()))
                            .unwrap()
                    }
                }
            }),
        );

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        Self {
            addr,
            _handle: handle,
            requests,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

fn chat_request(stream: bool) -> ChatCompletionRequest {
    let val = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": stream,
    });
    serde_json::from_value(val).unwrap()
}

struct RoutedOutcome {
    status: StatusCode,
    headers: HeaderMap,
    body: String,
    upstream_requests: usize,
}

async fn route_chat(script: Vec<PlannedResponse>, stream: bool) -> RoutedOutcome {
    let upstream = ScriptedUpstream::new(script).await;
    let ctx = crate::common::test_app::create_test_app_context().await;
    crate::common::test_app::register_external_worker(
        &ctx,
        &upstream.base_url(),
        Some(vec!["test-model"]),
    );
    let router = OpenAIRouter::new(&ctx).await.unwrap();

    let response = router.route_chat(None, &chat_request(stream), None).await;
    let status = response.status();
    let headers = response.headers().clone();
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body_bytes.to_vec()).unwrap();

    RoutedOutcome {
        status,
        headers,
        body,
        upstream_requests: upstream.requests.load(Ordering::SeqCst),
    }
}

fn retry_after(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
}

#[tokio::test]
async fn test_retry_after_forwarded_non_streaming_429() {
    let outcome = route_chat(
        vec![PlannedResponse {
            status: StatusCode::TOO_MANY_REQUESTS,
            retry_after: Some("30"),
            json_body: error_body(StatusCode::TOO_MANY_REQUESTS),
            sse_body: None,
        }],
        false,
    )
    .await;

    assert_eq!(outcome.status, StatusCode::TOO_MANY_REQUESTS);
    // The gateway retries 429s, so the upstream is hit more than once; the
    // final response must still carry the provider's retry guidance.
    assert!(outcome.upstream_requests > 1);
    assert_eq!(retry_after(&outcome.headers), Some("30"));
    assert!(outcome.body.contains("rate_limit_error"));
}

#[tokio::test]
async fn test_retry_after_forwarded_streaming_429() {
    let outcome = route_chat(
        vec![PlannedResponse {
            status: StatusCode::TOO_MANY_REQUESTS,
            retry_after: Some("30"),
            json_body: error_body(StatusCode::TOO_MANY_REQUESTS),
            sse_body: None,
        }],
        true,
    )
    .await;

    assert_eq!(outcome.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(retry_after(&outcome.headers), Some("30"));
}

#[tokio::test]
async fn test_retry_after_http_date_forwarded_on_503() {
    let outcome = route_chat(
        vec![PlannedResponse {
            status: StatusCode::SERVICE_UNAVAILABLE,
            retry_after: Some("Wed, 21 Oct 2026 07:28:00 GMT"),
            json_body: error_body(StatusCode::SERVICE_UNAVAILABLE),
            sse_body: None,
        }],
        false,
    )
    .await;

    assert_eq!(outcome.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        retry_after(&outcome.headers),
        Some("Wed, 21 Oct 2026 07:28:00 GMT")
    );
}

#[tokio::test]
async fn test_retry_after_forwarded_on_streaming_success() {
    let outcome = route_chat(
        vec![PlannedResponse {
            status: StatusCode::OK,
            retry_after: Some("5"),
            json_body: success_body(),
            sse_body: Some(SSE_BODY),
        }],
        true,
    )
    .await;

    assert_eq!(outcome.status, StatusCode::OK);
    assert_eq!(retry_after(&outcome.headers), Some("5"));
    assert!(outcome.body.contains("chat.completion.chunk"));
    assert!(outcome.body.contains("[DONE]"));
}

#[tokio::test]
async fn test_retry_after_absent_when_upstream_omits_it() {
    // Success without the header.
    let outcome = route_chat(
        vec![PlannedResponse {
            status: StatusCode::OK,
            retry_after: None,
            json_body: success_body(),
            sse_body: None,
        }],
        false,
    )
    .await;
    assert_eq!(outcome.status, StatusCode::OK);
    assert_eq!(retry_after(&outcome.headers), None);

    // Error without the header: no header must be invented.
    let outcome = route_chat(
        vec![PlannedResponse {
            status: StatusCode::TOO_MANY_REQUESTS,
            retry_after: None,
            json_body: error_body(StatusCode::TOO_MANY_REQUESTS),
            sse_body: None,
        }],
        false,
    )
    .await;
    assert_eq!(outcome.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(retry_after(&outcome.headers), None);
}

#[tokio::test]
async fn test_no_stale_retry_after_after_successful_retry() {
    // First attempt: 429 with Retry-After. The retry succeeds, so the final
    // response must not retain the intermediate response's header.
    let outcome = route_chat(
        vec![
            PlannedResponse {
                status: StatusCode::TOO_MANY_REQUESTS,
                retry_after: Some("30"),
                json_body: error_body(StatusCode::TOO_MANY_REQUESTS),
                sse_body: None,
            },
            PlannedResponse {
                status: StatusCode::OK,
                retry_after: None,
                json_body: success_body(),
                sse_body: None,
            },
        ],
        false,
    )
    .await;

    assert_eq!(outcome.status, StatusCode::OK);
    assert_eq!(outcome.upstream_requests, 2);
    assert_eq!(retry_after(&outcome.headers), None);
    assert!(outcome.body.contains("chat.completion"));
}
