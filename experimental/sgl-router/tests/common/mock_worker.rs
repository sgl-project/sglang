// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Minimal axum mock of an SGLang HTTP worker for routing tests.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Json;
use bytes::Bytes;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

/// Headers captured from the most recent inbound request.
#[derive(Default)]
pub struct CapturedHeaders {
    pub seen: HashSet<String>,            // names (kept for backwards compat)
    pub headers: HashMap<String, String>, // name -> value (last write wins)
    pub last_body: Option<Bytes>,
}

#[derive(Clone)]
#[allow(dead_code)] // Only used by some test files; mock_worker is shared.
pub struct MockWorkerState {
    pub captured: Arc<Mutex<CapturedHeaders>>,
    pub stream_chunks: Arc<Vec<&'static str>>,
}

/// A running mock SGLang worker. Shuts down on Drop via the oneshot sender.
pub struct MockWorker {
    pub url: String,
    // Used in header_forwarding_test; not every test file reads captured headers.
    #[allow(dead_code)]
    pub captured: Arc<Mutex<CapturedHeaders>>,
    _shutdown: oneshot::Sender<()>,
}

impl MockWorker {
    /// Bind to a random port on 127.0.0.1 and start serving.
    ///
    /// `stream_chunks` are the raw SSE bytes returned when a streaming
    /// chat-completion request arrives.
    #[allow(dead_code)] // Only used by some test files.
    pub async fn start(stream_chunks: Vec<&'static str>) -> Self {
        let captured = Arc::new(Mutex::new(CapturedHeaders::default()));
        let state = MockWorkerState {
            captured: captured.clone(),
            stream_chunks: Arc::new(stream_chunks),
        };
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(chat))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            url,
            captured,
            _shutdown: tx,
        }
    }

    /// Bind to a random port and start a worker that accepts the request,
    /// sleeps for `delay`, then returns `200 OK` with an empty JSON object.
    /// Used to test router behaviour when the upstream wedges after accepting
    /// the TCP connection but before sending response headers.
    #[allow(dead_code)]
    pub async fn start_hanging(delay: Duration) -> Self {
        let captured = Arc::new(Mutex::new(CapturedHeaders::default()));

        #[derive(Clone)]
        struct HangState {
            captured: Arc<Mutex<CapturedHeaders>>,
            delay: Duration,
        }

        async fn hang_handler(
            State(s): State<HangState>,
            headers: HeaderMap,
            body: Bytes,
        ) -> Response<Body> {
            {
                let mut g = s.captured.lock().unwrap();
                g.last_body = Some(body.clone());
                for (k, v) in headers.iter() {
                    g.seen.insert(k.as_str().to_string());
                    if let Ok(val) = v.to_str() {
                        g.headers.insert(k.as_str().to_string(), val.to_string());
                    }
                }
            }
            tokio::time::sleep(s.delay).await;
            let mut r = Response::new(Body::from("{}"));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                HeaderName::from_static("content-type"),
                HeaderValue::from_static("application/json"),
            );
            r
        }

        let state = HangState {
            captured: captured.clone(),
            delay,
        };
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(hang_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            url,
            captured,
            _shutdown: tx,
        }
    }

    /// Bind to a random port and start a worker that ALWAYS returns the given
    /// HTTP status code and JSON body with `Content-Type: application/json`.
    /// Used to test router behaviour when the upstream returns an error.
    #[allow(dead_code)]
    pub async fn start_returning_error(status: StatusCode, body: Value) -> Self {
        let captured = Arc::new(Mutex::new(CapturedHeaders::default()));
        let body_arc = Arc::new(body.to_string());

        #[derive(Clone)]
        struct ErrorState {
            captured: Arc<Mutex<CapturedHeaders>>,
            body_str: Arc<String>,
            status: StatusCode,
        }

        async fn error_handler(
            State(s): State<ErrorState>,
            headers: HeaderMap,
            body: Bytes,
        ) -> Response<Body> {
            {
                let mut g = s.captured.lock().unwrap();
                g.last_body = Some(body);
                for (k, v) in headers.iter() {
                    g.seen.insert(k.as_str().to_string());
                    if let Ok(val) = v.to_str() {
                        g.headers.insert(k.as_str().to_string(), val.to_string());
                    }
                }
            }
            let mut r = Response::new(Body::from(s.body_str.as_ref().clone()));
            *r.status_mut() = s.status;
            r.headers_mut().insert(
                HeaderName::from_static("content-type"),
                HeaderValue::from_static("application/json"),
            );
            r
        }

        let state = ErrorState {
            captured: captured.clone(),
            body_str: body_arc,
            status,
        };
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(error_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            url,
            captured,
            _shutdown: tx,
        }
    }
}

#[allow(dead_code)] // Used by `MockWorker::start`, only some test files need it.
async fn chat(State(s): State<MockWorkerState>, headers: HeaderMap, body: Bytes) -> Response<Body> {
    {
        let mut g = s.captured.lock().unwrap();
        g.last_body = Some(body.clone());
        for (k, v) in headers.iter() {
            g.seen.insert(k.as_str().to_string());
            if let Ok(val) = v.to_str() {
                g.headers.insert(k.as_str().to_string(), val.to_string());
            }
        }
    }
    let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let streaming = v.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);
    if streaming {
        let chunks: Vec<_> = s
            .stream_chunks
            .iter()
            .map(|c| Ok::<_, std::io::Error>(Bytes::from(*c)))
            .collect();
        let body = Body::from_stream(futures::stream::iter(chunks));
        let mut r = Response::new(body);
        *r.status_mut() = StatusCode::OK;
        r.headers_mut().insert(
            HeaderName::from_static("content-type"),
            "text/event-stream".parse().unwrap(),
        );
        return r;
    }
    let resp = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": v["model"].as_str().unwrap_or("unknown"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }]
    });
    Json(resp).into_response()
}
