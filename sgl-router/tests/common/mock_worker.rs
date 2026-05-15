// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Minimal axum mock of an SGLang HTTP worker for routing tests.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Json;
use bytes::Bytes;
use serde_json::Value;
use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

/// Headers captured from the most recent inbound request.
#[derive(Default)]
pub struct CapturedHeaders {
    pub seen: HashSet<String>,
    pub last_body: Option<Bytes>,
}

#[derive(Clone)]
pub struct MockWorkerState {
    pub captured: Arc<Mutex<CapturedHeaders>>,
    pub stream_chunks: Arc<Vec<&'static str>>,
}

/// A running mock SGLang worker. Shuts down on Drop via the oneshot sender.
pub struct MockWorker {
    pub url: String,
    pub captured: Arc<Mutex<CapturedHeaders>>,
    _shutdown: oneshot::Sender<()>,
}

impl MockWorker {
    /// Bind to a random port on 127.0.0.1 and start serving.
    ///
    /// `stream_chunks` are the raw SSE bytes returned when a streaming
    /// chat-completion request arrives.
    pub async fn start(stream_chunks: Vec<&'static str>) -> Self {
        let captured = Arc::new(Mutex::new(CapturedHeaders::default()));
        let state = MockWorkerState {
            captured: captured.clone(),
            stream_chunks: Arc::new(stream_chunks),
        };
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(chat))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
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

async fn chat(
    State(s): State<MockWorkerState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    {
        let mut g = s.captured.lock().unwrap();
        g.last_body = Some(body.clone());
        for (k, _v) in headers.iter() {
            g.seen.insert(k.as_str().to_string());
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
