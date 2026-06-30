// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Minimal axum mock of an SGLang HTTP worker for routing tests.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use bytes::Bytes;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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
    /// Every `/abort_request` POST this worker received, in arrival order.
    /// Populated by every axum-based `start_*` variant; not every test file
    /// reads it. `start_returning_partial_body` (raw TCP, no path routing)
    /// never populates it.
    #[allow(dead_code)]
    pub abort_log: Arc<Mutex<Vec<Value>>>,
    _shutdown: oneshot::Sender<()>,
}

/// `/abort_request` route shared by every axum-based `MockWorker::start_*`
/// variant: appends the POSTed JSON body to `log` and answers 200 OK. Generic
/// over `S` (the per-variant axum state type) because the handler closure
/// extracts no `State<S>` — the same shape as `serve_tiny_server_info` below.
#[allow(dead_code)] // shared across all axum variants
fn abort_request_route<S>(log: Arc<Mutex<Vec<Value>>>) -> axum::routing::MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    post(move |Json(body): Json<Value>| async move {
        log.lock().unwrap().push(body);
        StatusCode::OK
    })
}

impl MockWorker {
    /// Bind to a random port on 127.0.0.1 and start serving.
    ///
    /// `stream_chunks` are the raw SSE bytes returned when a streaming
    /// chat-completion request arrives.
    #[allow(dead_code)] // Only used by some test files.
    pub async fn start(stream_chunks: Vec<&'static str>) -> Self {
        let captured = Arc::new(Mutex::new(CapturedHeaders::default()));
        let abort_log: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let state = MockWorkerState {
            captured: captured.clone(),
            stream_chunks: Arc::new(stream_chunks),
        };
        // /server_info advertises served_model_name="tiny" so the
        // worker-manager introspect step resolves model_ids for the
        // "tiny" model the tests register a tokenizer + policy under.
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/server_info", get(serve_tiny_server_info))
            .route("/abort_request", abort_request_route(abort_log.clone()))
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
            abort_log,
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
        let abort_log: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));

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
            .route("/server_info", get(serve_tiny_server_info))
            .route("/abort_request", abort_request_route(abort_log.clone()))
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
            abort_log,
            _shutdown: tx,
        }
    }

    /// Bind to a random port and start a worker that streams `chunks` with a
    /// fixed `delay` between each chunk.  Used to test that load guards survive
    /// the full body lifetime for streaming responses.
    #[allow(dead_code)]
    pub async fn start_slow_stream(chunks: Vec<&'static str>, delay: Duration) -> Self {
        let captured = Arc::new(Mutex::new(CapturedHeaders::default()));
        let abort_log: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));

        #[derive(Clone)]
        struct SlowState {
            captured: Arc<Mutex<CapturedHeaders>>,
            chunks: Arc<Vec<&'static str>>,
            delay: Duration,
        }

        async fn slow_chat(
            State(s): State<SlowState>,
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
            let chunks = s.chunks.clone();
            let delay = s.delay;
            // Stream chunks via a channel, sleeping between each send.
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            tokio::spawn(async move {
                for chunk in chunks.iter() {
                    tokio::time::sleep(delay).await;
                    if tx.send(Ok(Bytes::from(*chunk))).await.is_err() {
                        break;
                    }
                }
            });
            let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
            let mut r = Response::new(body);
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                HeaderName::from_static("content-type"),
                "text/event-stream".parse().unwrap(),
            );
            r
        }

        let state = SlowState {
            captured: captured.clone(),
            chunks: Arc::new(chunks),
            delay,
        };
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(slow_chat))
            .route("/server_info", get(serve_tiny_server_info))
            .route("/abort_request", abort_request_route(abort_log.clone()))
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
            abort_log,
            _shutdown: tx,
        }
    }

    /// Bind to a raw TCP listener and start a worker that writes a status
    /// line + headers with a large declared `Content-Length`, then writes
    /// only `partial_body_bytes` of body before closing the connection.
    ///
    /// Used to test router behaviour when the upstream replies with a status
    /// but drops the connection mid-body. We can't build this with axum
    /// directly (it owns the response lifecycle); raw TCP gives us frame-level
    /// control to short-write the body and close.
    ///
    /// NOTE: unlike the axum-based variants, this helper does NOT serve
    /// `/server_info` (one-shot raw-TCP accept, no path routing). Callers
    /// that wire this through `spawn_discovery` will see introspect fail
    /// with empty `model_ids`. All current callers inject the worker via
    /// `registry.add()` directly, which bypasses introspect.
    #[allow(dead_code)]
    pub async fn start_returning_partial_body(
        status: StatusCode,
        partial_body_bytes: &'static [u8],
    ) -> Self {
        let captured = Arc::new(Mutex::new(CapturedHeaders::default()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let url = format!("http://{addr}");
        let (tx, mut rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            // Accept one connection (or exit on shutdown).
            tokio::select! {
                _ = &mut rx => (),
                accept = listener.accept() => {
                    let (mut sock, _) = match accept {
                        Ok(v) => v,
                        Err(_) => return,
                    };
                    // Drain the request bytes until we see end-of-headers
                    // (`\r\n\r\n`). We deliberately do NOT fully consume the
                    // request body — the router has already sent it before
                    // awaiting our response, and we want to write the
                    // truncated response promptly.
                    let mut buf = [0u8; 4096];
                    let mut acc: Vec<u8> = Vec::new();
                    while !acc.windows(4).any(|w| w == b"\r\n\r\n") {
                        let n = match sock.read(&mut buf).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => n,
                        };
                        acc.extend_from_slice(&buf[..n]);
                        if acc.len() > 64 * 1024 {
                            // Defensive: don't loop forever if the request
                            // never produces a header terminator.
                            break;
                        }
                    }
                    // Write a response with a Content-Length larger than the
                    // bytes we will actually write, then drop the socket
                    // before the body completes.
                    let declared_len = partial_body_bytes.len() + 1024;
                    let head = format!(
                        "HTTP/1.1 {status_u16} {phrase}\r\n\
                         content-type: application/json\r\n\
                         content-length: {declared_len}\r\n\
                         connection: close\r\n\
                         \r\n",
                        status_u16 = status.as_u16(),
                        phrase = status.canonical_reason().unwrap_or("OK"),
                    );
                    if sock.write_all(head.as_bytes()).await.is_err() {
                        return;
                    }
                    if sock.write_all(partial_body_bytes).await.is_err() {
                        return;
                    }
                    // Flush, then drop — the client should see content-length
                    // mismatch as a transport-level body read failure.
                    let _ = sock.flush().await;
                    drop(sock);
                }
            }
        });
        Self {
            url,
            captured,
            abort_log: Arc::new(Mutex::new(Vec::new())),
            _shutdown: tx,
        }
    }

    /// Bind to a random port and start a worker that ALWAYS returns the given
    /// HTTP status code and JSON body with `Content-Type: application/json`.
    /// Used to test router behaviour when the upstream returns an error.
    #[allow(dead_code)]
    pub async fn start_returning_error(status: StatusCode, body: Value) -> Self {
        let captured = Arc::new(Mutex::new(CapturedHeaders::default()));
        let abort_log: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
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
            .route("/server_info", get(serve_tiny_server_info))
            .route("/abort_request", abort_request_route(abort_log.clone()))
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
            abort_log,
            _shutdown: tx,
        }
    }
}

/// Stateless `/server_info` handler shared by every axum-based
/// `MockWorker::start_*` variant. Advertising `served_model_name="tiny"`
/// lets the worker manager's introspect step resolve `model_ids` for any
/// variant that flows through `spawn_discovery`, instead of burning 3 ×
/// `SERVER_INFO_TIMEOUT` of retries before registering with empty
/// `model_ids`. Adding it unconditionally is cheaper than tracking which
/// variants do or don't get introspected.
#[allow(dead_code)] // shared across all axum variants
async fn serve_tiny_server_info() -> Json<Value> {
    Json(serde_json::json!({"served_model_name": "tiny"}))
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
