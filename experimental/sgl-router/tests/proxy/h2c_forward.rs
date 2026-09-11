// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cleartext-HTTP/2 (h2c) forwarding tests for [`Proxy`].
//!
//! These spin up an **HTTP/2-only** server (hyper's `http2::Builder`, no
//! HTTP/1.1 path) and drive `forward_json_to` against it with an explicit
//! per-request [`WireProtocol`]. Together the two tests prove what a value-only
//! unit test cannot: that the proxy's h2c client genuinely speaks HTTP/2 on the
//! wire when a request selects [`WireProtocol::H2c`] (not just that the value
//! was threaded through), and that the HTTP/1.1 client cannot reach an h2c
//! worker — so a regression that dropped the `http2` Cargo feature, removed
//! `http2_prior_knowledge()`, or mismatched `build_client`'s arms would fail
//! here rather than slip through. They also pin the per-worker design: the
//! protocol is chosen per `forward_json_to` call, not committed fleet-wide.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use hyper::server::conn::http2;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use sgl_router::health::circuit_breaker::CircuitBreaker;
use sgl_router::proxy::Proxy;
use sgl_router::workers::WireProtocol;
use tokio::net::TcpListener;

/// Serve HTTP/2 only. `http2::Builder::serve_connection` speaks the HTTP/2
/// framing protocol with no HTTP/1.1 fallback, so a client that does not
/// send the HTTP/2 connection preface cannot complete a request. Returns the
/// base URL; the accept loop is aborted when the test runtime shuts down.
async fn spawn_h2c_only_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let _ = http2::Builder::new(TokioExecutor::new())
                    .serve_connection(
                        TokioIo::new(stream),
                        service_fn(|_req: Request<hyper::body::Incoming>| async {
                            Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(
                                b"{\"ok\":true}",
                            ))))
                        }),
                    )
                    .await;
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn h2c_client_reaches_http2_only_worker() {
    let url = spawn_h2c_only_server().await;
    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = CircuitBreaker::new();

    // Select h2c per request, as the chat handler does for a worker whose
    // `/model_info` reported --enable-http2 on a cleartext URL.
    let resp = proxy
        .forward_json_to(
            &url,
            WireProtocol::H2c,
            &breaker,
            "/v1/chat/completions",
            &axum::http::HeaderMap::new(),
            Bytes::from_static(b"{}"),
        )
        .await
        .expect("h2c client must reach an HTTP/2-only worker");
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn http1_client_cannot_reach_http2_only_worker() {
    let url = spawn_h2c_only_server().await;
    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = CircuitBreaker::new();

    // The HTTP/1.1 client never sends the HTTP/2 preface, so the h2c-only
    // server cannot serve it. This is what makes selecting the protocol
    // meaningful: Http1 and H2c are different protocols on the wire, not the
    // same client pointed at the same endpoint.
    let res = proxy
        .forward_json_to(
            &url,
            WireProtocol::Http1,
            &breaker,
            "/v1/chat/completions",
            &axum::http::HeaderMap::new(),
            Bytes::from_static(b"{}"),
        )
        .await;
    assert!(
        res.is_err(),
        "HTTP/1.1 client must not complete a request against an HTTP/2-only worker, got {res:?}",
    );
}

/// Serve HTTP/2 only, answering with a multi-frame SSE body. Each `data:`
/// chunk is its own HTTP/2 DATA frame, which is the framing the real engine
/// produces for a streaming generation.
async fn spawn_h2c_only_sse_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let _ = http2::Builder::new(TokioExecutor::new())
                    .serve_connection(
                        TokioIo::new(stream),
                        service_fn(|_req: Request<hyper::body::Incoming>| async {
                            let chunks: Vec<Result<Frame<Bytes>, Infallible>> = vec![
                                Ok(Frame::data(Bytes::from_static(b"data: {\"i\":0}\n\n"))),
                                Ok(Frame::data(Bytes::from_static(b"data: {\"i\":1}\n\n"))),
                                Ok(Frame::data(Bytes::from_static(b"data: [DONE]\n\n"))),
                            ];
                            let body = StreamBody::new(futures::stream::iter(chunks));
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .header("content-type", "text/event-stream")
                                    .body(body)
                                    .unwrap(),
                            )
                        }),
                    )
                    .await;
            });
        }
    });
    format!("http://{addr}")
}

/// The streaming forward — the path every chat generation actually takes — must
/// work over h2c, and must deliver the whole multi-frame body.
///
/// `forward_streaming_to` differs from `forward_json_to` in ways HTTP/2 reaches
/// differently: it omits the request timeout, and it hands `bytes_stream()` to
/// the SSE pump rather than buffering. A break confined to SSE-over-h2c — a
/// truncated body, a stall, a mid-stream reset surfacing as success — leaves
/// the buffered tests green, so cover it directly.
#[tokio::test]
async fn h2c_client_streams_sse_from_http2_only_worker() {
    let url = spawn_h2c_only_sse_server().await;
    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::new());

    let first_byte_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&first_byte_seen);

    let resp = proxy
        .forward_streaming_to(
            &url,
            WireProtocol::H2c,
            &breaker,
            "/v1/chat/completions",
            &axum::http::HeaderMap::new(),
            Bytes::from_static(b"{}"),
            None,
            Some(Box::new(move || {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            })),
        )
        .await
        .expect("h2c client must stream from an HTTP/2-only worker");
    assert_eq!(resp.status(), 200);

    let body = resp
        .into_body()
        .collect()
        .await
        .expect("streaming body must complete over h2c")
        .to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();

    // Every frame arrives, in order — not just the first.
    assert!(text.contains("\"i\":0"), "missing first chunk: {text}");
    assert!(text.contains("\"i\":1"), "missing second chunk: {text}");
    assert!(
        text.contains("[DONE]"),
        "stream truncated before DONE: {text}"
    );
    assert!(
        first_byte_seen.load(std::sync::atomic::Ordering::SeqCst),
        "on_first_byte must fire for an h2c stream (TTFT accounting depends on it)",
    );
}
