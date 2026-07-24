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
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Full;
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
    // `/server_info` reported --enable-http2 on a cleartext URL.
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
