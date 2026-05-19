// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shutdown-under-load gap-closer for M6.
//!
//! Pins the contract that an `axum::serve(...).with_graceful_shutdown(...)`
//! server, once asked to stop, finishes EVERY in-flight request before
//! the server future resolves. The sgl-router `main` relies on this so a
//! k8s SIGTERM doesn't truncate streaming completions.
//!
//! We don't bring up the full router here — that adds tokenizer / KV-
//! event / config plumbing without exercising more of `with_graceful_shutdown`
//! than a minimal axum::Router does. The point is to verify the wiring
//! mode `main.rs` uses.

use axum::{
    body::Body,
    http::Response,
    response::IntoResponse,
    routing::get,
    Router,
};
use bytes::Bytes;
use futures::stream;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

async fn slow_stream() -> impl IntoResponse {
    // 8 chunks * 60ms each ≈ ~480ms per request. Long enough that we
    // can race in many concurrent clients before the server starts
    // shutting down, short enough that the test wall-time stays low.
    let chunks: Vec<Result<Bytes, std::io::Error>> = (0..8)
        .map(|i| Ok::<Bytes, std::io::Error>(Bytes::from(format!("chunk-{i}\n"))))
        .collect();
    let s = stream::unfold(chunks.into_iter(), |mut it| async move {
        match it.next() {
            Some(c) => {
                tokio::time::sleep(Duration::from_millis(60)).await;
                Some((c, it))
            }
            None => None,
        }
    });
    let mut resp = Response::new(Body::from_stream(s));
    *resp.status_mut() = axum::http::StatusCode::OK;
    resp
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn graceful_shutdown_drains_inflight_requests() {
    let app = Router::new().route("/slow", get(slow_stream));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}/slow");

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    // Fire N concurrent slow clients. Each must complete with its full
    // body — never a truncated response or a connection-refused.
    const N: usize = 20;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        let c = client.clone();
        let u = url.clone();
        handles.push(tokio::spawn(async move {
            let resp = c.get(&u).send().await?;
            let bytes = resp.bytes().await?;
            Ok::<_, reqwest::Error>(bytes.len())
        }));
    }

    // Let every request grab a connection and start receiving chunks.
    // 100ms is comfortably past the first chunk (60ms delay each).
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Trigger shutdown. After this the server stops accepting new
    // connections but MUST wait for in-flight requests to finish.
    let started = Instant::now();
    shutdown_tx.send(()).unwrap();

    // Every in-flight request must complete with a non-empty body.
    let mut total_bytes = 0usize;
    for h in handles {
        let result = h.await.unwrap().expect("client completed");
        assert!(
            result > 0,
            "in-flight request must finish with a non-empty body during graceful shutdown"
        );
        total_bytes += result;
    }
    // Server task must exit cleanly once all in-flight requests drained.
    server.await.expect("server task joins after shutdown");

    let elapsed = started.elapsed();
    // The server should have waited at least until the slowest stream
    // finished (~7 * 60ms ≈ 420ms post-shutdown). Anything shorter would
    // imply we got short-cut response data, which is exactly what we're
    // testing against.
    assert!(
        elapsed >= Duration::from_millis(200),
        "graceful shutdown returned too fast ({elapsed:?}) — likely truncated streams"
    );
    assert!(
        total_bytes > 0,
        "expected non-zero body bytes across {N} clients"
    );
}

#[tokio::test]
async fn graceful_shutdown_with_no_inflight_returns_promptly() {
    // The complement: when nothing is in flight, shutdown resolves
    // immediately. Catches a regression where the server might hang
    // waiting on an idle connection pool.
    let app = Router::new().route("/slow", get(slow_stream));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    let started = Instant::now();
    shutdown_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("server resolves within 2s when idle")
        .expect("server task joined cleanly");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(1),
        "idle shutdown took too long: {elapsed:?}"
    );
}
