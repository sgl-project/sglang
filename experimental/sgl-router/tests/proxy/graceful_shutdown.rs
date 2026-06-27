// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Pins the contract that `axum::serve(...).with_graceful_shutdown(...)` —
//! exactly as wired in `src/main.rs` — drains every in-flight streaming
//! request through the **real** `build_router(ctx)` stack before the
//! server future resolves. A k8s SIGTERM must not truncate streaming
//! completions.
//!
//! Why route the test through the real router (chat handler + proxy +
//! SSE pump) rather than a synthetic `Router::new().route(...)`: a
//! truncation regression could live in `forward_streaming_to`'s
//! `bytes_stream_to_body` completion hook, in `chat::chat_completions`'
//! guards, or in the SSE pump's `tx.send().await` race — all of which
//! would be silently skipped by a synthetic-handler test.

use bytes::Bytes;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

const TEST_TIMEOUT: Duration = Duration::from_secs(15);

fn build_ctx_with_worker(worker_url: &str) -> Arc<AppContext> {
    let cfg = Config {
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("w1".into()),
            url: worker_url.to_string(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        })
        .expect("test worker accepted");
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    let ctx = AppContext::new(cfg, tokenizers, proxy, registry, policies);
    ctx.mark_ready();
    Arc::new(ctx)
}

/// Streaming chat-completions body the worker hands back chunk-by-chunk.
/// One ~60 ms delay per chunk × 8 chunks ≈ ~480 ms per request, long
/// enough that we can race in ~100 concurrent clients and trigger
/// shutdown while every stream is still mid-flight.
const SLOW_CHUNKS: &[&str] = &[
    "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"d\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"e\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"f\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"g\"}}]}\n\n",
    "data: [DONE]\n\n",
];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_drains_100_inflight_streaming_chat_completions() {
    // 1. Spin up a slow streaming worker.
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        SLOW_CHUNKS.to_vec(),
        Duration::from_millis(60),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);

    // 2. Serve the REAL `build_router(ctx)` on a random port with the
    //    `with_graceful_shutdown` wiring main.rs uses.
    let app = build_router(ctx);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}/v1/chat/completions");

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("axum::serve cleanly resolves on shutdown");
    });

    // 3. Fire 100 concurrent streaming clients.
    const N: usize = 100;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let body = serde_json::to_vec(&serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    }))
    .unwrap();

    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        let c = client.clone();
        let u = url.clone();
        let b = body.clone();
        handles.push(tokio::spawn(async move {
            let resp = c
                .post(&u)
                .header("content-type", "application/json")
                .body(b)
                .send()
                .await
                .map_err(|e| format!("client {i} send: {e}"))?;
            if !resp.status().is_success() {
                return Err(format!("client {i} non-2xx: {}", resp.status()));
            }
            let bytes: Bytes = resp
                .bytes()
                .await
                .map_err(|e| format!("client {i} body: {e}"))?;
            Ok::<Bytes, String>(bytes)
        }));
    }

    // 4. Let every request grab a connection and start receiving data.
    //    100 ms is past the first chunk delay (60 ms) for every stream
    //    but well before the last chunk fires.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 5. Trigger shutdown. axum stops accepting new connections but
    //    MUST drain the 100 already-attached streams.
    let started = Instant::now();
    shutdown_tx.send(()).unwrap();

    // 6. Every in-flight request must complete with a `[DONE]` terminator
    //    — proving the stream was NOT truncated by shutdown.
    let mut bytes_total: usize = 0;
    let mut done_count: usize = 0;
    for h in handles {
        let result = h
            .await
            .expect("client task panicked")
            .expect("client completed");
        bytes_total += result.len();
        let body_str = String::from_utf8_lossy(&result);
        if body_str.contains("data: [DONE]") {
            done_count += 1;
        }
    }
    // Server task must exit cleanly once all 100 in-flight requests drained.
    server.await.expect("server task joins after shutdown");

    let elapsed = started.elapsed();
    assert_eq!(
        done_count, N,
        "all {N} streams must terminate with `data: [DONE]` during graceful shutdown (got {done_count})"
    );
    assert!(
        bytes_total > 0,
        "expected non-zero body bytes across {N} clients"
    );
    // Drain MUST have taken at least ~400 ms (7 remaining chunks * 60ms).
    // A shorter wait implies the streams were truncated.
    assert!(
        elapsed >= Duration::from_millis(300),
        "graceful shutdown returned too fast ({elapsed:?}) — likely truncated streams"
    );
}

#[tokio::test]
async fn shutdown_with_no_inflight_returns_promptly() {
    // Complement of the load test: when nothing is in flight, the
    // shutdown future resolves quickly. Catches a regression where the
    // server might hang waiting on an idle connection pool.
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);
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

/// The readiness-drain contract: on SIGTERM the drain flips `/readyz` to 503
/// *while the server keeps serving* (`/healthz` stays 200, requests still
/// succeed), so the EndpointSlice controller deregisters this pod before it
/// stops accepting. Mirrors `src/main.rs`'s SIGTERM arm by driving the
/// shutdown future as "await the signal, then `drain_for_termination`" against
/// the real `build_router(ctx)` stack.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn readyz_flips_to_503_during_drain_while_still_serving() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    assert!(ctx.is_ready(), "ctx starts ready");

    let app = build_router(ctx.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // `sigterm_tx` stands in for SIGTERM delivery: the shutdown future awaits
    // it and then runs the same drain helper main.rs uses. The drain window is
    // long enough to observe the deregistered-but-still-serving state.
    let drain = Duration::from_millis(800);
    let ctx_for_shutdown = ctx.clone();
    let (sigterm_tx, sigterm_rx) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = sigterm_rx.await;
                sgl_router::server::shutdown::drain_for_termination(
                    &ctx_for_shutdown,
                    drain,
                    std::future::pending::<()>(),
                )
                .await;
            })
            .await
            .unwrap();
    });

    let client = reqwest::Client::new();
    let readyz = format!("http://{addr}/readyz");
    let healthz = format!("http://{addr}/healthz");

    // Before SIGTERM: ready + worker registered ⇒ /readyz 200.
    let pre = client.get(&readyz).send().await.unwrap();
    assert_eq!(
        pre.status(),
        reqwest::StatusCode::OK,
        "ready before SIGTERM"
    );

    // Fire SIGTERM, then observe mid-drain (200 ms is well inside the 800 ms
    // window): /readyz must be 503 but the server must still be serving.
    sigterm_tx.send(()).unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mid_ready = client.get(&readyz).send().await.unwrap();
    assert_eq!(
        mid_ready.status(),
        reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "/readyz must flip to 503 during the drain so k8s deregisters the pod",
    );
    let mid_health = client.get(&healthz).send().await.unwrap();
    assert_eq!(
        mid_health.status(),
        reqwest::StatusCode::OK,
        "the server must still be serving (accepting) during the drain window",
    );

    // A *real proxied* request (not just the local health handlers) must still
    // be accepted and served during the drain window — this is the request k8s
    // may still route before it observes the /readyz 503 and deregisters.
    let chat = format!("http://{addr}/v1/chat/completions");
    let body = serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let mid_chat = client.post(&chat).json(&body).send().await.unwrap();
    assert_eq!(
        mid_chat.status(),
        reqwest::StatusCode::OK,
        "a proxied chat request must still succeed during the drain window",
    );

    // The server resolves once the drain elapses.
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("server resolves after the drain elapses")
        .expect("server task joined cleanly");
}

/// After the drain elapses and the server future resolves, axum must have
/// stopped accepting: a *new* connection is refused. This is the property that
/// actually closes the rolling-update race — a regression where the socket
/// lingers or a second listener stays open would slip past the mid-drain
/// assertions above.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn new_connections_refused_after_drain_completes() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Short drain so the test is fast; the point is the post-resolve state.
    let drain = Duration::from_millis(100);
    let ctx_for_shutdown = ctx.clone();
    let (sigterm_tx, sigterm_rx) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = sigterm_rx.await;
                sgl_router::server::shutdown::drain_for_termination(
                    &ctx_for_shutdown,
                    drain,
                    std::future::pending::<()>(),
                )
                .await;
            })
            .await
            .unwrap();
    });

    // Server accepts before shutdown.
    let client = reqwest::Client::new();
    let healthz = format!("http://{addr}/healthz");
    assert_eq!(
        client.get(&healthz).send().await.unwrap().status(),
        reqwest::StatusCode::OK,
        "server serves before SIGTERM",
    );

    // Fire SIGTERM and wait for the drain + server future to fully resolve.
    sigterm_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("server resolves after the drain elapses")
        .expect("server task joined cleanly");

    // A fresh connection must now be refused — the listener is closed.
    let refused = reqwest::Client::new()
        .get(&healthz)
        .timeout(Duration::from_secs(2))
        .send()
        .await;
    assert!(
        refused.is_err(),
        "a new connection must be refused after the drain completes, got {refused:?}",
    );
}

/// End-to-end composition: SIGTERM → `drain_for_termination` (flip 503, pause)
/// → axum drains the already-attached streaming request to `[DONE]`. The
/// 100-inflight test above drives a bare oneshot shutdown future; this one runs
/// the *actual* main.rs drain path so a bug in the flip→pause→drain ordering
/// (e.g. truncating streams once /readyz flips) is caught.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn inflight_stream_completes_through_drain_for_termination() {
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        SLOW_CHUNKS.to_vec(),
        Duration::from_millis(60),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}/v1/chat/completions");

    let drain = Duration::from_millis(50);
    let ctx_for_shutdown = ctx.clone();
    let (sigterm_tx, sigterm_rx) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = sigterm_rx.await;
                sgl_router::server::shutdown::drain_for_termination(
                    &ctx_for_shutdown,
                    drain,
                    std::future::pending::<()>(),
                )
                .await;
            })
            .await
            .unwrap();
    });

    // Start one slow stream and let it grab the connection + first chunk.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let body = serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let inflight = tokio::spawn(async move {
        let resp = client.post(&url).json(&body).send().await.unwrap();
        assert!(
            resp.status().is_success(),
            "stream started: {}",
            resp.status()
        );
        resp.bytes().await.unwrap()
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Fire SIGTERM mid-stream: the drain must NOT truncate the in-flight stream.
    sigterm_tx.send(()).unwrap();

    let received = inflight.await.expect("client task joined");
    let body_str = String::from_utf8_lossy(&received);
    assert!(
        body_str.contains("data: [DONE]"),
        "the in-flight stream must terminate with `data: [DONE]` through the drain path, got: {body_str}",
    );
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("server resolves after in-flight stream drains")
        .expect("server task joined cleanly");
}
