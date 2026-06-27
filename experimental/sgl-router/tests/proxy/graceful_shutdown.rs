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
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
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
            min_priority: None,
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
