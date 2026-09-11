// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Binds protocol *resolution* to protocol *use*, across the PD split.
//!
//! Everywhere else the two halves are tested apart: the manager tests assert
//! what lands on the registry, and `h2c_forward.rs` passes a `WireProtocol` to
//! the proxy by hand. Nothing asserts that the protocol a worker resolved to is
//! the one its own forward actually uses — and in the PD arm of
//! `chat_completions` that join is three separate expressions, a
//! `prefill_protocol` captured before a `tokio::spawn` plus two live
//! `decode_worker.protocol()` reads. Passing the wrong one of those compiles,
//! and every other test in the suite stays green.
//!
//! So make the two workers disagree and let the wire enforce it: the prefill
//! mock speaks **HTTP/2 only**, the decode mock speaks **HTTP/1.1 only**, and
//! each is registered with the matching protocol. Any mix-up sends a client at
//! a server that cannot answer it. This is also why neither mock can be the
//! shared `MockWorker` — that one runs on `axum::serve`, which answers both
//! protocols and would pass no matter which was selected.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::server::conn::{http1, http2};
use hyper::service::service_fn;
use hyper::Response as HyperResponse;
use hyper_util::rt::{TokioExecutor, TokioIo};
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
use sgl_router::workers::{WireProtocol, WorkerRegistry};
use tokio::net::TcpListener;
use tower::ServiceExt;

/// A chat-completions response shaped enough for the handler to return 200.
fn chat_completion_json() -> Bytes {
    Bytes::from(
        serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "tiny",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

/// Serve HTTP/2 only, recording that a request body arrived. An HTTP/1.1
/// client cannot complete a request here — it never sends the HTTP/2 preface.
async fn spawn_h2_only_worker(hits: Arc<Mutex<usize>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let hits = Arc::clone(&hits);
            tokio::spawn(async move {
                let _ = http2::Builder::new(TokioExecutor::new())
                    .serve_connection(
                        TokioIo::new(stream),
                        service_fn(move |_req| {
                            let hits = Arc::clone(&hits);
                            async move {
                                *hits.lock().unwrap() += 1;
                                Ok::<_, Infallible>(HyperResponse::new(Full::new(
                                    chat_completion_json(),
                                )))
                            }
                        }),
                    )
                    .await;
            });
        }
    });
    format!("http://{addr}")
}

/// Serve HTTP/1.1 only. `http1::Builder` speaks no HTTP/2, so an h2c client
/// sending the preface cannot be served here.
async fn spawn_http1_only_worker(hits: Arc<Mutex<usize>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let hits = Arc::clone(&hits);
            tokio::spawn(async move {
                let _ = http1::Builder::new()
                    .serve_connection(
                        TokioIo::new(stream),
                        service_fn(move |_req| {
                            let hits = Arc::clone(&hits);
                            async move {
                                *hits.lock().unwrap() += 1;
                                Ok::<_, Infallible>(HyperResponse::new(Full::new(
                                    chat_completion_json(),
                                )))
                            }
                        }),
                    )
                    .await;
            });
        }
    });
    format!("http://{addr}")
}

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            decode_policy: Default::default(),
            bucket_config: None,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

/// Register both workers with their own resolved protocol, the way
/// `manager::register_one` does after introspecting `/model_info`.
fn build_ctx(prefill_url: String, decode_url: String) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());

    let prefill_id = WorkerId("p1".into());
    let decode_id = WorkerId("d1".into());
    // The two disagree on purpose. This is the state a mixed fleet reaches
    // when only some engines run with --enable-http2.
    registry
        .add_with_cb(
            WorkerSpec {
                id: prefill_id.clone(),
                url: prefill_url,
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            None,
            WireProtocol::H2c,
        )
        .unwrap();
    registry
        .add_with_cb(
            WorkerSpec {
                id: decode_id.clone(),
                url: decode_url,
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
            None,
            WireProtocol::Http1,
        )
        .unwrap();

    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            }))
            .unwrap(),
        ))
        .unwrap()
}

/// Each PD leg must forward over the protocol *its own* worker resolved.
///
/// The decode leg is awaited, so a protocol mix-up there fails the response
/// outright. The prefill leg is detached, so its mix-up shows up as a body that
/// never arrives — poll for it rather than reading once.
#[tokio::test]
async fn pd_legs_each_use_their_own_workers_protocol() {
    let prefill_hits = Arc::new(Mutex::new(0usize));
    let decode_hits = Arc::new(Mutex::new(0usize));
    let prefill_url = spawn_h2_only_worker(Arc::clone(&prefill_hits)).await;
    let decode_url = spawn_http1_only_worker(Arc::clone(&decode_hits)).await;

    let app = build_router(build_ctx(prefill_url, decode_url));
    let res = app.oneshot(chat_request()).await.unwrap();

    // Decode answered, so the decode leg used HTTP/1.1 — had it been handed the
    // prefill worker's H2c, this HTTP/1.1-only server could not have replied.
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "decode leg must forward over the decode worker's own protocol (HTTP/1.1)",
    );
    let body = res.into_body().collect().await.unwrap().to_bytes();
    assert!(
        !body.is_empty(),
        "decode response body must reach the client",
    );

    // Prefill is spawn-and-forget; it races the response back to the client.
    let reached = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if *prefill_hits.lock().unwrap() > 0 {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        reached.is_ok(),
        "prefill leg must forward over the prefill worker's own protocol (h2c); \
         an HTTP/1.1 client cannot reach this HTTP/2-only worker",
    );
    assert_eq!(
        *decode_hits.lock().unwrap(),
        1,
        "decode worker should be hit exactly once",
    );
}
