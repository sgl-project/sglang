// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end wiring of the cache-sim response tee: a chat request through the
//! full router must produce BOTH a `/ingest_ids` tee (the ingress prompt) and,
//! once the response completes, a `/extend_ids` tee carrying the LONGER
//! prompt+reply sequence — on the buffered path and on the SSE path. This is
//! the fix for the oracle hit rate missing the previous round's output tokens.

use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use http_body_util::BodyExt;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Mock cache-sim: records every `{model, input_ids}` body per endpoint,
/// always replies 204 (the real service's fire-and-forget contract).
struct MockCacheSim {
    url: String,
    /// path -> bodies received, in arrival order.
    received: Arc<Mutex<HashMap<String, Vec<Value>>>>,
}

impl MockCacheSim {
    async fn start() -> Self {
        type Received = Arc<Mutex<HashMap<String, Vec<Value>>>>;
        let received: Received = Arc::new(Mutex::new(HashMap::new()));
        let handler = |path: &'static str| {
            move |State(rec): State<Received>, body: axum::body::Bytes| async move {
                if let Ok(v) = serde_json::from_slice::<Value>(&body) {
                    rec.lock()
                        .unwrap()
                        .entry(path.to_string())
                        .or_default()
                        .push(v);
                }
                StatusCode::NO_CONTENT
            }
        };
        let app = axum::Router::new()
            .route("/ingest_ids", post(handler("/ingest_ids")))
            .route("/extend_ids", post(handler("/extend_ids")))
            .with_state(Arc::clone(&received));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { url, received }
    }

    /// Wait until `path` has received at least one body, returning them all;
    /// panics after the deadline (tees are async fire-and-forget).
    async fn await_bodies(&self, path: &str) -> Vec<Value> {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(bodies) = self.received.lock().unwrap().get(path) {
                if !bodies.is_empty() {
                    return bodies.clone();
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "cache-sim never received a POST on {path}",
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

// deepseek-v4 id → the tokenizer registry auto-attaches the built-in V4 chat
// encoder, so both tees run the engine-equivalent chat-encode path (the same
// one the next round's probe uses), independent of the routing policy.
const MODEL: &str = "deepseek-v4-tiny";

fn build_ctx(worker_url: &str, cache_sim_url: &str) -> Arc<AppContext> {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig {
            cache_sim_url: Some(cache_sim_url.to_string()),
            ..Default::default()
        },
        model: ModelConfig {
            id: MODEL.into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            max_output_tokens: None,
            sampling_overrides: Default::default(),
            forward_input_ids: true,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
        retry: sgl_router::config::RetryConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(tokenizers.has_chat_encoder(MODEL));
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: worker_url.to_string(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn chat_request(stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello world"}],
                "stream": stream
            }))
            .unwrap(),
        ))
        .unwrap()
}

fn ids_of(body: &Value) -> Vec<u64> {
    body["input_ids"]
        .as_array()
        .expect("input_ids array")
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect()
}

/// A non-streaming chat request in DSV4 thinking mode
/// (`chat_template_kwargs.thinking`), optionally carrying tools.
fn thinking_chat_request(with_tools: bool) -> Request<Body> {
    let mut body = serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello world"}],
        "chat_template_kwargs": {"thinking": true},
        "stream": false
    });
    if with_tools {
        body["tools"] = serde_json::json!([{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }]);
    }
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn non_streaming_response_tees_extend_ids_with_reply_tokens() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let sim = MockCacheSim::start().await;
    let ctx = build_ctx(&worker.url, &sim.url);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request(false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    // The client body must be byte-identical to the worker's response even
    // though the handler re-read it for the tee.
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "ok");

    let ingests = sim.await_bodies("/ingest_ids").await;
    let extends = sim.await_bodies("/extend_ids").await;
    assert_eq!(ingests.len(), 1);
    assert_eq!(extends.len(), 1);
    assert_eq!(extends[0]["model"], MODEL);
    let prompt_ids = ids_of(&ingests[0]);
    let extended_ids = ids_of(&extends[0]);
    // The extension is the prompt history PLUS the assistant reply ("ok"):
    // strictly longer than the ingress tee, and prompt-prefixed — the prefix
    // property is what makes the next round's chain-hashed blocks match.
    assert!(
        extended_ids.len() > prompt_ids.len(),
        "extend ids ({}) must be longer than prompt ids ({})",
        extended_ids.len(),
        prompt_ids.len(),
    );
    assert!(
        extended_ids.starts_with(&prompt_ids),
        "extend ids must extend the ingress prompt ids",
    );
}

#[tokio::test]
async fn streaming_response_tees_extend_ids_after_stream_completes() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"str\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"eamed\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    ])
    .await;
    let sim = MockCacheSim::start().await;
    let ctx = build_ctx(&worker.url, &sim.url);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request(true)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    // Drain the SSE body to completion so the pump finishes cleanly (an
    // undrained body is a client disconnect, which correctly suppresses the
    // extend tee).
    let _ = res.into_body().collect().await.unwrap();

    let ingests = sim.await_bodies("/ingest_ids").await;
    let extends = sim.await_bodies("/extend_ids").await;
    assert_eq!(ingests.len(), 1);
    assert_eq!(extends.len(), 1);
    assert_eq!(extends[0]["model"], MODEL);
    let prompt_ids = ids_of(&ingests[0]);
    let extended_ids = ids_of(&extends[0]);
    assert!(
        extended_ids.len() > prompt_ids.len(),
        "extend ids ({}) must include the streamed reply beyond the prompt ({})",
        extended_ids.len(),
        prompt_ids.len(),
    );
    assert!(
        extended_ids.starts_with(&prompt_ids),
        "extend ids must extend the ingress prompt ids",
    );
}

// DSV4 thinking mode WITHOUT tools: the next round's re-render diverges at the
// generation-prompt transition, so an extension could only ever insert dead
// blocks — the tee must skip it entirely (valve-pressure guard), while the
// ingress tee keeps measuring as usual.
#[tokio::test]
async fn thinking_without_tools_skips_extend_tee() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let sim = MockCacheSim::start().await;
    let ctx = build_ctx(&worker.url, &sim.url);
    let app = build_router(ctx);

    let res = app.oneshot(thinking_chat_request(false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();

    // The ingress tee still fires…
    let _ = sim.await_bodies("/ingest_ids").await;
    // …but no extension may follow. Give the async pipeline a beat to
    // (wrongly) deliver before asserting silence.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        sim.received
            .lock()
            .unwrap()
            .get("/extend_ids")
            .is_none_or(|v| v.is_empty()),
        "thinking-without-tools must not produce a (permanently unmatchable) extension",
    );
}

// DSV4 thinking mode WITH tools: history rendering keeps prior reasoning, so
// the extension is matchable and the tee must stay armed.
#[tokio::test]
async fn thinking_with_tools_still_tees_extend_ids() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let sim = MockCacheSim::start().await;
    let ctx = build_ctx(&worker.url, &sim.url);
    let app = build_router(ctx);

    let res = app.oneshot(thinking_chat_request(true)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();

    let ingests = sim.await_bodies("/ingest_ids").await;
    let extends = sim.await_bodies("/extend_ids").await;
    let prompt_ids = ids_of(&ingests[0]);
    let extended_ids = ids_of(&extends[0]);
    assert!(extended_ids.len() > prompt_ids.len());
    assert!(
        extended_ids.starts_with(&prompt_ids),
        "thinking+tools extension must extend the ingress prompt ids",
    );
}

#[tokio::test]
async fn upstream_error_response_does_not_tee_extend_ids() {
    let worker = crate::common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        serde_json::json!({"error": {"message": "boom"}}),
    )
    .await;
    let sim = MockCacheSim::start().await;
    let ctx = build_ctx(&worker.url, &sim.url);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request(false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);

    // The ingress tee still fires (the prompt was offered before dispatch)…
    let _ = sim.await_bodies("/ingest_ids").await;
    // …but no extension may arrive for an error response. Give the async
    // pipeline a beat to (wrongly) deliver before asserting silence.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        sim.received
            .lock()
            .unwrap()
            .get("/extend_ids")
            .is_none_or(|v| v.is_empty()),
        "an error response must never extend the cache-sim",
    );
}
