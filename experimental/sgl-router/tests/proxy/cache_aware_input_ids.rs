// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end at the HTTP layer: the router tokenizes the prompt once at
//! ingress and forwards the ids to the engine as `input_ids` (so the engine
//! skips re-tokenizing the same prompt). Asserts the gating contract through
//! the real chat handler + a MockWorker backend:
//!
//! * A plain text chat request on the engine-equivalent chat-encoder path →
//!   the forwarded body carries `input_ids` AND retains `messages`.
//! * Requests carrying `tools` or a thinking override → also forwarded (the
//!   dsv4 encoder mirrors the engine's tool rendering and per-request
//!   thinking resolution, so the ids are engine-equivalent).
//! * A request with multimodal (non-text parts) content → `input_ids`
//!   omitted (the engine-side mm-item plumbing is unverified).
//!
//! The model id contains `deepseek-v4` so the tokenizer registry auto-attaches
//! the built-in V4 chat encoder — the engine-equivalent path — without a
//! template fixture.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, CacheAwareConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::engine_load::EngineLoadTable;
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::kv_events::{BlockSizeOracle, HashTree};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

const MODEL: &str = "deepseek-v4-tiny";

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: MODEL.into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::CacheAwareZmq,
            circuit_breaker: None,
            cache_aware: Some(CacheAwareConfig::default()),
            sticky: None,
            max_output_tokens: None,
            forward_input_ids: true,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
        retry: sgl_router::config::RetryConfig::default(),
    }
}

fn build_ctx(url: String) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(
        tokenizers.has_chat_encoder(MODEL),
        "deepseek-v4 model id must auto-attach the built-in chat encoder"
    );
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    });
    // Use the real loaded tokenizers (not the empty-registry test default) so
    // the cache-aware policy can tokenize at ingress.
    let policies = Arc::new(
        build_registry(
            &cfg,
            Arc::new(HashTree::new()),
            Arc::clone(&tokenizers),
            BlockSizeOracle::new(),
            EngineLoadTable::new(),
        )
        .unwrap(),
    );
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn send(ctx: Arc<AppContext>, body: Value) -> StatusCode {
    let app = build_router(ctx);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    app.oneshot(req).await.unwrap().status()
}

fn captured(mock: &MockWorker) -> Value {
    let b = mock
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("worker captured a request body");
    serde_json::from_slice(&b).expect("captured body is valid JSON")
}

#[tokio::test]
async fn plain_chat_forwards_input_ids_and_keeps_messages() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there friend"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    let ids = body.get("input_ids").and_then(|v| v.as_array());
    assert!(
        ids.is_some_and(|a| !a.is_empty()),
        "engine must receive non-empty input_ids; got {body}"
    );
    assert!(
        body.get("messages").is_some(),
        "messages must be retained alongside input_ids; got {body}"
    );
}

/// The ingress records `sgl_router_tokenize_seconds` for a request it
/// tokenizes. The registry unit tests cover the histogram in isolation; only
/// this asserts the chat handler is wired to it, so deleting the call site or
/// inverting its gate fails loudly here instead of passing silently.
#[tokio::test]
async fn tokenized_request_records_tokenize_seconds() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        Arc::clone(&ctx),
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there friend"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let m = ctx.metrics.render();
    assert!(
        m.contains(&format!(
            r#"sgl_router_tokenize_seconds_count{{model_id="{MODEL}"}} 1"#
        )),
        "ingress tokenize must be recorded once for a tokenized request; got:\n{m}"
    );
}

#[tokio::test]
async fn tool_request_forwards_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    let ids = body.get("input_ids").and_then(|v| v.as_array());
    assert!(
        ids.is_some_and(|a| !a.is_empty()),
        "tool requests on the dsv4 encoder must forward input_ids; got {body}"
    );
}

#[tokio::test]
async fn thinking_request_forwards_input_ids() {
    // `chat_template_kwargs.thinking` steers engine-side thinking mode, and
    // the dsv4 encoder resolves it the same way (`resolve_render_opts`), so
    // the ids are engine-equivalent and forward.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"thinking": true},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    let ids = body.get("input_ids").and_then(|v| v.as_array());
    assert!(
        ids.is_some_and(|a| !a.is_empty()),
        "thinking-mode requests on the dsv4 encoder must forward input_ids; got {body}"
    );
}

#[tokio::test]
async fn multimodal_request_omits_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    assert!(
        body.get("input_ids").is_none(),
        "multimodal requests must not forward input_ids; got {body}"
    );
}
