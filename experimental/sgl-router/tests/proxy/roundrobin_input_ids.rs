// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `input_ids` forwarding is policy-independent: a load-only **round-robin**
//! policy on a chat-encoder model still forwards `input_ids` to the engine
//! (the engine-tokenization offload), even though it picks workers round-robin
//! and ignores the tokens for routing. Tokenization is gated on the model's
//! chat encoder at ingress, not on the policy.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
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
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

// deepseek-v4 id → the tokenizer registry auto-attaches the built-in V4 chat
// encoder, so the model has an engine-equivalent encode path.
const MODEL: &str = "deepseek-v4-tiny";

fn config(forward_input_ids: bool) -> Config {
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
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            max_output_tokens: None,
            default_top_k: None,
            default_top_p: None,
            pins: Default::default(),
            forward_input_ids,
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

fn build_ctx(url: String, forward_input_ids: bool) -> Arc<AppContext> {
    let cfg = config(forward_input_ids);
    // The handler tokenizes via the AppContext's registry (which carries the V4
    // encoder); the RoundRobin policy itself needs no tokenizer.
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(tokenizers.has_chat_encoder(MODEL));
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
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

/// A round-robin (load-only) policy still forwards `input_ids` on a
/// chat-encoder model — the offload is decoupled from routing.
#[tokio::test]
async fn round_robin_plain_chat_forwards_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), true);
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
        "round-robin must forward input_ids on a chat-encoder model; got {body}"
    );
    assert!(
        body.get("messages").is_some(),
        "messages must be retained alongside input_ids; got {body}"
    );
}

/// With the kill switch engaged (`forward_input_ids: false`, the
/// `--disable-input-ids-offload` state), a plain-chat request that would
/// otherwise be offloaded withholds `input_ids` end-to-end — the engine
/// tokenizes from the retained `messages`.
#[tokio::test]
async fn kill_switch_engaged_withholds_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), false);
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
    assert!(
        body.get("input_ids").is_none(),
        "the kill switch must withhold input_ids end-to-end; got {body}"
    );
    assert!(
        body.get("messages").is_some(),
        "messages must still reach the engine for tokenization; got {body}"
    );
}

/// A tool request on the dsv4 chat encoder ALSO forwards `input_ids` — the
/// built-in encoder renders tool schemas the engine's way, so the ids are
/// engine-equivalent — and that parity holds under every policy, load-only
/// round-robin included.
#[tokio::test]
async fn round_robin_tool_request_forwards_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), true);
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

/// A successful plain-chat forward on a chat-encoder model must NOT emit
/// `sgl_router_ingress_tokenize_errors_total` — that counter fires only when the
/// offload was expected but the encoder failed. A tool request on the same
/// model is a successful forward too (its ids are engine-equivalent on the dsv4
/// encoder), so it must not emit the error counter either.
#[tokio::test]
async fn successful_forward_does_not_emit_ingress_tokenize_error() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), true);

    let status = send(
        Arc::clone(&ctx),
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there friend"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let status = send(
        Arc::clone(&ctx),
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let m = ctx.metrics.render();
    assert!(
        m.contains("# TYPE sgl_router_ingress_tokenize_errors_total counter"),
        "the error counter family must be exposed; got:\n{m}",
    );
    assert!(
        !m.contains("sgl_router_ingress_tokenize_errors_total{"),
        "healthy forwards (and expected omissions) must not emit the error counter; got:\n{m}",
    );
}
