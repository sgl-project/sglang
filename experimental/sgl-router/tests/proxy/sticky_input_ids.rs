// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Tokenize-once at ingress under the STICKY policy. The engine-tokenization
//! offload (`input_ids` forwarding) is a property of the MODEL — does it have a
//! chat encoder? — not of the routing policy, so a sticky-routed request on a
//! chat-encoder model must forward `input_ids` exactly like cache-aware does,
//! while still pinning sessions O(1) by header.
//!
//! Asserts through the real chat handler + `MockWorker` backends:
//!
//! * A plain text chat request forwards `input_ids` AND retains `messages`,
//!   even though sticky never consults the tokens for routing.
//! * A request carrying `tools` / multimodal content omits `input_ids` — the
//!   same safe-to-forward predicate applies regardless of policy.
//! * Same-session-header requests still pin to a single worker (O(1) sticky
//!   routing is unchanged by the added tokenization).
//!
//! The model id contains `deepseek-v4` so the tokenizer registry auto-attaches
//! the built-in V4 chat encoder — the engine-equivalent path — without a
//! template fixture.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig, StickyConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
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
const HEADER: &str = "x-sgl-routing-key";

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: MODEL.into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::Sticky,
            circuit_breaker: None,
            cache_aware: None,
            // Push eviction far out so the background sweeper never fires
            // mid-test; round-robin fallback for the initial pin of a key.
            sticky: Some(StickyConfig {
                header_name: HEADER.to_string(),
                fallback_policy: PolicyKind::RoundRobin,
                idle_secs: 3600,
                eviction_interval_secs: 3600,
            }),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

/// Build an `AppContext` running the sticky policy over the given workers.
/// The tokenizer registry is loaded from config (real tiny tokenizer + the
/// auto-attached V4 chat encoder) so the ingress can tokenize — the sticky
/// policy itself holds no tokenizer.
fn build_ctx(worker_urls: &[String]) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(
        tokenizers.has_chat_encoder(MODEL),
        "deepseek-v4 model id must auto-attach the built-in chat encoder"
    );
    let registry = Arc::new(WorkerRegistry::default());
    for (i, url) in worker_urls.iter().enumerate() {
        let _ = registry.add(WorkerSpec {
            id: WorkerId(format!("w{i}")),
            url: url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(MODEL.into())],
            bootstrap_port: None,
            min_priority: None,
        });
    }
    // Sticky needs no cache-aware deps, so the defaults registry is fine — the
    // ingress tokenizes via `ctx.tokenizers`, not the policy.
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn send(ctx: Arc<AppContext>, routing_key: &str, body: Value) -> StatusCode {
    let app = build_router(ctx);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(HEADER, routing_key)
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
async fn sticky_plain_chat_forwards_input_ids_and_keeps_messages() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(std::slice::from_ref(&mock.url));
    let status = send(
        ctx,
        "alice",
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
        "sticky-routed chat must still forward non-empty input_ids; got {body}"
    );
    assert!(
        body.get("messages").is_some(),
        "messages must be retained alongside input_ids; got {body}"
    );
}

#[tokio::test]
async fn sticky_tool_request_omits_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(std::slice::from_ref(&mock.url));
    let status = send(
        ctx,
        "alice",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    assert!(
        body.get("input_ids").is_none(),
        "tool requests must not forward input_ids even under sticky; got {body}"
    );
}

#[tokio::test]
async fn sticky_thinking_request_omits_input_ids() {
    // `chat_template_kwargs` steers engine-side thinking mode the router's
    // encoder renders in the default mode only — the safe-to-forward predicate
    // is policy-independent, so sticky must omit ids here too.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(std::slice::from_ref(&mock.url));
    let status = send(
        ctx,
        "alice",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": true},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    assert!(
        body.get("input_ids").is_none(),
        "thinking-mode requests must not forward input_ids under sticky; got {body}"
    );
}

#[tokio::test]
async fn sticky_multimodal_request_omits_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(std::slice::from_ref(&mock.url));
    let status = send(
        ctx,
        "alice",
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
        "multimodal requests must not forward input_ids under sticky; got {body}"
    );
}

/// Routing is unchanged: same session header pins every request to one worker
/// (O(1) sticky), even though the ingress now also tokenizes. With two
/// backends, all same-key requests must land on exactly one of them.
#[tokio::test]
async fn sticky_pins_session_by_header_with_tokenization_on() {
    let w0 = MockWorker::start(vec![]).await;
    let w1 = MockWorker::start(vec![]).await;
    let ctx = build_ctx(&[w0.url.clone(), w1.url.clone()]);
    let app = build_router(ctx.clone());

    const N: usize = 5;
    for _ in 0..N {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(HEADER, "alice")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "hello there friend"}],
                }))
                .unwrap(),
            ))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    // Exactly one worker captured a body — all same-key requests pinned to it.
    let w0_hit = w0.captured.lock().unwrap().last_body.is_some();
    let w1_hit = w1.captured.lock().unwrap().last_body.is_some();
    assert!(
        w0_hit ^ w1_hit,
        "same routing key must pin to exactly one worker (w0_hit={w0_hit}, w1_hit={w1_hit})"
    );

    // And the pinned worker still received forwarded input_ids — the offload
    // and the pin coexist.
    let pinned = if w0_hit { &w0 } else { &w1 };
    let body = captured(pinned);
    assert!(
        body.get("input_ids")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty()),
        "the pinned worker must receive forwarded input_ids; got {body}"
    );
}
