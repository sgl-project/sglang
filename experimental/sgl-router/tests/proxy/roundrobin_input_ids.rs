// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `input_ids` forwarding is policy-independent: a load-only **round-robin**
//! policy on a chat-formatter model still forwards `input_ids` to the engine
//! (the engine-tokenization offload), even though it picks workers round-robin
//! and ignores the tokens for routing. Tokenization is gated on the model's
//! chat formatter at ingress, not on the policy.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    Config, DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::policies::{Policy, SelectionContext};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{Worker, WorkerRegistry};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

// deepseek-v4 id → the tokenizer registry auto-attaches the built-in V4 chat
// encoder, so the model has an engine-equivalent encode path.
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
            disable_input_ids_forwarding: false,
            policy: PolicyKind::RoundRobin,
            decode_policy: Default::default(),
            bucket_config: None,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        router_inflight_load: InflightLoadConfig::default(),
    }
}

fn build_ctx(url: String) -> Arc<AppContext> {
    build_ctx_with_config(url, config())
}

fn build_ctx_with_config(url: String, cfg: Config) -> Arc<AppContext> {
    // The handler tokenizes via the AppContext's registry (which carries the V4
    // encoder); the RoundRobin policy itself needs no tokenizer.
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(tokenizers.has_chat_formatter(MODEL));
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

fn template_config(tokenizer_config: Value) -> (tempfile::TempDir, Config) {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = dir.path().join("tokenizer.json");
    std::fs::copy("tests/fixtures/tiny_tokenizer.json", &tokenizer).unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        tokenizer_config.to_string(),
    )
    .unwrap();
    let mut cfg = config();
    cfg.model.tokenizer_path = tokenizer.to_str().unwrap().into();
    (dir, cfg)
}

fn without_forwarding(mut cfg: Config, policy: PolicyKind) -> Config {
    cfg.model.policy = policy;
    cfg.model.cache_aware = (policy == PolicyKind::CacheAware).then(Default::default);
    cfg.model.disable_input_ids_forwarding = true;
    cfg
}

async fn assert_forwarded_unchanged(ctx: &Arc<AppContext>, mock: &MockWorker, request: &Value) {
    assert_eq!(send(Arc::clone(ctx), request.clone()).await, StatusCode::OK);
    assert_eq!(captured(mock), *request);
    assert!(!ctx
        .metrics
        .render()
        .contains("sgl_router_ingress_tokenize_errors_total{"));
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
/// chat-formatter model — the offload is decoupled from routing.
#[tokio::test]
async fn round_robin_plain_chat_forwards_input_ids() {
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
        "round-robin must forward input_ids on a chat-formatter model; got {body}"
    );
    assert!(
        body.get("messages").is_some(),
        "messages must be retained alongside input_ids; got {body}"
    );
}

#[tokio::test]
async fn forwarding_opt_out_preserves_messages_and_caller_ids() {
    for policy in [PolicyKind::RoundRobin, PolicyKind::CacheAware] {
        let mock = MockWorker::start(vec![]).await;
        let ctx = build_ctx_with_config(mock.url.clone(), without_forwarding(config(), policy));
        let mut request =
            json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]});
        assert_forwarded_unchanged(&ctx, &mock, &request).await;
        request["input_ids"] = json!([42, 43]);
        assert_forwarded_unchanged(&ctx, &mock, &request).await;
    }
}

#[tokio::test]
async fn forwarding_opt_out_keeps_ingress_tokens_for_routing() {
    #[derive(Debug)]
    struct ExpectTokens(Vec<u32>);
    impl Policy for ExpectTokens {
        fn needs_request_tokens(&self) -> bool {
            true
        }

        fn select(
            &self,
            workers: &[Arc<Worker>],
            ctx: &SelectionContext<'_>,
        ) -> Option<Arc<Worker>> {
            assert_eq!(ctx.request_tokens(), Some(self.0.as_slice()));
            workers.first().cloned()
        }
    }

    let mock = MockWorker::start(vec![]).await;
    let cfg = without_forwarding(config(), PolicyKind::RoundRobin);
    let ctx = build_ctx_with_config(mock.url.clone(), cfg);
    let request = json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]});
    let expected = ctx.tokenizers.encode_chat(MODEL, &request).unwrap();
    ctx.policies
        .insert(ModelId(MODEL.into()), Arc::new(ExpectTokens(expected)));
    assert_forwarded_unchanged(&ctx, &mock, &request).await;
}

/// Array-only deployments must opt out until Dynamo exposes its conversion flag.
#[tokio::test]
async fn array_only_template_opt_out_preserves_engine_processing() {
    let (_dir, cfg) = template_config(json!({
        "chat_template": "{% for m in messages %}{% for part in m.content %}{{ part.text }}{% endfor %}{% endfor %}"
    }));
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_config(
        mock.url.clone(),
        without_forwarding(cfg, PolicyKind::CacheAware),
    );
    let request = json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]});
    assert!(!ctx
        .tokenizers
        .encode_chat(MODEL, &request)
        .unwrap()
        .is_empty());
    assert_forwarded_unchanged(&ctx, &mock, &request).await;
}

#[tokio::test]
async fn template_with_date_helper_forwards_input_ids() {
    // GPT-OSS uses strftime_now; a bare Jinja probe incorrectly blocks it.
    let (_dir, cfg) = template_config(json!({
        "chat_template": "{{ strftime_now('%Y-%m-%d') }}{% for m in messages %}{{ m.content }}{% endfor %}"
    }));
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_config(mock.url.clone(), cfg);
    let request = json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]});
    let expected = ctx.tokenizers.encode_chat(MODEL, &request).unwrap();
    assert_eq!(send(ctx, request.clone()).await, StatusCode::OK);
    let body = captured(&mock);
    assert_eq!(body["input_ids"], json!(expected));
    assert_eq!(body["messages"], request["messages"]);
}

#[tokio::test]
async fn disabled_forwarding_does_not_count_routing_render_failures_as_offload_errors() {
    let (_dir, cfg) =
        template_config(json!({"chat_template": "{{ raise_exception('cannot render') }}"}));
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_config(
        mock.url.clone(),
        without_forwarding(cfg, PolicyKind::CacheAware),
    );
    let request = json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]});
    assert!(ctx.tokenizers.encode_chat(MODEL, &request).is_none());
    assert_forwarded_unchanged(&ctx, &mock, &request).await;
}

/// Even under round-robin, a tool request omits `input_ids` (the safe predicate
/// is policy-independent too).
#[tokio::test]
async fn round_robin_tool_request_omits_input_ids() {
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
    assert!(
        body.get("input_ids").is_none(),
        "tool requests must not forward input_ids under any policy; got {body}"
    );
}

/// A successful plain-chat forward on a chat-formatter model must NOT emit
/// `sgl_router_ingress_tokenize_errors_total` — that counter fires only when the
/// offload was expected but the encoder failed. A tool request on the same model
/// is an *expected* omission (its ids are still engine-equivalent; the
/// safe-predicate withholds forwarding for other reasons), so it must not emit
/// the error counter either.
#[tokio::test]
async fn successful_forward_does_not_emit_ingress_tokenize_error() {
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

/// History that dynamo-render rewrites stays intact for engine-side tokenization.
#[tokio::test]
async fn reasoning_history_preserves_messages_without_forwarding_ids() {
    let (_dir, cfg) = template_config(json!({
        "chat_template": "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}"
    }));
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_config(mock.url.clone(), cfg);
    let mut request = json!({"model": MODEL, "messages": [
        {"role":"user", "content":"hi"},
        {"role":"assistant", "content":"answer", "reasoning_content":"prior reasoning"},
        {"role":"user", "content":"next"}
    ]});
    assert!(!ctx
        .tokenizers
        .encode_chat(MODEL, &request)
        .unwrap()
        .is_empty());
    assert_forwarded_unchanged(&ctx, &mock, &request).await;

    request["messages"][1]
        .as_object_mut()
        .unwrap()
        .remove("reasoning_content");
    assert_eq!(send(ctx, request).await, StatusCode::OK);
    assert!(captured(&mock).get("input_ids").is_some());
}

/// Strict-template rewrites are used for routing only; the engine gets the original turns.
#[tokio::test]
async fn role_rewrites_preserve_messages_without_forwarding_ids() {
    let template = concat!(
        "{%- set ns = namespace(prev='') -%}",
        "{%- for m in messages -%}",
        "{%- if m.role == 'system' and not loop.first -%}",
        "{{ raise_exception('System message must be first.') }}",
        "{%- endif -%}",
        "{%- if m.role == 'user' and ns.prev == 'user' -%}",
        "{{ raise_exception('Conversation roles must alternate.') }}",
        "{%- endif -%}",
        "{{ m.role }}:{{ m.content }};",
        "{%- set ns.prev = m.role -%}",
        "{%- endfor -%}"
    );
    let (_dir, cfg) = template_config(json!({
        "chat_template": template, "sp_model_kwargs": {"enable_sampling": false}
    }));
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_config(mock.url.clone(), cfg);
    for roles in [
        vec!["user", "user"],
        vec!["system", "system", "user"],
        vec!["user", "assistant", "system", "user"],
    ] {
        let messages: Vec<_> = roles
            .iter()
            .map(|role| json!({"role": role, "content": "text"}))
            .collect();
        let request = json!({"model": MODEL, "messages": messages});
        assert!(!ctx
            .tokenizers
            .encode_chat(MODEL, &request)
            .unwrap()
            .is_empty());
        assert_forwarded_unchanged(&ctx, &mock, &request).await;
    }
    let request = json!({"model": MODEL, "messages": [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "next"}
    ]});
    assert_eq!(send(ctx, request).await, StatusCode::OK);
    assert!(captured(&mock).get("input_ids").is_some());
}

#[path = "../fixtures/kimi_k3.rs"]
mod kimi_fixture;

#[tokio::test]
async fn kimi_ids_forward_with_engine_rendering_fallback() {
    let mock = MockWorker::start(vec![]).await;
    let fixture = kimi_fixture::tokenizer();
    let mut cfg = config();
    let path = fixture.path().join("tiktoken.model");
    cfg.model.tokenizer_path = path.display().to_string();
    let ctx = build_ctx_with_config(mock.url.clone(), cfg);
    for (content, kwargs) in [
        ("literal <|open|> text", None),
        ("hi", Some(json!({"thinking_effort": null}))),
    ] {
        let mut request =
            json!({"model": MODEL, "messages": [{"role": "user", "content": content}]});
        let forward = kwargs.is_none();
        if let Some(kwargs) = kwargs {
            request["chat_template_kwargs"] = kwargs;
        }
        let ids = ctx.tokenizers.encode_chat(MODEL, &request);
        assert_eq!(send(ctx.clone(), request.clone()).await, StatusCode::OK);
        if forward {
            request["input_ids"] = json!(ids.unwrap());
        } else {
            assert!(ids.is_none());
        }
        assert_eq!(captured(&mock), request);
    }
}
