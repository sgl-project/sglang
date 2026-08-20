// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--override-sampling-params` end to end: from the CLI flag an operator
//! writes in a manifest to the JSON body the engine actually receives.
//!
//! The unit tests in `config::cli` and `server::routes::chat` cover parsing
//! and the per-parameter decision; these drive the whole path, because the
//! failure this feature exists to prevent (an engine serving sampling
//! parameters the operator did not declare) is only observable on the wire.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{Cli, Config};
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

const MODEL: &str = "tiny";

/// Build the config the way a deployment does — through `Cli`, so the flag
/// spelling in a manifest is what these tests pin, not a hand-built struct
/// that could drift from what the parser produces.
fn config(flags: &[&str]) -> Config {
    let mut argv = vec![
        "sgl-router",
        "--model-id",
        MODEL,
        "--tokenizer-path",
        "tests/fixtures/tiny_tokenizer.json",
        "--worker-urls",
        "http://placeholder:0",
    ];
    argv.extend_from_slice(flags);
    <Cli as clap::Parser>::parse_from(argv)
        .into_config()
        .expect("flags must parse")
}

fn build_ctx(url: String, flags: &[&str]) -> Arc<AppContext> {
    let cfg = config(flags);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
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
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    build_router(ctx).oneshot(req).await.unwrap().status()
}

fn captured(mock: &MockWorker) -> Option<Value> {
    let b = mock.captured.lock().unwrap().last_body.clone()?;
    Some(serde_json::from_slice(&b).expect("captured body is valid JSON"))
}

const OVERRIDES: &str = r#"{"temperature": 1, "top_p": 0.95, "top_k": 1000,
                            "frequency_penalty": 0, "presence_penalty": 0, "n": 1}"#;

/// The flags an operator writes replace the engine's own defaults: a request
/// that names no sampling parameter reaches the engine carrying every
/// configured value.
#[tokio::test]
async fn configured_values_reach_the_engine_when_the_request_omits_them() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &["--override-sampling-params", OVERRIDES]);
    let status = send(
        ctx,
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), Some(&json!(1)));
    assert_eq!(body.get("top_p"), Some(&json!(0.95)));
    assert_eq!(body.get("top_k"), Some(&json!(1000)));
    assert_eq!(body.get("frequency_penalty"), Some(&json!(0)));
    assert_eq!(body.get("presence_penalty"), Some(&json!(0)));
    assert_eq!(body.get("n"), Some(&json!(1)));
}

/// Under the default `reject` mode a conflicting request is a 400 that never
/// reaches a worker — the immutability contract costs no engine round-trip
/// and no admission slot.
#[tokio::test]
async fn reject_mode_400s_a_conflicting_request_without_touching_the_engine() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &["--override-sampling-params", OVERRIDES]);
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        captured(&mock).is_none(),
        "a rejected request must not reach the engine"
    );
}

/// The same request under `allow` is forwarded with the client's value intact,
/// and the parameters it did not name still get the configured ones.
#[tokio::test]
async fn allow_mode_forwards_the_client_value_and_fills_the_rest() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        mock.url.clone(),
        &[
            "--override-sampling-params",
            OVERRIDES,
            "--sampling-param-conflict",
            "allow",
        ],
    );
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), Some(&json!(0.7)));
    assert_eq!(body.get("top_p"), Some(&json!(0.95)));
    assert_eq!(body.get("top_k"), Some(&json!(1000)));
    assert_eq!(body.get("n"), Some(&json!(1)));
}

/// A band accepts anything inside it, 400s outside, and injects nothing —
/// Kimi K3's contract: temperature tunable in [0, 1], everything else fixed.
#[tokio::test]
async fn a_temperature_band_admits_in_range_values_and_injects_nothing() {
    let flags = [
        "--override-sampling-params",
        r#"{"temperature": {"min": 0, "max": 1}, "top_p": 0.95}"#,
    ];

    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &flags);
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.6,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), Some(&json!(0.6)));
    assert_eq!(body.get("top_p"), Some(&json!(0.95)));

    // Omitted: the band names no value, so the engine's own default applies.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &flags);
    let status = send(
        ctx,
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), None);

    // Outside the band: 400.
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &flags);
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(captured(&mock).is_none());
}

/// With the flag unset the router touches nothing: the body the client sent is
/// the body the engine sees, including a sampling parameter the operator
/// could have pinned.
#[tokio::test]
async fn unset_flag_forwards_the_body_untouched() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), &[]);
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("temperature"), Some(&json!(0.7)));
    assert_eq!(body.get("top_p"), None);
    assert_eq!(body.get("top_k"), None);
    assert_eq!(body.get("n"), None);
}
