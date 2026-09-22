// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--max-output-tokens` and `--override-sampling-params` on the
//! sglang-native `/generate` surface: the SAME controls, translated to
//! where `GenerateReqInput` actually reads them (`sampling_params.*`), with
//! one deliberate asymmetry — the output cap ENFORCES but never INJECTS on
//! `/generate`, because an absent `sampling_params.max_new_tokens` takes the
//! engine's `SamplingParams` dataclass default (128), already far under any
//! sane cap. Injecting would turn a ceiling into a floor and RAISE
//! per-request output. The chat surface's absent budget maps to an explicit
//! unbounded `max_new_tokens=None`, so there injection is a genuine
//! tightening.
//!
//! Each test asserts against the body the `MockWorker` actually received
//! (or that none arrived): a cap that injected a key the engine discards
//! would return 200 while doing nothing, and only the wire shows it.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
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
const OVERRIDES: &str = r#"{"temperature": 1, "top_p": 0.95}"#;

/// The flag sets under test, named so each case reads as a configuration
/// rather than an argv fragment.
const NO_FLAGS: &[&str] = &[];
const CAP: &[&str] = &["--max-output-tokens", "4096"];
const PINS: &[&str] = &["--override-sampling-params", OVERRIDES];
const CAP_AND_PINS: &[&str] = &[
    "--max-output-tokens",
    "4096",
    "--override-sampling-params",
    OVERRIDES,
];

/// Build the config the way a deployment does — through `Cli`, so the flag
/// spelling in a manifest is what these tests pin.
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

/// What the worker saw for one request, plus the client-facing status. The
/// `Option` fields are `None` exactly when the router rejected the request
/// before dispatch — which is itself the assertion most of these tests make.
struct Dispatched {
    status: StatusCode,
    /// The error message on a rejection, empty on success.
    error: String,
    /// The body the worker received.
    body: Option<Value>,
    /// The path the worker was called on.
    path: Option<String>,
}

impl Dispatched {
    /// Assert a pre-dispatch rejection whose message names `key`, and that
    /// nothing reached the engine.
    fn assert_rejected_naming(&self, key: &str) {
        assert_eq!(self.status, StatusCode::BAD_REQUEST);
        assert!(
            self.error.contains(key),
            "the 400 must name the key the client sent: {}",
            self.error
        );
        assert!(
            self.body.is_none(),
            "a rejected request must not reach the engine"
        );
    }

    /// Assert a pre-dispatch rejection, without pinning the message.
    fn assert_rejected(&self) {
        assert_eq!(self.status, StatusCode::BAD_REQUEST);
        assert!(
            self.body.is_none(),
            "a rejected request must not reach the engine"
        );
    }

    /// Assert a 200 and return the body the worker received.
    fn forwarded(&self) -> &Value {
        assert_eq!(self.status, StatusCode::OK);
        self.body.as_ref().expect("worker received a request")
    }

    /// The forwarded `sampling_params`, `None` when the body grew no such key.
    fn sampling_params(&self) -> Option<&Value> {
        self.forwarded().get("sampling_params")
    }
}

/// Send one request through a freshly-built router at `flags` and report what
/// the worker saw.
async fn run(flags: &[&str], path: &str, body: Value) -> Dispatched {
    let mock = MockWorker::start(vec![]).await;
    let cfg = config(flags);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(mock.url.clone()),
        url: mock.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
        transfer_group: None,
    });
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));

    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = build_router(ctx).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let error = parsed
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or_default()
        .to_string();

    let captured = mock.captured.lock().unwrap();
    Dispatched {
        status,
        error,
        body: captured
            .last_body
            .as_ref()
            .map(|b| serde_json::from_slice(b).expect("captured body is valid JSON")),
        path: captured.last_path.clone(),
    }
}

/// A plain `/generate` body carrying nothing the controls act on.
fn plain() -> Value {
    json!({"model": MODEL, "text": "hi"})
}

/// A `/generate` body whose `sampling_params` is exactly `sp`.
fn with_sampling_params(sp: Value) -> Value {
    json!({"model": MODEL, "text": "hi", "sampling_params": sp})
}

/// The test that pins enforce-don't-inject: a cap is configured and the
/// request sets no output budget, so the forwarded body must carry NEITHER
/// `sampling_params.max_new_tokens` NOR a top-level `max_tokens` — the
/// engine's 128 default stands. An implementation that reused the chat
/// surface's inject-when-absent arm would write one of them and fail here.
#[tokio::test]
async fn cap_enforces_but_never_injects_on_generate() {
    let d = run(CAP, "/generate", plain()).await;
    let body = d.forwarded();
    assert_eq!(
        d.path.as_deref(),
        Some("/generate"),
        "the request must be proxied to the worker's /generate path"
    );
    assert_eq!(
        d.sampling_params().and_then(|sp| sp.get("max_new_tokens")),
        None,
        "no budget may be injected on /generate: {body}"
    );
    assert_eq!(
        body.get("max_tokens"),
        None,
        "a top-level max_tokens would be silently dropped by the engine: {body}"
    );
}

/// An explicit over-cap `sampling_params.max_new_tokens` is a 400 before
/// dispatch, and the message names the key the client actually sent — not
/// `max_tokens`, a field `/generate` clients never set.
#[tokio::test]
async fn over_cap_max_new_tokens_is_400_naming_the_generate_key() {
    run(
        CAP,
        "/generate",
        with_sampling_params(json!({"max_new_tokens": 999999})),
    )
    .await
    .assert_rejected_naming("sampling_params.max_new_tokens");
}

/// An explicit `"max_new_tokens": null` is NOT the absent case: the engine
/// reads it as UNBOUNDED (`init_req_max_new_tokens` maps null to `1 << 30`),
/// so with a cap configured it is a 400 — treating it as absent would let a
/// client null its way past `--max-output-tokens`. With no cap configured it
/// forwards untouched.
#[tokio::test]
async fn explicit_null_max_new_tokens_is_a_400_under_a_cap() {
    let body = || with_sampling_params(json!({"max_new_tokens": null}));

    run(CAP, "/generate", body())
        .await
        .assert_rejected_naming("sampling_params.max_new_tokens");

    assert_eq!(
        run(NO_FLAGS, "/generate", body()).await.status,
        StatusCode::OK,
        "with no cap configured the null forwards untouched"
    );
}

/// An explicit under-cap budget forwards unchanged.
#[tokio::test]
async fn under_cap_max_new_tokens_forwards_unchanged() {
    let d = run(
        CAP,
        "/generate",
        with_sampling_params(json!({"max_new_tokens": 100})),
    )
    .await;
    let body = d.forwarded();
    assert_eq!(
        d.sampling_params(),
        Some(&json!({"max_new_tokens": 100})),
        "{body}"
    );
    assert_eq!(body.get("max_tokens"), None, "{body}");
}

/// Under the default `reject` mode a conflicting `sampling_params` value is
/// a 400 that never reaches a worker.
#[tokio::test]
async fn reject_mode_400s_a_conflicting_sampling_params_value() {
    run(
        PINS,
        "/generate",
        with_sampling_params(json!({"temperature": 0.7})),
    )
    .await
    .assert_rejected();
}

/// A configured parameter the request omits is injected UNDER
/// `sampling_params` — alongside anything the object already carried — and
/// never as a top-level key the engine would discard.
#[tokio::test]
async fn omitted_temperature_is_injected_under_sampling_params() {
    let d = run(
        PINS,
        "/generate",
        with_sampling_params(json!({"top_k": 50})),
    )
    .await;
    let body = d.forwarded();
    assert_eq!(
        d.sampling_params(),
        Some(&json!({"temperature": 1, "top_p": 0.95, "top_k": 50})),
        "{body}"
    );
    assert_eq!(body.get("temperature"), None, "{body}");
    assert_eq!(body.get("top_p"), None, "{body}");
}

/// A request with no `sampling_params` key at all gets the object created
/// for the injections; an explicit `null` is treated the same way, rather
/// than the null being forwarded.
#[tokio::test]
async fn absent_or_null_sampling_params_object_is_created() {
    for body in [plain(), with_sampling_params(Value::Null)] {
        let d = run(PINS, "/generate", body.clone()).await;
        assert_eq!(
            d.sampling_params(),
            Some(&json!({"temperature": 1, "top_p": 0.95})),
            "sent {body}, worker saw {}",
            d.forwarded()
        );
    }
}

/// A present but non-object `sampling_params` is a 400 BEFORE dispatch —
/// forwarding it would crash the engine's `_handle_parallel_sampling` into a
/// 500 after the router skipped its own budget check.
#[tokio::test]
async fn non_object_sampling_params_is_400_before_dispatch() {
    for bad in [json!("x"), json!([])] {
        run(NO_FLAGS, "/generate", with_sampling_params(bad))
            .await
            .assert_rejected();
    }
}

/// Chat regression: the SAME cap/override configs against
/// `/v1/chat/completions` still write top-level keys and never emit a
/// `sampling_params` object.
#[tokio::test]
async fn chat_surface_still_writes_top_level_keys() {
    let d = run(
        CAP_AND_PINS,
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    let body = d.forwarded();
    assert_eq!(body.get("max_tokens"), Some(&json!(4096)), "{body}");
    assert_eq!(body.get("temperature"), Some(&json!(1)), "{body}");
    assert_eq!(body.get("top_p"), Some(&json!(0.95)), "{body}");
    assert_eq!(
        d.sampling_params(),
        None,
        "the chat surface must never grow a sampling_params object: {body}"
    );
}
