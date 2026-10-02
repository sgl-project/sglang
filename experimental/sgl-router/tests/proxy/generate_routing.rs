// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Native `/generate`: the engine's schema passes through, routed like chat.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::Router;
use serde_json::{json, Value};
use sgl_router::config::{AffinityConfig, CachePrefixProvider};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::prefix_provider::RadixTreePrefixProvider;
use sgl_router::policies::request_tokens_for;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::state::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, MODEL};
use crate::common::mock_worker::MockWorker;
use crate::common::streaming::{collect_body, parse_sse_data};

/// A cache-aware router over `workers` whose KV prefixes come from `tree`.
fn router(workers: &[(&MockWorker, WorkerMode)], tree: HashTree) -> Router {
    let mut cfg = config();
    cfg.model.cache_aware.as_mut().unwrap().prefix_provider = CachePrefixProvider::RadixTree;
    cfg.model.affinity = Some(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(0),
        cache_candidate_min_workers: 1,
        cache_candidate_ratio: 1.0,
        cache_candidate_max_workers: 1,
        ..Default::default()
    });
    let registry = WorkerRegistry::default();
    for &(worker, mode) in workers {
        let spec = WorkerSpec {
            id: WorkerId(worker.url.clone()),
            url: worker.url.clone(),
            mode,
            model_ids: vec![ModelId(MODEL.into())],
            bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
            ..Default::default()
        };
        registry.add(spec).unwrap();
    }
    let (tree, oracle) = (Arc::new(tree), BlockSizeOracle::new());
    oracle.try_set(1).unwrap();
    let policies = build_registry(&cfg, Arc::clone(&tree), Arc::clone(&oracle)).unwrap();
    let mut ctx = AppContext::new(
        cfg.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap()),
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        Arc::new(registry),
        Arc::new(policies),
    );
    ctx.radix_tree_prefix_provider = Some(RadixTreePrefixProvider::new(tree, Arc::clone(&oracle)));
    ctx.block_size_oracle = oracle;
    build_router(Arc::new(ctx))
}

async fn send(app: &Router, method: &str, body: &Value) -> Response {
    let request = Request::builder()
        .method(method)
        .uri("/generate")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    app.clone().oneshot(request).await.unwrap()
}

#[tokio::test]
async fn generate_forwards_the_engine_body_and_response() {
    let engine = MockWorker::start(vec![]).await;
    let app = router(&[(&engine, WorkerMode::Plain)], HashTree::new());
    let sent = json!({"text": "hi", "sampling_params": {"max_new_tokens": 8}});

    let res = send(&app, "POST", &sent).await;
    assert_eq!(res.status(), StatusCode::OK);
    let response: Value = serde_json::from_slice(&collect_body(res.into_body()).await).unwrap();

    // Only an rid for abort-on-disconnect is added: no `model`, no router `input_ids`.
    let mut forwarded = engine.captured_json().await;
    let rid = forwarded.as_object_mut().unwrap().remove("rid").unwrap();
    assert!(rid
        .as_str()
        .is_some_and(crate::common::is_engine_shaped_rid));
    assert_eq!(forwarded, sent);
    assert_eq!(response["meta_info"]["id"], rid);
}

#[tokio::test]
async fn generate_accepts_put_and_keeps_a_caller_rid() {
    let engine = MockWorker::start(vec![]).await;
    let app = router(&[(&engine, WorkerMode::Plain)], HashTree::new());
    let sent = json!({"text": ["a", "b"], "rid": ["r0", "r1"]});

    assert_eq!(send(&app, "PUT", &sent).await.status(), StatusCode::OK);
    assert_eq!(engine.captured_json().await, sent);
}

#[tokio::test]
async fn generate_streams_engine_events_through() {
    let engine = MockWorker::start(vec!["data: {\"text\":\"ok\"}\n\n", "data: [DONE]\n\n"]).await;
    let app = router(&[(&engine, WorkerMode::Plain)], HashTree::new());

    let res = send(&app, "POST", &json!({"text": "hi", "stream": true})).await;
    assert_eq!(res.headers()["content-type"], "text/event-stream");
    let events = parse_sse_data(&collect_body(res.into_body()).await);
    assert_eq!(events, ["{\"text\":\"ok\"}", "[DONE]"]);
}

#[tokio::test]
async fn pd_generate_sends_one_bootstrap_room_to_both_workers() {
    let (prefill, decode) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let app = router(
        &[
            (&prefill, WorkerMode::Prefill),
            (&decode, WorkerMode::Decode),
        ],
        HashTree::new(),
    );

    let res = send(&app, "POST", &json!({"text": "hi"})).await;
    assert_eq!(res.status(), StatusCode::OK);
    let (p, d) = (prefill.captured_json().await, decode.captured_json().await);
    for key in [
        "text",
        "rid",
        "bootstrap_host",
        "bootstrap_port",
        "bootstrap_room",
    ] {
        assert_eq!(p[key], d[key], "{key}");
    }
    assert_eq!(p["bootstrap_port"], 8997);
}

#[tokio::test]
async fn generate_routes_text_to_the_worker_caching_its_prefix() {
    let (cached, uncached) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let body = json!({"text": "local radix cache hit"});
    let tokenizers = TokenizerRegistry::load_from_config(&config()).unwrap();
    let tokens = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body).unwrap();
    let tree = HashTree::new();
    tree.insert(
        &KvWorkerId::new(cached.url.clone(), 0),
        None,
        &compute_block_hashes(&tokens.ids, 1),
    );
    let app = router(
        &[(&uncached, WorkerMode::Plain), (&cached, WorkerMode::Plain)],
        tree,
    );

    // Without tokens the pick is a load tie broken at random, so one request proves nothing.
    for _ in 0..16 {
        assert_eq!(send(&app, "POST", &body).await.status(), StatusCode::OK);
    }
    assert!(uncached.captured.lock().unwrap().last_body.is_none());
}
