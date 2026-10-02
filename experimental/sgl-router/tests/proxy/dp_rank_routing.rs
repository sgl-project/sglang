// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--dp-aware` forwards the chosen DP rank as `X-Data-Parallel-Rank` and `routed_dp_rank`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::Request;
use serde_json::json;
use sgl_router::config::{Config, PolicyKind, StickyConfig, StickyFallbackKind};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::prefix_provider::RadixTreePrefixProvider;
use sgl_router::policies::request_tokens_for;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::state::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{EngineProfile, WireProtocol, WorkerRegistry};
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, MODEL};
use crate::common::mock_worker::MockWorker;

const RANK: &str = "x-data-parallel-rank";
const KEY: &str = "x-conversation-id";

fn sticky_config() -> Config {
    let mut cfg = config();
    cfg.model.dp_aware = true;
    cfg.model.policy = PolicyKind::Sticky;
    cfg.model.cache_aware = None;
    cfg.model.sticky = Some(StickyConfig {
        header_name: KEY.into(),
        fallback_policy: StickyFallbackKind::RoundRobin,
        idle_secs: 3600,
        eviction_interval_secs: 3600,
    });
    cfg
}

/// Builds a router over `(worker, mode, dp_ranks)`, with `tree` as its local KV index.
fn router(
    cfg: Config,
    workers: &[(&MockWorker, WorkerMode, u32)],
    tree: Arc<HashTree>,
) -> axum::Router {
    let registry = Arc::new(WorkerRegistry::default());
    for &(worker, mode, dp_ranks) in workers {
        let spec = WorkerSpec {
            id: WorkerId(worker.url.clone()),
            url: worker.url.clone(),
            mode,
            model_ids: vec![ModelId(MODEL.into())],
            bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
            version_group: None,
        };
        let profile = EngineProfile {
            protocol: WireProtocol::default(),
            dp_ranks,
        };
        registry.add_with_cb(spec, None, profile).unwrap();
    }
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    let policies = Arc::new(build_registry(&cfg, Arc::clone(&tree), Arc::clone(&oracle)).unwrap());
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let mut ctx = AppContext::new(cfg, tokenizers, proxy, registry, policies);
    ctx.dp_rank_prefix_provider = Some(RadixTreePrefixProvider::new(tree, oracle));
    build_router(Arc::new(ctx))
}

fn body() -> serde_json::Value {
    json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
}

/// Sends one chat request and returns the rank header `worker` received.
async fn send(app: &axum::Router, worker: &MockWorker, headers: &[(&str, &str)]) -> Option<String> {
    let mut request =
        Request::post("/v1/chat/completions").header("content-type", "application/json");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let request = request.body(Body::from(body().to_string())).unwrap();
    worker.captured.lock().unwrap().headers.clear();
    assert!(app
        .clone()
        .oneshot(request)
        .await
        .unwrap()
        .status()
        .is_success());
    worker.captured.lock().unwrap().headers.get(RANK).cloned()
}

#[tokio::test]
async fn sticky_key_pins_a_rank_and_overrides_the_client() {
    let worker = MockWorker::start(vec![]).await;
    let app = router(
        sticky_config(),
        &[(&worker, WorkerMode::Plain, 4)],
        Default::default(),
    );
    let rank = send(&app, &worker, &[(KEY, "conv-a")]).await.unwrap();
    for _ in 0..3 {
        let spoofed = [(KEY, "conv-a"), (RANK, "9")];
        assert_eq!(send(&app, &worker, &spoofed).await.as_ref(), Some(&rank));
    }
}

#[tokio::test]
async fn no_rank_without_the_flag_or_for_a_single_rank_worker() {
    let worker = MockWorker::start(vec![]).await;
    let mut off = sticky_config();
    off.model.dp_aware = false;
    let app = router(off, &[(&worker, WorkerMode::Plain, 4)], Default::default());
    assert_eq!(send(&app, &worker, &[(RANK, "2")]).await, None);
    let app = router(
        sticky_config(),
        &[(&worker, WorkerMode::Plain, 1)],
        Default::default(),
    );
    assert_eq!(send(&app, &worker, &[]).await, None);
}

#[tokio::test]
async fn pd_room_maps_decode_to_the_prefill_rank() {
    let (prefill, decode) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let workers = [
        (&prefill, WorkerMode::Prefill, 4),
        (&decode, WorkerMode::Decode, 2),
    ];
    let app = router(sticky_config(), &workers, Default::default());
    assert!(send(&app, &decode, &[(KEY, "conv-pd")]).await.is_some());

    let start = Instant::now();
    let (body, rank) = loop {
        let captured = {
            let captured = prefill.captured.lock().unwrap();
            captured
                .last_body
                .clone()
                .zip(captured.headers.get(RANK).cloned())
        };
        if let Some(captured) = captured {
            break captured;
        }
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "no prefill request"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    let room = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["bootstrap_room"]
        .as_u64()
        .unwrap();
    assert_eq!(room % 4, rank.parse::<u64>().unwrap());
}

#[tokio::test]
async fn any_policy_picks_the_rank_with_the_deepest_prefix() {
    let worker = MockWorker::start(vec![]).await;
    let mut cfg = config();
    cfg.model.dp_aware = true;
    // Neither the policy nor input_ids forwarding asks for tokens.
    cfg.model.policy = PolicyKind::PowerOfTwo;
    cfg.model.cache_aware = None;
    cfg.model.disable_input_ids_forwarding = true;
    let tokenizers = TokenizerRegistry::load_from_config(&cfg).unwrap();
    let tokens = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body()).unwrap();
    let hashes = compute_block_hashes(&tokens.ids, 1);
    let tree = Arc::new(HashTree::new());
    tree.insert(&KvWorkerId::new(worker.url.clone(), 1), None, &hashes[..1]);
    tree.insert(&KvWorkerId::new(worker.url.clone(), 3), None, &hashes);

    let app = router(cfg, &[(&worker, WorkerMode::Plain, 4)], tree);
    for _ in 0..3 {
        assert_eq!(send(&app, &worker, &[]).await.as_deref(), Some("3"));
    }
}

/// Sends `body` to `path` under a sticky key over a PD pair with 4 prefill and 2 decode ranks,
/// returning the prefill and decode bodies.
async fn send_pd(path: &str, body: serde_json::Value) -> (serde_json::Value, serde_json::Value) {
    let (prefill, decode) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let workers = [
        (&prefill, WorkerMode::Prefill, 4),
        (&decode, WorkerMode::Decode, 2),
    ];
    let request = Request::post(path)
        .header("content-type", "application/json")
        .header(KEY, "conv-pd")
        .body(Body::from(body.to_string()))
        .unwrap();
    let app = router(sticky_config(), &workers, Default::default());
    assert!(app.oneshot(request).await.unwrap().status().is_success());
    (prefill.captured_json().await, decode.captured_json().await)
}

/// Native `/generate` ignores the rank header, so each PD worker gets its rank in its own body.
#[tokio::test]
async fn pd_generate_carries_each_rank_in_its_body() {
    let (p, d) = send_pd("/generate", json!({"text": "hi"})).await;
    assert_eq!(
        p["routed_dp_rank"],
        p["bootstrap_room"].as_u64().unwrap() % 4
    );
    assert!(d["routed_dp_rank"].as_u64().is_some_and(|rank| rank < 2));
}

/// The engine gives fan-out item i the room `room + i`, so one pinned prefill rank would break decode.
#[tokio::test]
async fn pd_fan_out_leaves_the_prefill_rank_to_the_engine() {
    let mut chat = body();
    chat["n"] = 2.into();
    for (path, mut body) in [
        ("/v1/chat/completions", chat),
        ("/generate", json!({"text": ["a", "b"]})),
    ] {
        // A caller's rank is replaced too.
        body["routed_dp_rank"] = 3.into();
        body["data_parallel_rank"] = 3.into();
        let (p, d) = send_pd(path, body).await;
        assert!(
            p["routed_dp_rank"].is_null() && p["data_parallel_rank"].is_null(),
            "{path}"
        );
        assert!(
            d["routed_dp_rank"].as_u64().is_some_and(|rank| rank < 2),
            "{path}"
        );
    }
}

/// The engine's `/v1/embeddings` reads no rank, so the router pins none.
#[tokio::test]
async fn embeddings_leave_the_rank_to_the_engine() {
    let worker = MockWorker::start(vec![]).await;
    let app = router(
        sticky_config(),
        &[(&worker, WorkerMode::Plain, 4)],
        Default::default(),
    );
    let request = Request::post("/v1/embeddings")
        .header("content-type", "application/json")
        .header(KEY, "conv-a")
        .body(Body::from(
            json!({"model": MODEL, "input": "hi"}).to_string(),
        ))
        .unwrap();
    assert!(app.oneshot(request).await.unwrap().status().is_success());
    assert_eq!(worker.captured.lock().unwrap().headers.get(RANK), None);
    assert!(worker.captured_json().await.get("routed_dp_rank").is_none());
}
