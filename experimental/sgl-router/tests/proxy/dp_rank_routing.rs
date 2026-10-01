// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--dp-aware`: the router picks a DP rank inside each selected multi-rank
//! worker and forwards it as `X-Data-Parallel-Rank`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    AffinityConfig, CachePrefixProvider, Config, DiscoveryBackend, InflightLoadConfig, ModelConfig,
    ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
    StickyConfig, StickyFallbackKind,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::{build_registry, build_registry_with_defaults};
use sgl_router::policies::prefix_provider::RadixTreePrefixProvider;
use sgl_router::policies::request_tokens_for;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::state::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{EngineProfile, WireProtocol, WorkerRegistry};
use tower::ServiceExt;

use crate::common::cache_aware_fixture;
use crate::common::mock_worker::MockWorker;

const RANK_HEADER: &str = "x-data-parallel-rank";
const STICKY_HEADER: &str = "x-conversation-id";

fn config(policy: PolicyKind, dp_aware: bool) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy,
            decode_policy: Default::default(),
            dp_aware,
            bucket_config: None,
            circuit_breaker: None,
            cache_aware: None,
            sticky: (policy == PolicyKind::Sticky).then(|| StickyConfig {
                header_name: STICKY_HEADER.into(),
                fallback_policy: StickyFallbackKind::RoundRobin,
                idle_secs: 3600,
                eviction_interval_secs: 3600,
            }),
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            default_chat_template_kwargs: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        router_inflight_load: InflightLoadConfig::default(),
    }
}

fn add_worker(registry: &WorkerRegistry, model: &str, url: &str, mode: WorkerMode, dp_ranks: u32) {
    registry
        .add_with_cb(
            WorkerSpec {
                id: WorkerId(url.into()),
                url: url.into(),
                mode,
                model_ids: vec![ModelId(model.into())],
                bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
            },
            None,
            EngineProfile {
                protocol: WireProtocol::default(),
                dp_ranks,
            },
        )
        .unwrap();
}

fn build_ctx(cfg: Config, workers: &[(&str, WorkerMode, u32)]) -> Arc<AppContext> {
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for &(url, mode, dp_ranks) in workers {
        add_worker(&registry, "tiny", url, mode, dp_ranks);
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn chat_request(model: &str, headers: &[(&str, &str)]) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        ))
        .unwrap()
}

/// Sends one request and returns the DP rank header the worker received.
async fn send(ctx: &Arc<AppContext>, mock: &MockWorker, headers: &[(&str, &str)]) -> Option<u32> {
    mock.captured.lock().unwrap().headers.clear();
    let response = build_router(Arc::clone(ctx))
        .oneshot(chat_request(&ctx.config.model.id, headers))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    received_rank(mock)
}

fn received_rank(mock: &MockWorker) -> Option<u32> {
    mock.captured
        .lock()
        .unwrap()
        .headers
        .get(RANK_HEADER)
        .map(|rank| rank.parse().expect("rank header is an integer"))
}

/// The PD prefill request is spawned detached; wait for it to land.
async fn await_body(mock: &MockWorker) -> Value {
    let start = Instant::now();
    loop {
        let body = mock.captured.lock().unwrap().last_body.clone();
        if let Some(body) = body {
            return serde_json::from_slice(&body).unwrap();
        }
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "no request captured"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn sticky_key_pins_a_stable_rank_and_overrides_client_rank() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        config(PolicyKind::Sticky, true),
        &[(&mock.url, WorkerMode::Plain, 4)],
    );

    let first = send(&ctx, &mock, &[(STICKY_HEADER, "conv-a")])
        .await
        .unwrap();
    assert!(first < 4);
    for _ in 0..3 {
        assert_eq!(
            send(&ctx, &mock, &[(STICKY_HEADER, "conv-a")]).await,
            Some(first)
        );
    }
    // A client cannot pin a rank behind the router's back.
    let spoofed = ((first + 1) % 4).to_string();
    assert_eq!(
        send(
            &ctx,
            &mock,
            &[(STICKY_HEADER, "conv-a"), (RANK_HEADER, &spoofed)]
        )
        .await,
        Some(first)
    );

    let mut ranks = HashSet::new();
    for i in 0..32 {
        let key = format!("conv-{i}");
        ranks.insert(send(&ctx, &mock, &[(STICKY_HEADER, &key)]).await.unwrap());
    }
    assert_eq!(ranks.len(), 4, "distinct keys spread over every rank");
}

#[tokio::test]
async fn keyless_requests_rotate_across_idle_ranks() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        config(PolicyKind::RoundRobin, true),
        &[(&mock.url, WorkerMode::Plain, 3)],
    );
    let mut ranks = Vec::new();
    for _ in 0..6 {
        ranks.push(send(&ctx, &mock, &[]).await.unwrap());
    }
    assert_eq!(ranks, [0, 1, 2, 0, 1, 2]);
}

#[tokio::test]
async fn no_rank_without_the_flag_or_for_a_single_rank_worker() {
    let mock = MockWorker::start(vec![]).await;
    let off = build_ctx(
        config(PolicyKind::RoundRobin, false),
        &[(&mock.url, WorkerMode::Plain, 4)],
    );
    assert_eq!(send(&off, &mock, &[(RANK_HEADER, "2")]).await, None);

    let single = build_ctx(
        config(PolicyKind::RoundRobin, true),
        &[(&mock.url, WorkerMode::Plain, 1)],
    );
    assert_eq!(send(&single, &mock, &[]).await, None);
}

#[tokio::test]
async fn pd_ranks_prefill_and_decode_separately_with_an_aligned_room() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        config(PolicyKind::Sticky, true),
        &[
            (&prefill.url, WorkerMode::Prefill, 4),
            (&decode.url, WorkerMode::Decode, 2),
        ],
    );
    let decode_rank = send(&ctx, &decode, &[(STICKY_HEADER, "conv-pd")]).await;
    let prefill_body = await_body(&prefill).await;
    let prefill_rank = received_rank(&prefill).expect("prefill gets a rank");

    assert!(decode_rank.is_some_and(|rank| rank < 2));
    // The decode engine resolves the prefill rank as `bootstrap_room % prefill_dp_size`.
    let room = prefill_body["bootstrap_room"].as_u64().unwrap();
    assert_eq!(room % 4, u64::from(prefill_rank));
    assert!(room <= i64::MAX as u64);
}

#[tokio::test]
async fn cache_aware_routes_to_the_rank_holding_the_prefix() {
    let mock = MockWorker::start(vec![]).await;
    let mut cfg = cache_aware_fixture::config();
    cfg.model.dp_aware = true;
    cfg.model.cache_aware.as_mut().unwrap().prefix_provider = CachePrefixProvider::RadixTree;
    cfg.model.affinity = Some(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(0),
        ..Default::default()
    });
    let model = cache_aware_fixture::MODEL;
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
    let tokens = request_tokens_for(&tokenizers, &ModelId(model.into()), &body).unwrap();

    let tree = Arc::new(HashTree::new());
    let hashes = compute_block_hashes(&tokens.ids, 1);
    tree.insert(&KvWorkerId::new(mock.url.clone(), 2), None, &hashes[..1]);
    tree.insert(&KvWorkerId::new(mock.url.clone(), 3), None, &hashes);
    let registry = Arc::new(WorkerRegistry::default());
    add_worker(&registry, model, &mock.url, WorkerMode::Plain, 4);
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    let policies = Arc::new(build_registry(&cfg, Arc::clone(&tree), Arc::clone(&oracle)).unwrap());
    let mut ctx = AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    );
    ctx.radix_tree_prefix_provider = Some(RadixTreePrefixProvider::new(tree, Arc::clone(&oracle)));
    ctx.block_size_oracle = oracle;
    let ctx = Arc::new(ctx);

    for _ in 0..3 {
        assert_eq!(send(&ctx, &mock, &[]).await, Some(3));
    }
}
