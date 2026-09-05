// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP contract for static P/D buckets.
//!
//! Buckets narrow the candidate domain before policy selection. Prefill SLO
//! profiles may override rank, while decode uses `input_tokens + max_tokens`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use sgl_kv_indexer::{PrefixIndex, PrefixIndexError, PrefixMatch, PrefixOutcome};
use sgl_router::config::{
    ActiveLoadConfig, AffinityConfig, BucketConfig, BucketSpec, BucketStage, CacheAwareConfig,
    CachePrefixProvider, Config, DiscoveryBackend, KvIndexerEndpointConfig, ModelConfig,
    ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig, SessionAffinityMode,
    SloBucketPolicy, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn bucket(id: &str, stage: BucketStage, rank: u32, worker_id: &str) -> BucketSpec {
    BucketSpec {
        id: id.into(),
        stage,
        rank,
        worker_ids: vec![worker_id.into()],
        min_extend_tokens: None,
        max_extend_tokens: None,
        min_sequence_tokens: None,
        max_sequence_tokens: None,
        max_context_tokens: Some(16_384),
        ttft_p95_at_capacity_ms: None,
        tps_p05_at_capacity: None,
        max_pending_prefill_tokens: None,
    }
}

fn build_app_context(
    specs: Vec<WorkerSpec>,
    bucket_config: BucketConfig,
    policy: PolicyKind,
    affinity: Option<AffinityConfig>,
) -> AppContext {
    let config = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy,
            decode_policy: Default::default(),
            bucket_config: Some(bucket_config),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity,
            fused: None,
            eligibility: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&config).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for spec in specs {
        let _ = registry.add(spec);
    }
    let policies = Arc::new(build_registry_with_defaults(&config).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    AppContext::new(config, tokenizers, proxy, registry, policies)
}

fn build_ctx(
    specs: Vec<WorkerSpec>,
    bucket_config: BucketConfig,
    policy: PolicyKind,
    affinity: Option<AffinityConfig>,
) -> Arc<AppContext> {
    Arc::new(build_app_context(specs, bucket_config, policy, affinity))
}

struct FakePrefixIndex {
    address: Option<String>,
    calls: AtomicUsize,
}

impl FakePrefixIndex {
    fn matched(address: String) -> Arc<Self> {
        Arc::new(Self {
            address: Some(address),
            calls: AtomicUsize::new(0),
        })
    }

    fn no_signal() -> Arc<Self> {
        Arc::new(Self {
            address: None,
            calls: AtomicUsize::new(0),
        })
    }
}

#[tonic::async_trait]
impl PrefixIndex for FakePrefixIndex {
    async fn match_prefix(&self, hashes: Vec<i64>) -> Result<PrefixOutcome, PrefixIndexError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let Some(address) = &self.address else {
            return Ok(PrefixOutcome::Empty);
        };
        let matched_prefix_blocks =
            u32::try_from(hashes.len().saturating_sub(1)).unwrap_or(u32::MAX);
        Ok(PrefixOutcome::Matched {
            matches: vec![PrefixMatch {
                address: address.clone(),
                matched_prefix_blocks,
                worker_id: "fake-index-worker".into(),
            }],
            best_prefix_blocks: matched_prefix_blocks,
        })
    }
}

struct TwoPrefixIndex {
    best_address: String,
    lower_ranked_address: String,
}

impl TwoPrefixIndex {
    fn new(best_address: String, lower_ranked_address: String) -> Arc<Self> {
        Arc::new(Self {
            best_address,
            lower_ranked_address,
        })
    }
}

#[tonic::async_trait]
impl PrefixIndex for TwoPrefixIndex {
    async fn match_prefix(&self, hashes: Vec<i64>) -> Result<PrefixOutcome, PrefixIndexError> {
        let best_prefix_blocks = u32::try_from(hashes.len().saturating_sub(1)).unwrap_or(u32::MAX);
        let lower_ranked_prefix_blocks = (best_prefix_blocks / 2).max(1);
        Ok(PrefixOutcome::Matched {
            matches: vec![
                PrefixMatch {
                    address: self.best_address.clone(),
                    matched_prefix_blocks: best_prefix_blocks,
                    worker_id: "best-index-worker".into(),
                },
                PrefixMatch {
                    address: self.lower_ranked_address.clone(),
                    matched_prefix_blocks: lower_ranked_prefix_blocks,
                    worker_id: "lower-index-worker".into(),
                },
            ],
            best_prefix_blocks,
        })
    }
}

fn build_cache_ctx(
    specs: Vec<WorkerSpec>,
    bucket_config: BucketConfig,
    prefix_index: Arc<dyn PrefixIndex>,
) -> Arc<AppContext> {
    build_cache_ctx_with_affinity(
        specs,
        bucket_config,
        prefix_index,
        AffinityConfig::default(),
    )
}

fn build_cache_ctx_with_affinity(
    specs: Vec<WorkerSpec>,
    bucket_config: BucketConfig,
    prefix_index: Arc<dyn PrefixIndex>,
    affinity: AffinityConfig,
) -> Arc<AppContext> {
    let mut context =
        build_app_context(specs, bucket_config, PolicyKind::CacheAware, Some(affinity));
    context.config.model.cache_aware = Some(CacheAwareConfig {
        prefix_provider: CachePrefixProvider::Indexer,
        kv_indexer_endpoint: Some(KvIndexerEndpointConfig {
            url: "http://fake-indexer".into(),
            query_timeout_ms: 100,
            query_max_inflight: 32,
        }),
    });
    context.prefix_index = Some(prefix_index);
    context.block_size_oracle.try_set(1).unwrap();
    Arc::new(context)
}

fn worker_spec(id: &str, url: String, mode: WorkerMode) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url,
        mode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
    }
}

fn chat_request(ttft_slo_ms: Option<u64>, max_tokens: Option<u64>) -> Request<Body> {
    chat_request_with_content("bucket routing", ttft_slo_ms, max_tokens, None)
}

fn chat_request_with_content(
    content: &str,
    ttft_slo_ms: Option<u64>,
    max_tokens: Option<u64>,
    session_id: Option<&str>,
) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(ttft_slo_ms) = ttft_slo_ms {
        builder = builder.header("x-sgl-ttft-slo-ms", ttft_slo_ms.to_string());
    }
    if let Some(session_id) = session_id {
        builder = builder.header("x-session-id", session_id);
    }
    builder
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
            }))
            .unwrap(),
        ))
        .unwrap()
}

async fn wait_for_prefill(mock: &crate::common::mock_worker::MockWorker) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if mock.captured.lock().unwrap().last_body.is_some() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("selected prefill worker must receive the detached request");
}

async fn wait_for_prefill_body_containing(
    mock: &crate::common::mock_worker::MockWorker,
    expected: &str,
) -> Vec<u8> {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let captured = mock.captured.lock().unwrap().last_body.clone();
            if let Some(body) = captured {
                if String::from_utf8_lossy(&body).contains(expected) {
                    return body.to_vec();
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("selected prefill worker must receive the expected request body")
}

#[tokio::test]
async fn prefill_slo_first_uses_eligible_ttft_bucket_before_lower_rank_bucket() {
    let cheap = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let fast = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let mut cheap_bucket = bucket("p-cheap", BucketStage::Prefill, 10, "p-cheap");
    cheap_bucket.ttft_p95_at_capacity_ms = Some(400);
    let mut fast_bucket = bucket("p-fast", BucketStage::Prefill, 20, "p-fast");
    fast_bucket.ttft_p95_at_capacity_ms = Some(100);
    let bucket_config = BucketConfig {
        buckets: vec![
            cheap_bucket,
            fast_bucket,
            bucket("d-catch-all", BucketStage::Decode, 30, "d"),
        ],
        ttft_slo_policy: SloBucketPolicy::SloFirst,
        tps_slo_policy: SloBucketPolicy::Disabled,
    };
    let ctx = build_ctx(
        vec![
            worker_spec("p-cheap", cheap.url.clone(), WorkerMode::Prefill),
            worker_spec("p-fast", fast.url.clone(), WorkerMode::Prefill),
            worker_spec("d", decode.url.clone(), WorkerMode::Decode),
        ],
        bucket_config,
        PolicyKind::PowerOfTwo,
        None,
    );

    let response = build_router(ctx)
        .oneshot(chat_request(Some(200), Some(16)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_prefill(&fast).await;
    assert!(
        cheap.captured.lock().unwrap().last_body.is_none(),
        "lower-rank but TTFT-ineligible P Bucket must not be dispatched first"
    );
}

#[tokio::test]
async fn decode_bucket_uses_input_plus_requested_output_budget() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let short_decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let long_decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let mut short_bucket = bucket("d-short", BucketStage::Decode, 20, "d-short");
    short_bucket.max_sequence_tokens = Some(1_024);
    let mut long_bucket = bucket("d-long", BucketStage::Decode, 30, "d-long");
    long_bucket.min_sequence_tokens = Some(1_025);
    let bucket_config = BucketConfig {
        buckets: vec![
            bucket("p", BucketStage::Prefill, 10, "p"),
            short_bucket,
            long_bucket,
        ],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    };
    let ctx = build_ctx(
        vec![
            worker_spec("p", prefill.url.clone(), WorkerMode::Prefill),
            worker_spec("d-short", short_decode.url.clone(), WorkerMode::Decode),
            worker_spec("d-long", long_decode.url.clone(), WorkerMode::Decode),
        ],
        bucket_config,
        PolicyKind::PowerOfTwo,
        None,
    );

    let response = build_router(ctx)
        .oneshot(chat_request(None, Some(2_000)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-sgl-decode-url")
            .and_then(|value| value.to_str().ok()),
        Some(long_decode.url.as_str()),
        "peak sequence length must exclude the short Decode Bucket"
    );
    assert!(
        long_decode.captured.lock().unwrap().last_body.is_some(),
        "the selected long Decode worker is awaited before the response"
    );
    assert!(
        short_decode.captured.lock().unwrap().last_body.is_none(),
        "the incompatible short Decode Bucket must not receive the request"
    );
}

#[tokio::test]
async fn prefill_only_bucket_configuration_keeps_global_decode_routing() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let bucket_config = BucketConfig {
        buckets: vec![bucket("p", BucketStage::Prefill, 10, "p")],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    };
    let ctx = build_ctx(
        vec![
            worker_spec("p", prefill.url.clone(), WorkerMode::Prefill),
            worker_spec("d", decode.url.clone(), WorkerMode::Decode),
        ],
        bucket_config,
        PolicyKind::PowerOfTwo,
        None,
    );

    let response = build_router(ctx)
        .oneshot(chat_request(None, Some(16)))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-sgl-decode-url")
            .and_then(|value| value.to_str().ok()),
        Some(decode.url.as_str()),
        "a Prefill-only Bucket rollout must retain the Step 1 global Decode domain"
    );
}

#[tokio::test]
async fn global_rebind_session_affinity_can_keep_a_cross_length_bucket_primary() {
    let short = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let long = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let mut short_bucket = bucket("p-short", BucketStage::Prefill, 10, "p-short");
    short_bucket.max_extend_tokens = Some(256);
    short_bucket.max_context_tokens = Some(16_384);
    short_bucket.ttft_p95_at_capacity_ms = Some(80);
    let mut long_bucket = bucket("p-long", BucketStage::Prefill, 20, "p-long");
    long_bucket.min_extend_tokens = Some(257);
    long_bucket.max_context_tokens = Some(16_384);
    long_bucket.ttft_p95_at_capacity_ms = Some(300);
    let bucket_config = BucketConfig {
        buckets: vec![
            short_bucket,
            long_bucket,
            bucket("d-catch-all", BucketStage::Decode, 30, "d"),
        ],
        ttft_slo_policy: SloBucketPolicy::SloFirst,
        tps_slo_policy: SloBucketPolicy::Disabled,
    };
    let ctx = build_ctx(
        vec![
            worker_spec("p-short", short.url.clone(), WorkerMode::Prefill),
            worker_spec("p-long", long.url.clone(), WorkerMode::Prefill),
            worker_spec("d", decode.url.clone(), WorkerMode::Decode),
        ],
        bucket_config,
        PolicyKind::SessionAware,
        Some(AffinityConfig {
            session_affinity_mode: SessionAffinityMode::GlobalRebind,
            ..Default::default()
        }),
    );
    let app = build_router(ctx);

    let first = app
        .clone()
        .oneshot(chat_request_with_content(
            "short",
            Some(120),
            Some(8),
            Some("s-1"),
        ))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    wait_for_prefill(&short).await;

    let long_content = "length ".repeat(128);
    let second = app
        .oneshot(chat_request_with_content(
            &long_content,
            Some(120),
            Some(8),
            Some("s-1"),
        ))
        .await
        .unwrap();
    assert_eq!(second.status(), StatusCode::OK);
    let short_body = wait_for_prefill_body_containing(&short, &long_content).await;
    assert!(
        String::from_utf8_lossy(&short_body).contains(&long_content),
        "the second, long request must retain the existing cross-Bucket session primary"
    );
    assert!(
        long.captured.lock().unwrap().last_body.is_none(),
        "target length Bucket is skipped only because the primary's own Hard TTFT profile is eligible"
    );
}

#[tokio::test]
async fn global_preserve_establishes_then_reuses_a_new_assignment() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let bucket_config = BucketConfig {
        buckets: vec![
            bucket("p", BucketStage::Prefill, 10, "p"),
            bucket("d", BucketStage::Decode, 20, "d"),
        ],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    };
    let ctx = build_ctx(
        vec![
            worker_spec("p", prefill.url.clone(), WorkerMode::Prefill),
            worker_spec("d", decode.url.clone(), WorkerMode::Decode),
        ],
        bucket_config,
        PolicyKind::SessionAware,
        Some(AffinityConfig {
            session_affinity_mode: SessionAffinityMode::GlobalPreserve,
            ..Default::default()
        }),
    );
    let app = build_router(Arc::clone(&ctx));

    for content in ["first global request", "second global request"] {
        let response = app
            .clone()
            .oneshot(chat_request_with_content(
                content,
                None,
                Some(8),
                Some("global-session"),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    let metrics = ctx.metrics.render();
    assert!(
        metrics.contains(
            r#"sgl_router_policy_decisions_total{policy="session_aware",reason="assigned"} 1"#
        ),
        "the first global-preserve request must establish an assignment: {metrics}"
    );
    assert!(
        metrics.contains(
            r#"sgl_router_policy_decisions_total{policy="session_aware",reason="session_primary"} 1"#
        ),
        "the second global-preserve request must reuse the assignment: {metrics}"
    );
}

#[tokio::test]
async fn cache_winner_uses_target_uncached_work_before_prompt_length_bucket() {
    let short = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let long = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let mut short_bucket = bucket("p-short", BucketStage::Prefill, 10, "p-short");
    short_bucket.max_extend_tokens = Some(8);
    let mut long_bucket = bucket("p-long", BucketStage::Prefill, 20, "p-long");
    long_bucket.min_extend_tokens = Some(9);
    let bucket_config = BucketConfig {
        buckets: vec![
            short_bucket,
            long_bucket,
            bucket("d-catch-all", BucketStage::Decode, 30, "d"),
        ],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    };
    let index = FakePrefixIndex::matched(short.url.clone());
    let prefix_index: Arc<dyn PrefixIndex> = index.clone();
    let ctx = build_cache_ctx(
        vec![
            worker_spec("p-short", short.url.clone(), WorkerMode::Prefill),
            worker_spec("p-long", long.url.clone(), WorkerMode::Prefill),
            worker_spec("d", decode.url.clone(), WorkerMode::Decode),
        ],
        bucket_config,
        prefix_index,
    );

    let content = "cached-prefix ".repeat(128);
    let response = build_router(ctx)
        .oneshot(chat_request_with_content(&content, None, Some(8), None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_prefill(&short).await;
    assert!(
        long.captured.lock().unwrap().last_body.is_none(),
        "a cache winner with small target-specific uncached work must not be replaced by the full-length Bucket"
    );
    assert_eq!(
        index.calls.load(Ordering::Relaxed),
        1,
        "the async Indexer query must run once at ingress, not once per Bucket"
    );
}

#[tokio::test]
async fn cache_candidate_bucket_binding_happens_before_candidate_limit() {
    let best = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let lower_ranked = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let mut best_bucket = bucket("p-best", BucketStage::Prefill, 10, "p-best");
    best_bucket.min_extend_tokens = Some(32);
    let mut lower_ranked_bucket = bucket("p-lower", BucketStage::Prefill, 20, "p-lower");
    lower_ranked_bucket.min_extend_tokens = Some(32);
    let bucket_config = BucketConfig {
        buckets: vec![
            best_bucket,
            lower_ranked_bucket,
            bucket("d-catch-all", BucketStage::Decode, 30, "d"),
        ],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    };
    let index: Arc<dyn PrefixIndex> =
        TwoPrefixIndex::new(best.url.clone(), lower_ranked.url.clone());
    let ctx = build_cache_ctx_with_affinity(
        vec![
            worker_spec("p-best", best.url.clone(), WorkerMode::Prefill),
            worker_spec("p-lower", lower_ranked.url.clone(), WorkerMode::Prefill),
            worker_spec("d", decode.url.clone(), WorkerMode::Decode),
        ],
        bucket_config,
        index,
        AffinityConfig {
            cache_candidate_min_workers: 1,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 1,
            ..AffinityConfig::default()
        },
    );

    let content = "cached bucket candidate ".repeat(256);
    let response = build_router(Arc::clone(&ctx))
        .oneshot(chat_request_with_content(&content, None, Some(8), None))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    wait_for_prefill(&lower_ranked).await;
    assert!(
        best.captured.lock().unwrap().last_body.is_none(),
        "the top Indexer hit is Bucket-incompatible and must not consume K=1"
    );
    assert!(
        ctx.metrics.render().contains(
            r#"sgl_router_policy_decisions_total{policy="cache_aware",reason="cache_candidate"} 1"#
        ),
        "the compatible lower-ranked cache holder must remain a cache candidate"
    );
}

#[tokio::test]
async fn cache_no_signal_restarts_normal_prompt_length_bucket_fallback() {
    let short = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let long = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let mut short_bucket = bucket("p-short", BucketStage::Prefill, 10, "p-short");
    short_bucket.max_extend_tokens = Some(8);
    let mut long_bucket = bucket("p-long", BucketStage::Prefill, 20, "p-long");
    long_bucket.min_extend_tokens = Some(9);
    let bucket_config = BucketConfig {
        buckets: vec![
            short_bucket,
            long_bucket,
            bucket("d-catch-all", BucketStage::Decode, 30, "d"),
        ],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    };
    let index = FakePrefixIndex::no_signal();
    let prefix_index: Arc<dyn PrefixIndex> = index.clone();
    let ctx = build_cache_ctx(
        vec![
            worker_spec("p-short", short.url.clone(), WorkerMode::Prefill),
            worker_spec("p-long", long.url.clone(), WorkerMode::Prefill),
            worker_spec("d", decode.url.clone(), WorkerMode::Decode),
        ],
        bucket_config,
        prefix_index,
    );

    let content = "uncached-prompt ".repeat(128);
    let response = build_router(ctx)
        .oneshot(chat_request_with_content(&content, None, Some(8), None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_prefill(&long).await;
    assert!(
        short.captured.lock().unwrap().last_body.is_none(),
        "without a cache winner the request must restart the normal full-input Bucket path"
    );
    assert_eq!(index.calls.load(Ordering::Relaxed), 1);
}
