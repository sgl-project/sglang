// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sgl_kv_indexer::{PrefixIndex, PrefixIndexError, PrefixMatch, PrefixOutcome};
use sgl_router::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
use sgl_router::config::AffinityConfig;
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies::prefix_provider::RadixTreePrefixProvider;
use sgl_router::policies_reorg::admission::{Decision, EngineAdmission, EngineMetrics};
use sgl_router::policies_reorg::cache_aware::{CacheAwarePolicy, CacheSource, PrefixMemo};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies_reorg::{PickError, PickRequest, Policy, Stage};
use sgl_router::state::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree, KvWorkerId,
};
use sgl_router::state::load_monitor::engine_reported_load::{
    EngineReportedLoadTable, LoadStat, NativeCacheRankLoad,
};
use sgl_router::workers::Worker;

const TOKENS: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

fn engine(id: &str, active: usize) -> Arc<Worker> {
    let engine = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}"),
        mode: Stage::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }));
    engine.active_requests.store(active, Ordering::Relaxed);
    engine
}

fn config() -> AffinityConfig {
    AffinityConfig {
        cache_affinity_min_matched_tokens: Some(1),
        cache_switch_margin_tokens: 0,
        ..Default::default()
    }
}

fn oracle() -> Arc<BlockSizeOracle> {
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    oracle
}

fn local(entries: &[(&Arc<Worker>, usize)]) -> Arc<CacheSource> {
    let tree = Arc::new(HashTree::new());
    let hashes = compute_block_hashes(&TOKENS, 1);
    for (worker, depth) in entries {
        tree.insert(
            &KvWorkerId::new(worker.url.clone(), 0),
            None,
            &hashes[..*depth],
        );
    }
    Arc::new(CacheSource::Local(RadixTreePrefixProvider::new(
        tree,
        oracle(),
    )))
}

fn request(model: &ModelId) -> PickRequest<'_> {
    PickRequest {
        token_ids: Some(&TOKENS),
        ..PickRequest::new(model, Stage::Plain, 8)
    }
}

fn report(
    table: &EngineReportedLoadTable,
    engine: &Worker,
    waiting: u64,
    pending: u64,
    at: Instant,
) {
    table.set(
        &engine.url,
        0,
        LoadStat {
            num_running_reqs: 1,
            num_waiting_reqs: waiting,
            num_tokens: 10,
            max_total_num_tokens: 100,
            native_cache: Some(NativeCacheRankLoad {
                num_waiting_uncached_tokens: pending,
                num_total_tokens: 10,
                max_running_requests: 100,
                total_prefill_uncached_tokens: 0,
                total_prefill_busy_us: 0,
            }),
        },
        at,
    );
}

#[derive(Debug)]
struct Reject {
    id: &'static str,
    calls: Mutex<Vec<(String, Option<u64>)>>,
}

impl Reject {
    fn new(id: &'static str) -> Arc<Self> {
        Arc::new(Self {
            id,
            calls: Mutex::new(Vec::new()),
        })
    }
}

impl EngineAdmission for Reject {
    fn check(&self, engine: &Worker, metrics: &EngineMetrics) -> Result<Decision, PickError> {
        self.calls
            .lock()
            .unwrap()
            .push((engine.id.0.clone(), metrics.waiting_requests));
        Ok(if engine.id.0 == self.id {
            Decision::Reject("full".into())
        } else {
            Decision::Allow
        })
    }
}

struct Index {
    result: Result<PrefixOutcome, PrefixIndexError>,
    calls: AtomicUsize,
    hashes: Mutex<Vec<Vec<i64>>>,
}

#[tonic::async_trait]
impl PrefixIndex for Index {
    async fn match_prefix(&self, hashes: Vec<i64>) -> Result<PrefixOutcome, PrefixIndexError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.hashes.lock().unwrap().push(hashes);
        tokio::task::yield_now().await;
        self.result.clone()
    }
}

fn remote(
    result: Result<PrefixOutcome, PrefixIndexError>,
    oracle: Arc<BlockSizeOracle>,
) -> (Arc<CacheSource>, Arc<Index>) {
    let index = Arc::new(Index {
        result,
        calls: AtomicUsize::new(0),
        hashes: Mutex::new(Vec::new()),
    });
    (
        Arc::new(CacheSource::Remote {
            index: index.clone(),
            block_size: oracle,
        }),
        index,
    )
}

fn matches(entries: &[(&str, u32)]) -> PrefixOutcome {
    PrefixOutcome::Matched {
        matches: entries
            .iter()
            .map(|(url, depth)| PrefixMatch {
                address: (*url).into(),
                worker_id: "not-a-routing-identity".into(),
                matched_prefix_blocks: *depth,
            })
            .collect(),
        best_prefix_blocks: entries.iter().map(|(_, depth)| *depth).max().unwrap_or(0),
    }
}

#[tokio::test]
async fn local_prefix_wins_within_group_and_threshold_misses_use_load() {
    let engines = [
        engine("deep", 9),
        engine("shallow", 0),
        engine("outside", 0),
    ];
    let source = local(&[(&engines[0], 7), (&engines[1], 4), (&engines[2], 8)]);
    let model = ModelId("m".into());
    for (minimum, ratio, expected) in [(Some(1), None, 0), (Some(8), None, 1), (None, Some(0.9), 1)]
    {
        let policy = CacheAwarePolicy::new(
            source.clone(),
            EngineReportedLoadTable::new(),
            AffinityConfig {
                cache_affinity_min_matched_tokens: minimum,
                cache_affinity_min_match_ratio: ratio,
                ..config()
            },
        )
        .unwrap();
        for stage in [Stage::Plain, Stage::Prefill] {
            let request = PickRequest {
                stage,
                ..request(&model)
            };
            let pick = policy.pick(&engines[..2], &request).await.unwrap();
            assert!(Arc::ptr_eq(&pick.engine, &engines[expected]));
        }
    }
}

#[tokio::test]
async fn remote_urls_are_exact_duplicate_depths_merge_and_block_counts_are_capped() {
    let engines = [engine("a", 9), engine("b", 0)];
    let (source, _) = remote(
        Ok(matches(&[
            ("http://a", 1),
            ("http://a", u32::MAX),
            ("http://b/", u32::MAX),
        ])),
        oracle(),
    );
    let policy = CacheAwarePolicy::new(
        source,
        EngineReportedLoadTable::new(),
        AffinityConfig {
            cache_affinity_min_match_ratio: Some(1.0),
            ..config()
        },
    )
    .unwrap();
    let model = ModelId("m".into());
    assert_eq!(
        policy
            .pick(&engines, &request(&model))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "a"
    );
}

#[tokio::test]
async fn memo_reuses_io_but_reruns_admission_and_group_selection() {
    let engines = [engine("a", 0), engine("b", 0)];
    let (source, index) = remote(Ok(matches(&[("http://a", 8), ("http://b", 7)])), oracle());
    let mut policy =
        CacheAwarePolicy::new(source.clone(), EngineReportedLoadTable::new(), config()).unwrap();
    let admission = Reject::new("a");
    policy.admission = admission.clone();
    let memo = PrefixMemo::default();
    let model = ModelId("m".into());
    let request = PickRequest {
        prefix: Some(&memo),
        ..request(&model)
    };
    assert!(matches!(
        policy.pick(&engines[..1], &request).await,
        Err(PickError::NoAdmissibleEngine(_))
    ));
    assert_eq!(
        policy
            .pick(&engines[1..], &request)
            .await
            .unwrap()
            .engine
            .id
            .0,
        "b"
    );
    assert_eq!(
        policy.pick(&engines, &request).await.unwrap().engine.id.0,
        "b"
    );
    assert_eq!(index.calls.load(Ordering::Relaxed), 1);
    assert_eq!(admission.calls.lock().unwrap().len(), 4);
    // A second namespace in the same request must query its own backend.
    let (other, other_index) = remote(Ok(matches(&[("http://a", 8)])), oracle());
    let policy = CacheAwarePolicy::new(other, EngineReportedLoadTable::new(), config()).unwrap();
    assert_eq!(
        policy.pick(&engines, &request).await.unwrap().engine.id.0,
        "a"
    );
    assert_eq!(other_index.calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn concurrent_picks_share_one_query_and_new_requests_query_again() {
    let engines = [engine("a", 0)];
    let (source, index) = remote(Ok(PrefixOutcome::Empty), oracle());
    let policy = CacheAwarePolicy::new(source, EngineReportedLoadTable::new(), config()).unwrap();
    let model = ModelId("m".into());
    let memo = PrefixMemo::default();
    let request = PickRequest {
        prefix: Some(&memo),
        ..request(&model)
    };
    let (left, right) = tokio::join!(
        policy.pick(&engines, &request),
        policy.pick(&engines, &request)
    );
    assert!(left.is_ok() && right.is_ok());
    assert_eq!(index.calls.load(Ordering::Relaxed), 1);
    let next = PrefixMemo::default();
    policy
        .pick(
            &engines,
            &PickRequest {
                prefix: Some(&next),
                ..request
            },
        )
        .await
        .unwrap();
    assert_eq!(index.calls.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn unavailable_index_falls_back_but_rejected_queries_fail() {
    let engines = [engine("a", 9), engine("b", 0)];
    let model = ModelId("m".into());
    for error in [
        PrefixIndexError::Timeout,
        PrefixIndexError::Unreachable,
        PrefixIndexError::Overloaded,
        PrefixIndexError::QueryTooLarge,
        PrefixIndexError::Rejected(sgl_kv_indexer::RpcCode::InvalidArgument),
    ] {
        let rejected = matches!(error, PrefixIndexError::Rejected(_));
        let (source, index) = remote(Err(error), oracle());
        let policy =
            CacheAwarePolicy::new(source, EngineReportedLoadTable::new(), config()).unwrap();
        let memo = PrefixMemo::default();
        let request = PickRequest {
            prefix: Some(&memo),
            ..request(&model)
        };
        let result = policy.pick(&engines, &request).await;
        if rejected {
            assert!(matches!(result, Err(PickError::InvalidSignal(_))));
        } else {
            assert_eq!(result.unwrap().engine.id.0, "b");
            policy.pick(&engines, &request).await.unwrap();
        }
        assert_eq!(index.calls.load(Ordering::Relaxed), 1);
    }
}

#[tokio::test]
async fn missing_tokens_or_block_size_skip_io_and_bigram_hashes_match_workers() {
    let engines = [engine("a", 0)];
    let model = ModelId("m".into());
    let (source, index) = remote(Ok(PrefixOutcome::Empty), BlockSizeOracle::new());
    let policy = CacheAwarePolicy::new(source, EngineReportedLoadTable::new(), config()).unwrap();
    policy.pick(&engines, &request(&model)).await.unwrap();
    assert_eq!(index.calls.load(Ordering::Relaxed), 0);
    let oracle = oracle();
    oracle.set_bigram(true);
    let (source, index) = remote(Ok(PrefixOutcome::Empty), oracle);
    let policy = CacheAwarePolicy::new(source, EngineReportedLoadTable::new(), config()).unwrap();
    policy
        .pick(&engines, &PickRequest::new(&model, Stage::Plain, 8))
        .await
        .unwrap();
    assert_eq!(index.calls.load(Ordering::Relaxed), 0);
    policy.pick(&engines, &request(&model)).await.unwrap();
    assert_eq!(
        index.hashes.lock().unwrap()[0],
        compute_block_hashes_bigram(&TOKENS, 1)
    );
}

#[tokio::test]
async fn queue_diversion_and_saturation_use_only_this_group() {
    let engines = [engine("owner", 0), engine("cold", 9), engine("outside", 0)];
    let model = ModelId("m".into());
    for (cold_waiting, floor, expected, reason) in [
        (0, None, "cold", "no_cache_candidate"),
        (5, None, "owner", "saturation_pin"),
        (3, Some(2), "owner", "saturation_pin"),
    ] {
        let table = EngineReportedLoadTable::new();
        report(&table, &engines[0], 5, 100, Instant::now());
        report(&table, &engines[1], cold_waiting, 1, Instant::now());
        report(&table, &engines[2], 0, 0, Instant::now());
        let policy = CacheAwarePolicy::new(
            local(&[(&engines[0], 8)]),
            table,
            AffinityConfig {
                worker_queue_limit: Some(4),
                saturation_queue_floor: floor,
                ..config()
            },
        )
        .unwrap();
        let pick = policy.pick(&engines[..2], &request(&model)).await.unwrap();
        assert_eq!(pick.engine.id.0, expected);
        assert_eq!(pick.reason, reason);
    }
}

#[tokio::test]
async fn hard_rejection_never_becomes_cold_fallback_or_saturation_bypass() {
    let engines = [engine("owner", 0), engine("cold", 9)];
    let model = ModelId("m".into());
    for floor in [None, Some(2)] {
        let table = EngineReportedLoadTable::new();
        for engine in &engines {
            report(&table, engine, 5, 100, Instant::now());
        }
        let mut policy = CacheAwarePolicy::new(
            local(&[(&engines[0], 8)]),
            table,
            AffinityConfig {
                worker_queue_limit: Some(4),
                saturation_queue_floor: floor,
                ..config()
            },
        )
        .unwrap();
        let admission = Reject::new("owner");
        policy.admission = admission.clone();
        assert!(matches!(
            policy.pick(&engines, &request(&model)).await,
            Err(PickError::NoAdmissibleEngine(_))
        ));
        assert_eq!(
            *admission.calls.lock().unwrap(),
            vec![("owner".into(), Some(5))]
        );
    }
    let mut policy =
        CacheAwarePolicy::new(local(&[]), EngineReportedLoadTable::new(), config()).unwrap();
    policy.admission = Reject::new("owner");
    assert!(matches!(
        policy.pick(&engines, &request(&model)).await,
        Err(PickError::AdmissionRejected(_))
    ));
}

#[tokio::test]
async fn guard_switches_near_ties_only_with_complete_fresh_telemetry() {
    let engines = [engine("deep", 0), engine("shallow", 9)];
    let model = ModelId("m".into());
    for (margin, stale, expected) in [(0, false, "deep"), (1, false, "shallow"), (1, true, "deep")]
    {
        let table = EngineReportedLoadTable::new();
        report(&table, &engines[0], 5, 100, Instant::now());
        report(
            &table,
            &engines[1],
            1,
            1,
            Instant::now()
                - if stale {
                    Duration::from_secs(3600)
                } else {
                    Duration::ZERO
                },
        );
        let policy = CacheAwarePolicy::new(
            local(&[(&engines[0], 8), (&engines[1], 7)]),
            table,
            AffinityConfig {
                cache_switch_margin_tokens: margin,
                pressure_abs_threshold_tokens: 10,
                ..config()
            },
        )
        .unwrap();
        assert_eq!(
            policy
                .pick(&engines, &request(&model))
                .await
                .unwrap()
                .engine
                .id
                .0,
            expected
        );
    }
}

#[tokio::test]
async fn candidate_cap_is_applied_before_admission() {
    let engines = [engine("deep", 9), engine("shallow", 0)];
    let model = ModelId("m".into());
    let mut policy = CacheAwarePolicy::new(
        local(&[(&engines[0], 8), (&engines[1], 7)]),
        EngineReportedLoadTable::new(),
        AffinityConfig {
            cache_candidate_min_workers: 1,
            cache_candidate_max_workers: 1,
            ..config()
        },
    )
    .unwrap();
    let admission = Reject::new("deep");
    policy.admission = admission.clone();
    assert!(matches!(
        policy.pick(&engines, &request(&model)).await,
        Err(PickError::NoAdmissibleEngine(_))
    ));
    assert_eq!(admission.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn decode_group_and_invalid_configuration_are_rejected() {
    let source = local(&[]);
    for invalid in [
        AffinityConfig {
            cache_candidate_min_workers: 0,
            ..config()
        },
        AffinityConfig {
            cache_candidate_ratio: f64::NAN,
            ..config()
        },
        AffinityConfig {
            saturation_queue_floor: Some(1),
            ..config()
        },
    ] {
        assert!(matches!(
            CacheAwarePolicy::new(source.clone(), EngineReportedLoadTable::new(), invalid),
            Err(PickError::InvalidConfiguration(_))
        ));
    }
    let policy =
        Arc::new(CacheAwarePolicy::new(source, EngineReportedLoadTable::new(), config()).unwrap());
    let model = ModelId("m".into());
    assert!(matches!(
        policy
            .pick(
                &[engine("a", 0)],
                &PickRequest::new(&model, Stage::Decode, 8)
            )
            .await,
        Err(PickError::InvalidConfiguration(_))
    ));
    let pd = |prefill: Arc<dyn Policy>, decode: Arc<dyn Policy>| {
        Bucket::new(
            "pd",
            BucketGroups::Pd {
                prefill: EngineGroup::new(prefill),
                decode: EngineGroup::new(decode),
            },
        )
    };
    let load = Arc::new(PowerOfTwoPolicy::new(EngineReportedLoadTable::new()));
    assert!(matches!(
        BucketResolver::new(vec![pd(load.clone(), policy.clone())]),
        Err(PickError::InvalidConfiguration(_))
    ));
    assert!(BucketResolver::new(vec![pd(policy, load)]).is_ok());
}
