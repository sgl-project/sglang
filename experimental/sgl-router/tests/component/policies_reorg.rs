// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::BoxFuture;
use sgl_router::buckets_reorg::{
    Bucket, BucketResolver, Pool, Pools, SelectionRequest, SloPreference, TokenLimits,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{Admission, Decision, EngineAdmission, Placement};
use sgl_router::policies_reorg::affinity::{AffinityKind, AffinityPolicy};
use sgl_router::policies_reorg::{Pick, PickError, PickRequest, Policy, Stage};
use sgl_router::server::metrics::MetricsRegistry;
use sgl_router::state::engine_load::EngineLoadTable;
use sgl_router::state::{AffinityStore, LoadView};
use sgl_router::workers::{Worker, WorkerRegistry};

#[derive(Debug, Default)]
struct TestPolicy {
    admission: Admission,
    result: Option<Arc<Worker>>,
    miss: bool,
    invalid: bool,
    calls: Mutex<Vec<String>>,
}

impl Policy for TestPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            self.calls.lock().unwrap().push(request.bucket.to_owned());
            if self.invalid {
                return Err(PickError::InvalidSignal("test signal".into()));
            }
            if self.miss {
                return Err(PickError::NoCandidates);
            }
            let admitted = self.admission.admit(engines, request)?;
            let engine = self.result.clone().unwrap_or_else(|| admitted[0].clone());
            self.admission.verify(
                Pick {
                    engine,
                    reason: "test",
                },
                request,
            )
        })
    }
}

#[derive(Debug)]
struct Reject(&'static str);

impl EngineAdmission for Reject {
    fn check(&self, engine: &Worker, _: &PickRequest<'_>) -> Result<Decision, PickError> {
        Ok(if engine.id.0 == self.0 {
            Decision::Reject("full".into())
        } else {
            Decision::Allow
        })
    }
}

fn spec(id: &str, mode: Stage, model: &str) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}"),
        mode,
        model_ids: vec![ModelId(model.into())],
        bootstrap_port: None,
    }
}

fn registry() -> Arc<WorkerRegistry> {
    let workers = Arc::new(WorkerRegistry::default());
    for (id, mode, model) in [
        ("a", Stage::Plain, "m"),
        ("b", Stage::Plain, "m"),
        ("unhealthy", Stage::Plain, "m"),
        ("other", Stage::Plain, "other"),
        ("p", Stage::Prefill, "pd"),
        ("d", Stage::Decode, "pd"),
    ] {
        workers.add(spec(id, mode, model)).unwrap();
    }
    let unhealthy = workers.get(&WorkerId("unhealthy".into())).unwrap();
    for _ in 0..3 {
        unhealthy.breaker.record_failure();
    }
    workers
}

fn implicit() -> Pool {
    Pool::implicit(Arc::new(TestPolicy::default()))
}

fn pool(buckets: Vec<Bucket>) -> Pool {
    Pool {
        buckets,
        ..Default::default()
    }
}

fn bucket(id: &str, rank: u32, members: &[&str], policy: Arc<dyn Policy>) -> Bucket {
    Bucket {
        id: id.into(),
        rank,
        worker_ids: Some(members.iter().map(|id| WorkerId((*id).into())).collect()),
        limits: TokenLimits::default(),
        ttft_ms: None,
        tokens_per_second: None,
        policy,
        pending_prefill_budget: None,
    }
}

fn request<'a>(model: &'a ModelId, stage: Stage, load: &'a LoadView<'a>) -> SelectionRequest<'a> {
    SelectionRequest::new(PickRequest::new(model, stage, 10, load))
}

#[tokio::test]
async fn pools_isolate_model_health_and_stage() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("m".into());
    let request = request(&model, Stage::Plain, &load);
    let plain = BucketResolver::new(registry(), Pools::plain(implicit()));
    let groups = plain.ordered_groups(&request).unwrap();
    assert_eq!(
        groups[0]
            .engines
            .iter()
            .map(|w| w.id.0.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(plain.pick(&request).await.unwrap().engine.id.0, "a");

    let pd_model = ModelId("pd".into());
    let pd = BucketResolver::new(
        registry(),
        Pools {
            prefill: implicit(),
            decode: implicit(),
            ..Default::default()
        },
    );
    for (stage, id) in [(Stage::Prefill, "p"), (Stage::Decode, "d")] {
        let request = self::request(&pd_model, stage, &load);
        assert_eq!(pd.pick(&request).await.unwrap().engine.id.0, id);
    }
    // A plain deployment has no decode pool.
    let decode = self::request(&pd_model, Stage::Decode, &load);
    assert!(matches!(
        plain.pick(&decode).await,
        Err(PickError::NoCandidates)
    ));
}

#[test]
fn groups_order_by_slo_then_rank_and_id_and_filter_limits() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("m".into());
    let mut request = request(&model, Stage::Plain, &load);
    request.ttft_ms = Some(100);
    let policy = Arc::new(TestPolicy::default());
    let mut fast = bucket("fast", 10, &["a", "other", "unhealthy"], policy.clone());
    fast.ttft_ms = Some(50);
    let mut small = bucket("small", 0, &["a"], policy.clone());
    small.limits.max = Some(9);
    let mut context = bucket("context", 0, &["a"], policy.clone());
    context.limits.context = Some(9);
    let buckets = vec![
        fast,
        small,
        context,
        bucket("z", 1, &["b"], policy.clone()),
        bucket("a", 1, &["a"], policy.clone()),
        bucket("empty", 0, &["missing"], policy),
    ];
    let mut resolver = BucketResolver::new(registry(), Pools::plain(pool(buckets)));
    for (slo, expected) in [
        (SloPreference::Disabled, vec!["a", "z", "fast"]),
        (SloPreference::SloFirst, vec!["fast", "a", "z"]),
        (SloPreference::BestEffort, vec!["a", "z", "fast"]),
    ] {
        resolver.pools.prefill.slo = slo;
        let groups = resolver.ordered_groups(&request).unwrap();
        assert_eq!(
            groups.iter().map(|g| g.bucket).collect::<Vec<_>>(),
            expected
        );
        assert!(groups.iter().all(|g| g.engines.len() == 1));
    }
}

#[test]
fn decode_unknown_output_uses_only_unbounded_sequence_ranges() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("pd".into());
    let mut request = request(&model, Stage::Decode, &load);
    let policy = Arc::new(TestPolicy::default());
    let mut bounded = bucket("bounded", 0, &["d"], policy.clone());
    bounded.limits = TokenLimits {
        min: Some(10),
        max: Some(20),
        context: Some(20),
    };
    let mut catch_all = bucket("catch-all", 1, &["d"], policy);
    catch_all.limits.context = Some(30);
    let resolver = BucketResolver::new(
        registry(),
        Pools {
            prefill: implicit(),
            decode: pool(vec![bounded, catch_all]),
            ..Default::default()
        },
    );
    assert_eq!(
        resolver.ordered_groups(&request).unwrap()[0].bucket,
        "catch-all"
    );
    request.pick.expected_peak_tokens = Some(20);
    assert_eq!(resolver.ordered_groups(&request).unwrap().len(), 2);
    request.pick.expected_peak_tokens = Some(31);
    assert!(resolver.ordered_groups(&request).unwrap().is_empty());
    request.pick.expected_peak_tokens = Some(9);
    assert!(matches!(
        resolver.ordered_groups(&request),
        Err(PickError::InvalidSignal(_))
    ));
}

#[tokio::test]
async fn rejection_advances_once_or_stops_without_relaxing_admission() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("m".into());
    let request = request(&model, Stage::Plain, &load);
    for placement in [Placement::BeforeSelection, Placement::AfterSelection] {
        let first = Arc::new(TestPolicy {
            admission: Admission {
                check: Arc::new(Reject("a")),
                placement,
            },
            ..Default::default()
        });
        let second = Arc::new(TestPolicy::default());
        let mut resolver = BucketResolver::new(
            registry(),
            Pools::plain(pool(vec![
                bucket("first", 0, &["a"], first.clone()),
                bucket("second", 1, &["b"], second.clone()),
            ])),
        );
        assert_eq!(resolver.pick(&request).await.unwrap().engine.id.0, "b");
        assert_eq!(*first.calls.lock().unwrap(), ["first"]);
        resolver.fallback_on_rejection = false;
        assert!(resolver.pick(&request).await.is_err());
        assert_eq!(second.calls.lock().unwrap().len(), 1);
        resolver.fallback_on_rejection = true;
        resolver.workers.remove(&WorkerId("b".into()));
        let Err(PickError::NoAdmissibleEngine(reasons)) = resolver.pick(&request).await else {
            panic!("expected admission exhaustion")
        };
        assert_eq!(reasons.len(), 1);
        assert_eq!(reasons[0].reason, "full");
    }
}

#[tokio::test]
async fn misses_advance_but_invalid_signals_and_foreign_picks_stop() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("m".into());
    let request = request(&model, Stage::Plain, &load);
    for (first, advances) in [
        (
            TestPolicy {
                miss: true,
                ..Default::default()
            },
            true,
        ),
        (
            TestPolicy {
                invalid: true,
                ..Default::default()
            },
            false,
        ),
        // Even a recreated worker with the same ID is outside the supplied set.
        (
            TestPolicy {
                result: Some(Arc::new(Worker::new(spec("a", Stage::Plain, "m")))),
                ..Default::default()
            },
            false,
        ),
    ] {
        let second = Arc::new(TestPolicy::default());
        let resolver = BucketResolver::new(
            registry(),
            Pools::plain(pool(vec![
                bucket("first", 0, &["a"], Arc::new(first)),
                bucket("second", 1, &["b"], second.clone()),
            ])),
        );
        assert_eq!(resolver.pick(&request).await.is_ok(), advances);
        assert_eq!(second.calls.lock().unwrap().len(), usize::from(advances));
    }
}

#[tokio::test]
async fn admission_placement_changes_whether_an_alternative_can_win() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10, &load);
    let engines = [
        Arc::new(Worker::new(spec("a", Stage::Plain, "m"))),
        Arc::new(Worker::new(spec("b", Stage::Plain, "m"))),
    ];
    let mut policy = TestPolicy {
        admission: Admission {
            check: Arc::new(Reject("a")),
            placement: Placement::BeforeSelection,
        },
        ..Default::default()
    };
    assert_eq!(
        policy.pick(&engines, &request).await.unwrap().engine.id.0,
        "b"
    );
    policy.admission.placement = Placement::AfterSelection;
    assert!(matches!(
        policy.pick(&engines, &request).await,
        Err(PickError::AdmissionRejected(_))
    ));
    assert!(matches!(
        policy.pick(&[], &request).await,
        Err(PickError::NoCandidates)
    ));
}

#[tokio::test]
async fn affinity_group_is_probed_first_and_preserve_keeps_a_stranded_binding() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("m".into());
    let store = AffinityStore::new(Duration::from_secs(60));
    let session = || -> Arc<dyn Policy> {
        Arc::new(AffinityPolicy {
            kind: AffinityKind::Session,
            admission: Admission::default(),
            store: store.clone(),
            global: true,
            fallback: Arc::new(TestPolicy::default()),
            metrics: MetricsRegistry::new(),
        })
    };
    let mut small = bucket("small", 0, &["a"], session());
    small.limits.max = Some(5);
    let mut pool = pool(vec![small, bucket("big", 1, &["b"], session())]);
    pool.affinity = Some(session());
    pool.preserve_global_binding = true;
    let resolver = BucketResolver::new(registry(), Pools::plain(pool));
    let mut request = request(&model, Stage::Plain, &load);
    request.pick.session_key = Some("s");
    // Only "big" fits 10 tokens: the session binds to b.
    assert_eq!(resolver.pick(&request).await.unwrap().engine.id.0, "b");
    // "small" now fits first, but the affinity group returns the bound engine.
    request.pick.input_tokens = 3;
    let pick = resolver.pick(&request).await.unwrap();
    assert!(pick.engine.id.0 == "b" && pick.reason == "session_primary");
    // The bound engine leaves: later buckets serve without rebinding.
    resolver.workers.remove(&WorkerId("b".into()));
    let pick = resolver.pick(&request).await.unwrap();
    assert!(pick.engine.id.0 == "a" && pick.reason == "test");
    assert_eq!(store.binding("Plain/session/global/s").unwrap().0, "b");
}
