// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use sgl_router::buckets_reorg::{Bucket, BucketResolver, EngineGroup, TokenLimits};
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{Admission, Decision, EngineAdmission, Placement};
use sgl_router::policies_reorg::{Pick, PickError, PickRequest, Policy, Stage};
use sgl_router::state::load_monitor::engine_load::EngineLoadTable;
use sgl_router::state::LoadView;
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

fn group(rank: u32, members: &[&str], policy: Arc<dyn Policy>) -> EngineGroup {
    EngineGroup {
        rank,
        worker_ids: Some(members.iter().map(|id| WorkerId((*id).into())).collect()),
        ..EngineGroup::new(policy)
    }
}

fn bucket(id: &str, rank: u32, members: &[&str], policy: Arc<dyn Policy>) -> Bucket {
    Bucket {
        id: id.into(),
        plain: Some(group(rank, members, policy)),
        ..Default::default()
    }
}

fn request<'a>(model: &'a ModelId, stage: Stage, load: &'a LoadView<'a>) -> PickRequest<'a> {
    PickRequest::new(model, stage, 10, load)
}

#[tokio::test]
async fn groups_isolate_model_health_and_stage() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("m".into());
    let request = request(&model, Stage::Plain, &load);
    let plain = BucketResolver::new(
        registry(),
        vec![Bucket {
            id: "default".into(),
            plain: Some(EngineGroup::new(Arc::new(TestPolicy::default()))),
            ..Default::default()
        }],
    );
    assert_eq!(plain.pick(&request).await.unwrap().engine.id.0, "a");
    plain.workers.remove(&WorkerId("a".into()));
    assert_eq!(plain.pick(&request).await.unwrap().engine.id.0, "b");
    plain.workers.remove(&WorkerId("b".into()));
    assert!(matches!(
        plain.pick(&request).await,
        Err(PickError::NoCandidates)
    ));

    let pd_model = ModelId("pd".into());
    let pd = BucketResolver::new(
        registry(),
        vec![Bucket {
            id: "default".into(),
            prefill: Some(EngineGroup::new(Arc::new(TestPolicy::default()))),
            decode: Some(EngineGroup::new(Arc::new(TestPolicy::default()))),
            ..Default::default()
        }],
    );
    for (stage, id) in [(Stage::Prefill, "p"), (Stage::Decode, "d")] {
        let request = self::request(&pd_model, stage, &load);
        assert_eq!(pd.pick(&request).await.unwrap().engine.id.0, id);
        assert!(matches!(
            plain.pick(&request).await,
            Err(PickError::NoCandidates)
        ));
    }
}

#[tokio::test]
async fn buckets_match_length_then_order_by_rank_and_id() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("m".into());
    let request = request(&model, Stage::Plain, &load);
    let policy = Arc::new(TestPolicy::default());
    let mut small = bucket("small", 0, &["a"], policy.clone());
    small.plain.as_mut().unwrap().limits.max = Some(9);
    let mut context = bucket("context", 0, &["a"], policy.clone());
    context.max_context_tokens = Some(9);
    let resolver = BucketResolver::new(
        registry(),
        vec![
            bucket("last", 10, &["b"], policy.clone()),
            small,
            context,
            bucket("z", 1, &["b"], policy.clone()),
            bucket("a", 1, &["a", "other", "unhealthy"], policy.clone()),
            bucket("empty", 0, &["missing"], policy),
        ],
    );
    assert_eq!(
        resolver
            .matching_buckets(&request)
            .unwrap()
            .iter()
            .map(|b| b.id.as_str())
            .collect::<Vec<_>>(),
        ["empty", "a", "z", "last"]
    );
    // Matching buckets can be empty; selection skips them before invoking policies.
    assert_eq!(resolver.pick(&request).await.unwrap().engine.id.0, "a");
}

#[tokio::test]
async fn one_bucket_uses_separate_role_memberships_and_policies() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("pd".into());
    let workers = registry();
    workers.add(spec("p2", Stage::Prefill, "pd")).unwrap();
    workers.add(spec("d2", Stage::Decode, "pd")).unwrap();
    let prefill = Arc::new(TestPolicy::default());
    let decode = Arc::new(TestPolicy::default());
    let resolver = BucketResolver::new(
        workers,
        vec![Bucket {
            id: "shared".into(),
            // Even explicitly listed members must match the request's model and role.
            prefill: Some(group(0, &["p2", "d", "a"], prefill.clone())),
            decode: Some(group(0, &["d2", "p", "other"], decode.clone())),
            ..Default::default()
        }],
    );
    for (stage, expected) in [(Stage::Prefill, "p2"), (Stage::Decode, "d2")] {
        let request = request(&model, stage, &load);
        assert_eq!(resolver.pick(&request).await.unwrap().engine.id.0, expected);
    }
    assert_eq!(*prefill.calls.lock().unwrap(), ["shared"]);
    assert_eq!(*decode.calls.lock().unwrap(), ["shared"]);
    assert!(matches!(
        resolver.pick(&request(&model, Stage::Plain, &load)).await,
        Err(PickError::NoCandidates)
    ));
}

#[test]
fn stages_have_independent_ranges_and_ranks() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("pd".into());
    let policy = Arc::new(TestPolicy::default());
    let mut resolver = BucketResolver::new(
        registry(),
        vec![
            Bucket {
                id: "short".into(),
                prefill: Some(group(0, &["p"], policy.clone())),
                decode: Some(group(1, &["d"], policy.clone())),
                ..Default::default()
            },
            Bucket {
                id: "long".into(),
                prefill: Some(group(1, &["p"], policy.clone())),
                decode: Some(group(0, &["d"], policy)),
                ..Default::default()
            },
        ],
    );
    let prefill = request(&model, Stage::Prefill, &load);
    let mut decode = request(&model, Stage::Decode, &load);
    decode.expected_peak_tokens = Some(20);
    assert_eq!(resolver.matching_buckets(&prefill).unwrap()[0].id, "short");
    assert_eq!(resolver.matching_buckets(&decode).unwrap()[0].id, "long");

    resolver.buckets[0].prefill.as_mut().unwrap().limits.max = Some(10);
    resolver.buckets[0].decode.as_mut().unwrap().limits.max = Some(15);
    resolver.buckets[1].prefill.as_mut().unwrap().limits.min = Some(11);
    resolver.buckets[1].decode.as_mut().unwrap().limits.min = Some(16);
    for (request, expected) in [(&prefill, "short"), (&decode, "long")] {
        let groups = resolver.matching_buckets(request).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, expected);
    }

    // The same bucket context limit applies to both of its role groups.
    resolver.buckets[0].max_context_tokens = Some(9);
    resolver.buckets[1].max_context_tokens = Some(19);
    assert!(resolver.matching_buckets(&prefill).unwrap().is_empty());
    assert!(resolver.matching_buckets(&decode).unwrap().is_empty());
}

#[test]
fn decode_unknown_output_uses_only_unbounded_sequence_ranges() {
    let table = EngineLoadTable::new();
    let load = LoadView::new(&table);
    let model = ModelId("pd".into());
    let mut request = request(&model, Stage::Decode, &load);
    let policy = Arc::new(TestPolicy::default());
    let bounded = Bucket {
        id: "bounded".into(),
        max_context_tokens: Some(20),
        decode: Some(EngineGroup {
            limits: TokenLimits {
                min: Some(10),
                max: Some(20),
            },
            ..group(0, &["d"], policy.clone())
        }),
        ..Default::default()
    };
    let catch_all = Bucket {
        id: "catch-all".into(),
        max_context_tokens: Some(30),
        decode: Some(group(1, &["d"], policy)),
        ..Default::default()
    };
    let resolver = BucketResolver::new(registry(), vec![bounded, catch_all]);
    assert_eq!(
        resolver.matching_buckets(&request).unwrap()[0].id,
        "catch-all"
    );
    request.expected_peak_tokens = Some(20);
    assert_eq!(resolver.matching_buckets(&request).unwrap().len(), 2);
    request.expected_peak_tokens = Some(31);
    assert!(resolver.matching_buckets(&request).unwrap().is_empty());
    request.expected_peak_tokens = Some(9);
    assert!(matches!(
        resolver.matching_buckets(&request),
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
            vec![
                bucket("first", 0, &["a"], first.clone()),
                bucket("second", 1, &["b"], second.clone()),
            ],
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
            vec![
                bucket("first", 0, &["a"], Arc::new(first)),
                bucket("second", 1, &["b"], second.clone()),
            ],
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
