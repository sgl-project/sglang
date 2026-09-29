// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use sgl_router::buckets_reorg::{
    Bucket, BucketGroups, BucketRequest, BucketResolver, EngineGroup, TokenLimits,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{
    AdmissionLimits, Decision, EngineAdmission, EngineMetrics,
};
use sgl_router::policies_reorg::{Pick, PickError, PickRequest, Policy, Rejection, Stage};
use sgl_router::workers::{Worker, WorkerRegistry};

#[derive(Debug)]
struct TestPolicy {
    admission: Arc<dyn EngineAdmission>,
    result: Option<Arc<Worker>>,
    miss: bool,
    invalid: bool,
    calls: Mutex<Vec<String>>,
}

impl Default for TestPolicy {
    fn default() -> Self {
        Self {
            admission: Arc::new(AdmissionLimits::default()),
            result: None,
            miss: false,
            invalid: false,
            calls: Mutex::new(Vec::new()),
        }
    }
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
            if engines.is_empty() {
                return Err(PickError::NoCandidates);
            }
            let engine = self.result.clone().unwrap_or_else(|| engines[0].clone());
            if let Decision::Reject(reason) =
                self.admission.check(&engine, &EngineMetrics::default())?
            {
                return Err(PickError::AdmissionRejected(Rejection {
                    engine: engine.id.clone(),
                    reason,
                }));
            }
            Ok(Pick {
                engine,
                reason: "test",
            })
        })
    }
}

#[derive(Debug)]
struct Reject(&'static str);

impl EngineAdmission for Reject {
    fn check(&self, engine: &Worker, _: &EngineMetrics) -> Result<Decision, PickError> {
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

fn group(members: &[&str], policy: Arc<dyn Policy>) -> EngineGroup {
    EngineGroup {
        worker_ids: Some(members.iter().map(|id| WorkerId((*id).into())).collect()),
        policy,
    }
}

fn bucket(id: &str, max: Option<u64>, policy: Arc<dyn Policy>) -> Bucket {
    let mut bucket = Bucket::new(id, BucketGroups::Plain(EngineGroup::new(policy)));
    bucket.limits.max = max;
    bucket
}

#[tokio::test]
async fn groups_isolate_model_health_stage_and_membership() {
    let workers = registry();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let group = EngineGroup::new(Arc::new(TestPolicy::default()));
    assert_eq!(
        group.pick(&workers, &request).await.unwrap().engine.id.0,
        "a"
    );
    workers.remove(&WorkerId("a".into()));
    assert_eq!(
        group.pick(&workers, &request).await.unwrap().engine.id.0,
        "b"
    );
    workers.remove(&WorkerId("b".into()));
    assert!(matches!(
        group.pick(&workers, &request).await,
        Err(PickError::NoCandidates)
    ));

    let pd = ModelId("pd".into());
    for (stage, expected) in [(Stage::Prefill, "p"), (Stage::Decode, "d")] {
        let request = PickRequest::new(&pd, stage, 10);
        let group = self::group(
            &["p", "d", "other", "unhealthy"],
            Arc::new(TestPolicy::default()),
        );
        assert_eq!(
            group.pick(&workers, &request).await.unwrap().engine.id.0,
            expected
        );
    }
    let empty = self::group(&[], Arc::new(TestPolicy::default()));
    assert!(matches!(
        empty.pick(&workers, &request).await,
        Err(PickError::NoCandidates)
    ));
}

#[test]
fn resolve_orders_all_length_fits_by_capacity_rank_and_id() {
    let policy = Arc::new(TestPolicy::default());
    let mut z = bucket("z", Some(20), policy.clone());
    z.rank = 1;
    let mut a = bucket("a", Some(20), policy.clone());
    a.rank = 1;
    let mut later = bucket("later", Some(20), policy.clone());
    later.rank = 2;
    let mut min = bucket("min", Some(15), policy.clone());
    min.limits.min = Some(11);
    let resolver = BucketResolver::new(vec![
        bucket("catch-all", None, policy.clone()),
        bucket("too-small", Some(9), policy),
        z,
        later,
        a,
        min,
    ])
    .unwrap();
    assert_eq!(
        resolver
            .resolve(10, None, None, None)
            .unwrap()
            .iter()
            .map(|bucket| bucket.id.as_str())
            .collect::<Vec<_>>(),
        ["a", "z", "later", "catch-all"]
    );
    assert_eq!(resolver.resolve(11, None, None, None).unwrap()[0].id, "min");
    assert_eq!(resolver.resolve(15, None, None, None).unwrap()[0].id, "min");
    assert_eq!(resolver.resolve(20, None, None, None).unwrap()[0].id, "a");
    assert_eq!(
        resolver.resolve(21, None, None, None).unwrap()[0].id,
        "catch-all"
    );
}

#[test]
fn context_capacity_checks_peak_when_known_and_input_otherwise() {
    let policy = Arc::new(TestPolicy::default());
    let mut short = bucket("short", None, policy.clone());
    short.max_context_tokens = Some(20);
    let mut long = bucket("long", None, policy);
    long.max_context_tokens = Some(30);
    let resolver = BucketResolver::new(vec![long, short]).unwrap();
    assert_eq!(
        resolver.resolve(10, None, None, None).unwrap()[0].id,
        "short"
    );
    assert_eq!(
        resolver.resolve(10, Some(20), None, None).unwrap()[0].id,
        "short"
    );
    assert_eq!(
        resolver.resolve(10, Some(21), None, None).unwrap()[0].id,
        "long"
    );
    assert!(resolver
        .resolve(10, Some(31), None, None)
        .unwrap()
        .is_empty());
    assert!(resolver.resolve(31, None, None, None).unwrap().is_empty());
    assert!(matches!(
        resolver.resolve(10, Some(9), None, None),
        Err(PickError::InvalidSignal(_))
    ));
    assert!(BucketResolver::default()
        .resolve(1, None, None, None)
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn selected_pd_bucket_owns_both_memberships_and_policies() {
    let workers = registry();
    workers.add(spec("p2", Stage::Prefill, "pd")).unwrap();
    workers.add(spec("d2", Stage::Decode, "pd")).unwrap();
    let model = ModelId("pd".into());
    let prefill_policy = Arc::new(TestPolicy::default());
    let decode_policy = Arc::new(TestPolicy::default());
    let resolver = BucketResolver::new(vec![Bucket::new(
        "shared",
        BucketGroups::Pd {
            prefill: group(&["p2", "d", "a"], prefill_policy.clone()),
            decode: group(&["d2", "p", "other"], decode_policy.clone()),
        },
    )])
    .unwrap();
    let bucket = resolver.resolve(10, Some(20), None, None).unwrap()[0];
    let request = BucketRequest {
        prefix: None,
        model: &model,
        input_tokens: 10,
        expected_peak_tokens: Some(20),
        token_ids: None,
        session_key: None,
        routing_key: None,
    };
    let picks = bucket.pick_engines(&workers, &request).await.unwrap();
    assert_eq!(picks.prefill.engine.id.0, "p2");
    assert_eq!(picks.decode.unwrap().engine.id.0, "d2");
    assert_eq!(*prefill_policy.calls.lock().unwrap(), ["shared"]);
    assert_eq!(*decode_policy.calls.lock().unwrap(), ["shared"]);
}

#[tokio::test]
async fn resolver_includes_empty_groups_without_invoking_policies() {
    let workers = registry();
    let model = ModelId("m".into());
    let policy = Arc::new(TestPolicy::default());
    let mut empty = Bucket::new(
        "empty",
        BucketGroups::Plain(group(&["missing"], policy.clone())),
    );
    empty.limits = TokenLimits {
        min: None,
        max: Some(10),
    };
    let resolver =
        BucketResolver::new(vec![empty, bucket("available", Some(20), policy.clone())]).unwrap();
    let buckets = resolver.resolve(10, None, None, None).unwrap();
    assert_eq!(
        buckets
            .iter()
            .map(|bucket| bucket.id.as_str())
            .collect::<Vec<_>>(),
        ["empty", "available"]
    );
    let bucket = buckets[0];
    let BucketGroups::Plain(group) = &bucket.groups else {
        panic!("expected plain")
    };
    let request = PickRequest {
        bucket: &bucket.id,
        ..PickRequest::new(&model, Stage::Plain, 10)
    };
    assert!(matches!(
        group.pick(&workers, &request).await,
        Err(PickError::NoCandidates)
    ));
    assert!(policy.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn group_propagates_rejections_misses_and_invalid_signals() {
    let workers = registry();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let rejected = group(
        &["a"],
        Arc::new(TestPolicy {
            admission: Arc::new(Reject("a")),
            ..Default::default()
        }),
    );
    assert!(matches!(
        rejected.pick(&workers, &request).await,
        Err(PickError::AdmissionRejected(_))
    ));
    let miss = group(
        &["a"],
        Arc::new(TestPolicy {
            miss: true,
            ..Default::default()
        }),
    );
    assert!(matches!(
        miss.pick(&workers, &request).await,
        Err(PickError::NoCandidates)
    ));
    let invalid = group(
        &["a"],
        Arc::new(TestPolicy {
            invalid: true,
            ..Default::default()
        }),
    );
    assert!(matches!(
        invalid.pick(&workers, &request).await,
        Err(PickError::InvalidSignal(_))
    ));
}

#[tokio::test]
async fn group_rejects_foreign_pick_even_with_same_worker_id() {
    let workers = registry();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let group = group(
        &["a"],
        Arc::new(TestPolicy {
            result: Some(Arc::new(Worker::new(spec("a", Stage::Plain, "m")))),
            ..Default::default()
        }),
    );
    assert!(matches!(
        group.pick(&workers, &request).await,
        Err(PickError::OutsideCandidates(_))
    ));
}

#[tokio::test]
async fn selected_engine_rejection_does_not_try_an_alternative() {
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let engines = [
        Arc::new(Worker::new(spec("a", Stage::Plain, "m"))),
        Arc::new(Worker::new(spec("b", Stage::Plain, "m"))),
    ];
    let policy = TestPolicy {
        admission: Arc::new(Reject("a")),
        ..Default::default()
    };
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
async fn bucket_scopes_plain_pick_and_preserves_request_facts() {
    #[derive(Debug)]
    struct InspectRequest;

    impl Policy for InspectRequest {
        fn pick<'a>(
            &'a self,
            engines: &'a [Arc<Worker>],
            request: &'a PickRequest<'a>,
        ) -> BoxFuture<'a, Result<Pick, PickError>> {
            Box::pin(async move {
                assert_eq!(request.model.0, "m");
                assert_eq!(request.bucket, "plain-bucket");
                assert_eq!(request.stage, Stage::Plain);
                assert_eq!(request.input_tokens, 2);
                assert_eq!(request.expected_peak_tokens, Some(12));
                assert_eq!(request.token_ids, Some([7, 9].as_slice()));
                assert_eq!(request.session_key, Some("session"));
                assert_eq!(request.routing_key, Some("routing"));
                Ok(Pick {
                    engine: engines[0].clone(),
                    reason: "inspected",
                })
            })
        }
    }

    let workers = registry();
    let model = ModelId("m".into());
    let bucket = Bucket::new(
        "plain-bucket",
        BucketGroups::Plain(group(&["b"], Arc::new(InspectRequest))),
    );
    let request = BucketRequest {
        prefix: None,
        model: &model,
        input_tokens: 2,
        expected_peak_tokens: Some(12),
        token_ids: Some(&[7, 9]),
        session_key: Some("session"),
        routing_key: Some("routing"),
    };
    let picks = bucket.pick_engines(&workers, &request).await.unwrap();
    assert_eq!(picks.prefill.engine.id.0, "b");
    assert_eq!(picks.prefill.reason, "inspected");
    assert!(picks.decode.is_none());
}

#[tokio::test]
async fn power_of_two_checks_selected_engine_and_propagates_rejection_without_fallback() {
    use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
    use sgl_router::state::load_monitor::engine_reported_load::EngineReportedLoadTable;

    #[derive(Debug)]
    struct Check {
        calls: Mutex<Vec<WorkerId>>,
        reject: bool,
        invalid: bool,
    }

    impl EngineAdmission for Check {
        fn check(&self, engine: &Worker, _: &EngineMetrics) -> Result<Decision, PickError> {
            self.calls.lock().unwrap().push(engine.id.clone());
            if self.invalid {
                Err(PickError::InvalidSignal("admission input".into()))
            } else if self.reject {
                Ok(Decision::Reject("full".into()))
            } else {
                Ok(Decision::Allow)
            }
        }
    }

    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let engine = Arc::new(Worker::new(spec("a", Stage::Plain, "m")));
    let other = Arc::new(Worker::new(spec("b", Stage::Plain, "m")));
    let _busy = other.load_guard();
    for engines in [vec![engine.clone()], vec![other.clone(), engine.clone()]] {
        for (reject, invalid) in [(false, false), (true, false), (false, true)] {
            let check = Arc::new(Check {
                calls: Mutex::new(Vec::new()),
                reject,
                invalid,
            });
            let mut policy = PowerOfTwoPolicy::new(EngineReportedLoadTable::new());
            policy.admission = check.clone();
            assert!(matches!(
                policy.pick(&[], &request).await,
                Err(PickError::NoCandidates)
            ));
            assert!(check.calls.lock().unwrap().is_empty());
            let result = policy.pick(&engines, &request).await;
            if invalid {
                assert!(matches!(result, Err(PickError::InvalidSignal(_))));
            } else if reject {
                assert!(matches!(result, Err(PickError::AdmissionRejected(reason))
                if reason.engine == engine.id && reason.reason == "full"));
            } else {
                assert!(Arc::ptr_eq(&result.unwrap().engine, &engine));
            }
            assert_eq!(
                check.calls.lock().unwrap().as_slice(),
                std::slice::from_ref(&engine.id)
            );
        }
    }
}
