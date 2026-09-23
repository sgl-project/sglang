// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{Decision, EngineAdmission, EngineMetrics};
use sgl_router::policies_reorg::session_aware::SessionAwarePolicy;
use sgl_router::policies_reorg::{PickError, PickRequest, Policy, Stage};
use sgl_router::state::load_monitor::engine_reported_load::{EngineReportedLoadTable, LoadStat};
use sgl_router::state::load_monitor::router_inflight_load::MockClock;
use sgl_router::state::AffinityStore;
use sgl_router::workers::Worker;

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

fn request(model: &ModelId) -> PickRequest<'_> {
    PickRequest {
        bucket: "bucket",
        session_key: Some("session"),
        ..PickRequest::new(model, Stage::Plain, 8)
    }
}

fn policy() -> (SessionAwarePolicy, Arc<AffinityStore>) {
    let store = AffinityStore::new(Duration::from_secs(60));
    (
        SessionAwarePolicy::new(store.clone(), EngineReportedLoadTable::new()),
        store,
    )
}

#[derive(Debug, Default)]
struct Admission {
    reject: AtomicBool,
    invalid: AtomicBool,
    calls: Mutex<Vec<(String, Option<u64>)>>,
}

impl EngineAdmission for Admission {
    fn check(&self, engine: &Worker, metrics: &EngineMetrics) -> Result<Decision, PickError> {
        self.calls
            .lock()
            .unwrap()
            .push((engine.id.0.clone(), metrics.waiting_requests));
        if self.invalid.load(Ordering::Relaxed) {
            return Err(PickError::InvalidSignal("invalid admission input".into()));
        }
        Ok(if self.reject.load(Ordering::Relaxed) {
            Decision::Reject("full".into())
        } else {
            Decision::Allow
        })
    }
}

#[tokio::test]
async fn new_sessions_choose_lower_pressure_then_reuse_binding() {
    let (policy, store) = policy();
    let engines = [engine("a", 0), engine("b", 9)];
    let model = ModelId("m".into());
    let request = request(&model);
    let first = policy.pick(&engines, &request).await.unwrap();
    assert_eq!(first.engine.id.0, "a");
    assert_eq!(first.reason, "assigned");
    engines[0].active_requests.store(100, Ordering::Relaxed);
    for _ in 0..5 {
        let pick = policy.pick(&engines, &request).await.unwrap();
        assert!(Arc::ptr_eq(&pick.engine, &engines[0]));
        assert_eq!(pick.reason, "session_primary");
    }
    assert_eq!(store.len(), 1);
}

#[tokio::test]
async fn missing_and_empty_keys_use_admitted_power_of_two_without_binding() {
    let (mut policy, store) = policy();
    let admission = Arc::new(Admission::default());
    policy.admission = admission.clone();
    let model = ModelId("m".into());
    let engines = [engine("a", 9), engine("b", 0)];
    for key in [None, Some("")] {
        let request = PickRequest {
            session_key: key,
            ..request(&model)
        };
        let pick = policy.pick(&engines, &request).await.unwrap();
        assert_eq!(pick.engine.id.0, "b");
        assert_eq!(pick.reason, "no_session");
        admission.reject.store(true, Ordering::Relaxed);
        assert!(matches!(
            policy.pick(&engines, &request).await,
            Err(PickError::AdmissionRejected(_))
        ));
        admission.reject.store(false, Ordering::Relaxed);
    }
    assert!(store.is_empty());
    assert_eq!(admission.calls.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn rejected_new_and_existing_sessions_never_rebind_or_try_another_engine() {
    let (mut policy, store) = policy();
    let admission = Arc::new(Admission::default());
    policy.admission = admission.clone();
    let engines = [engine("a", 0), engine("b", 9)];
    let model = ModelId("m".into());
    let request = request(&model);
    admission.reject.store(true, Ordering::Relaxed);
    assert!(matches!(
        policy.pick(&engines, &request).await,
        Err(PickError::AdmissionRejected(_))
    ));
    assert!(store.is_empty());
    admission.reject.store(false, Ordering::Relaxed);
    policy.pick(&engines, &request).await.unwrap();
    engines[0].active_requests.store(100, Ordering::Relaxed);
    admission.reject.store(true, Ordering::Relaxed);
    assert!(matches!(
        policy.pick(&engines, &request).await,
        Err(PickError::AdmissionRejected(_))
    ));
    admission.reject.store(false, Ordering::Relaxed);
    assert_eq!(
        policy.pick(&engines, &request).await.unwrap().engine.id.0,
        "a"
    );
    assert_eq!(store.len(), 1);
    assert_eq!(admission.calls.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn removed_binding_is_replaced_only_after_admission() {
    let (mut policy, store) = policy();
    let admission = Arc::new(Admission::default());
    policy.admission = admission.clone();
    let model = ModelId("m".into());
    let request = request(&model);
    let original = [engine("a", 0)];
    let replacement = [engine("b", 0)];
    policy.pick(&original, &request).await.unwrap();
    admission.reject.store(true, Ordering::Relaxed);
    assert!(matches!(
        policy.pick(&replacement, &request).await,
        Err(PickError::AdmissionRejected(_))
    ));
    admission.reject.store(false, Ordering::Relaxed);
    assert_eq!(
        policy.pick(&original, &request).await.unwrap().reason,
        "session_primary"
    );
    assert_eq!(
        policy.pick(&replacement, &request).await.unwrap().reason,
        "assigned"
    );
    assert_eq!(store.len(), 1);
}

#[tokio::test]
async fn bindings_are_scoped_by_model_bucket_role_and_session() {
    let (policy, store) = policy();
    let models = [ModelId("m".into()), ModelId("other".into())];
    let original = [engine("a", 0)];
    let fleet = [original[0].clone(), engine("b", 0)];
    policy.pick(&original, &request(&models[0])).await.unwrap();
    original[0].active_requests.store(100, Ordering::Relaxed);
    let requests = [
        request(&models[1]),
        PickRequest {
            bucket: "other",
            ..request(&models[0])
        },
        PickRequest {
            stage: Stage::Prefill,
            ..request(&models[0])
        },
        PickRequest {
            stage: Stage::Decode,
            ..request(&models[0])
        },
        PickRequest {
            session_key: Some("other"),
            ..request(&models[0])
        },
    ];
    for request in requests {
        let pick = policy.pick(&fleet, &request).await.unwrap();
        assert_eq!(pick.engine.id.0, "b");
        assert_eq!(pick.reason, "assigned");
    }
    assert_eq!(
        policy
            .pick(&fleet, &request(&models[0]))
            .await
            .unwrap()
            .engine
            .id
            .0,
        "a"
    );
    assert_eq!(store.len(), 6);
}

#[tokio::test]
async fn embedded_delimiters_do_not_alias_scopes() {
    let (policy, store) = policy();
    let models = [ModelId("a\0b".into()), ModelId("a".into())];
    let first = PickRequest {
        bucket: "c",
        session_key: Some("d\0e"),
        ..request(&models[0])
    };
    let second = PickRequest {
        bucket: "b\0c",
        session_key: Some("d\0e"),
        ..request(&models[1])
    };
    let third = PickRequest {
        bucket: "c\0d",
        session_key: Some("e"),
        ..request(&models[0])
    };
    let fleet = [engine("a", 0), engine("b", 9)];
    policy.pick(&fleet, &first).await.unwrap();
    fleet[0].active_requests.store(100, Ordering::Relaxed);
    for request in [second, third] {
        assert_eq!(
            policy.pick(&fleet, &request).await.unwrap().engine.id.0,
            "b"
        );
    }
    assert_eq!(store.len(), 3);
}

#[tokio::test]
async fn same_id_replacement_returns_the_live_candidate_instance() {
    let (policy, _) = policy();
    let model = ModelId("m".into());
    let request = request(&model);
    policy.pick(&[engine("a", 0)], &request).await.unwrap();
    let replacement = [engine("a", 0)];
    let pick = policy.pick(&replacement, &request).await.unwrap();
    assert!(Arc::ptr_eq(&pick.engine, &replacement[0]));
    assert_eq!(pick.reason, "session_primary");
}

#[tokio::test]
async fn empty_candidates_skip_admission_and_invalid_signals_never_bind() {
    let (mut policy, store) = policy();
    let admission = Arc::new(Admission::default());
    policy.admission = admission.clone();
    let model = ModelId("m".into());
    let request = request(&model);
    assert!(matches!(
        policy.pick(&[], &request).await,
        Err(PickError::NoCandidates)
    ));
    assert!(admission.calls.lock().unwrap().is_empty());
    admission.invalid.store(true, Ordering::Relaxed);
    assert!(matches!(
        policy.pick(&[engine("a", 0)], &request).await,
        Err(PickError::InvalidSignal(_))
    ));
    assert!(store.is_empty());
}

#[tokio::test]
async fn admission_receives_fresh_load_on_assignment_and_reuse() {
    let store = AffinityStore::new(Duration::from_secs(60));
    let table = EngineReportedLoadTable::new();
    let mut policy = SessionAwarePolicy::new(store, table.clone());
    let admission = Arc::new(Admission::default());
    policy.admission = admission.clone();
    let engines = [engine("a", 0)];
    let model = ModelId("m".into());
    let request = request(&model);
    for waiting in [3, 7] {
        table.set(
            &engines[0].url,
            0,
            LoadStat {
                num_waiting_reqs: waiting,
                num_running_reqs: 1,
                num_tokens: 10,
                max_total_num_tokens: 100,
                native_cache: None,
            },
            Instant::now(),
        );
        policy.pick(&engines, &request).await.unwrap();
    }
    assert_eq!(
        *admission.calls.lock().unwrap(),
        vec![("a".into(), Some(3)), ("a".into(), Some(7))]
    );
}

#[tokio::test]
async fn shared_store_refreshes_active_sessions_and_expires_idle_ones() {
    let clock = Arc::new(MockClock::new(Instant::now()));
    let store = AffinityStore::with_clock(Duration::from_secs(10), clock.clone());
    let policy = SessionAwarePolicy::new(store.clone(), EngineReportedLoadTable::new());
    let model = ModelId("m".into());
    let fleet = [engine("a", 0), engine("b", 9)];
    let hot = request(&model);
    let cold = PickRequest {
        session_key: Some("cold"),
        ..hot
    };
    policy.pick(&fleet, &hot).await.unwrap();
    policy.pick(&fleet, &cold).await.unwrap();
    clock.advance(Duration::from_secs(8));
    fleet[0].active_requests.store(100, Ordering::Relaxed);
    policy.pick(&fleet, &hot).await.unwrap();
    clock.advance(Duration::from_secs(8));
    assert_eq!(store.sweep_expired(), 1);
    assert_eq!(policy.pick(&fleet, &hot).await.unwrap().engine.id.0, "a");
    assert_eq!(policy.pick(&fleet, &cold).await.unwrap().engine.id.0, "b");
}

#[derive(Debug)]
struct RacingAdmission {
    model: ModelId,
    competitor: SessionAwarePolicy,
    winner: Arc<Worker>,
    reject_winner: bool,
    calls: Mutex<Vec<String>>,
}

impl EngineAdmission for RacingAdmission {
    fn check(&self, engine: &Worker, _: &EngineMetrics) -> Result<Decision, PickError> {
        self.calls.lock().unwrap().push(engine.id.0.clone());
        if engine.id.0 == "a" {
            // Complete a competing first request after this request selected a,
            // but before it can commit. The effective binding is now b.
            futures::executor::block_on(
                self.competitor
                    .pick(std::slice::from_ref(&self.winner), &request(&self.model)),
            )?;
        }
        Ok(if self.reject_winner && engine.id == self.winner.id {
            Decision::Reject("racing winner full".into())
        } else {
            Decision::Allow
        })
    }
}

#[tokio::test]
async fn concurrent_assignment_winner_is_checked_and_preserved_on_rejection() {
    for reject_winner in [false, true] {
        let (mut policy, store) = policy();
        let engines = [engine("a", 0), engine("b", 9)];
        let admission = Arc::new(RacingAdmission {
            model: ModelId("m".into()),
            competitor: SessionAwarePolicy::new(store.clone(), EngineReportedLoadTable::new()),
            winner: engines[1].clone(),
            reject_winner,
            calls: Mutex::new(Vec::new()),
        });
        policy.admission = admission.clone();
        let model = ModelId("m".into());
        let request = request(&model);
        let result = policy.pick(&engines, &request).await;
        if reject_winner {
            assert!(
                matches!(result, Err(PickError::AdmissionRejected(rejection)) if rejection.engine.0 == "b")
            );
        } else {
            let pick = result.unwrap();
            assert_eq!(pick.engine.id.0, "b");
            assert_eq!(pick.reason, "session_primary");
        }
        assert_eq!(*admission.calls.lock().unwrap(), vec!["a", "b"]);
        assert_eq!(
            admission
                .competitor
                .pick(&engines, &request)
                .await
                .unwrap()
                .engine
                .id
                .0,
            "b"
        );
        assert_eq!(store.len(), 1);
    }
}
