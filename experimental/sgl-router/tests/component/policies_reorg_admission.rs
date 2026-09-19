// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{
    AdmissionLoad, AllOf, Capacity, Decision, EngineAdmission, InFlightLimit, QueueLimit,
};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies_reorg::{PickError, PickRequest, Policy, Stage};
use sgl_router::state::load_monitor::engine_load::{
    EngineLoadTable, LoadStat, NativeCacheRankLoad,
};
use sgl_router::workers::Worker;

fn worker() -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("w".into()),
        url: "http://w".into(),
        mode: Stage::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

fn report() -> LoadStat {
    LoadStat {
        num_running_reqs: 1,
        num_waiting_reqs: 2,
        num_tokens: 10,
        max_total_num_tokens: 100,
        native_cache: Some(NativeCacheRankLoad {
            num_waiting_uncached_tokens: 5,
            num_total_tokens: 80,
            max_running_requests: 3,
            total_prefill_uncached_tokens: 0,
            total_prefill_busy_us: 0,
        }),
    }
}

#[tokio::test]
async fn capacity_uses_total_kv_and_peak_tokens_and_enforces_running_limit() {
    let engine = worker();
    let model = ModelId("m".into());
    for (peak, running, total, capacity, allowed) in [
        (Some(20), 1, 80, 100, true),
        (Some(21), 1, 80, 100, false),
        (None, 1, 80, 100, true),
        (None, 3, 80, 100, false),
        (Some(u64::MAX), 1, 80, u64::MAX, false),
    ] {
        let table = EngineLoadTable::new();
        let mut sample = report();
        sample.num_running_reqs = running;
        sample.max_total_num_tokens = capacity;
        sample.native_cache.as_mut().unwrap().num_total_tokens = total;
        table.set(&engine.url, 0, sample, Instant::now());
        let mut policy = PowerOfTwoPolicy::new(table);
        policy.admission = Arc::new(Capacity);
        let mut request = PickRequest::new(&model, Stage::Plain, 10);
        request.expected_peak_tokens = peak;
        let result = policy.pick(std::slice::from_ref(&engine), &request).await;
        if allowed {
            assert!(result.is_ok());
        } else {
            assert!(
                matches!(result, Err(PickError::AdmissionRejected(reason)) if reason.reason == "kv_capacity")
            );
        }
    }
}

#[test]
fn checks_use_supplied_reports_and_compose_in_order() {
    let engine = worker();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let table = EngineLoadTable::new();
    table.set(&engine.url, 0, report(), Instant::now());
    let snapshot = table.capture_snapshot(Instant::now());
    let load = AdmissionLoad {
        reported: snapshot.fresh_load_for_url(&engine.url),
        native: snapshot.fresh_native_cache_load_for_url(&engine.url),
    };
    // A newer report cannot change checks against the selected observation.
    let mut newer = report();
    newer.num_waiting_reqs = 0;
    table.set(&engine.url, 0, newer, Instant::now());
    assert_eq!(
        QueueLimit(2).check(&engine, &request, load).unwrap(),
        Decision::Reject("queue_limit".into())
    );
    assert_eq!(
        QueueLimit(3).check(&engine, &request, load).unwrap(),
        Decision::Allow
    );
    let all = AllOf(vec![
        Arc::new(Capacity),
        Arc::new(InFlightLimit(1)),
        Arc::new(QueueLimit(2)),
    ]);
    assert_eq!(
        all.check(&engine, &request, load).unwrap(),
        Decision::Reject("queue_limit".into())
    );
    let _busy = engine.load_guard();
    assert_eq!(
        all.check(&engine, &request, load).unwrap(),
        Decision::Reject("in_flight_limit".into())
    );
    assert_eq!(
        Capacity
            .check(&engine, &request, AdmissionLoad::default())
            .unwrap(),
        Decision::Allow
    );
    assert_eq!(
        QueueLimit(0)
            .check(&engine, &request, AdmissionLoad::default())
            .unwrap(),
        Decision::Allow
    );
}

#[test]
fn all_of_stops_on_rejection_or_invalid_signal() {
    #[derive(Debug)]
    struct Invalid;
    impl EngineAdmission for Invalid {
        fn check(
            &self,
            _: &Worker,
            _: &PickRequest<'_>,
            _: AdmissionLoad<'_>,
        ) -> Result<Decision, PickError> {
            Err(PickError::InvalidSignal("test".into()))
        }
    }
    #[derive(Debug)]
    struct Unexpected;
    impl EngineAdmission for Unexpected {
        fn check(
            &self,
            _: &Worker,
            _: &PickRequest<'_>,
            _: AdmissionLoad<'_>,
        ) -> Result<Decision, PickError> {
            panic!("composition must stop after the first rejection or error")
        }
    }
    let engine = worker();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let rejected = AllOf(vec![Arc::new(InFlightLimit(0)), Arc::new(Unexpected)]);
    assert_eq!(
        rejected
            .check(&engine, &request, AdmissionLoad::default())
            .unwrap(),
        Decision::Reject("in_flight_limit".into())
    );
    let invalid = AllOf(vec![Arc::new(Invalid), Arc::new(Unexpected)]);
    assert!(matches!(
        invalid.check(&engine, &request, AdmissionLoad::default()),
        Err(PickError::InvalidSignal(_))
    ));
    assert_eq!(
        AllOf(vec![])
            .check(&engine, &request, AdmissionLoad::default())
            .unwrap(),
        Decision::Allow
    );
}
