// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{
    AdmissionConfig, AdmissionState, Decision, EngineAdmission,
};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies_reorg::{PickError, PickRequest, Policy, Stage};
use sgl_router::state::load_monitor::engine_load::{
    EngineLoadTable, LoadStat, NativeCacheRankLoad,
};
use sgl_router::workers::Worker;

fn engine() -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("w".into()),
        url: "http://w".into(),
        mode: Stage::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

#[test]
fn config_requires_exactly_the_limits_for_its_name() {
    let value = json!({
        "name": "running_plus_kv_capacity",
        "max_running_requests": 4,
        "max_kv_tokens": 100,
    });
    let config: AdmissionConfig = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        config,
        AdmissionConfig::RunningPlusKvCapacity {
            max_running_requests: 4,
            max_kv_tokens: 100,
        }
    );
    assert_eq!(serde_json::to_value(config).unwrap(), value);
    assert_eq!(
        serde_json::from_value::<AdmissionConfig>(json!({"name": "allow_all"})).unwrap(),
        AdmissionConfig::default()
    );
    for invalid in [
        json!({"name": "unknown"}),
        json!({"name": "running_plus_kv_capacity", "max_running_requests": 4}),
        json!({"name": "running_plus_kv_capacity", "max_kv_tokens": 100}),
        json!({"name": "allow_all", "max_running_requests": 4}),
        json!({"name": "running_plus_kv_capacity", "max_running_requests": 4,
               "max_kv_tokens": 100, "queue_limit": 3}),
    ] {
        assert!(
            serde_json::from_value::<AdmissionConfig>(invalid.clone()).is_err(),
            "unexpectedly accepted {invalid}"
        );
    }
}

#[test]
fn capacity_checks_projected_usage_and_preserves_unknown_measurements() {
    let engine = engine();
    let model = ModelId("m".into());
    let config = AdmissionConfig::RunningPlusKvCapacity {
        max_running_requests: 4,
        max_kv_tokens: 100,
    };
    for (running, kv, peak, rejection) in [
        (Some(3), Some(80), Some(20), None),
        (Some(4), Some(80), Some(20), Some("running_capacity")),
        (Some(3), Some(80), Some(21), Some("kv_capacity")),
        (Some(3), Some(90), None, None),
        (Some(3), Some(91), None, Some("kv_capacity")),
        (Some(u64::MAX), Some(0), None, Some("running_capacity")),
        (Some(0), Some(1), Some(u64::MAX), Some("kv_capacity")),
        (None, None, None, None),
        (Some(4), None, None, Some("running_capacity")),
        (None, Some(100), None, Some("kv_capacity")),
    ] {
        let mut request = PickRequest::new(&model, Stage::Plain, 10);
        request.expected_peak_tokens = peak;
        let state = AdmissionState {
            running_requests: running,
            kv_tokens: kv,
        };
        assert_eq!(
            config.check(&engine, &request, state).unwrap(),
            rejection.map_or(Decision::Allow, |reason| Decision::Reject(reason.into())),
            "state={state:?}, peak={peak:?}"
        );
        assert_eq!(
            AdmissionConfig::default()
                .check(&engine, &request, state)
                .unwrap(),
            Decision::Allow
        );
    }
    let unlimited = AdmissionConfig::RunningPlusKvCapacity {
        max_running_requests: u64::MAX,
        max_kv_tokens: u64::MAX,
    };
    assert_eq!(
        unlimited
            .check(
                &engine,
                &PickRequest::new(&model, Stage::Plain, 1),
                AdmissionState {
                    running_requests: Some(0),
                    kv_tokens: Some(u64::MAX)
                },
            )
            .unwrap(),
        Decision::Reject("kv_capacity".into())
    );
}

#[tokio::test]
async fn capacity_uses_configured_limits_and_total_kv_from_selection_snapshot() {
    let engine = engine();
    let table = EngineLoadTable::new();
    let mut report = LoadStat {
        num_running_reqs: 1,
        num_waiting_reqs: 0,
        num_tokens: 10,
        max_total_num_tokens: 1000,
        native_cache: Some(NativeCacheRankLoad {
            num_waiting_uncached_tokens: 0,
            num_total_tokens: 80,
            max_running_requests: 100,
            total_prefill_uncached_tokens: 0,
            total_prefill_busy_us: 0,
        }),
    };
    table.set(&engine.url, 0, report.clone(), Instant::now());
    let snapshot = table.capture_snapshot(Instant::now());
    let state = AdmissionState::from_snapshot(&snapshot, &engine);
    assert_eq!(state.kv_tokens, Some(80));
    let config = AdmissionConfig::RunningPlusKvCapacity {
        max_running_requests: 2,
        max_kv_tokens: 100,
    };
    let mut policy = PowerOfTwoPolicy::new(table.clone());
    policy.admission = Arc::new(config.clone());
    let model = ModelId("m".into());
    let mut request = PickRequest::new(&model, Stage::Plain, 10);
    request.expected_peak_tokens = Some(21);
    assert!(matches!(
        policy.pick(std::slice::from_ref(&engine), &request).await,
        Err(PickError::AdmissionRejected(reason)) if reason.reason == "kv_capacity"
    ));
    // A later report affects the next selection, not the retained observation.
    report.native_cache.as_mut().unwrap().num_total_tokens = 0;
    table.set(&engine.url, 0, report.clone(), Instant::now());
    assert_eq!(
        config.check(&engine, &request, state).unwrap(),
        Decision::Reject("kv_capacity".into())
    );
    assert!(policy
        .pick(std::slice::from_ref(&engine), &request)
        .await
        .is_ok());
    report.num_running_reqs = 2;
    table.set(&engine.url, 0, report, Instant::now());
    assert!(matches!(
        policy.pick(std::slice::from_ref(&engine), &request).await,
        Err(PickError::AdmissionRejected(reason)) if reason.reason == "running_capacity"
    ));
}
