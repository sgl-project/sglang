// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{
    AdmissionLimits, Decision, EngineAdmission, EngineMetrics,
};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies_reorg::{PickError, PickRequest, Policy, Stage};
use sgl_router::state::load_monitor::engine_reported_load::{
    EngineReportedLoadTable, LoadStat, NativeCacheRankLoad,
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

fn report(running: u64, waiting: u64, kv_tokens: u64, pending: u64) -> LoadStat {
    LoadStat {
        num_running_reqs: running,
        num_waiting_reqs: waiting,
        num_tokens: kv_tokens,
        max_total_num_tokens: 1000,
        native_cache: Some(NativeCacheRankLoad {
            num_waiting_uncached_tokens: pending,
            num_total_tokens: kv_tokens,
            max_running_requests: 100,
            total_prefill_uncached_tokens: 0,
            total_prefill_busy_us: 0,
        }),
    }
}

fn reject(name: &str) -> Decision {
    Decision::Reject(name.into())
}

#[test]
fn limits_are_flat_optional_fields() {
    let limits: AdmissionLimits =
        serde_json::from_value(json!({"max_inflight_requests": 4, "max_kv_tokens": 100})).unwrap();
    assert_eq!(
        limits,
        AdmissionLimits {
            max_inflight_requests: Some(4),
            max_kv_tokens: Some(100),
            ..Default::default()
        }
    );
    assert_eq!(
        serde_json::from_value::<AdmissionLimits>(json!({})).unwrap(),
        AdmissionLimits::default()
    );
    assert!(serde_json::from_value::<AdmissionLimits>(json!({"max_in_flight": 4})).is_err());
}

#[test]
fn each_limit_caps_its_metric_and_fails_open_when_unknown() {
    let engine = engine();
    let known = EngineMetrics {
        running_requests: Some(3),
        waiting_requests: Some(1),
        kv_tokens: Some(80),
        pending_prefill_tokens: Some(90),
        inflight_requests: 2,
    };
    let unknown = EngineMetrics {
        inflight_requests: 2,
        ..Default::default()
    };
    let limit = |field: &str, max| {
        let mut limits = AdmissionLimits::default();
        *match field {
            "max_running_requests" => &mut limits.max_running_requests,
            "max_waiting_requests" => &mut limits.max_waiting_requests,
            "max_kv_tokens" => &mut limits.max_kv_tokens,
            "max_pending_prefill_tokens" => &mut limits.max_pending_prefill_tokens,
            "max_inflight_requests" => &mut limits.max_inflight_requests,
            _ => unreachable!(),
        } = Some(max);
        limits
    };
    assert_eq!(
        AdmissionLimits::default().check(&engine, &known).unwrap(),
        Decision::Allow
    );
    for (field, below, at) in [
        ("max_running_requests", 4, 3),
        ("max_waiting_requests", 2, 1),
        ("max_kv_tokens", 81, 80),
        ("max_pending_prefill_tokens", 91, 90),
        ("max_inflight_requests", 3, 2),
    ] {
        let (below, at) = (limit(field, below), limit(field, at));
        assert_eq!(below.check(&engine, &known).unwrap(), Decision::Allow);
        assert_eq!(at.check(&engine, &known).unwrap(), reject(field));
        // Without a report only the router-local in-flight count applies.
        let expected = if field == "max_inflight_requests" {
            reject(field)
        } else {
            Decision::Allow
        };
        assert_eq!(at.check(&engine, &unknown).unwrap(), expected);
    }
}

#[tokio::test]
async fn metrics_come_from_the_selection_snapshot_and_live_inflight_count() {
    let engine = engine();
    let table = EngineReportedLoadTable::new();
    table.set(&engine.url, 0, report(1, 2, 80, 5), Instant::now());
    let guard = engine.load_guard();
    assert_eq!(
        EngineMetrics::observe(&engine, &table.capture_snapshot(Instant::now())),
        EngineMetrics {
            running_requests: Some(1),
            waiting_requests: Some(2),
            kv_tokens: Some(80),
            pending_prefill_tokens: Some(5),
            inflight_requests: 1,
        }
    );
    drop(guard);
    // Basic reports carry request counts but no KV or pending-prefill tokens.
    let basic = LoadStat {
        native_cache: None,
        ..report(1, 2, 80, 5)
    };
    table.set(&engine.url, 0, basic, Instant::now());
    let metrics = EngineMetrics::observe(&engine, &table.capture_snapshot(Instant::now()));
    assert_eq!(
        (
            metrics.running_requests,
            metrics.kv_tokens,
            metrics.pending_prefill_tokens
        ),
        (Some(1), None, None)
    );

    let mut policy = PowerOfTwoPolicy::new(table.clone());
    policy.admission = Arc::new(AdmissionLimits {
        max_running_requests: Some(2),
        max_kv_tokens: Some(100),
        ..Default::default()
    });
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let engines = [engine];
    for (running, kv_tokens, rejection) in [
        (1, 100, Some("max_kv_tokens")),
        (1, 99, None),
        (2, 0, Some("max_running_requests")),
    ] {
        table.set(
            &engines[0].url,
            0,
            report(running, 0, kv_tokens, 0),
            Instant::now(),
        );
        let result = policy.pick(&engines, &request).await;
        match rejection {
            None => assert!(result.is_ok()),
            Some(reason) => assert!(matches!(
                result,
                Err(PickError::AdmissionRejected(rejected)) if rejected.reason == reason
            )),
        }
    }
}
