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
        ..Default::default()
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
        serde_json::from_value(json!({"max_inflight_requests": 4, "max_kv_usage": 0.9})).unwrap();
    assert_eq!(
        limits,
        AdmissionLimits {
            max_inflight_requests: Some(4),
            max_kv_usage: Some(0.9),
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
fn each_count_caps_its_metric_and_fails_open_when_unknown() {
    let engine = engine();
    let known = EngineMetrics {
        waiting_requests: Some(1),
        pending_prefill_tokens: Some(90),
        inflight_requests: 2,
        ..Default::default()
    };
    let unknown = EngineMetrics {
        inflight_requests: 2,
        ..Default::default()
    };
    let limit = |field: &str, max| {
        let mut limits = AdmissionLimits::default();
        *match field {
            "max_waiting_requests" => &mut limits.max_waiting_requests,
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
        ("max_waiting_requests", 2, 1),
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
            running_capacity: Some(100),
            waiting_requests: Some(2),
            kv_tokens: Some(80),
            kv_capacity: Some(1000),
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
        max_running_usage: Some(0.02),
        max_kv_usage: Some(0.1),
        ..Default::default()
    });
    let engines = [engine];
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    // Capacities are 100 running requests and 1000 KV tokens.
    for (running, kv_tokens, rejection) in [
        (1, 100, Some("max_kv_usage")),
        (1, 99, None),
        (2, 0, Some("max_running_usage")),
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

#[test]
fn usages_admit_below_the_share_and_fail_open_without_capacity() {
    let limits = AdmissionLimits {
        max_running_usage: Some(0.9),
        max_kv_usage: Some(0.9),
        ..Default::default()
    };
    let metrics = |running, kv_tokens, capacity: Option<u64>| EngineMetrics {
        running_requests: Some(running),
        running_capacity: capacity.map(|_| 10),
        kv_tokens: Some(kv_tokens),
        kv_capacity: capacity,
        ..Default::default()
    };
    let check = |m| limits.check(&engine(), &m).unwrap();
    assert_eq!(check(metrics(8, 899, Some(1000))), Decision::Allow);
    assert_eq!(
        check(metrics(9, 0, Some(1000))),
        reject("max_running_usage")
    );
    assert_eq!(check(metrics(0, 900, Some(1000))), reject("max_kv_usage"));
    assert_eq!(check(metrics(9, 900, None)), Decision::Allow);
    let kv = |share, kv_tokens| {
        let limits = AdmissionLimits {
            max_kv_usage: Some(share),
            ..Default::default()
        };
        let metrics = EngineMetrics {
            kv_tokens: Some(kv_tokens),
            kv_capacity: Some(100),
            ..Default::default()
        };
        limits.check(&engine(), &metrics).unwrap()
    };
    // A full share still admits one more request below it, whatever its size.
    assert_eq!(kv(1.0, 99), Decision::Allow);
    assert_eq!(kv(1.0, 100), reject("max_kv_usage"));
    // `0.07 * 100.0` rounds above 7; the boundary still rejects exactly at the share.
    assert_eq!(kv(0.07, 6), Decision::Allow);
    assert_eq!(kv(0.07, 7), reject("max_kv_usage"));
}

#[test]
fn validation_rejects_out_of_range_usages_and_zero_counts() {
    for bad in [
        json!({"max_kv_usage": 1.5}),
        json!({"max_running_usage": 0.0}),
        json!({"max_waiting_requests": 0}),
        json!({"max_pending_prefill_tokens": 0}),
        json!({"max_inflight_requests": 0}),
    ] {
        let limits: AdmissionLimits = serde_json::from_value(bad.clone()).unwrap();
        assert!(limits.validate().is_err(), "accepted {bad}");
    }
    let ok: AdmissionLimits =
        serde_json::from_value(json!({"max_kv_usage": 1.0, "max_inflight_requests": 1})).unwrap();
    assert!(ok.validate().is_ok());
}

#[test]
fn group_limits_override_defaults_field_by_field() {
    let defaults = AdmissionLimits {
        max_inflight_requests: Some(64),
        max_kv_usage: Some(0.95),
        ..Default::default()
    };
    let group = AdmissionLimits {
        max_kv_usage: Some(0.9),
        max_waiting_requests: Some(8),
        ..Default::default()
    };
    assert_eq!(
        group.or(&defaults),
        AdmissionLimits {
            max_inflight_requests: Some(64),
            max_kv_usage: Some(0.9),
            max_waiting_requests: Some(8),
            ..Default::default()
        }
    );
}
