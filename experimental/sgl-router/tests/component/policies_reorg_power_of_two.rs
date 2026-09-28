// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies_reorg::{PickRequest, Policy, Stage};
use sgl_router::state::load_monitor::engine_reported_load::{
    EngineReportedLoadTable, LoadStat, NativeCacheRankLoad,
};
use sgl_router::workers::Worker;

fn engine(id: &str, stage: Stage, active: usize) -> Arc<Worker> {
    let worker = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}"),
        mode: stage,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }));
    worker.active_requests.store(active, Ordering::Relaxed);
    worker
}

fn load(running: u64, waiting: u64, tokens: u64, capacity: u64, pending: u64) -> LoadStat {
    LoadStat {
        num_running_reqs: running,
        num_waiting_reqs: waiting,
        num_tokens: tokens,
        max_total_num_tokens: capacity,
        native_cache: Some(NativeCacheRankLoad {
            num_waiting_uncached_tokens: pending,
            num_total_tokens: tokens,
            max_running_requests: 100,
            total_prefill_uncached_tokens: 0,
            total_prefill_busy_us: 0,
        }),
    }
}

async fn assert_winner(
    policy: &PowerOfTwoPolicy,
    engines: &[Arc<Worker>],
    stage: Stage,
    expected: &Arc<Worker>,
) {
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, stage, 10);
    // With exactly two candidates the winner is independent of sample order.
    for _ in 0..16 {
        let pick = policy.pick(engines, &request).await.unwrap();
        assert!(Arc::ptr_eq(&pick.engine, expected), "stage: {stage:?}");
        assert_eq!(pick.reason, "power_of_two");
    }
}

#[tokio::test]
async fn plain_and_prefill_use_pending_work_while_decode_uses_request_pressure() {
    for stage in [Stage::Plain, Stage::Prefill, Stage::Decode] {
        let engines = [engine("a", stage, 100), engine("b", stage, 0)];
        let table = EngineReportedLoadTable::new();
        table.set(&engines[0].url, 0, load(5, 8, 10, 100, 1), Instant::now());
        table.set(&engines[1].url, 0, load(1, 1, 10, 100, 100), Instant::now());
        let expected = if stage == Stage::Decode { 1 } else { 0 };
        assert_winner(
            &PowerOfTwoPolicy::new(table),
            &engines,
            stage,
            &engines[expected],
        )
        .await;
    }
}

#[tokio::test]
async fn prefill_uses_estimated_queue_time_only_when_both_engines_have_rates() {
    for both_have_rates in [true, false] {
        let engines = [
            engine("a", Stage::Prefill, 0),
            engine("b", Stage::Prefill, 0),
        ];
        let table = EngineReportedLoadTable::new();
        for (i, worker) in engines.iter().enumerate() {
            let mut report = load(1, 1, 10, 100, if i == 0 { 10 } else { 20 });
            if i == 0 || both_have_rates {
                table.set(&worker.url, 0, report.clone(), Instant::now());
            }
            let native = report.native_cache.as_mut().unwrap();
            native.total_prefill_uncached_tokens = if i == 0 { 100 } else { 1000 };
            native.total_prefill_busy_us = 1_000_000;
            table.set(&worker.url, 0, report, Instant::now());
        }
        // B has more queued tokens, but its higher throughput gives a shorter
        // estimated queue. Without B's rate, compare queued tokens for both.
        let expected = usize::from(both_have_rates);
        assert_winner(
            &PowerOfTwoPolicy::new(table),
            &engines,
            Stage::Prefill,
            &engines[expected],
        )
        .await;
    }
}

#[tokio::test]
async fn decode_orders_by_waiting_running_kv_fraction_then_tokens() {
    let cases = [
        (load(50, 1, 90, 100, 0), load(1, 2, 1, 100, 0)),
        (load(1, 1, 90, 100, 0), load(2, 1, 1, 100, 0)),
        (load(1, 1, 100, 1000, 0), load(1, 1, 20, 100, 0)),
        (load(1, 1, 10, 100, 0), load(1, 1, 100, 1000, 0)),
    ];
    for (left, right) in cases {
        let engines = [
            engine("a", Stage::Decode, 100),
            engine("b", Stage::Decode, 0),
        ];
        let table = EngineReportedLoadTable::new();
        table.set(&engines[0].url, 0, left, Instant::now());
        table.set(&engines[1].url, 0, right, Instant::now());
        assert_winner(
            &PowerOfTwoPolicy::new(table),
            &engines,
            Stage::Decode,
            &engines[0],
        )
        .await;
    }
}

#[tokio::test]
async fn unusable_telemetry_falls_back_to_local_load_for_both_candidates() {
    for stage in [Stage::Plain, Stage::Prefill, Stage::Decode] {
        for case in [
            "missing",
            "stale",
            "incomplete",
            "old_publisher",
            "unknown_capacity",
        ] {
            let engines = [engine("a", stage, 1), engine("b", stage, 5)];
            let table = EngineReportedLoadTable::new();
            // A's high reported pressure must not be compared to B's local
            // count or to a fabricated zero for its unavailable telemetry.
            table.set(
                &engines[0].url,
                0,
                load(90, 90, 90, 100, 900),
                Instant::now(),
            );
            let mut right = load(0, 0, 0, 100, 0);
            let mut at = Instant::now();
            match case {
                "missing" => {}
                "stale" => at -= Duration::from_secs(3600),
                "incomplete" => {
                    table.mark_expected_rank(&engines[1].url, 0);
                    table.mark_expected_rank(&engines[1].url, 1);
                }
                "old_publisher" => right.native_cache = None,
                "unknown_capacity" => right.max_total_num_tokens = 0,
                _ => unreachable!(),
            }
            if case != "missing" {
                table.set(&engines[1].url, 0, right, at);
            }
            assert_winner(&PowerOfTwoPolicy::new(table), &engines, stage, &engines[0]).await;
        }
    }
}

#[tokio::test]
async fn equal_reported_pressure_uses_local_active_load_as_tiebreaker() {
    for stage in [Stage::Plain, Stage::Prefill, Stage::Decode] {
        let engines = [engine("a", stage, 5), engine("b", stage, 1)];
        let table = EngineReportedLoadTable::new();
        for worker in &engines {
            table.set(&worker.url, 0, load(1, 1, 10, 100, 10), Instant::now());
        }
        assert_winner(&PowerOfTwoPolicy::new(table), &engines, stage, &engines[1]).await;
    }
}

#[tokio::test]
async fn multiple_candidates_never_select_the_unique_busiest_engine() {
    let engines: Vec<_> = (0..8)
        .map(|i| engine(&i.to_string(), Stage::Plain, i))
        .collect();
    let policy = PowerOfTwoPolicy::new(EngineReportedLoadTable::new());
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    for _ in 0..64 {
        let pick = policy.pick(&engines, &request).await.unwrap();
        // Every distinct pair has an engine less busy than the last candidate.
        assert!(engines[..7]
            .iter()
            .any(|engine| Arc::ptr_eq(engine, &pick.engine)));
    }
}
