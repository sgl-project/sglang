// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{Decision, EngineAdmission, EngineMetrics};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies_reorg::{PickError, PickRequest, Policy, Stage};
use sgl_router::state::load_monitor::engine_reported_load::{EngineReportedLoadTable, LoadStat};
use sgl_router::workers::Worker;

const URL: &str = "http://engine";

fn report(table: &EngineReportedLoadTable, rank: u32, running: u64, at: Instant) {
    table.set(
        URL,
        rank,
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: 2,
            num_tokens: 30,
            max_total_num_tokens: 100,
            native_cache: None,
        },
        at,
    );
}

fn engine() -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("a".into()),
        url: URL.into(),
        mode: Stage::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

#[derive(Debug)]
struct ObserveAdmission {
    table: Arc<EngineReportedLoadTable>,
    observations: Mutex<Vec<EngineMetrics>>,
}

impl EngineAdmission for ObserveAdmission {
    fn check(&self, engine: &Worker, metrics: &EngineMetrics) -> Result<Decision, PickError> {
        assert_eq!(engine.url, URL);
        // A new report arriving after selection must not change the observation
        // supplied to admission. The next pick should read the new report.
        report(&self.table, 0, 99, Instant::now());
        self.observations.lock().unwrap().push(*metrics);
        Ok(Decision::Allow)
    }
}

#[tokio::test]
async fn selected_load_reaches_admission_and_next_pick_reads_fresh_state() {
    let table = EngineReportedLoadTable::new();
    let first_at = Instant::now();
    report(&table, 0, 1, first_at);
    report(&table, 1, 3, first_at);
    table.mark_expected_rank(URL, 0);
    table.mark_expected_rank(URL, 1);
    let admission = Arc::new(ObserveAdmission {
        table: table.clone(),
        observations: Mutex::default(),
    });
    let mut policy = PowerOfTwoPolicy::new(table);
    policy.admission = admission.clone();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let alternative = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("b".into()),
        url: "http://other".into(),
        mode: Stage::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }));
    // These old-format reports lack native pressure metrics, so selection uses
    // local active counts for both candidates and chooses the second engine.
    let _busy = alternative.load_guard();
    let engines = [alternative, engine()];
    for _ in 0..2 {
        let pick = policy.pick(&engines, &request).await.unwrap();
        assert!(Arc::ptr_eq(&pick.engine, &engines[1]));
    }
    let observations = admission.observations.lock().unwrap();
    assert_eq!(observations.len(), 2);
    // Basic reports carry request counts but no KV or pending-prefill tokens.
    assert_eq!(
        observations[0],
        EngineMetrics {
            running_requests: Some(4),
            waiting_requests: Some(4),
            ..EngineMetrics::default()
        }
    );
    assert_eq!(observations[1].running_requests, Some(102));
}

#[tokio::test]
async fn missing_stale_and_incomplete_reports_reach_admission_as_unknown() {
    for case in ["missing", "stale", "incomplete"] {
        let table = EngineReportedLoadTable::new();
        match case {
            "missing" => {}
            "stale" => report(&table, 0, 1, Instant::now() - Duration::from_secs(3600)),
            "incomplete" => {
                report(&table, 0, 1, Instant::now());
                table.mark_expected_rank(URL, 0);
                table.mark_expected_rank(URL, 1);
            }
            _ => unreachable!(),
        }
        let admission = Arc::new(ObserveAdmission {
            table: table.clone(),
            observations: Mutex::default(),
        });
        let mut policy = PowerOfTwoPolicy::new(table);
        policy.admission = admission.clone();
        let model = ModelId("m".into());
        policy
            .pick(&[engine()], &PickRequest::new(&model, Stage::Plain, 10))
            .await
            .unwrap();
        assert_eq!(
            *admission.observations.lock().unwrap(),
            [EngineMetrics::default()],
            "{case}"
        );
    }
}
