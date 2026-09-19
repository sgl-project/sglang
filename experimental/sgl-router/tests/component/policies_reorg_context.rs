// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::BoxFuture;
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{AllowAll, Decision, EngineAdmission};
use sgl_router::policies_reorg::{
    Pick, PickContext, PickError, PickRequest, Policy, Rejection, Stage,
};
use sgl_router::state::load_monitor::engine_load::{EngineLoadSnapshot, EngineLoadTable, LoadStat};
use sgl_router::workers::Worker;

const URL: &str = "http://engine";

fn report(table: &EngineLoadTable, running: u64) {
    table.set(
        URL,
        0,
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: 0,
            num_tokens: 0,
            max_total_num_tokens: 100,
            native_cache: None,
        },
        Instant::now(),
    );
}

#[test]
fn observations_are_lazy_cached_and_isolated_by_source() {
    let context = PickContext::default();
    let first = EngineLoadTable::new();
    report(&first, 1);
    // Creating the context does not capture reports.
    report(&first, 2);
    let snapshot = context.load(&first);
    assert_eq!(
        snapshot.fresh_load_for_url(URL).unwrap().num_running_reqs,
        2
    );
    report(&first, 3);
    assert!(Arc::ptr_eq(&snapshot, &context.load(&first)));

    let second = EngineLoadTable::new();
    report(&second, 7);
    let other = context.load(&second);
    assert_eq!(other.fresh_load_for_url(URL).unwrap().num_running_reqs, 7);
    assert!(!Arc::ptr_eq(&snapshot, &other));
    assert!(Arc::ptr_eq(&other, &context.load(&second)));
    // The source remains alive with its cached snapshot, preventing address reuse.
    assert_eq!(Arc::strong_count(&first), 2);
    drop(context);
    assert_eq!(Arc::strong_count(&first), 1);
}

#[test]
fn concurrent_consumers_share_the_same_snapshot() {
    let context = PickContext::default();
    let table = EngineLoadTable::new();
    report(&table, 1);
    let snapshots = std::thread::scope(|scope| {
        let readers: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| context.load(&table)))
            .collect();
        readers
            .into_iter()
            .map(|reader| reader.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(snapshots
        .iter()
        .all(|snapshot| Arc::ptr_eq(snapshot, &snapshots[0])));
}

type Reads = Arc<Mutex<Vec<(&'static str, Arc<EngineLoadSnapshot>)>>>;

#[derive(Debug)]
struct ObservingAdmission {
    table: Arc<EngineLoadTable>,
    reads: Reads,
}

impl EngineAdmission for ObservingAdmission {
    fn check(
        &self,
        _: &Worker,
        _: &PickRequest<'_>,
        context: &PickContext,
    ) -> Result<Decision, PickError> {
        self.reads
            .lock()
            .unwrap()
            .push(("admission", context.load(&self.table)));
        report(&self.table, 10);
        Ok(Decision::Allow)
    }
}

#[derive(Debug)]
struct ObservingPolicy {
    table: Arc<EngineLoadTable>,
    label: &'static str,
    reads: Reads,
    admission: Arc<dyn EngineAdmission>,
    fallback: Option<Arc<dyn Policy>>,
}

impl Policy for ObservingPolicy {
    fn pick_with_context<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
        context: &'a PickContext,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            self.reads
                .lock()
                .unwrap()
                .push((self.label, context.load(&self.table)));
            report(&self.table, 20);
            let pick = if self.fallback.is_some() {
                self.pick_fallback(engines, request, context).await?
            } else {
                Pick {
                    engine: engines.first().ok_or(PickError::NoCandidates)?.clone(),
                    reason: "observed",
                }
            };
            if let Decision::Reject(reason) =
                self.admission.check(&pick.engine, request, context)?
            {
                return Err(PickError::AdmissionRejected(Rejection {
                    engine: pick.engine.id.clone(),
                    reason,
                }));
            }
            Ok(pick)
        })
    }

    fn fallback(&self) -> Option<&dyn Policy> {
        self.fallback.as_deref()
    }
}

fn engine(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: URL.into(),
        mode: Stage::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

#[tokio::test]
async fn admission_selection_and_nested_fallback_share_observations_but_next_pick_is_fresh() {
    let table = EngineLoadTable::new();
    report(&table, 1);
    let reads: Reads = Arc::default();
    let leaf = Arc::new(ObservingPolicy {
        table: table.clone(),
        label: "leaf",
        reads: reads.clone(),
        admission: Arc::new(AllowAll),
        fallback: None,
    });
    let middle = Arc::new(ObservingPolicy {
        table: table.clone(),
        label: "middle",
        reads: reads.clone(),
        admission: Arc::new(AllowAll),
        fallback: Some(leaf),
    });
    let primary = ObservingPolicy {
        table: table.clone(),
        label: "primary",
        reads: reads.clone(),
        admission: Arc::new(ObservingAdmission {
            table: table.clone(),
            reads: reads.clone(),
        }),
        fallback: Some(middle),
    };
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 1);
    let engines = [engine("a"), engine("b")];
    let expected = vec!["primary", "middle", "leaf", "admission"];
    primary.pick(&engines, &request).await.unwrap();
    let first_reads = std::mem::take(&mut *reads.lock().unwrap());
    assert_eq!(
        first_reads
            .iter()
            .map(|(label, _)| *label)
            .collect::<Vec<_>>(),
        expected
    );
    let first = &first_reads[0].1;
    assert_eq!(first.fresh_load_for_url(URL).unwrap().num_running_reqs, 1);
    assert!(first_reads
        .iter()
        .all(|(_, snapshot)| Arc::ptr_eq(first, snapshot)));

    // Use the same policy and request again: observations must not leak across attempts.
    report(&table, 99);
    primary.pick(&engines, &request).await.unwrap();
    let next_reads = reads.lock().unwrap();
    let next = &next_reads[0].1;
    assert_eq!(next.fresh_load_for_url(URL).unwrap().num_running_reqs, 99);
    assert!(next.version > first.version);
    assert!(!Arc::ptr_eq(first, next));
    assert!(next_reads
        .iter()
        .all(|(_, snapshot)| Arc::ptr_eq(next, snapshot)));
}
