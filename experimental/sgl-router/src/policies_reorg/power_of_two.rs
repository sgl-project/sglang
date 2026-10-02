// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use rand::Rng;

use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
use crate::workers::Worker;

use super::admission::{AdmissionLimits, Decision, EngineAdmission, EngineMetrics};
use super::{Pick, PickError, PickRequest, Policy, Rejection, Stage};

/// Samples two distinct engines and selects the one with lower stage pressure.
/// Checks admission only on the selected engine; rejection never resamples.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Shared application state; snapshots are local to each pick.
    engine_load: Arc<EngineReportedLoadTable>,
    pub admission: Arc<dyn EngineAdmission>,
}

impl PowerOfTwoPolicy {
    pub fn new(engine_load: Arc<EngineReportedLoadTable>) -> Self {
        Self {
            engine_load,
            admission: Arc::new(AdmissionLimits::default()),
        }
    }
}

impl Policy for PowerOfTwoPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            if engines.is_empty() {
                return Err(PickError::NoCandidates);
            }
            // Selection and admission use the same load observation.
            let load = self.engine_load.capture_snapshot(Instant::now());
            let engine = match engines {
                [engine] => Arc::clone(engine),
                _ => {
                    let mut rng = rand::thread_rng();
                    let i = rng.gen_range(0..engines.len());
                    let mut j = rng.gen_range(0..engines.len() - 1);
                    if j >= i {
                        j += 1;
                    }
                    let (left, right) = (&engines[i], &engines[j]);
                    let reports = [left, right]
                        .map(|engine| load.fresh_native_cache_load_for_url(&engine.url));
                    let use_reported = reports.iter().all(Option::is_some);
                    let use_queue_time = reports.iter().all(|report| {
                        report.is_some_and(|r| r.estimated_prefill_queue_ms.is_some())
                    });
                    // A common denominator keeps the two KV fractions exact.
                    let kv_capacity: u128 = reports
                        .iter()
                        .flatten()
                        .map(|report| u128::from(report.max_total_num_tokens))
                        .product();
                    // Lower wins; array elements are compared left to right.
                    let score = |engine: &Worker| -> [u128; 5] {
                        let inflight = engine.router_inflight_load() as u128;
                        let Some(report) = load
                            .fresh_native_cache_load_for_url(&engine.url)
                            .filter(|_| use_reported)
                        else {
                            return [0, 0, 0, 0, inflight];
                        };
                        match request.stage {
                            Stage::Plain | Stage::Prefill => [
                                // Nonnegative f64 bits preserve queue-time order.
                                if use_queue_time {
                                    report.estimated_prefill_queue_ms.unwrap().to_bits() as u128
                                } else {
                                    0
                                },
                                report.num_waiting_uncached_tokens.into(),
                                report.num_waiting_reqs.into(),
                                report.num_running_reqs.into(),
                                inflight,
                            ],
                            Stage::Decode => [
                                report.num_waiting_reqs.into(),
                                report.num_running_reqs.into(),
                                u128::from(report.num_used_tokens)
                                    * (kv_capacity
                                        / u128::from(report.max_total_num_tokens.max(1))),
                                report.num_used_tokens.into(),
                                inflight,
                            ],
                        }
                    };
                    Arc::clone(if score(left) > score(right) {
                        right
                    } else {
                        left
                    })
                }
            };
            let metrics = EngineMetrics::observe(&engine, &load);
            if let Decision::Reject(reason) = self.admission.check(&engine, &metrics)? {
                return Err(PickError::AdmissionRejected(Rejection {
                    engine: engine.id.clone(),
                    reason,
                }));
            }
            Ok(Pick {
                engine,
                reason: "power_of_two",
            })
        })
    }
}
