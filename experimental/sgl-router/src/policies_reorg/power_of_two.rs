// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use rand::Rng;

use crate::policies::admission::{compare_decode_pressure, compare_prefill_pressure};
use crate::server::metrics::MetricsRegistry;
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
    /// Records admitted Plain/Prefill picks. Unset on fallback instances,
    /// whose picks belong to the wrapping policy.
    pub metrics: Option<Arc<MetricsRegistry>>,
}

impl PowerOfTwoPolicy {
    pub fn new(engine_load: Arc<EngineReportedLoadTable>) -> Self {
        Self {
            engine_load,
            admission: Arc::new(AdmissionLimits::default()),
            metrics: None,
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
            let (engine, load_source) = match engines {
                [engine] => (Arc::clone(engine), "single_candidate"),
                _ => {
                    let mut rng = rand::thread_rng();
                    let i = rng.gen_range(0..engines.len());
                    let mut j = rng.gen_range(0..engines.len() - 1);
                    if j >= i {
                        j += 1;
                    }
                    let (left, right) = (&engines[i], &engines[j]);
                    let pressure = match request.stage {
                        Stage::Plain | Stage::Prefill => {
                            compare_prefill_pressure(left, right, Some(&load))
                        }
                        Stage::Decode => compare_decode_pressure(left, right, Some(&load)),
                    };
                    // The comparators fall back to Router-local in-flight load
                    // unless both engines have fresh, complete reports.
                    let load_source = if [left, right]
                        .iter()
                        .all(|engine| load.fresh_native_cache_load_for_url(&engine.url).is_some())
                    {
                        "engine_load"
                    } else {
                        "router_local"
                    };
                    let winner = if pressure.is_gt() { right } else { left };
                    (Arc::clone(winner), load_source)
                }
            };
            let metrics = EngineMetrics::observe(&engine, &load);
            if let Decision::Reject(reason) = self.admission.check(&engine, &metrics)? {
                return Err(PickError::AdmissionRejected(Rejection {
                    engine: engine.id.clone(),
                    reason,
                }));
            }
            if let (Some(registry), Stage::Plain | Stage::Prefill) = (&self.metrics, request.stage)
            {
                registry.record_policy_decision("power_of_two", load_source);
            }
            Ok(Pick {
                engine,
                reason: "power_of_two",
            })
        })
    }
}
