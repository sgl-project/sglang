// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use rand::Rng;

use crate::state::load_monitor::engine_ranking::{compare_decode_engines, compare_prefill_engines};
use crate::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedLoadTable,
};
use crate::workers::Worker;

use super::admission::{AdmissionLimits, Decision, EngineAdmission, EngineMetrics};
use super::{Pick, PickError, PickRequest, Policy, Rejection, Stage};

/// Samples two distinct engines and selects the better-ranked one for its stage.
/// A rejected engine is dropped and the rest resampled, so the group fails only
/// when no engine is admitted.
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

    /// Picks with `admission`, which affinity policies pass for their fallback.
    pub fn pick_admitted(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
        admission: &dyn EngineAdmission,
    ) -> Result<Pick, PickError> {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        // Selection and admission use the same load observation.
        let load = self.engine_load.capture_snapshot(Instant::now());
        let mut pool: Vec<_> = engines.iter().collect();
        let mut rejections = Vec::new();
        while !pool.is_empty() {
            let engine = sample(&pool, request.stage, &load);
            let metrics = EngineMetrics::observe(engine, &load);
            match admission.check(engine, &metrics)? {
                Decision::Allow => {
                    return Ok(Pick {
                        engine: Arc::clone(engine),
                        reason: "power_of_two",
                    })
                }
                Decision::Reject(reason) => rejections.push(Rejection {
                    engine: engine.id.clone(),
                    reason,
                }),
            }
            pool.retain(|e| !Arc::ptr_eq(e, engine));
        }
        Err(if rejections.len() == 1 {
            PickError::AdmissionRejected(rejections.remove(0))
        } else {
            PickError::NoAdmissibleEngine(rejections)
        })
    }
}

/// The better-ranked engine of two distinct samples.
fn sample<'e>(
    pool: &[&'e Arc<Worker>],
    stage: Stage,
    load: &EngineReportedLoadSnapshot,
) -> &'e Arc<Worker> {
    if pool.len() == 1 {
        return pool[0];
    }
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(0..pool.len());
    let mut j = rng.gen_range(0..pool.len() - 1);
    if j >= i {
        j += 1;
    }
    let (left, right) = (pool[i], pool[j]);
    let ordering = match stage {
        Stage::Plain | Stage::Prefill => compare_prefill_engines(left, right, Some(load)),
        Stage::Decode => compare_decode_engines(left, right, Some(load)),
    };
    if ordering.is_gt() {
        right
    } else {
        left
    }
}

impl Policy for PowerOfTwoPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        let result = self.pick_admitted(engines, request, self.admission.as_ref());
        Box::pin(async move { result })
    }
}
