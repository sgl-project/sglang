// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use rand::Rng;

use crate::state::load_monitor::engine_ranking::{compare_decode_engines, compare_prefill_engines};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
use crate::workers::Worker;

use super::admission::{partition_admitted, AdmissionLimits, EngineAdmission};
use super::{Pick, PickError, PickRequest, Policy, Stage};

/// Samples two distinct admitted engines and selects the better-ranked one for
/// its stage. Engines that fail admission are never sampled, so the pick fails
/// only when no engine is admitted.
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
            let (admitted, rejections) =
                partition_admitted(self.admission.as_ref(), engines, &load)?;
            let engine = match admitted.as_slice() {
                [] => return Err(PickError::NoAdmissibleEngine(rejections)),
                [engine] => Arc::clone(engine),
                _ => {
                    let mut rng = rand::thread_rng();
                    let i = rng.gen_range(0..admitted.len());
                    let mut j = rng.gen_range(0..admitted.len() - 1);
                    if j >= i {
                        j += 1;
                    }
                    let (left, right) = (&admitted[i], &admitted[j]);
                    let ordering = match request.stage {
                        Stage::Plain | Stage::Prefill => {
                            compare_prefill_engines(left, right, Some(&load))
                        }
                        Stage::Decode => compare_decode_engines(left, right, Some(&load)),
                    };
                    Arc::clone(if ordering.is_gt() { right } else { left })
                }
            };
            Ok(Pick {
                engine,
                reason: "power_of_two",
            })
        })
    }
}
