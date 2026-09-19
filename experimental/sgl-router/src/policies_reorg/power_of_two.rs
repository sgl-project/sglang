// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use rand::Rng;

use crate::policies::admission::{compare_decode_pressure, compare_prefill_pressure};
use crate::state::load_monitor::engine_load::EngineLoadTable;
use crate::workers::Worker;

use super::admission::{AdmissionConfig, AdmissionState, Decision, EngineAdmission};
use super::{Pick, PickError, PickRequest, Policy, Rejection, Stage};

/// Samples two distinct engines and selects the one with lower stage pressure.
/// Checks admission only on the selected engine; rejection never resamples.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Shared application state; snapshots are local to each pick.
    engine_load: Arc<EngineLoadTable>,
    pub admission: Arc<dyn EngineAdmission>,
    pub fallback: Option<Arc<dyn Policy>>,
}

impl PowerOfTwoPolicy {
    pub fn new(engine_load: Arc<EngineLoadTable>) -> Self {
        Self {
            engine_load,
            admission: Arc::new(AdmissionConfig::default()),
            fallback: None,
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
                    let pressure = match request.stage {
                        Stage::Plain | Stage::Prefill => {
                            compare_prefill_pressure(left, right, Some(&load))
                        }
                        Stage::Decode => compare_decode_pressure(left, right, Some(&load)),
                    };
                    Arc::clone(if pressure.is_gt() { right } else { left })
                }
            };
            let state = AdmissionState::from_snapshot(&load, &engine);
            if let Decision::Reject(reason) = self.admission.check(&engine, request, state)? {
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

    fn fallback(&self) -> Option<&dyn Policy> {
        self.fallback.as_deref()
    }
}
