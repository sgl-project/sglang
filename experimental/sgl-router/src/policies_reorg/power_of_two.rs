// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;

use crate::state::load_monitor::engine_load::{EngineLoadTable, EngineWorkerLoad};
use crate::workers::Worker;

use super::admission::{AllowAll, Decision, EngineAdmission};
use super::{Pick, PickError, PickRequest, Policy, Rejection};

/// Selects an engine, then checks its admission; rejection never resamples.
/// Multi-candidate sampling and load comparison remain a follow-up.
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
            admission: Arc::new(AllowAll),
            fallback: None,
        }
    }

    fn select_engine(
        &self,
        engines: &[Arc<Worker>],
        _request: &PickRequest<'_>,
    ) -> Result<(Arc<Worker>, Option<EngineWorkerLoad>), PickError> {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        // Keep load local to selection. Admission receives the chosen engine's
        // record from this same snapshot, including capacity and report time.
        let load = self.engine_load.capture_snapshot(Instant::now());
        match engines {
            [engine] => Ok((
                Arc::clone(engine),
                load.fresh_load_for_url(&engine.url).cloned(),
            )),
            _ => {
                todo!("sample two engines and compare load for request.stage")
            }
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
            let (engine, load) = self.select_engine(engines, request)?;
            if let Decision::Reject(reason) =
                self.admission.check(&engine, request, load.as_ref())?
            {
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
