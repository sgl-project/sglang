// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket-scoped session placement with admitted fallback and rebinding.

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;

use crate::config::{AffinityConfig, AffinityMode};

use crate::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedLoadTable,
};
use crate::state::AffinityStore;
use crate::workers::Worker;

use super::admission::{AdmissionLimits, Decision, EngineAdmission, EngineMetrics};
use super::affinity;
use super::power_of_two::PowerOfTwoPolicy;
use super::{Pick, PickError, PickRequest, Policy, Rejection};

#[derive(Debug)]
pub struct SessionAwarePolicy {
    store: Arc<AffinityStore>,
    engine_load: Arc<EngineReportedLoadTable>,
    fallback: PowerOfTwoPolicy,
    pub admission: Arc<dyn EngineAdmission>,
    pub config: AffinityConfig,
}

impl SessionAwarePolicy {
    /// The caller owns the shared store's idle timeout and sweeper lifecycle.
    pub fn new(store: Arc<AffinityStore>, engine_load: Arc<EngineReportedLoadTable>) -> Self {
        Self {
            store,
            fallback: PowerOfTwoPolicy::new(Arc::clone(&engine_load)),
            engine_load,
            admission: Arc::new(AdmissionLimits::default()),
            config: AffinityConfig {
                mode: AffinityMode::Prefer,
                ..Default::default()
            },
        }
    }

    fn assignment_key(request: &PickRequest<'_>) -> Option<String> {
        let session = request.session_key.filter(|key| !key.is_empty())?;
        // Length prefixes keep arbitrary model, bucket and session strings
        // unambiguous, including embedded delimiters. Roles never share bindings.
        Some(format!(
            "session:{:?}:{}:{}{}:{}{}",
            request.stage,
            request.model.0.len(),
            request.model.0,
            request.bucket.len(),
            request.bucket,
            session
        ))
    }

    fn check(&self, engine: &Worker, load: &EngineReportedLoadSnapshot) -> Result<(), PickError> {
        let metrics = EngineMetrics::observe(engine, load);
        match self.admission.check(engine, &metrics)? {
            Decision::Allow => Ok(()),
            Decision::Reject(reason) => Err(PickError::AdmissionRejected(Rejection {
                engine: engine.id.clone(),
                reason,
            })),
        }
    }
}

impl Policy for SessionAwarePolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            if engines.is_empty() {
                return Err(PickError::NoCandidates);
            }
            let key = Self::assignment_key(request);
            if let Some(bound) = key.as_ref().and_then(|key| self.store.bound(key, engines)) {
                let load = self.engine_load.capture_snapshot(Instant::now());
                let rejection = match self.check(bound, &load) {
                    Ok(()) => None,
                    Err(error @ PickError::AdmissionRejected(_)) => Some(error),
                    Err(error) => return Err(error),
                };
                let primary = Pick {
                    engine: Arc::clone(bound),
                    reason: "session_primary",
                };
                if rejection.is_none() && self.config.mode != AffinityMode::Balanced {
                    return Ok(primary);
                }
                let alternatives: Vec<_> = engines
                    .iter()
                    .filter(|e| e.id != bound.id)
                    .cloned()
                    .collect();
                if alternatives.is_empty() {
                    return rejection.map_or(Ok(primary), Err);
                }
                let fallback =
                    self.fallback
                        .pick_admitted(&alternatives, request, self.admission.as_ref());
                let mut pick = affinity::choose(
                    &self.config,
                    rejection.is_none().then_some(primary),
                    fallback,
                    &load,
                    // Without a prefix signal, either engine prefills the whole input.
                    |_| request.input_tokens,
                )?;
                if pick.engine.id == bound.id {
                    return Ok(pick);
                }
                // Excluding the old binding lets a concurrent replacement win.
                let effective = self.store.bind(key.unwrap(), &pick.engine, &alternatives);
                if !Arc::ptr_eq(effective, &pick.engine) {
                    self.check(effective, &load)?;
                }
                pick.engine = Arc::clone(effective);
                pick.reason = "session_rebound";
                return Ok(pick);
            }

            let mut pick =
                self.fallback
                    .pick_admitted(engines, request, self.admission.as_ref())?;
            let load = self.engine_load.capture_snapshot(Instant::now());
            let Some(key) = key else {
                pick.reason = "no_session";
                return Ok(pick);
            };

            let effective = self.store.bind(key, &pick.engine, engines);
            if !Arc::ptr_eq(effective, &pick.engine) {
                // A racing first assignment wins. Check it once, without
                // rewriting a rejected binding or retrying another engine.
                self.check(effective, &load)?;
                pick.reason = "session_primary";
            } else {
                pick.reason = "assigned";
            }
            pick.engine = Arc::clone(effective);
            Ok(pick)
        })
    }
}
