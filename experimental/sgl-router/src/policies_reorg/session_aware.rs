// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket-scoped session placement. Admission rejection preserves the binding
//! and returns to the bucket loop; it never selects a backup inside the group.

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;

use crate::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedLoadTable,
};
use crate::state::AffinityStore;
use crate::workers::Worker;

use super::admission::{AllowAll, Decision, EngineAdmission};
use super::power_of_two::PowerOfTwoPolicy;
use super::{Pick, PickError, PickRequest, Policy, Rejection};

#[derive(Debug)]
pub struct SessionAwarePolicy {
    store: Arc<AffinityStore>,
    engine_load: Arc<EngineReportedLoadTable>,
    fallback: PowerOfTwoPolicy,
    pub admission: Arc<dyn EngineAdmission>,
}

impl SessionAwarePolicy {
    /// The caller owns the shared store's idle timeout and sweeper lifecycle.
    /// This policy implements bucket-scoped affinity, without legacy global modes
    /// or primary/backup pressure escape.
    pub fn new(store: Arc<AffinityStore>, engine_load: Arc<EngineReportedLoadTable>) -> Self {
        Self {
            store,
            fallback: PowerOfTwoPolicy::new(Arc::clone(&engine_load)),
            engine_load,
            admission: Arc::new(AllowAll),
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

    fn check(
        &self,
        engine: &Worker,
        request: &PickRequest<'_>,
        load: &EngineReportedLoadSnapshot,
    ) -> Result<(), PickError> {
        match self
            .admission
            .check(engine, request, load.fresh_load_for_url(&engine.url))?
        {
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
                self.check(bound, request, &load)?;
                return Ok(Pick {
                    engine: Arc::clone(bound),
                    reason: "session_primary",
                });
            }

            // The nested power-of-two policy uses AllowAll. The session owner
            // checks its chosen engine before creating or replacing a binding.
            let mut pick = self.pick_fallback(engines, request).await?;
            let load = self.engine_load.capture_snapshot(Instant::now());
            self.check(&pick.engine, request, &load)?;
            let Some(key) = key else {
                pick.reason = "no_session";
                return Ok(pick);
            };

            let effective = self.store.bind(key, &pick.engine, engines);
            if !Arc::ptr_eq(effective, &pick.engine) {
                // A racing first assignment wins. Check it once, without
                // rewriting a rejected binding or retrying another engine.
                self.check(effective, request, &load)?;
                pick.reason = "session_primary";
            } else {
                pick.reason = "assigned";
            }
            pick.engine = Arc::clone(effective);
            Ok(pick)
        })
    }

    fn fallback(&self) -> Option<&dyn Policy> {
        Some(&self.fallback)
    }
}
