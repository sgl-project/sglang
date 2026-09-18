// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;
use std::sync::Arc;

use crate::workers::Worker;

use super::{Pick, PickError, PickRequest, Rejection};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Reject(String),
}

pub trait EngineAdmission: Send + Sync + Debug {
    fn check(&self, engine: &Worker, request: &PickRequest<'_>) -> Result<Decision, PickError>;
}

#[derive(Debug)]
pub struct AllowAll;

impl EngineAdmission for AllowAll {
    fn check(&self, _: &Worker, _: &PickRequest<'_>) -> Result<Decision, PickError> {
        Ok(Decision::Allow)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Placement {
    #[default]
    BeforeSelection,
    AfterSelection,
}

#[derive(Debug, Clone)]
pub struct Admission {
    pub check: Arc<dyn EngineAdmission>,
    pub placement: Placement,
}

impl Default for Admission {
    fn default() -> Self {
        Self {
            check: Arc::new(AllowAll),
            placement: Placement::BeforeSelection,
        }
    }
}

impl Admission {
    /// Candidates a policy may choose from; a no-op under `AfterSelection`.
    pub fn before(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Result<Vec<Arc<Worker>>, PickError> {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        if self.placement == Placement::AfterSelection {
            return Ok(engines.to_vec());
        }
        let mut admitted = Vec::new();
        let mut rejected = Vec::new();
        for engine in engines {
            match self.check.check(engine, request)? {
                Decision::Allow => admitted.push(engine.clone()),
                Decision::Reject(reason) => rejected.push(Rejection {
                    engine: engine.id.clone(),
                    reason,
                }),
            }
        }
        if admitted.is_empty() {
            Err(PickError::NoAdmissibleEngine(rejected))
        } else {
            Ok(admitted)
        }
    }

    /// Checks the chosen engine under `AfterSelection`; never picks a replacement.
    pub fn after(&self, pick: Pick, request: &PickRequest<'_>) -> Result<Pick, PickError> {
        if self.placement == Placement::AfterSelection {
            if let Decision::Reject(reason) = self.check.check(&pick.engine, request)? {
                return Err(PickError::AdmissionRejected(Rejection {
                    engine: pick.engine.id.clone(),
                    reason,
                }));
            }
        }
        Ok(pick)
    }
}
