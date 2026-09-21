// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use crate::state::load_monitor::engine_reported_load::EngineReportedWorkerLoad;
use crate::workers::Worker;

use super::{PickError, PickRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Reject(String),
}

/// Checks one engine using the load observation retained by selection.
/// `None` means no usable load observation, never zero load. Each check defines
/// its missing-data behavior and owns any other state handles it needs.
/// Each policy decides when to check an engine and how to handle rejection.
pub trait EngineAdmission: Send + Sync + Debug {
    fn check(
        &self,
        engine: &Worker,
        request: &PickRequest<'_>,
        load: Option<&EngineReportedWorkerLoad>,
    ) -> Result<Decision, PickError>;
}

#[derive(Debug)]
pub struct AllowAll;

impl EngineAdmission for AllowAll {
    fn check(
        &self,
        _: &Worker,
        _: &PickRequest<'_>,
        _: Option<&EngineReportedWorkerLoad>,
    ) -> Result<Decision, PickError> {
        Ok(Decision::Allow)
    }
}
