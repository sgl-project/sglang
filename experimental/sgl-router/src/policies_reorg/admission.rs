// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use crate::workers::Worker;

use super::{PickContext, PickError, PickRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Reject(String),
}

/// Checks use their injected state handles with the policy's observation context.
/// Each policy decides when to check an engine and how to handle rejection.
pub trait EngineAdmission: Send + Sync + Debug {
    fn check(
        &self,
        engine: &Worker,
        request: &PickRequest<'_>,
        context: &PickContext,
    ) -> Result<Decision, PickError>;
}

#[derive(Debug)]
pub struct AllowAll;

impl EngineAdmission for AllowAll {
    fn check(
        &self,
        _: &Worker,
        _: &PickRequest<'_>,
        _: &PickContext,
    ) -> Result<Decision, PickError> {
        Ok(Decision::Allow)
    }
}
