// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Side-by-side implementation of POLICY_DESIGN.md. Chat routing can opt into
//! this interface through AppContext; `policies` remains the default.

pub mod admission;
pub mod cache_aware;
pub mod power_of_two;
pub mod session_aware;

use std::fmt::Debug;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::discovery::{ModelId, WorkerId};
use crate::workers::Worker;

pub use crate::discovery::WorkerMode as Stage;

/// Request facts for engine selection. Policies own their shared-state handles.
#[derive(Debug, Clone, Copy)]
pub struct PickRequest<'a> {
    pub model: &'a ModelId,
    pub stage: Stage,
    pub bucket: &'a str,
    pub input_tokens: u64,
    pub expected_peak_tokens: Option<u64>,
    pub prefix: Option<&'a cache_aware::PrefixMemo>,
    pub token_ids: Option<&'a [u32]>,
    pub session_key: Option<&'a str>,
    pub routing_key: Option<&'a str>,
}

impl<'a> PickRequest<'a> {
    pub fn new(model: &'a ModelId, stage: Stage, input_tokens: u64) -> Self {
        Self {
            model,
            stage,
            bucket: "",
            input_tokens,
            expected_peak_tokens: None,
            prefix: None,
            token_ids: None,
            session_key: None,
            routing_key: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pick {
    pub engine: Arc<Worker>,
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub engine: WorkerId,
    pub reason: String,
}

#[derive(Debug, thiserror::Error)]
pub enum PickError {
    #[error("no bucket matches the request length")]
    NoMatchingBucket,
    #[error("no candidates")]
    NoCandidates,
    #[error("no admissible engine: {0:?}")]
    NoAdmissibleEngine(Vec<Rejection>),
    #[error("selected engine rejected: {0:?}")]
    AdmissionRejected(Rejection),
    #[error("invalid signal: {0}")]
    InvalidSignal(String),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("policy selected an engine outside its candidates: {0:?}")]
    OutsideCandidates(WorkerId),
}

/// Returns one admitted engine from exactly the supplied candidates.
/// Implementations receive shared load, KV, and affinity handles at construction;
/// they obtain their own observations rather than asking callers to supply them.
pub trait Policy: Send + Sync + Debug {
    /// Read required state locally and pass the selected engine's observations to admission.
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>>;

    /// Whether this policy may serve `stage`; checked when a resolver is built.
    fn supports(&self, _stage: Stage) -> bool {
        true
    }

    /// Runs on a miss within the same candidates; never on an admission rejection.
    fn fallback(&self) -> Option<&dyn Policy> {
        None
    }

    /// Delegate within the same candidates; the fallback reads its own state.
    fn pick_fallback<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        match self.fallback() {
            Some(fallback) => fallback.pick(engines, request),
            None => Box::pin(async { Err(PickError::NoCandidates) }),
        }
    }
}
