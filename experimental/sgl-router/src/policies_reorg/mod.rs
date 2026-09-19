// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Side-by-side implementation of POLICY_DESIGN.md. Not wired into serving;
//! `policies` stays live until the switch PR replaces it.

pub mod admission;
pub mod affinity;
pub mod factory;
pub mod least_load;
pub mod power_of_two;
pub mod random;
pub mod round_robin;

use std::fmt::Debug;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::discovery::{ModelId, WorkerId};
use crate::state::LoadView;
use crate::workers::Worker;

pub use crate::discovery::WorkerMode as Stage;

/// `HitRequired` returns only an existing affinity hit: no fallback, no new binding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PickMode {
    #[default]
    Normal,
    HitRequired,
}

/// Request facts a policy may read. Resolver-only facts live in `SelectionRequest`.
#[derive(Debug, Clone, Copy)]
pub struct PickRequest<'a> {
    pub model: &'a ModelId,
    pub stage: Stage,
    pub bucket: &'a str,
    pub mode: PickMode,
    /// False once global-preserve has decided later groups may not look up or bind.
    pub affinity_enabled: bool,
    pub input_tokens: u64,
    pub expected_peak_tokens: Option<u64>,
    pub token_ids: Option<&'a [u32]>,
    pub session_key: Option<&'a str>,
    pub routing_key: Option<&'a str>,
    pub load: &'a LoadView<'a>,
}

impl<'a> PickRequest<'a> {
    pub fn new(
        model: &'a ModelId,
        stage: Stage,
        input_tokens: u64,
        load: &'a LoadView<'a>,
    ) -> Self {
        Self {
            model,
            stage,
            bucket: "",
            mode: PickMode::Normal,
            affinity_enabled: true,
            input_tokens,
            expected_peak_tokens: None,
            token_ids: None,
            session_key: None,
            routing_key: None,
            load,
        }
    }

    /// KV the request will hold: the input, or the projected peak when known.
    pub fn kv_tokens(&self) -> u64 {
        self.expected_peak_tokens.unwrap_or(self.input_tokens)
    }

    /// Binding key for `kind`, scoped to this bucket unless `global`.
    pub fn affinity_key(&self, kind: &str, global: bool, value: &str) -> String {
        let scope = if global { "global" } else { self.bucket };
        format!("{:?}/{kind}/{scope}/{value}", self.stage)
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
    #[error("no candidates")]
    NoCandidates,
    #[error("no admissible engine: {0:?}")]
    NoAdmissibleEngine(Vec<Rejection>),
    #[error("selected engine rejected: {0:?}")]
    AdmissionRejected(Rejection),
    #[error("invalid signal: {0}")]
    InvalidSignal(String),
    #[error("policy selected an engine outside its candidates: {0:?}")]
    OutsideCandidates(WorkerId),
}

/// Returns one admitted engine from exactly the supplied candidates.
pub trait Policy: Send + Sync + Debug {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>>;

    /// The nested policy a miss falls back to, within the same candidates.
    fn fallback(&self) -> Option<&dyn Policy> {
        None
    }
}

/// Lifts a synchronous result into the trait's future.
pub(crate) fn ready(
    result: Result<Pick, PickError>,
) -> BoxFuture<'static, Result<Pick, PickError>> {
    Box::pin(std::future::ready(result))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::admission::Admission;
    use super::*;
    use crate::discovery::WorkerSpec;
    use crate::state::engine_load::EngineLoadTable;

    pub(crate) fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}"),
            mode: Stage::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    pub(crate) async fn pick(
        policy: &dyn Policy,
        engines: &[Arc<Worker>],
    ) -> Result<Pick, PickError> {
        pick_as(policy, engines, None, PickMode::Normal).await
    }

    pub(crate) async fn pick_as(
        policy: &dyn Policy,
        engines: &[Arc<Worker>],
        session_key: Option<&str>,
        mode: PickMode,
    ) -> Result<Pick, PickError> {
        let table = EngineLoadTable::new();
        let load = LoadView::new(&table);
        let model = ModelId("m".into());
        let mut request = PickRequest::new(&model, Stage::Plain, 10, &load);
        request.session_key = session_key;
        request.mode = mode;
        policy.pick(engines, &request).await
    }

    /// First admitted engine under `admission`.
    pub(crate) async fn pick_with(
        admission: &Admission,
        engines: &[Arc<Worker>],
    ) -> Result<Pick, PickError> {
        let table = EngineLoadTable::new();
        let load = LoadView::new(&table);
        let model = ModelId("m".into());
        let request = PickRequest::new(&model, Stage::Plain, 10, &load);
        admission.select(engines, &request, "test", |admitted| {
            admitted.first().cloned()
        })
    }
}
