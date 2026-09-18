// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bucket-attached engine selection (see `POLICY_DESIGN.md`).
//!
//! A [`Policy`] picks one engine from the candidates it is handed and owns its
//! own fallback; its attached [`Admission`] decides which of those engines may
//! accept the request. Bucket resolution and ordering live one layer up.

pub mod admission;
pub mod least_load;
pub mod power_of_two;
pub mod random;
pub mod round_robin;
pub mod session_aware;
pub mod sticky;

pub use admission::*;

use crate::config::{DecodePolicyKind, FilterKind, ModelConfig, PolicyKind, StickyFallbackKind};
use crate::discovery::{ModelId, WorkerId};
use crate::policies::state::engine_load::LoadView;
use crate::policies::state::AffinityStore;
use crate::server::metrics::MetricsRegistry;
use crate::workers::Worker;
use futures::future::BoxFuture;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStage {
    Plain,
    Prefill,
    Decode,
}

/// Where affinity keys and candidate limits are scoped for this pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinityScope {
    Global,
    Bucket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickMode {
    /// Run the policy's full logic, including its fallback.
    Normal,
    /// Return an admitted affinity winner or `NoCandidates`; never fall back or bind.
    HitRequired,
}

/// Request facts a policy may read. Scoped to one model and stage.
pub struct PickRequest<'a> {
    pub model: &'a ModelId,
    pub stage: RoutingStage,
    pub bucket_id: &'a str,
    pub scope: AffinityScope,
    pub mode: PickMode,
    pub input_tokens: u64,
    /// Input plus requested output; decode KV projection and bucket fit.
    pub expected_peak_sequence_tokens: Option<u64>,
    pub session_id: Option<&'a str>,
    pub routing_key: Option<&'a str>,
    /// False when the resolver wants this pick to neither look up nor create bindings.
    pub affinity_enabled: bool,
    pub load: &'a LoadView<'a>,
}

impl PickRequest<'_> {
    /// Store key for a non-empty affinity `value`, scoped per the request.
    pub fn affinity_key(&self, kind: &str, value: Option<&str>) -> Option<String> {
        let value = value.filter(|value| self.affinity_enabled && !value.is_empty())?;
        let scope = match self.scope {
            AffinityScope::Global => "global",
            AffinityScope::Bucket => self.bucket_id,
        };
        Some(format!("{:?}/{kind}/{scope}/{value}", self.stage))
    }

    pub fn admission(&self) -> AdmissionContext<'_> {
        AdmissionContext {
            load: self.load,
            kv_tokens: self
                .expected_peak_sequence_tokens
                .unwrap_or(self.input_tokens),
            uncached_tokens: self.input_tokens,
        }
    }
}

/// The selected engine and the label the decision metric records for it.
#[derive(Debug, Clone)]
pub struct Pick {
    pub engine: Arc<Worker>,
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineRejection {
    pub engine: WorkerId,
    pub reason: AdmissionReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickError {
    /// Nothing to choose from here; the caller may move to the next bucket.
    NoCandidates,
    NoAdmissibleEngine(Vec<EngineRejection>),
    AdmissionRejected(EngineRejection),
    InvalidSignal(String),
}

impl PickError {
    /// Every candidate was `engine`, and it was rejected.
    pub fn rejected_all(engine: &Worker, reason: AdmissionReason) -> Self {
        Self::NoAdmissibleEngine(vec![EngineRejection {
            engine: engine.id.clone(),
            reason,
        }])
    }
}

pub type PickResult = Result<Pick, PickError>;

pub trait Policy: Send + Sync + fmt::Debug {
    /// Selects one engine from `engines`; never returns an engine outside it.
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, PickResult>;
}

/// Lifts a synchronous selection into the trait's future.
pub(crate) fn ready(result: PickResult) -> BoxFuture<'static, PickResult> {
    Box::pin(std::future::ready(result))
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("policy `{0}` is not available in the bucket engine")]
    Unsupported(PolicyKind),
    #[error("decode policy `{0:?}` is not available in the bucket engine")]
    UnsupportedDecode(DecodePolicyKind),
}

/// Shared services policies hold handles to; started once by application wiring.
pub struct PolicyDependencies {
    pub metrics: Arc<MetricsRegistry>,
    pub affinity: Arc<AffinityStore>,
}

pub fn build_policy(
    kind: PolicyKind,
    model: &ModelConfig,
    deps: &PolicyDependencies,
) -> Result<Arc<dyn Policy>, BuildError> {
    let admission = migrated_admission(kind, model);
    Ok(match kind {
        PolicyKind::SessionAware => Arc::new(session_aware::SessionAwarePolicy::new(
            admission,
            Arc::clone(&deps.affinity),
        )),
        PolicyKind::Sticky => {
            let fallback = model
                .sticky
                .as_ref()
                .map_or(StickyFallbackKind::RoundRobin, |sticky| {
                    sticky.fallback_policy
                });
            Arc::new(sticky::StickyPolicy::new(
                admission,
                Arc::clone(&deps.affinity),
                load_only(fallback.into(), Admission::allow_all())?,
                Arc::clone(&deps.metrics),
            ))
        }
        kind => load_only(kind, admission)?,
    })
}

fn load_only(kind: PolicyKind, admission: Admission) -> Result<Arc<dyn Policy>, BuildError> {
    Ok(match kind {
        PolicyKind::RoundRobin => Arc::new(round_robin::RoundRobinPolicy::new(admission)),
        PolicyKind::Random => Arc::new(random::RandomPolicy::new(admission)),
        PolicyKind::PowerOfTwo => Arc::new(power_of_two::PowerOfTwoPolicy::new(admission)),
        PolicyKind::LoadBased => Arc::new(least_load::LeastLoadPolicy::new(admission)),
        other => return Err(BuildError::Unsupported(other)),
    })
}

impl From<StickyFallbackKind> for PolicyKind {
    fn from(kind: StickyFallbackKind) -> Self {
        match kind {
            StickyFallbackKind::RoundRobin => Self::RoundRobin,
            StickyFallbackKind::Random => Self::Random,
            StickyFallbackKind::PowerOfTwo => Self::PowerOfTwo,
            StickyFallbackKind::LoadBased => Self::LoadBased,
        }
    }
}

pub fn build_decode_policy(kind: DecodePolicyKind) -> Result<Arc<dyn Policy>, BuildError> {
    match kind {
        DecodePolicyKind::PowerOfTwo => Ok(Arc::new(power_of_two::PowerOfTwoPolicy::new(
            Admission::before(CapacityAdmission),
        ))),
        other => Err(BuildError::UnsupportedDecode(other)),
    }
}

/// The acceptance checks the current selection path runs implicitly for this
/// kind, made explicit so a migrated configuration does not lose them.
fn migrated_admission(kind: PolicyKind, model: &ModelConfig) -> Admission {
    let mut checks: Vec<Box<dyn EngineAdmission>> = Vec::new();
    if matches!(
        kind,
        PolicyKind::PowerOfTwo | PolicyKind::SessionAware | PolicyKind::CacheAware
    ) {
        checks.push(Box::new(CapacityAdmission));
        if let Some(budgets) = model
            .bucket_config
            .as_ref()
            .and_then(PendingPrefillAdmission::from_buckets)
        {
            checks.push(Box::new(budgets));
        }
    }
    let overloaded = model
        .eligibility
        .as_ref()
        .filter(|eligibility| eligibility.filters.contains(&FilterKind::Overloaded))
        .and_then(|eligibility| eligibility.max_in_flight);
    if let Some(max_in_flight) = overloaded {
        checks.push(Box::new(InFlightLimitAdmission { max_in_flight }));
    }
    match checks.len() {
        0 => Admission::allow_all(),
        _ => Admission::before(AllOfAdmission(checks)),
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::discovery::{WorkerMode, WorkerSpec};
    use crate::policies::state::engine_load::EngineLoadTable;

    pub(crate) fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    pub(crate) async fn pick(policy: &dyn Policy, engines: &[Arc<Worker>]) -> PickResult {
        Request::default().pick(policy, engines).await
    }

    #[derive(Default)]
    pub(crate) struct Request {
        session_id: Option<&'static str>,
        routing_key: Option<&'static str>,
        mode: Option<PickMode>,
    }

    impl Request {
        pub(crate) fn session(id: &'static str) -> Self {
            Self {
                session_id: Some(id),
                ..Self::default()
            }
        }

        pub(crate) fn routing_key(key: &'static str) -> Self {
            Self {
                routing_key: Some(key),
                ..Self::default()
            }
        }

        pub(crate) fn hit_required(self) -> Self {
            Self {
                mode: Some(PickMode::HitRequired),
                ..self
            }
        }

        pub(crate) async fn pick(
            &self,
            policy: &dyn Policy,
            engines: &[Arc<Worker>],
        ) -> PickResult {
            let table = EngineLoadTable::new();
            let load = LoadView::new(&table);
            let model = ModelId("m".into());
            let request = PickRequest {
                model: &model,
                stage: RoutingStage::Plain,
                bucket_id: "global",
                scope: AffinityScope::Bucket,
                mode: self.mode.unwrap_or(PickMode::Normal),
                input_tokens: 16,
                expected_peak_sequence_tokens: None,
                session_id: self.session_id,
                routing_key: self.routing_key,
                affinity_enabled: true,
                load: &load,
            };
            policy.pick(engines, &request).await
        }
    }

    pub(crate) fn id(result: &PickResult) -> &str {
        &result.as_ref().expect("a pick").engine.id.0
    }
}
