// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-model PD pool resolution.
//!
//! Carries forward the fix from `sgl-project/sglang#25184`: in
//! prefill/decode (PD) disaggregation deployments, prefill traffic must
//! never select a decode worker and vice versa. This module is the
//! single chokepoint that classifies a model as PD or non-PD and exposes
//! pool-restricted candidate sets.
//!
//! # Classification
//!
//! A model is **PD-mode** if its [`WorkerRegistry`] contains workers
//! with [`WorkerMode::Prefill`] OR [`WorkerMode::Decode`]. A model is
//! **plain-mode** if it has only `WorkerMode::Plain` workers (or no
//! workers — both queries return empty for an unknown model). The
//! `(prefill, decode, plain)` partition is computed eagerly per call;
//! tests show this is cheaper than maintaining a side-table and
//! avoiding a race against discovery events.
//!
//! # Why not `Worker::mode` directly in the chat handler?
//!
//! Two reasons:
//!
//! 1. The classification is a *cohort* decision (does the model use PD?),
//!    not a per-worker decision. Putting it in the handler means every
//!    request route reimplements the same "are any of these prefill?"
//!    walk. A central [`PdPoolResolver`] returns the same answer with
//!    one call.
//! 2. Errors. The handler needs to distinguish "no workers at all"
//!    (existing `NoHealthyWorkers`) from "no prefill workers
//!    available for a PD-mode model" (new `NoPrefillWorkersAvailable`)
//!    — only the resolver has the cohort context to tell which is which.

use crate::discovery::{ModelId, WorkerMode};
use crate::workers::{Worker, WorkerRegistry};
use std::sync::Arc;

/// Resolution result for a single request route. The handler picks
/// `prefill` / `decode` based on whether it is dispatching prefill or
/// decode traffic; `plain` is for non-PD models.
#[derive(Debug)]
pub enum PdPools {
    /// Non-PD deployment: the model is served by plain workers.
    Plain { workers: Vec<Arc<Worker>> },
    /// PD-disaggregation deployment: the model has prefill and/or decode
    /// workers. Either pool may be empty (e.g. all prefill workers'
    /// circuit breakers are open).
    Pd {
        prefill: Vec<Arc<Worker>>,
        decode: Vec<Arc<Worker>>,
    },
}

/// Reason the resolver could not satisfy a request — exposed so the
/// handler can map to the right HTTP error code.
#[derive(Debug, PartialEq, Eq)]
pub enum PdResolveError {
    /// The model has no workers registered at all, healthy or not.
    /// Surfaced as 503 `no_healthy_workers`.
    NoHealthyWorkers,
    /// PD-mode deployment whose prefill pool is empty (all
    /// breakers-open or no prefill workers ever registered).
    /// Surfaced as 503 `no_prefill_workers_available`.
    NoPrefillWorkersAvailable,
    /// PD-mode deployment whose decode pool is empty.
    /// Surfaced as 503 `no_decode_workers_available` (added alongside
    /// `no_prefill_workers_available` in M4 carry-forward).
    NoDecodeWorkersAvailable,
}

/// Thin façade over [`WorkerRegistry`] that returns the per-pool
/// candidate sets for a model. Cheap to construct; the registry is
/// shared.
#[derive(Debug, Clone)]
pub struct PdPoolResolver {
    workers: Arc<WorkerRegistry>,
}

impl PdPoolResolver {
    pub fn new(workers: Arc<WorkerRegistry>) -> Self {
        Self { workers }
    }

    /// Classify a model and return its pool partition over healthy
    /// workers. Workers whose circuit breaker is open are filtered out
    /// at this layer so the policy never has to re-check.
    ///
    /// Returns `Err(NoHealthyWorkers)` only when the entire model has
    /// zero healthy workers; PD-mode partial failures (one pool empty,
    /// the other non-empty) succeed here — the handler decides whether
    /// the empty pool matters for the request it is dispatching.
    pub fn resolve(&self, model: &ModelId) -> Result<PdPools, PdResolveError> {
        let all = self.workers.healthy_workers_for(model);
        if all.is_empty() {
            return Err(PdResolveError::NoHealthyWorkers);
        }
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        let mut plain = Vec::new();
        for w in all {
            match w.mode() {
                WorkerMode::Prefill => prefill.push(w),
                WorkerMode::Decode => decode.push(w),
                WorkerMode::Plain => plain.push(w),
            }
        }
        // PD-mode iff any prefill OR any decode worker exists. Mixing
        // plain + prefill on the same model_id is a discovery-level
        // misconfiguration we do not try to repair here — we treat any
        // role tag at all as PD intent. The plain workers in that case
        // become unreachable, which is loud enough at the metrics layer
        // for operators to notice.
        if !prefill.is_empty() || !decode.is_empty() {
            Ok(PdPools::Pd { prefill, decode })
        } else {
            Ok(PdPools::Plain { workers: plain })
        }
    }

    /// Convenience for the prefill dispatch path. Returns the prefill
    /// pool for a PD model, or the full plain pool for a non-PD model.
    /// Errors when the relevant pool is empty.
    pub fn prefill_candidates(&self, model: &ModelId) -> Result<Vec<Arc<Worker>>, PdResolveError> {
        match self.resolve(model)? {
            PdPools::Plain { workers } => Ok(workers),
            PdPools::Pd { prefill, .. } => {
                if prefill.is_empty() {
                    Err(PdResolveError::NoPrefillWorkersAvailable)
                } else {
                    Ok(prefill)
                }
            }
        }
    }

    /// Convenience for the decode dispatch path. Mirror of
    /// [`Self::prefill_candidates`].
    pub fn decode_candidates(&self, model: &ModelId) -> Result<Vec<Arc<Worker>>, PdResolveError> {
        match self.resolve(model)? {
            PdPools::Plain { workers } => Ok(workers),
            PdPools::Pd { decode, .. } => {
                if decode.is_empty() {
                    Err(PdResolveError::NoDecodeWorkersAvailable)
                } else {
                    Ok(decode)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec};

    fn spec(id: &str, mode: WorkerMode, model: &str) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}"),
            mode,
            model_ids: vec![ModelId(model.into())],
        }
    }

    fn registry(specs: &[WorkerSpec]) -> Arc<WorkerRegistry> {
        let r = Arc::new(WorkerRegistry::default());
        for s in specs {
            r.add(s.clone());
        }
        r
    }

    /// Model with only Plain workers → Plain partition.
    #[test]
    fn plain_mode_returns_all_plain_workers() {
        let r = registry(&[
            spec("w1", WorkerMode::Plain, "m"),
            spec("w2", WorkerMode::Plain, "m"),
        ]);
        let res = PdPoolResolver::new(r)
            .resolve(&ModelId("m".into()))
            .unwrap();
        match res {
            PdPools::Plain { workers } => assert_eq!(workers.len(), 2),
            PdPools::Pd { .. } => panic!("expected Plain"),
        }
    }

    /// Model with prefill + decode → Pd partition, both pools populated.
    #[test]
    fn pd_mode_returns_distinct_pools() {
        let r = registry(&[
            spec("p1", WorkerMode::Prefill, "m"),
            spec("d1", WorkerMode::Decode, "m"),
            spec("d2", WorkerMode::Decode, "m"),
        ]);
        let res = PdPoolResolver::new(r)
            .resolve(&ModelId("m".into()))
            .unwrap();
        match res {
            PdPools::Pd { prefill, decode } => {
                assert_eq!(prefill.len(), 1);
                assert_eq!(decode.len(), 2);
                // No cross-contamination: each worker carries the
                // right mode.
                assert!(prefill.iter().all(|w| w.mode() == WorkerMode::Prefill));
                assert!(decode.iter().all(|w| w.mode() == WorkerMode::Decode));
            }
            PdPools::Plain { .. } => panic!("expected Pd"),
        }
    }

    /// Unknown model → NoHealthyWorkers.
    #[test]
    fn unknown_model_returns_no_healthy_workers() {
        let r = Arc::new(WorkerRegistry::default());
        let err = PdPoolResolver::new(r)
            .resolve(&ModelId("ghost".into()))
            .unwrap_err();
        assert_eq!(err, PdResolveError::NoHealthyWorkers);
    }

    /// Gap closer #1: PD mode with no prefill workers → resolve()
    /// returns a Pd partition with an empty prefill pool, and
    /// `prefill_candidates()` errors with NoPrefillWorkersAvailable.
    #[test]
    fn pd_mode_with_no_prefill_errors_on_prefill_dispatch() {
        let r = registry(&[
            spec("d1", WorkerMode::Decode, "m"),
            spec("d2", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let model = ModelId("m".into());
        // resolve() succeeds — we have decode workers.
        match resolver.resolve(&model).unwrap() {
            PdPools::Pd { prefill, decode } => {
                assert!(prefill.is_empty());
                assert_eq!(decode.len(), 2);
            }
            other => panic!("expected Pd, got {other:?}"),
        }
        // prefill_candidates errors.
        let err = resolver.prefill_candidates(&model).unwrap_err();
        assert_eq!(err, PdResolveError::NoPrefillWorkersAvailable);
        // decode_candidates succeeds.
        let decode = resolver.decode_candidates(&model).unwrap();
        assert_eq!(decode.len(), 2);
    }

    /// Symmetric: PD mode with no decode workers → decode dispatch
    /// errors.
    #[test]
    fn pd_mode_with_no_decode_errors_on_decode_dispatch() {
        let r = registry(&[
            spec("p1", WorkerMode::Prefill, "m"),
            spec("p2", WorkerMode::Prefill, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let model = ModelId("m".into());
        let err = resolver.decode_candidates(&model).unwrap_err();
        assert_eq!(err, PdResolveError::NoDecodeWorkersAvailable);
    }

    /// PR #25184 carry-forward: separate models don't cross-contaminate.
    /// One model is PD, the other is plain; resolving one must not return
    /// workers from the other's pool.
    #[test]
    fn distinct_models_isolated_across_pd_and_plain() {
        let r = registry(&[
            spec("plain1", WorkerMode::Plain, "plainmodel"),
            spec("p1", WorkerMode::Prefill, "pdmodel"),
            spec("d1", WorkerMode::Decode, "pdmodel"),
        ]);
        let resolver = PdPoolResolver::new(r);
        match resolver.resolve(&ModelId("plainmodel".into())).unwrap() {
            PdPools::Plain { workers } => assert_eq!(workers.len(), 1),
            _ => panic!("plainmodel should resolve to Plain"),
        }
        match resolver.resolve(&ModelId("pdmodel".into())).unwrap() {
            PdPools::Pd { prefill, decode } => {
                assert_eq!(prefill.len(), 1);
                assert_eq!(decode.len(), 1);
            }
            _ => panic!("pdmodel should resolve to Pd"),
        }
    }

    /// Plain-mode prefill_candidates returns the plain pool (non-PD
    /// shorthand: dispatch helpers Just Work for plain models).
    #[test]
    fn plain_mode_prefill_candidates_returns_plain_pool() {
        let r = registry(&[spec("w1", WorkerMode::Plain, "m")]);
        let resolver = PdPoolResolver::new(r);
        let v = resolver.prefill_candidates(&ModelId("m".into())).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].mode(), WorkerMode::Plain);
    }
}
