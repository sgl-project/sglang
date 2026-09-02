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
use crate::server::metrics::DecodeAffinityOutcome;
use crate::workers::{Worker, WorkerRegistry};
use std::sync::Arc;

/// Multiplier over the median decode-pool load above which a same-host
/// decode peer is considered "too hot" — we fall back to the lowest-load
/// peer outside the affinity preference. Two-times-median keeps short
/// load bursts on the same host (NCCL chatter, GPU sharing) from being
/// treated as overload while still avoiding pinning to a wedged peer.
const AFFINITY_LOAD_TOLERANCE: f64 = 2.0;

/// Resolution result for a single request route. The handler picks
/// `prefill` / `decode` based on whether it is dispatching prefill or
/// decode traffic; `plain` is for non-PD models.
#[derive(Debug)]
pub enum PdPools {
    /// Non-PD deployment: the model is served by plain workers.
    Plain { workers: Vec<Arc<Worker>> },
    /// PD-disaggregation deployment: the model has prefill and/or decode
    /// workers. Either OR BOTH pools may be empty (e.g. every prefill
    /// worker's circuit breaker is open, or every PD worker on the
    /// model is currently unhealthy). The `*_candidates` helpers are
    /// the only safe consumers — they map an empty pool to the
    /// appropriate `NoPrefillWorkersAvailable` / `NoDecodeWorkersAvailable`
    /// error. Callers that read this variant directly MUST treat an
    /// empty pool as a transient failure, not as "zero work".
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
    /// Surfaced as 503 `no_decode_workers_available`.
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
    /// Returns `Err(NoHealthyWorkers)` only when the model has zero
    /// **registered** workers (healthy or not). When the model is
    /// registered as PD but every PD worker is currently unhealthy
    /// (any failure path that flips `breaker.allow()` to false),
    /// returns `Ok(Pd { prefill: [], decode: [] })` so
    /// `prefill_candidates` / `decode_candidates` can surface the more
    /// specific `NoPrefillWorkersAvailable` / `NoDecodeWorkersAvailable`
    /// code — operators alerting on partial-pool failures see the same
    /// code whether the empty pool is empty by registration or by
    /// transient health state.
    pub fn resolve(&self, model: &ModelId) -> Result<PdPools, PdResolveError> {
        let all = self.workers.healthy_workers_for(model);
        if all.is_empty() {
            // No healthy workers — distinguish "model never registered"
            // (true 404-ish, operator misconfiguration) from "PD model
            // with all breakers currently open" (transient health
            // issue, deserves the per-pool code).
            let registered = self.workers.workers_for(model);
            let pd_intent = registered
                .iter()
                .any(|w| matches!(w.mode(), WorkerMode::Prefill | WorkerMode::Decode));
            return if pd_intent {
                Ok(PdPools::Pd {
                    prefill: Vec::new(),
                    decode: Vec::new(),
                })
            } else {
                Err(PdResolveError::NoHealthyWorkers)
            };
        }
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        let mut plain = Vec::new();
        for w in all {
            match w.mode() {
                // A prefill worker without a bootstrap port cannot take a
                // PD request: the router would inject `bootstrap_port:
                // null` and the decode side could never connect for the
                // KV transfer. Exclude it from the routable pool (it stays
                // registered — the manager re-introspects `/server_info`
                // until the port shows up). Mirrors AIgate's rule that a
                // P/D endpoint with incomplete metadata is not selectable.
                WorkerMode::Prefill if w.bootstrap_port().is_none() => {
                    tracing::debug!(
                        worker_url = %w.url,
                        "prefill worker has no bootstrap_port; excluded from routing",
                    );
                }
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

    /// Pick a decode worker for a PD-mode handoff with **host affinity**
    /// to the prefill worker. Resolves the decode pool for `model`, then
    /// applies the affinity rules in [`select_decode_with_affinity`].
    ///
    /// Returns `Err(NoDecodeWorkersAvailable)` if the decode pool is
    /// empty (PD-mode partial failure) — the chat handler then maps to
    /// 503 `no_decode_workers_available`. For non-PD (plain) models
    /// this is a no-op call — there is no decode peer to find — and
    /// the caller should NOT use this helper.
    pub fn decode_with_affinity(
        &self,
        model: &ModelId,
        prefill_url: &str,
    ) -> Result<Arc<Worker>, PdResolveError> {
        let candidates = self.decode_candidates(model)?;
        select_decode_with_affinity(prefill_url, &candidates)
            .ok_or(PdResolveError::NoDecodeWorkersAvailable)
    }
}

/// Restrict a decode candidate set to the prefill worker's transfer group.
/// Ungrouped (`None`) decode workers pair only with ungrouped prefill
/// workers — a labelled and an unlabelled worker are never paired, since
/// the label is the operator's statement of KV-transfer reachability.
pub fn same_transfer_group(candidates: &[Arc<Worker>], group: Option<&str>) -> Vec<Arc<Worker>> {
    candidates
        .iter()
        .filter(|w| w.transfer_group() == group)
        .cloned()
        .collect()
}

/// Keep only prefill workers whose transfer group has at least one decode
/// candidate in `decode_pool`, so a prefill that could never be paired
/// (its group's decode side is down / stale / absent) is not selected.
pub fn prefill_with_compatible_group(
    prefill: Vec<Arc<Worker>>,
    decode_pool: &[Arc<Worker>],
) -> Vec<Arc<Worker>> {
    prefill
        .into_iter()
        .filter(|p| {
            decode_pool
                .iter()
                .any(|d| d.transfer_group() == p.transfer_group())
        })
        .collect()
}

/// Pick a decode worker from `candidates` preferring the one whose URL
/// shares a host with `prefill_url`. Falls back to lowest-load when no
/// same-host peer exists, when the same-host peer's breaker is open,
/// or when the same-host peer is overloaded relative to the pool.
///
/// # Rules
///
/// 1. **Same-host preference.** Parse the host portion of both URLs
///    (`url::Url::host_str`). If any candidate shares the host AND has
///    a closed circuit breaker AND has `active_load <=
///    AFFINITY_LOAD_TOLERANCE × median(decode_pool_load)`, return it.
/// 2. **Fallback: min-load among closed-breaker candidates.** No
///    same-host peer, or the same-host peer was filtered by rule 1's
///    health/load gates.
/// 3. **Last resort: min-load over ALL candidates.** Every candidate
///    has its breaker open; the next dispatch will likely fail too,
///    but a min-load fallback keeps the selection function total.
///    Callers should observe the breaker-open error and surface it as
///    `BreakerOpen`, not silently retry.
///
/// Returns `None` only when `candidates` is empty.
///
/// # Why a free-standing function vs a `Policy::select` extension?
///
/// The current `Policy` trait carries `(workers, ctx)`; adding an
/// `affinity_hint` argument would touch every policy implementation
/// (`round_robin`, `random`, `power_of_two`, `cache_aware_zmq`).
/// Affinity is a PD-routing concern — orthogonal to the in-pool
/// scoring the trait abstracts — so keeping it as a sibling helper
/// keeps the trait's responsibility narrow.
pub fn select_decode_with_affinity(
    prefill_url: &str,
    candidates: &[Arc<Worker>],
) -> Option<Arc<Worker>> {
    select_decode_with_affinity_outcome(prefill_url, candidates).map(|(w, _)| w)
}

/// [`select_decode_with_affinity`] plus WHY the pick was made, for the
/// `sgl_router_decode_affinity_total{outcome}` counter:
///
/// * `SameHostPicked` — rule 1 held.
/// * `FallbackBreaker` — a same-host peer exists but every such peer's
///   breaker is open.
/// * `FallbackLoadImbalance` — a same-host peer exists and is healthy but
///   sits above the load tolerance.
/// * `NoSameHostPeer` — no candidate shares the prefill host.
pub fn select_decode_with_affinity_outcome(
    prefill_url: &str,
    candidates: &[Arc<Worker>],
) -> Option<(Arc<Worker>, DecodeAffinityOutcome)> {
    if candidates.is_empty() {
        return None;
    }
    let prefill_host = host_of(prefill_url);

    // Build the closed-breaker subset once; both the affinity branch
    // and the fallback branch read from it. `would_allow` (non-mutating)
    // is the right filter — `allow()` would claim a half-open probe for
    // every candidate we look at, including ones we never dispatch to.
    let healthy: Vec<&Arc<Worker>> = candidates
        .iter()
        .filter(|w| w.breaker.would_allow())
        .collect();

    // Compute the median load over the closed-breaker subset.  Empty
    // subset → median is 0 (means: every peer's breaker is open; the
    // affinity gate is moot, we'll fall through to the last-resort
    // branch).
    let load_tolerance = if healthy.is_empty() {
        0
    } else {
        let mut loads: Vec<usize> = healthy.iter().map(|w| w.active_load()).collect();
        loads.sort_unstable();
        let median = loads[loads.len() / 2];
        ((median as f64) * AFFINITY_LOAD_TOLERANCE).ceil() as usize
    };

    // Rule 1: same-host AND healthy AND not overloaded.
    let mut outcome = DecodeAffinityOutcome::NoSameHostPeer;
    if let Some(host) = prefill_host.as_deref() {
        let same_host: Vec<&Arc<Worker>> = candidates
            .iter()
            .filter(|w| host_of(&w.url).as_deref() == Some(host))
            .collect();
        if !same_host.is_empty() {
            if let Some(w) = same_host.iter().find(|w| {
                w.breaker.would_allow()
                    && (load_tolerance == 0 || w.active_load() <= load_tolerance)
            }) {
                return Some((Arc::clone(w), DecodeAffinityOutcome::SameHostPicked));
            }
            outcome = if same_host.iter().any(|w| w.breaker.would_allow()) {
                DecodeAffinityOutcome::FallbackLoadImbalance
            } else {
                DecodeAffinityOutcome::FallbackBreaker
            };
        }
    }

    // Rule 2: min-load among healthy.
    if let Some(w) = healthy.iter().min_by_key(|w| w.active_load()) {
        return Some((Arc::clone(w), outcome));
    }

    // Rule 3: last-resort min-load over all candidates (every
    // breaker is open). The caller's dispatch will likely fail and
    // surface `BreakerOpen`, but the selection function stays total.
    candidates
        .iter()
        .min_by_key(|w| w.active_load())
        .cloned()
        .map(|w| (w, outcome))
}

/// Parse the host portion of a worker URL. Returns `None` when the URL
/// fails to parse or has no host (rare; discovery emits URLs the proxy
/// has already used at least once for /server_info, so this is mostly
/// defensive).
fn host_of(worker_url: &str) -> Option<String> {
    url::Url::parse(worker_url)
        .ok()?
        .host_str()
        .map(str::to_owned)
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
            // Prefill workers need a bootstrap port to be routable.
            bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
            transfer_group: None,
        }
    }

    fn registry(specs: &[WorkerSpec]) -> Arc<WorkerRegistry> {
        let r = Arc::new(WorkerRegistry::default());
        for s in specs {
            let _ = r.add(s.clone());
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

    /// PD mode where every breaker is open (e.g. the upstream pool went
    /// hard down) must NOT collapse to the generic `NoHealthyWorkers`
    /// code. Both `prefill_candidates` and `decode_candidates` should
    /// still surface the per-pool variant so operators can alert on
    /// "prefill tier degraded" independently from "model misconfigured".
    #[test]
    fn pd_mode_all_breakers_open_keeps_per_pool_codes() {
        let r = registry(&[
            spec("p1", WorkerMode::Prefill, "m"),
            spec("d1", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let model = ModelId("m".into());
        // Trip both breakers. Loop on `allow()` (not a fixed count) so
        // the test stays correct if the default `CircuitBreakerConfig`
        // threshold ever changes.
        for w in resolver.workers.workers_for(&model) {
            while w.breaker.allow() {
                w.breaker.record_failure();
            }
        }
        // resolve() still returns a PD shape (both pools empty) — the
        // PD intent is preserved across the breaker-open state.
        match resolver.resolve(&model).unwrap() {
            PdPools::Pd { prefill, decode } => {
                assert!(prefill.is_empty());
                assert!(decode.is_empty());
            }
            other => panic!("expected Pd, got {other:?}"),
        }
        // prefill dispatch → NoPrefillWorkersAvailable (not NoHealthyWorkers).
        assert_eq!(
            resolver.prefill_candidates(&model).unwrap_err(),
            PdResolveError::NoPrefillWorkersAvailable,
        );
        // decode dispatch → NoDecodeWorkersAvailable (not NoHealthyWorkers).
        assert_eq!(
            resolver.decode_candidates(&model).unwrap_err(),
            PdResolveError::NoDecodeWorkersAvailable,
        );
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

    // === Decoder affinity (Task C) ===

    /// Build a `WorkerSpec` with an explicit URL — the affinity tests
    /// distinguish workers by host, so they care about the URL string
    /// directly, not the generated `http://{id}` form.
    fn spec_with_url(id: &str, url: &str, mode: WorkerMode, model: &str) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(id.into()),
            url: url.into(),
            mode,
            model_ids: vec![ModelId(model.into())],
            bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
            transfer_group: None,
        }
    }

    /// A prefill worker that has not (yet) disclosed its bootstrap port is
    /// registered but NOT routable: the resolver drops it from the prefill
    /// pool so the router never injects `bootstrap_port: null`. With no
    /// other prefill peer the pool is empty → `NoPrefillWorkersAvailable`.
    #[test]
    fn prefill_without_bootstrap_port_is_excluded_from_pool() {
        let m = "m";
        let mut portless = spec("p-noport", WorkerMode::Prefill, m);
        portless.bootstrap_port = None;
        let r = registry(&[portless, spec("d1", WorkerMode::Decode, m)]);
        let resolver = PdPoolResolver::new(r.clone());
        assert_eq!(
            resolver.prefill_candidates(&ModelId(m.into())).unwrap_err(),
            PdResolveError::NoPrefillWorkersAvailable,
        );
        // Decode pool is unaffected.
        assert_eq!(
            resolver
                .decode_candidates(&ModelId(m.into()))
                .unwrap()
                .len(),
            1
        );
        // A sibling WITH a port is the only prefill candidate.
        let _ = r.add(spec("p-ok", WorkerMode::Prefill, m));
        let prefill = resolver.prefill_candidates(&ModelId(m.into())).unwrap();
        assert_eq!(prefill.len(), 1);
        assert_eq!(prefill[0].id, WorkerId("p-ok".into()));
    }

    fn spec_in_group(id: &str, mode: WorkerMode, model: &str, group: Option<&str>) -> WorkerSpec {
        let mut s = spec(id, mode, model);
        s.transfer_group = group.map(str::to_string);
        s
    }

    /// Decode candidates are restricted to the prefill's transfer group;
    /// ungrouped only pairs with ungrouped.
    #[test]
    fn same_transfer_group_filters_by_exact_group() {
        let r = registry(&[
            spec_in_group("d-a", WorkerMode::Decode, "m", Some("a")),
            spec_in_group("d-b", WorkerMode::Decode, "m", Some("b")),
            spec_in_group("d-none", WorkerMode::Decode, "m", None),
        ]);
        let pool = r.workers_for(&ModelId("m".into()));
        let ids = |v: Vec<Arc<Worker>>| {
            let mut ids: Vec<String> = v.iter().map(|w| w.id.0.clone()).collect();
            ids.sort();
            ids
        };
        assert_eq!(ids(same_transfer_group(&pool, Some("a"))), vec!["d-a"]);
        assert_eq!(ids(same_transfer_group(&pool, Some("b"))), vec!["d-b"]);
        assert_eq!(ids(same_transfer_group(&pool, None)), vec!["d-none"]);
        assert!(same_transfer_group(&pool, Some("zzz")).is_empty());
    }

    /// A prefill whose group has no decode candidate is dropped from the
    /// prefill candidate set; groups with both sides survive.
    #[test]
    fn prefill_with_compatible_group_drops_incomplete_groups() {
        let r = registry(&[
            spec_in_group("p-a", WorkerMode::Prefill, "m", Some("a")),
            spec_in_group("p-b", WorkerMode::Prefill, "m", Some("b")),
            spec_in_group("p-none", WorkerMode::Prefill, "m", None),
            spec_in_group("d-a", WorkerMode::Decode, "m", Some("a")),
        ]);
        let resolver = PdPoolResolver::new(r.clone());
        let m = ModelId("m".into());
        let prefill = resolver.prefill_candidates(&m).unwrap();
        let decode = resolver.decode_candidates(&m).unwrap();
        let compatible = prefill_with_compatible_group(prefill, &decode);
        assert_eq!(compatible.len(), 1);
        assert_eq!(compatible[0].id, WorkerId("p-a".into()));
    }

    /// Same-host affinity: a request that lands on `prefill@host_a`
    /// picks `decode@host_a` even when `decode@host_b` has lower load.
    /// Pin: the affinity branch wins over load tiebreak when both
    /// candidates are healthy and not overloaded.
    #[test]
    fn decoder_picks_same_host_when_available() {
        let r = registry(&[
            spec_with_url("p1", "http://host_a:30000", WorkerMode::Prefill, "m"),
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let prefill_url = "http://host_a:30000";

        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), prefill_url)
            .unwrap();
        assert_eq!(
            chosen.url, "http://host_a:30001",
            "same-host decode peer must win over remote peer",
        );
    }

    /// Affinity peer's breaker is open → fall back to the remote
    /// healthy peer. Pin: the affinity rule must not pin a request to
    /// a known-bad worker just because the host matches.
    #[test]
    fn decoder_falls_back_when_affinity_peer_breaker_open() {
        let r = registry(&[
            spec_with_url("p1", "http://host_a:30000", WorkerMode::Prefill, "m"),
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);

        // Trip d1's breaker by saturating record_failure() against the
        // default config (threshold = 3). The breaker then denies
        // `allow()` until the cooldown elapses.
        let d1 = resolver
            .workers
            .healthy_workers_for(&ModelId("m".into()))
            .into_iter()
            .find(|w| w.url == "http://host_a:30001")
            .unwrap();
        for _ in 0..3 {
            d1.breaker.record_failure();
        }
        assert!(!d1.breaker.allow(), "d1 breaker must be open");

        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap();
        assert_eq!(
            chosen.url, "http://host_b:30001",
            "breaker-open affinity peer must fall back to the remote healthy peer",
        );
    }

    /// Affinity peer is overloaded (load > 2× median) → fall back to
    /// the remote lower-load peer. Pin: the load gate prevents a single
    /// host's wedged decode worker from absorbing every co-located
    /// prefill request.
    #[test]
    fn decoder_falls_back_when_affinity_peer_load_imbalance() {
        let r = registry(&[
            spec_with_url("p1", "http://host_a:30000", WorkerMode::Prefill, "m"),
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
            spec_with_url("d3", "http://host_c:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);

        // Loads: d1=20, d2=2, d3=2. Median = 2. 2× tolerance = 4.
        // d1 is overloaded (20 > 4) → affinity rule rejects d1.
        let decode_pool = resolver
            .workers
            .healthy_workers_for(&ModelId("m".into()))
            .into_iter()
            .filter(|w| w.mode() == WorkerMode::Decode)
            .collect::<Vec<_>>();
        let d1 = decode_pool
            .iter()
            .find(|w| w.url == "http://host_a:30001")
            .unwrap();
        let d2 = decode_pool
            .iter()
            .find(|w| w.url == "http://host_b:30001")
            .unwrap();
        let d3 = decode_pool
            .iter()
            .find(|w| w.url == "http://host_c:30001")
            .unwrap();
        let mut guards = Vec::new();
        for _ in 0..20 {
            guards.push(d1.load_guard());
        }
        for _ in 0..2 {
            guards.push(d2.load_guard());
            guards.push(d3.load_guard());
        }

        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap();
        assert!(
            chosen.url == "http://host_b:30001" || chosen.url == "http://host_c:30001",
            "overloaded affinity peer must fall back to a remote min-load peer, got: {}",
            chosen.url,
        );
        // Drop guards explicitly so the test cleanup doesn't depend on
        // RAII order against the resolver / registry.
        drop(guards);
    }

    /// No same-host decode peer exists → fall back to min-load remote.
    #[test]
    fn decoder_falls_back_when_no_same_host_peer() {
        let r = registry(&[
            spec_with_url("p1", "http://host_a:30000", WorkerMode::Prefill, "m"),
            spec_with_url("d1", "http://host_b:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_c:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);

        // Bump d1 to 1, d2 stays at 0 — min-load picks d2.
        let pool = resolver
            .workers
            .healthy_workers_for(&ModelId("m".into()))
            .into_iter()
            .filter(|w| w.mode() == WorkerMode::Decode)
            .collect::<Vec<_>>();
        let d1 = pool
            .iter()
            .find(|w| w.url == "http://host_b:30001")
            .unwrap();
        let _g = d1.load_guard();

        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap();
        assert_eq!(
            chosen.url, "http://host_c:30001",
            "no same-host peer → min-load fallback over remote candidates",
        );
    }

    /// Empty decode pool → `NoDecodeWorkersAvailable`. The chat
    /// handler maps this to 503 `no_decode_workers_available`.
    #[test]
    fn decoder_with_affinity_returns_error_when_pool_empty() {
        let r = registry(&[spec_with_url(
            "p1",
            "http://host_a:30000",
            WorkerMode::Prefill,
            "m",
        )]);
        let resolver = PdPoolResolver::new(r);
        let err = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap_err();
        assert_eq!(err, PdResolveError::NoDecodeWorkersAvailable);
    }

    /// Prefill URL is malformed (no host) → still picks a min-load
    /// decode peer. Affinity is best-effort; a parse failure must not
    /// kill the request.
    #[test]
    fn decoder_handles_malformed_prefill_url_via_min_load_fallback() {
        let r = registry(&[
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let chosen = resolver
            .decode_with_affinity(&ModelId("m".into()), "not-a-url")
            .unwrap();
        // Both d1 and d2 are at load 0 → either is acceptable. The
        // assertion is only that the function returns Some, not None
        // / panic.
        assert!(
            chosen.url == "http://host_a:30001" || chosen.url == "http://host_b:30001",
            "unexpected decode worker chosen: {}",
            chosen.url,
        );
    }

    /// All decode peers' breakers are open → `decode_with_affinity`
    /// surfaces `NoDecodeWorkersAvailable` (the per-pool variant), not
    /// the generic `NoHealthyWorkers`. The PD intent is preserved
    /// through `resolve` so operators alerting on "decode tier down"
    /// see the same code regardless of whether the pool is empty by
    /// registration or by breaker state.
    ///
    /// The lower-level helper [`select_decode_with_affinity`] is total
    /// even when every candidate's breaker is open (rule 3 in the
    /// docstring): tests that call it directly with breaker-open
    /// candidates get a min-load result.
    #[test]
    fn decoder_with_affinity_errors_when_all_breakers_open() {
        let r = registry(&[
            spec_with_url("d1", "http://host_a:30001", WorkerMode::Decode, "m"),
            spec_with_url("d2", "http://host_b:30001", WorkerMode::Decode, "m"),
        ]);
        let resolver = PdPoolResolver::new(r);
        let pool = resolver
            .workers
            .workers_for(&ModelId("m".into()))
            .into_iter()
            .filter(|w| w.mode() == WorkerMode::Decode)
            .collect::<Vec<_>>();
        // Trip every decode breaker; loop on `allow()` for threshold
        // resilience.
        for w in &pool {
            while w.breaker.allow() {
                w.breaker.record_failure();
            }
        }
        // resolver path: healthy_workers_for returns empty, but the
        // model is registered as PD (decode peers exist), so resolve()
        // preserves PD shape and decode_with_affinity surfaces the
        // per-pool code.
        let err = resolver
            .decode_with_affinity(&ModelId("m".into()), "http://host_a:30000")
            .unwrap_err();
        assert_eq!(err, PdResolveError::NoDecodeWorkersAvailable);

        // helper path with a non-empty (but all-breaker-open) slice
        // returns Some via the last-resort branch — selection function
        // stays total, caller sees `BreakerOpen` on dispatch.
        let any = select_decode_with_affinity("http://host_a:30000", &pool).unwrap();
        assert!(
            any.url == "http://host_a:30001" || any.url == "http://host_b:30001",
            "last-resort path must return some candidate, got: {}",
            any.url,
        );
    }
}
