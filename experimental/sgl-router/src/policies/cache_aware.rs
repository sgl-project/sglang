// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware selection: prefer the engine holding the longest usable prefix
//! of the prompt, gated by queue depth and admission, guarded against sending
//! a near-tie to a much more pressured engine; a miss falls back to a
//! power-of-two choice.

use super::{
    power_of_two, Admission, AdmissionContext, EngineRejection, Pick, PickError, PickMode,
    PickRequest, PickResult, Policy,
};
use crate::config::AffinityConfig;
use crate::discovery::ModelId;
use crate::policies::state::engine_load::{EngineLoadSnapshot, FreshLoadLookup};
use crate::policies::state::kv_events::{BlockSizeOracle, KvEventIndex};
use crate::policies::state::PrefixLookup;
use crate::server::metrics::{CacheAwareDecision, MetricsRegistry};
use crate::workers::Worker;
use futures::future::BoxFuture;
use sgl_kv_indexer::{PrefixIndex, PrefixIndexError};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};

/// Sampled operator diagnostics: one line per this many occurrences.
const LOG_SAMPLE: u64 = 64;
static QUEUE_GATE_BLIND_LOGS: AtomicU64 = AtomicU64::new(0);
static SATURATION_PIN_LOGS: AtomicU64 = AtomicU64::new(0);

fn sampled(counter: &AtomicU64) -> bool {
    counter
        .fetch_add(1, AtomicOrdering::Relaxed)
        .is_multiple_of(LOG_SAMPLE)
}

/// Where prefix ownership comes from: the router-local KV-event index or an
/// external indexer.
pub enum CacheSource {
    Local(Arc<KvEventIndex>),
    Remote {
        index: Arc<dyn PrefixIndex>,
        block_size: Arc<BlockSizeOracle>,
    },
}

impl fmt::Debug for CacheSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Local(_) => "CacheSource::Local",
            Self::Remote { .. } => "CacheSource::Remote",
        })
    }
}

impl CacheSource {
    /// An unavailable backend reads as a miss; a rejected query is an error.
    async fn lookup(
        &self,
        tokens: Option<&[u32]>,
        model: &ModelId,
    ) -> Result<Option<PrefixLookup>, PickError> {
        let Some(tokens) = tokens else {
            return Ok(None);
        };
        let (index, block_size) = match self {
            Self::Local(index) => return Ok(index.match_prefix(tokens)),
            Self::Remote { index, block_size } => (index, block_size),
        };
        let Some(hashes) = block_size.block_hashes(tokens) else {
            return Ok(None);
        };
        let query_blocks = hashes.len();
        if hashes.is_empty() {
            return Ok(Some(PrefixLookup::from_indexer(
                sgl_kv_indexer::PrefixOutcome::Empty,
                0,
            )));
        }
        let outcome = match index.match_prefix(hashes).await {
            Ok(outcome) => outcome,
            Err(
                error @ (PrefixIndexError::Overloaded
                | PrefixIndexError::Timeout
                | PrefixIndexError::Unreachable
                | PrefixIndexError::QueryTooLarge),
            ) => {
                tracing::warn!(%model, error = %error, "KV Indexer unavailable; falling back to min-load routing");
                sgl_kv_indexer::PrefixOutcome::Empty
            }
            Err(error) => {
                tracing::warn!(%model, error = %error, "KV Indexer rejected the query");
                return Err(PickError::InvalidSignal(error.to_string()));
            }
        };
        Ok(Some(PrefixLookup::from_indexer(outcome, query_blocks)))
    }
}

/// Per-request memo shared by every bucket's pick: the prefix lookup runs
/// once, and the first tournament's gate audit decides the miss label.
#[derive(Default)]
pub struct PrefixMemo {
    lookup: tokio::sync::OnceCell<Option<PrefixLookup>>,
    audit: OnceLock<GateAudit>,
}

impl PrefixMemo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resolved(lookup: Option<PrefixLookup>) -> Self {
        Self {
            lookup: tokio::sync::OnceCell::from(lookup),
            audit: OnceLock::new(),
        }
    }
}

/// What the queue gate and admission did to the candidate set; see
/// `fallback_decision` for how it is read.
#[derive(Debug, Clone, Copy, Default)]
struct GateAudit {
    queue_gate_rejected: u64,
    admission_evaluated: u64,
    fleet_all_queued: bool,
    best_rejected_blocks: u32,
}

struct Candidate<'e> {
    engine: &'e Arc<Worker>,
    matched_prefix_tokens: u64,
    uncached_tokens: u64,
    matched_prefix_blocks: u32,
}

struct Resolution<'e> {
    winner: Option<(&'e Candidate<'e>, &'static str)>,
    audit: GateAudit,
    rejections: Vec<EngineRejection>,
}

#[derive(Debug)]
pub struct CacheAwarePolicy {
    source: CacheSource,
    admission: Admission,
    config: AffinityConfig,
    metrics: Arc<MetricsRegistry>,
}

impl CacheAwarePolicy {
    pub fn new(
        source: CacheSource,
        admission: Admission,
        config: AffinityConfig,
        metrics: Arc<MetricsRegistry>,
    ) -> Self {
        Self {
            source,
            admission,
            config,
            metrics,
        }
    }

    async fn pick_async(&self, engines: &[Arc<Worker>], request: &PickRequest<'_>) -> PickResult {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        // An earlier group already ran the tournament over a superset; only the fallback is left.
        if let Some(audit) = request.prefix.and_then(|memo| memo.audit.get()) {
            if request.mode == PickMode::Normal {
                return self.fallback(engines, request, *audit);
            }
        }
        let owned;
        let lookup = match request.prefix {
            Some(memo) => memo
                .lookup
                .get_or_try_init(|| self.source.lookup(request.tokens, request.model))
                .await?
                .as_ref(),
            None => {
                owned = self.source.lookup(request.tokens, request.model).await?;
                owned.as_ref()
            }
        };
        let candidates = self.candidates(
            engines,
            request.input_tokens,
            lookup,
            request.load.snapshot(),
        );
        let resolution = self.resolve(&candidates, engines, request);
        if let Some(memo) = request.prefix {
            let _ = memo.audit.set(resolution.audit);
        }
        match (resolution.winner, request.mode) {
            (Some((candidate, reason)), _) => {
                self.book_hit(request, candidate, reason, resolution.audit);
                Ok(Pick {
                    engine: Arc::clone(candidate.engine),
                    reason,
                })
            }
            (None, PickMode::Normal) => self.fallback(engines, request, resolution.audit),
            (None, PickMode::HitRequired) if resolution.rejections.is_empty() => {
                Err(PickError::NoCandidates)
            }
            (None, PickMode::HitRequired) => {
                Err(PickError::NoAdmissibleEngine(resolution.rejections))
            }
        }
    }

    /// Bounded, ordered candidates among `engines` that pass the minimum-hit gate.
    fn candidates<'e>(
        &self,
        engines: &'e [Arc<Worker>],
        input_tokens: u64,
        lookup: Option<&PrefixLookup>,
        snapshot: &EngineLoadSnapshot,
    ) -> Vec<Candidate<'e>> {
        let Some(lookup) = lookup.filter(|lookup| lookup.query_blocks > 0) else {
            return Vec::new();
        };
        let by_url: HashMap<&str, &'e Arc<Worker>> =
            engines.iter().map(|e| (e.url.as_str(), e)).collect();
        let mut seen = HashSet::new();
        let mut candidates: Vec<Candidate<'e>> = lookup
            .matches
            .iter()
            .filter_map(|m| {
                let engine = by_url.get(m.address.as_str())?;
                if m.matched_prefix_blocks == 0 || !seen.insert(&engine.id) {
                    return None;
                }
                let blocks = m
                    .matched_prefix_blocks
                    .min(u32::try_from(lookup.query_blocks).unwrap_or(u32::MAX));
                let matched_prefix_tokens =
                    input_tokens.saturating_mul(u64::from(blocks)) / lookup.query_blocks as u64;
                self.passes_gate(input_tokens, matched_prefix_tokens)
                    .then(|| Candidate {
                        engine,
                        matched_prefix_tokens,
                        uncached_tokens: input_tokens.saturating_sub(matched_prefix_tokens),
                        matched_prefix_blocks: blocks,
                    })
            })
            .collect();
        let limit = self.candidate_limit(engines.len());
        if limit == 0 {
            return Vec::new();
        }
        let loads = FreshLoadLookup::new(Some(snapshot), candidates.iter().map(|c| c.engine));
        let order = |left: &Candidate<'_>, right: &Candidate<'_>| {
            right
                .matched_prefix_tokens
                .cmp(&left.matched_prefix_tokens)
                .then_with(|| loads.compare_prefill_pressure(left.engine, right.engine))
                .then_with(|| left.engine.id.0.cmp(&right.engine.id.0))
        };
        if candidates.len() > limit {
            candidates.select_nth_unstable_by(limit, order);
            candidates.truncate(limit);
        }
        candidates.sort_by(order);
        candidates
    }

    fn passes_gate(&self, input_tokens: u64, matched_prefix_tokens: u64) -> bool {
        self.config
            .cache_affinity_min_matched_tokens
            .is_none_or(|minimum| matched_prefix_tokens >= minimum)
            && self
                .config
                .cache_affinity_min_match_ratio
                .is_none_or(|minimum| {
                    input_tokens > 0
                        && matched_prefix_tokens as f64 / input_tokens as f64 >= minimum
                })
    }

    fn candidate_limit(&self, pool: usize) -> usize {
        let proportional =
            (self.config.cache_candidate_ratio.clamp(0.0, 1.0) * pool as f64).ceil() as usize;
        pool.min(self.config.cache_candidate_max_workers)
            .min(self.config.cache_candidate_min_workers.max(proportional))
    }

    fn admits(
        &self,
        candidate: &Candidate<'_>,
        request: &PickRequest<'_>,
    ) -> Result<(), EngineRejection> {
        let ctx = AdmissionContext {
            load: request.load,
            kv_tokens: request.input_tokens,
            uncached_tokens: candidate.uncached_tokens,
        };
        self.admission
            .check(candidate.engine, &ctx)
            .map_err(|reason| EngineRejection {
                engine: candidate.engine.id.clone(),
                reason,
            })
    }

    /// The queue gate runs before admission so a queueing owner never counts
    /// as a capacity rejection, then the saturation pin, then the tournament
    /// among admitted candidates within the switch margin of the work floor.
    fn resolve<'c, 'e>(
        &self,
        candidates: &'c [Candidate<'e>],
        fleet: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Resolution<'c> {
        let snapshot = request.load.snapshot();
        let limit = self.config.worker_queue_limit;
        let gate_admits = |engine: &Worker| {
            limit.is_none_or(|limit| {
                snapshot
                    .fresh_load_for_url(&engine.url)
                    .is_none_or(|load| load.num_waiting_reqs < limit)
            })
        };
        if limit.is_some()
            && fleet
                .iter()
                .all(|e| snapshot.fresh_load_for_url(&e.url).is_none())
            && sampled(&QUEUE_GATE_BLIND_LOGS)
        {
            tracing::warn!(model = %request.model, worker_queue_limit = limit, workers = fleet.len(),
                "--worker-queue-limit is set but no worker has a fresh engine load sample, so the queue gate is inert");
        }
        let fleet_all_queued =
            limit.is_some() && !fleet.is_empty() && fleet.iter().all(|e| !gate_admits(e));
        let (mut evaluated, gated): (Vec<&Candidate<'e>>, Vec<&Candidate<'e>>) =
            candidates.iter().partition(|c| gate_admits(c.engine));
        // Nowhere unqueued to divert to: keep the prefix rather than trade it for the same wait.
        let fell_back = evaluated.is_empty()
            && !gated.is_empty()
            && fleet_all_queued
            && self.config.saturation_queue_floor.is_none();
        if fell_back {
            evaluated = candidates.iter().collect();
        }
        let loads = FreshLoadLookup::new(Some(snapshot), evaluated.iter().map(|c| c.engine));
        let mut rejections = Vec::new();
        let admitted: Vec<&Candidate<'e>> = evaluated
            .iter()
            .copied()
            .filter(|c| {
                self.admits(c, request)
                    .map_err(|r| rejections.push(r))
                    .is_ok()
            })
            .collect();
        let audit = GateAudit {
            queue_gate_rejected: gated.len() as u64,
            admission_evaluated: evaluated.len() as u64,
            fleet_all_queued,
            best_rejected_blocks: gated
                .iter()
                .map(|c| c.matched_prefix_blocks)
                .max()
                .unwrap_or(0),
        };
        self.metrics
            .record_cache_admission_evaluations(audit.admission_evaluated);
        self.metrics
            .record_cache_admission_rejections(rejections.len() as u64);
        self.metrics
            .record_cache_monitor_decision(loads.prefill_pressure_source());

        let Some(floor) = admitted.iter().copied().min_by_key(|c| c.uncached_tokens) else {
            let pinned = self.saturation_pin(&gated, fleet, request);
            self.metrics.record_cache_pressure_guard(0, 0);
            return Resolution {
                winner: pinned.map(|c| (c, "saturation_pin")),
                audit,
                rejections,
            };
        };
        let ceiling = floor
            .uncached_tokens
            .saturating_add(self.config.cache_switch_margin_tokens);
        let (mut winner, mut compared, mut overrides) = (floor, 0, 0);
        for candidate in admitted {
            if candidate.engine.id == winner.engine.id || candidate.uncached_tokens > ceiling {
                continue;
            }
            let baseline = self.compare(winner, candidate, &loads, false);
            let ordering = if self.config.pressure_guard
                && loads.comparable_get(&winner.engine.id).is_some()
                && loads.comparable_get(&candidate.engine.id).is_some()
            {
                compared += 1;
                let guarded = self.compare(winner, candidate, &loads, true);
                overrides += u64::from(guarded != baseline);
                guarded
            } else {
                baseline
            };
            if ordering.is_gt() {
                winner = candidate;
            }
        }
        self.metrics
            .record_cache_pressure_guard(compared, overrides);
        let reason = if fell_back {
            "saturation_pin"
        } else {
            "cache_candidate"
        };
        Resolution {
            winner: Some((winner, reason)),
            audit,
            rejections,
        }
    }

    /// With a floor configured and no fleet worker provably below it, wait at
    /// the least-pressured admitted prefix owner instead of cold-prefilling.
    fn saturation_pin<'c, 'e>(
        &self,
        gated: &[&'c Candidate<'e>],
        fleet: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Option<&'c Candidate<'e>> {
        let floor = self.config.saturation_queue_floor?;
        let snapshot = request.load.snapshot();
        if gated.is_empty()
            || snapshot.any_fresh_queue_below(fleet.iter().map(|e| e.url.as_str()), floor)
        {
            return None;
        }
        let loads = FreshLoadLookup::new(Some(snapshot), gated.iter().map(|c| c.engine));
        gated
            .iter()
            .copied()
            .filter(|c| self.admits(c, request).is_ok())
            .min_by(|l, r| {
                loads
                    .compare_prefill_pressure(l.engine, r.engine)
                    .then_with(|| l.engine.id.0.cmp(&r.engine.id.0))
            })
    }

    fn compare(
        &self,
        left: &Candidate<'_>,
        right: &Candidate<'_>,
        loads: &FreshLoadLookup<'_>,
        guard: bool,
    ) -> Ordering {
        if guard {
            if self.more_pressured(left.engine, right.engine, loads) {
                return Ordering::Greater;
            }
            if self.more_pressured(right.engine, left.engine, loads) {
                return Ordering::Less;
            }
        }
        left.uncached_tokens
            .cmp(&right.uncached_tokens)
            .then_with(|| loads.compare_prefill_pressure(left.engine, right.engine))
            .then_with(|| left.engine.id.0.cmp(&right.engine.id.0))
    }

    fn more_pressured(
        &self,
        candidate: &Worker,
        other: &Worker,
        loads: &FreshLoadLookup<'_>,
    ) -> bool {
        let (Some(c), Some(o)) = (
            loads.comparable_get(&candidate.id),
            loads.comparable_get(&other.id),
        ) else {
            return false;
        };
        if let (Some(ms), Some(c_ms), Some(o_ms)) = (
            self.config.pressure_abs_threshold_ms,
            c.estimated_prefill_queue_ms,
            o.estimated_prefill_queue_ms,
        ) {
            return c_ms - o_ms > ms && c_ms > o_ms * self.config.pressure_rel_threshold;
        }
        c.num_waiting_uncached_tokens
            .saturating_sub(o.num_waiting_uncached_tokens)
            > self.config.pressure_abs_threshold_tokens
            && c.num_waiting_uncached_tokens as f64
                > o.num_waiting_uncached_tokens as f64 * self.config.pressure_rel_threshold
    }

    /// Miss path: a power-of-two choice among admitted engines, preferring
    /// those the queue gate admits.
    fn fallback(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
        audit: GateAudit,
    ) -> PickResult {
        let admitted = self.admission.admit(engines, &request.admission())?;
        let snapshot = request.load.snapshot();
        let unqueued: Vec<Arc<Worker>> = match self.config.worker_queue_limit {
            Some(limit) => admitted
                .iter()
                .filter(|e| {
                    snapshot
                        .fresh_load_for_url(&e.url)
                        .is_none_or(|l| l.num_waiting_reqs < limit)
                })
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        let pool = if unqueued.is_empty() {
            &admitted[..]
        } else {
            &unqueued[..]
        };
        let engine =
            Arc::clone(power_of_two::choose(pool, request).ok_or(PickError::NoCandidates)?);
        let decision = fallback_decision(audit);
        if matches!(decision, CacheAwareDecision::CacheWorkerQueued) {
            self.metrics.observe_diverted_overlap_blocks(
                &request.model.0,
                u64::from(audit.best_rejected_blocks),
            );
        }
        self.metrics
            .record_cache_aware_decision(&request.model.0, decision);
        Ok(Pick {
            engine,
            reason: "no_cache_candidate",
        })
    }

    fn book_hit(
        &self,
        request: &PickRequest<'_>,
        candidate: &Candidate<'_>,
        reason: &str,
        audit: GateAudit,
    ) {
        let saturated = reason == "saturation_pin";
        tracing::debug!(model = %request.model, selected = %candidate.engine.url, reason,
            input_tokens = request.input_tokens, matched_prefix_tokens = candidate.matched_prefix_tokens,
            uncached_tokens = candidate.uncached_tokens, "cache candidate winner");
        if saturated && !audit.fleet_all_queued && sampled(&SATURATION_PIN_LOGS) {
            tracing::info!(model = %request.model, worker = %candidate.engine.url,
                saturation_queue_floor = self.config.saturation_queue_floor,
                worker_queue_limit = self.config.worker_queue_limit,
                "fleet saturated, keeping affinity with a queueing prefix owner instead of diverting");
        }
        let decision = if saturated {
            CacheAwareDecision::AllQueued
        } else {
            CacheAwareDecision::CacheHit
        };
        self.metrics
            .record_cache_aware_decision(&request.model.0, decision);
    }
}

/// Saturation is asked before capacity: a queueing fleet whose owners are
/// also out of KV is still saturated. `admission_evaluated > 0` then means
/// capacity, not the gate, emptied the set.
fn fallback_decision(audit: GateAudit) -> CacheAwareDecision {
    if audit.queue_gate_rejected == 0 {
        CacheAwareDecision::CacheMiss
    } else if audit.fleet_all_queued {
        CacheAwareDecision::AllQueued
    } else if audit.admission_evaluated > 0 {
        CacheAwareDecision::CacheMiss
    } else {
        CacheAwareDecision::CacheWorkerQueued
    }
}

impl Policy for CacheAwarePolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, PickResult> {
        Box::pin(self.pick_async(engines, request))
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{native, worker, Request};
    use super::super::{CapacityAdmission, PendingPrefillAdmission};
    use super::*;
    use crate::policies::state::PrefixMatch;

    fn policy(admission: Admission, config: AffinityConfig) -> CacheAwarePolicy {
        CacheAwarePolicy::new(
            CacheSource::Local(KvEventIndex::new()),
            admission,
            config,
            MetricsRegistry::new(),
        )
    }

    fn lookup(matches: &[(&Arc<Worker>, u32)]) -> PrefixLookup {
        PrefixLookup {
            matches: matches
                .iter()
                .map(|(engine, blocks)| PrefixMatch {
                    address: engine.url.clone(),
                    matched_prefix_blocks: *blocks,
                })
                .collect(),
            query_blocks: 10,
        }
    }

    #[tokio::test]
    async fn pending_budget_uses_uncached_work_but_kv_capacity_uses_the_full_input() {
        let candidate = worker("candidate");
        let budgets = PendingPrefillAdmission::new([("candidate".to_string(), 30)].into());
        let policy = policy(
            Admission::before(super::super::AllOfAdmission(vec![
                Box::new(CapacityAdmission),
                Box::new(budgets),
            ])),
            AffinityConfig {
                cache_affinity_min_matched_tokens: None,
                ..AffinityConfig::default()
            },
        );
        // 80 of 100 tokens cached: E=20 fits the budget of 30 with 5 waiting.
        let request = Request::default()
            .input_tokens(100)
            .prefix(lookup(&[(&candidate, 8)]))
            .hit_required();
        let fits = request
            .clone()
            .snapshot([native(&candidate, 0, 5, 0, 1_000)]);
        assert_eq!(
            fits.pick(&policy, std::slice::from_ref(&candidate))
                .await
                .unwrap()
                .reason,
            "cache_candidate"
        );
        // KV must fit the whole input: 30 used + 100 > 100.
        let full = request.snapshot([native(&candidate, 0, 5, 30, 100)]);
        assert!(matches!(
            full.pick(&policy, std::slice::from_ref(&candidate)).await,
            Err(PickError::NoAdmissibleEngine(rejections)) if rejections.len() == 1
        ));
    }

    #[tokio::test]
    async fn queue_gate_diverts_off_a_queueing_owner_and_books_the_diversion() {
        let (owner, other) = (worker("owner"), worker("other"));
        let policy = policy(
            Admission::allow_all(),
            AffinityConfig {
                cache_affinity_min_matched_tokens: None,
                worker_queue_limit: Some(4),
                ..AffinityConfig::default()
            },
        );
        let fleet = [owner.clone(), other.clone()];
        let request = Request::default()
            .input_tokens(100)
            .prefix(lookup(&[(&owner, 9), (&other, 4)]))
            .snapshot([
                native(&owner, 1, 4, 10, 10_000),
                native(&other, 1, 0, 10, 10_000),
            ]);
        let pick = request.pick(&policy, &fleet).await.unwrap();
        assert!(pick.engine.id == other.id && pick.reason == "cache_candidate");

        // Both owners queueing with the whole fleet queueing: keep the prefix.
        let saturated = Request::default()
            .input_tokens(100)
            .prefix(lookup(&[(&owner, 9)]))
            .snapshot([
                native(&owner, 1, 4, 10, 10_000),
                native(&other, 1, 9, 10, 10_000),
            ]);
        assert_eq!(
            saturated.pick(&policy, &fleet).await.unwrap().engine.id,
            owner.id
        );
    }

    #[tokio::test]
    async fn a_miss_is_no_candidates_when_a_hit_is_required_and_falls_back_otherwise() {
        let policy = policy(Admission::allow_all(), AffinityConfig::default());
        let fleet = [worker("a"), worker("b")];
        let memo = PrefixMemo::resolved(Some(lookup(&[])));
        let miss = Request::default()
            .input_tokens(100)
            .memo(&memo)
            .hit_required();
        assert_eq!(
            miss.pick(&policy, &fleet).await.unwrap_err(),
            PickError::NoCandidates
        );
        let fallback = Request::default()
            .input_tokens(100)
            .memo(&memo)
            .pick(&policy, &fleet)
            .await
            .unwrap();
        assert_eq!(fallback.reason, "no_cache_candidate");
        assert!(policy.metrics.render().contains("cache_miss"));
    }
}
