// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware selection within one engine group, following the legacy
//! `policies::cache_aware` proposal and `resolve_cache_candidates` rules.
//! Prefix I/O is memoized per request; candidate bounding, the queue gate and
//! admission run per pick against a fresh load snapshot.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::BoxFuture;
use sgl_kv_indexer::{PrefixIndex, PrefixIndexError, PrefixOutcome};
use tokio::sync::OnceCell;

use crate::config::AffinityConfig;
use crate::policies::admission::{fleet_is_all_queued, queue_gate_admits, FreshLoadLookup};
use crate::policies::prefix_provider::RadixTreePrefixProvider;
use crate::policies::ExternalPrefixSignal;
use crate::state::kv_events::{compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle};
use crate::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedLoadTable, EngineReportedSchedulingLoad,
};
use crate::workers::Worker;

use super::admission::{AdmissionLimits, Decision, EngineAdmission, EngineMetrics};
use super::power_of_two::PowerOfTwoPolicy;
use super::{Pick, PickError, PickRequest, Policy, Rejection, Stage};

type Signal = Option<Arc<ExternalPrefixSignal>>;
type Lookup = Arc<OnceCell<Signal>>;

/// Local radix tree or remote indexer. Groups sharing an index namespace share
/// one `Arc<CacheSource>`.
pub enum CacheSource {
    Local(RadixTreePrefixProvider),
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
    async fn lookup(&self, tokens: Option<&[u32]>) -> Result<Signal, PickError> {
        let Some(tokens) = tokens else {
            return Ok(None);
        };
        let (index, oracle) = match self {
            Self::Local(provider) => {
                return Ok(provider.match_request_tokens(tokens).map(Arc::new))
            }
            Self::Remote { index, block_size } => (index, block_size),
        };
        let Some(block_size) = oracle.get() else {
            return Ok(None);
        };
        let hashes = if oracle.is_bigram() {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        let query_blocks = hashes.len();
        if query_blocks == 0 {
            return Ok(None);
        }
        match index.match_prefix(hashes).await {
            Ok(outcome) => Ok(Some(Arc::new(ExternalPrefixSignal {
                outcome,
                query_blocks,
            }))),
            Err(PrefixIndexError::Rejected(code)) => Err(PickError::InvalidSignal(format!(
                "KV Indexer rejected the query: {code}"
            ))),
            Err(error) => {
                tracing::warn!(%error, "KV Indexer unavailable; using cache policy fallback");
                Ok(None)
            }
        }
    }
}

/// One per prepared request. Keyed by source identity so distinct index
/// namespaces never share an answer.
#[derive(Default)]
pub struct PrefixMemo {
    cells: Mutex<Vec<(Arc<CacheSource>, Lookup)>>,
}

impl fmt::Debug for PrefixMemo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PrefixMemo")
    }
}

impl PrefixMemo {
    fn cell(&self, source: &Arc<CacheSource>) -> Lookup {
        let mut cells = self.cells.lock().unwrap_or_else(|e| e.into_inner());
        match cells.iter().find(|(s, _)| Arc::ptr_eq(s, source)) {
            Some((_, cell)) => Arc::clone(cell),
            None => {
                let cell = Arc::default();
                cells.push((Arc::clone(source), Arc::clone(&cell)));
                cell
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate<'a> {
    engine: &'a Arc<Worker>,
    uncached_tokens: u64,
}

/// Less uncached work first, then lower prefill pressure, then worker id.
fn rank(loads: &FreshLoadLookup<'_>, left: &Candidate<'_>, right: &Candidate<'_>) -> Ordering {
    left.uncached_tokens
        .cmp(&right.uncached_tokens)
        .then_with(|| loads.compare_prefill_pressure(left.engine, right.engine))
        .then_with(|| left.engine.id.0.cmp(&right.engine.id.0))
}

#[derive(Debug)]
pub struct CacheAwarePolicy {
    source: Arc<CacheSource>,
    engine_load: Arc<EngineReportedLoadTable>,
    config: AffinityConfig,
    pub admission: Arc<dyn EngineAdmission>,
    /// Runs on a miss; this policy checks admission on its pick.
    pub fallback: Arc<dyn Policy>,
}

impl CacheAwarePolicy {
    pub fn new(
        source: Arc<CacheSource>,
        engine_load: Arc<EngineReportedLoadTable>,
        config: AffinityConfig,
    ) -> Result<Self, PickError> {
        let unit = |ratio: f64| ratio.is_finite() && (0.0..=1.0).contains(&ratio);
        let nonnegative = |ms: f64| ms.is_finite() && ms >= 0.0;
        let floor_fits = |floor| {
            config
                .worker_queue_limit
                .is_some_and(|limit| floor <= limit)
        };
        let valid = (1..=config.cache_candidate_max_workers)
            .contains(&config.cache_candidate_min_workers)
            && unit(config.cache_candidate_ratio)
            && config.cache_affinity_min_match_ratio.is_none_or(unit)
            && config.pressure_rel_threshold.is_finite()
            && config.pressure_rel_threshold > 1.0
            && config.pressure_abs_threshold_ms.is_none_or(nonnegative)
            && config.saturation_queue_floor.is_none_or(floor_fits);
        if !valid {
            return Err(PickError::InvalidConfiguration(
                "invalid cache candidate bounds, thresholds or saturation floor".into(),
            ));
        }
        Ok(Self {
            source,
            fallback: Arc::new(PowerOfTwoPolicy::new(Arc::clone(&engine_load))),
            engine_load,
            config,
            admission: Arc::new(AdmissionLimits::default()),
        })
    }

    /// Prefix holders in `engines` (matched by exact URL) that pass the hit
    /// thresholds, ranked and bounded to the configured candidate count.
    fn candidates<'e>(
        &self,
        engines: &'e [Arc<Worker>],
        request: &PickRequest<'_>,
        signal: Option<&ExternalPrefixSignal>,
        load: &EngineReportedLoadSnapshot,
    ) -> Vec<Candidate<'e>> {
        let Some(ExternalPrefixSignal {
            outcome: PrefixOutcome::Matched { matches, .. },
            query_blocks,
        }) = signal.filter(|signal| signal.query_blocks > 0)
        else {
            return Vec::new();
        };
        let query_blocks = *query_blocks as u64;
        let mut depths = HashMap::<&str, u64>::new();
        for entry in matches {
            let depth = depths.entry(entry.address.as_str()).or_default();
            *depth = (*depth).max(u64::from(entry.matched_prefix_blocks));
        }
        let config = &self.config;
        let input = request.input_tokens;
        let mut candidates: Vec<_> = engines
            .iter()
            .filter_map(|engine| {
                let blocks = depths.get(engine.url.as_str())?.min(&query_blocks);
                let matched = input.saturating_mul(*blocks) / query_blocks;
                let ratio = matched as f64 / input.max(1) as f64;
                let hit = *blocks > 0
                    && config
                        .cache_affinity_min_matched_tokens
                        .is_none_or(|min| matched >= min)
                    && config
                        .cache_affinity_min_match_ratio
                        .is_none_or(|min| ratio >= min);
                let uncached_tokens = input - matched;
                hit.then_some(Candidate {
                    engine,
                    uncached_tokens,
                })
            })
            .collect();
        let proportional = (config.cache_candidate_ratio * engines.len() as f64).ceil() as usize;
        let limit = engines
            .len()
            .min(config.cache_candidate_max_workers)
            .min(config.cache_candidate_min_workers.max(proportional));
        let loads = FreshLoadLookup::new(Some(load), candidates.iter().map(|c| c.engine));
        candidates.sort_by(|left, right| rank(&loads, left, right));
        candidates.truncate(limit);
        candidates
    }

    /// `None` when admitted.
    fn check(
        &self,
        engine: &Worker,
        load: &EngineReportedLoadSnapshot,
    ) -> Result<Option<Rejection>, PickError> {
        let metrics = EngineMetrics::observe(engine, load);
        Ok(match self.admission.check(engine, &metrics)? {
            Decision::Allow => None,
            Decision::Reject(reason) => Some(Rejection {
                engine: engine.id.clone(),
                reason,
            }),
        })
    }

    fn admit<'e>(
        &self,
        candidates: &[Candidate<'e>],
        load: &EngineReportedLoadSnapshot,
        rejections: &mut Vec<Rejection>,
    ) -> Result<Vec<Candidate<'e>>, PickError> {
        let mut admitted = Vec::new();
        for &candidate in candidates {
            match self.check(candidate.engine, load)? {
                None => admitted.push(candidate),
                Some(rejection) => rejections.push(rejection),
            }
        }
        Ok(admitted)
    }

    fn more_pressured(
        &self,
        left: &EngineReportedSchedulingLoad,
        right: &EngineReportedSchedulingLoad,
    ) -> bool {
        let config = &self.config;
        let queue_ms = |load: &EngineReportedSchedulingLoad| load.estimated_prefill_queue_ms;
        match (
            config.pressure_abs_threshold_ms,
            queue_ms(left),
            queue_ms(right),
        ) {
            (Some(abs), Some(left), Some(right)) => {
                left - right > abs && left > right * config.pressure_rel_threshold
            }
            _ => {
                let (left, right) = (
                    left.num_waiting_uncached_tokens,
                    right.num_waiting_uncached_tokens,
                );
                left.saturating_sub(right) > config.pressure_abs_threshold_tokens
                    && left as f64 > right as f64 * config.pressure_rel_threshold
            }
        }
    }

    /// Pressure guard between near-tied candidates; `None` defers to `rank`.
    fn guard(
        &self,
        left: &Worker,
        right: &Worker,
        load: &EngineReportedLoadSnapshot,
    ) -> Option<Ordering> {
        let left = load.fresh_native_cache_load_for_url(&left.url)?;
        let right = load.fresh_native_cache_load_for_url(&right.url)?;
        if self.more_pressured(left, right) {
            Some(Ordering::Greater)
        } else if self.more_pressured(right, left) {
            Some(Ordering::Less)
        } else {
            None
        }
    }

    /// Soft queue gate, saturation rules and hard admission over the bounded
    /// candidates. `Ok(None)` is a miss; a rejection never becomes a cold fallback.
    fn resolve(
        &self,
        candidates: &[Candidate<'_>],
        engines: &[Arc<Worker>],
        load: &EngineReportedLoadSnapshot,
    ) -> Result<Option<Pick>, PickError> {
        let limit = self.config.worker_queue_limit;
        let (mut evaluated, gated): (Vec<Candidate<'_>>, Vec<_>) = candidates
            .iter()
            .partition(|c| queue_gate_admits(load, c.engine, limit));
        // Diverting off an all-queued group buys nothing, so keep the prefix.
        // A configured floor replaces this tier with the pressure-ranked pin below.
        let saturated = evaluated.is_empty()
            && !gated.is_empty()
            && fleet_is_all_queued(load, engines, limit)
            && self.config.saturation_queue_floor.is_none();
        if saturated {
            evaluated.extend(&gated);
        }
        let mut rejections = Vec::new();
        let admitted = self.admit(&evaluated, load, &mut rejections)?;
        if let Some(&least) = admitted.iter().min_by_key(|c| c.uncached_tokens) {
            let loads = FreshLoadLookup::new(Some(load), evaluated.iter().map(|c| c.engine));
            let guarded = self.config.pressure_guard
                && evaluated.iter().all(|c| {
                    load.fresh_native_cache_load_for_url(&c.engine.url)
                        .is_some()
                });
            let ceiling = least
                .uncached_tokens
                .saturating_add(self.config.cache_switch_margin_tokens);
            let winner = admitted
                .iter()
                .filter(|c| c.uncached_tokens <= ceiling)
                .fold(least, |winner, &candidate| {
                    // The guard applies only to pairs within the margin of
                    // each other, not merely of the work floor.
                    let near_tie = winner.uncached_tokens.abs_diff(candidate.uncached_tokens)
                        <= self.config.cache_switch_margin_tokens;
                    let guard = (guarded && near_tie)
                        .then(|| self.guard(winner.engine, candidate.engine, load))
                        .flatten();
                    match guard.unwrap_or_else(|| rank(&loads, &winner, &candidate)) {
                        Ordering::Greater => candidate,
                        _ => winner,
                    }
                });
            return Ok(Some(Pick {
                engine: Arc::clone(winner.engine),
                reason: if saturated {
                    "saturation_pin"
                } else {
                    "cache_candidate"
                },
            }));
        }
        // Saturation pin: with no engine below the floor the request waits
        // anywhere, so wait at the least-pressured admitted prefix owner.
        let pinned = self.config.saturation_queue_floor.is_some_and(|floor| {
            !load.any_fresh_queue_below(engines.iter().map(|e| e.url.as_str()), floor)
        });
        if pinned {
            let loads = FreshLoadLookup::new(Some(load), gated.iter().map(|c| c.engine));
            let owner = self
                .admit(&gated, load, &mut rejections)?
                .into_iter()
                .min_by(|left, right| {
                    loads
                        .compare_prefill_pressure(left.engine, right.engine)
                        .then_with(|| left.engine.id.0.cmp(&right.engine.id.0))
                });
            if let Some(owner) = owner {
                return Ok(Some(Pick {
                    engine: Arc::clone(owner.engine),
                    reason: "saturation_pin",
                }));
            }
        }
        if rejections.is_empty() {
            Ok(None)
        } else {
            Err(PickError::NoAdmissibleEngine(rejections))
        }
    }
}

impl Policy for CacheAwarePolicy {
    fn supports(&self, stage: Stage) -> bool {
        stage != Stage::Decode
    }

    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            if engines.is_empty() {
                return Err(PickError::NoCandidates);
            }
            if request.stage == Stage::Decode {
                return Err(PickError::InvalidConfiguration(
                    "cache-aware selection requires a plain or prefill group".into(),
                ));
            }
            let lookup = || self.source.lookup(request.token_ids);
            let signal = match request.prefix {
                Some(memo) => memo
                    .cell(&self.source)
                    .get_or_try_init(lookup)
                    .await?
                    .clone(),
                None => lookup().await?,
            };
            // Capture load after remote I/O; selection and admission share it.
            let load = self.engine_load.capture_snapshot(Instant::now());
            let candidates = self.candidates(engines, request, signal.as_deref(), &load);
            if let Some(pick) = self.resolve(&candidates, engines, &load)? {
                return Ok(pick);
            }
            // Miss: fall back within the unqueued tier when one exists.
            let unqueued: Vec<_> = engines
                .iter()
                .filter(|e| queue_gate_admits(&load, e, self.config.worker_queue_limit))
                .cloned()
                .collect();
            let pool = if unqueued.is_empty() {
                engines
            } else {
                &unqueued
            };
            let mut pick = self.pick_fallback(pool, request).await?;
            if !pool.iter().any(|e| Arc::ptr_eq(e, &pick.engine)) {
                return Err(PickError::OutsideCandidates(pick.engine.id.clone()));
            }
            if let Some(rejection) = self.check(&pick.engine, &load)? {
                return Err(PickError::AdmissionRejected(rejection));
            }
            pick.reason = "no_cache_candidate";
            Ok(pick)
        })
    }

    fn fallback(&self) -> Option<&dyn Policy> {
        Some(self.fallback.as_ref())
    }
}
