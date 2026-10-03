// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefer admitted prefix owners, optionally balancing affinity against load.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::BoxFuture;
use sgl_kv_indexer::{PrefixIndex, PrefixIndexError, PrefixOutcome};
use tokio::sync::OnceCell;

use crate::config::{AffinityConfig, AffinityMode};
use crate::policies::admission::FreshLoadLookup;
use crate::policies::prefix_provider::RadixTreePrefixProvider;
use crate::policies::ExternalPrefixSignal;
use crate::state::kv_events::{compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle};
use crate::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedLoadTable,
};
use crate::workers::Worker;

use super::admission::{AdmissionLimits, Decision, EngineAdmission, EngineMetrics};
use super::affinity;
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
    /// This policy checks admission on the fallback's pick.
    pub fallback: Arc<dyn Policy>,
}

impl CacheAwarePolicy {
    pub fn new(
        source: Arc<CacheSource>,
        engine_load: Arc<EngineReportedLoadTable>,
        config: AffinityConfig,
    ) -> Result<Self, PickError> {
        let unit = |ratio: f64| ratio.is_finite() && (0.0..=1.0).contains(&ratio);
        let valid = (1..=config.cache_candidate_max_workers)
            .contains(&config.cache_candidate_min_workers)
            && unit(config.cache_candidate_ratio)
            && config.cache_affinity_min_match_ratio.is_none_or(unit)
            && config.load_factor.is_finite()
            && config.load_factor >= 1.0
            && config.worker_queue_limit.is_none()
            && config.saturation_queue_floor.is_none();
        if !valid {
            return Err(PickError::InvalidConfiguration(
                "invalid cache candidate bounds or affinity settings".into(),
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
}

impl Policy for CacheAwarePolicy {
    fn supports(&self, stage: Stage) -> bool {
        stage != Stage::Decode
    }

    fn needs_request_tokens(&self) -> bool {
        true
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
            let mut rejections = Vec::new();
            let admitted = self.admit(&candidates, &load, &mut rejections)?;
            let affinity = admitted.first().map(|c| Pick {
                engine: Arc::clone(c.engine),
                reason: "cache_candidate",
            });
            if self.config.mode != AffinityMode::Balanced {
                if let Some(pick) = affinity {
                    return Ok(pick);
                }
            }
            let pool: Vec<_> = engines
                .iter()
                .filter(|e| {
                    !rejections.iter().any(|r| r.engine == e.id)
                        && affinity.as_ref().is_none_or(|p| p.engine.id != e.id)
                })
                .cloned()
                .collect();
            if pool.is_empty() && affinity.is_none() && !rejections.is_empty() {
                return Err(PickError::NoAdmissibleEngine(rejections));
            }
            let fallback = async {
                let mut pick = self.pick_fallback(&pool, request).await?;
                if !pool.iter().any(|e| Arc::ptr_eq(e, &pick.engine)) {
                    return Err(PickError::OutsideCandidates(pick.engine.id.clone()));
                }
                if let Some(rejection) = self.check(&pick.engine, &load)? {
                    return Err(PickError::AdmissionRejected(rejection));
                }
                pick.reason = if affinity.is_some() {
                    "affinity_load"
                } else {
                    "no_cache_candidate"
                };
                Ok(pick)
            }
            .await;
            affinity::choose(&self.config, affinity, fallback, &load)
        })
    }

    fn fallback(&self) -> Option<&dyn Policy> {
        Some(self.fallback.as_ref())
    }
}
