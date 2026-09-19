// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefer the engine holding the longest usable prefix, gated by queue depth
//! and admission and guarded against sending a near-tie to a much busier
//! engine. A miss falls back to a power-of-two choice.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::config::AffinityConfig;
use crate::policies::admission::FreshLoadLookup;
use crate::server::metrics::{CacheAwareDecision, MetricsRegistry};
use crate::state::engine_load::EngineLoadSnapshot;
use crate::state::prefix::{PrefixLookup, PrefixSource};
use crate::workers::Worker;

use super::admission::{Admission, Decision};
use super::{power_of_two, Pick, PickError, PickMode, PickRequest, Policy, Rejection};

#[derive(Debug)]
pub struct CacheAwarePolicy {
    pub source: Arc<PrefixSource>,
    pub admission: Admission,
    pub config: AffinityConfig,
    pub metrics: Arc<MetricsRegistry>,
}

#[derive(Clone, Copy)]
struct Candidate<'e> {
    engine: &'e Arc<Worker>,
    matched: u64,
    uncached: u64,
}

impl CacheAwarePolicy {
    async fn pick_async(
        &self,
        engines: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Result<Pick, PickError> {
        if engines.is_empty() {
            return Err(PickError::NoCandidates);
        }
        let owned;
        let lookup = match (request.token_ids, request.prefix) {
            (None, _) => None,
            (Some(tokens), Some(memo)) => memo
                .get_or_try_init(|| self.source.lookup(tokens))
                .await
                .map_err(|error| PickError::InvalidSignal(error.to_string()))?
                .as_ref(),
            (Some(tokens), None) => {
                owned = self
                    .source
                    .lookup(tokens)
                    .await
                    .map_err(|error| PickError::InvalidSignal(error.to_string()))?;
                owned.as_ref()
            }
        };
        let snapshot = request.load.snapshot();
        let candidates = self.candidates(engines, request.input_tokens, lookup, snapshot);
        let limit = self.config.worker_queue_limit;
        let unqueued = |engine: &Worker| {
            limit.is_none_or(|limit| {
                snapshot
                    .fresh_load_for_url(&engine.url)
                    .is_none_or(|load| load.num_waiting_reqs < limit)
            })
        };
        let (mut evaluated, gated): (Vec<&Candidate<'_>>, Vec<&Candidate<'_>>) =
            candidates.iter().partition(|c| unqueued(c.engine));
        // Nowhere unqueued to divert to: keep the prefix rather than trade it for the same wait.
        let fleet_all_queued = limit.is_some() && !engines.iter().any(|e| unqueued(e));
        if evaluated.is_empty() && fleet_all_queued && self.config.saturation_queue_floor.is_none()
        {
            evaluated = gated.clone();
        }
        let mut admitted = Vec::new();
        let mut rejections = Vec::new();
        self.metrics
            .record_cache_admission_evaluations(evaluated.len() as u64);
        for candidate in evaluated {
            match self.admission.check.check(candidate.engine, request)? {
                Decision::Allow => admitted.push(candidate),
                Decision::Reject(reason) => rejections.push(Rejection {
                    engine: candidate.engine.id.clone(),
                    reason,
                }),
            }
        }
        self.metrics
            .record_cache_admission_rejections(rejections.len() as u64);
        if let Some(winner) = self.tournament(&admitted, snapshot) {
            self.metrics
                .record_cache_aware_decision(&request.model.0, CacheAwareDecision::CacheHit);
            return Ok(Pick {
                engine: winner.engine.clone(),
                reason: "cache_candidate",
            });
        }
        if let Some(pinned) = self.saturation_pin(&gated, engines, request)? {
            return Ok(Pick {
                engine: pinned.engine.clone(),
                reason: "saturation_pin",
            });
        }
        if request.mode == PickMode::HitRequired {
            return Err(match rejections.is_empty() {
                true => PickError::NoCandidates,
                false => PickError::NoAdmissibleEngine(rejections),
            });
        }
        let decision = match (gated.is_empty(), fleet_all_queued) {
            (true, _) => CacheAwareDecision::CacheMiss,
            (false, true) => CacheAwareDecision::AllQueued,
            (false, false) => CacheAwareDecision::CacheWorkerQueued,
        };
        self.metrics
            .record_cache_aware_decision(&request.model.0, decision);
        let admitted = self.admission.admit(engines, request)?;
        let preferred: Vec<_> = admitted.iter().filter(|e| unqueued(e)).cloned().collect();
        let pool = if preferred.is_empty() {
            &admitted
        } else {
            &preferred
        };
        let engine = power_of_two::choose(pool, request)
            .ok_or(PickError::NoCandidates)?
            .clone();
        self.admission.verify(
            Pick {
                engine,
                reason: "no_cache_candidate",
            },
            request,
        )
    }

    /// Prefix holders among `engines` that pass the minimum-hit gate, best first, bounded.
    fn candidates<'e>(
        &self,
        engines: &'e [Arc<Worker>],
        input: u64,
        lookup: Option<&PrefixLookup>,
        snapshot: &EngineLoadSnapshot,
    ) -> Vec<Candidate<'e>> {
        let Some(lookup) = lookup.filter(|lookup| lookup.query_blocks > 0) else {
            return Vec::new();
        };
        let by_url: HashMap<&str, &'e Arc<Worker>> =
            engines.iter().map(|e| (e.url.as_str(), e)).collect();
        let mut candidates: Vec<Candidate<'e>> = lookup
            .matches
            .iter()
            .filter_map(|m| {
                let engine = *by_url.get(m.address.as_str())?;
                let blocks = u64::from(m.matched_prefix_blocks).min(lookup.query_blocks as u64);
                let matched = input.saturating_mul(blocks) / lookup.query_blocks as u64;
                (matched > 0 && self.passes_gate(input, matched)).then(|| Candidate {
                    engine,
                    matched,
                    uncached: input - matched,
                })
            })
            .collect();
        let loads = FreshLoadLookup::new(Some(snapshot), candidates.iter().map(|c| c.engine));
        candidates.sort_by(|l, r| {
            r.matched
                .cmp(&l.matched)
                .then_with(|| loads.compare_prefill_pressure(l.engine, r.engine))
                .then_with(|| l.engine.id.0.cmp(&r.engine.id.0))
        });
        candidates.dedup_by(|l, r| l.engine.id == r.engine.id);
        candidates.truncate(self.candidate_limit(engines.len()));
        candidates
    }

    fn passes_gate(&self, input: u64, matched: u64) -> bool {
        self.config
            .cache_affinity_min_matched_tokens
            .is_none_or(|minimum| matched >= minimum)
            && self
                .config
                .cache_affinity_min_match_ratio
                .is_none_or(|minimum| matched as f64 / input as f64 >= minimum)
    }

    fn candidate_limit(&self, pool: usize) -> usize {
        let ratio = self.config.cache_candidate_ratio.clamp(0.0, 1.0);
        let proportional = (ratio * pool as f64).ceil() as usize;
        pool.min(self.config.cache_candidate_max_workers)
            .min(self.config.cache_candidate_min_workers.max(proportional))
    }

    /// Among admitted holders within the switch margin of the least uncached
    /// work, the least pressured wins.
    fn tournament<'c, 'e>(
        &self,
        admitted: &[&'c Candidate<'e>],
        snapshot: &EngineLoadSnapshot,
    ) -> Option<&'c Candidate<'e>> {
        let loads = FreshLoadLookup::new(Some(snapshot), admitted.iter().map(|c| c.engine));
        let floor = *admitted.iter().min_by_key(|c| c.uncached)?;
        let ceiling = floor
            .uncached
            .saturating_add(self.config.cache_switch_margin_tokens);
        Some(
            admitted
                .iter()
                .copied()
                .filter(|c| c.uncached <= ceiling)
                .fold(floor, |winner, c| {
                    if self.compare(winner, c, &loads).is_gt() {
                        c
                    } else {
                        winner
                    }
                }),
        )
    }

    fn compare(
        &self,
        l: &Candidate<'_>,
        r: &Candidate<'_>,
        loads: &FreshLoadLookup<'_>,
    ) -> Ordering {
        if self.config.pressure_guard {
            if self.more_pressured(l.engine, r.engine, loads) {
                return Ordering::Greater;
            }
            if self.more_pressured(r.engine, l.engine, loads) {
                return Ordering::Less;
            }
        }
        l.uncached
            .cmp(&r.uncached)
            .then_with(|| loads.compare_prefill_pressure(l.engine, r.engine))
            .then_with(|| l.engine.id.0.cmp(&r.engine.id.0))
    }

    fn more_pressured(
        &self,
        candidate: &Worker,
        other: &Worker,
        loads: &FreshLoadLookup<'_>,
    ) -> bool {
        let (Some(c), Some(o)) = (loads.get(&candidate.id), loads.get(&other.id)) else {
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

    /// With a floor configured and no fleet queue provably below it, wait at the
    /// least pressured admitted prefix holder instead of cold-prefilling.
    fn saturation_pin<'c, 'e>(
        &self,
        gated: &[&'c Candidate<'e>],
        fleet: &[Arc<Worker>],
        request: &PickRequest<'_>,
    ) -> Result<Option<&'c Candidate<'e>>, PickError> {
        let Some(floor) = self.config.saturation_queue_floor else {
            return Ok(None);
        };
        let snapshot = request.load.snapshot();
        if gated.is_empty()
            || snapshot.any_fresh_queue_below(fleet.iter().map(|e| e.url.as_str()), floor)
        {
            return Ok(None);
        }
        let loads = FreshLoadLookup::new(Some(snapshot), gated.iter().map(|c| c.engine));
        let mut best: Option<&'c Candidate<'e>> = None;
        for candidate in gated {
            if self.admission.check.check(candidate.engine, request)? != Decision::Allow {
                continue;
            }
            if best.is_none_or(|b| {
                loads
                    .compare_prefill_pressure(candidate.engine, b.engine)
                    .then_with(|| candidate.engine.id.0.cmp(&b.engine.id.0))
                    .is_lt()
            }) {
                best = Some(candidate);
            }
        }
        Ok(best)
    }
}

impl Policy for CacheAwarePolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(self.pick_async(engines, request))
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::worker;
    use super::super::Stage;
    use super::*;
    use crate::discovery::ModelId;
    use crate::state::engine_load::EngineLoadTable;
    use crate::state::kv_events::BlockSizeOracle;
    use crate::state::LoadView;
    use sgl_kv_indexer::{PrefixIndex, PrefixIndexError, PrefixOutcome};

    #[derive(Debug)]
    struct Fake(Vec<(&'static str, u32)>);

    #[tonic::async_trait]
    impl PrefixIndex for Fake {
        async fn match_prefix(&self, _: Vec<i64>) -> Result<PrefixOutcome, PrefixIndexError> {
            if self.0.is_empty() {
                return Ok(PrefixOutcome::Empty);
            }
            let matches = self
                .0
                .iter()
                .map(|(address, blocks)| sgl_kv_indexer::PrefixMatch {
                    address: address.to_string(),
                    matched_prefix_blocks: *blocks,
                    worker_id: address.to_string(),
                })
                .collect();
            Ok(PrefixOutcome::Matched {
                matches,
                best_prefix_blocks: 4,
            })
        }
    }

    fn policy(matches: Vec<(&'static str, u32)>) -> CacheAwarePolicy {
        let block_size = BlockSizeOracle::new();
        block_size.try_set(16).unwrap();
        CacheAwarePolicy {
            source: Arc::new(PrefixSource::Remote {
                index: Arc::new(Fake(matches)),
                block_size,
            }),
            admission: Admission::default(),
            config: AffinityConfig {
                cache_affinity_min_matched_tokens: Some(16),
                ..AffinityConfig::default()
            },
            metrics: MetricsRegistry::new(),
        }
    }

    async fn pick(
        policy: &CacheAwarePolicy,
        engines: &[Arc<Worker>],
        mode: PickMode,
    ) -> Result<Pick, PickError> {
        let table = EngineLoadTable::new();
        let load = LoadView::new(&table);
        let model = ModelId("m".into());
        let tokens: Vec<u32> = (0..64).collect();
        let mut request = PickRequest::new(&model, Stage::Plain, 64, &load);
        request.token_ids = Some(&tokens);
        request.mode = mode;
        policy.pick(engines, &request).await
    }

    #[tokio::test]
    async fn longest_admitted_prefix_wins_and_a_miss_falls_back() {
        let fleet: Vec<_> = ["a", "b", "c"].map(worker).into();
        let hit = pick(
            &policy(vec![("http://a", 4), ("http://b", 1)]),
            &fleet,
            PickMode::Normal,
        )
        .await
        .unwrap();
        assert!(hit.engine.id.0 == "a" && hit.reason == "cache_candidate");
        let miss = policy(vec![]);
        assert_eq!(
            pick(&miss, &fleet, PickMode::Normal).await.unwrap().reason,
            "no_cache_candidate"
        );
        assert!(matches!(
            pick(&miss, &fleet, PickMode::HitRequired).await,
            Err(PickError::NoCandidates)
        ));
    }
}
