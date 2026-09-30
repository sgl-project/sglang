// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Attention-DP rank selection for `--dp-aware`.
//!
//! A dp-attention engine is one URL but `dp_size` independent schedulers, each
//! with its own KV cache. Without `--dp-aware` the engine's DP controller picks
//! the rank, so a prefix the router routed for (it lives on rank 3) is only a
//! hit when the controller happens to pick rank 3 too. With it, the router
//! picks the rank and sends it as `X-Data-Parallel-Rank`, which the engine
//! honors ahead of its own load-balance method (`routed_dp_rank`).
//!
//! The rank is chosen AFTER the worker, inside it — the worker stays one
//! routing destination for admission, breakers, health and every per-URL
//! table. sgl-model-gateway's `--dp-aware` instead registers `url@rank` as
//! separate workers; that shape does not fit a router keyed by URL
//! everywhere, and it would re-pick the rank on every admission retry against
//! a request body built once.
//!
//! Order of preference, per request:
//!   1. a rank holding the request's prefix (from [`RankPrefix`], supplied by
//!      the cache-aware policy) whose queue is under the per-rank limit —
//!      deepest holding first, then the better tier, then the lighter load;
//!   2. otherwise the least-loaded rank.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::policies::engine_load::RankLoad;
use crate::policies::kv_events::RankHold;
use crate::workers::Worker;

/// Which ranks of ONE worker hold the request's prefix, as the cache-aware
/// policy saw it. Only holdings that cleared the policy's `cache_threshold`
/// are listed: a below-threshold holding must not beat a lighter rank, for
/// the same reason it does not win the worker selection.
#[derive(Debug, Clone, Default)]
pub struct RankPrefix {
    pub holds: HashMap<u32, RankHold>,
}

/// Why a rank was picked. Label values of `sgl_router_dp_rank_selections_total`
/// (`prefix_owner` / `owners_queued` / `min_load` — deliberately not the
/// worker-level `cache_hit` vocabulary, which this is one level below).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankDecision {
    /// A rank holding the prefix, under the per-rank queue limit.
    PrefixOwner,
    /// Some rank holds the prefix but every holder is at the queue limit, so
    /// the least-loaded rank took it.
    OwnersQueued,
    /// No rank holds the prefix (or the policy is not cache-aware): least-loaded.
    MinLoad,
}

impl RankDecision {
    pub const ALL: [RankDecision; 3] = [Self::PrefixOwner, Self::OwnersQueued, Self::MinLoad];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::PrefixOwner => "prefix_owner",
            Self::OwnersQueued => "owners_queued",
            Self::MinLoad => "min_load",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::PrefixOwner => 0,
            Self::OwnersQueued => 1,
            Self::MinLoad => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankPick {
    pub rank: u32,
    pub decision: RankDecision,
}

/// Window over which router-side dispatches stand in for a rank's load when
/// the engine gauges are not all fresh. Short on purpose: it approximates
/// "in flight", and a request dispatched long ago has most likely finished.
const FALLBACK_LOAD_WINDOW: Duration = Duration::from_secs(5);

/// Per-rank load of one worker for this pick.
///
/// With every rank's engine gauge fresh: the gauge (`running + waiting`) plus
/// this router's dispatches to that rank since the gauge — the per-rank twin
/// of `WorkerLoads::load_of`. Otherwise the gauges are not comparable across
/// ranks (a stale rank would read as idle), so every rank falls back to this
/// router's dispatches over [`FALLBACK_LOAD_WINDOW`], which at least spreads
/// traffic evenly.
struct RankLoads {
    load: Vec<usize>,
    /// Fresh engine queue, or `None` when unknown (gates fail open on it).
    waiting: Vec<Option<usize>>,
}

impl RankLoads {
    fn build(worker: &Worker, dp: u32, gauges: &HashMap<u32, RankLoad>, now: Instant) -> Self {
        let all_fresh = (0..dp).all(|r| gauges.get(&r).is_some_and(|g| g.fresh));
        let since = now.checked_sub(FALLBACK_LOAD_WINDOW).unwrap_or(now);
        let mut load = Vec::with_capacity(dp as usize);
        let mut waiting = Vec::with_capacity(dp as usize);
        for r in 0..dp {
            match gauges.get(&r).filter(|_| all_fresh) {
                Some(g) => {
                    load.push(
                        g.depth
                            .depth()
                            .saturating_add(worker.rank_dispatches_since(r, g.at)),
                    );
                    waiting.push(Some(g.depth.waiting()));
                }
                None => {
                    load.push(worker.rank_dispatches_since(r, since));
                    waiting.push(
                        gauges
                            .get(&r)
                            .filter(|g| g.fresh)
                            .map(|g| g.depth.waiting()),
                    );
                }
            }
        }
        Self { load, waiting }
    }

    fn admits(&self, rank: u32, limit: Option<usize>) -> bool {
        match (limit, self.waiting[rank as usize]) {
            (Some(limit), Some(q)) => q < limit,
            _ => true,
        }
    }

    /// Least-loaded rank; ties broken uniformly at random so equal ranks share
    /// traffic instead of rank 0 taking every tie.
    fn min_load(&self) -> u32 {
        let min = *self.load.iter().min().expect("dp >= 2");
        let ties: Vec<u32> = (0..self.load.len() as u32)
            .filter(|&r| self.load[r as usize] == min)
            .collect();
        if ties.len() == 1 {
            return ties[0];
        }
        use rand::seq::SliceRandom;
        *ties.choose(&mut rand::thread_rng()).expect("non-empty")
    }
}

/// Pick the attention-DP rank of `worker` for one dispatch, or `None` when the
/// worker is not multi-rank (`dp_size <= 1`, or not introspected yet) — the
/// request then goes without a rank and the engine picks, as before.
pub fn pick_rank(
    worker: &Worker,
    prefix: Option<&RankPrefix>,
    gauges: &HashMap<u32, RankLoad>,
    rank_queue_limit: Option<usize>,
    now: Instant,
) -> Option<RankPick> {
    let dp = worker.dp_size();
    if dp <= 1 {
        return None;
    }
    let loads = RankLoads::build(worker, dp, gauges, now);

    let owners: Vec<(u32, RankHold)> = prefix
        .map(|p| {
            p.holds
                .iter()
                .filter(|(&r, _)| r < dp)
                .map(|(&r, &h)| (r, h))
                .collect()
        })
        .unwrap_or_default();
    let best_owner = owners
        .iter()
        .filter(|(r, _)| loads.admits(*r, rank_queue_limit))
        // Deepest holding, then best tier (lower slot is better), then lighter.
        .min_by_key(|(r, h)| {
            (
                std::cmp::Reverse(h.depth),
                h.tiers.best_slot().unwrap_or(usize::MAX),
                loads.load[*r as usize],
            )
        });
    if let Some(&(rank, _)) = best_owner {
        return Some(RankPick {
            rank,
            decision: RankDecision::PrefixOwner,
        });
    }
    Some(RankPick {
        rank: loads.min_load(),
        decision: if owners.is_empty() {
            RankDecision::MinLoad
        } else {
            RankDecision::OwnersQueued
        },
    })
}

/// The per-rank queue limit in effect: `--dp-rank-queue-limit`, else the
/// worker-level limit split across the worker's ranks (rounded up, so a limit
/// never rounds down to "always queued").
pub fn effective_rank_queue_limit(
    explicit: Option<usize>,
    worker_queue_limit: Option<usize>,
    dp_size: u32,
) -> Option<usize> {
    explicit.or_else(|| worker_queue_limit.map(|l| l.div_ceil(dp_size.max(1) as usize).max(1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::engine_load::WorkerDepth;
    use crate::policies::kv_events::Tiers;

    fn worker(dp: u32) -> Worker {
        let w = Worker::new(WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://w:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
            transfer_group: None,
        });
        w.set_dp_size(dp);
        w
    }

    fn gauge(running: usize, waiting: usize, at: Instant) -> RankLoad {
        RankLoad {
            depth: WorkerDepth::new(running, waiting),
            at,
            fresh: true,
        }
    }

    fn fresh_gauges(loads: &[(usize, usize)], at: Instant) -> HashMap<u32, RankLoad> {
        loads
            .iter()
            .enumerate()
            .map(|(r, &(run, wait))| (r as u32, gauge(run, wait, at)))
            .collect()
    }

    fn hold(depth: usize, tiers: Tiers) -> RankHold {
        RankHold { depth, tiers }
    }

    #[test]
    fn single_rank_worker_gets_no_rank() {
        let now = Instant::now();
        assert_eq!(
            pick_rank(&worker(1), None, &HashMap::new(), None, now),
            None
        );
        assert_eq!(
            pick_rank(&worker(0), None, &HashMap::new(), None, now),
            None,
            "dp_size not introspected yet ⇒ the engine picks",
        );
    }

    #[test]
    fn prefix_owner_beats_a_lighter_rank() {
        let now = Instant::now();
        let w = worker(4);
        let gauges = fresh_gauges(&[(0, 0), (0, 0), (9, 0), (0, 0)], now);
        let prefix = RankPrefix {
            holds: HashMap::from([(2, hold(10, Tiers::DEVICE))]),
        };
        let pick = pick_rank(&w, Some(&prefix), &gauges, None, now).unwrap();
        assert_eq!(pick.rank, 2);
        assert_eq!(pick.decision, RankDecision::PrefixOwner);
    }

    #[test]
    fn deeper_holding_wins_then_tier_then_load() {
        let now = Instant::now();
        let w = worker(4);
        let gauges = fresh_gauges(&[(5, 0), (1, 0), (0, 0), (0, 0)], now);
        let prefix = RankPrefix {
            holds: HashMap::from([
                (0, hold(8, Tiers::DEVICE)),
                (1, hold(8, Tiers::DEVICE)),
                (2, hold(8, Tiers::HOST)),
                (3, hold(3, Tiers::DEVICE)),
            ]),
        };
        // Depth 8 beats depth 3; among depth 8, device beats host; among the
        // two device holders, rank 1 is lighter.
        assert_eq!(
            pick_rank(&w, Some(&prefix), &gauges, None, now)
                .unwrap()
                .rank,
            1
        );
    }

    #[test]
    fn queued_owner_is_skipped_for_another_owner_then_min_load() {
        let now = Instant::now();
        let w = worker(3);
        // Rank 0 holds the deepest prefix but has 4 queued; rank 1 holds a
        // shallower one and is under the limit.
        let gauges = fresh_gauges(&[(4, 4), (2, 0), (0, 0)], now);
        let prefix = RankPrefix {
            holds: HashMap::from([(0, hold(10, Tiers::DEVICE)), (1, hold(5, Tiers::DEVICE))]),
        };
        let pick = pick_rank(&w, Some(&prefix), &gauges, Some(4), now).unwrap();
        assert_eq!((pick.rank, pick.decision), (1, RankDecision::PrefixOwner));

        // Only the queued owner holds it ⇒ least-loaded rank, labelled as such.
        let prefix = RankPrefix {
            holds: HashMap::from([(0, hold(10, Tiers::DEVICE))]),
        };
        let pick = pick_rank(&w, Some(&prefix), &gauges, Some(4), now).unwrap();
        assert_eq!((pick.rank, pick.decision), (2, RankDecision::OwnersQueued));
    }

    #[test]
    fn no_prefix_goes_to_the_least_loaded_rank() {
        let now = Instant::now();
        let w = worker(3);
        let gauges = fresh_gauges(&[(3, 0), (1, 0), (2, 0)], now);
        let pick = pick_rank(&w, None, &gauges, None, now).unwrap();
        assert_eq!((pick.rank, pick.decision), (1, RankDecision::MinLoad));
    }

    #[test]
    fn dispatches_since_the_gauge_count_toward_the_rank() {
        let at = Instant::now();
        let w = worker(2);
        let gauges = fresh_gauges(&[(1, 0), (2, 0)], at);
        // Two requests went to rank 0 after the gauge: 1 + 2 = 3 > 2.
        let now = at + Duration::from_millis(10);
        w.record_rank_dispatch(0, now);
        w.record_rank_dispatch(0, now);
        assert_eq!(pick_rank(&w, None, &gauges, None, now).unwrap().rank, 1);
    }

    #[test]
    fn stale_gauges_fall_back_to_router_dispatch_counts() {
        let now = Instant::now();
        let w = worker(2);
        // Rank 1's gauge is stale and reads idle; it must not attract
        // everything. Rank 1 got two recent dispatches, rank 0 none.
        let mut gauges = fresh_gauges(&[(9, 0), (0, 0)], now);
        gauges.get_mut(&1).unwrap().fresh = false;
        w.record_rank_dispatch(1, now);
        w.record_rank_dispatch(1, now);
        assert_eq!(pick_rank(&w, None, &gauges, None, now).unwrap().rank, 0);
    }

    #[test]
    fn owner_ranks_outside_dp_size_are_ignored() {
        let now = Instant::now();
        let w = worker(2);
        let gauges = fresh_gauges(&[(0, 0), (1, 0)], now);
        let prefix = RankPrefix {
            holds: HashMap::from([(7, hold(10, Tiers::DEVICE))]),
        };
        let pick = pick_rank(&w, Some(&prefix), &gauges, None, now).unwrap();
        assert_eq!((pick.rank, pick.decision), (0, RankDecision::MinLoad));
    }

    #[test]
    fn rank_queue_limit_defaults_from_the_worker_limit() {
        assert_eq!(effective_rank_queue_limit(Some(3), Some(64), 8), Some(3));
        assert_eq!(effective_rank_queue_limit(None, Some(32), 8), Some(4));
        assert_eq!(effective_rank_queue_limit(None, Some(4), 8), Some(1));
        assert_eq!(effective_rank_queue_limit(None, None, 8), None);
    }
}
