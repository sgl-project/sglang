// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DP-rank selection inside an already selected worker (`--dp-aware`).
//!
//! An engine launched with `--dp-size` / `--attn-dp-size` runs several DP
//! ranks, each with its own scheduler and KV cache, behind one HTTP endpoint.
//! Worker policies pick the engine; this module then picks the rank inside it,
//! and the handler sends that rank as `X-Data-Parallel-Rank`, which the
//! engine's DP controller honors over its own load balancing.
//!
//! Tiers, first match wins:
//!
//! 1. **Affinity** — the request carries a sticky routing key or session id:
//!    a stable hash of it picks the rank, so a conversation keeps hitting the
//!    rank that holds its KV cache. The hash is fixed (SHA-256), so router
//!    replicas and restarts agree without shared state.
//! 2. **Prefix** — the router's local KV-event tree shows some rank of this
//!    worker holding the prompt's prefix: the deepest one wins.
//! 3. **Least load** — the rank with the fewest requests this router has in
//!    flight on it.
//!
//! Ties within a tier go to the least-loaded rank, rotating among equals.

use crate::workers::Worker;
use sha2::{Digest, Sha256};

/// Which tier chose the rank; the `reason` label on
/// `sgl_router_dp_rank_selections_total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpRankReason {
    Affinity,
    Prefix,
    LeastLoad,
}

impl DpRankReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Affinity => "affinity",
            Self::Prefix => "prefix",
            Self::LeastLoad => "least_load",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DpRankPick {
    pub rank: u32,
    pub reason: DpRankReason,
}

/// Pick a DP rank on `worker`, or `None` for a single-rank worker.
///
/// `prefix_depths` lists `(rank, cached prefix blocks)` for this worker's
/// ranks; ranks the worker does not have are ignored.
pub fn select_dp_rank(
    worker: &Worker,
    affinity_key: Option<&str>,
    prefix_depths: &[(u32, usize)],
) -> Option<DpRankPick> {
    let ranks = worker.dp_ranks();
    if ranks <= 1 {
        return None;
    }
    if let Some(key) = affinity_key {
        return Some(DpRankPick {
            rank: affinity_rank(key, ranks),
            reason: DpRankReason::Affinity,
        });
    }
    let deepest = prefix_depths
        .iter()
        .filter(|&&(rank, depth)| rank < ranks && depth > 0)
        .map(|&(_, depth)| depth)
        .max();
    if let Some(deepest) = deepest {
        let holders: Vec<u32> = prefix_depths
            .iter()
            .filter(|&&(rank, depth)| rank < ranks && depth == deepest)
            .map(|&(rank, _)| rank)
            .collect();
        return Some(DpRankPick {
            rank: least_loaded(worker, &holders),
            reason: DpRankReason::Prefix,
        });
    }
    let all: Vec<u32> = (0..ranks).collect();
    Some(DpRankPick {
        rank: least_loaded(worker, &all),
        reason: DpRankReason::LeastLoad,
    })
}

/// Stable rank for an affinity key: the first 8 bytes of its SHA-256,
/// reduced modulo the rank count.
fn affinity_rank(key: &str, ranks: u32) -> u32 {
    let digest = Sha256::digest(key.as_bytes());
    let prefix = u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 has 32 bytes"));
    (prefix % u64::from(ranks)) as u32
}

/// The candidate with the fewest router in-flight requests, starting the scan
/// at a rotating offset so equally loaded ranks share traffic.
fn least_loaded(worker: &Worker, candidates: &[u32]) -> u32 {
    let start = worker.next_dp_rank_offset() % candidates.len();
    candidates[start..]
        .iter()
        .chain(&candidates[..start])
        .copied()
        .min_by_key(|&rank| worker.dp_rank_inflight(rank))
        .expect("candidates is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::workers::{EngineProfile, WireProtocol};
    use std::collections::HashSet;

    fn worker(dp_ranks: u32) -> Worker {
        Worker::with_cb_config(
            WorkerSpec {
                id: WorkerId("w".into()),
                url: "http://w".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
            },
            None,
            EngineProfile {
                protocol: WireProtocol::default(),
                dp_ranks,
            },
        )
    }

    #[test]
    fn single_rank_worker_gets_no_rank() {
        assert_eq!(select_dp_rank(&worker(1), Some("key"), &[(0, 4)]), None);
    }

    #[test]
    fn affinity_key_maps_to_a_stable_rank() {
        let w = worker(8);
        let first = select_dp_rank(&w, Some("session-42"), &[]).unwrap();
        assert_eq!(first.reason, DpRankReason::Affinity);
        // Load and prefix state do not move an affinity-keyed request.
        let _busy = w.dp_rank_guard(first.rank);
        let again = select_dp_rank(&w, Some("session-42"), &[(7, 100)]).unwrap();
        assert_eq!(again, first);
        // Pinned value: replicas and restarts must agree on the mapping.
        assert_eq!(affinity_rank("session-42", 8), first.rank);
        assert_eq!(affinity_rank("session-42", 8), 1);
    }

    #[test]
    fn affinity_keys_spread_across_ranks() {
        let w = worker(4);
        let ranks: HashSet<u32> = (0..64)
            .map(|i| {
                select_dp_rank(&w, Some(&format!("k{i}")), &[])
                    .unwrap()
                    .rank
            })
            .collect();
        assert_eq!(ranks.len(), 4);
    }

    #[test]
    fn deepest_prefix_rank_wins() {
        let w = worker(4);
        let pick = select_dp_rank(&w, None, &[(0, 2), (2, 5), (3, 1)]).unwrap();
        assert_eq!(
            pick,
            DpRankPick {
                rank: 2,
                reason: DpRankReason::Prefix
            }
        );
    }

    #[test]
    fn prefix_tie_goes_to_least_loaded_holder() {
        let w = worker(4);
        let _busy = w.dp_rank_guard(1);
        for _ in 0..4 {
            let pick = select_dp_rank(&w, None, &[(1, 3), (3, 3)]).unwrap();
            assert_eq!(pick.rank, 3);
        }
    }

    #[test]
    fn ranks_outside_the_worker_and_empty_depths_are_ignored() {
        let w = worker(2);
        let pick = select_dp_rank(&w, None, &[(0, 0), (5, 9)]).unwrap();
        assert_eq!(pick.reason, DpRankReason::LeastLoad);
    }

    #[test]
    fn least_load_avoids_busy_ranks_and_rotates_among_idle_ones() {
        let w = worker(4);
        let _busy = [w.dp_rank_guard(0), w.dp_rank_guard(2)];
        let picks: HashSet<u32> = (0..4)
            .map(|_| select_dp_rank(&w, None, &[]).unwrap())
            .inspect(|pick| assert_eq!(pick.reason, DpRankReason::LeastLoad))
            .map(|pick| pick.rank)
            .collect();
        assert_eq!(picks, HashSet::from([1, 3]));
    }

    #[test]
    fn rank_guard_releases_on_drop() {
        let w = worker(2);
        let guard = w.dp_rank_guard(1);
        assert_eq!(w.dp_rank_inflight(1), 1);
        drop(guard);
        assert_eq!(w.dp_rank_inflight(1), 0);
    }
}
