// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DP-rank selection inside an already selected worker (`--dp-aware`).
//!
//! First match wins: a sticky/session key hashes to a fixed rank, so router
//! replicas agree without shared state; else the rank with the deepest cached
//! prefix; else [`DpRankPolicy`] picks: the fewest router in-flight requests,
//! or the worker's ranks in turn.

use crate::config::DpRankPolicy;
use crate::workers::Worker;
use sha2::{Digest, Sha256};

/// `None` for a single-rank worker. `prefix_depths` is `(rank, cached blocks)`.
pub fn select_dp_rank(
    worker: &Worker,
    affinity_key: Option<&str>,
    prefix_depths: &[(u32, usize)],
    policy: DpRankPolicy,
) -> Option<u32> {
    let ranks = worker.dp_ranks();
    if ranks <= 1 {
        return None;
    }
    if let Some(key) = affinity_key {
        let digest = Sha256::digest(key.as_bytes());
        let hash = u64::from_be_bytes(digest[..8].try_into().unwrap());
        return Some((hash % u64::from(ranks)) as u32);
    }
    let cached: Vec<_> = prefix_depths
        .iter()
        .filter(|&&(rank, depth)| rank < ranks && depth > 0)
        .collect();
    let candidates: Vec<u32> = match cached.iter().map(|&&(_, depth)| depth).max() {
        Some(deepest) => cached
            .iter()
            .filter(|&&&(_, depth)| depth == deepest)
            .map(|&&(rank, _)| rank)
            .collect(),
        None => (0..ranks).collect(),
    };
    if policy == DpRankPolicy::RoundRobin {
        return Some(candidates[worker.next_dp_rank_turn() % candidates.len()]);
    }
    // Random start so equally loaded ranks share traffic.
    let start = rand::random::<usize>() % candidates.len();
    candidates[start..]
        .iter()
        .chain(&candidates[..start])
        .copied()
        .min_by_key(|&rank| worker.dp_rank_inflight(rank))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::workers::{EngineProfile, WireProtocol};

    fn worker(dp_ranks: u32) -> Worker {
        let spec = WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://w".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
            version_group: None,
            services: Default::default(),
        };
        let profile = EngineProfile {
            protocol: WireProtocol::default(),
            dp_ranks,
            openai: None,
        };
        Worker::with_cb_config(spec, None, profile)
    }

    const LEAST: DpRankPolicy = DpRankPolicy::LeastInFlight;
    const RR: DpRankPolicy = DpRankPolicy::RoundRobin;

    #[test]
    fn affinity_then_prefix_then_load() {
        let w = worker(4);
        assert_eq!(select_dp_rank(&worker(1), Some("k"), &[], LEAST), None);
        // Pinned: replicas and releases must agree on the mapping.
        assert_eq!(
            select_dp_rank(&w, Some("session-42"), &[(0, 9)], LEAST),
            Some(1)
        );
        assert_eq!(
            select_dp_rank(&w, None, &[(1, 2), (3, 5), (9, 9)], LEAST),
            Some(3)
        );
        let _busy = [0, 1, 3].map(|rank| w.dp_rank_guard(rank));
        assert_eq!(select_dp_rank(&w, None, &[], LEAST), Some(2));
    }

    #[test]
    fn round_robin_takes_turns_regardless_of_load() {
        let w = worker(4);
        let _busy = w.dp_rank_guard(0);
        let turns: Vec<_> = (0..8).map(|_| select_dp_rank(&w, None, &[], RR)).collect();
        assert_eq!(turns, [0, 1, 2, 3, 0, 1, 2, 3].map(Some));
        // A key or the deepest prefix still decides before the turn does.
        assert_eq!(select_dp_rank(&w, Some("session-42"), &[], RR), Some(1));
        let tied = [(1, 5), (3, 5), (2, 1)];
        let turns: Vec<_> = (0..4)
            .map(|_| select_dp_rank(&w, None, &tied, RR))
            .collect();
        assert_eq!(turns, [1, 3, 1, 3].map(Some));
    }
}
