// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DP-rank selection inside an already selected worker (`--dp-aware`).
//!
//! First match wins: a sticky/session key hashes to a fixed rank, so router
//! replicas agree without shared state; else the rank with the deepest cached
//! prefix; else [`DpRankPolicy`] picks: the fewest router in-flight requests,
//! the fewest engine-reported KV tokens, or the worker's ranks in turn.

use crate::config::DpRankPolicy;
use crate::workers::Worker;
use sha2::{Digest, Sha256};

/// `None` for a single-rank worker. `prefix_depths` is `(rank, cached blocks)`;
/// `rank_tokens` is `(rank, KV tokens in use)` from the engine's load feed, read
/// only under `least_tokens`, and ignored unless it covers every candidate so a
/// rank with no reading never looks idle.
pub fn select_dp_rank(
    worker: &Worker,
    affinity_key: Option<&str>,
    prefix_depths: &[(u32, usize)],
    rank_tokens: &[(u32, u64)],
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
    let tokens_of = |rank: u32| {
        rank_tokens
            .iter()
            .find(|&&(reported, _)| reported == rank)
            .map(|&(_, tokens)| tokens)
    };
    let by_tokens = policy == DpRankPolicy::LeastTokens
        && candidates.iter().all(|&rank| tokens_of(rank).is_some());
    // Random start so equally loaded ranks share traffic.
    let start = rand::random::<usize>() % candidates.len();
    candidates[start..]
        .iter()
        .chain(&candidates[..start])
        .copied()
        .min_by_key(|&rank| {
            let tokens = if by_tokens {
                tokens_of(rank).unwrap_or(0)
            } else {
                0
            };
            (tokens, worker.dp_rank_inflight(rank))
        })
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
            grpc_port: None,
            supports_inline_system: false,
        };
        Worker::with_cb_config(spec, None, profile)
    }

    const LEAST: DpRankPolicy = DpRankPolicy::LeastInFlight;
    const TOKENS: DpRankPolicy = DpRankPolicy::LeastTokens;

    #[test]
    fn affinity_then_prefix_then_load() {
        let w = worker(4);
        assert_eq!(select_dp_rank(&worker(1), Some("k"), &[], &[], LEAST), None);
        // Pinned: replicas and releases must agree on the mapping.
        assert_eq!(
            select_dp_rank(&w, Some("session-42"), &[(0, 9)], &[], LEAST),
            Some(1)
        );
        assert_eq!(
            select_dp_rank(&w, None, &[(1, 2), (3, 5), (9, 9)], &[], LEAST),
            Some(3)
        );
        let _busy = [0, 1, 3].map(|rank| w.dp_rank_guard(rank));
        assert_eq!(select_dp_rank(&w, None, &[], &[], LEAST), Some(2));
    }

    #[test]
    fn least_tokens_reads_engine_load_and_falls_back_to_in_flight() {
        let w = worker(4);
        let tokens = [(0, 50), (1, 10), (2, 10), (3, 90)];
        let _busy_1 = w.dp_rank_guard(1);
        // Fewest tokens; the router's in-flight count breaks the tie.
        assert_eq!(select_dp_rank(&w, None, &[], &tokens, TOKENS), Some(2));
        // A cached prefix still outranks load.
        assert_eq!(
            select_dp_rank(&w, None, &[(3, 4)], &tokens, TOKENS),
            Some(3)
        );
        // Tokens outrank in-flight: rank 0 is idle but holds more KV.
        let _busy_2 = [2, 2].map(|rank| w.dp_rank_guard(rank));
        assert_eq!(select_dp_rank(&w, None, &[], &tokens, TOKENS), Some(1));
        // No reading, or one that misses a candidate: least in flight (0 or 3).
        for partial in [&[][..], &[(1, 0)][..]] {
            let picked = select_dp_rank(&w, None, &[], partial, TOKENS).unwrap();
            assert!(picked == 0 || picked == 3, "{picked}");
        }
        // least_in_flight never reads the tokens.
        let picked = select_dp_rank(&w, None, &[], &tokens, LEAST).unwrap();
        assert!(picked == 0 || picked == 3, "{picked}");
    }
}
