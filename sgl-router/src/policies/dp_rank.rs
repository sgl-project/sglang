// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DP-rank selection inside an already selected worker (`--dp-aware`).
//!
//! First match wins: a sticky/session key hashes to a fixed rank, so router
//! replicas agree without shared state; else the rank with the deepest cached
//! prefix; else [`DpRankPolicy`] picks: the fewest router in-flight requests,
//! the fewest engine-reported KV tokens (in use or queued), or the worker's ranks
//! in turn.

use crate::config::DpRankPolicy;
use crate::state::load_monitor::engine_reported_load::RankLoad;
use crate::workers::Worker;
use sha2::{Digest, Sha256};

/// `rank_tokens` for [`select_dp_rank`]: each reading moved forward by the
/// stamped requests this router sent to the rank since it, at the worker's
/// mean KV per request, so a burst between two engine reports does not all
/// land on the rank that was lightest at the last one.
pub fn projected_rank_tokens(worker: &Worker, loads: &[RankLoad]) -> Vec<(u32, u64)> {
    let (tokens, reqs) = loads.iter().fold((0u64, 0u64), |(tokens, reqs), load| {
        (
            tokens.saturating_add(load.num_tokens),
            reqs.saturating_add(load.num_reqs),
        )
    });
    let per_req = tokens.checked_div(reqs).unwrap_or(0);
    loads
        .iter()
        .map(|load| {
            let sent = worker.dp_rank_dispatched_since(load.rank, load.at) as u64;
            (
                load.rank,
                load.num_tokens.saturating_add(sent.saturating_mul(per_req)),
            )
        })
        .collect()
}

/// `None` for a single-rank worker. `prefix_depths` is `(rank, cached blocks)`;
/// `rank_tokens` is `(rank, committed KV tokens)` from the engine's load feed, read
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
    use std::time::{Duration, Instant};

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

    #[test]
    fn projection_counts_stamped_dispatches_since_each_reading() {
        let w = worker(2);
        let reading = |rank, num_tokens, num_reqs, at| RankLoad {
            rank,
            num_tokens,
            num_reqs,
            at,
        };
        let at = Instant::now();
        // Mean KV per request over the worker: (100 + 400) / (1 + 4).
        let loads = [reading(0, 100, 1, at), reading(1, 400, 4, at)];
        assert_eq!(projected_rank_tokens(&w, &loads), [(0, 100), (1, 400)]);
        let _sent = [0, 0, 0, 0].map(|rank| w.dp_rank_timestamped_guard(rank));
        let _plain = w.dp_rank_guard(0);
        assert_eq!(projected_rank_tokens(&w, &loads), [(0, 500), (1, 400)]);
        // A newer reading already covers them.
        let newer = Instant::now() + Duration::from_secs(1);
        let loads = [reading(0, 100, 1, newer), reading(1, 400, 4, newer)];
        assert_eq!(projected_rank_tokens(&w, &loads), [(0, 100), (1, 400)]);
        // No request on the worker: nothing to scale by.
        let loads = [reading(0, 0, 0, at), reading(1, 0, 0, at)];
        assert_eq!(projected_rank_tokens(&w, &loads), [(0, 0), (1, 0)]);
    }
}
