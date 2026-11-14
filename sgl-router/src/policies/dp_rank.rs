// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DP-rank selection inside an already selected worker (`--dp-aware`).
//!
//! First match wins: a sticky/session key hashes to a fixed rank, so router
//! replicas agree without shared state; else the rank with the deepest cached
//! prefix; else [`DpRankPolicy`] picks: the fewest router in-flight requests,
//! the worker's ranks in turn, or the rank whose heavier phase, prefill or
//! decode, stays lightest.

use crate::config::DpRankPolicy;
use crate::workers::{RankLoad, Worker};
use sha2::{Digest, Sha256};

/// Output tokens charged to a request that sets no cap; the engine's default
/// `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`.
pub const UNBOUNDED_OUTPUT_TOKENS: u64 = 4096;

/// What a request adds to a rank or worker, as the shape-aware policies score it.
/// `output_tokens` is `None` when the request sets no cap.
pub fn request_rank_load(input_tokens: usize, output_tokens: Option<u64>) -> RankLoad {
    RankLoad {
        input_tokens: input_tokens as u64,
        output_tokens: output_tokens
            .unwrap_or(UNBOUNDED_OUTPUT_TOKENS)
            .min(UNBOUNDED_OUTPUT_TOKENS),
    }
}

/// `None` for a single-rank worker. `prefix_depths` is `(rank, cached blocks)`;
/// `request` is the load the request would add.
pub fn select_dp_rank(
    worker: &Worker,
    affinity_key: Option<&str>,
    prefix_depths: &[(u32, usize)],
    policy: DpRankPolicy,
    request: RankLoad,
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
    let rotated = candidates[start..]
        .iter()
        .chain(&candidates[..start])
        .copied();
    if policy == DpRankPolicy::ShapeAware {
        return rotated.min_by_key(|&rank| {
            let load = worker.dp_rank_load(rank);
            let bottleneck = (load.input_tokens.saturating_add(request.input_tokens))
                .max(load.output_tokens.saturating_add(request.output_tokens));
            (bottleneck, worker.dp_rank_inflight(rank))
        });
    }
    rotated.min_by_key(|&rank| worker.dp_rank_inflight(rank))
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
        };
        Worker::with_cb_config(spec, None, profile)
    }

    const LEAST: DpRankPolicy = DpRankPolicy::LeastInFlight;
    const NONE: RankLoad = RankLoad {
        input_tokens: 0,
        output_tokens: 0,
    };

    #[test]
    fn affinity_then_prefix_then_load() {
        let w = worker(4);
        assert_eq!(
            select_dp_rank(&worker(1), Some("k"), &[], LEAST, NONE),
            None
        );
        // Pinned: replicas and releases must agree on the mapping.
        assert_eq!(
            select_dp_rank(&w, Some("session-42"), &[(0, 9)], LEAST, NONE),
            Some(1)
        );
        assert_eq!(
            select_dp_rank(&w, None, &[(1, 2), (3, 5), (9, 9)], LEAST, NONE),
            Some(3)
        );
        let _busy = [0, 1, 3].map(|rank| w.dp_rank_guard(rank, NONE));
        assert_eq!(select_dp_rank(&w, None, &[], LEAST, NONE), Some(2));
    }

    fn load(input_tokens: u64, output_tokens: u64) -> RankLoad {
        RankLoad {
            input_tokens,
            output_tokens,
        }
    }

    const SHAPE: DpRankPolicy = DpRankPolicy::ShapeAware;

    #[test]
    fn shape_aware_balances_the_post_admission_bottleneck() {
        let w = worker(3);
        let _held = [
            w.dp_rank_guard(0, load(100, 10)),
            w.dp_rank_guard(1, load(20, 90)),
            w.dp_rank_guard(2, load(60, 50)),
        ];
        // Rank 2's heavier phase stays at 80, against 120 and 110.
        assert_eq!(select_dp_rank(&w, None, &[], SHAPE, load(20, 20)), Some(2));
        // A decode-heavy request avoids the rank already deepest in decode.
        assert_eq!(select_dp_rank(&w, None, &[], SHAPE, load(1, 80)), Some(0));
    }

    #[test]
    fn shape_aware_ties_break_on_request_count() {
        let w = worker(2);
        let _held = [w.dp_rank_guard(0, NONE), w.dp_rank_guard(0, NONE)];
        assert_eq!(select_dp_rank(&w, None, &[], SHAPE, NONE), Some(1));
    }

    #[test]
    fn shape_aware_releases_load_when_the_guard_drops() {
        let w = worker(2);
        drop(w.dp_rank_guard(0, load(500, 500)));
        assert_eq!(w.dp_rank_load(0), NONE);
        assert_eq!(w.dp_rank_inflight(0), 0);
    }

    #[test]
    fn unset_output_cap_scores_as_the_ceiling() {
        assert_eq!(request_rank_load(7, None), load(7, UNBOUNDED_OUTPUT_TOKENS));
        assert_eq!(request_rank_load(7, Some(10)), load(7, 10));
        assert_eq!(
            request_rank_load(7, Some(131_000)),
            load(7, UNBOUNDED_OUTPUT_TOKENS)
        );
    }
}
