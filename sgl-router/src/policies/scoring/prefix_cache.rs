// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefix-cache scores from the KV-event [`HashTree`].

use super::{EligibilityFilter, ScoringPolicy};
use crate::policies::SelectionContext;
use crate::state::kv_events::{BlockSizeOracle, HashTree};
use crate::workers::Worker;
use std::collections::HashMap;
use std::sync::Arc;

/// Score for a miss or unavailable prefix signal.
const NO_HOLDING: f32 = 0.0;

/// Default fused-term weight.
pub const DEFAULT_WEIGHT: f32 = 1.0;

// Building an index costs more than scanning a small worker or depth set.
const URL_INDEX_MIN_ENTRIES: usize = 16;

pub struct PrefixCachePolicy {
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
    weight: f32,
    /// Minimum cached share for eligibility; zero disables filtering.
    min_share: f32,
}

impl std::fmt::Debug for PrefixCachePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixCachePolicy")
            .field("weight", &self.weight)
            .field("min_share", &self.min_share)
            .field("tree_nodes", &self.tree.node_count())
            .finish()
    }
}

impl PrefixCachePolicy {
    pub fn new(tree: Arc<HashTree>, block_size_oracle: Arc<BlockSizeOracle>, weight: f32) -> Self {
        Self {
            tree,
            block_size_oracle,
            weight,
            min_share: 0.0,
        }
    }

    /// Require a cached share for eligibility.
    pub fn with_min_share(mut self, share: f32) -> Self {
        self.min_share = share;
        self
    }
}

impl PrefixCachePolicy {
    /// Returns each worker's cached prompt share.
    fn shares(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        let flat = || vec![NO_HOLDING; workers.len()];

        let Some(tokens) = ctx.request_tokens().filter(|t| !t.is_empty()) else {
            return flat();
        };
        let Some(hashes) = self
            .block_size_oracle
            .block_hashes(tokens, ctx.cache_namespace())
            .filter(|hashes| !hashes.is_empty())
        else {
            return flat();
        };

        let depths = self.tree.prefix_depths(None, &hashes);
        let total = hashes.len() as f32;
        if depths.len() <= URL_INDEX_MIN_ENTRIES || workers.len() <= URL_INDEX_MIN_ENTRIES {
            return workers
                .iter()
                .map(|w| {
                    depths
                        .iter()
                        .filter(|(worker, _)| worker.url == w.url)
                        .map(|(_, &depth)| depth)
                        .max()
                        .map_or(NO_HOLDING, |depth| depth as f32 / total)
                })
                .collect();
        }

        let mut depth_by_url: HashMap<String, usize> = HashMap::with_capacity(depths.len());
        for (worker, depth) in depths {
            depth_by_url
                .entry(worker.url)
                .and_modify(|current| *current = (*current).max(depth))
                .or_insert(depth);
        }
        workers
            .iter()
            .map(|w| {
                depth_by_url
                    .get(w.url.as_str())
                    .map_or(NO_HOLDING, |&depth| depth as f32 / total)
            })
            .collect()
    }
}

impl ScoringPolicy for PrefixCachePolicy {
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        self.shares(workers, ctx)
    }

    fn weight(&self) -> f32 {
        self.weight
    }

    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        (self.min_share > 0.0).then_some(self as &dyn EligibilityFilter)
    }

    fn needs_tokens(&self) -> bool {
        true
    }
}

impl EligibilityFilter for PrefixCachePolicy {
    fn keep(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<bool> {
        (self.shares(workers, ctx).into_iter())
            .map(|share| share >= self.min_share)
            .collect()
    }

    fn needs_tokens(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::state::kv_events::{compute_block_hashes, compute_block_hashes_bigram, KvWorkerId};

    const BLOCK: usize = 4;

    fn worker(url: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            ..Default::default()
        }))
    }

    fn tokens() -> Vec<u32> {
        (0..(BLOCK as u32 * 4)).collect()
    }

    fn insert(tree: &HashTree, url: &str, rank: u32, from: usize, blocks: usize) {
        let all = compute_block_hashes(&tokens(), BLOCK);
        let parent = if from == 0 { None } else { Some(all[from - 1]) };
        tree.insert(
            &KvWorkerId::new(url.into(), rank),
            parent,
            &all[from..from + blocks],
        );
    }

    fn policy(tree: Arc<HashTree>) -> PrefixCachePolicy {
        let oracle = BlockSizeOracle::new();
        oracle
            .try_set(BLOCK as u32)
            .expect("a fresh oracle accepts the first block size");
        PrefixCachePolicy::new(tree, oracle, 1.0)
    }

    fn shares(p: &PrefixCachePolicy, ws: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        assert!(
            ScoringPolicy::as_filter(p).is_none(),
            "no floor configured, so this term must not be a filter at all",
        );
        p.scores(ws, ctx)
    }

    #[test]
    fn depth_is_a_fraction_and_a_tail_under_another_workers_block_counts() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 3);
        insert(&tree, "tail", 0, 2, 2);
        let ws = vec![worker("deep"), worker("tail"), worker("cold")];

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let scores = shares(&policy(tree), &ws, &ctx);
        assert_eq!(scores[0], 0.75, "3 of 4 blocks held, not a neutral 1.0");
        assert_eq!(
            scores[1], 1.0,
            "a chain stored under block 1 means its engine holds blocks 0 and 1"
        );
        assert_eq!(scores[2], 0.0, "never seen");
    }

    #[test]
    fn several_dp_ranks_of_one_worker_collapse_to_the_deepest() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "dp", 0, 0, 1);
        insert(&tree, "dp", 1, 0, 3);
        let ws = vec![worker("dp")];

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        assert_eq!(
            shares(&policy(tree), &ws, &ctx),
            vec![0.75],
            "3 of 4, not 1"
        );
    }

    #[test]
    fn indexed_scores_stay_aligned_with_worker_order_after_rank_collapse() {
        let tree = Arc::new(HashTree::new());
        let urls: Vec<_> = (0..20).map(|i| format!("worker-{i}")).collect();
        for (i, url) in urls.iter().enumerate() {
            insert(&tree, url, 0, 0, i % 4 + 1);
        }
        insert(&tree, &urls[18], 1, 0, 4);
        let ws: Vec<_> = urls.iter().rev().map(|url| worker(url)).collect();

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let expected: Vec<_> = (0..20)
            .rev()
            .map(|i| {
                if i == 18 {
                    1.0
                } else {
                    (i % 4 + 1) as f32 / 4.0
                }
            })
            .collect();
        assert_eq!(shares(&policy(tree), &ws, &ctx), expected);
    }

    #[test]
    fn without_tokens_every_worker_scores_the_same() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 3);
        let ws = vec![worker("deep"), worker("cold")];
        let model = ModelId("tiny".into());
        let policy = policy(tree);

        let ids = tokens();
        let with = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        assert_eq!(
            shares(&policy, &ws, &with),
            vec![0.75, 0.0],
            "signal is live"
        );

        let without = SelectionContext::new(&model, None);
        assert_eq!(
            shares(&policy, &ws, &without),
            vec![0.0, 0.0],
            "and inert here"
        );
    }

    #[test]
    fn without_a_block_size_no_worker_looks_like_a_hit() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 3);
        let ws = vec![worker("deep"), worker("cold")];

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let cold = PrefixCachePolicy::new(tree, BlockSizeOracle::new(), 1.0);
        assert_eq!(shares(&cold, &ws, &ctx), vec![0.0, 0.0]);
    }

    #[test]
    fn the_bigram_branch_queries_a_different_chain() {
        let ids = tokens();
        let unigram = compute_block_hashes(&ids, BLOCK);
        let bigram = compute_block_hashes_bigram(&ids, BLOCK);
        assert_ne!(unigram, bigram, "the two hashers must disagree, else this");

        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 4);
        let ws = vec![worker("deep")];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let oracle = BlockSizeOracle::new();
        oracle.try_set(BLOCK as u32).unwrap();
        oracle.set_bigram(true);
        let p = PrefixCachePolicy::new(Arc::clone(&tree), oracle, 1.0);
        assert_eq!(
            shares(&p, &ws, &ctx),
            vec![0.0],
            "bigram query, unigram tree"
        );
        assert_eq!(shares(&policy(tree), &ws, &ctx), vec![1.0]);
    }
}
