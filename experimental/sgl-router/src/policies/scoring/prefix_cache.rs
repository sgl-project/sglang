// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Continuous prefix-cache preference, from the KV-event [`HashTree`].
//!
//! Where `cache_aware_zmq` answers "is the deepest match good enough?" with a
//! threshold and a fleet-wide veto, this answers "how much of this prompt does
//! each worker already hold?" as a number in `[0, 1]` — which is what a
//! weighted sum needs. Usable standalone via `--policy prefix_cache`, or as a
//! fused term alongside load.

use super::{EligibilityFilter, ScoringPolicy};
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::policies::SelectionContext;
use crate::workers::Worker;
use std::sync::Arc;

/// Score for a worker that holds none of the prompt, and for every worker when
/// there is no signal at all (no tokens, no block size, no blocks).
///
/// Deliberately the same value: a uniform score vector is a no-op in a fused
/// sum, so "we cannot tell" cedes the decision to the other terms rather than
/// nudging it. It is 0.0 and not 1.0 because a neutral 1.0 once let a prefix
/// test pass without ever computing a depth.
const NO_HOLDING: f32 = 0.0;

/// Default fused-term weight, for a factory that has no `--fuse name=W`
/// override to apply. Deliberately the trait's own 1.0 and NOT a tuned ratio
/// against load: standalone it cannot matter (argmax is scale-invariant under
/// a positive weight), and fused, `=W` is the knob. A constant that looks
/// tuned but was never measured is the worse of the two failures.
pub const DEFAULT_WEIGHT: f32 = 1.0;

pub struct PrefixCachePolicy {
    /// Per-process KV-event tree, fed by the indexer. Read-only from here.
    tree: Arc<HashTree>,
    /// Worker-published block size. Without it the router cannot hash a prompt
    /// the way the workers did, so every score would be a false miss.
    block_size_oracle: Arc<BlockSizeOracle>,
    weight: f32,
    /// Share of the prompt a worker must already hold to stay ELIGIBLE. `0.0`
    /// is off, and off is the default: the term is then a pure preference and
    /// can never reject anyone. Above zero it turns cache affinity into a
    /// constraint, which is the thing a weight cannot buy — no fixed weight
    /// out-ranks load at every prompt length, because prompt length is a
    /// per-request quantity.
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

    /// Make affinity a constraint above `share` of the prompt: anyone holding
    /// less is rejected outright rather than merely out-scored. See
    /// [`Self::min_share`]; `0.0` restores the pure-preference behaviour.
    pub fn with_min_share(mut self, share: f32) -> Self {
        self.min_share = share;
        self
    }
}

impl PrefixCachePolicy {
    /// The share of the prompt each worker already holds, in `[0, 1]`.
    ///
    /// `NO_HOLDING` for every worker whenever there is no signal at all -- no
    /// ingress ids, no published block size, not one whole block. Both layers
    /// read this ONE walk, so the hard and soft halves cannot disagree and the
    /// tree is descended once per decision, not twice.
    fn shares(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        let flat = || vec![NO_HOLDING; workers.len()];

        // Only the ingress-computed ids. Those are the chat-templated tokens
        // the engine itself cached; re-deriving them from the raw joined
        // message content shifts every block boundary, so block 0 already
        // mismatches and the whole term goes silently inert. That was a real
        // bug, not a hypothetical.
        let Some(tokens) = ctx.request_tokens().filter(|t| !t.is_empty()) else {
            return flat();
        };
        let Some(block_size) = self.block_size_oracle.get() else {
            return flat();
        };
        // EAGLE-family workers hash over token bigrams; querying with the
        // wrong hashing scheme misses every block.
        let hashes = if self.block_size_oracle.is_bigram() {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        if hashes.is_empty() {
            return flat();
        }

        // ONE tree walk for the whole fleet. A per-candidate query here is the
        // regression to watch for: it turns a flat ~110µs into 262µs→4.6ms as
        // the fleet grows.
        let depths = self.tree.prefix_depths(None, &hashes);
        let total = hashes.len() as f32;
        workers
            .iter()
            .map(|w| {
                // `depths` is keyed {url, dp_rank}, so a worker with several
                // ranks appears once per rank: take its BEST. Taking the first
                // match read an arbitrary rank in HashMap order, which moved
                // the score run to run. O(workers x ranks), nil at fleet size.
                depths
                    .iter()
                    .filter(|(kw, _)| kw.url == w.url)
                    .map(|(_, &d)| d)
                    .max()
                    .map_or(NO_HOLDING, |d| d as f32 / total)
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

    /// The hard half, and only when a floor was actually configured. `None`
    /// below that is not cosmetic: it makes `--filter prefix_cache` without
    /// `--prefix-cache-min-share` fail at startup instead of installing a
    /// constraint that admits everybody.
    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        (self.min_share > 0.0).then_some(self as &dyn EligibilityFilter)
    }

    /// The whole point of this policy is the prompt, so ingress must tokenize.
    /// Saying `false` here would score every request on an empty prompt.
    fn needs_tokens(&self) -> bool {
        true
    }
}

impl EligibilityFilter for PrefixCachePolicy {
    /// Admit only the workers already holding at least `min_share` of the
    /// prompt -- the claim no weight can make, because out-ranking load by
    /// depth would need a weight that holds at every prompt length, and prompt
    /// length is a per-request quantity.
    ///
    /// No signal means ABSTAIN, not reject. "We cannot tell" is not "nobody is
    /// fit": vetoing here would make an unhashable prompt unroutable, and under
    /// a filter chain would cost the request every lower-priority constraint
    /// too.
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
    use crate::policies::kv_events::KvWorkerId;

    const BLOCK: usize = 4;

    fn worker(url: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }))
    }

    /// Four blocks' worth of tokens, so a depth is a clean quarter.
    fn tokens() -> Vec<u32> {
        (0..(BLOCK as u32 * 4)).collect()
    }

    /// Insert `blocks` of the prompt's hash chain for `url`'s `rank`, starting
    /// at `from` — so a worker can be given a TAIL it holds without block 0.
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

    /// The held shares, plus the claim that with no floor configured the term
    /// constrains nothing -- so every case below is reading the SOFT half.
    fn shares(p: &PrefixCachePolicy, ws: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        assert!(
            ScoringPolicy::as_filter(p).is_none(),
            "no floor configured, so this term must not be a filter at all",
        );
        p.scores(ws, ctx)
    }

    /// The two failure modes that matter together, in one call: a deep holder
    /// must score its real fraction (not a neutral 1.0), and a worker holding
    /// a TAIL without block 0 must score the miss value — `prefix_depths`
    /// freezes it at the first level that omits it rather than counting past
    /// the hole.
    #[test]
    fn depth_is_a_fraction_and_a_tail_without_block_zero_misses() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 3);
        insert(&tree, "tail", 0, 2, 2);
        let ws = vec![worker("deep"), worker("tail"), worker("cold")];

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let scores = shares(&policy(tree), &ws, &ctx);
        assert_eq!(scores[0], 0.75, "3 of 4 blocks held, not a neutral 1.0");
        // Literal 0.0, NOT the NO_HOLDING constant: asserting against the
        // constant the code uses makes the test move with the bug, and a
        // mutation to NO_HOLDING = 1.0 passed until this was a literal.
        assert_eq!(scores[1], 0.0, "tail without block 0 holds nothing");
        assert_eq!(scores[2], 0.0, "never seen");
    }

    /// A worker running data parallel publishes one KV stream PER dp rank, so
    /// the tree holds several entries for one url. The score must collapse
    /// them to the best rank; reading whichever the HashMap yielded first made
    /// this worker's score flip between 0.25 and 0.75 across processes.
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

    /// No ingress tokens is the common case for `/generate` on a model with no
    /// chat encoder. Every worker must come out equal, so the term is a no-op
    /// in a fused sum instead of inventing a preference.
    ///
    /// Scored twice over the SAME tree and fleet, because the uniform half
    /// alone cannot fail: `vec![0.0, 0.0]` is also what a `scores()` that is
    /// unconditionally flat returns, and that policy is dead code, not a no-op.
    /// The with-tokens half is what tells the two apart.
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

    /// No worker has published a block size yet, so the router cannot hash the
    /// prompt the way the workers did. Every score must be the miss value —
    /// dropping this guard does not fail loudly, it hashes at some default and
    /// returns confident WRONG depths, which is the worse outcome.
    #[test]
    fn without_a_block_size_no_worker_looks_like_a_hit() {
        let tree = Arc::new(HashTree::new());
        insert(&tree, "deep", 0, 0, 3);
        let ws = vec![worker("deep"), worker("cold")];

        let model = ModelId("tiny".into());
        let ids = tokens();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        // Same tree and same tokens that score [0.75, 0.0] above — the ONLY
        // difference is the unset oracle, so a pass here cannot be the tree
        // being empty.
        let cold = PrefixCachePolicy::new(tree, BlockSizeOracle::new(), 1.0);
        assert_eq!(shares(&cold, &ws, &ctx), vec![0.0, 0.0]);
    }

    /// EAGLE-family workers hash over token BIGRAMS. The two hashers must not
    /// be interchangeable: if the branch picked the wrong one the query would
    /// miss every block and the term would go silently inert at full score
    /// range — a term that always returns 0.0 is indistinguishable from
    /// "no cache" in a fused sum.
    #[test]
    fn the_bigram_branch_queries_a_different_chain() {
        let ids = tokens();
        let unigram = compute_block_hashes(&ids, BLOCK);
        let bigram = compute_block_hashes_bigram(&ids, BLOCK);
        assert_ne!(unigram, bigram, "the two hashers must disagree, else this");

        // Tree holds the UNIGRAM chain (what `insert` writes); the oracle says
        // bigram, so the policy queries the other chain and must miss.
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
        // The control: same tree, same tokens, unigram oracle -> full hit. It
        // is what proves the 0.0 above is the hashing scheme and not a broken
        // fixture.
        assert_eq!(shares(&policy(tree), &ws, &ctx), vec![1.0]);
    }
}
