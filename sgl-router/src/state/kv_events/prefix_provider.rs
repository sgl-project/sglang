// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{BlockSizeOracle, CacheNamespace, HashTree};
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Prefix lookup result from the local radix tree or remote KV indexer,
/// consumed by cache-aware routing.
pub struct PrefixLookupResult {
    pub outcome: PrefixOutcome,
    pub query_blocks: usize,
    /// Workers whose reported depth came from a pending stake rather than
    /// from KV events (empty when the confirmed tree decided the lookup).
    /// Routing to one of them is a "follow" of that stake.
    pub pending_winners: Vec<String>,
    /// The query's block hashes when the local tree answered, so routing can
    /// record the placement without rehashing.
    pub block_hashes: Option<Arc<[i64]>>,
}

#[derive(Clone, Debug)]
pub struct RadixTreePrefixProvider {
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
    /// Confirmed-first mode, on when set: a pending stake is consulted only
    /// when it is strictly deeper than the best confirmed KV-event match, and
    /// only while the confirmed match covers less than this share of the
    /// prompt. Above it the confirmed tree already routes the prompt and its
    /// successor turns, so the pending walk and the stake write are skipped:
    /// a warm request pays nothing for a tier it cannot benefit from. `None`
    /// keeps the original behavior: pending depths are merged with the
    /// confirmed ones by maximum on every request, and every request stakes.
    pending_consult_below: Option<f64>,
}

impl RadixTreePrefixProvider {
    pub fn new(tree: Arc<HashTree>, block_size_oracle: Arc<BlockSizeOracle>) -> Self {
        Self {
            tree,
            block_size_oracle,
            pending_consult_below: None,
        }
    }

    /// Switch on confirmed-first mode: consult and stake pending prefixes
    /// only when the confirmed match covers less than `ratio` of the prompt
    /// and only where a stake is strictly deeper (see `pending_consult_below`).
    pub fn with_pending_consult_below(mut self, ratio: f64) -> Self {
        self.pending_consult_below = Some(ratio);
        self
    }

    pub fn match_request_tokens(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
    ) -> Option<PrefixLookupResult> {
        let hashes = self.block_hashes(tokens, namespace)?;

        let mut depth_by_url = BTreeMap::<String, u32>::new();
        for (worker, depth) in self.tree.prefix_depths(None, &hashes) {
            merge_depth(&mut depth_by_url, worker.url, depth);
        }
        let confirmed = depth_by_url.values().copied().max();
        let confirmed_by_url = depth_by_url.clone();
        // Stakes at their follower cap are already filtered out by
        // `PendingPrefixes::depths`.
        let (consult_pending, merge_pending) = match self.pending_consult_below {
            // Original behavior: walk the pending store on every request and
            // merge its depths with the confirmed ones by maximum.
            None => (self.tree.pending().is_enabled(), true),
            // Confirmed-first: pay for the pending tier only when the confirmed
            // tree has not already placed this prompt (a first turn, or a turn
            // racing its predecessor's KV events), and let a stake count only
            // where it is strictly deeper than the best confirmed match, so a
            // real match that is at least as good always decides.
            Some(below) => {
                let covered = confirmed.unwrap_or(0) as f64 / hashes.len() as f64;
                (self.tree.pending().is_enabled() && covered < below, false)
            }
        };
        let pending = if consult_pending {
            self.tree.pending().depths(&hashes)
        } else {
            Vec::new()
        };
        // Confirmed-first merges a stake only where it is strictly deeper than
        // the best confirmed match, each on its own: a stake no deeper than
        // that match never becomes a candidate, even beside one that is.
        for (url, depth) in pending {
            let deeper = Some(u32::try_from(depth).unwrap_or(u32::MAX)) > confirmed;
            if merge_pending || deeper {
                merge_depth(&mut depth_by_url, url.to_string(), depth);
            }
        }
        let pending_winners: Vec<String> = depth_by_url
            .iter()
            .filter(|(url, depth)| confirmed_by_url.get(*url) < Some(depth))
            .map(|(url, _)| url.clone())
            .collect();
        let outcome = match depth_by_url.values().copied().max() {
            None => PrefixOutcome::Empty,
            Some(best_prefix_blocks) => {
                if confirmed < Some(best_prefix_blocks) {
                    self.tree.pending().record_hit();
                }
                let matches = depth_by_url
                    .into_iter()
                    .map(|(address, matched_prefix_blocks)| PrefixMatch {
                        worker_id: address.clone(),
                        address,
                        matched_prefix_blocks,
                    })
                    .collect();
                PrefixOutcome::Matched {
                    matches,
                    best_prefix_blocks,
                }
            }
        };
        // Empty is still returned so routing can record where this prompt went.
        Some(PrefixLookupResult {
            outcome,
            query_blocks: hashes.len(),
            pending_winners,
            block_hashes: consult_pending.then(|| hashes.into()),
        })
    }

    /// Remember that `signal`'s prompt was just routed to `url`. When the
    /// placement followed `url`'s pending stake, charge that stake a follower
    /// first, so the cap sees every request the stake attracted.
    pub fn record_route(&self, signal: &PrefixLookupResult, url: &str) {
        if let Some(hashes) = &signal.block_hashes {
            if signal.pending_winners.iter().any(|w| w == url) {
                self.tree.pending().record_follow(url, hashes);
            }
            self.tree.pending().record(url, hashes);
        }
    }

    /// `(dp_rank, cached prefix blocks)` for each rank of `worker_url`.
    pub fn rank_depths(
        &self,
        tokens: &[u32],
        namespace: &CacheNamespace,
        worker_url: &str,
    ) -> Vec<(u32, usize)> {
        let Some(hashes) = self.block_hashes(tokens, namespace) else {
            return Vec::new();
        };
        self.tree
            .prefix_depths(None, &hashes)
            .into_iter()
            .filter(|(worker, _)| worker.url == worker_url)
            .map(|(worker, depth)| (worker.dp_rank, depth))
            .collect()
    }

    fn block_hashes(&self, tokens: &[u32], namespace: &CacheNamespace) -> Option<Vec<i64>> {
        let hashes = self.block_size_oracle.block_hashes(tokens, namespace)?;
        (!hashes.is_empty()).then_some(hashes)
    }
}

fn merge_depth(depth_by_url: &mut BTreeMap<String, u32>, url: String, depth: usize) {
    let depth = u32::try_from(depth).unwrap_or(u32::MAX);
    depth_by_url
        .entry(url)
        .and_modify(|current| *current = (*current).max(depth))
        .or_insert(depth);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::kv_events::tree::KvWorkerId;
    use crate::state::kv_events::wire::BlockStored;
    use std::time::Duration;

    const BLOCK: u32 = 2;

    fn provider(ttl_ms: u64, max_followers: u64) -> RadixTreePrefixProvider {
        let tree = Arc::new(HashTree::new());
        let oracle = BlockSizeOracle::new();
        oracle.try_set(BLOCK).expect("block size");
        tree.pending().enable(Duration::from_millis(ttl_ms));
        tree.pending().set_max_followers(max_followers);
        RadixTreePrefixProvider::new(tree, oracle)
    }

    fn tokens(blocks: usize) -> Vec<u32> {
        (1..=(blocks * BLOCK as usize) as u32).collect()
    }

    /// Publish the first `blocks` blocks of `tokens` for `url` as KV events.
    fn confirm(provider: &RadixTreePrefixProvider, url: &str, tokens: &[u32], blocks: usize) {
        let hashes = provider
            .block_hashes(tokens, &CacheNamespace::default())
            .expect("hashes");
        let event = BlockStored {
            block_hashes: hashes[..blocks].to_vec(),
            parent_block_hash: None,
            token_ids: tokens[..blocks * BLOCK as usize].to_vec(),
            block_size: BLOCK,
            lora_id: None,
            lora_name: None,
            cache_salt: None,
            is_bigram: false,
            medium: None,
        };
        provider
            .tree
            .store_event(&KvWorkerId::new(url.to_owned(), 0), &event);
    }

    fn lookup(provider: &RadixTreePrefixProvider, tokens: &[u32]) -> PrefixLookupResult {
        provider
            .match_request_tokens(tokens, &CacheNamespace::default())
            .expect("lookup")
    }

    fn depths(signal: &PrefixLookupResult) -> Vec<(String, u32)> {
        match &signal.outcome {
            PrefixOutcome::Matched { matches, .. } => {
                let mut v: Vec<_> = matches
                    .iter()
                    .map(|m| (m.address.clone(), m.matched_prefix_blocks))
                    .collect();
                v.sort();
                v
            }
            _ => Vec::new(),
        }
    }

    #[test]
    fn confirmed_match_wins_over_a_shallower_or_equal_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 6);
        // A stake on b covering 4 blocks: shallower than a's confirmed 6.
        let probe = lookup(&provider, &prompt[..4 * BLOCK as usize]);
        provider.record_route(&probe, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 6)]);
        assert!(signal.pending_winners.is_empty());
        assert_eq!(provider.tree.pending().hits(), 0);
        // Equal depth on another worker: still the confirmed tree decides.
        let probe = lookup(&provider, &prompt[..6 * BLOCK as usize]);
        provider.record_route(&probe, "http://c");
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 6)]);
        assert!(signal.pending_winners.is_empty());
    }

    #[test]
    fn without_the_switch_pending_merges_by_maximum_as_before() {
        let provider = provider(60_000, 0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 6);
        // An equal-depth stake on another worker stays a candidate, as it did
        // before confirmed-first mode existed, and every request stakes.
        let probe = lookup(&provider, &prompt[..6 * BLOCK as usize]);
        assert!(probe.block_hashes.is_some());
        provider.record_route(&probe, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 6), ("http://b".into(), 6)]
        );
        assert!(signal.block_hashes.is_some(), "every request stakes");
        assert_eq!(
            provider.tree.pending().hits(),
            0,
            "not deeper than confirmed"
        );
    }

    #[test]
    fn deeper_stake_is_consulted_and_marked_as_pending_winner() {
        let provider = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 3);
        let signal = lookup(&provider, &prompt);
        provider.record_route(&signal, "http://b");
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 3), ("http://b".into(), 10)]
        );
        assert_eq!(signal.pending_winners, vec!["http://b".to_string()]);
        assert_eq!(provider.tree.pending().hits(), 1);
    }

    #[test]
    fn only_strictly_deeper_stakes_join_the_candidates() {
        let first = provider(60_000, 0).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&first, "http://a", &prompt, 2);
        // c stakes 3 blocks (deeper than a's 2), d stakes 2 (equal) and e 1.
        for (url, blocks) in [("http://c", 3), ("http://d", 2), ("http://e", 1)] {
            let probe = lookup(&first, &prompt[..blocks * BLOCK as usize]);
            first.record_route(&probe, url);
        }
        let signal = lookup(&first, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 2), ("http://c".into(), 3)],
            "the stakes no deeper than a's match are not candidates"
        );
        assert_eq!(signal.pending_winners, vec!["http://c".to_string()]);
        // Without the switch, every stake merges by maximum as before.
        let plain = provider(60_000, 0);
        confirm(&plain, "http://a", &prompt, 2);
        for (url, blocks) in [("http://c", 3), ("http://d", 2), ("http://e", 1)] {
            let probe = lookup(&plain, &prompt[..blocks * BLOCK as usize]);
            plain.record_route(&probe, url);
        }
        let signal = lookup(&plain, &prompt);
        assert_eq!(
            depths(&signal),
            [
                ("http://a".into(), 2),
                ("http://c".into(), 3),
                ("http://d".into(), 2),
                ("http://e".into(), 1)
            ]
        );
    }

    #[test]
    fn follower_cap_falls_back_to_the_confirmed_tree() {
        let provider = provider(60_000, 2).with_pending_consult_below(1.0);
        let prompt = tokens(10);
        confirm(&provider, "http://a", &prompt, 3);
        let first = lookup(&provider, &prompt);
        provider.record_route(&first, "http://b");
        // Two followers may ride the stake.
        for _ in 0..2 {
            let signal = lookup(&provider, &prompt);
            assert_eq!(signal.pending_winners, vec!["http://b".to_string()]);
            provider.record_route(&signal, "http://b");
        }
        assert_eq!(provider.tree.pending().followed(), 2);
        // The third sees only the confirmed tree: a at 3 blocks, b gone.
        let signal = lookup(&provider, &prompt);
        assert_eq!(depths(&signal), [("http://a".into(), 3)]);
        assert!(signal.pending_winners.is_empty());
        assert_eq!(provider.tree.pending().capped(), 1);
        // Once b's events arrive, b is a confirmed match again.
        confirm(&provider, "http://b", &prompt, 10);
        let signal = lookup(&provider, &prompt);
        assert_eq!(
            depths(&signal),
            [("http://a".into(), 3), ("http://b".into(), 10)]
        );
        assert!(signal.pending_winners.is_empty());
    }

    #[test]
    fn warm_prompt_skips_the_pending_tier_and_its_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(0.5);
        let prompt = tokens(10);
        // A cold prompt (nothing confirmed) consults the tier and is staked.
        let cold = lookup(&provider, &prompt);
        assert!(cold.block_hashes.is_some(), "a cold prompt is staked");
        provider.record_route(&cold, "http://b");
        // Once a confirmed match covers at least half the prompt, the deeper
        // stake on b is not even consulted, and the prompt writes no stake.
        confirm(&provider, "http://a", &prompt, 6);
        let warm = lookup(&provider, &prompt);
        assert_eq!(depths(&warm), [("http://a".into(), 6)]);
        assert!(warm.pending_winners.is_empty());
        assert!(warm.block_hashes.is_none(), "a warm prompt pays no stake");
        assert_eq!(
            provider.tree.pending().hits(),
            0,
            "the tier was never walked"
        );
    }

    #[test]
    fn cold_prompt_below_the_gate_still_uses_the_stake() {
        let provider = provider(60_000, 0).with_pending_consult_below(0.5);
        let prompt = tokens(10);
        // Only the shared head is confirmed: 2 of 10 blocks, below the gate.
        confirm(&provider, "http://a", &prompt, 2);
        let first = lookup(&provider, &prompt);
        assert!(first.block_hashes.is_some());
        provider.record_route(&first, "http://b");
        let next = lookup(&provider, &prompt);
        assert_eq!(next.pending_winners, vec!["http://b".to_string()]);
        assert_eq!(
            depths(&next),
            [("http://a".into(), 2), ("http://b".into(), 10)]
        );
        assert_eq!(provider.tree.pending().hits(), 1);
    }

    #[test]
    fn routing_to_a_confirmed_worker_is_not_a_follow() {
        let provider = provider(60_000, 1);
        let prompt = tokens(4);
        confirm(&provider, "http://a", &prompt, 4);
        for _ in 0..3 {
            let signal = lookup(&provider, &prompt);
            assert!(signal.pending_winners.is_empty());
            provider.record_route(&signal, "http://a");
        }
        assert_eq!(provider.tree.pending().followed(), 0);
        assert_eq!(provider.tree.pending().capped(), 0);
    }
}
