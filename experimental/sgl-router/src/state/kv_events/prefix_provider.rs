// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree};
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Prefix lookup result from the local radix tree or remote KV indexer,
/// consumed by cache-aware routing.
pub struct PrefixLookupResult {
    pub outcome: PrefixOutcome,
    pub query_blocks: usize,
    /// The query's block hashes when the local tree answered, so routing can
    /// record the placement without rehashing.
    pub block_hashes: Option<Arc<[i64]>>,
}

#[derive(Clone, Debug)]
pub struct RadixTreePrefixProvider {
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
}

impl RadixTreePrefixProvider {
    pub fn new(tree: Arc<HashTree>, block_size_oracle: Arc<BlockSizeOracle>) -> Self {
        Self {
            tree,
            block_size_oracle,
        }
    }

    pub fn match_request_tokens(&self, tokens: &[u32]) -> Option<PrefixLookupResult> {
        let hashes = self.block_hashes(tokens)?;

        let mut depth_by_url = BTreeMap::<String, u32>::new();
        for (worker, depth) in self.tree.prefix_depths(None, &hashes) {
            merge_depth(&mut depth_by_url, worker.url, depth);
        }
        let confirmed = depth_by_url.values().copied().max();
        for (url, depth) in self.tree.pending().depths(&hashes) {
            merge_depth(&mut depth_by_url, url.to_string(), depth);
        }
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
            block_hashes: self.tree.pending().is_enabled().then(|| hashes.into()),
        })
    }

    /// Remember that `signal`'s prompt was just routed to `url`.
    pub fn record_route(&self, signal: &PrefixLookupResult, url: &str) {
        if let Some(hashes) = &signal.block_hashes {
            self.tree.pending().record(url, hashes);
        }
    }

    /// `(dp_rank, cached prefix blocks)` for each rank of `worker_url`.
    pub fn rank_depths(&self, tokens: &[u32], worker_url: &str) -> Vec<(u32, usize)> {
        let Some(hashes) = self.block_hashes(tokens) else {
            return Vec::new();
        };
        self.tree
            .prefix_depths(None, &hashes)
            .into_iter()
            .filter(|(worker, _)| worker.url == worker_url)
            .map(|(worker, depth)| (worker.dp_rank, depth))
            .collect()
    }

    fn block_hashes(&self, tokens: &[u32]) -> Option<Vec<i64>> {
        let block_size = self.block_size_oracle.get()?;
        let hashes = if self.block_size_oracle.is_bigram() {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
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
