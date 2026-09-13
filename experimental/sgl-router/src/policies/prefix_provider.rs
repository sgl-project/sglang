// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{ExternalPrefixSignal, TreePrefixView};
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree, PrefixDepth,
};
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

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

    pub fn match_request_tokens(&self, tokens: &[u32]) -> Option<ExternalPrefixSignal> {
        let block_size = self.block_size_oracle.get()?;
        let hashes = if self.block_size_oracle.is_bigram() {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        let &block0 = hashes.first()?;

        // One entry per worker URL. A URL can appear under several dp ranks;
        // the deepest rank is the one routing gets, and its tier is the tier
        // that hit will be served from — so depth picks the winner and the
        // tier only breaks a tie. Taking the best tier across ranks
        // independently would report an in-place hit for a prefix that is
        // actually loaded back.
        let mut best_by_url = BTreeMap::<String, PrefixDepth>::new();
        for (worker, depth) in self.tree.prefix_depths(None, &hashes) {
            best_by_url
                .entry(worker.url)
                .and_modify(|current| {
                    if (depth.blocks, current.best_tier_rank())
                        > (current.blocks, depth.best_tier_rank())
                    {
                        *current = depth;
                    }
                })
                .or_insert(depth);
        }

        let tree_view = TreePrefixView {
            owner_tiers: best_by_url
                .iter()
                .filter_map(|(url, depth)| Some((url.clone(), depth.best_tier_label()?)))
                .collect::<HashMap<_, _>>(),
            // Only a miss needs attributing, and this is a second read of the
            // tree under its own lock — so a fleet that is matching well never
            // pays for it.
            block0_in_tree: best_by_url
                .is_empty()
                .then(|| self.tree.contains_hash(block0)),
        };

        // A walk that reached nobody is still an answer: it says the tree was
        // consulted and matched nothing, which `block0_in_tree` then explains.
        let Some(best_prefix_blocks) = best_by_url.values().map(|d| d.blocks).max() else {
            return Some(ExternalPrefixSignal {
                outcome: PrefixOutcome::Empty,
                query_blocks: hashes.len(),
                tree_view: Some(tree_view),
            });
        };
        let matches = best_by_url
            .into_iter()
            .map(|(address, depth)| PrefixMatch {
                worker_id: address.clone(),
                address,
                matched_prefix_blocks: u32::try_from(depth.blocks).unwrap_or(u32::MAX),
            })
            .collect();
        Some(ExternalPrefixSignal {
            outcome: PrefixOutcome::Matched {
                matches,
                best_prefix_blocks: u32::try_from(best_prefix_blocks).unwrap_or(u32::MAX),
            },
            query_blocks: hashes.len(),
            tree_view: Some(tree_view),
        })
    }
}
