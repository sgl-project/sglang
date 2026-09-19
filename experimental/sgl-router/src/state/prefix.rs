// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use sgl_kv_indexer::{PrefixIndex, PrefixIndexError, PrefixOutcome};

use super::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, KvEventIndex,
};

/// Which engines hold a prefix of the request, in blocks of the query.
/// Empty `matches` is a confirmed miss.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixLookup {
    pub matches: Vec<PrefixMatch>,
    pub query_blocks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    /// The engine's registered URL.
    pub address: String,
    pub matched_prefix_blocks: u32,
}

/// One lookup per request, shared by every bucket's pick.
pub type PrefixMemo = tokio::sync::OnceCell<Option<PrefixLookup>>;

/// Prefix ownership from the router-local KV-event index or an external indexer.
pub enum PrefixSource {
    Local(Arc<KvEventIndex>),
    Remote {
        index: Arc<dyn PrefixIndex>,
        block_size: Arc<BlockSizeOracle>,
    },
}

impl fmt::Debug for PrefixSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Local(_) => "PrefixSource::Local",
            Self::Remote { .. } => "PrefixSource::Remote",
        })
    }
}

impl PrefixSource {
    /// `None` until an engine has reported its block size. An unavailable
    /// backend reads as a miss; a rejected query is an error.
    pub async fn lookup(&self, tokens: &[u32]) -> Result<Option<PrefixLookup>, PrefixIndexError> {
        let oracle = match self {
            Self::Local(index) => index.block_size_oracle(),
            Self::Remote { block_size, .. } => Arc::clone(block_size),
        };
        let Some(block_size) = oracle.get() else {
            return Ok(None);
        };
        let hashes = if oracle.is_bigram() {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        let query_blocks = hashes.len();
        let matches = match self {
            Self::Local(index) => {
                // DP ranks of one engine collapse to the deepest match.
                let mut depth_by_url = BTreeMap::<String, u32>::new();
                for (worker, depth) in index.tree().prefix_depths(None, &hashes) {
                    let depth = u32::try_from(depth).unwrap_or(u32::MAX);
                    let current = depth_by_url.entry(worker.url).or_default();
                    *current = (*current).max(depth);
                }
                depth_by_url
                    .into_iter()
                    .map(|(address, matched_prefix_blocks)| PrefixMatch {
                        address,
                        matched_prefix_blocks,
                    })
                    .collect()
            }
            Self::Remote { index, .. } => match index.match_prefix(hashes).await {
                Ok(PrefixOutcome::Matched { matches, .. }) => matches
                    .into_iter()
                    .map(|m| PrefixMatch {
                        address: m.address,
                        matched_prefix_blocks: m.matched_prefix_blocks,
                    })
                    .collect(),
                Ok(PrefixOutcome::Empty) => Vec::new(),
                Err(
                    error @ (PrefixIndexError::Overloaded
                    | PrefixIndexError::Timeout
                    | PrefixIndexError::Unreachable
                    | PrefixIndexError::QueryTooLarge),
                ) => {
                    tracing::warn!(%error, "KV Indexer unavailable; treating as a cache miss");
                    Vec::new()
                }
                Err(error) => return Err(error),
            },
        };
        Ok(Some(PrefixLookup {
            matches,
            query_blocks,
        }))
    }
}
