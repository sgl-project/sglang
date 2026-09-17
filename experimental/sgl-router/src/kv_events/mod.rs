//! ZMQ-based KV-cache event indexer for cache-aware routing.
//!
//! Decodes the msgpack wire format emitted by SGLang's `ZmqEventPublisher`
//! (`python/sglang/srt/utils/event_publisher.py`; KV event types in
//! `python/sglang/srt/disaggregation/kv_events.py`) and maintains the
//! router-side index used for cache-aware request routing.
//!
//! # Submodules
//!
//! - [`wire`] — msgpack types and [`decode_event_batch`]; the contract
//!   with the SGLang publisher. Pure decoding; no I/O.
//! - [`hash`] — block-hash compute mirroring SGLang `RadixKey.hash_page`.
//! - [`tree`] — hash-keyed radix tree consumed by the routing path,
//!   tracking the storage tier each worker holds a block on.
//! - [`tally`] — per-(kind, medium) counters of the events the pump applied.
//! - [`subscriber`] — per-worker ZMQ SUB tasks.
//! - [`discovery`] — `/server_info` parse → publisher endpoint.
//! - [`index`] — public façade bundling the tree + subscribers + pump.

pub mod block_size_oracle;
pub mod discovery;
pub mod hash;
pub mod index;
pub mod subscriber;
pub mod tally;
pub mod tree;
pub mod wire;

pub use block_size_oracle::BlockSizeOracle;
pub(crate) use discovery::classify_bigram;
pub use discovery::{fetch_event_config, EventConfig};
pub use hash::{compute_block_hashes, compute_block_hashes_bigram, sha256_to_i64};
pub use index::{KvEventIndex, KvIndexMetrics};
pub use subscriber::{KvEventSubscriberRegistry, SubKind, WorkerEvent};
pub use tally::{EventKind, EventTally, TallyRow};
pub use tree::{
    HashTree, KvWorkerId, MatchResult, TierCounts, Tiers, ACCOUNTING_REASONS, TIER_SLOT_COUNT,
};
pub use wire::{
    decode_event_batch, BlockRemoved, BlockStored, DecodeError, KvCacheEvent, KvEventBatch,
};

use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Prefix observations prepared at ingress from either the local or remote index.
pub struct PrefixSignal {
    pub outcome: PrefixOutcome,
    pub query_blocks: usize,
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

    pub fn match_request_tokens(&self, tokens: &[u32]) -> Option<PrefixSignal> {
        let (query_blocks, depth_by_url) =
            prefix_depths_by_url(&self.tree, &self.block_size_oracle, tokens)?;
        let best_prefix_blocks = depth_by_url.values().copied().max()?;
        let matches = depth_by_url
            .into_iter()
            .map(|(address, matched_prefix_blocks)| PrefixMatch {
                worker_id: address.clone(),
                address,
                matched_prefix_blocks,
            })
            .collect();
        Some(PrefixSignal {
            outcome: PrefixOutcome::Matched {
                matches,
                best_prefix_blocks,
            },
            query_blocks,
        })
    }
}

/// Hash full prompt blocks using the engine's advertised key format.
pub fn request_block_hashes(oracle: &BlockSizeOracle, tokens: &[u32]) -> Option<Vec<i64>> {
    let block_size = oracle.get()? as usize;
    Some(if oracle.is_bigram() {
        compute_block_hashes_bigram(tokens, block_size)
    } else {
        compute_block_hashes(tokens, block_size)
    })
}

/// Collapse DP ranks to the deepest prefix held at each dispatch URL.
pub(crate) fn prefix_depths_by_url(
    tree: &HashTree,
    oracle: &BlockSizeOracle,
    tokens: &[u32],
) -> Option<(usize, BTreeMap<String, u32>)> {
    let hashes = request_block_hashes(oracle, tokens)?;
    if hashes.is_empty() {
        return None;
    }
    let mut depths = BTreeMap::<String, u32>::new();
    for (worker, depth) in tree.prefix_depths(None, &hashes) {
        let depth = u32::try_from(depth).unwrap_or(u32::MAX);
        depths
            .entry(worker.url)
            .and_modify(|current| *current = (*current).max(depth))
            .or_insert(depth);
    }
    Some((hashes.len(), depths))
}
