// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared, mutable state that selection policies read: the KV-event cache
//! index, engine load accounting, and affinity assignments.

pub mod affinity_store;
pub mod engine_load;
pub mod kv_events;

pub use affinity_store::AffinityStore;

/// Which engines hold a prefix of the request, measured in blocks of the
/// query. Empty `matches` is a confirmed miss.
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

impl PrefixLookup {
    pub fn from_indexer(outcome: sgl_kv_indexer::PrefixOutcome, query_blocks: usize) -> Self {
        let matches = match outcome {
            sgl_kv_indexer::PrefixOutcome::Matched { matches, .. } => matches
                .into_iter()
                .map(|m| PrefixMatch {
                    address: m.address,
                    matched_prefix_blocks: m.matched_prefix_blocks,
                })
                .collect(),
            sgl_kv_indexer::PrefixOutcome::Empty => Vec::new(),
        };
        Self {
            matches,
            query_blocks,
        }
    }
}
