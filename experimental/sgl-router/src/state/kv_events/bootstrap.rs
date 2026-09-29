// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Peer-snapshot bootstrap for the KV-event tree: the wire shape
//! ([`PeerSnapshot`]), its constants, and the registry of peers a snapshot
//! may be pulled from.
//!
//! A freshly started replica subscribes to each worker's KV topic mid-stream:
//! ZMQ SUB delivers deltas from whatever sequence the publisher has reached,
//! so every block already resident in the engine's radix cache is invisible to
//! that replica. Until traffic re-stores those blocks the replica routes
//! cache-blind, and its own dispatches scatter prefixes across workers that
//! warm replicas were keeping consolidated. A tree snapshot pulled from a warm
//! sibling over HTTP, grafted beneath the live delta stream, closes that gap.
//!
//! # Why the body carries cursors and not just tree state
//!
//! A graft is only sound if the recipient can prove its own live stream picks
//! up exactly where the snapshot stops. The producer's per-rank last-applied
//! sequence is what makes that checkable: as a FILTER it says which buffered
//! deltas the snapshot already reflects, and as a WATERMARK it says the stream
//! must resume at `cursor + 1`. A hole there means a delta was lost, and a
//! lost `BlockRemoved` is a permanent false cache hit.
//!
//! The wire carries [`WireWorker`], never [`super::tree::KvWorkerId`], and
//! [`super::tree::HashTree::restore_snapshot`] stays module-internal, so
//! network input cannot become a `KvWorkerId` directly.

use serde::{Deserialize, Serialize};

use super::tree::{KvWorkerId, SnapshotNode};

mod peers;

pub use peers::PeerRegistry;

/// Wire-format version. Bump on any incompatible change to [`PeerSnapshot`].
pub const SNAPSHOT_FORMAT: u32 = 1;

/// Path the producer serves snapshots on.
pub const SNAPSHOT_PATH: &str = "/internal/kv_snapshot";

/// Query parameter by which a caller states how stale a cached snapshot it
/// will accept, in milliseconds. See [`PRODUCER_CACHE_TTL`].
pub const MAX_AGE_PARAM: &str = "max_age_ms";

/// Query parameter by which a caller asks for the cursor table alone, with
/// no tree.
pub const CURSORS_ONLY_PARAM: &str = "cursors_only";

/// Default reuse window for callers that send no [`MAX_AGE_PARAM`]. A cached
/// export predates the caller's subscription, and events in that gap reach
/// neither side, so callers that care send `max_age_ms` and the producer
/// rebuilds.
pub const PRODUCER_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(2);

/// A worker identity as it appears on the snapshot wire. Not a
/// [`KvWorkerId`]; see the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireWorker {
    pub url: String,
    pub dp_rank: u32,
}

/// Outbound only: describing a local id on the wire mints nothing. No
/// conversion exists the other way.
impl From<&KvWorkerId> for WireWorker {
    fn from(w: &KvWorkerId) -> Self {
        Self {
            url: w.url.clone(),
            dp_rank: w.dp_rank,
        }
    }
}

/// A peer replica's view of the KV tree, plus the cursors needed to splice it
/// under a live delta stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeerSnapshot {
    /// See [`SNAPSHOT_FORMAT`].
    pub format: u32,
    /// Producer's established block size. Block hashes are only comparable at
    /// the same page size, so a mismatch is fatal to the snapshot.
    pub block_size: u32,
    /// Producer's hashing mode (EAGLE-family workers hash token bigrams).
    pub is_bigram: bool,
    /// True when the producer's hash config is established and its tree holds
    /// nodes, so a cold or half-configured replica is never copied.
    pub producer_ready: bool,
    /// Worker table; node carrier lists index into this.
    pub workers: Vec<WireWorker>,
    /// `(worker-table index, last-applied seq)` at export time; the stream
    /// resumes at `cursor + 1`.
    pub cursors: Vec<(u32, i64)>,
    /// Tree nodes in dependency order; see [`SnapshotNode`].
    pub nodes: Vec<SnapshotNode>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A body with every field populated survives a JSON round trip.
    #[test]
    fn snapshot_round_trips_through_json() {
        let snap = PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 64,
            is_bigram: true,
            producer_ready: true,
            workers: vec![
                WireWorker {
                    url: "http://a:30000".into(),
                    dp_rank: 0,
                },
                WireWorker {
                    url: "http://a:30000".into(),
                    dp_rank: 1,
                },
            ],
            cursors: vec![(0, 41), (1, 7)],
            nodes: vec![
                SnapshotNode {
                    parent: None,
                    block_hash: 100,
                    workers: vec![0, 1],
                    tiers: vec![1, 2],
                },
                SnapshotNode {
                    parent: Some(0),
                    block_hash: 200,
                    workers: vec![1],
                    tiers: vec![],
                },
            ],
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: PeerSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(back, snap);
    }

    /// A node with no `tiers` field parses as device-only.
    #[test]
    fn a_node_without_tiers_parses_as_device_only() {
        let json = r#"{
            "format": 1,
            "block_size": 64,
            "is_bigram": false,
            "producer_ready": true,
            "workers": [{"url": "http://a:30000", "dp_rank": 0}],
            "cursors": [[0, 12]],
            "nodes": [{"parent": null, "block_hash": 5, "workers": [0]}]
        }"#;
        let snap: PeerSnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(snap.nodes.len(), 1);
        assert!(snap.nodes[0].tiers.is_empty());
    }
}
