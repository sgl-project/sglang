// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Peer-snapshot bootstrap for the KV-event tree — wire format and producer.
//!
//! A freshly started replica subscribes to each worker's KV topic mid-stream:
//! ZMQ SUB delivers deltas from whatever sequence the publisher has reached,
//! so every block already resident in the engine's radix cache is invisible to
//! that replica. Until traffic re-stores those blocks the replica routes
//! cache-blind, and — worse for the fleet — its own dispatches scatter
//! prefixes across workers that warm replicas were keeping consolidated.
//!
//! The answer is to pull a tree snapshot from a warm sibling replica over HTTP
//! and graft it beneath the live delta stream. This module holds the shared
//! wire shape ([`PeerSnapshot`]) and the producer's half of that exchange; the
//! consumer lands on top of it.
//!
//! # Why the body carries cursors and not just tree state
//!
//! A graft is only sound if the consumer can prove its own live stream picks
//! up exactly where the snapshot stops. The producer's per-rank last-applied
//! sequence is what makes that checkable: as a FILTER it tells the consumer
//! which buffered deltas the snapshot already reflects, and as a WATERMARK it
//! says the stream must resume at `cursor + 1`. A hole there means a delta was
//! lost, and a lost `BlockRemoved` is a permanent false cache hit — so the
//! consumer must be able to see the hole and refuse the graft.
//!
//! # Trust boundary
//!
//! Snapshots cross the network, so the boundary is enforced by types rather
//! than by discipline. [`PeerSnapshot`] carries [`WireWorker`], never
//! [`super::tree::KvWorkerId`]: deserialising straight into the latter would
//! mint routing identities from network input, which its provenance contract
//! forbids. Resolving a wire identity against the locally discovered worker
//! set is the only way across, and
//! [`super::tree::HashTree::restore_snapshot`] stays module-internal so that
//! step cannot be skipped.

use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::tree::SnapshotNode;

/// Wire-format version. Bump on any incompatible change to
/// [`PeerSnapshot`]; a receiver rejects anything it does not recognise, so a
/// mixed-version fleet degrades to cold boots rather than corrupt trees.
pub const SNAPSHOT_FORMAT: u32 = 1;

/// Path served by the producer and fetched by the consumer.
pub const SNAPSHOT_PATH: &str = "/internal/kv_snapshot";

/// Query parameter by which a consumer states how stale a cached snapshot it
/// will accept, in milliseconds. See [`PRODUCER_CACHE_TTL`].
pub const MAX_AGE_PARAM: &str = "max_age_ms";

/// Query parameter by which a consumer asks for the cursor table alone, with
/// no tree.
pub const CURSORS_ONLY_PARAM: &str = "cursors_only";

/// How long a producer may reuse an already-built snapshot for a request that
/// states no freshness requirement of its own.
///
/// # Why staleness is a correctness input, not a tuning knob
///
/// A cached snapshot was exported some time BEFORE the consumer that receives
/// it subscribed. Every event the publisher emitted in that window is in
/// neither place — not in the snapshot, not in the consumer's held queue — so
/// the live stream resumes above `cursor + 1`, the watermark reads a hole, and
/// the graft has to be discarded. On a fleet whose ranks publish continuously
/// that window is the dominant loss.
///
/// A shorter fixed TTL only makes that less likely. What removes it is letting
/// the CONSUMER state the requirement, since only the consumer knows when its
/// ranks began holding: [`MAX_AGE_PARAM`] carries "no older than this", and the
/// producer rebuilds when its cached entry does not meet it.
///
/// So this constant is only the DEFAULT for requests that state no
/// requirement — an older router image that does not send the parameter. It
/// can be generous, which is what keeps a boot herd sharing one tree walk.
pub const PRODUCER_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(2);

/// A worker identity as it appears on the snapshot wire.
///
/// Deliberately NOT [`super::tree::KvWorkerId`]: deserialising straight into
/// that type would mint routing identities from network input, which its
/// provenance contract forbids. A consumer must resolve these against its own
/// live worker set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireWorker {
    pub url: String,
    pub dp_rank: u32,
}

/// A peer replica's view of the KV tree, plus the cursors needed to splice it
/// under a live delta stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerSnapshot {
    /// See [`SNAPSHOT_FORMAT`].
    pub format: u32,
    /// Producer's established block size. Block hashes are only comparable at
    /// the same page size, so a mismatch is fatal to the snapshot.
    pub block_size: u32,
    /// Producer's hashing mode (EAGLE-family workers hash token bigrams).
    /// Reported for diagnosis and for older consumers that vet on it.
    pub is_bigram: bool,
    /// Whether the producer is a tree worth copying: it knows its own hashing
    /// config AND it actually holds nodes.
    ///
    /// The node check matters during a rolling update. Without it two new
    /// replicas could bootstrap from each other and both inherit nothing.
    ///
    /// The primary defence is upstream of this flag: a replica that has not
    /// passed `/readyz` is not `ready` in its EndpointSlice, so peer discovery
    /// never offers it as a candidate. This field covers the propagation race
    /// where readiness and EndpointSlice state briefly disagree.
    pub producer_ready: bool,
    /// Worker table; node carrier lists index into this.
    pub workers: Vec<WireWorker>,
    /// `(worker-table index, last-applied seq)` on the producer at export
    /// time. The consumer seeds these so its pump can filter the deltas the
    /// snapshot already reflects.
    pub cursors: Vec<(u32, i64)>,
    /// Tree nodes in dependency order; see [`SnapshotNode`].
    pub nodes: Vec<SnapshotNode>,
}

/// Sibling replicas a snapshot may be pulled from, kept current by peer
/// discovery.
///
/// Empty disables peer bootstrap without any other configuration — but "empty"
/// has two meanings and conflating them is the failure mode this type exists
/// to prevent, so it tracks enough state to tell them apart.
#[derive(Debug, Default)]
pub struct PeerRegistry {
    peers: Mutex<Vec<String>>,
    /// Whether peer discovery has reported at least once, even with an empty
    /// result.
    ///
    /// WHY this matters: an empty peer set is ambiguous. It means either "the
    /// watch has not delivered yet" — routine, since worker discovery
    /// regularly wins the race against it — or "this replica genuinely has no
    /// siblings". Treating the first as the second makes a joining replica give
    /// up before its peers are even known and boot cold, which silently
    /// defeats the whole feature. Only after a sync is an empty set
    /// conclusive.
    synced: AtomicBool,
    /// Whether a non-empty peer set has ever been observed.
    ///
    /// The peer set legitimately dips to empty for an instant — an
    /// EndpointSlice repack deletes the last slice before its replacement
    /// arrives, and during a rolling update every sibling can be `notReady` at
    /// once. A bootstrap retry loop samples [`Self::known_to_have_no_peers`]
    /// many times per boot, so it has many chances to catch such a dip, and one
    /// hit is permanent for that boot. Once siblings have been seen, "empty" is
    /// treated as transient and the loop keeps waiting rather than concluding
    /// this replica is alone.
    ever_had_peers: AtomicBool,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the peer set wholesale, as a relist or a slice event produces.
    pub fn replace(&self, peers: Vec<String>) {
        self.synced.store(true, Ordering::Relaxed);
        if !peers.is_empty() {
            self.ever_had_peers.store(true, Ordering::Relaxed);
        }
        let mut guard = self.peers.lock();
        if *guard != peers {
            info!(count = peers.len(), peers = ?peers, "kv-bootstrap: peer set updated");
        }
        *guard = peers;
    }

    pub fn len(&self) -> usize {
        self.peers.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.lock().is_empty()
    }

    /// Whether peer discovery has reported at least once. An empty peer set is
    /// only conclusive once this is true.
    pub fn synced(&self) -> bool {
        self.synced.load(Ordering::Relaxed)
    }

    /// True when discovery has confirmed this replica has no siblings, so
    /// there is no point waiting for one.
    pub fn known_to_have_no_peers(&self) -> bool {
        self.synced() && self.is_empty() && !self.ever_had_peers.load(Ordering::Relaxed)
    }

    /// Candidate peers in shuffled order, so simultaneous boots spread their
    /// snapshot fetches instead of stampeding whichever peer sorts first.
    pub fn candidates(&self) -> Vec<String> {
        let mut peers = self.peers.lock().clone();
        peers.shuffle(&mut rand::thread_rng());
        peers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The format is the fleet's compatibility contract, so the round trip has
    /// to hold for a body with every field populated — including the tier
    /// lists a pre-tiering peer omits.
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

        assert_eq!(back.format, snap.format);
        assert_eq!(back.block_size, snap.block_size);
        assert_eq!(back.is_bigram, snap.is_bigram);
        assert_eq!(back.producer_ready, snap.producer_ready);
        assert_eq!(back.workers, snap.workers);
        assert_eq!(back.cursors, snap.cursors);
        assert_eq!(back.nodes, snap.nodes);
    }

    /// A body from a producer that predates tiering omits `tiers` entirely.
    /// It must still parse — that is what lets a mixed-version fleet bootstrap
    /// at all, and why the tier addition needed no format bump.
    #[test]
    fn a_pre_tiering_body_still_parses() {
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

    #[test]
    fn a_transient_empty_peer_set_is_not_conclusive() {
        let reg = PeerRegistry::new();
        assert!(
            !reg.known_to_have_no_peers(),
            "before any sync, empty means the watch has not delivered",
        );

        reg.replace(vec!["http://a:30000".into()]);
        assert!(reg.synced());
        assert!(!reg.known_to_have_no_peers());

        // An EndpointSlice repack, or a rolling update with every sibling
        // notReady, empties the set for an instant. Having once seen siblings
        // is what keeps that from reading as "I am alone".
        reg.replace(vec![]);
        assert!(reg.is_empty());
        assert!(
            !reg.known_to_have_no_peers(),
            "a dip after siblings were seen is transient, not conclusive",
        );
    }

    #[test]
    fn a_synced_empty_set_with_no_history_means_no_siblings() {
        let reg = PeerRegistry::new();
        reg.replace(vec![]);
        assert!(reg.synced());
        assert!(reg.known_to_have_no_peers());
    }

    #[test]
    fn candidates_shuffle_without_losing_entries() {
        let reg = PeerRegistry::new();
        let peers: Vec<String> = (0..16).map(|i| format!("http://p{i}:30000")).collect();
        reg.replace(peers.clone());

        let mut got = reg.candidates();
        assert_eq!(got.len(), peers.len());
        got.sort();
        let mut want = peers;
        want.sort();
        assert_eq!(got, want, "shuffling must not drop or duplicate a peer");
    }
}
