// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Peer-snapshot bootstrap for the KV-event tree: the wire shape
//! ([`PeerSnapshot`]) and its producer constants, the fetch client
//! ([`fetch_snapshot`], [`fetch_cursors`]), the [`VettedSnapshot`] pass that
//! turns wire input into something graftable, the per-rank
//! [`BootstrapTracker`], and the [`PeerRegistry`] of peers a snapshot may be
//! pulled from.
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

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::tree::{KvWorkerId, SnapshotNode};

mod fetch;
mod peers;
mod tracker;
mod vet;

pub use fetch::{fetch_cursors, fetch_snapshot, FetchAnswer};
pub use peers::PeerRegistry;
pub use tracker::BootstrapTracker;
pub use vet::{VetError, VettedSnapshot};

/// Wire-format version. Bump on any incompatible change to [`PeerSnapshot`].
pub const SNAPSHOT_FORMAT: u32 = 1;

/// Path the producer serves snapshots on.
pub const SNAPSHOT_PATH: &str = "/internal/kv_snapshot";

/// Query parameter by which a caller states how stale a cached snapshot it
/// will accept, in milliseconds. See [`PRODUCER_CACHE_TTL`].
pub const MAX_AGE_PARAM: &str = "max_age_ms";

/// Query parameter by which a caller asks for the cursor table alone, with
/// no tree. See [`fetch_cursors`].
pub const CURSORS_ONLY_PARAM: &str = "cursors_only";

/// Default reuse window for callers that send no [`MAX_AGE_PARAM`]. A cached
/// export predates the caller's subscription, and events in that gap reach
/// neither side, so callers that care send `max_age_ms` and the producer
/// rebuilds.
pub const PRODUCER_CACHE_TTL: Duration = Duration::from_secs(2);

/// Default upper bound on one snapshot fetch that is making progress. Stalls
/// are caught by [`SNAPSHOT_FETCH_READ_TIMEOUT`] and dead peers by
/// [`SNAPSHOT_FETCH_CONNECT_TIMEOUT`], so this is sized for a large healthy
/// body end to end: the producer's export build before the first byte, then
/// tens of MB gzipped on the wire.
pub const DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP: Duration = Duration::from_secs(300);

/// Bound on establishing the connection to a peer. A peer that is gone fails
/// this fast, so a caller reaches its next candidate without spending the
/// per-fetch budget on a dead one.
pub const SNAPSHOT_FETCH_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Bound on the gap BETWEEN body chunks, not on the whole response.
///
/// This is what separates "hung" from "big": a total-request timeout fires
/// identically on a peer that sent nothing and on one streaming a large body
/// steadily. Bounding idle time lets an arbitrarily large healthy snapshot
/// complete while still cutting a stalled peer loose.
///
/// The first gap includes the producer building its export before it sends
/// the first byte, which takes tens of seconds on a fleet-sized tree, so the
/// stall bound must outlast a healthy build.
pub const SNAPSHOT_FETCH_READ_TIMEOUT: Duration = Duration::from_secs(60);

/// Per-rank bootstrap state. `settled()` waits on `Pending` ranks until the
/// deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapState {
    /// Registered; no snapshot applied yet.
    Pending,
    /// Snapshot grafted and cursor seeded.
    Recovered,
    /// No usable snapshot; the rank runs on live deltas only.
    Failed,
}

impl BootstrapState {
    /// Numeric encoding for the `sgl_router_kv_bootstrap_state` gauge.
    pub fn as_metric(self) -> u64 {
        match self {
            Self::Pending => 0,
            Self::Recovered => 1,
            Self::Failed => 2,
        }
    }

    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Pending)
    }
}

/// How one fetch from one peer turned out; counted per peer attempt, not per
/// rank (see [`RankOutcome`]). A closed set so no call site can mint a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotOutcome {
    /// Usable body.
    Accepted,
    /// Peer did not answer, or answered non-200 (including the 404 an older
    /// router image returns for an endpoint it does not serve).
    Unreachable,
    /// Answered, but holds no state to graft for the ranks being bootstrapped.
    /// That includes WARM peers whose tree shares no carriers with our live
    /// workers or tracks none of those ranks, so this is not
    /// [`PeerSnapshot::is_cold`] and does not count toward a cold-fleet verdict.
    ColdPeer,
    /// Answered with an untrustworthy body: any [`VetError`] whose
    /// [`VetError::outcome`] is `Rejected`.
    Rejected,
}

impl SnapshotOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Unreachable => "unreachable",
            Self::ColdPeer => "cold_peer",
            Self::Rejected => "rejected",
        }
    }
}

/// How one bounded peer sweep ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepOutcome {
    /// A peer supplied a usable snapshot.
    Found,
    /// Discovery confirmed there are no siblings.
    NoPeers,
    /// Every sibling proved it holds no state.
    FleetCold,
    /// The deadline expired first.
    TimedOut,
    /// Every rank the sweep was run for left `Pending` without its snapshot:
    /// resolved from its own stream's origin, settled cold because every
    /// sibling proved it holds nothing for that rank, or forgotten.
    RanksResolved,
}

impl SweepOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Found => "found",
            Self::NoPeers => "no_peers",
            Self::FleetCold => "fleet_cold",
            Self::TimedOut => "timed_out",
            Self::RanksResolved => "ranks_resolved",
        }
    }
}

/// How one rank's bootstrap ended; recorded once per rank per incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankOutcome {
    /// Grafted, and the splice with the live stream proven.
    Warm,
    /// The live stream did not join the snapshot's watermark; graft discarded.
    Gap,
    /// Grafted and kept, but no batch or peer witnessed the splice.
    WarmUnwitnessed,
    /// The accepted snapshot carried no cursor for this rank.
    Uncovered,
    /// No peer supplied a usable snapshot: the deadline expired, discovery
    /// confirmed there are no siblings, or every sibling proved it has nothing
    /// to give — an empty tree, a permanent incompatibility, or this rank named
    /// in [`PeerSnapshot::empty_ranks`].
    Abandoned,
    /// Held batches hit their cap before a snapshot arrived; bootstrap was
    /// abandoned and they were replayed as live deltas.
    Overflow,
    /// The publisher restarted mid-bootstrap.
    PublisherReset,
    /// The tree refused the snapshot's structure.
    TreeRejected,
    /// The rank's first held batch was its publisher's first batch ever, so its
    /// history is complete without a snapshot and none was grafted.
    FromOrigin,
}

impl RankOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Gap => "gap",
            Self::WarmUnwitnessed => "warm_unwitnessed",
            Self::Uncovered => "uncovered",
            Self::Abandoned => "abandoned",
            Self::Overflow => "overflow",
            Self::PublisherReset => "publisher_reset",
            Self::TreeRejected => "tree_rejected",
            Self::FromOrigin => "from_origin",
        }
    }
}

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
    /// True when the producer's own bootstrap has settled, its hash config is
    /// established, and its tree holds nodes, so a cold, still-bootstrapping
    /// or half-configured replica is never copied.
    pub producer_ready: bool,
    /// Worker table; node carrier lists index into this.
    pub workers: Vec<WireWorker>,
    /// `(worker-table index, last-applied seq)` at export time; the stream
    /// resumes at `cursor + 1`.
    pub cursors: Vec<(u32, i64)>,
    /// Tree nodes in dependency order; see [`SnapshotNode`].
    pub nodes: Vec<SnapshotNode>,
    /// Ranks the producer is subscribed to but held no tree node for at export
    /// time: ranks it is still bootstrapping itself (their batches are held,
    /// not applied), and ranks whose applied stream left nothing standing.
    ///
    /// Evidence for the consumer's per-rank cold settle: a rank every sibling
    /// names here is one no sibling can supply, and the export was taken after
    /// the consumer subscribed (see [`MAX_AGE_PARAM`]), so nothing a sibling
    /// learns about it later is anything the consumer's own subscription will
    /// not also deliver. Without it, "has nothing for this rank" and "has not
    /// discovered this rank yet" look identical — a rank absent from
    /// `workers` — and only the second is worth waiting for.
    ///
    /// Backward compatible in both directions, so no [`SNAPSHOT_FORMAT`] bump:
    /// an older consumer ignores the unknown field, and an older producer omits
    /// it, which reads as empty — no evidence, so the consumer keeps waiting
    /// exactly as it did before the field existed. Omitted from the body when
    /// empty for the same reason.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub empty_ranks: Vec<WireWorker>,
}

impl PeerSnapshot {
    /// Not ready, or empty tree.
    pub fn is_cold(&self) -> bool {
        !self.producer_ready || self.holds_no_state()
    }

    /// Empty tree. Weaker than [`Self::is_cold`]: a peer mid-bootstrap reports
    /// `producer_ready: false` with nodes.
    pub fn holds_no_state(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Whether this body names `(url, dp_rank)` in [`Self::empty_ranks`]: the
    /// producer is subscribed to it and holds nothing for it. Addressed by wire
    /// identity, like [`Self::wire_cursor_for`], because it is evidence, not
    /// tree state.
    pub fn holds_nothing_for(&self, url: &str, dp_rank: u32) -> bool {
        self.empty_ranks
            .iter()
            .any(|w| w.url == url && w.dp_rank == dp_rank)
    }

    /// Last-applied seq the producer reports for `(url, dp_rank)`. Sequence
    /// numbers are the publisher's, so any observer's value is evidence of
    /// publisher progress.
    pub fn wire_cursor_for(&self, url: &str, dp_rank: u32) -> Option<i64> {
        let idx = self
            .workers
            .iter()
            .position(|w| w.url == url && w.dp_rank == dp_rank)? as u32;
        self.cursors
            .iter()
            .find(|(i, _)| *i == idx)
            .map(|(_, seq)| *seq)
    }
}

#[cfg(test)]
mod test_support {
    use super::*;

    pub(super) fn wire_worker(url: &str, dp_rank: u32) -> WireWorker {
        WireWorker {
            url: url.into(),
            dp_rank,
        }
    }

    pub(super) fn sample_snapshot() -> PeerSnapshot {
        PeerSnapshot {
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
            empty_ranks: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{sample_snapshot, wire_worker};
    use super::*;

    /// A wire cursor is read by worker identity, without vetting. A worker the
    /// snapshot does not mention yields `None` rather than a misaddressed
    /// cursor from another rank's table slot.
    #[test]
    fn wire_cursor_is_addressed_by_identity_not_table_position() {
        let snap = PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 4,
            is_bigram: false,
            producer_ready: true,
            workers: vec![wire_worker("http://w1", 0), wire_worker("http://w2", 1)],
            // Out of table order, and with no entry for index 0.
            cursors: vec![(1, 77)],
            nodes: vec![],
            empty_ranks: vec![],
        };
        assert_eq!(snap.wire_cursor_for("http://w2", 1), Some(77));
        assert_eq!(
            snap.wire_cursor_for("http://w1", 0),
            None,
            "a worker in the table without a cursor must not borrow another's",
        );
        assert_eq!(
            snap.wire_cursor_for("http://w2", 0),
            None,
            "dp_rank matters"
        );
        assert_eq!(snap.wire_cursor_for("http://nope", 0), None);
    }

    /// A body with every field populated survives a JSON round trip.
    /// `empty_ranks` is additive on the wire. An older producer's body has no
    /// such key and must decode as "no evidence"; an empty list is omitted, so
    /// an older consumer sees the body it always has.
    #[test]
    fn empty_ranks_is_backward_compatible_on_the_wire() {
        let old_producer = serde_json::json!({
            "format": SNAPSHOT_FORMAT,
            "block_size": 64,
            "is_bigram": false,
            "producer_ready": true,
            "workers": [],
            "cursors": [],
            "nodes": [],
        });
        let decoded: PeerSnapshot = serde_json::from_value(old_producer).unwrap();
        assert!(decoded.empty_ranks.is_empty());
        assert!(!decoded.holds_nothing_for("http://a", 0));

        let encoded = serde_json::to_value(sample_snapshot()).unwrap();
        assert!(encoded.get("empty_ranks").is_none(), "omitted when empty");

        let mut named = sample_snapshot();
        named.empty_ranks = vec![wire_worker("http://a", 1)];
        let round: PeerSnapshot =
            serde_json::from_slice(&serde_json::to_vec(&named).unwrap()).unwrap();
        assert!(round.holds_nothing_for("http://a", 1));
        assert!(!round.holds_nothing_for("http://a", 0), "dp_rank matters");
    }

    #[test]
    fn snapshot_round_trips_through_json() {
        let snap = sample_snapshot();
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

    /// A rank the producer does not list, or lists without a cursor, has no
    /// cursor.
    #[test]
    fn wire_cursor_for_resolves_by_wire_identity() {
        let mut snap = sample_snapshot();
        assert_eq!(snap.wire_cursor_for("http://a:30000", 0), Some(41));
        assert_eq!(snap.wire_cursor_for("http://a:30000", 1), Some(7));
        assert_eq!(snap.wire_cursor_for("http://a:30000", 2), None);
        assert_eq!(snap.wire_cursor_for("http://b:30000", 0), None);

        snap.cursors.retain(|(i, _)| *i != 1);
        assert_eq!(
            snap.wire_cursor_for("http://a:30000", 1),
            None,
            "a listed worker with no cursor entry is not a witness",
        );
    }
}
