//! Turn untrusted wire input into a graftable `VettedSnapshot`.

use std::collections::HashSet;

use tracing::debug;

use super::{PeerSnapshot, SnapshotOutcome, SNAPSHOT_FORMAT};
use crate::state::kv_events::tree::{
    HashTree, KvWorkerId, RestoreError, ShapeViolation, SnapshotNode,
};

/// Why a wire snapshot was refused before any tree mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VetError {
    UnknownFormat {
        got: u32,
        want: u32,
    },
    /// A node's `parent` was not a backward reference into the node list.
    InvalidParentReference {
        index: usize,
        parent: u32,
    },
    /// A node's `tiers` was neither empty nor the same length as its
    /// `workers`, so carriers cannot be paired with their tiers.
    /// [`SnapshotNode::retain_carriers`] drops a carrier that lacks a tier
    /// entry, so this check keeps malformed pairing from being silently
    /// dropped.
    TierTableMismatch {
        index: usize,
    },
    /// Two cursor entries resolve to the same worker, so its watermark is
    /// ambiguous and [`VettedSnapshot::cursor_for`] would silently pick one.
    DuplicateCursor {
        worker: u32,
    },
    /// Two worker-table entries name the same live worker. An honest export
    /// lists each worker once, and merging the entries would leave nodes that
    /// name one carrier twice.
    DuplicateWorker {
        index: usize,
    },
    BlockSizeMismatch {
        peer: u32,
        local: u32,
    },
    ProducerCold,
    /// The snapshot was well formed, but nothing in it survives vetting for this
    /// replica — every carrier was a worker we do not know, or held the node
    /// only on tiers this build does not rank. Unlike `ProducerCold`, the peer
    /// has a real tree with no overlap with ours.
    ///
    /// Refused rather than accepted as empty: a seeded cursor with no tree
    /// behind it filters away every live delta at or below its watermark.
    NothingUsable {
        wire_nodes: usize,
        dropped_workers: usize,
    },
}

impl std::fmt::Display for VetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFormat { got, want } => {
                write!(f, "unknown snapshot format {got} (want {want})")
            }
            Self::InvalidParentReference { index, parent } => write!(
                f,
                "snapshot node {index} has parent {parent}, not a backward reference",
            ),
            Self::TierTableMismatch { index } => write!(
                f,
                "snapshot node {index} has a tiers list that does not pair with its workers",
            ),
            Self::DuplicateCursor { worker } => write!(
                f,
                "snapshot has more than one cursor for worker {worker}; the watermark is ambiguous",
            ),
            Self::DuplicateWorker { index } => write!(
                f,
                "snapshot worker {index} repeats an earlier worker-table entry",
            ),
            Self::BlockSizeMismatch { peer, local } => write!(
                f,
                "peer block_size {peer} disagrees with local {local}; block hashes are incomparable",
            ),
            Self::NothingUsable {
                wire_nodes,
                dropped_workers,
            } => write!(
                f,
                "none of the peer's {wire_nodes} nodes survive vetting \
                 ({dropped_workers} of its workers are unknown here)",
            ),
            Self::ProducerCold => write!(
                f,
                "peer is not a usable source (still bootstrapping, or settled with an empty tree)",
            ),
        }
    }
}

impl VetError {
    pub fn outcome(&self) -> SnapshotOutcome {
        match self {
            // Not the peer's fault and not permanent — it may discover our
            // workers moments later, so this must stay retriable.
            Self::NothingUsable { .. } | Self::ProducerCold => SnapshotOutcome::ColdPeer,
            _ => SnapshotOutcome::Rejected,
        }
    }
}

/// A snapshot whose worker identities have been resolved against the local
/// live set and whose carrier lists have been remapped onto the surviving
/// workers.
///
/// Only [`VettedSnapshot::from_wire`] builds one, and
/// [`VettedSnapshot::graft_into`] is the only route from it to the tree's
/// restore path; the module docs describe the trust boundary this upholds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VettedSnapshot {
    /// Locally-minted worker ids, resolved from the live set.
    /// [`Self::retain_workers`] filters carriers and cursors, never this
    /// table.
    worker_table: Vec<KvWorkerId>,
    /// Nodes with carrier indices remapped into `worker_table`, pruned by
    /// [`VettedSnapshot::prune_carrier_less`].
    nodes: Vec<SnapshotNode>,
    /// `(worker, last-applied seq)` for workers that survived vetting.
    cursors: Vec<(KvWorkerId, i64)>,
    /// Wire workers the local replica does not know. Expected and benign
    /// during a rolling worker change; logged, and worth watching if
    /// persistently large.
    dropped_workers: usize,
}

impl VettedSnapshot {
    /// Nodes that survived vetting, for logging.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Distinct workers that survived vetting.
    pub fn worker_count(&self) -> usize {
        self.worker_table.len()
    }

    /// Whether `id` survived vetting.
    pub fn has_worker(&self, id: &KvWorkerId) -> bool {
        self.worker_table.contains(id)
    }

    /// Wire workers this replica does not know; see the field docs.
    pub fn dropped_workers(&self) -> usize {
        self.dropped_workers
    }

    /// `(carried, structure)`: nodes with at least one surviving carrier vs
    /// carrier-less interior kept on a path to a carried descendant. Structure
    /// nodes cannot answer a query but can turn one into a hit-shaped miss, so
    /// their count is surfaced rather than buried inside `node_count`.
    pub(crate) fn carrier_counts(&self) -> (usize, usize) {
        let structure = self.nodes.iter().filter(|n| n.workers.is_empty()).count();
        (self.nodes.len() - structure, structure)
    }

    /// Graft this snapshot's nodes into `tree`, returning how many were applied.
    ///
    /// Hands over the ids vetting resolved against the live set, the
    /// precondition the restore path trusts. Call it from the tree's single
    /// writer, like every tree mutation.
    pub fn graft_into(&self, tree: &HashTree) -> Result<usize, RestoreError> {
        tree.restore_snapshot(&self.worker_table, &self.nodes)
    }

    /// Build one directly, for tests that need a specific shape without a
    /// wire round trip. Test-only because it skips vetting.
    #[cfg(test)]
    pub fn from_parts_for_test(
        worker_table: Vec<KvWorkerId>,
        nodes: Vec<SnapshotNode>,
        cursors: Vec<(KvWorkerId, i64)>,
        dropped_workers: usize,
    ) -> Self {
        Self {
            worker_table,
            nodes,
            cursors,
            dropped_workers,
        }
    }

    /// Drop every carrier and cursor whose worker is not in `keep`, then prune
    /// the structure that strands. Lets a caller graft for a subset of the
    /// vetted workers.
    pub fn retain_workers(&mut self, keep: &HashSet<KvWorkerId>) {
        let allowed: Vec<bool> = self.worker_table.iter().map(|w| keep.contains(w)).collect();
        if allowed.iter().all(|&a| a) {
            return;
        }
        let is_allowed = |w: u32| allowed.get(w as usize) == Some(&true);
        for node in &mut self.nodes {
            if node.workers.iter().all(|&w| is_allowed(w)) {
                continue;
            }
            node.retain_carriers(|w, _| is_allowed(w).then_some(w));
        }
        self.cursors.retain(|(w, _)| keep.contains(w));
        let pruned = self.prune_carrier_less();
        if pruned > 0 {
            debug!(
                pruned,
                remaining = self.nodes.len(),
                "kv-bootstrap: dropped carrier-less nodes after filtering to pending ranks",
            );
        }
    }

    /// Drop nodes that carry no worker and have no surviving descendant that
    /// does, remapping parent indices. Returns the number of nodes dropped.
    ///
    /// `match_prefix` reports the carriers of the DEEPEST matched node, so a
    /// carrier-less node grafted below a carrying one turns a real cache hit
    /// into a match with no workers, and nothing ever reclaims it. Vetting
    /// leaves such nodes routinely: carriers of workers this replica has not
    /// discovered, carriers held only on tiers this build does not rank, and
    /// carriers [`Self::retain_workers`] filters out.
    pub fn prune_carrier_less(&mut self) -> usize {
        let n = self.nodes.len();
        // Parents precede children, so one reverse pass settles every node,
        // and a kept node's parent is always kept.
        let mut keep = vec![false; n];
        for (i, node) in self.nodes.iter().enumerate().rev() {
            keep[i] |= !node.workers.is_empty();
            if let (true, Some(p)) = (keep[i], node.parent) {
                keep[p as usize] = true;
            }
        }
        let pruned = keep.iter().filter(|&&k| !k).count();
        if pruned == 0 {
            return 0;
        }

        const DROPPED: u32 = u32::MAX;
        let mut new_index = vec![DROPPED; n];
        let mut out: Vec<SnapshotNode> = Vec::with_capacity(n - pruned);
        let nodes = std::mem::take(&mut self.nodes);
        for (i, (mut node, kept)) in nodes.into_iter().zip(keep).enumerate() {
            if !kept {
                continue;
            }
            if let Some(p) = node.parent {
                let np = new_index[p as usize];
                debug_assert_ne!(np, DROPPED, "node {i} is kept but its parent {p} is not");
                if np == DROPPED {
                    continue;
                }
                node.parent = Some(np);
            }
            new_index[i] = out.len() as u32;
            out.push(node);
        }
        self.nodes = out;
        n - self.nodes.len()
    }

    /// Whether the snapshot has a cursor for any of `ranks`; one without
    /// cannot seed a watermark for them.
    pub fn covers_any(&self, ranks: &[KvWorkerId]) -> bool {
        ranks.iter().any(|r| self.cursor_for(r).is_some())
    }

    /// Every rank [`Self::covers_any`] would accept, for checking many ranks
    /// against one snapshot.
    pub fn covered_ranks(&self) -> HashSet<&KvWorkerId> {
        self.cursors.iter().map(|(w, _)| w).collect()
    }

    /// Last-applied sequence the producer had for `worker`, or `None` when it
    /// did not track that worker.
    pub fn cursor_for(&self, worker: &KvWorkerId) -> Option<i64> {
        self.cursors
            .iter()
            .find(|(w, _)| w == worker)
            .map(|(_, seq)| *seq)
    }

    /// Validate a wire snapshot and resolve its worker table against `live`.
    ///
    /// `local_block_size` is `None` before any worker has established one, in
    /// which case the peer's value is accepted — there is nothing yet to
    /// contradict it, and `add_worker` will reject any worker that disagrees.
    ///
    /// The producer's `is_bigram` stamp is not vetted: hashing mode belongs to
    /// the publishing worker, and every surviving node is carried by a worker
    /// this replica discovered and hashes for in its own established mode.
    pub fn from_wire(
        snap: PeerSnapshot,
        live: &HashSet<KvWorkerId>,
        local_block_size: Option<u32>,
    ) -> Result<Self, VetError> {
        if snap.format != SNAPSHOT_FORMAT {
            return Err(VetError::UnknownFormat {
                got: snap.format,
                want: SNAPSHOT_FORMAT,
            });
        }
        // Refuse a producer that reports not-ready, or whose tree is empty
        // whatever it reports.
        if snap.is_cold() {
            return Err(VetError::ProducerCold);
        }
        if let Some(local) = local_block_size {
            if local != snap.block_size {
                return Err(VetError::BlockSizeMismatch {
                    peer: snap.block_size,
                    local,
                });
            }
        }

        // `prune_carrier_less` indexes by `parent`, so bound it and pair tiers
        // before anything reads the nodes.
        for (i, rec) in snap.nodes.iter().enumerate() {
            rec.check_shape(i).map_err(|v| match v {
                ShapeViolation::NonBackwardParent { parent } => {
                    VetError::InvalidParentReference { index: i, parent }
                }
                ShapeViolation::TierTableMismatch => VetError::TierTableMismatch { index: i },
            })?;
        }

        // Resolve wire identities to live ones. `remap[i]` is the new index of
        // wire worker `i`, or `None` when this replica does not know it.
        let mut worker_table: Vec<KvWorkerId> = Vec::new();
        let mut seen: HashSet<&KvWorkerId> = HashSet::new();
        let mut remap: Vec<Option<u32>> = Vec::with_capacity(snap.workers.len());
        let mut dropped_workers = 0usize;
        for (index, w) in snap.workers.iter().enumerate() {
            // Construct only to look up; the id kept is the one from `live`,
            // preserving registry provenance.
            let probe = KvWorkerId::new(w.url.clone(), w.dp_rank);
            match live.get(&probe) {
                Some(known) => {
                    if !seen.insert(known) {
                        return Err(VetError::DuplicateWorker { index });
                    }
                    remap.push(Some(worker_table.len() as u32));
                    worker_table.push(known.clone());
                }
                None => {
                    remap.push(None);
                    dropped_workers += 1;
                }
            }
        }
        let resolve = |w: u32| remap.get(w as usize).copied().flatten();

        // Drop carriers of unknown workers and carriers on no ranked tier, so
        // `prune_carrier_less` sees every node the restore path would leave
        // carrier-less.
        let nodes: Vec<SnapshotNode> = snap
            .nodes
            .into_iter()
            .map(|mut n| {
                // Rebuilding both lists costs two allocations per node, and on
                // the common path — every wire worker known, every tier ranked —
                // nothing changes. Only pay for the nodes it actually changes.
                let changes = n
                    .workers
                    .iter()
                    .enumerate()
                    .any(|(k, &w)| resolve(w) != Some(w) || n.carrier_tiers(k).is_empty());
                if changes {
                    n.retain_carriers(|w, tiers| if tiers.is_empty() { None } else { resolve(w) });
                }
                n
            })
            .collect();

        let mut cursors: Vec<(KvWorkerId, i64)> = Vec::with_capacity(snap.cursors.len());
        let mut cursor_seen = vec![false; worker_table.len()];
        for (idx, seq) in snap.cursors {
            let Some(slot) = resolve(idx) else {
                continue;
            };
            if std::mem::replace(&mut cursor_seen[slot as usize], true) {
                return Err(VetError::DuplicateCursor { worker: idx });
            }
            cursors.push((worker_table[slot as usize].clone(), seq));
        }

        let wire_nodes = nodes.len();
        let mut vetted = Self {
            worker_table,
            nodes,
            cursors,
            dropped_workers,
        };
        let pruned = vetted.prune_carrier_less();
        if pruned > 0 {
            debug!(
                pruned,
                remaining = vetted.nodes.len(),
                dropped_workers,
                "kv-bootstrap: dropped carrier-less nodes left by unknown workers",
            );
        }
        // Pruning can empty a node list the cold gate saw non-empty; see
        // `VetError::NothingUsable`.
        if vetted.nodes.is_empty() {
            return Err(VetError::NothingUsable {
                wire_nodes,
                dropped_workers,
            });
        }
        Ok(vetted)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::wire_worker;
    use super::super::WireWorker;
    use super::*;
    use crate::state::kv_events::Tiers;

    fn node(parent: Option<u32>, block_hash: i64, workers: Vec<u32>) -> SnapshotNode {
        SnapshotNode {
            parent,
            block_hash,
            workers,
            tiers: vec![],
        }
    }

    fn snapshot(workers: Vec<WireWorker>, nodes: Vec<SnapshotNode>) -> PeerSnapshot {
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 64,
            is_bigram: false,
            producer_ready: true,
            workers,
            cursors: vec![],
            nodes,
            empty_ranks: vec![],
        }
    }

    fn live(ids: &[(&str, u32)]) -> HashSet<KvWorkerId> {
        ids.iter()
            .map(|(u, r)| KvWorkerId::new((*u).to_string(), *r))
            .collect()
    }

    #[test]
    fn vet_rejects_unknown_format() {
        let mut snap = snapshot(vec![], vec![]);
        snap.format = SNAPSHOT_FORMAT + 1;
        let err = VettedSnapshot::from_wire(snap, &live(&[]), Some(64)).unwrap_err();
        assert_eq!(
            err,
            VetError::UnknownFormat {
                got: SNAPSHOT_FORMAT + 1,
                want: SNAPSHOT_FORMAT
            }
        );
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// A producer that reports not-ready is refused even when its tree holds
    /// nodes.
    #[test]
    fn vet_rejects_a_not_ready_producer_even_with_nodes() {
        let mut snap = snapshot(
            vec![wire_worker("http://w1:30000", 0)],
            vec![node(None, 7, vec![0])],
        );
        snap.producer_ready = false;
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://w1:30000", 0)]), Some(64))
            .unwrap_err();
        assert_eq!(err, VetError::ProducerCold);
        assert_eq!(err.outcome(), SnapshotOutcome::ColdPeer);
    }

    /// An out-of-range `parent` from a peer is refused.
    #[test]
    fn vet_rejects_out_of_range_parent_reference() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0]), node(Some(99), 2, vec![0])],
        );
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64))
            .expect_err("an out-of-range parent must be refused");
        assert_eq!(
            err,
            VetError::InvalidParentReference {
                index: 1,
                parent: 99
            },
        );
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// A forward `parent` reference is refused.
    #[test]
    fn vet_rejects_forward_parent_reference_on_the_wire() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(Some(1), 1, vec![0]), node(None, 2, vec![0])],
        );
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64))
            .expect_err("a forward parent must be refused");
        assert_eq!(
            err,
            VetError::InvalidParentReference {
                index: 0,
                parent: 1
            }
        );
    }

    /// A `tiers` list that does not pair with `workers` is refused, not
    /// repaired by filtering.
    #[test]
    fn vet_rejects_tiers_that_do_not_pair_with_workers() {
        let mut bad = node(None, 1, vec![0, 1]);
        bad.tiers = vec![Tiers::HOST.bits()]; // one entry for two carriers
        let snap = snapshot(
            vec![wire_worker("http://a", 0), wire_worker("http://b", 0)],
            vec![node(None, 7, vec![0]), bad],
        );
        let err =
            VettedSnapshot::from_wire(snap, &live(&[("http://a", 0), ("http://b", 0)]), Some(64))
                .expect_err("mispaired tiers must be refused, not repaired");
        assert_eq!(err, VetError::TierTableMismatch { index: 1 });
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// A snapshot that pruning empties after the cold gate is refused as
    /// `NothingUsable`.
    #[test]
    fn vet_rejects_snapshot_that_prunes_to_nothing() {
        let snap = snapshot(
            vec![wire_worker("http://rogue", 0)],
            vec![node(None, 1, vec![0]), node(Some(0), 2, vec![0])],
        );
        // The only carrier is a worker this replica has never discovered, so
        // every node becomes carrier-less and is pruned.
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://known", 0)]), Some(64))
            .expect_err("a snapshot that prunes to nothing must be refused");
        assert_eq!(
            err,
            VetError::NothingUsable {
                wire_nodes: 2,
                dropped_workers: 1
            },
        );
        // Retriable: the peer may discover our workers moments later.
        assert_eq!(err.outcome(), SnapshotOutcome::ColdPeer);
    }

    /// A snapshot with no cursor for any of the given ranks does not cover
    /// them.
    #[test]
    fn covers_any_is_false_when_the_peer_knows_none_of_our_ranks() {
        let mut snap = snapshot(
            vec![wire_worker("http://known", 0)],
            vec![node(None, 1, vec![0])],
        );
        snap.cursors = vec![(0, 7)];
        let vetted =
            VettedSnapshot::from_wire(snap, &live(&[("http://known", 0)]), Some(64)).unwrap();

        assert!(vetted.covers_any(&[KvWorkerId::new("http://known".into(), 0)]));
        assert!(
            !vetted.covers_any(&[KvWorkerId::new("http://other".into(), 0)]),
            "a snapshot with no cursor for our rank cannot bootstrap it",
        );
    }

    /// A producer that claims ready with an empty tree is refused.
    #[test]
    fn vet_rejects_snapshot_with_no_nodes_even_when_producer_claims_ready() {
        let snap = snapshot(vec![wire_worker("http://a", 0)], vec![]);
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap_err();
        assert_eq!(err, VetError::ProducerCold);
        assert_eq!(err.outcome(), SnapshotOutcome::ColdPeer);
    }

    #[test]
    fn vet_rejects_block_size_mismatch() {
        let snap = snapshot(vec![], vec![node(None, 1, vec![])]);
        let err = VettedSnapshot::from_wire(snap, &live(&[]), Some(32)).unwrap_err();
        assert_eq!(
            err,
            VetError::BlockSizeMismatch {
                peer: 64,
                local: 32
            }
        );
    }

    /// The producer's `is_bigram` stamp does not affect vetting.
    #[test]
    fn vet_ignores_the_producer_bigram_stamp() {
        let mut snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0])],
        );
        for stamp in [false, true] {
            snap.is_bigram = stamp;
            VettedSnapshot::from_wire(snap.clone(), &live(&[("http://a", 0)]), Some(64))
                .unwrap_or_else(|e| panic!("is_bigram={stamp} must not affect vetting: {e}"));
        }
    }

    /// Before any worker establishes a block size there is nothing to
    /// contradict the peer, so the snapshot is accepted.
    #[test]
    fn vet_accepts_when_local_block_size_unset() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0])],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), None).unwrap();
        assert_eq!(vetted.worker_table.len(), 1);
    }

    /// The core trust-boundary test: a peer naming a worker this replica has
    /// never discovered must not be able to introduce it.
    #[test]
    fn vet_drops_unknown_workers_and_remaps_carriers() {
        let snap = snapshot(
            vec![
                wire_worker("http://known", 0),
                wire_worker("http://rogue", 0),
                wire_worker("http://known", 1),
            ],
            vec![node(None, 100, vec![0, 1, 2]), node(Some(0), 200, vec![1])],
        );
        let vetted = VettedSnapshot::from_wire(
            snap,
            &live(&[("http://known", 0), ("http://known", 1)]),
            Some(64),
        )
        .unwrap();

        assert_eq!(vetted.dropped_workers, 1);
        assert_eq!(
            vetted.worker_table,
            vec![
                KvWorkerId::new("http://known".into(), 0),
                KvWorkerId::new("http://known".into(), 1),
            ],
        );
        // Wire indices 0 and 2 survive as 0 and 1; the rogue index vanishes.
        assert_eq!(vetted.nodes[0].workers, vec![0, 1]);
        // The node whose only carrier was the rogue worker is pruned; see
        // `prune_carrier_less`.
        assert_eq!(
            vetted.nodes.len(),
            1,
            "carrier-less leaf must be pruned, not retained as structure",
        );
    }

    /// Interior structure leading to a surviving carrier must be KEPT — pruning
    /// it would detach the carrier and lose the match entirely.
    #[test]
    fn vet_keeps_carrier_less_interior_nodes_on_a_live_path() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![
                node(None, 1, vec![]),     // carrier-less interior
                node(Some(0), 2, vec![]),  // carrier-less interior
                node(Some(1), 3, vec![0]), // the surviving carrier, at depth 3
            ],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap();
        assert_eq!(vetted.nodes.len(), 3, "path to a carrier must survive");
        assert_eq!(vetted.nodes[2].workers, vec![0]);
        // Parent links must still be backward references after any remap.
        for (i, rec) in vetted.nodes.iter().enumerate() {
            assert!(rec.parent.is_none_or(|p| (p as usize) < i));
        }
    }

    /// The graft-observability split: carried nodes are matchable, structure
    /// nodes are match paths only. And both accessors describe the SURVIVING
    /// population: a carrier nobody knows is already gone from the count.
    #[test]
    fn carrier_counts_split_carrying_nodes_from_kept_structure() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0), wire_worker("http://drained", 0)],
            vec![
                node(None, 1, vec![]),        // interior on a live path: kept structure
                node(Some(0), 2, vec![0, 1]), // carried (by both)
                node(Some(1), 3, vec![1]),    // carried only by the drained worker
            ],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap();
        // The drained carrier leaves the worker table; its exclusive node is
        // pruned as a carrier-less leaf, and the shared node keeps the live
        // worker as its only carrier.
        assert_eq!(vetted.worker_count(), 1);
        assert_eq!(vetted.dropped_workers(), 1);
        assert_eq!(vetted.carrier_counts(), (1, 1));
    }

    /// Pruning must remap parent indices, not just drop entries.
    #[test]
    fn prune_remaps_parent_indices() {
        let mut vetted = VettedSnapshot {
            worker_table: vec![KvWorkerId::new("http://a".into(), 0)],
            nodes: vec![
                node(None, 10, vec![]),     // 0: dead leaf, pruned
                node(None, 20, vec![]),     // 1: interior on a live path, kept -> 0
                node(Some(1), 30, vec![0]), // 2: carrier, kept -> 1
                node(Some(1), 40, vec![]),  // 3: dead leaf, pruned
            ],
            cursors: vec![],
            dropped_workers: 0,
        };
        assert_eq!(vetted.prune_carrier_less(), 2);
        assert_eq!(vetted.nodes.len(), 2);
        assert_eq!(vetted.nodes[0].block_hash, 20);
        assert_eq!(vetted.nodes[0].parent, None);
        assert_eq!(vetted.nodes[1].block_hash, 30);
        assert_eq!(
            vetted.nodes[1].parent,
            Some(0),
            "surviving child must point at its parent's NEW index",
        );
    }

    #[test]
    fn vet_drops_cursors_for_unknown_workers() {
        let mut snap = snapshot(
            vec![
                wire_worker("http://known", 0),
                wire_worker("http://rogue", 0),
            ],
            vec![node(None, 1, vec![0])],
        );
        snap.cursors = vec![(0, 42), (1, 99)];
        let vetted =
            VettedSnapshot::from_wire(snap, &live(&[("http://known", 0)]), Some(64)).unwrap();
        assert_eq!(
            vetted.cursors,
            vec![(KvWorkerId::new("http://known".into(), 0), 42)],
        );
    }

    /// Out-of-range carrier indices from a malformed peer are dropped.
    #[test]
    fn vet_ignores_out_of_range_carrier_indices() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0, 7])],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap();
        assert_eq!(vetted.nodes[0].workers, vec![0]);
    }

    /// A node whose only carrier holds no ranked tier is pruned, so it never
    /// reaches the tree.
    #[test]
    fn vet_prunes_nodes_carried_only_on_unranked_tiers() {
        let unranked: u8 = 1 << 7;
        let mut carried = node(Some(0), 2, vec![0]);
        carried.tiers = vec![Tiers::DEVICE.bits()];
        let mut future_tier_only = node(Some(1), 3, vec![0]);
        future_tier_only.tiers = vec![unranked];
        let mut root = node(None, 1, vec![0]);
        root.tiers = vec![Tiers::DEVICE.bits() | unranked];
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![root, carried, future_tier_only],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap();
        assert_eq!(
            vetted.node_count(),
            2,
            "a node whose only carrier holds no ranked tier must be pruned",
        );
        let tree = HashTree::new();
        assert_eq!(vetted.graft_into(&tree), Ok(2));
        assert_eq!(tree.node_count(), 2, "no carrier-less node may be grafted");
    }

    /// Every carrier on unranked tiers leaves nothing usable, which is refused
    /// like any other snapshot that prunes to nothing.
    #[test]
    fn vet_refuses_a_snapshot_carried_only_on_unranked_tiers() {
        let mut only = node(None, 1, vec![0]);
        only.tiers = vec![0];
        let snap = snapshot(vec![wire_worker("http://a", 0)], vec![only]);
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap_err();
        assert_eq!(
            err,
            VetError::NothingUsable {
                wire_nodes: 1,
                dropped_workers: 0
            },
        );
    }

    /// Two cursors for one worker make its watermark ambiguous.
    #[test]
    fn vet_rejects_duplicate_cursors_for_one_worker() {
        let mut snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0])],
        );
        snap.cursors = vec![(0, 5), (0, 100)];
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap_err();
        assert_eq!(err, VetError::DuplicateCursor { worker: 0 });
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// A worker table that names one live worker twice is malformed.
    #[test]
    fn vet_rejects_a_repeated_wire_identity() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0), wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0, 1])],
        );
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap_err();
        assert_eq!(err, VetError::DuplicateWorker { index: 1 });
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// Filtering to a subset of workers drops the others' carriers and
    /// cursors, prunes what that strands, and keeps a shared node with only
    /// the kept carrier.
    #[test]
    fn retain_workers_filters_to_pending_ranks_and_prunes_stranded_nodes() {
        let mut snap = snapshot(
            vec![
                wire_worker("http://pending", 0),
                wire_worker("http://done", 0),
            ],
            vec![
                node(None, 1, vec![0, 1]), // shared
                node(Some(0), 2, vec![1]), // only the filtered-out worker
            ],
        );
        snap.cursors = vec![(0, 3), (1, 4)];
        let ids = [("http://pending", 0), ("http://done", 0)];
        let mut vetted = VettedSnapshot::from_wire(snap, &live(&ids), Some(64)).unwrap();

        vetted.retain_workers(&live(&[("http://pending", 0)]));

        let pending = KvWorkerId::new("http://pending".into(), 0);
        let done = KvWorkerId::new("http://done".into(), 0);
        assert_eq!(
            vetted.node_count(),
            1,
            "the done rank's exclusive node is pruned"
        );
        assert_eq!(vetted.nodes[0].workers, vec![0]);
        assert_eq!(vetted.cursor_for(&pending), Some(3));
        assert_eq!(vetted.cursor_for(&done), None);
    }
}
