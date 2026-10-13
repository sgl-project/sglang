//! The tree's flat export and restore format: [`SnapshotNode`] and the `HashTree` snapshot pass.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};

use super::{add_tiers, now_millis, shard_of, tally_tiers, HashTree, KvWorkerId, NodeId};
use super::{TierCounts, Tiers, N_SHARDS, ROOT_ID};

/// One node of a tree snapshot, as produced by
/// [`HashTree::export_snapshot`] and consumed by
/// [`HashTree::restore_snapshot`].
///
/// Records are parent-linked by their **index in the snapshot's node list**,
/// not by block hash or by root-to-node hash path. The same block hash
/// legitimately occupies several tree positions (see the reverse-index
/// module docs), so a hash-keyed replay would land in `resolve_parent`'s
/// ambiguous branch and could graft a chain under the wrong node; a hash
/// path is exact but quadratic in depth. An index is exact and linear.
///
/// `workers` are indices into the snapshot's worker table, kept sorted so a
/// record's carrier list does not depend on hash-map iteration order.
/// Records are in dependency order (see [`Self::parent`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotNode {
    /// Index of this node's parent record, or `None` when the node hangs
    /// directly off the root. Always a backward reference — strictly less
    /// than this record's own index — so a single forward pass places every
    /// parent before its children.
    pub parent: Option<u32>,
    pub block_hash: i64,
    pub workers: Vec<u32>,
    /// Tier bits of each carrier, parallel to `workers` (`tiers[k]` describes
    /// `workers[k]`), as [`Tiers::bits`]. Empty means every carrier holds the
    /// node on device.
    ///
    /// That empty-means-device reading is what lets the snapshot format stay
    /// fixed across the tiering change in *both* directions: a producer that
    /// predates tiers omits the field and its carriers restore exactly as
    /// they did before, and a consumer that predates tiers ignores the field
    /// and reads the old meaning. `serde(default)` fills the gap.
    #[serde(default)]
    pub tiers: Vec<u8>,
}

impl SnapshotNode {
    /// Keep the carriers `map` accepts, renumbering each to what it returns,
    /// and keep their tier entries in lockstep.
    ///
    /// The only correct way to filter a record's carriers: `workers` and
    /// `tiers` are parallel by index, so filtering one alone silently re-pairs
    /// every later carrier with another carrier's tiers.
    ///
    /// Filtering cannot preserve a `tiers` list that does not pair with
    /// `workers` — the output is always well formed — so a caller holding
    /// untrusted records rejects that mismatch BEFORE filtering. On such a
    /// record a carrier with no tier entry is dropped, never padded onto
    /// device: padding would invent the device holding [`Tiers::from_bits`]
    /// refuses to.
    pub fn retain_carriers(&mut self, mut map: impl FnMut(u32) -> Option<u32>) {
        let tiered = !self.tiers.is_empty();
        let mut workers = Vec::with_capacity(self.workers.len());
        let mut tiers = Vec::with_capacity(if tiered { self.workers.len() } else { 0 });
        for (k, &w) in self.workers.iter().enumerate() {
            let bits = self.tiers.get(k).copied();
            if tiered && bits.is_none() {
                continue;
            }
            let Some(kept) = map(w) else {
                continue;
            };
            workers.push(kept);
            tiers.extend(bits);
        }
        self.workers = workers;
        self.tiers = tiers;
    }

    /// The tiers carrier `k` restores onto. Absent `tiers` reads as device;
    /// otherwise [`Tiers::from_bits`]. Empty means the carrier is skipped.
    fn carrier_tiers(&self, k: usize) -> Tiers {
        match self.tiers.get(k) {
            Some(&bits) => Tiers::from_bits(bits),
            None => Tiers::DEVICE,
        }
    }

    /// Whether restoring this record would leave at least one carrier on it.
    fn has_restorable_carrier(&self) -> bool {
        (0..self.workers.len()).any(|k| !self.carrier_tiers(k).is_empty())
    }
}

/// Why a [`HashTree::restore_snapshot`] was rejected.
///
/// Snapshot input is untrusted, so its shape is validated rather than
/// assumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreError {
    /// A record's `parent` pointed at itself or at a later record, so the
    /// list is not in dependency order and placement cannot be resolved.
    ForwardParentReference { index: usize },
    /// A record referenced a worker-table slot that does not exist.
    WorkerIndexOutOfRange { index: usize, worker: u32 },
    /// A record's `tiers` was neither empty nor the same length as its
    /// `workers`, so carriers cannot be paired with their tiers.
    TierTableMismatch { index: usize },
    /// A node could not be created because its parent vanished — a tree
    /// invariant violation, not a bad snapshot. Already logged by
    /// `create_child`.
    TreeInvariant { index: usize },
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForwardParentReference { index } => write!(
                f,
                "snapshot node {index} has a non-backward parent reference",
            ),
            Self::WorkerIndexOutOfRange { index, worker } => write!(
                f,
                "snapshot node {index} references out-of-range worker index {worker}",
            ),
            Self::TierTableMismatch { index } => write!(
                f,
                "snapshot node {index} has a tiers list that does not pair with its workers",
            ),
            Self::TreeInvariant { index } => write!(
                f,
                "snapshot node {index} could not be grafted: parent missing",
            ),
        }
    }
}

impl std::error::Error for RestoreError {}

/// Snapshot records one [`HashTree::export_snapshot`] or
/// [`HashTree::restore_snapshot`] pass holds a SHARD's lock for before
/// yielding it.
///
/// Sharding caps how much of the tree one lock covers, not how long a pass
/// over that much of it runs: an unchunked walk of one shard of a
/// fleet-sized tree still stalls every match and every insert rooted in that
/// shard for the walk's whole duration. The chunk bounds that hold at the
/// cost of one uncontended re-acquire per chunk.
const SNAPSHOT_LOCK_CHUNK: usize = 4096;

impl HashTree {
    /// Export every node as a flat, parent-linked list that
    /// [`HashTree::restore_snapshot`] can rebuild an identical tree from.
    ///
    /// Returns `(worker_table, nodes)`. Each shard is walked in DFS pre-order
    /// and the walks are concatenated, which yields the dependency order
    /// [`SnapshotNode::parent`] requires.
    ///
    /// No shard is recorded; [`HashTree::restore_snapshot`] re-derives it.
    ///
    /// # Cost
    ///
    /// Takes the shards one at a time, never holding two, and re-takes each
    /// shard's read lock every [`SNAPSHOT_LOCK_CHUNK`] records. The result is
    /// therefore not one consistent instant: a node pruned mid-walk is
    /// skipped along with its unvisited descendants.
    pub fn export_snapshot(&self) -> (Vec<KvWorkerId>, Vec<SnapshotNode>) {
        let mut worker_table: Vec<KvWorkerId> = Vec::new();
        let mut worker_index: HashMap<KvWorkerId, u32> = HashMap::new();
        let mut nodes: Vec<SnapshotNode> = Vec::with_capacity(self.node_count());
        // Reused across nodes so each record costs only its own two lists.
        let mut carriers: Vec<(u32, u8)> = Vec::new();
        for shard in &self.shards {
            // (node id, parent's index in `nodes`); `None` for the sentinel's
            // own children, which are chain roots.
            let mut stack: Vec<(NodeId, Option<u32>)> = {
                let st = shard.read();
                st.nodes
                    .get(&ROOT_ID)
                    .map(|root| root.children.values().map(|&id| (id, None)).collect())
                    .unwrap_or_default()
            };
            while !stack.is_empty() {
                let st = shard.read();
                for _ in 0..SNAPSHOT_LOCK_CHUNK {
                    let Some((id, parent_idx)) = stack.pop() else {
                        break;
                    };
                    let Some(node) = st.nodes.get(&id) else {
                        continue;
                    };
                    carriers.clear();
                    carriers.extend(node.workers.iter().map(|(w, tiers)| {
                        let idx = match worker_index.get(w) {
                            Some(&idx) => idx,
                            None => {
                                let idx = worker_table.len() as u32;
                                worker_table.push(w.clone());
                                worker_index.insert(w.clone(), idx);
                                idx
                            }
                        };
                        (idx, tiers.bits())
                    }));
                    // Deterministic order; tiers ride along.
                    carriers.sort_unstable();
                    let (workers, tiers): (Vec<u32>, Vec<u8>) = carriers.iter().copied().unzip();
                    // Pushed before its children.
                    let my_idx = nodes.len() as u32;
                    nodes.push(SnapshotNode {
                        parent: parent_idx,
                        block_hash: node.block_hash,
                        workers,
                        tiers,
                    });
                    for &child in node.children.values() {
                        stack.push((child, Some(my_idx)));
                    }
                }
            }
        }
        (worker_table, nodes)
    }

    /// Rebuild tree state from a snapshot.
    ///
    /// Grafts each record under its recorded parent: creates the node when
    /// absent, unions the carrier's tiers when present, so restoring onto a
    /// non-empty tree is well defined. A record with `parent == None` is a
    /// chain root and lands under the sentinel of `shard_of(block_hash)` —
    /// where [`HashTree::insert`] would have put it, so a restored tree routes
    /// identically to the tree it came from. Descendants inherit their
    /// parent's shard, so a restored chain still lives inside one shard.
    /// Snapshots carry no shard: re-deriving it rebuilds a chain that a
    /// racing prune had stranded under some other shard's sentinel (the
    /// re-rooting the module header describes) where `match_prefix` can
    /// reach it, instead of copying it stranded.
    ///
    /// A record is grafted only when it or a descendant leaves a carrier: an
    /// empty leaf would turn its parent's hit into a match with no workers,
    /// and nothing prunes it. Records that carry nothing arise from the tier
    /// filter.
    ///
    /// Returns the number of records applied.
    ///
    /// # Untrusted input
    ///
    /// The node list is untrusted, so its shape is validated up front rather
    /// than assumed: `parent` must be a backward reference, every worker
    /// index must be in range, and a `tiers` list must either be absent or
    /// pair one-to-one with `workers`. Validation happens before any
    /// mutation, so a rejected snapshot leaves the tree untouched.
    ///
    /// Carriers whose tiers come back empty from [`Tiers::from_bits`] are
    /// skipped. The tree's "no bits ⇒ no entry" invariant depends on that.
    ///
    /// `worker_table` must hold ids resolved against the local worker
    /// registry, NOT ids deserialized straight off the wire — see the
    /// provenance note on [`KvWorkerId`]. This method trusts the ids it is
    /// handed, which is why it is module-internal.
    ///
    /// # Locking
    ///
    /// Holds [`HashTree::writer`] for the whole pass and takes one shard's
    /// write lock at a time, releasing it every [`SNAPSHOT_LOCK_CHUNK`]
    /// records so a concurrent match on that shard waits at most a chunk.
    /// Holding `writer` across those gaps is what makes the chunking safe:
    /// the other writer (`KvEventIndex::remove_worker`, off the discovery
    /// task) cannot prune a parent that an earlier chunk placed and a later
    /// chunk is about to hang a child from.
    #[allow(dead_code)]
    pub(in crate::state::kv_events) fn restore_snapshot(
        &self,
        worker_table: &[KvWorkerId],
        nodes: &[SnapshotNode],
    ) -> Result<usize, RestoreError> {
        // Validate before mutating. The backward-reference check is what makes
        // the placement lookup below infallible; the bounds check keeps
        // malformed input from silently dropping cache carriers.
        for (i, rec) in nodes.iter().enumerate() {
            if rec.parent.is_some_and(|p| p as usize >= i) {
                return Err(RestoreError::ForwardParentReference { index: i });
            }
            if let Some(&worker) = rec
                .workers
                .iter()
                .find(|&&w| w as usize >= worker_table.len())
            {
                return Err(RestoreError::WorkerIndexOutOfRange { index: i, worker });
            }
            if !rec.tiers.is_empty() && rec.tiers.len() != rec.workers.len() {
                return Err(RestoreError::TierTableMismatch { index: i });
            }
        }

        // Which records leave a carrier in their subtree. Parents precede
        // children, so one reverse pass settles every record, and a kept
        // record's parent is always kept.
        let mut keep: Vec<bool> = vec![false; nodes.len()];
        for (i, rec) in nodes.iter().enumerate().rev() {
            keep[i] |= rec.has_restorable_carrier();
            if let (true, Some(p)) = (keep[i], rec.parent) {
                keep[p as usize] = true;
            }
        }

        // Route every record to a shard before touching any of them, then walk
        // shard by shard: one write-lock acquisition per chunk instead of one
        // per chain. A bucket keeps snapshot order, and a child shares its
        // parent's bucket, so a parent is still placed before any child reads
        // it.
        let mut rec_shard: Vec<u32> = Vec::with_capacity(nodes.len());
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); N_SHARDS];
        for (i, rec) in nodes.iter().enumerate() {
            let idx = match rec.parent {
                None => shard_of(rec.block_hash) as u32,
                Some(p) => rec_shard[p as usize],
            };
            if keep[i] {
                buckets[idx as usize].push(i as u32);
            }
            rec_shard.push(idx);
        }
        let applied = buckets.iter().map(Vec::len).sum();

        // Record index -> where it landed.
        let mut placed: Vec<NodeId> = vec![ROOT_ID; nodes.len()];
        // Occupancy is booked once per (chunk, carrier) rather than once per
        // node, the same amortisation `insert` does for a chain.
        let mut booked: Vec<TierCounts> = vec![TierCounts::default(); worker_table.len()];
        let now = now_millis();
        let _writer = self.writer.lock();
        for (shard_idx, bucket) in buckets.iter().enumerate() {
            for chunk in bucket.chunks(SNAPSHOT_LOCK_CHUNK) {
                booked.fill(TierCounts::default());
                let mut failed: Option<RestoreError> = None;
                let mut st = self.shards[shard_idx].write();
                for &ri in chunk {
                    let i = ri as usize;
                    let rec = &nodes[i];
                    let (parent_id, parent_block_hash) = match rec.parent {
                        None => (ROOT_ID, None),
                        Some(p) => (placed[p as usize], Some(nodes[p as usize].block_hash)),
                    };
                    let Some(id) = st.child_or_create(parent_id, rec.block_hash, parent_block_hash)
                    else {
                        failed = Some(RestoreError::TreeInvariant { index: i });
                        break;
                    };
                    if let Some(node) = st.nodes.get_mut(&id) {
                        // Only a fresh node: on a populated one `reserve`
                        // would grow capacity for carriers it already holds.
                        if node.workers.is_empty() {
                            node.workers.reserve(rec.workers.len());
                        }
                        for (k, &w) in rec.workers.iter().enumerate() {
                            let tiers = rec.carrier_tiers(k);
                            if tiers.is_empty() {
                                continue;
                            }
                            let added =
                                add_tiers(&mut node.workers, &worker_table[w as usize], tiers);
                            tally_tiers(&mut booked[w as usize], added);
                        }
                        node.last_used.store(now, Ordering::Relaxed);
                    }
                    placed[i] = id;
                }
                // Book before surfacing the error: the tier bits of the records
                // this chunk did apply are already in the nodes, so bailing
                // without booking would leave the occupancy gauge permanently
                // short by exactly those.
                for (worker, delta) in worker_table.iter().zip(&booked) {
                    st.account_add(worker, *delta);
                }
                drop(st);
                if let Some(e) = failed {
                    return Err(e);
                }
            }
        }
        Ok(applied)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::super::test_support::{worker, workers};
    use super::*;

    // -----------------------------------------------------------------------
    // Snapshot export / restore
    // -----------------------------------------------------------------------

    fn rec(
        parent: Option<u32>,
        block_hash: i64,
        workers: Vec<u32>,
        tiers: Vec<u8>,
    ) -> SnapshotNode {
        SnapshotNode {
            parent,
            block_hash,
            workers,
            tiers,
        }
    }

    /// Roots of the many-sibling-roots part of [`populated_tree`].
    fn sibling_roots() -> impl Iterator<Item = i64> {
        (0..32i64).map(|r| r * 4096 + 11)
    }

    /// Build a tree exercising the cases a real snapshot has to survive:
    /// multi-worker shared prefixes, divergent branches, many sibling roots,
    /// and the same block hash occupying more than one position.
    fn populated_tree() -> (HashTree, Vec<KvWorkerId>) {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        let c = worker("http://b", 1); // same url, different dp rank

        // Shared prefix, divergent tails.
        tree.insert(&a, None, &[1, 2, 3, 4]);
        tree.insert(&b, None, &[1, 2, 5, 6]);
        // Single-block chain.
        tree.insert(&c, None, &[7]);
        // Hash 2 reappears as a chain root elsewhere, and hash 3 as an
        // interior block of a different chain — the ambiguity that makes
        // hash-keyed replay wrong.
        tree.insert(&a, None, &[2, 3, 9]);
        // Many sibling roots.
        for root in sibling_roots() {
            tree.insert(&b, None, &[root, root + 1]);
        }
        (tree, vec![a, b, c])
    }

    /// The queries a restored tree must answer identically to its source.
    fn probe_queries() -> Vec<Vec<i64>> {
        let mut q = vec![
            vec![1],
            vec![1, 2],
            vec![1, 2, 3],
            vec![1, 2, 3, 4],
            vec![1, 2, 5],
            vec![1, 2, 5, 6],
            vec![7],
            vec![2],
            vec![2, 3],
            vec![2, 3, 9],
            vec![1, 2, 3, 4, 99],
            vec![404],
        ];
        for root in sibling_roots() {
            q.push(vec![root]);
            q.push(vec![root, root + 1]);
        }
        q
    }

    #[test]
    fn export_restore_round_trips_identically() {
        let (src, table) = populated_tree();
        let (worker_table, nodes) = src.export_snapshot();

        // The worker table must cover exactly the carriers in the tree.
        let exported: HashSet<KvWorkerId> = worker_table.iter().cloned().collect();
        assert_eq!(exported, table.iter().cloned().collect::<HashSet<_>>());

        let dst = HashTree::new();
        let applied = dst.restore_snapshot(&worker_table, &nodes).unwrap();
        assert_eq!(applied, nodes.len());

        assert_eq!(
            dst.node_count(),
            src.node_count(),
            "restored tree must have the same node count",
        );
        for q in probe_queries() {
            let want = src.match_prefix(None, &q);
            let got = dst.match_prefix(None, &q);
            assert_eq!(
                (got.matched_blocks, got.workers()),
                (want.matched_blocks, want.workers()),
                "match_prefix diverged for {q:?}",
            );
        }
    }

    /// A snapshot carries no shard column — [`HashTree::restore_snapshot`]
    /// re-derives each chain's shard from its root hash. Check that placement
    /// directly, shard for shard, on a fixture whose roots demonstrably span
    /// more than one shard: `match_prefix` parity alone would stop proving
    /// anything about placement the day the fixture's roots happened to
    /// collide into a single shard.
    #[test]
    fn restore_rebuilds_the_same_shard_layout() {
        let (src, _) = populated_tree();
        let spanned = sibling_roots()
            .map(shard_of)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(
            spanned > 1,
            "fixture roots must span shards for this to test anything, got {spanned}",
        );

        let (worker_table, nodes) = src.export_snapshot();
        let dst = HashTree::new();
        dst.restore_snapshot(&worker_table, &nodes).unwrap();

        for i in 0..N_SHARDS {
            assert_eq!(
                dst.shards[i].read().node_count(),
                src.shards[i].read().node_count(),
                "shard {i} holds a different number of nodes after restore",
            );
        }
    }

    /// Occupancy must survive the round trip too. It is booked incrementally
    /// per chunk, so a restore that placed the tier bits but skipped the
    /// booking would pass every `match_prefix` assertion above and still
    /// report the worker as publishing nothing on `/metrics`.
    #[test]
    fn export_restore_round_trips_occupancy() {
        let (src, _) = populated_tree();
        let (worker_table, nodes) = src.export_snapshot();
        let dst = HashTree::new();
        dst.restore_snapshot(&worker_table, &nodes).unwrap();

        assert_eq!(dst.tier_occupancy(), src.tier_occupancy());
        assert_eq!(dst.tier_occupancy(), dst.debug_recount_occupancy());
        assert_eq!(dst.accounting_errors(), [0, 0]);
    }

    /// A restore must be exact even when a carrier was dropped from an
    /// interior node but still holds a descendant — the state a `BlockRemoved`
    /// for a mid-chain hash produces, and the case a chain-replay through
    /// `insert` would silently "repair" by re-adding the ancestor.
    #[test]
    fn export_restore_preserves_interior_carrier_gaps() {
        let src = HashTree::new();
        let a = worker("http://a", 0);
        src.insert(&a, None, &[10, 20, 30]);
        // Drop the middle block only. Node 20 survives because it has a child.
        src.remove(&a, &[20]);
        assert!(!src.match_prefix(None, &[10, 20]).workers().contains(&a));

        let (worker_table, nodes) = src.export_snapshot();
        let dst = HashTree::new();
        dst.restore_snapshot(&worker_table, &nodes).unwrap();

        assert_eq!(dst.node_count(), src.node_count());
        for q in [vec![10], vec![10, 20], vec![10, 20, 30]] {
            let want = src.match_prefix(None, &q);
            let got = dst.match_prefix(None, &q);
            assert_eq!(
                (got.matched_blocks, got.workers()),
                (want.matched_blocks, want.workers()),
                "interior carrier gap not preserved for {q:?}",
            );
        }
    }

    #[test]
    fn export_of_empty_tree_is_empty() {
        let tree = HashTree::new();
        let (worker_table, nodes) = tree.export_snapshot();
        assert!(worker_table.is_empty());
        assert!(nodes.is_empty());

        let dst = HashTree::new();
        assert_eq!(dst.restore_snapshot(&worker_table, &nodes).unwrap(), 0);
        assert_eq!(dst.node_count(), 0);
    }

    #[test]
    fn export_emits_only_backward_parent_references() {
        let (src, _) = populated_tree();
        let (_, nodes) = src.export_snapshot();
        assert!(!nodes.is_empty());
        for (i, rec) in nodes.iter().enumerate() {
            if let Some(p) = rec.parent {
                assert!(
                    (p as usize) < i,
                    "record {i} references parent {p}, not a backward reference",
                );
            }
        }
    }

    /// Both passes release a shard's lock every `SNAPSHOT_LOCK_CHUNK` records,
    /// so the boundary is a real seam in the walk. A tree several chunks deep
    /// must round-trip across it, including the parent references that span
    /// two chunks.
    #[test]
    fn export_restore_round_trips_across_lock_chunks() {
        let src = HashTree::new();
        let a = worker("http://a", 0);
        // One deep chain, so nearly every record's parent is the record
        // before it and a chunk boundary always falls mid-chain.
        let deep: Vec<i64> = (1..=(SNAPSHOT_LOCK_CHUNK as i64 * 2 + 37)).collect();
        src.insert(&a, None, &deep);
        assert!(src.node_count() > SNAPSHOT_LOCK_CHUNK * 2);

        let (worker_table, nodes) = src.export_snapshot();
        assert_eq!(nodes.len(), src.node_count());
        let dst = HashTree::new();
        assert_eq!(
            dst.restore_snapshot(&worker_table, &nodes).unwrap(),
            nodes.len(),
        );

        assert_eq!(dst.node_count(), src.node_count());
        let m = dst.match_prefix(None, &deep);
        assert_eq!(m.matched_blocks, deep.len());
        assert_eq!(m.workers(), workers(&[&a]));
        assert_eq!(dst.tier_occupancy(), src.tier_occupancy());
    }

    #[test]
    fn restore_rejects_forward_parent_reference() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let nodes = vec![
            rec(Some(1), 1, vec![0], vec![]), // forward
            rec(None, 2, vec![0], vec![]),
        ];
        assert_eq!(
            tree.restore_snapshot(&[a], &nodes),
            Err(RestoreError::ForwardParentReference { index: 0 }),
        );
        // Rejected before any mutation.
        assert_eq!(tree.node_count(), 0);
    }

    #[test]
    fn restore_rejects_self_parent_reference() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let nodes = vec![rec(Some(0), 1, vec![0], vec![])];
        assert_eq!(
            tree.restore_snapshot(&[a], &nodes),
            Err(RestoreError::ForwardParentReference { index: 0 }),
        );
        assert_eq!(tree.node_count(), 0);
    }

    #[test]
    fn restore_rejects_out_of_range_worker_index() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let nodes = vec![
            rec(None, 1, vec![0], vec![]),
            rec(Some(0), 2, vec![7], vec![]), // table has one entry
        ];
        assert_eq!(
            tree.restore_snapshot(&[a], &nodes),
            Err(RestoreError::WorkerIndexOutOfRange {
                index: 1,
                worker: 7
            }),
        );
        assert_eq!(tree.node_count(), 0);
    }

    #[test]
    fn restore_rejects_tiers_that_do_not_pair_with_workers() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        // One tier entry for two carriers.
        let nodes = vec![rec(None, 1, vec![0, 1], vec![Tiers::HOST.bits()])];
        assert_eq!(
            tree.restore_snapshot(&[a, b], &nodes),
            Err(RestoreError::TierTableMismatch { index: 0 }),
        );
        assert_eq!(tree.node_count(), 0);
    }

    /// Tiers survive an export → restore round trip per carrier, so the
    /// restored tree ranks device and host owners the same way the source did.
    #[test]
    fn snapshot_round_trip_preserves_tiers_per_carrier() {
        let src = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        src.insert_tiered(&a, None, &[1, 2], Tiers::DEVICE);
        src.insert_tiered(&a, None, &[1, 2], Tiers::HOST);
        src.insert_tiered(&b, None, &[1, 2], Tiers::HOST);

        let (table, nodes) = src.export_snapshot();
        for n in &nodes {
            assert_eq!(n.tiers.len(), n.workers.len(), "tiers pair with workers");
        }
        let dst = HashTree::new();
        dst.restore_snapshot(&table, &nodes).unwrap();

        let m = dst.match_prefix(None, &[1, 2]);
        assert_eq!(m.workers(), workers(&[&a, &b]));
        assert_eq!(
            m.device_workers(),
            workers(&[&a]),
            "b was host-only in the source",
        );
    }

    /// A snapshot from a producer that predates tiers carries no `tiers`; its
    /// carriers restore as device owners, which is what its `workers` list
    /// meant when it was written.
    #[test]
    fn legacy_snapshot_without_tiers_restores_as_device() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let nodes = vec![rec(None, 1, vec![0], vec![])];
        tree.restore_snapshot(std::slice::from_ref(&a), &nodes)
            .unwrap();
        assert_eq!(
            tree.match_prefix(None, &[1]).device_workers(),
            workers(&[&a]),
        );
    }

    /// A carrier naming only tiers this build does not rank is dropped, not
    /// folded onto device. Folding would make a future tier's carrier a
    /// preferred device owner here — the same trade `Tiers::for_store` makes
    /// for an unknown `medium`, and for the same reason.
    #[test]
    fn restore_drops_a_carrier_that_names_no_known_tier() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        let unknown = !Tiers::ALL.bits(); // every bit this build has no name for
        let nodes = vec![rec(None, 1, vec![0, 1], vec![Tiers::HOST.bits(), unknown])];
        tree.restore_snapshot(&[a.clone(), b], &nodes).unwrap();

        let m = tree.match_prefix(None, &[1]);
        assert_eq!(m.workers(), workers(&[&a]), "b held no tier we rank");
        assert!(tree.debug_no_empty_carrier());
        assert_eq!(tree.accounting_errors(), [0, 0]);
    }

    /// A carrier holding a known tier AND an unknown one keeps the known half
    /// rather than being dropped outright.
    #[test]
    fn restore_keeps_the_known_half_of_a_mixed_tier_entry() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let mixed = Tiers::HOST.bits() | !Tiers::ALL.bits();
        let nodes = vec![rec(None, 1, vec![0], vec![mixed])];
        tree.restore_snapshot(std::slice::from_ref(&a), &nodes)
            .unwrap();

        let m = tree.match_prefix(None, &[1]);
        assert_eq!(m.workers(), workers(&[&a]));
        assert!(
            m.device_workers().is_empty(),
            "the unknown bit must not read as device",
        );
    }

    /// `retain_carriers` is the only correct way to filter a record's
    /// carriers: workers and tiers are parallel by index, so dropping a worker
    /// must drop its tier entry, not shift a neighbour's onto it.
    #[test]
    fn retain_carriers_keeps_tiers_aligned() {
        let mut node = rec(
            None,
            1,
            vec![0, 1, 2],
            vec![Tiers::DEVICE.bits(), Tiers::HOST.bits(), Tiers::ALL.bits()],
        );
        // Drop worker 1, renumber 2 → 1.
        node.retain_carriers(|w| match w {
            0 => Some(0),
            2 => Some(1),
            _ => None,
        });
        assert_eq!(node.workers, vec![0, 1]);
        assert_eq!(node.tiers, vec![Tiers::DEVICE.bits(), Tiers::ALL.bits()]);

        // A legacy record stays legacy: no tiers are invented.
        let mut legacy = rec(None, 1, vec![0, 1], vec![]);
        legacy.retain_carriers(|w| (w == 1).then_some(0));
        assert_eq!(legacy.workers, vec![0]);
        assert!(legacy.tiers.is_empty());
    }

    /// A tiers list shorter than its workers cannot be re-paired. Filtering
    /// must drop the carrier that has no entry rather than pad it onto
    /// device — padding would invent a device owner out of malformed input.
    #[test]
    fn retain_carriers_drops_a_carrier_with_no_tier_entry() {
        let mut node = rec(None, 1, vec![0, 1], vec![Tiers::HOST.bits()]);
        node.retain_carriers(Some);
        assert_eq!(node.workers, vec![0]);
        assert_eq!(node.tiers, vec![Tiers::HOST.bits()]);
    }

    /// Only records whose subtree leaves a carrier are grafted.
    #[test]
    fn restore_skips_structure_that_carries_nothing() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let unknown = !Tiers::ALL.bits();
        let nodes = vec![
            rec(None, 1, vec![0], vec![]),
            // Its only carrier names no tier this build ranks.
            rec(Some(0), 2, vec![0], vec![unknown]),
            rec(Some(1), 3, vec![], vec![]),
            // Carrier-less, but a carrying record hangs beneath it.
            rec(Some(0), 4, vec![], vec![]),
            rec(Some(3), 5, vec![0], vec![]),
        ];
        assert_eq!(
            tree.restore_snapshot(std::slice::from_ref(&a), &nodes)
                .unwrap(),
            3,
        );
        assert_eq!(tree.node_count(), 3);

        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 1);
        assert_eq!(m.workers(), workers(&[&a]));
        let m = tree.match_prefix(None, &[1, 4, 5]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));
        assert_eq!(tree.tier_occupancy(), tree.debug_recount_occupancy());
    }

    /// Restoring onto a tree that already holds live state must union
    /// carriers, not duplicate nodes.
    #[test]
    fn restore_onto_populated_tree_unions_carriers() {
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);

        let src = HashTree::new();
        src.insert(&a, None, &[1, 2, 3]);
        let (worker_table, nodes) = src.export_snapshot();

        let dst = HashTree::new();
        dst.insert(&b, None, &[1, 2, 3]);
        let before = dst.node_count();
        dst.restore_snapshot(&worker_table, &nodes).unwrap();

        assert_eq!(dst.node_count(), before, "restore must not duplicate nodes");
        let m = dst.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a, &b]));
        assert_eq!(dst.tier_occupancy(), dst.debug_recount_occupancy());
    }

    /// Re-restoring the same snapshot must not double-book occupancy: the
    /// second pass adds no tier bits, so `add_tiers` returns nothing and the
    /// chunk's booking stays at zero.
    #[test]
    fn restore_is_idempotent() {
        let (src, _) = populated_tree();
        let (worker_table, nodes) = src.export_snapshot();
        let dst = HashTree::new();
        dst.restore_snapshot(&worker_table, &nodes).unwrap();
        let after_first = dst.tier_occupancy();
        dst.restore_snapshot(&worker_table, &nodes).unwrap();

        assert_eq!(dst.node_count(), src.node_count());
        assert_eq!(dst.tier_occupancy(), after_first);
        assert_eq!(dst.tier_occupancy(), dst.debug_recount_occupancy());
    }

    /// The JSON wire tolerates a record written before `tiers` existed.
    #[test]
    fn snapshot_node_deserialises_without_tiers() {
        let rec: SnapshotNode =
            serde_json::from_str(r#"{"parent":null,"block_hash":7,"workers":[0]}"#).unwrap();
        assert_eq!(rec.block_hash, 7);
        assert_eq!(rec.workers, vec![0]);
        assert!(rec.tiers.is_empty());
    }
}
