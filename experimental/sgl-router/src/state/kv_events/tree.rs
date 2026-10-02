//! Hash-keyed radix tree for KV-cache event indexing.
//!
//! Each non-root node is one block hash (`i64`), with children keyed by the next
//! hash in the chain; each node records which [`KvWorkerId`]s hold the chain
//! ending there, and on which [`Tiers`]. Fed by SGLang `BlockStored` /
//! `BlockRemoved` / `AllBlocksCleared` events ([`super::wire`]),
//! queried via [`HashTree::match_prefix`].
//!
//! # Concurrency
//!
//! The tree is split into [`N_SHARDS`] [`TreeState`]s keyed by the chain's root
//! hash, so the event pump's per-event write lock blocks only one shard's reads.
//! [`FxHashMap`] on block hashes is safe not because they are unguessable
//! (they are unsalted SHA256, [`super::hash`]) but because planting one costs a
//! real inference request; worker-keyed maps keep SipHash.
//!
//! [`HashTree::route_insert`] / [`HashTree::route_match`] resolve a parent hash
//! across shards; `remove` / `clear_worker` / `evict_lru` fan out, and the
//! node cap is global. A writer landing between such a scan and its write could
//! re-root a chain where `match_prefix(None, ..)` never finds it, so
//! [`HashTree::writer`] serialises the two writers (the pump and
//! `KvEventIndex::remove_worker`); readers never take it.
//! [`HashTree::descend_match_shard`] re-checks its resolved parent instead,
//! falling back to `shard_of(block_hashes[0])`.
//!
//! # Reverse index
//!
//! `BlockRemoved` carries no parent, so [`TreeState::by_hash`] maps a hash to the
//! *set* of nodes carrying it; one hash can sit at several positions in a shard.
//!
//! # Pruning
//!
//! A node with no workers and no children is detached, cascading upward
//! iteratively since chains can be deep. A chain never crosses a shard.
//!
//! # Storage tiers
//!
//! A hierarchical-cache engine publishes each tier transition as its own event
//! tagged with a `medium`. Write-through publishes the host store before the
//! device eviction, write-back the inverse, so neither order may be assumed:
//! a removal clears only its own tier, and a later store re-adds the chain.
//!
//! A worker owns a block while it holds it on ANY tier, because a device
//! eviction that leaves a host copy still lets the worker load the prefix back
//! cheaply. [`MatchResult::tiers`] reports the tier per owner;
//! [`HashTree::prefix_depths`] is tier-blind by design.
//!
//! An untagged store is a device store and an untagged remove clears every tier.
//! A store with an unrankable `medium` is dropped, a removal with one clears
//! every tier. See [`Tiers`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use parking_lot::{Mutex, RwLock};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, error};

use super::pending::PendingPrefixes;

mod snapshot;

pub(super) use snapshot::ShapeViolation;
pub use snapshot::{RestoreError, SnapshotNode};

/// A power of two so `shard_of` selects with a shift; large enough that chains
/// rarely collide, small enough that a per-shard lock and arena are free.
const N_SHARDS: usize = 32;

// `shard_of` shifts by `64 - log2(N_SHARDS)` and indexes `shards`, both
// only correct for a power of two >= 2.
const _: () = assert!(
    N_SHARDS.is_power_of_two() && N_SHARDS >= 2,
    "N_SHARDS must be a power of two and at least 2",
);

/// Fibonacci-hashing constant. One worker emits many distinct chains;
/// mixing the root hash keeps that write load off a single shard.
const SHARD_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Map a chain-root block hash to its shard index, taken from the
/// best-mixed top bits of the multiplicative hash.
fn shard_of(root_hash: i64) -> usize {
    let mixed = (root_hash as u64).wrapping_mul(SHARD_MIX);
    (mixed >> (64 - N_SHARDS.trailing_zeros())) as usize
}

/// Epoch for the cheap millisecond timestamps in [`Node::last_used`].
static PROCESS_EPOCH: OnceLock<Instant> = OnceLock::new();

/// Milliseconds since [`PROCESS_EPOCH`]; the `u64` truncation is safe.
fn now_millis() -> u64 {
    PROCESS_EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// A worker endpoint refined by DP-attention rank: each rank publishes its own
/// event stream over a disjoint slice of the KV cache.
/// Distinct from the registry's UUID [`crate::core::worker_registry::WorkerId`].
///
/// Mint only inside kv_events, so `url` is always the registry's URL;
/// an id built from user input would let routing resolve to an unregistered
/// endpoint.
#[derive(Clone, Eq, Hash, PartialEq, Debug)]
pub struct KvWorkerId {
    pub url: String,
    pub dp_rank: u32,
}

impl KvWorkerId {
    /// Prefer over a struct literal, so provenance has a single chokepoint.
    pub fn new(url: String, dp_rank: u32) -> Self {
        Self { url, dp_rank }
    }
}

/// Bitset of the tiers on which one worker holds one block, since a backed-up
/// block sits on device and host at once. Any set bit makes it a routing owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Tiers(u8);

impl Tiers {
    /// Device HBM: `medium = "GPU"`, or an untagged event from a publisher
    /// that predates tiering.
    pub const DEVICE: Tiers = Tiers(1);
    /// Host pinned memory: `medium = "CPU_PINNED"`.
    pub const HOST: Tiers = Tiers(1 << 1);
    /// L3 local SSD / NVMe: `medium = "DISK"`.
    pub const DISK: Tiers = Tiers(1 << 2);
    /// L4 shared / remote pool, e.g. Mooncake: `medium = "EXTERNAL"`.
    /// Its own bit because the engine's `StorageMedium` keeps L3 and L4 apart;
    /// sharing one would let an L3 eviction erase an L4 copy.
    pub const EXTERNAL: Tiers = Tiers(1 << 3);
    /// Every tier — what an untagged `BlockRemoved` clears.
    pub const ALL: Tiers = Self::DEVICE
        .union(Self::HOST)
        .union(Self::DISK)
        .union(Self::EXTERNAL);
    /// Every tier with its metric label; [`TierCounts`] is indexed by position.
    /// Bit order is also preference order, cheapest load-back first.
    pub const SLOTS: [(Tiers, &'static str); TIER_SLOT_COUNT] = [
        (Self::DEVICE, "device"),
        (Self::HOST, "host"),
        (Self::DISK, "disk"),
        (Self::EXTERNAL, "external"),
    ];

    /// SGLang's wire `StorageMedium` strings
    /// (`python/sglang/srt/disaggregation/kv_events.py`) and their tiers;
    /// the single source for both the tree's ranking and the tally's labels.
    pub const WIRE_MEDIA: [(&'static str, Tiers); 4] = [
        ("GPU", Self::DEVICE),
        ("CPU_PINNED", Self::HOST),
        ("DISK", Self::DISK),
        ("EXTERNAL", Self::EXTERNAL),
    ];

    /// The tiers a `BlockStored` tagged `medium` lands on; untagged is device.
    /// An unknown tag lands on none, so the store is dropped: guessing a tier
    /// could price a remote fetch as cheap, while dropping costs one cold
    /// prefill and still shows on the tally's `unknown` row.
    pub fn for_store(medium: Option<&str>) -> Tiers {
        match medium {
            None => Self::DEVICE,
            Some(m) => Self::known(m).unwrap_or_default(),
        }
    }

    /// The tiers a `BlockRemoved` tagged `medium` clears: its own, or every
    /// tier when the tag is absent or unrecognised.
    /// Unlike [`Self::for_store`] it guesses wide, because over-removal costs
    /// one cold prefill while a permanently stale owner must never happen.
    pub fn for_remove(medium: Option<&str>) -> Tiers {
        medium.and_then(Self::known).unwrap_or(Self::ALL)
    }

    fn known(medium: &str) -> Option<Tiers> {
        Self::WIRE_MEDIA
            .iter()
            .find(|(name, _)| *name == medium)
            .map(|(_, tier)| *tier)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every bit of `other` is set here.
    pub const fn contains(self, other: Tiers) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Tiers) {
        self.0 |= other.0;
    }

    pub fn remove(&mut self, other: Tiers) {
        self.0 &= !other.0;
    }

    /// The bits set in both.
    pub const fn intersection(self, other: Tiers) -> Tiers {
        Tiers(self.0 & other.0)
    }

    /// The bits set here and not in `other`.
    pub const fn difference(self, other: Tiers) -> Tiers {
        Tiers(self.0 & !other.0)
    }

    /// The bits set in either; `const` so the tier tables can use it.
    pub const fn union(self, other: Tiers) -> Tiers {
        Tiers(self.0 | other.0)
    }

    /// Raw bits; a tier set crosses a process boundary only in [`SnapshotNode::tiers`].
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Read a [`SnapshotNode::tiers`] entry back, keeping only the tiers this
    /// build ranks. A restore is a store, so like [`Self::for_store`] unknown
    /// bits are dropped; an empty result (including zero) makes
    /// [`HashTree::restore_snapshot`] skip that carrier.
    pub const fn from_bits(bits: u8) -> Tiers {
        Tiers(bits & Self::ALL.0)
    }
}

/// Number of entries in [`Tiers::SLOTS`].
pub const TIER_SLOT_COUNT: usize = 4;

// Keeps the three tier tables in step when a tier is added: a bit missing from
// `ALL` is never cleared by an untagged remove (a stale owner), and one missing
// from `SLOTS` is invisible to the occupancy metric and its debug recount.
const _: () = {
    let mut union = Tiers(0);
    let mut i = 0;
    while i < TIER_SLOT_COUNT {
        union = union.union(Tiers::SLOTS[i].0);
        i += 1;
    }
    assert!(
        union.0 == Tiers::ALL.0,
        "Tiers::ALL must be exactly the union of Tiers::SLOTS",
    );
    let mut i = 0;
    while i < Tiers::WIRE_MEDIA.len() {
        assert!(
            Tiers::ALL.contains(Tiers::WIRE_MEDIA[i].1),
            "every wire medium must map to a tier that SLOTS ranks",
        );
        i += 1;
    }
};

/// How many nodes one carrier holds on each tier, indexed like
/// [`Tiers::SLOTS`]. A node held on device and host counts under both.
pub type TierCounts = [u64; TIER_SLOT_COUNT];

/// Count one node's worth of `bits` into `counts`.
fn tally_tiers(counts: &mut TierCounts, bits: Tiers) {
    for (slot, (tier, _)) in Tiers::SLOTS.iter().enumerate() {
        if bits.contains(*tier) {
            counts[slot] += 1;
        }
    }
}

/// Add `tiers` to `worker`'s hold and return the newly set bits.
/// Never leaves an entry with no bits, which [`TreeState::remove`] relies on.
fn add_tiers(
    carriers: &mut HashMap<KvWorkerId, Tiers>,
    worker: &KvWorkerId,
    tiers: Tiers,
) -> Tiers {
    match carriers.get_mut(worker) {
        Some(held) => {
            let added = tiers.difference(*held);
            held.insert(tiers);
            added
        }
        None => {
            carriers.insert(worker.clone(), tiers);
            tiers
        }
    }
}

/// Result of [`HashTree::match_prefix`]; `Default` is the no-match result.
#[derive(Debug, Clone, Default)]
pub struct MatchResult {
    /// Leading input hashes that matched a path from the root.
    pub matched_blocks: usize,
    /// Every worker holding the deepest matched node, with its tiers; empty when
    /// `matched_blocks == 0`. A worker without the device bit must load back.
    /// The single carrier list, so the derived views can never disagree.
    pub tiers: HashMap<KvWorkerId, Tiers>,
}

impl MatchResult {
    /// Whether `worker` holds the deepest matched node on any tier;
    /// unlike `workers().contains(..)` it clones no carrier.
    pub fn holds(&self, worker: &KvWorkerId) -> bool {
        self.tiers.contains_key(worker)
    }

    /// Workers holding the deepest matched node on ANY tier.
    pub fn workers(&self) -> HashSet<KvWorkerId> {
        self.tiers.keys().cloned().collect()
    }

    /// The subset of [`Self::workers`] holding it on device.
    pub fn device_workers(&self) -> HashSet<KvWorkerId> {
        self.tiers
            .iter()
            .filter(|(_, tiers)| tiers.contains(Tiers::DEVICE))
            .map(|(w, _)| w.clone())
            .collect()
    }
}

/// Stable arena key for a node, so whole-tree passes can enumerate nodes and
/// the reverse index has a stable key. Unique within a shard only.
type NodeId = u64;

/// One tree node. `last_used` (ms since [`PROCESS_EPOCH`]) is atomic so the
/// match path can bump it under a read lock; `Relaxed` suffices because
/// eviction needs only approximate freshness, with ties broken by [`NodeId`].
#[derive(Debug)]
struct Node {
    block_hash: i64,
    /// The parent hash the producing event named, kept even on a root-attached
    /// chain; diagnostic only (read by tests), the real link is [`Node::parent`].
    #[allow(dead_code)]
    parent_block_hash: Option<i64>,
    /// `None` only for the root sentinel.
    parent: Option<NodeId>,
    /// Carriers of the chain ending here, with their tiers;
    /// an entry with empty tiers never exists (see [`TreeState::insert`]).
    workers: HashMap<KvWorkerId, Tiers>,
    /// Children keyed by next-block hash.
    children: FxHashMap<i64, NodeId>,
    last_used: AtomicU64,
}

impl Node {
    fn new_child(block_hash: i64, parent_block_hash: Option<i64>, parent: NodeId) -> Self {
        Self {
            block_hash,
            parent_block_hash,
            parent: Some(parent),
            workers: HashMap::new(),
            children: FxHashMap::default(),
            last_used: AtomicU64::new(now_millis()),
        }
    }
}

/// Where one `insert` hangs its chain. Two fields because a forced root-attach
/// ([`HashTree::route_insert`]) must still record the parent hash as a
/// breadcrumb; one `Option` would silently drop it.
#[derive(Clone, Copy, Debug)]
struct ParentLink {
    /// Fed to [`TreeState::resolve_parent`]; `None` forces a root-attach.
    attach: Option<i64>,
    /// Recorded as the first node's `parent_block_hash`.
    breadcrumb: Option<i64>,
}

/// One shard's state. Invariants:
///
/// * `nodes[ROOT_ID]` always exists and is the only node with no parent.
/// * Every non-root node's parent maps its `block_hash` back to it.
/// * `by_hash[h]` holds every non-root node with hash `h`, never the root.
/// * A node emptied of workers and children is pruned at once.
#[derive(Debug)]
struct TreeState {
    nodes: FxHashMap<NodeId, Node>,
    by_hash: FxHashMap<i64, FxHashSet<NodeId>>,
    next_id: NodeId,
    /// Nodes each carrier holds, per tier, so a scrape need not walk the tree;
    /// a carrier holding nothing has no row.
    occupancy: HashMap<KvWorkerId, TierCounts>,
    /// Times occupancy bookkeeping contradicted itself, by reason; the only
    /// signal telling a booking bug apart from a worker that publishes nothing.
    accounting_errors: [u64; ACCOUNTING_REASONS.len()],
}

/// Reasons in [`TreeState::accounting_errors`] order.
pub const ACCOUNTING_REASONS: [&str; 2] = ["missing_row", "underflow"];
const REASON_MISSING_ROW: usize = 0;
const REASON_UNDERFLOW: usize = 1;

const ROOT_ID: NodeId = 0;
/// Sentinel block_hash for the root; a real `i64::MIN` cannot collide with it
/// because the root is never looked up via `by_hash`.
const ROOT_HASH_SENTINEL: i64 = i64::MIN;

impl TreeState {
    fn new() -> Self {
        let mut nodes = FxHashMap::default();
        nodes.insert(
            ROOT_ID,
            Node {
                block_hash: ROOT_HASH_SENTINEL,
                parent_block_hash: None,
                parent: None,
                workers: HashMap::new(),
                children: FxHashMap::default(),
                last_used: AtomicU64::new(now_millis()),
            },
        );
        Self {
            nodes,
            by_hash: FxHashMap::default(),
            next_id: 1,
            occupancy: HashMap::new(),
            accounting_errors: [0; ACCOUNTING_REASONS.len()],
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Book `delta` nodes-per-tier newly held by `worker`, one lookup per chain.
    fn account_add(&mut self, worker: &KvWorkerId, delta: TierCounts) {
        if delta.iter().all(|&c| c == 0) {
            return;
        }
        match self.occupancy.get_mut(worker) {
            Some(counts) => {
                for (acc, d) in counts.iter_mut().zip(delta) {
                    *acc += d;
                }
            }
            None => {
                self.occupancy.insert(worker.clone(), delta);
            }
        }
    }

    /// Book one node's worth of `removed` tier bits `worker` no longer holds.
    fn account_remove(&mut self, worker: &KvWorkerId, removed: Tiers) {
        let mut delta = TierCounts::default();
        tally_tiers(&mut delta, removed);
        self.account_remove_counts(worker, delta);
    }

    /// Book `delta` nodes-per-tier `worker` no longer holds, dropping its row at
    /// zero. Takes a whole-carrier delta so `clear_worker` logs a failure once,
    /// not once per node while holding the write lock.
    fn account_remove_counts(&mut self, worker: &KvWorkerId, delta: TierCounts) {
        if delta.iter().all(|&c| c == 0) {
            return;
        }
        let Some(counts) = self.occupancy.get_mut(worker) else {
            self.accounting_errors[REASON_MISSING_ROW] += 1;
            error!(
                worker = %worker.url,
                dp_rank = worker.dp_rank,
                "tree invariant violation: releasing tiers for a carrier with no occupancy row",
            );
            return;
        };
        let mut underflowed = false;
        for (acc, d) in counts.iter_mut().zip(delta) {
            underflowed |= *acc < d;
            *acc = acc.saturating_sub(d);
        }
        if underflowed {
            // Saturating can drop the row of a carrier that still holds nodes,
            // so count it in release builds and stop debug runs here.
            self.accounting_errors[REASON_UNDERFLOW] += 1;
            debug_assert!(
                false,
                "occupancy underflow: a tier was released that was never booked",
            );
        }
        if counts.iter().all(|&c| c == 0) {
            self.occupancy.remove(worker);
        }
    }

    /// Insert a new child under `parent_id`, overwriting any existing slot.
    /// A missing parent logs and returns `None`: a panic would kill the pump.
    fn create_child(
        &mut self,
        parent_id: NodeId,
        block_hash: i64,
        parent_block_hash: Option<i64>,
    ) -> Option<NodeId> {
        let id = self.alloc_id();
        self.nodes.insert(
            id,
            Node::new_child(block_hash, parent_block_hash, parent_id),
        );
        let Some(parent) = self.nodes.get_mut(&parent_id) else {
            error!(
                parent_id,
                block_hash,
                "tree invariant violation: create_child called with unknown parent_id; discarding new node",
            );
            self.nodes.remove(&id);
            return None;
        };
        parent.children.insert(block_hash, id);
        self.by_hash.entry(block_hash).or_default().insert(id);
        Some(id)
    }

    /// The child of `parent_id` keyed by `block_hash`, created when absent.
    /// `None` only when `parent_id` itself is missing.
    fn child_or_create(
        &mut self,
        parent_id: NodeId,
        block_hash: i64,
        parent_block_hash: Option<i64>,
    ) -> Option<NodeId> {
        let existing = self
            .nodes
            .get(&parent_id)
            .and_then(|n| n.children.get(&block_hash).copied());
        existing.or_else(|| self.create_child(parent_id, block_hash, parent_block_hash))
    }

    /// Pick the parent node for a `BlockStored` in this shard: the sole node
    /// carrying `parent_hash`, else one `worker` already holds, else the root.
    /// [`HashTree::route_insert`] has already decided globally, so the root
    /// fallback here only guards a routing/local disagreement.
    fn resolve_parent(&self, worker: &KvWorkerId, parent_hash: Option<i64>) -> NodeId {
        let Some(parent_hash) = parent_hash else {
            return ROOT_ID;
        };
        let Some(candidates) = self.by_hash.get(&parent_hash) else {
            debug!(
                worker = %worker.url,
                dp_rank = worker.dp_rank,
                parent_hash,
                "parent_hash not in tree; attaching new chain to root",
            );
            return ROOT_ID;
        };
        if candidates.len() == 1 {
            return *candidates.iter().next().unwrap();
        }
        // Multiple candidates — prefer one this worker already holds.
        for &cand in candidates {
            if self
                .nodes
                .get(&cand)
                .is_some_and(|n| n.workers.contains_key(worker))
            {
                return cand;
            }
        }
        debug!(
            worker = %worker.url,
            dp_rank = worker.dp_rank,
            parent_hash,
            n_candidates = candidates.len(),
            "ambiguous parent_hash with no worker-owned candidate; attaching to root",
        );
        ROOT_ID
    }

    /// Add `tiers` to `worker`'s hold along `block_hashes`. Empty `tiers` is a
    /// no-op, since `remove` prunes on "no bits means no entry".
    fn insert(
        &mut self,
        worker: &KvWorkerId,
        parent: ParentLink,
        block_hashes: &[i64],
        tiers: Tiers,
    ) {
        if block_hashes.is_empty() || tiers.is_empty() {
            return;
        }
        let mut current = self.resolve_parent(worker, parent.attach);
        let mut prev_hash = parent.breadcrumb;
        let now = now_millis();
        // Occupancy is booked once for the whole chain, not per block.
        let mut delta = TierCounts::default();
        for &h in block_hashes {
            // `break`, not `return`: the nodes visited so far still need
            // booking in `account_add` below.
            let Some(child_id) = self.child_or_create(current, h, prev_hash) else {
                break;
            };
            let Some(child) = self.nodes.get_mut(&child_id) else {
                error!(
                    child_id,
                    block_hash = h,
                    "tree invariant violation: child node missing immediately after fetch/create; aborting chain",
                );
                break;
            };
            let added = add_tiers(&mut child.workers, worker, tiers);
            child.last_used.store(now, Ordering::Relaxed);
            tally_tiers(&mut delta, added);
            current = child_id;
            prev_hash = Some(h);
        }
        self.account_add(worker, delta);
    }

    /// Whether this shard carries any of `block_hashes`, so a removal can skip
    /// it under a read lock.
    fn carries_any(&self, block_hashes: &[i64]) -> bool {
        block_hashes.iter().any(|h| self.by_hash.contains_key(h))
    }

    /// Clear `tiers` from `worker`'s hold on every node carrying any of
    /// `block_hashes`, pruning nodes left empty and childless.
    fn remove(&mut self, worker: &KvWorkerId, block_hashes: &[i64], tiers: Tiers) {
        // Snapshot the ids first: pruning mutates `by_hash`.
        let mut targets: Vec<NodeId> = Vec::new();
        for h in block_hashes {
            if let Some(set) = self.by_hash.get(h) {
                targets.extend(set.iter().copied());
            }
        }
        for id in targets {
            // An earlier prune in this batch may already have removed it.
            let (prunable, released) = match self.nodes.get_mut(&id) {
                Some(node) => {
                    let mut released = Tiers::default();
                    if let Some(held) = node.workers.get_mut(worker) {
                        released = held.intersection(tiers);
                        held.remove(tiers);
                        if held.is_empty() {
                            node.workers.remove(worker);
                        }
                    }
                    (
                        node.workers.is_empty() && node.children.is_empty(),
                        released,
                    )
                }
                None => (false, Tiers::default()),
            };
            self.account_remove(worker, released);
            if prunable {
                self.prune_cascade(id);
            }
        }
    }

    fn clear_worker(&mut self, worker: &KvWorkerId) {
        // Snapshot ids before mutation.
        let ids: Vec<NodeId> = self
            .nodes
            .keys()
            .copied()
            .filter(|&id| id != ROOT_ID)
            .collect();
        let mut prune_candidates: Vec<NodeId> = Vec::new();
        // Booked once for every node; see `account_remove_counts`.
        let mut delta = TierCounts::default();
        for id in ids {
            let removed = self.nodes.get_mut(&id).and_then(|node| {
                node.workers
                    .remove(worker)
                    .map(|held| (held, node.workers.is_empty() && node.children.is_empty()))
            });
            if let Some((held, prunable)) = removed {
                tally_tiers(&mut delta, held);
                if prunable {
                    prune_candidates.push(id);
                }
            }
        }
        self.account_remove_counts(worker, delta);
        for id in prune_candidates {
            // A sibling's cascading prune may already have removed it.
            if self.nodes.contains_key(&id) {
                self.prune_cascade(id);
            }
        }
    }

    /// Prune `start` and every ancestor it empties; iterative, since chains can be long.
    fn prune_cascade(&mut self, start: NodeId) {
        let mut cursor = start;
        loop {
            if cursor == ROOT_ID {
                return;
            }
            // Peek at the node before removal so we know its parent + hash.
            let (parent_id, block_hash) = match self.nodes.get(&cursor) {
                Some(n) => match n.parent {
                    Some(p) => (p, n.block_hash),
                    None => {
                        error!(
                            cursor,
                            "tree invariant violation: non-root node has no parent; aborting prune",
                        );
                        return;
                    }
                },
                None => return,
            };
            // Confirm prune precondition (cheap defensive check).
            let prunable = self
                .nodes
                .get(&cursor)
                .map(|n| n.workers.is_empty() && n.children.is_empty())
                .unwrap_or(false);
            if !prunable {
                return;
            }
            // Detach from parent's children map.
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.children.remove(&block_hash);
            }
            // Remove from reverse index.
            if let Some(set) = self.by_hash.get_mut(&block_hash) {
                set.remove(&cursor);
                if set.is_empty() {
                    self.by_hash.remove(&block_hash);
                }
            }
            // Drop the node itself.
            self.nodes.remove(&cursor);
            // Walk up.
            cursor = parent_id;
            // Stop unless the parent is now also empty + childless.
            let parent_prunable = self
                .nodes
                .get(&cursor)
                .map(|n| cursor != ROOT_ID && n.workers.is_empty() && n.children.is_empty())
                .unwrap_or(false);
            if !parent_prunable {
                return;
            }
        }
    }

    /// Takes `&self` so routing holds only a read lock. Unlike
    /// [`TreeState::resolve_parent`] it has no worker context,
    /// so an ambiguous `parent_hash` falls back to root.
    fn match_prefix(&self, parent_hash: Option<i64>, block_hashes: &[i64]) -> MatchResult {
        if block_hashes.is_empty() {
            return MatchResult::default();
        }
        // Start at the unique node carrying `parent_hash`, else the root.
        let start = match parent_hash {
            None => ROOT_ID,
            Some(p) => match self.by_hash.get(&p) {
                Some(set) if set.len() == 1 => *set.iter().next().unwrap(),
                _ => ROOT_ID,
            },
        };

        let mut current = start;
        let mut matched = 0usize;
        let now = now_millis();
        for &h in block_hashes {
            let next = self
                .nodes
                .get(&current)
                .and_then(|n| n.children.get(&h).copied());
            match next {
                Some(child_id) => {
                    // Touch as we descend; atomic, so no `&mut` needed.
                    if let Some(child) = self.nodes.get(&child_id) {
                        child.last_used.store(now, Ordering::Relaxed);
                    }
                    current = child_id;
                    matched += 1;
                }
                None => break,
            }
        }
        // With nothing matched `current` is still `start`,
        // whose carriers belong to the caller's parent and must not be reported.
        let tiers: HashMap<KvWorkerId, Tiers> = (matched > 0)
            .then(|| self.nodes.get(&current))
            .flatten()
            .map(|n| n.workers.clone())
            .unwrap_or_default();
        MatchResult {
            matched_blocks: matched,
            tiers,
        }
    }

    /// Contiguous prefix depth for every worker on the chain, in one descent.
    /// A worker is frozen at the first level that omits it,
    /// so a removed interior node stops the count at the hole.
    fn prefix_depths(
        &self,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
    ) -> HashMap<KvWorkerId, usize> {
        if block_hashes.is_empty() {
            return HashMap::new();
        }
        // Same start resolution as `match_prefix`.
        let start = match parent_hash {
            None => ROOT_ID,
            Some(p) => match self.by_hash.get(&p) {
                Some(set) if set.len() == 1 => *set.iter().next().unwrap(),
                _ => ROOT_ID,
            },
        };
        // Keys borrow the arena for the walk; ids are cloned once on the way out.
        let mut depths: HashMap<&KvWorkerId, usize> = HashMap::new();
        let mut alive: Vec<&KvWorkerId> = Vec::new();
        let mut current = start;
        let mut reached = 0usize;
        let now = now_millis();
        for &h in block_hashes {
            let Some(child_id) = self
                .nodes
                .get(&current)
                .and_then(|n| n.children.get(&h).copied())
            else {
                break;
            };
            let Some(child) = self.nodes.get(&child_id) else {
                break;
            };
            child.last_used.store(now, Ordering::Relaxed);
            current = child_id;
            reached += 1;
            if reached == 1 {
                // A tail held without block 0 counts as nothing,
                // and presence on any tier holds the level.
                alive = child.workers.keys().collect();
            } else {
                let mut still = Vec::with_capacity(alive.len());
                for w in alive {
                    if child.workers.contains_key(w) {
                        still.push(w);
                    } else {
                        depths.insert(w, reached - 1);
                    }
                }
                alive = still;
            }
            if alive.is_empty() {
                break;
            }
        }
        // Whoever is still tracked held every level the walk reached.
        for w in alive {
            depths.insert(w, reached);
        }
        depths.into_iter().map(|(w, d)| (w.clone(), d)).collect()
    }

    /// Count of *non-root* nodes in this shard.
    fn node_count(&self) -> usize {
        // Subtract one for the root sentinel.
        self.nodes.len().saturating_sub(1)
    }

    /// Drop already-empty leaves left by pruning races, returning the number of
    /// nodes dropped including cascaded ancestors.
    fn drop_empty_leaves(&mut self) -> usize {
        let count_before = self.nodes.len();
        let empty_leaves: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|(&id, n)| {
                if id != ROOT_ID && n.workers.is_empty() && n.children.is_empty() {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();
        for id in empty_leaves {
            if self.nodes.contains_key(&id) {
                self.prune_cascade(id);
            }
        }
        count_before - self.nodes.len()
    }

    /// Timestamp and shard-local id of the LRU leaf, for global eviction
    /// ordering; millisecond ties break by `NodeId`.
    fn lru_leaf(&self) -> Option<(u64, NodeId)> {
        let mut oldest: Option<(u64, NodeId)> = None;
        for (&id, n) in &self.nodes {
            if id == ROOT_ID || !n.children.is_empty() {
                continue;
            }
            let ts = n.last_used.load(Ordering::Relaxed);
            match oldest {
                None => oldest = Some((ts, id)),
                Some((cur_ts, cur_id)) if (ts, id) < (cur_ts, cur_id) => oldest = Some((ts, id)),
                _ => {}
            }
        }
        oldest
    }

    /// Force-evict leaf `victim` and cascade-prune, returning the nodes removed;
    /// 0 if it raced away. Carriers are drained through `account_remove`
    /// so occupancy stays in step even though they still hold the block.
    fn evict_leaf(&mut self, victim: NodeId) -> usize {
        let is_leaf = self
            .nodes
            .get(&victim)
            .map(|n| n.children.is_empty())
            .unwrap_or(false);
        if !is_leaf {
            return 0;
        }
        let count_before = self.nodes.len();
        let carriers: Vec<(KvWorkerId, Tiers)> = match self.nodes.get_mut(&victim) {
            Some(node) => node.workers.drain().collect(),
            None => Vec::new(),
        };
        for (worker, held) in &carriers {
            self.account_remove(worker, *held);
        }
        self.prune_cascade(victim);
        count_before - self.nodes.len()
    }
}

/// Public hash-keyed radix tree, `Send + Sync` and shared behind an [`Arc`].
/// See the module docs for sharding and [`Self::writer`].
#[derive(Debug)]
pub struct HashTree {
    shards: Vec<RwLock<TreeState>>,
    /// Held for every whole mutation, so a cross-shard scan and the write it
    /// feeds are atomic against the other writer. Readers never take it.
    writer: Mutex<()>,
    /// Route-time predictions, kept apart from the event-driven shards.
    pending: PendingPrefixes,
}

impl Default for HashTree {
    fn default() -> Self {
        Self::new()
    }
}

impl HashTree {
    pub fn new() -> Self {
        let mut shards = Vec::with_capacity(N_SHARDS);
        for _ in 0..N_SHARDS {
            shards.push(RwLock::new(TreeState::new()));
        }
        Self {
            shards,
            writer: Mutex::new(()),
            pending: PendingPrefixes::default(),
        }
    }

    pub fn pending(&self) -> &PendingPrefixes {
        &self.pending
    }

    /// Resolve an `insert`'s `parent_hash` over every shard's carriers, like
    /// [`TreeState::resolve_parent`] but globally, returning the target shard
    /// and [`ParentLink`]. `block_hashes` must be non-empty.
    ///
    /// An ambiguous parent with no `worker`-owned carrier must attach `None`:
    /// a shard holding one local copy would otherwise attach under a node the
    /// global decision rejected. The breadcrumb stays `Some(p)` regardless.
    fn route_insert(
        &self,
        worker: &KvWorkerId,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
    ) -> (usize, ParentLink) {
        let root_shard = shard_of(block_hashes[0]);
        let Some(p) = parent_hash else {
            return (
                root_shard,
                ParentLink {
                    attach: None,
                    breadcrumb: None,
                },
            );
        };
        // How many nodes carry `p`, and which shard holds one `worker` owns.
        let mut total_carriers = 0usize;
        let mut single_carrier_shard: Option<usize> = None;
        let mut worker_owned_shard: Option<usize> = None;
        for (idx, shard) in self.shards.iter().enumerate() {
            let st = shard.read();
            let Some(ids) = st.by_hash.get(&p) else {
                continue;
            };
            // An empty set should not exist, but would nominate this shard
            // and root the chain where `match_prefix(None, ..)` cannot reach it.
            if ids.is_empty() {
                continue;
            }
            total_carriers += ids.len();
            single_carrier_shard = Some(idx);
            if worker_owned_shard.is_none()
                && ids.iter().any(|id| {
                    st.nodes
                        .get(id)
                        .is_some_and(|n| n.workers.contains_key(worker))
                })
            {
                worker_owned_shard = Some(idx);
            }
        }
        let breadcrumb = Some(p);
        match total_carriers {
            // No shard can resolve `p`, so it root-attaches inside the shard.
            0 => (
                root_shard,
                ParentLink {
                    attach: breadcrumb,
                    breadcrumb,
                },
            ),
            1 => (
                single_carrier_shard.unwrap_or(root_shard),
                ParentLink {
                    attach: breadcrumb,
                    breadcrumb,
                },
            ),
            _ => match worker_owned_shard {
                Some(idx) => (
                    idx,
                    ParentLink {
                        attach: breadcrumb,
                        breadcrumb,
                    },
                ),
                // Forced root-attach, breadcrumb kept.
                None => (
                    root_shard,
                    ParentLink {
                        attach: None,
                        breadcrumb,
                    },
                ),
            },
        }
    }

    /// Resolve the match start across shards as `(shard, effective_parent)`;
    /// only a unique carrier of `p` is honored, else the root shard and `None`.
    /// Advisory once the locks drop; [`Self::descend_match_shard`] re-checks it.
    fn route_match(&self, parent_hash: Option<i64>, block_hashes: &[i64]) -> (usize, Option<i64>) {
        let root_shard = shard_of(block_hashes[0]);
        let Some(p) = parent_hash else {
            return (root_shard, None);
        };
        let mut total = 0usize;
        let mut only_shard: Option<usize> = None;
        for (idx, shard) in self.shards.iter().enumerate() {
            if let Some(ids) = shard.read().by_hash.get(&p).filter(|ids| !ids.is_empty()) {
                total += ids.len();
                only_shard = Some(idx);
                if total > 1 {
                    return (root_shard, None);
                }
            }
        }
        match total {
            1 => (only_shard.unwrap_or(root_shard), Some(p)),
            _ => (root_shard, None),
        }
    }

    /// Descend the shard that can answer for `parent_hash`, re-checking
    /// [`Self::route_match`]'s resolution under the descent's own read lock.
    /// A write in between can prune `p`, and falling back to that shard's
    /// sentinel would match nothing; a stale resolution falls back to
    /// `shard_of(block_hashes[0])` instead. A carrier newly in another shard
    /// is not re-checked.
    fn descend_match_shard<R>(
        &self,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
        descend: impl Fn(&TreeState, Option<i64>) -> R,
    ) -> R {
        let (idx, effective_parent) = self.route_match(parent_hash, block_hashes);
        self.descend_routed(idx, effective_parent, block_hashes, descend)
    }

    /// The post-scan half of [`Self::descend_match_shard`], split out so a test
    /// can hand it a stale resolution that only a racing writer can produce.
    fn descend_routed<R>(
        &self,
        idx: usize,
        effective_parent: Option<i64>,
        block_hashes: &[i64],
        descend: impl Fn(&TreeState, Option<i64>) -> R,
    ) -> R {
        if let Some(p) = effective_parent {
            let shard = self.shards[idx].read();
            if shard.by_hash.get(&p).is_some_and(|ids| ids.len() == 1) {
                return descend(&shard, Some(p));
            }
        }
        descend(&self.shards[shard_of(block_hashes[0])].read(), None)
    }

    /// Apply an untagged `BlockStored` event: a device store.
    pub fn insert(&self, worker: &KvWorkerId, parent_hash: Option<i64>, block_hashes: &[i64]) {
        self.insert_tiered(worker, parent_hash, block_hashes, Tiers::DEVICE);
    }

    /// Apply a `BlockStored` event on `tiers` (from [`Tiers::for_store`]),
    /// adding them to `worker`'s hold along the chain. Empty input is a no-op.
    pub fn insert_tiered(
        &self,
        worker: &KvWorkerId,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
        tiers: Tiers,
    ) {
        if block_hashes.is_empty() || tiers.is_empty() {
            return;
        }
        // The scan and the write it feeds must be one critical section;
        // a prune in between would orphan the chain.
        let _writer = self.writer.lock();
        let (idx, parent) = self.route_insert(worker, parent_hash, block_hashes);
        self.shards[idx]
            .write()
            .insert(worker, parent, block_hashes, tiers);
    }

    /// Apply an untagged `BlockRemoved` event: the worker loses every tier.
    pub fn remove(&self, worker: &KvWorkerId, block_hashes: &[i64]) {
        self.remove_tiered(worker, block_hashes, Tiers::ALL);
    }

    /// Apply a `BlockRemoved` event for `tiers` (from [`Tiers::for_remove`]).
    /// The worker stays an owner while it holds any other tier.
    ///
    /// Fans out, since a hash can sit in several shards; a shard carrying none
    /// is skipped under a read lock, safe because [`Self::writer`] is held.
    pub fn remove_tiered(&self, worker: &KvWorkerId, block_hashes: &[i64], tiers: Tiers) {
        // Empty `tiers` would write-lock every carrying shard for nothing.
        if block_hashes.is_empty() || tiers.is_empty() {
            return;
        }
        let _writer = self.writer.lock();
        for shard in &self.shards {
            if !shard.read().carries_any(block_hashes) {
                continue;
            }
            shard.write().remove(worker, block_hashes, tiers);
        }
    }

    /// Apply an `AllBlocksCleared` event for `worker`. Also the scale-down path,
    /// which runs off the pump task, hence [`Self::writer`].
    pub fn clear_worker(&self, worker: &KvWorkerId) {
        let _writer = self.writer.lock();
        for shard in &self.shards {
            shard.write().clear_worker(worker);
        }
    }

    /// Longest match of a prefix of `block_hashes`, from the root or from the
    /// node carrying `parent_hash`. Touches `last_used` along the path,
    /// keeping it hot for [`HashTree::evict_lru`], under one shard's read lock.
    ///
    /// A `parent_hash` carried by several nodes cannot be disambiguated
    /// without a worker, so matching falls back to the root.
    pub fn match_prefix(&self, parent_hash: Option<i64>, block_hashes: &[i64]) -> MatchResult {
        if block_hashes.is_empty() {
            return MatchResult::default();
        }
        self.descend_match_shard(parent_hash, block_hashes, |shard, parent| {
            shard.match_prefix(parent, block_hashes)
        })
    }

    /// How many leading blocks of `block_hashes` each worker holds
    /// contiguously, in one descent; an absent worker holds none.
    pub fn prefix_depths(
        &self,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
    ) -> HashMap<KvWorkerId, usize> {
        if block_hashes.is_empty() {
            return HashMap::new();
        }
        self.descend_match_shard(parent_hash, block_hashes, |shard, parent| {
            shard.prefix_depths(parent, block_hashes)
        })
    }

    /// Number of non-root nodes across all shards, summed shard by shard,
    /// so not one consistent instant.
    pub fn node_count(&self) -> usize {
        self.shards.iter().map(|s| s.read().node_count()).sum()
    }

    /// Reverse-index keys summed across shards, for invariant tests:
    /// nonzero with `node_count() == 0` means a prune leaked `by_hash`.
    pub fn reverse_index_size(&self) -> usize {
        self.shards.iter().map(|s| s.read().by_hash.len()).sum()
    }

    /// Occupancy bookkeeping contradictions by [`ACCOUNTING_REASONS`]; zero on a
    /// correct tree, and visible in release builds where the assertion is off.
    pub fn accounting_errors(&self) -> [u64; ACCOUNTING_REASONS.len()] {
        let mut total = [0u64; ACCOUNTING_REASONS.len()];
        for shard in &self.shards {
            for (acc, c) in total.iter_mut().zip(shard.read().accounting_errors) {
                *acc += c;
            }
        }
        total
    }

    /// Nodes each carrier holds per tier, summed across shards and sorted by
    /// carrier, as rendered in `sgl_router_kv_tree_blocks`. Read off the
    /// per-shard accounting, so not one consistent instant.
    pub fn tier_occupancy(&self) -> Vec<(KvWorkerId, TierCounts)> {
        let mut total: HashMap<KvWorkerId, TierCounts> = HashMap::new();
        for shard in &self.shards {
            for (worker, counts) in &shard.read().occupancy {
                let acc = total.entry(worker.clone()).or_default();
                for (a, c) in acc.iter_mut().zip(counts) {
                    *a += c;
                }
            }
        }
        let mut rows: Vec<(KvWorkerId, TierCounts)> = total.into_iter().collect();
        rows.sort_by(|a, b| (&a.0.url, a.0.dp_rank).cmp(&(&b.0.url, b.0.dp_rank)));
        rows
    }

    /// Evict LRU leaves, globally across shards, until `node_count() <= max_size`;
    /// returns the nodes pruned. Holds [`Self::writer`] throughout, so the cap is
    /// a postcondition; keep it off the hot path.
    /// Only tests call it, so the tree is bounded by `BlockRemoved` turnover alone.
    pub fn evict_lru(&self, max_size: usize) -> usize {
        let _writer = self.writer.lock();
        let mut remaining = self.node_count();
        if remaining <= max_size {
            return 0;
        }
        let mut pruned = 0usize;

        // Free already-empty leaves first; `remaining` tracks the global count.
        for shard in &self.shards {
            let dropped = shard.write().drop_empty_leaves();
            pruned += dropped;
            remaining -= dropped;
            if remaining <= max_size {
                return pruned;
            }
        }

        // Then evict the globally-oldest leaf, bounded so a degenerate tree cannot spin.
        let mut iters = 0usize;
        let max_iters = remaining.saturating_add(1);
        while remaining > max_size && iters < max_iters {
            iters += 1;
            // Oldest leaf globally; (ts, node id, shard) breaks ties deterministically.
            let mut target: Option<(u64, NodeId, usize)> = None;
            for (idx, shard) in self.shards.iter().enumerate() {
                if let Some((ts, id)) = shard.read().lru_leaf() {
                    let cand = (ts, id, idx);
                    match target {
                        None => target = Some(cand),
                        Some(cur) if cand < cur => target = Some(cand),
                        _ => {}
                    }
                }
            }
            let Some((_, victim, idx)) = target else {
                break; // no leaves anywhere
            };
            let dropped = self.shards[idx].write().evict_leaf(victim);
            if dropped == 0 {
                // No longer a leaf; re-scan rather than spin on a stale pick.
                continue;
            }
            pruned += dropped;
            remaining -= dropped;
        }
        pruned
    }
}

// Whitebox helpers for the in-module tests, which assert on per-shard structure.

#[cfg(test)]
impl HashTree {
    /// Whether every carrier on every node holds at least one tier;
    /// otherwise a node can never be pruned nor a worker dropped.
    fn debug_no_empty_carrier(&self) -> bool {
        self.shards.iter().all(|s| {
            s.read()
                .nodes
                .values()
                .all(|n| n.workers.values().all(|t| !t.is_empty()))
        })
    }

    /// Whether every chain root lives in `shard_of` its own hash,
    /// the only shard `match_prefix(None, ..)` looks in.
    fn debug_roots_in_own_shard(&self) -> bool {
        self.shards.iter().enumerate().all(|(idx, s)| {
            s.read().nodes[&ROOT_ID]
                .children
                .keys()
                .all(|&h| shard_of(h) == idx)
        })
    }

    /// Whether any shard's reverse index carries `hash`.
    fn debug_has_hash(&self, hash: i64) -> bool {
        self.shards
            .iter()
            .any(|s| s.read().by_hash.contains_key(&hash))
    }

    /// Distinct nodes carrying `hash`, summed across shards.
    fn debug_hash_node_count(&self, hash: i64) -> usize {
        self.shards
            .iter()
            .map(|s| {
                s.read()
                    .by_hash
                    .get(&hash)
                    .map(|set| set.len())
                    .unwrap_or(0)
            })
            .sum()
    }

    /// `parent_block_hash` of the node carrying `hash`. Panics unless
    /// exactly one node carries it; callers build unambiguous chains.
    fn debug_parent_block_hash(&self, hash: i64) -> Option<i64> {
        let mut found: Option<Option<i64>> = None;
        for shard in &self.shards {
            let st = shard.read();
            if let Some(set) = st.by_hash.get(&hash) {
                assert_eq!(set.len(), 1, "debug_parent_block_hash: hash not unique");
                let id = *set.iter().next().unwrap();
                assert!(
                    found.is_none(),
                    "debug_parent_block_hash: hash present in multiple shards",
                );
                found = Some(st.nodes[&id].parent_block_hash);
            }
        }
        found.expect("debug_parent_block_hash: hash not present")
    }

    /// Recompute [`Self::tier_occupancy`] by a full walk. It shares
    /// [`Tiers::SLOTS`] with the code it checks, so it cannot catch a wrong
    /// tier; the example tests and compile-time asserts own that.
    fn debug_recount_occupancy(&self) -> Vec<(KvWorkerId, TierCounts)> {
        let mut total: HashMap<KvWorkerId, TierCounts> = HashMap::new();
        for shard in &self.shards {
            let st = shard.read();
            for (&id, node) in &st.nodes {
                if id == ROOT_ID {
                    continue;
                }
                for (worker, held) in &node.workers {
                    let acc = total.entry(worker.clone()).or_default();
                    tally_tiers(acc, *held);
                }
            }
        }
        let mut rows: Vec<(KvWorkerId, TierCounts)> = total.into_iter().collect();
        rows.sort_by(|a, b| (&a.0.url, a.0.dp_rank).cmp(&(&b.0.url, b.0.dp_rank)));
        rows
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test_support {
    use super::*;

    pub(super) fn worker(url: &str, dp_rank: u32) -> KvWorkerId {
        KvWorkerId {
            url: url.to_string(),
            dp_rank,
        }
    }

    pub(super) fn workers(ids: &[&KvWorkerId]) -> HashSet<KvWorkerId> {
        ids.iter().map(|w| (*w).clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{worker, workers};
    use super::*;

    /// Unlike `match_prefix`, which names only the deepest node's holders;
    /// a removed interior block stops the count at the hole.
    #[test]
    fn prefix_depths_answers_every_worker_at_its_own_depth() {
        let chain = [1i64, 2, 3, 4];
        let (deep, shallow, holed) = (
            worker("http://a", 0),
            worker("http://b", 0),
            worker("http://c", 0),
        );
        let tree = HashTree::new();
        tree.insert(&deep, None, &chain);
        tree.insert(&shallow, None, &chain[..2]);
        tree.insert(&holed, None, &chain);
        tree.remove(&holed, &chain[1..2]);

        let depths = tree.prefix_depths(None, &chain);
        assert_eq!(depths.get(&deep), Some(&4), "holds the whole chain");
        assert_eq!(depths.get(&shallow), Some(&2), "holds two of four");
        assert_eq!(depths.get(&holed), Some(&1), "stops at the cleared block");
        assert_eq!(depths.get(&worker("http://d", 0)), None, "holds nothing");

        // Contiguity matters: `remove` left `holed` listed at the deepest node,
        // so `match_prefix` credits it with the full chain scored here at 1.
        let m = tree.match_prefix(None, &chain);
        assert_eq!(m.matched_blocks, 4);
        assert_eq!(m.workers(), workers(&[&deep, &holed]));
    }

    #[test]
    fn empty_match_returns_zero_no_workers() {
        let tree = HashTree::new();
        let m = tree.match_prefix(None, &[]);
        assert_eq!(m.matched_blocks, 0);
        assert!(m.workers().is_empty());

        let m2 = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m2.matched_blocks, 0);
        assert!(m2.workers().is_empty());
    }

    /// The engine's sequence for a backed-up block: device store, host store,
    /// device removal. Only the host removal ends ownership.
    #[test]
    fn device_eviction_after_host_backup_keeps_the_worker_as_host_owner() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert_tiered(&a, None, &[1, 2, 3], Tiers::for_store(Some("GPU")));
        tree.insert_tiered(&a, None, &[1, 2, 3], Tiers::for_store(Some("CPU_PINNED")));

        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));
        assert_eq!(m.device_workers(), workers(&[&a]), "held on device too");

        tree.remove_tiered(&a, &[3], Tiers::for_remove(Some("GPU")));
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3, "host copy keeps the chain matchable");
        assert_eq!(
            m.workers(),
            workers(&[&a]),
            "host-only holder is still an owner"
        );
        assert!(
            m.device_workers().is_empty(),
            "but no longer a device owner"
        );

        tree.remove_tiered(&a, &[3], Tiers::for_remove(Some("CPU_PINNED")));
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 2, "last tier gone: node pruned");
    }

    /// Routing reads `prefix_depths`, so a host backup must keep the full depth
    /// there; the untagged removal is the control that clears every tier.
    #[test]
    fn prefix_depths_survive_a_device_eviction_with_a_host_backup() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert_tiered(&a, None, &[1, 2, 3], Tiers::for_store(Some("GPU")));
        tree.insert_tiered(&a, None, &[1, 2, 3], Tiers::for_store(Some("CPU_PINNED")));

        // Block 1 included on purpose: `prefix_depths` seeds its live set there,
        // and a seed narrowed to device owners must fail this test.
        tree.remove_tiered(&a, &[1, 2, 3], Tiers::for_remove(Some("GPU")));
        assert_eq!(
            tree.prefix_depths(None, &[1, 2, 3]).get(&a).copied(),
            Some(3),
            "host backup keeps every level attributed to the worker",
        );

        tree.remove_tiered(&a, &[1, 2, 3], Tiers::for_remove(None));
        assert_eq!(
            tree.prefix_depths(None, &[1, 2, 3]).get(&a).copied(),
            None,
            "an untagged removal still clears every tier, host backup included",
        );
    }

    /// A bare `BlockStored` is a device store, a bare `BlockRemoved` clears
    /// every tier.
    #[test]
    fn untagged_events_keep_the_legacy_meaning() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2]);
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(
            m.device_workers(),
            workers(&[&a]),
            "untagged store is device"
        );

        // Add a host copy, then an UNTAGGED removal: must clear both.
        tree.insert_tiered(&a, None, &[1, 2], Tiers::HOST);
        tree.remove(&a, &[2]);
        assert_eq!(tree.match_prefix(None, &[1, 2]).matched_blocks, 1);
    }

    /// Asymmetric on purpose: see [`Tiers::for_store`] and
    /// [`Tiers::for_remove`].
    #[test]
    fn unknown_medium_drops_the_store_and_removes_everything() {
        assert_eq!(Tiers::for_store(Some("NVLINK_PEER")), Tiers::default());
        assert_eq!(Tiers::for_remove(Some("NVLINK_PEER")), Tiers::ALL);
        assert_eq!(Tiers::for_store(None), Tiers::DEVICE);
        assert_eq!(Tiers::for_remove(None), Tiers::ALL);
        assert_eq!(Tiers::for_store(Some("CPU_PINNED")), Tiers::HOST);
        assert_eq!(Tiers::for_store(Some("DISK")), Tiers::DISK);
        assert_eq!(Tiers::for_store(Some("EXTERNAL")), Tiers::EXTERNAL);
        // A known tag clears its own tier and nothing else.
        assert_eq!(Tiers::for_remove(Some("DISK")), Tiers::DISK);
        assert_eq!(Tiers::for_remove(Some("EXTERNAL")), Tiers::EXTERNAL);

        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert_tiered(&a, None, &[1], Tiers::DEVICE);
        tree.insert_tiered(&a, None, &[1], Tiers::HOST);
        tree.insert_tiered(&b, None, &[1], Tiers::for_store(Some("NVLINK_PEER")));
        let m = tree.match_prefix(None, &[1]);
        assert_eq!(
            m.workers(),
            workers(&[&a]),
            "a store on an unrankable tier must not make the worker an owner"
        );
        tree.remove_tiered(&a, &[1], Tiers::for_remove(Some("NVLINK_PEER")));
        assert_eq!(
            tree.match_prefix(None, &[1]).matched_blocks,
            0,
            "an unknown-medium removal clears every tier the worker held",
        );
    }

    /// One worker's device removal must not touch another's hold on the node,
    /// and the device subset is reported exactly.
    #[test]
    fn tiers_are_tracked_per_worker() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert_tiered(&a, None, &[1, 2], Tiers::DEVICE);
        tree.insert_tiered(&b, None, &[1, 2], Tiers::HOST);

        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.workers(), workers(&[&a, &b]));
        assert_eq!(m.device_workers(), workers(&[&a]));

        tree.remove_tiered(&a, &[2], Tiers::DEVICE);
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.workers(), workers(&[&b]), "a dropped, b untouched");
        assert!(m.device_workers().is_empty());
    }

    /// A store on no tier must not create a carrier,
    /// since `remove` prunes on "no bits means no entry".
    #[test]
    fn empty_tier_insert_is_noop() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert_tiered(&a, None, &[1], Tiers::default());
        assert_eq!(tree.node_count(), 0);
    }

    /// Occupancy is booked incrementally at every mutation site,
    /// so each kind of mutation is checked against a full recount.
    #[test]
    fn tier_occupancy_matches_a_full_recount_after_mixed_mutations() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 1);
        let c = worker("http://c", 0);

        tree.insert_tiered(&a, None, &[1, 2, 3, 4], Tiers::DEVICE);
        tree.insert_tiered(&a, None, &[1, 2], Tiers::HOST);
        tree.insert_tiered(&b, None, &[1, 2, 3], Tiers::HOST);
        tree.insert_tiered(&b, None, &[1, 2, 5, 6], Tiers::DEVICE);
        tree.insert_tiered(&c, None, &[7], Tiers::for_store(Some("EXTERNAL")));
        tree.insert_tiered(&c, None, &[8], Tiers::for_store(Some("NVLINK_PEER")));
        for r in 0..40i64 {
            tree.insert(&c, None, &[r * 4096 + 11, r * 4096 + 12]);
        }
        assert_eq!(tree.tier_occupancy(), tree.debug_recount_occupancy());

        // Spot-check the shape: a holds 4 device nodes and 2 host nodes.
        let rows = tree.tier_occupancy();
        let (_, a_counts) = rows.iter().find(|(w, _)| *w == a).unwrap();
        assert_eq!(a_counts[0], 4, "device");
        assert_eq!(a_counts[1], 2, "host");
        assert_eq!(a_counts[2], 0, "disk");
        let (_, c_counts) = rows.iter().find(|(w, _)| *w == c).unwrap();
        assert_eq!(c_counts[3], 1, "external");
        assert_eq!(
            tree.match_prefix(None, &[8]).matched_blocks,
            0,
            "a store on an unrankable medium is booked nowhere because it is              never applied",
        );

        // A device-only removal on a two-tier node, then one that prunes.
        tree.remove_tiered(&a, &[2], Tiers::DEVICE);
        tree.remove_tiered(&a, &[4], Tiers::ALL);
        assert_eq!(tree.tier_occupancy(), tree.debug_recount_occupancy());

        // Whole-worker clear, then LRU eviction down to a small cap.
        tree.clear_worker(&b);
        assert_eq!(tree.tier_occupancy(), tree.debug_recount_occupancy());
        assert!(tree.evict_lru(10) > 0);
        assert_eq!(tree.tier_occupancy(), tree.debug_recount_occupancy());
        assert!(
            tree.tier_occupancy().iter().all(|(w, _)| *w != b),
            "a cleared worker must leave no occupancy row",
        );
    }

    /// The engine's `StorageMedium` keeps L3 and L4 apart, so an L3 eviction
    /// must not erase the L4 copy.
    #[test]
    fn disk_and_external_are_independent_tiers() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert_tiered(&a, None, &[1], Tiers::for_store(Some("EXTERNAL")));
        tree.insert_tiered(&a, None, &[1], Tiers::for_store(Some("DISK")));

        tree.remove_tiered(&a, &[1], Tiers::for_remove(Some("DISK")));
        assert_eq!(
            tree.match_prefix(None, &[1]).workers(),
            workers(&[&a]),
            "an L3 eviction must not take the L4 copy with it",
        );

        tree.remove_tiered(&a, &[1], Tiers::for_remove(Some("EXTERNAL")));
        assert_eq!(tree.match_prefix(None, &[1]).matched_blocks, 0);
    }

    /// After every step of a deterministic walk over every operation and medium:
    /// occupancy equals a recount, no carrier has empty tiers,
    /// and every chain root sits in `shard_of` its own hash.
    #[test]
    fn tree_invariants_hold_under_a_random_walk() {
        // xorshift64*, so the walk is reproducible without a dev-dependency.
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 >> 12;
                self.0 ^= self.0 << 25;
                self.0 ^= self.0 >> 27;
                self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
        }

        const MEDIA: [Option<&str>; 6] = [
            Some("GPU"),
            Some("CPU_PINNED"),
            Some("DISK"),
            Some("EXTERNAL"),
            Some("NVLINK_PEER"),
            None,
        ];

        for seed in 1..=16u64 {
            let tree = HashTree::new();
            let ws: Vec<KvWorkerId> = (0..4)
                .map(|i| worker(&format!("http://w{i}"), i % 2))
                .collect();
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

            for step in 0..400 {
                let w = &ws[rng.below(ws.len() as u64) as usize];
                let medium = MEDIA[rng.below(MEDIA.len() as u64) as usize];
                // A small hash space so chains collide and share nodes.
                let hashes: Vec<i64> = (0..1 + rng.below(4))
                    .map(|_| rng.below(12) as i64)
                    .collect();
                // Sometimes anchor to a maybe-absent hash, to exercise the fallbacks.
                let parent = (rng.below(4) == 0).then(|| rng.below(12) as i64);

                match rng.below(16) {
                    0 => tree.clear_worker(w),
                    1 => {
                        tree.evict_lru(rng.below(20) as usize);
                    }
                    2..=6 => tree.remove_tiered(w, &hashes, Tiers::for_remove(medium)),
                    _ => tree.insert_tiered(w, parent, &hashes, Tiers::for_store(medium)),
                }

                assert_eq!(
                    tree.tier_occupancy(),
                    tree.debug_recount_occupancy(),
                    "seed {seed} step {step}: incremental occupancy diverged from a full recount",
                );
                assert!(
                    tree.debug_no_empty_carrier(),
                    "seed {seed} step {step}: a carrier is present holding no tier",
                );
                assert!(
                    tree.debug_roots_in_own_shard(),
                    "seed {seed} step {step}: a chain root is in the wrong shard, \
                     so match_prefix(None, ..) can never reach it again",
                );
            }
        }
    }

    /// `AllBlocksCleared` is also the scale-down path, so it must drop a carrier
    /// whatever tiers it held.
    #[test]
    fn clear_worker_drops_carriers_on_every_tier() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert_tiered(&a, None, &[1, 2], Tiers::HOST);
        tree.insert_tiered(&a, None, &[3], Tiers::for_store(Some("EXTERNAL")));
        tree.insert_tiered(&b, None, &[1, 2], Tiers::DEVICE);

        tree.clear_worker(&a);
        assert_eq!(
            tree.match_prefix(None, &[1, 2]).workers(),
            workers(&[&b]),
            "a host-only carrier must be cleared like any other",
        );
        assert_eq!(tree.match_prefix(None, &[3]).matched_blocks, 0);
        assert!(
            tree.tier_occupancy().iter().all(|(w, _)| *w != a),
            "a cleared worker must leave no occupancy row on any tier",
        );
        assert_eq!(tree.tier_occupancy(), tree.debug_recount_occupancy());
    }

    /// An ambiguous parent prefers a candidate the worker holds on ANY tier;
    /// device-only would re-root a continuation the host tier still serves.
    #[test]
    fn ambiguous_parent_resolves_through_a_host_only_hold() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert_tiered(&a, None, &[5, 7], Tiers::DEVICE);
        tree.insert_tiered(&a, None, &[5, 7], Tiers::HOST);
        tree.remove_tiered(&a, &[5, 7], Tiers::for_remove(Some("GPU")));
        // A second chain carrying hash 7, so `parent_hash = 7` is ambiguous.
        tree.insert_tiered(&b, None, &[6, 7], Tiers::DEVICE);

        tree.insert_tiered(&a, Some(7), &[8], Tiers::DEVICE);
        assert_eq!(
            tree.prefix_depths(None, &[5, 7, 8]).get(&a).copied(),
            Some(3),
            "the continuation must attach under the host-only hold, not at root",
        );
    }

    #[test]
    fn single_insert_and_match() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);

        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));

        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&a]));

        // Diverges at depth 3 (input asks for 4, tree has 3).
        let m = tree.match_prefix(None, &[1, 2, 4]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&a]));

        // No match at root.
        let m = tree.match_prefix(None, &[9, 9]);
        assert_eq!(m.matched_blocks, 0);
        assert!(m.workers().is_empty());
    }

    #[test]
    fn two_workers_overlapping_prefix() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert(&a, None, &[1, 2, 3]);
        tree.insert(&b, None, &[1, 2, 4]);

        // Common prefix node carries both.
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&a, &b]));

        // Divergent leaf carries only the matching worker.
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));

        let m = tree.match_prefix(None, &[1, 2, 4]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&b]));
    }

    #[test]
    fn continuation_insert_chains_via_parent_hash() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2]);
        tree.insert(&a, Some(2), &[3]);

        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));
    }

    #[test]
    fn remove_specific_blocks_drops_worker_at_those_nodes() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);
        // Sanity.
        assert_eq!(tree.node_count(), 3);

        // Node 2 loses A but survives (it has child 3); descendants are untouched.
        tree.remove(&a, &[2]);

        // Match length 2 lands on node 2, whose worker set is now empty.
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2);
        assert!(m.workers().is_empty());

        // Match length 3 lands on node 3 (workers still has A).
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));

        // Reverse-index sanity for hash 2: still present (node holds it).
        assert!(tree.debug_has_hash(2));
    }

    #[test]
    fn clear_worker_drops_exclusive_branches_keeps_shared_nodes() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert(&a, None, &[1, 2, 3]);
        tree.insert(&b, None, &[1, 2, 4]);
        let n_before = tree.node_count();
        assert_eq!(n_before, 4); // 1, 2, 3, 4

        tree.clear_worker(&a);

        // [1,2,3] no longer has A; node 3 prunes (only A held it).
        let m = tree.match_prefix(None, &[1, 2, 3]);
        // Node 3 was pruned, so only 2 levels match.
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&b]));

        // B still holds 1 and 2.
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&b]));

        // Node count: root + 1 + 2 + 4 (no 3) = 3 non-root nodes.
        assert_eq!(tree.node_count(), 3);
    }

    #[test]
    fn pruning_cascades_when_only_worker_clears() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);
        assert_eq!(tree.node_count(), 3);

        tree.clear_worker(&a);
        // Whole chain prunes; only the root sentinel remains.
        assert_eq!(tree.node_count(), 0);
        // Reverse index for these hashes should be empty.
        assert!(!tree.debug_has_hash(1));
        assert!(!tree.debug_has_hash(2));
        assert!(!tree.debug_has_hash(3));
    }

    #[test]
    fn pruning_cascades_via_remove_blockhashes() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);

        // Remove all of A's blocks at once.
        tree.remove(&a, &[1, 2, 3]);
        assert_eq!(tree.node_count(), 0);
    }

    #[test]
    fn same_hash_in_two_chains_both_tracked_in_reverse_index() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        // Two chains share hash=5 but at different positions.
        tree.insert(&a, None, &[1, 5]);
        tree.insert(&a, None, &[2, 5]);

        // Both chains exist independently.
        let m = tree.match_prefix(None, &[1, 5]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&a]));

        let m = tree.match_prefix(None, &[2, 5]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&a]));

        // Hash 5 is carried by 2 distinct nodes, summed across shards.
        assert_eq!(tree.debug_hash_node_count(5), 2);

        // Removing [5] clears both nodes carrying it; both are leaves, so both prune.
        tree.remove(&a, &[5]);
        // Remaining nodes: 1 and 2 (still hold A).
        assert_eq!(tree.node_count(), 2);
        let m = tree.match_prefix(None, &[1, 5]);
        assert_eq!(m.matched_blocks, 1);
        assert_eq!(m.workers(), workers(&[&a]));
        let m = tree.match_prefix(None, &[2, 5]);
        assert_eq!(m.matched_blocks, 1);
        assert_eq!(m.workers(), workers(&[&a]));
    }

    /// The only multi-root test whose roots share a shard.
    #[test]
    fn colliding_roots_in_same_shard_stay_independent() {
        // Guarded so a change to N_SHARDS / SHARD_MIX cannot silently void the test.
        assert_eq!(
            shard_of(1),
            shard_of(22),
            "test premise: roots 1 and 22 must share a shard",
        );
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert(&a, None, &[1, 900]);
        tree.insert(&b, None, &[22, 901]);

        // Each chain matches in full with only its own worker.
        let m = tree.match_prefix(None, &[1, 900]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&a]));
        let m = tree.match_prefix(None, &[22, 901]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&b]));

        // Removing A's chain leaves B's chain in the shared shard untouched.
        tree.remove(&a, &[1, 900]);
        assert_eq!(tree.match_prefix(None, &[1, 900]).matched_blocks, 0);
        let m = tree.match_prefix(None, &[22, 901]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&b]));
        assert_eq!(tree.node_count(), 2);
    }

    #[test]
    fn dp_rank_distinguishes_workers() {
        let tree = HashTree::new();
        let w0 = worker("http://u", 0);
        let w1 = worker("http://u", 1);
        tree.insert(&w0, None, &[1, 2, 3]);
        tree.insert(&w1, None, &[1, 2, 4]);

        // Common prefix has both ranks.
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&w0, &w1]));

        // Divergent leaves: each rank on its own.
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&w0]));

        let m = tree.match_prefix(None, &[1, 2, 4]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&w1]));
    }

    #[test]
    fn parent_hash_resolution_picks_worker_owned_node() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        // Two nodes both carry hash=5.
        tree.insert(&a, None, &[1, 5]);
        tree.insert(&b, None, &[2, 5]);
        // A continues from its 5.
        tree.insert(&a, Some(5), &[7]);

        // The chain 1->5->7 must exist with A.
        let m = tree.match_prefix(None, &[1, 5, 7]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));

        // The chain 2->5 should NOT have a 7-child (we routed to A's branch).
        let m = tree.match_prefix(None, &[2, 5, 7]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers(), workers(&[&b]));
    }

    #[test]
    fn ambiguous_parent_hash_unowned_falls_back_to_root() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        let c = worker("http://c", 0);
        // Two nodes carry hash=5, neither is owned by C.
        tree.insert(&a, None, &[1, 5]);
        tree.insert(&b, None, &[2, 5]);
        // C extends from parent_hash=5, which falls back to root.
        tree.insert(&c, Some(5), &[9]);

        // C is reachable as a fresh root child at hash=9.
        let m = tree.match_prefix(None, &[9]);
        assert_eq!(m.matched_blocks, 1);
        assert_eq!(m.workers(), workers(&[&c]));
    }

    /// A `parent_hash` no shard carries root-attaches, but global resolution
    /// must not drop it as the first node's breadcrumb.
    #[test]
    fn unknown_parent_hash_root_attaches_but_keeps_the_breadcrumb() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, Some(99), &[1, 2]);

        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2, "the chain must attach at root");
        assert!(m.holds(&a));
        assert_eq!(
            tree.debug_parent_block_hash(1),
            Some(99),
            "the unresolved parent must stay recorded on the chain's first node",
        );
    }

    /// The forced root-attach of an unowned-ambiguous parent keeps its breadcrumb.
    #[test]
    fn ambiguous_unowned_parent_root_attaches_but_keeps_the_breadcrumb() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        let c = worker("http://c", 0);
        // Two carriers of hash 5, neither held by C.
        tree.insert(&a, None, &[1, 5]);
        tree.insert(&b, None, &[2, 5]);
        tree.insert(&c, Some(5), &[1009]);

        assert_eq!(
            tree.match_prefix(None, &[1009]).matched_blocks,
            1,
            "the chain must still root-attach",
        );
        assert_eq!(
            tree.debug_parent_block_hash(1009),
            Some(5),
            "forcing the attach point must not drop the recorded parent",
        );
    }

    /// Regression: an unowned-ambiguous parent roots the chain even in a shard
    /// that locally carries `parent_hash` exactly once.
    #[test]
    fn unowned_ambiguous_parent_force_roots_even_on_carrier_shard() {
        // Guard the premise; a new shard count or mix needs new constants.
        assert_ne!(
            shard_of(1),
            shard_of(2),
            "test premise: roots 1 and 2 must be on different shards",
        );
        assert_eq!(
            shard_of(1009),
            shard_of(1),
            "test premise: continuation root 1009 must collide with root 1's shard",
        );

        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        let c = worker("http://c", 0);
        tree.insert(&a, None, &[1, 5]); // node-5 in shard_of(1)
        tree.insert(&b, None, &[2, 5]); // node-5 in shard_of(2)

        // C owns neither node-5, so 1009 must become a fresh root child.
        tree.insert(&c, Some(5), &[1009]);

        let m = tree.match_prefix(None, &[1009]);
        assert_eq!(
            m.matched_blocks, 1,
            "1009 must attach at root, reachable as a top-level child",
        );
        assert_eq!(m.workers(), workers(&[&c]));

        // And 1->5 must NOT have grown a 1009 child.
        let m = tree.match_prefix(None, &[1, 5, 1009]);
        assert_eq!(
            m.matched_blocks, 2,
            "1009 must NOT be attached under the shard's node carrying 5",
        );
    }

    /// Regression: a parent pruned between [`HashTree::route_match`] and the
    /// descent must fall back to the chain's root shard, not the nominated one.
    /// Uses a hand-made stale pair, since sequential calls always agree.
    #[test]
    fn a_pruned_parent_falls_back_to_the_chains_root_shard() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let chain = [1i64, 2, 3];
        // Negative so it cannot collide with the chain,
        // and off the chain's shard so a wrong descent is observable.
        let mut p = -7i64;
        while shard_of(p) == shard_of(chain[0]) {
            p -= 1;
        }
        tree.insert(&a, None, &chain);
        tree.insert(&a, None, &[p]);

        let (idx, parent) = tree.route_match(Some(p), &chain);
        assert_eq!(
            (idx, parent),
            (shard_of(p), Some(p)),
            "test premise: `p` resolves uniquely, to a shard that is not the chain's",
        );

        // The racing writer prunes `p` — empty and childless, so the node goes.
        tree.remove(&a, &[p]);

        assert_eq!(
            tree.shards[idx]
                .read()
                .match_prefix(Some(p), &chain)
                .matched_blocks,
            0,
            "test premise: the nominated shard's sentinel carries no chain, \
             so an in-shard fall-back to root matches nothing",
        );

        let m = tree.descend_routed(idx, parent, &chain, |shard, parent| {
            shard.match_prefix(parent, &chain)
        });
        assert_eq!(
            m.matched_blocks,
            chain.len(),
            "a stale resolution must fall back to the chain's root shard",
        );
        assert_eq!(m.workers(), workers(&[&a]));

        let depths = tree.descend_routed(idx, parent, &chain, |shard, parent| {
            shard.prefix_depths(parent, &chain)
        });
        assert_eq!(
            depths.get(&a),
            Some(&chain.len()),
            "`prefix_depths` shares the fall-back",
        );
    }

    /// The same staleness when `p` gains a second carrier in the nominated
    /// shard between the scan and the descent.
    #[test]
    fn a_parent_that_gained_a_carrier_falls_back_to_the_chains_root_shard() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let chain = [1i64, 2, 3];
        let p = -7i64;
        // Two roots sharing one shard that is not the chain's.
        let mut h1 = 4i64;
        while shard_of(h1) == shard_of(chain[0]) {
            h1 += 1;
        }
        let mut h2 = h1 + 1;
        while shard_of(h2) != shard_of(h1) {
            h2 += 1;
        }
        tree.insert(&a, None, &chain);
        tree.insert(&a, None, &[h1, p]);

        let (idx, parent) = tree.route_match(Some(p), &chain);
        assert_eq!(
            (idx, parent),
            (shard_of(h1), Some(p)),
            "test premise: one carrier of `p`, off the chain's shard",
        );

        // The racing writer roots a second chain carrying `p` in that shard.
        tree.insert(&a, None, &[h2, p]);

        assert_eq!(
            tree.shards[idx]
                .read()
                .match_prefix(Some(p), &chain)
                .matched_blocks,
            0,
            "test premise: ambiguity sends the descent to the wrong sentinel",
        );

        let m = tree.descend_routed(idx, parent, &chain, |shard, parent| {
            shard.match_prefix(parent, &chain)
        });
        assert_eq!(
            m.matched_blocks,
            chain.len(),
            "a resolution invalidated by a second carrier must fall back to \
             the chain's root shard",
        );
        assert_eq!(m.workers(), workers(&[&a]));
    }

    /// A prune landing between `insert`'s shard scan and its write must not
    /// re-root the chain in the wrong shard. Fails without [`HashTree::writer`].
    #[test]
    fn concurrent_writers_never_orphan_a_chain() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let tree = Arc::new(HashTree::new());
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        // `h0` is off `r`'s shard, so a wrongly-rooted continuation is observable.
        let (r, p) = (1i64, 5i64);
        let mut h0 = 2i64;
        while shard_of(h0) == shard_of(r) {
            h0 += 1;
        }

        let stop = Arc::new(AtomicBool::new(false));
        let clearer = {
            let (tree, b, stop) = (tree.clone(), b.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    tree.clear_worker(&b);
                }
            })
        };

        for _ in 0..20_000 {
            tree.insert(&b, None, &[r, p]);
            tree.insert(&a, Some(p), &[h0]);
            assert!(
                tree.debug_roots_in_own_shard(),
                "a continuation was re-rooted in the wrong shard: unreachable \
                 from match_prefix(None, ..) for the rest of the process",
            );
            tree.clear_worker(&a);
        }

        stop.store(true, Ordering::Relaxed);
        clearer.join().expect("clearer thread panicked");
    }

    #[test]
    fn reinsert_same_chain_idempotent() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);
        tree.insert(&a, None, &[1, 2, 3]);

        assert_eq!(tree.node_count(), 3);
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));
    }

    #[test]
    fn empty_block_hashes_insert_is_noop() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[]);
        assert_eq!(tree.node_count(), 0);
    }

    #[test]
    fn eviction_smoke_drops_to_below_cap() {
        let tree = HashTree::new();
        // 50 distinct chains of length 1. Each chain gets its own root child.
        for i in 0..50i64 {
            let w = worker("http://w", i as u32);
            tree.insert(&w, None, &[i]);
        }
        assert_eq!(tree.node_count(), 50);

        let evicted = tree.evict_lru(10);
        // Each leaf hangs off root, so no prune cascades past it.
        assert_eq!(evicted, 40, "expected to evict exactly 40, got {evicted}");
        assert_eq!(
            tree.node_count(),
            10,
            "expected node_count == 10, got {}",
            tree.node_count()
        );
    }

    #[test]
    fn eviction_under_cap_is_noop() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);

        let evicted = tree.evict_lru(100);
        assert_eq!(evicted, 0);
        assert_eq!(tree.node_count(), 3);
    }

    #[test]
    fn eviction_prefers_oldest_leaves() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        // First chain: oldest.
        tree.insert(&a, None, &[100, 101, 102]);
        // Exceed the 1ms resolution of last_used.
        std::thread::sleep(std::time::Duration::from_millis(2));
        // Second chain: newer.
        tree.insert(&a, None, &[200, 201, 202]);

        // Match the newer chain to bump its last_used.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = tree.match_prefix(None, &[200, 201, 202]);

        // Leaf 102 is the LRU, and pruning it cascades through the older chain.
        let evicted = tree.evict_lru(3);
        assert_eq!(evicted, 3, "expected to evict exactly 3, got {evicted}");
        assert_eq!(tree.node_count(), 3);

        // The newer chain should still match fully.
        let m = tree.match_prefix(None, &[200, 201, 202]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));
    }

    #[test]
    fn batched_block_stored_chains_correctly() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        // Each hash chains off its predecessor; parent_hash applies to the first.
        tree.insert(&a, None, &[10, 20, 30]);

        let m = tree.match_prefix(None, &[10, 20, 30]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));

        assert_eq!(tree.debug_parent_block_hash(30), Some(20));
        assert_eq!(tree.debug_parent_block_hash(20), Some(10));
        assert_eq!(tree.debug_parent_block_hash(10), None);
    }

    #[test]
    fn remove_does_not_drop_node_held_by_other_workers() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert(&a, None, &[1, 2, 3]);
        tree.insert(&b, None, &[1, 2, 3]);
        assert_eq!(tree.node_count(), 3);

        // A removes its blocks; B still holds them.
        tree.remove(&a, &[1, 2, 3]);
        assert_eq!(tree.node_count(), 3);

        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&b]));
    }

    /// Roots spread over several shards, and every chain still matches in full.
    #[test]
    fn distinct_roots_spread_across_shards() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        for r in 0..64i64 {
            tree.insert(&a, None, &[r * 1000, r * 1000 + 1]);
        }
        assert_eq!(tree.node_count(), 128);

        // Else the test is not exercising cross-shard routing at all.
        let used_shards = (0..64i64)
            .map(|r| shard_of(r * 1000))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(
            used_shards > 1,
            "expected roots to span multiple shards, got {used_shards}",
        );

        // Every chain still matches in full.
        for r in 0..64i64 {
            let m = tree.match_prefix(None, &[r * 1000, r * 1000 + 1]);
            assert_eq!(m.matched_blocks, 2, "chain {r} must match fully");
            assert_eq!(m.workers(), workers(&[&a]));
        }
    }
}
