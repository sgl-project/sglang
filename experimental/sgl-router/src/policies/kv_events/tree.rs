//! Hash-keyed radix tree for KV-cache event indexing.
//!
//! Each non-root node represents one block hash (`i64`). A node's children
//! are keyed by the *next* block hash in a chain, so a path from the root
//! down to depth `n` represents a chain of `n` block hashes. Every node
//! tracks the [`KvWorkerId`]s that hold the chain ending at that node, and
//! on which storage [`Tiers`] each of them holds it.
//!
//! The tree is fed by `BlockStored` / `BlockRemoved` / `AllBlocksCleared`
//! events from SGLang workers (decoded by [`super::wire`]) and is queried
//! via [`HashTree::match_prefix`] to find which workers already hold the
//! longest prefix of an incoming request's block-hash chain.
//!
//! # Concurrency
//!
//! The whole tree lives behind a single [`parking_lot::RwLock`]
//! ([`HashTree::state`]). The match path takes a read-lock and updates
//! `last_used` via an [`AtomicU64`] so that routing decisions across tokio
//! worker threads do not serialise on the lock. Mutations (insert / remove
//! / clear / evict) take a write-lock. We accept the coarse granularity
//! for v1 on the write side — correctness over throughput — and the
//! existing text-tree at `super::super::tree` is what serves the high-RPS
//! mesh-fallback path. This module is only on the cache-aware-from-events
//! path.
//!
//! # Reverse index
//!
//! `BlockRemoved` events carry only `block_hashes` and no parent context,
//! so without an index from `block_hash → set of nodes carrying that hash`
//! we'd have to walk the whole tree. We maintain that reverse index as
//! [`TreeState::by_hash`]. The same hash can legitimately appear at
//! multiple positions in the tree (e.g. as the last block of one chain and
//! as the second block of another), so each entry is a *set* of node IDs.
//!
//! # Pruning
//!
//! When a worker is dropped from a node and the node has no remaining
//! workers AND no children, we detach it from its parent and remove it
//! from the reverse index. Pruning cascades upward iteratively (chains
//! can be deep — the recursive form would risk stack-overflow for
//! pathological inputs).
//!
//! # Storage tiers
//!
//! An engine running a hierarchical cache holds a block on device and, once
//! it has been backed up, on host pinned memory (or a storage backend) as
//! well. It publishes every tier transition as its own event, tagged with a
//! `medium`: a host-tier `BlockStored` when the backup lands, a device-tier
//! `BlockRemoved` when the device copy is evicted, a host-tier `BlockRemoved`
//! when the host copy goes.
//!
//! The two orderings differ by write policy, and neither may be assumed.
//! Under write-through the pending D2H copy holds a lock ref, so the host
//! store is published before the device eviction. Under write-back
//! `_detach_backuped` publishes the device removal as soon as host slots are
//! reserved and the host store follows only when the copy lands — the
//! inverse. The tree therefore has to converge either way, and it does: a
//! removal clears only its own tier, and a later store re-adds the chain.
//!
//! WHY the tree keeps the tiers apart instead of treating any `BlockRemoved`
//! as "the worker lost the block": a device eviction that leaves a host copy
//! behind does not end the worker's ability to serve the prefix — it loads
//! the copy back at memory speed instead of recomputing it. Dropping the
//! worker on that event makes every prefix unroutable after one device
//! turnover, even though the fleet still holds it for the whole host
//! retention horizon, and the repeat request lands on a random worker that
//! then prefills it cold. Ownership here is "any tier", which is what makes
//! the prefix routable again. [`MatchResult::tiers`] reports which tier each
//! owner holds it on so a policy can price a load-back against an in-place
//! hit; no policy consumes it yet — the routing path reads
//! [`HashTree::prefix_depths`], which is tier-blind by design.
//!
//! Untagged events keep their pre-tiering meaning: an untagged store is a
//! device store, an untagged remove clears every tier. A store tagged with a
//! `medium` this build cannot rank is dropped rather than filed under a
//! guess, while a removal tagged with one clears every tier. See [`Tiers`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use parking_lot::RwLock;
use tracing::{debug, error};

/// Process-wide monotonic epoch used to derive cheap millisecond-resolution
/// timestamps for [`Node::last_used`]. Initialised lazily on first use.
static PROCESS_EPOCH: OnceLock<Instant> = OnceLock::new();

/// Milliseconds elapsed since [`PROCESS_EPOCH`]. Truncates from `u128` to
/// `u64`; with `u64` ms we have ~584 million years of headroom which is
/// fine.
fn now_millis() -> u64 {
    PROCESS_EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// Identifier for a worker endpoint, refined by DP-attention rank.
///
/// Workers running with multiple DP-attention ranks emit independent event
/// streams (one per rank), and each rank holds a disjoint slice of the KV
/// cache. We therefore track them as separate cache-holders.
///
/// The name is intentionally namespaced (`KvWorkerId`) to avoid collision
/// with [`crate::core::worker_registry::WorkerId`], which is a UUID-string
/// identity used by the worker registry.
///
/// # Provenance
///
/// Instances should only be minted by the kv_events module itself
/// (subscriber registry → pump → tree) so the `url` always comes from
/// the worker registry's authoritative URL. External callers can read
/// the fields and use them to query the tree, but constructing fresh
/// IDs from arbitrary URLs would let routing logic resolve to
/// non-registered endpoints. Use [`KvWorkerId::new`] when constructing
/// from a tested path; do not assemble struct literals from
/// user-controlled input.
#[derive(Clone, Eq, Hash, PartialEq, Debug)]
pub struct KvWorkerId {
    pub url: String,
    pub dp_rank: u32,
}

impl KvWorkerId {
    /// Explicit constructor — preferred over struct-literal syntax so
    /// future tightening of provenance has a single chokepoint.
    pub fn new(url: String, dp_rank: u32) -> Self {
        Self { url, dp_rank }
    }
}

/// The storage tiers on which one worker holds one block — a bitset, because
/// a backed-up block sits on device AND host at once (module docs, "Storage
/// tiers").
///
/// A worker owns a block, and is a routing candidate for it, while ANY bit is
/// set.
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
    ///
    /// Separate from [`Self::DISK`] rather than folded into one "storage" bit
    /// because the engine treats them as distinct tiers
    /// (`StorageMedium` in `python/sglang/srt/disaggregation/kv_events.py`).
    /// Sharing a bit would make an L3 eviction erase the router's knowledge of
    /// an L4 copy on a fleet running both.
    pub const EXTERNAL: Tiers = Tiers(1 << 3);
    /// Every tier — what an untagged `BlockRemoved` clears.
    pub const ALL: Tiers = Self::DEVICE
        .union(Self::HOST)
        .union(Self::DISK)
        .union(Self::EXTERNAL);
    /// Every tier with its metric label, in bit order. [`TierCounts`] is
    /// indexed by position in this table, and the order is preference order:
    /// a device copy is served in place, a host copy by load-back from host
    /// memory, local disk slower still, a remote pool slower again.
    pub const SLOTS: [(Tiers, &'static str); TIER_SLOT_COUNT] = [
        (Self::DEVICE, "device"),
        (Self::HOST, "host"),
        (Self::DISK, "disk"),
        (Self::EXTERNAL, "external"),
    ];

    /// The `StorageMedium` strings SGLang puts on the wire
    /// (`python/sglang/srt/disaggregation/kv_events.py`) and the tier each
    /// lands on. The single source for both the tree's ranking and the event
    /// tally's medium labels, so a medium the tree ranks can never be one the
    /// tally reports as unknown.
    pub const WIRE_MEDIA: [(&'static str, Tiers); 4] = [
        ("GPU", Self::DEVICE),
        ("CPU_PINNED", Self::HOST),
        ("DISK", Self::DISK),
        ("EXTERNAL", Self::EXTERNAL),
    ];

    /// The tiers a `BlockStored` tagged `medium` lands on. Untagged reads as
    /// device, which is what the event meant before tiers existed. An unknown
    /// tag lands on NO tier: the store is dropped rather than filed under a
    /// guess.
    ///
    /// Filing it on device would turn every store on a future medium into a
    /// device hit. Filing it on an "unranked" catch-all tier is no better — it
    /// makes the worker a routing candidate for a tier whose cost this build
    /// cannot price, and a load-back the router assumes is cheap may be a
    /// remote fetch dearer than recomputing the prefix elsewhere. Dropping
    /// costs at most one cold prefill and is visible: the event still counts
    /// on the tally's `unknown` medium row, and the string is logged once.
    pub fn for_store(medium: Option<&str>) -> Tiers {
        match medium {
            None => Self::DEVICE,
            Some(m) => Self::known(m).unwrap_or_default(),
        }
    }

    /// The tiers a `BlockRemoved` tagged `medium` clears: its own, or every
    /// tier when the tag is absent or unrecognised.
    ///
    /// Deliberately asymmetric with [`Self::for_store`], which drops an
    /// unknown tag instead of widening it. A removal is the one direction
    /// where guessing wide is safe and guessing narrow is not: over-removal
    /// costs at most one cold prefill, while a permanently stale owner is the
    /// one failure a routing index must never manufacture. An engine that
    /// stores under a `medium` this build knows but frees the block under one
    /// it does not would otherwise strand that bit until the worker is
    /// dropped entirely.
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

    /// The bits set in either. `const` so the tier tables can be built from
    /// the individual tiers rather than from raw bit arithmetic.
    pub const fn union(self, other: Tiers) -> Tiers {
        Tiers(self.0 | other.0)
    }
}

/// Number of entries in [`Tiers::SLOTS`].
pub const TIER_SLOT_COUNT: usize = 4;

// The three tier tables have to stay in step, and nothing about adding a
// `pub const` tier would otherwise force it:
//
// * a bit missing from `ALL` is a bit an untagged `BlockRemoved` never
//   clears — a permanently stale owner, the one failure `for_remove` exists
//   to prevent;
// * a bit missing from `SLOTS` is never counted by `tally_tiers` nor
//   decremented by `account_remove`, so the tier is invisible in
//   `sgl_router_kv_tree_blocks` — this bug class, one tier later. The
//   `debug_recount_occupancy` oracle cannot catch it either, because it
//   walks the same `SLOTS`.
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

/// Add `tiers` to `worker`'s hold in `carriers`, creating the entry on first
/// sight, and return the bits that were newly set. One lookup on the re-store
/// path — under a hierarchical cache the host backup of a chain the worker
/// already holds on device, the common case — and two on first sight. Never
/// leaves an entry with no bits, which [`TreeState::remove`] relies on.
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

/// Result of [`HashTree::match_prefix`]. `Default` is the no-match result:
/// zero depth, no carriers on any tier.
#[derive(Debug, Clone, Default)]
pub struct MatchResult {
    /// Number of leading block hashes from the input slice that matched a
    /// path from the root.
    pub matched_blocks: usize,
    /// Every worker holding the deepest matched node, and the tiers it holds
    /// it on. A worker without the device bit serves the prefix by loading it
    /// back from a lower tier — cheaper than a cold prefill, dearer than
    /// serving in place — which is the ordering a policy would prefer on.
    /// Empty when `matched_blocks == 0`.
    ///
    /// The single carrier list: [`Self::workers`] and [`Self::device_workers`]
    /// are derived from it, so no two views of one match can disagree.
    pub tiers: HashMap<KvWorkerId, Tiers>,
}

impl MatchResult {
    /// Workers holding the deepest matched node on ANY tier.
    pub fn workers(&self) -> HashSet<KvWorkerId> {
        self.tiers.keys().cloned().collect()
    }

    /// The subset of [`Self::workers`] holding the deepest matched node on
    /// device.
    pub fn device_workers(&self) -> HashSet<KvWorkerId> {
        self.tiers
            .iter()
            .filter(|(_, tiers)| tiers.contains(Tiers::DEVICE))
            .map(|(w, _)| w.clone())
            .collect()
    }
}

/// Internal stable handle to a tree node.
///
/// We use an arena (`HashMap<NodeId, Node>`) instead of `Arc<RwLock<Node>>`
/// + `Weak` because:
/// 1. We need to enumerate every node (e.g. for `clear_worker` and
///    `evict_lru`); a flat map is direct and cheap.
/// 2. The reverse index needs a *stable* key per node — `Weak` would force
///    upgrades on every lookup and complicate prune semantics.
type NodeId = u64;

/// A single tree node. Non-root nodes are keyed by their `block_hash`
/// (which is shared across siblings only insofar as the reverse index
/// records every position); within a single parent's children map there is
/// at most one child per `block_hash`.
///
/// `last_used` is an [`AtomicU64`] of milliseconds since [`PROCESS_EPOCH`].
/// Storing it atomically lets the match path mutate it under a *read* lock
/// on [`TreeState`], which is essential because matching is on the routing
/// hot path. `Relaxed` ordering is sufficient: eviction only needs
/// approximate freshness, and ties at the millisecond boundary tie-break
/// by [`NodeId`].
#[derive(Debug)]
struct Node {
    block_hash: i64,
    /// Hash of the parent block on the chain that produced this node, or
    /// `None` if this node hangs directly off the root sentinel.
    /// Stored for diagnostic / chain-reconstruction only — the actual
    /// parent pointer lives in [`Node::parent`]. Tests and future
    /// inspectors read this; suppress dead-code warning in non-test builds.
    #[allow(dead_code)]
    parent_block_hash: Option<i64>,
    /// `None` only for the root sentinel.
    parent: Option<NodeId>,
    /// Carriers of the chain ending here, each with the tiers it holds the
    /// block on. A carrier is dropped the moment its last tier bit clears;
    /// an entry with empty tiers never exists (see [`TreeState::insert`]).
    workers: HashMap<KvWorkerId, Tiers>,
    /// Children keyed by next-block hash.
    children: HashMap<i64, NodeId>,
    last_used: AtomicU64,
}

impl Node {
    fn new_child(block_hash: i64, parent_block_hash: Option<i64>, parent: NodeId) -> Self {
        Self {
            block_hash,
            parent_block_hash,
            parent: Some(parent),
            workers: HashMap::new(),
            children: HashMap::new(),
            last_used: AtomicU64::new(now_millis()),
        }
    }
}

/// Inner mutable tree state. Single-lock for v1; document any cross-method
/// invariants here:
///
/// * `nodes[ROOT_ID]` is always present and is the only node with
///   `parent == None`.
/// * For every non-root node `n`: `nodes[n.parent].children[&n.block_hash]
///   == n`'s id (i.e., parent's child pointer round-trips).
/// * `by_hash[h]` contains the id of every non-root node `n` with
///   `n.block_hash == h`. Root is never in `by_hash`.
/// * Pruning runs after every worker-removal that empties a node: prune
///   detaches from parent, removes from `by_hash`, and recurses upward.
#[derive(Debug)]
struct TreeState {
    nodes: HashMap<NodeId, Node>,
    by_hash: HashMap<i64, HashSet<NodeId>>,
    next_id: NodeId,
    /// Nodes each carrier holds, per tier. Booked at every site where a
    /// carrier's tier bits on a node change (`account_add` /
    /// `account_remove`), so a scrape reads it without walking the tree. A
    /// carrier holding nothing has no row.
    occupancy: HashMap<KvWorkerId, TierCounts>,
    /// Times the occupancy bookkeeping contradicted itself, by reason. Both
    /// reasons mean the same class of bug — bits added without being booked —
    /// but the symptom the operator sees is a worker's
    /// `sgl_router_kv_tree_blocks` rows vanishing while it still holds nodes,
    /// which the metric's own HELP text says means "this worker publishes
    /// nothing". A counter is the only thing separating those.
    accounting_errors: [u64; ACCOUNTING_REASONS.len()],
}

/// Reasons in [`TreeState::accounting_errors`] order.
pub const ACCOUNTING_REASONS: [&str; 2] = ["missing_row", "underflow"];
const REASON_MISSING_ROW: usize = 0;
const REASON_UNDERFLOW: usize = 1;

const ROOT_ID: NodeId = 0;
/// Sentinel block_hash for the root. Real workers can in principle emit
/// `i64::MIN`, but the root is never looked up via `by_hash` so collisions
/// don't matter.
const ROOT_HASH_SENTINEL: i64 = i64::MIN;

impl TreeState {
    fn new() -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(
            ROOT_ID,
            Node {
                block_hash: ROOT_HASH_SENTINEL,
                parent_block_hash: None,
                parent: None,
                workers: HashMap::new(),
                children: HashMap::new(),
                last_used: AtomicU64::new(now_millis()),
            },
        );
        Self {
            nodes,
            by_hash: HashMap::new(),
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

    /// Book `delta` nodes-per-tier newly held by `worker` — one row lookup
    /// for a whole chain, which is how `insert` uses it.
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

    /// Book `delta` nodes-per-tier `worker` no longer holds, dropping the
    /// carrier's row once it holds nothing.
    ///
    /// Takes a whole-carrier delta rather than one node's bits so
    /// `clear_worker` can book a fleet-sized tree in ONE call: booking per
    /// node would emit one `error!` per node on the failure path below, which
    /// is hundreds of thousands of lines written while holding the tree's
    /// write lock — every routing decision blocked behind a log flood.
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
            // Saturating so a release can never wrap the gauge — but then the
            // all-zero test below would drop a row for a carrier that still
            // holds nodes, and a missing series reads as "this worker
            // publishes nothing". Count it so a release build is not blind;
            // assert so a debug run stops at the release that exposed it.
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

    /// Insert a brand-new child under `parent_id` and wire up the reverse
    /// index. Caller is responsible for ensuring `parent_id`'s child slot
    /// for `block_hash` is empty (else this overwrites it).
    ///
    /// Returns `None` if `parent_id` does not exist — an invariant
    /// violation. The pump runs in a long-lived task; panicking here would
    /// take down the entire cache-aware path, so we log and bail.
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

    /// Pick the parent node id for an incoming `BlockStored` event.
    ///
    /// Resolution order (matches doc-comment on `HashTree::insert`):
    /// 1. `parent_hash == None` → root.
    /// 2. There's exactly one node carrying `parent_hash` → use it.
    /// 3. Multiple candidates: prefer one already containing `worker`.
    /// 4. None contain the worker: log at debug, fall back to root. The
    ///    new chain still carries `parent_hash` on its first node so that
    ///    if the parent's `BlockStored` arrives later we can reconstruct
    ///    the link via the reverse index.
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

    /// Mark every node along `block_hashes` as held by `worker` on `tiers`,
    /// adding to whatever tiers it already holds there. Empty `tiers` is a
    /// no-op: a carrier entry with no tier would be an owner of nothing, and
    /// `remove` relies on "no bits ⇒ no entry" to know when to prune.
    fn insert(
        &mut self,
        worker: &KvWorkerId,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
        tiers: Tiers,
    ) {
        if block_hashes.is_empty() || tiers.is_empty() {
            return;
        }
        let mut current = self.resolve_parent(worker, parent_hash);
        let mut prev_hash = parent_hash;
        let now = now_millis();
        // Occupancy is booked once for the whole chain, not per block.
        let mut delta = TierCounts::default();
        for &h in block_hashes {
            let child_id = match self
                .nodes
                .get(&current)
                .and_then(|n| n.children.get(&h).copied())
            {
                Some(id) => id,
                // `break`, not `return`: the tier bits are already written
                // into the nodes visited so far, so bailing without reaching
                // `account_add` below would leave the occupancy gauge
                // permanently short by exactly those blocks.
                None => match self.create_child(current, h, prev_hash) {
                    Some(id) => id,
                    None => break,
                },
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

    /// Clear `tiers` from `worker`'s hold on every node carrying any hash in
    /// `block_hashes`. The worker leaves a node once no tier bit remains, and
    /// a node that becomes empty + childless is pruned.
    fn remove(&mut self, worker: &KvWorkerId, block_hashes: &[i64], tiers: Tiers) {
        // Collect all node ids to touch (fixed snapshot — avoids iterator
        // invalidation when pruning mutates `by_hash`).
        let mut targets: Vec<NodeId> = Vec::new();
        for h in block_hashes {
            if let Some(set) = self.by_hash.get(h) {
                targets.extend(set.iter().copied());
            }
        }
        for id in targets {
            // Node may already be gone if a previous prune in this batch
            // cascaded through it — skip silently.
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
        // Accumulated over every node, then booked once — see
        // `account_remove_counts`.
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
            // Re-check: cascading prune from a sibling may have already
            // removed this id.
            if self.nodes.contains_key(&id) {
                self.prune_cascade(id);
            }
        }
    }

    /// Detach `start` and walk up, pruning every ancestor that becomes
    /// empty + childless. Iterative — chains can be long.
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

    /// Read-only match path. Takes `&self` (not `&mut self`) so the public
    /// [`HashTree::match_prefix`] can hold only a read lock — matching is
    /// the routing hot path and write-locking it would serialise all
    /// routing decisions across tokio worker threads. `last_used` is an
    /// [`AtomicU64`] specifically so the touch-on-descend can happen
    /// through a shared reference.
    ///
    /// Note the asymmetry with [`TreeState::resolve_parent`] (used by
    /// `insert`): that function disambiguates a multi-candidate
    /// `parent_hash` by preferring a worker-owned node. This function has
    /// no worker context to do the same, so multiple candidates fall back
    /// to root. The asymmetry is intentional for v1; the public doc on
    /// [`HashTree::match_prefix`] documents the policy for callers.
    fn match_prefix(&self, parent_hash: Option<i64>, block_hashes: &[i64]) -> MatchResult {
        if block_hashes.is_empty() {
            return MatchResult::default();
        }
        // Determine starting node: root, or the unique node carrying
        // `parent_hash`. Multiple matches: bail to root (caller should
        // have a single canonical context).
        let start = match parent_hash {
            None => ROOT_ID,
            Some(p) => match self.by_hash.get(&p) {
                Some(set) if set.len() == 1 => *set.iter().next().unwrap(),
                _ => ROOT_ID,
            },
        };

        let mut current = start;
        let mut matched = 0usize;
        let mut last_match_node: Option<NodeId> = None;
        let now = now_millis();
        for &h in block_hashes {
            let next = self
                .nodes
                .get(&current)
                .and_then(|n| n.children.get(&h).copied());
            match next {
                Some(child_id) => {
                    // Touch as we descend. Atomic store under a shared
                    // borrow — no &mut needed.
                    if let Some(child) = self.nodes.get(&child_id) {
                        child.last_used.store(now, Ordering::Relaxed);
                    }
                    current = child_id;
                    matched += 1;
                    last_match_node = Some(child_id);
                }
                None => break,
            }
        }
        let tiers: HashMap<KvWorkerId, Tiers> = last_match_node
            .and_then(|id| self.nodes.get(&id))
            .map(|n| n.workers.clone())
            .unwrap_or_default();
        MatchResult {
            matched_blocks: matched,
            tiers,
        }
    }

    /// Contiguous prefix depth for every worker on the chain, in one descent.
    /// `insert` marks a worker at every node it descends, so "holds the first
    /// `d` blocks" means present at each of levels `1..=d`. A worker is frozen
    /// at the first level that omits it, so a `remove`d interior node stops the
    /// count at the hole instead of counting past it.
    fn prefix_depths(
        &self,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
    ) -> HashMap<KvWorkerId, usize> {
        if block_hashes.is_empty() {
            return HashMap::new();
        }
        // Same start resolution as `match_prefix`: an ambiguous `parent_hash`
        // has no worker context to disambiguate, so fall back to root.
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
                // Absent here means never in `alive`: a tail held without
                // block 0 is reported as holding nothing. Presence is "on any
                // tier" — a worker whose device copy was evicted but whose
                // host backup remains still holds the level.
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

    /// Approximate count of *non-root* nodes in the tree.
    fn node_count(&self) -> usize {
        // Subtract one for the root sentinel.
        self.nodes.len().saturating_sub(1)
    }

    fn evict_lru(&mut self, max_size: usize) -> usize {
        // Fast-path: already under cap.
        if self.node_count() <= max_size {
            return 0;
        }
        // Count by total node-count delta so cascade prunes (which may
        // remove multiple ancestors per `prune_cascade` call) are
        // accounted for accurately, not just the cascade entry point.
        let count_before = self.nodes.len();

        // Phase 1: drop empty (no-worker) leaves first. These hang around
        // only because of pruning races — they're free wins.
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
            if self.node_count() <= max_size {
                break;
            }
            if self.nodes.contains_key(&id) {
                self.prune_cascade(id);
            }
        }

        // Phase 2: evict oldest leaves (with workers) until we hit cap.
        // We re-snapshot leaves each pass because pruning can promote a
        // parent into "leaf" status. The outer loop bounds work to
        // O(node_count) so we don't spin on a degenerate tree.
        let mut iters = 0usize;
        let max_iters = self.nodes.len().saturating_add(1);
        while self.node_count() > max_size && iters < max_iters {
            iters += 1;
            // Find the LRU leaf. `last_used` is read with `Relaxed` —
            // approximate freshness is fine for eviction. Equality at
            // the millisecond boundary tie-breaks by NodeId.
            let mut oldest: Option<(u64, NodeId)> = None;
            for (&id, n) in &self.nodes {
                if id == ROOT_ID || !n.children.is_empty() {
                    continue;
                }
                let ts = n.last_used.load(Ordering::Relaxed);
                match oldest {
                    None => oldest = Some((ts, id)),
                    Some((cur, _)) if ts < cur => oldest = Some((ts, id)),
                    _ => {}
                }
            }
            let Some((_, victim)) = oldest else {
                break; // No leaves at all (shouldn't happen with non-empty tree).
            };
            // Force-prune even if the leaf still holds workers — eviction
            // intentionally evicts. We clear workers first so the cascade
            // precondition holds, releasing each carrier's tiers as we go so
            // the occupancy counters stay in step.
            let carriers: Vec<(KvWorkerId, Tiers)> = match self.nodes.get_mut(&victim) {
                Some(node) => node.workers.drain().collect(),
                None => Vec::new(),
            };
            for (worker, held) in &carriers {
                self.account_remove(worker, *held);
            }
            self.prune_cascade(victim);
        }
        count_before - self.nodes.len()
    }
}

/// Public hash-keyed radix tree. Cheap to clone an [`Arc`] of; the
/// underlying state is `Send + Sync` (single `RwLock`).
#[derive(Debug)]
pub struct HashTree {
    state: RwLock<TreeState>,
}

impl Default for HashTree {
    fn default() -> Self {
        Self::new()
    }
}

impl HashTree {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(TreeState::new()),
        }
    }

    /// Apply an untagged `BlockStored` event: a device store. See
    /// [`Self::insert_tiered`].
    pub fn insert(&self, worker: &KvWorkerId, parent_hash: Option<i64>, block_hashes: &[i64]) {
        self.insert_tiered(worker, parent_hash, block_hashes, Tiers::DEVICE);
    }

    /// Apply a `BlockStored` event on `tiers` (from its `medium`, via
    /// [`Tiers::for_store`]).
    ///
    /// Walks from `parent_hash`'s node (or root) and descends along
    /// `block_hashes`, marking every visited node as held by `worker` on
    /// `tiers` in addition to any tier it already holds there. Empty
    /// `block_hashes` or empty `tiers` is a no-op.
    pub fn insert_tiered(
        &self,
        worker: &KvWorkerId,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
        tiers: Tiers,
    ) {
        let mut state = self.state.write();
        state.insert(worker, parent_hash, block_hashes, tiers);
    }

    /// Apply an untagged `BlockRemoved` event: the worker loses the blocks on
    /// every tier. See [`Self::remove_tiered`].
    pub fn remove(&self, worker: &KvWorkerId, block_hashes: &[i64]) {
        self.remove_tiered(worker, block_hashes, Tiers::ALL);
    }

    /// Apply a `BlockRemoved` event for `tiers` (from its `medium`, via
    /// [`Tiers::for_remove`]).
    ///
    /// For every node carrying any hash in `block_hashes`, clear `tiers`
    /// from `worker`'s hold on it. The worker stays an owner of the node
    /// while it holds the block on any other tier — a device eviction after a
    /// host backup leaves the worker a host-tier owner. Once no tier remains
    /// the worker is dropped, and nodes that become empty AND childless are
    /// pruned (cascading upward).
    ///
    /// Removing the worker from a node does NOT remove the node if other
    /// workers still hold it.
    pub fn remove_tiered(&self, worker: &KvWorkerId, block_hashes: &[i64], tiers: Tiers) {
        let mut state = self.state.write();
        state.remove(worker, block_hashes, tiers);
    }

    /// Apply an `AllBlocksCleared` event for `worker`.
    pub fn clear_worker(&self, worker: &KvWorkerId) {
        let mut state = self.state.write();
        state.clear_worker(worker);
    }

    /// Find the longest path from the root that matches a prefix of
    /// `block_hashes`, optionally starting from the node carrying
    /// `parent_hash`.
    ///
    /// Returns the deepest matched node's worker set and how many blocks
    /// matched.
    ///
    /// As a side-effect, touches `last_used` on every node visited along
    /// the match — so frequently-matched paths are kept hot for
    /// [`HashTree::evict_lru`]. The touch is an atomic `Relaxed` store, so
    /// this method only needs a read lock and many threads can match
    /// concurrently.
    ///
    /// # Ambiguous `parent_hash`
    /// If `parent_hash == Some(p)` and `p` is carried by multiple nodes
    /// (the "same hash in two chains" case), this method cannot
    /// disambiguate and falls back to matching from the root. Callers
    /// that need a specific chain should split the request or call with
    /// `parent_hash = None`. (`insert` resolves the same ambiguity by
    /// preferring a worker-owned candidate; `match_prefix` has no worker
    /// context, so the asymmetry is intentional.)
    pub fn match_prefix(&self, parent_hash: Option<i64>, block_hashes: &[i64]) -> MatchResult {
        let state = self.state.read();
        state.match_prefix(parent_hash, block_hashes)
    }

    /// How many leading blocks of `block_hashes` each worker holds contiguously,
    /// in one descent under one read lock. [`Self::match_prefix`] names only the
    /// deepest matched node's holders, so it cannot answer this. Absent = none.
    pub fn prefix_depths(
        &self,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
    ) -> HashMap<KvWorkerId, usize> {
        let state = self.state.read();
        state.prefix_depths(parent_hash, block_hashes)
    }

    /// Approximate number of non-root nodes in the tree (the root sentinel
    /// is not counted). Useful for metrics and to decide when to call
    /// [`HashTree::evict_lru`].
    pub fn node_count(&self) -> usize {
        self.state.read().node_count()
    }

    /// Number of distinct block-hash keys carried by the reverse index.
    /// Exposed for invariant tests: when `node_count() == 0` this must
    /// also be 0. A nonzero value here with zero nodes means a `prune`
    /// path forgot to clean up `by_hash` and the index has leaked.
    pub fn reverse_index_size(&self) -> usize {
        self.state.read().by_hash.len()
    }

    /// Times the occupancy bookkeeping contradicted itself, by
    /// [`ACCOUNTING_REASONS`]. Always zero on a correct tree; rendered so a
    /// release build, where the debug assertion is compiled out, still says
    /// so out loud.
    pub fn accounting_errors(&self) -> [u64; ACCOUNTING_REASONS.len()] {
        self.state.read().accounting_errors
    }

    /// Nodes each carrier holds on each tier, sorted by carrier. Rendered as
    /// `sgl_router_kv_tree_blocks`; against the engine's own per-tier
    /// occupancy for the same pod it is the tree's coverage of that tier —
    /// the number that says whether a tier the engine holds is visible to
    /// routing at all (module docs, "Storage tiers").
    ///
    /// Read off the incrementally maintained accounting rather than walked,
    /// so a scrape costs one read lock. A carrier holding nothing has no row.
    pub fn tier_occupancy(&self) -> Vec<(KvWorkerId, TierCounts)> {
        let state = self.state.read();
        let mut rows: Vec<(KvWorkerId, TierCounts)> = state
            .occupancy
            .iter()
            .map(|(w, counts)| (w.clone(), *counts))
            .collect();
        rows.sort_by(|a, b| (&a.0.url, a.0.dp_rank).cmp(&(&b.0.url, b.0.dp_rank)));
        rows
    }

    /// Evict least-recently-used nodes until `node_count() <= max_size`.
    ///
    /// Strategy:
    /// 1. Drop already-empty leaves (no workers, no children) first.
    /// 2. If still over cap, evict oldest leaves (force-clearing workers
    ///    on the victim) and cascade-prune.
    ///
    /// Returns the exact total number of nodes pruned, including any
    /// ancestors removed by cascade-pruning. Suitable for wiring into a
    /// metric counter.
    pub fn evict_lru(&self, max_size: usize) -> usize {
        let mut state = self.state.write();
        state.evict_lru(max_size)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
impl HashTree {
    /// Whether every carrier on every node holds at least one tier. `remove`
    /// and `prune_cascade` rely on "no bits ⇒ no entry"; a violation means a
    /// node can never be pruned and a worker never dropped.
    fn debug_no_empty_carrier(&self) -> bool {
        let state = self.state.read();
        state
            .nodes
            .values()
            .all(|n| n.workers.values().all(|t| !t.is_empty()))
    }

    /// Recompute [`Self::tier_occupancy`] by walking every node — the oracle
    /// the incrementally maintained counters are checked against.
    ///
    /// Its ceiling: it shares `tally_tiers` and [`Tiers::SLOTS`] with the code
    /// it checks, so it proves only that incremental booking agrees with a
    /// full walk. It is blind by construction to WHICH tier is right — a
    /// SLOTS/bit mismatch would mis-tally identically on both sides. The
    /// example-based tier tests and the compile-time coupling asserts own
    /// that half.
    fn debug_recount_occupancy(&self) -> Vec<(KvWorkerId, TierCounts)> {
        let state = self.state.read();
        let mut total: HashMap<KvWorkerId, TierCounts> = HashMap::new();
        for (&id, node) in &state.nodes {
            if id == ROOT_ID {
                continue;
            }
            for (worker, held) in &node.workers {
                let acc = total.entry(worker.clone()).or_default();
                tally_tiers(acc, *held);
            }
        }
        let mut rows: Vec<(KvWorkerId, TierCounts)> = total.into_iter().collect();
        rows.sort_by(|a, b| (&a.0.url, a.0.dp_rank).cmp(&(&b.0.url, b.0.dp_rank)));
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(url: &str, dp_rank: u32) -> KvWorkerId {
        KvWorkerId {
            url: url.to_string(),
            dp_rank,
        }
    }

    fn workers(ids: &[&KvWorkerId]) -> HashSet<KvWorkerId> {
        ids.iter().map(|w| (*w).clone()).collect()
    }

    /// One descent answers for every worker at its *own* depth, where
    /// `match_prefix` names only whoever sits at the deepest matched node.
    /// Also pins that a `remove`d interior block stops the count at the hole.
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

    /// The write-back sequence the engine publishes for a backed-up block:
    /// device store, host store once the D2H copy lands, then a DEVICE-tagged
    /// removal when the device copy is evicted. The worker still holds the
    /// block on host, so it must stay an owner — just no longer a device one.
    /// Only the host-tagged removal ends ownership.
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

    /// The routing path reads `prefix_depths`, not `match_prefix`, so the
    /// tier fix has to show up there: a chain whose device copy was evicted
    /// after a host backup still scores at its full depth. The untagged
    /// removal is the control — it clears every tier and the depth collapses,
    /// which is what the tier-blind tree did for the tagged case too.
    #[test]
    fn prefix_depths_survive_a_device_eviction_with_a_host_backup() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert_tiered(&a, None, &[1, 2, 3], Tiers::for_store(Some("GPU")));
        tree.insert_tiered(&a, None, &[1, 2, 3], Tiers::for_store(Some("CPU_PINNED")));

        // Block 1 included on purpose. `prefix_depths` SEEDS its live set from
        // level 1 and narrows from there, so evicting only from block 2 onward
        // would leave that seed exercised against device-held state alone — and
        // a seed narrowed to device owners would send every host-only worker to
        // depth 0, restoring this bug with the suite still green.
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

    /// Untagged events keep the pre-tiering contract: a bare `BlockStored` is
    /// a device store, a bare `BlockRemoved` clears every tier at once. A
    /// publisher that never tags must see exactly the behaviour it always had.
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

    /// An unknown medium is asymmetric on purpose: the store is dropped (no
    /// tier at all), while a removal tagged with it clears every tier. Filing
    /// the store under a guess would make the worker a routing candidate on a
    /// tier this build cannot price; narrowing the removal to a bit that was
    /// never set would leave a stale owner forever.
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

    /// Tiers are per (node, worker): a device removal by one worker must not
    /// touch another worker's hold on the same node, and a node whose
    /// carriers differ by tier reports the device subset exactly.
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

    /// A store on no tier must not create a carrier: `remove` relies on
    /// "no bits ⇒ no entry" to know when a node is prunable.
    #[test]
    fn empty_tier_insert_is_noop() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert_tiered(&a, None, &[1], Tiers::default());
        assert_eq!(tree.node_count(), 0);
    }

    /// The per-tier occupancy is maintained incrementally at every mutation
    /// site, so it is checked against a full recount after a sequence that
    /// exercises all of them: tiered stores, partial and full removals,
    /// `clear_worker`, and LRU eviction.
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

        // Partial removal (device only) on a node held on both tiers, then a
        // removal that clears the last tier and prunes.
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

    /// `DISK` (L3) and `EXTERNAL` (L4) are distinct tiers in the engine's own
    /// `StorageMedium`, so they must not share a bit: on a fleet running both,
    /// folding them would make an L3 eviction erase the router's knowledge of
    /// the L4 copy.
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

    /// The occupancy counters are maintained incrementally at four mutation
    /// sites, which is exactly the shape that rots under a later refactor. A
    /// deterministic random walk over every operation and every medium, with
    /// both invariants asserted after EVERY step:
    ///
    /// * `tier_occupancy()` equals a full node-walk recount, and
    /// * no carrier entry with empty tiers exists (`remove` relies on
    ///   "no bits ⇒ no entry" to know when a node is prunable).
    #[test]
    fn occupancy_and_carrier_invariants_hold_under_a_random_walk() {
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
                // Sometimes anchor to a hash that may or may not be in the
                // tree, to exercise `resolve_parent`'s fallbacks.
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
            }
        }
    }

    /// `AllBlocksCleared` is the pod-restart / scale-down path
    /// (`remove_worker` clears every rank), so it must drop a carrier
    /// regardless of which tiers it held — a hold on any lower tier is still
    /// a hold. Asserted at the tree, not only through the rendered metric, so
    /// a metrics refactor cannot take the coverage with it.
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

    /// `resolve_parent` disambiguates a shared hash by preferring a candidate
    /// the worker already holds — on ANY tier. Narrowing that to device owners
    /// would re-anchor a continuation at the root once the worker's device
    /// copy was evicted, fragmenting the very prefix the host tier still
    /// serves. The shape is a shared system prompt whose block hash also
    /// appears in another chain.
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

        // Remove A from the node carrying hash=2. Per spec: that node loses
        // A; descendants are NOT recursively touched, but `match_prefix`
        // returns the deepest matched *node*'s worker set. Node 2 still
        // exists (it has child 3), but its worker set is now empty.
        tree.remove(&a, &[2]);

        // Node 2 still in tree (has child 3).
        // Match length 2 lands on node 2 (workers empty), so workers={}.
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2);
        assert!(m.workers().is_empty());

        // Match length 3 lands on node 3 (workers still has A).
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));

        // Reverse-index sanity for hash 2: still present (node holds it).
        {
            let st = tree.state.read();
            assert!(st.by_hash.contains_key(&2));
        }
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

        // [1,2] now has only B (A was the only other holder of node 2;
        // wait — actually A held 1 and 2 too. But B also holds 1 and 2.)
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
        // node_count() returns *non-root* count, so it should be 0.
        assert_eq!(tree.node_count(), 0);
        // Reverse index for these hashes should be empty.
        {
            let st = tree.state.read();
            assert!(!st.by_hash.contains_key(&1));
            assert!(!st.by_hash.contains_key(&2));
            assert!(!st.by_hash.contains_key(&3));
        }
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

        // Reverse index for hash 5 has 2 distinct nodes.
        {
            let st = tree.state.read();
            assert_eq!(st.by_hash.get(&5).map(|s| s.len()), Some(2));
        }

        // BlockRemoved [5] should remove A from BOTH nodes-carrying-5.
        // Both nodes are leaves, so both prune.
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
        // Two nodes both end up carrying hash=5 (same trick as the
        // "same hash in two chains" test).
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
        // C tries to extend with parent_hash=5; resolution should fall
        // back to root with the new chain rooted at hash=9.
        tree.insert(&c, Some(5), &[9]);

        // C is reachable as a fresh root child at hash=9.
        let m = tree.match_prefix(None, &[9]);
        assert_eq!(m.matched_blocks, 1);
        assert_eq!(m.workers(), workers(&[&c]));
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
        // Each leaf hangs directly off root, so cascade-pruning never
        // cascades past the leaf itself: count must equal exactly the
        // number of nodes we needed to drop.
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
        // Tiny sleep to force last_used differentiation at millisecond
        // resolution. The 2ms gap is generous vs. the 1ms tick.
        std::thread::sleep(std::time::Duration::from_millis(2));
        // Second chain: newer.
        tree.insert(&a, None, &[200, 201, 202]);

        // Match the newer chain to bump its last_used.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = tree.match_prefix(None, &[200, 201, 202]);

        // Force eviction down to 3 nodes; the older chain should go first.
        // The leaf 102 is the LRU; pruning it cascades up through 101 and
        // 100 (each becomes empty + childless), so a single victim drops
        // the whole older chain — exactly 3 nodes evicted.
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
        // BlockStored carrying multiple hashes: each chains off its
        // predecessor, and parent_hash applies to the FIRST.
        tree.insert(&a, None, &[10, 20, 30]);

        let m = tree.match_prefix(None, &[10, 20, 30]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers(), workers(&[&a]));

        // Confirm parent_block_hash chain: node carrying 30 should record
        // parent_block_hash = Some(20).
        let st = tree.state.read();
        let n30_id = *st.by_hash.get(&30).unwrap().iter().next().unwrap();
        assert_eq!(st.nodes[&n30_id].parent_block_hash, Some(20));
        let n20_id = *st.by_hash.get(&20).unwrap().iter().next().unwrap();
        assert_eq!(st.nodes[&n20_id].parent_block_hash, Some(10));
        let n10_id = *st.by_hash.get(&10).unwrap().iter().next().unwrap();
        assert_eq!(st.nodes[&n10_id].parent_block_hash, None);
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
}
