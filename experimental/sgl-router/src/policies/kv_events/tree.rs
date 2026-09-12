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
//! # Concurrency: sharded so the event-write path stops blocking matches
//!
//! WHY this is sharded: the routing hot path (`prefix_depths` /
//! `match_prefix`) takes a read lock while the ZMQ KV-event pump takes a
//! write lock per event per worker (`insert` / `remove` / `clear_worker`).
//! With a single process-wide lock every write blocks every concurrent
//! routing match, and under load the per-request routing overhead grows
//! several-fold. We therefore split the tree into [`N_SHARDS`] independent
//! [`TreeState`]s, each behind its own [`parking_lot::RwLock`], keyed by
//! the chain's ROOT block hash.
//!
//! A radix chain is rooted at its first block hash and lives entirely
//! inside one shard, so an `insert(parent_hash = None, [h0, h1, …])` and a
//! `match_prefix(None, [h0, …])` both touch only `shard_of(h0)`. A write to
//! one chain blocks only readers walking a chain in the same shard; readers
//! on every other root proceed in parallel. The integer-keyed maps
//! (`children`, `by_hash`, `nodes` — trusted block hashes and node ids) use
//! [`FxHashMap`] / [`FxHashSet`]: the DoS-resistant SipHash default is pure
//! overhead on these keys. Worker-keyed maps stay on the std hasher.
//!
//! ## Routing operations across shards
//!
//! * `parent_hash == None` → the shard of `block_hashes[0]` (or, for the
//!   match path, the empty-input early return).
//! * `parent_hash == Some(p)` → the shard whose local reverse index carries
//!   `p`. `insert` continuations need this so a chain extending an existing
//!   one lands in the same shard and stays whole; if `p` is not in any
//!   shard (parent absent), we fall back exactly as the single-shard code
//!   did — `insert` roots the new chain at its own first hash, the match
//!   path starts from the root of `shard_of(block_hashes[0])`. In practice
//!   the match path is only ever called with `None` (both production
//!   callers route from root); the `Some` scatter exists only for `insert`
//!   continuations and whitebox tests, neither on the hot path.
//! * `remove([h, …])` and `clear_worker(w)` may touch state in several
//!   shards (the same hash can be a chain root in one shard and an interior
//!   block in another; a worker can hold chains in many shards), so they
//!   fan out across all shards.
//! * `node_count` / `reverse_index_size` / `evict_lru` aggregate across all
//!   shards; `evict_lru` enforces a single global node cap (see its doc).
//!
//! ## One writer, many readers (why the cross-shard scans are safe)
//!
//! Every mutation — `insert` / `remove` / `clear_worker` / `evict_lru` —
//! spans several shards without holding them all at once: `route_insert` /
//! `route_match` scan shards by taking and releasing each shard's read lock
//! in turn, then write-lock the one they chose, while `remove` /
//! `clear_worker` / `evict_lru` walk the shards one at a time. None of that
//! is a consistent snapshot, so a second concurrent writer can invalidate a
//! routing decision between the scan and the write — a continuation whose
//! chosen parent node is pruned in the gap gets re-rooted at the chosen
//! SHARD's root sentinel, which is not `shard_of(block_hashes[0])`, and the
//! chain is then unreachable from `match_prefix(None, …)` forever.
//!
//! The mutators are NOT all on one task: the KV-event pump drives
//! `insert` / `remove` / `clear_worker` from its own task, and
//! `KvEventIndex::remove_worker` (service discovery, worker scale-down)
//! calls `clear_worker` from a different one. [`HashTree::writer`]
//! therefore serialises writers explicitly, so the scan and the write it
//! feeds are atomic with respect to every other mutation. Readers never
//! take it: the match path stays lock-free of everything but the single
//! shard read lock, which is the whole point of sharding.
//!
//! # Reverse index
//!
//! `BlockRemoved` events carry only `block_hashes` and no parent context,
//! so without an index from `block_hash → set of nodes carrying that hash`
//! we'd have to walk the whole tree. We maintain that reverse index per
//! shard as [`TreeState::by_hash`]. The same hash can legitimately appear
//! at multiple positions (e.g. as the last block of one chain and as the
//! second block of another) within a shard, so each entry is a *set* of
//! node IDs.
//!
//! # Pruning
//!
//! When a worker is dropped from a node and the node has no remaining
//! workers AND no children, we detach it from its parent and remove it
//! from the reverse index. Pruning cascades upward iteratively (chains
//! can be deep — the recursive form would risk stack-overflow for
//! pathological inputs). Pruning is shard-local: a chain never crosses a
//! shard boundary.
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

use parking_lot::{Mutex, RwLock};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, error, warn};

/// Number of independent tree shards. A power of two so shard selection is
/// a mask, not a modulo. Sized generously relative to typical fleet sizes
/// so distinct chains rarely collide on a shard while keeping per-shard
/// overhead (one `RwLock` + one arena) negligible.
const N_SHARDS: usize = 32;

// `shard_of` shifts by `64 - log2(N_SHARDS)` and indexes
// `shards[..N_SHARDS]`, both of which are only correct for a power of
// two ≥ 2. Make a bad value a compile error rather than a runtime panic /
// out-of-bounds.
const _: () = assert!(
    N_SHARDS.is_power_of_two() && N_SHARDS >= 2,
    "N_SHARDS must be a power of two and at least 2",
);

/// Multiplicative hash constant (Fibonacci hashing) used to spread chain
/// roots across shards. A single worker emits many distinct chains; mixing
/// the root hash keeps that write load from piling onto one shard.
const SHARD_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Map a chain-root block hash to its shard index.
fn shard_of(root_hash: i64) -> usize {
    let mixed = (root_hash as u64).wrapping_mul(SHARD_MIX);
    // Top bits of a multiplicative hash are the best-mixed; fold them down
    // to the shard count.
    (mixed >> (64 - N_SHARDS.trailing_zeros())) as usize
}

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
    /// The `StorageMedium` strings SGLang puts on the wire
    /// (`python/sglang/srt/disaggregation/kv_events.py`) and the tier each
    /// lands on. The single source for the tree's ranking.
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
    /// costs at most one cold prefill, and it must not be silent: an engine
    /// that starts publishing a medium this build predates would otherwise
    /// go invisible to cache-aware routing with nothing in the logs, so each
    /// unrecognised string is warned about once per process.
    pub fn for_store(medium: Option<&str>) -> Tiers {
        match medium {
            None => Self::DEVICE,
            Some(m) => Self::known(m).unwrap_or_else(|| {
                warn_unknown_medium(m);
                Tiers::default()
            }),
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

    /// The bits set in either. `const` so the tier tables can be built from
    /// the individual tiers rather than from raw bit arithmetic.
    pub const fn union(self, other: Tiers) -> Tiers {
        Tiers(self.0 | other.0)
    }
}

// `ALL` has to stay in step with the individual tiers, and nothing about
// adding a `pub const` tier would otherwise force it: a bit missing from
// `ALL` is a bit an untagged `BlockRemoved` never clears — a permanently
// stale owner, the one failure `for_remove` exists to prevent.
const _: () = {
    let mut i = 0;
    while i < Tiers::WIRE_MEDIA.len() {
        assert!(
            Tiers::ALL.contains(Tiers::WIRE_MEDIA[i].1),
            "every wire medium must map to a tier `ALL` clears",
        );
        i += 1;
    }
};

/// Warn once per process for each `medium` string this build cannot rank.
/// The set is only touched on the unknown path, so a fleet publishing known
/// media never pays for it; a fleet publishing one unknown medium pays a
/// single uncontended lock per event, on the pump task.
fn warn_unknown_medium(medium: &str) {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let mut seen = SEEN.get_or_init(Default::default).lock();
    if seen.insert(medium.to_string()) {
        warn!(
            medium,
            known = ?Tiers::WIRE_MEDIA.map(|(name, _)| name),
            "kv-events: BlockStored on an unrecognised storage medium; \
             dropping the store rather than filing it under a guessed tier. \
             Prefixes held only on this medium will not be routable.",
        );
    }
}

/// Add `tiers` to `worker`'s hold in `carriers`, creating the entry on first
/// sight. One lookup on the re-store path — under a hierarchical cache the
/// host backup of a chain the worker already holds on device, the common
/// case — and two on first sight. Never leaves an entry with no bits, which
/// [`TreeState::remove`] relies on.
fn add_tiers(carriers: &mut HashMap<KvWorkerId, Tiers>, worker: &KvWorkerId, tiers: Tiers) {
    match carriers.get_mut(worker) {
        Some(held) => held.insert(tiers),
        None => {
            carriers.insert(worker.clone(), tiers);
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
    /// Whether `worker` holds the deepest matched node on any tier.
    /// Prefer this to `workers().contains(…)`, which clones every carrier
    /// (each one a `String` allocation) to answer a membership question.
    pub fn holds(&self, worker: &KvWorkerId) -> bool {
        self.tiers.contains_key(worker)
    }

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
/// We use an arena (`FxHashMap<NodeId, Node>`) instead of `Arc<RwLock<Node>>`
/// + `Weak` because:
/// 1. We need to enumerate every node (e.g. for `clear_worker` and
///    `evict_lru`); a flat map is direct and cheap.
/// 2. The reverse index needs a *stable* key per node — `Weak` would force
///    upgrades on every lookup and complicate prune semantics.
///
/// Node ids are unique *within a shard* only; the shard a node lives in is
/// implied by the chain root, never stored.
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

/// Inner mutable state of one shard. Cross-method invariants:
///
/// * `nodes[ROOT_ID]` is always present and is the only node with
///   `parent == None`.
/// * For every non-root node `n`: `nodes[n.parent].children[&n.block_hash]
///   == n`'s id (i.e., parent's child pointer round-trips).
/// * `by_hash[h]` contains the id of every non-root node `n` with
///   `n.block_hash == h`. Root is never in `by_hash`.
/// * Pruning runs after every worker-removal that empties a node: prune
///   detaches from parent, removes from `by_hash`, and recurses upward.
///
/// Node ids are unique within this shard only; two shards may both mint
/// id 1.
#[derive(Debug)]
struct TreeState {
    nodes: FxHashMap<NodeId, Node>,
    by_hash: FxHashMap<i64, FxHashSet<NodeId>>,
    next_id: NodeId,
}

const ROOT_ID: NodeId = 0;
/// Sentinel block_hash for the root. Real workers can in principle emit
/// `i64::MIN`, but the root is never looked up via `by_hash` so collisions
/// don't matter.
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
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
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

    /// Pick the parent node id for an incoming `BlockStored` event, given
    /// that this shard is already known to own (or be the fallback for)
    /// the chain.
    ///
    /// Resolution order:
    /// 1. `parent_hash == None` → root.
    /// 2. There's exactly one node carrying `parent_hash` → use it.
    /// 3. Multiple candidates: prefer one already containing `worker`.
    /// 4. None contain the worker: log at debug, fall back to root. The
    ///    new chain still carries `parent_hash` on its first node so that
    ///    if the parent's `BlockStored` arrives later we can reconstruct
    ///    the link via the reverse index.
    ///
    /// [`HashTree::route_insert`] resolves cases 1 and 4 globally and
    /// passes `None` for them, so in practice only 2 and 3 are reached;
    /// the root fallbacks stay as defence against a routing/local
    /// disagreement, which under [`HashTree::writer`] cannot happen.
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
        for &h in block_hashes {
            let child_id = match self
                .nodes
                .get(&current)
                .and_then(|n| n.children.get(&h).copied())
            {
                Some(id) => id,
                None => match self.create_child(current, h, prev_hash) {
                    Some(id) => id,
                    None => return,
                },
            };
            let Some(child) = self.nodes.get_mut(&child_id) else {
                error!(
                    child_id,
                    block_hash = h,
                    "tree invariant violation: child node missing immediately after fetch/create; aborting chain",
                );
                return;
            };
            add_tiers(&mut child.workers, worker, tiers);
            child.last_used.store(now, Ordering::Relaxed);
            current = child_id;
            prev_hash = Some(h);
        }
    }

    /// Whether this shard's reverse index carries any of `block_hashes` —
    /// i.e. whether [`Self::remove`] could do anything here. Lets
    /// [`HashTree::remove_tiered`] skip a shard under a read lock instead
    /// of write-locking it to discover there was nothing to do.
    fn carries_any(&self, block_hashes: &[i64]) -> bool {
        block_hashes.iter().any(|h| self.by_hash.contains_key(h))
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
            let prunable = match self.nodes.get_mut(&id) {
                Some(node) => {
                    if let Some(held) = node.workers.get_mut(worker) {
                        held.remove(tiers);
                        if held.is_empty() {
                            node.workers.remove(worker);
                        }
                    }
                    node.workers.is_empty() && node.children.is_empty()
                }
                None => false,
            };
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
        for id in ids {
            if let Some(node) = self.nodes.get_mut(&id) {
                if node.workers.remove(worker).is_some()
                    && node.workers.is_empty()
                    && node.children.is_empty()
                {
                    prune_candidates.push(id);
                }
            }
        }
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
                }
                None => break,
            }
        }
        // `current` is the deepest matched node once anything matched; with
        // nothing matched it is still `start`, whose carriers belong to the
        // caller-supplied parent and must not be reported.
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

    /// Count of *non-root* nodes in this shard.
    fn node_count(&self) -> usize {
        // Subtract one for the root sentinel.
        self.nodes.len().saturating_sub(1)
    }

    /// Drop already-empty (no-worker, no-child) leaves in this shard. These
    /// hang around only because of pruning races — they're free wins.
    /// Returns the number of nodes dropped (including cascade ancestors).
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

    /// Timestamp + shard-local id of the LRU leaf, for global eviction
    /// ordering. `None` if the shard has no leaves. Ties at the millisecond
    /// boundary break deterministically by `NodeId` (the single global cap
    /// is preserved; the per-millisecond victim among equal timestamps is
    /// an arbitrary-but-deterministic choice). `last_used` is read with
    /// `Relaxed` — approximate freshness is fine for eviction.
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

    /// Force-evict the leaf with shard-local id `victim` (clearing its
    /// workers first so the cascade precondition holds) and cascade-prune.
    /// Returns the number of nodes removed, or 0 if the node is gone or is
    /// no longer a leaf (raced away between selection and eviction).
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
        if let Some(node) = self.nodes.get_mut(&victim) {
            node.workers.clear();
        }
        self.prune_cascade(victim);
        count_before - self.nodes.len()
    }
}

/// Public hash-keyed radix tree. Cheap to clone an [`Arc`] of; the
/// underlying state is `Send + Sync`. Split into [`N_SHARDS`] independent
/// [`TreeState`]s keyed by chain root, with writers serialised by
/// [`Self::writer`] — see the module docs for both.
#[derive(Debug)]
pub struct HashTree {
    shards: Vec<RwLock<TreeState>>,
    /// Held for the whole of every mutation so that a multi-shard write
    /// (and the cross-shard routing scan that chose its target) is atomic
    /// against other writers. Readers never take it — see the module docs,
    /// "One writer, many readers".
    writer: Mutex<()>,
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
        }
    }

    /// Resolve an `insert`'s `parent_hash` GLOBALLY, replicating the
    /// single-tree `resolve_parent` decision across all shards, and return
    /// `(shard, effective_parent_hash)` for the local insert.
    ///
    /// WHY return an effective parent: the single-tree `resolve_parent`
    /// decides "attach under node X" or "attach at root" from the COMPLETE
    /// set of nodes carrying `parent_hash`. A per-shard `resolve_parent`
    /// only sees its own slice of that set, so when the global decision is
    /// "attach at root" but the chosen root shard ALSO happens to carry
    /// `parent_hash` in exactly one local node, the local resolve would
    /// wrongly attach under it. To stay byte-for-byte identical we pass
    /// `None` to the local insert whenever the global decision was a
    /// root-attach, forcing the shard to root the chain regardless of its
    /// partial reverse-index view.
    ///
    /// `block_hashes` is non-empty (the caller early-returns on empty).
    /// Resolution mirrors `TreeState::resolve_parent`:
    /// 1. `parent_hash == None` → root shard, parent `None`.
    /// 2. No shard carries `parent_hash` → root shard, parent `None`.
    /// 3. Exactly one node (in one shard) carries it → that shard, keep
    ///    `parent_hash` (the unique local node is the parent).
    /// 4. Multiple nodes carry it → a `worker`-owned carrier's shard
    ///    keeping `parent_hash` ("owned" = holds it on ANY tier, matching
    ///    the single-tree rule); if none is owned → root shard, parent
    ///    `None` (single-tree "attach to root" fallback).
    ///
    /// Only invoked on `insert`; the match hot path never scatters.
    fn route_insert(
        &self,
        worker: &KvWorkerId,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
    ) -> (usize, Option<i64>) {
        let root_shard = shard_of(block_hashes[0]);
        let Some(p) = parent_hash else {
            return (root_shard, None);
        };
        // Gather, across shards, how many nodes carry `p` and which shard
        // (if any) holds a node `worker` already owns.
        let mut total_carriers = 0usize;
        let mut single_carrier_shard: Option<usize> = None;
        let mut worker_owned_shard: Option<usize> = None;
        for (idx, shard) in self.shards.iter().enumerate() {
            let st = shard.read();
            let Some(ids) = st.by_hash.get(&p) else {
                continue;
            };
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
        match total_carriers {
            // Parent absent everywhere → attach at the new chain's own root.
            0 => (root_shard, None),
            // Unique carrier → its shard; the local node IS the parent.
            1 => (single_carrier_shard.unwrap_or(root_shard), Some(p)),
            // Ambiguous: a worker-owned carrier keeps the parent link; with
            // none owned the single-tree path attaches at root — force-root
            // locally so a single-carrier root shard can't re-derive a node.
            _ => match worker_owned_shard {
                Some(idx) => (idx, Some(p)),
                None => (root_shard, None),
            },
        }
    }

    /// Resolve the match path's start point GLOBALLY and return
    /// `(shard, effective_parent_hash)`. The hot path uses
    /// `parent_hash == None` and never scatters; the rarely-exercised
    /// `Some(p)` form mirrors the single-tree rule (start from the UNIQUE
    /// node carrying `p`, else from root). As with `route_insert`, the
    /// "else from root" cases pass `None` to the local match so a
    /// single-carrier root shard cannot re-derive a node from its partial
    /// reverse index.
    ///
    /// `block_hashes` is non-empty (the caller early-returns on empty).
    fn route_match(&self, parent_hash: Option<i64>, block_hashes: &[i64]) -> (usize, Option<i64>) {
        let root_shard = shard_of(block_hashes[0]);
        let Some(p) = parent_hash else {
            return (root_shard, None);
        };
        // The single-tree match path only honors a UNIQUE carrier of `p`;
        // zero or multiple → root.
        let mut total = 0usize;
        let mut only_shard: Option<usize> = None;
        for (idx, shard) in self.shards.iter().enumerate() {
            if let Some(ids) = shard.read().by_hash.get(&p) {
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
        if block_hashes.is_empty() || tiers.is_empty() {
            return;
        }
        // The routing scan and the write it feeds must be one critical
        // section: a concurrent prune between them would re-root the chain
        // under the chosen shard's sentinel and orphan it.
        let _writer = self.writer.lock();
        let (idx, effective_parent) = self.route_insert(worker, parent_hash, block_hashes);
        self.shards[idx]
            .write()
            .insert(worker, effective_parent, block_hashes, tiers);
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
    ///
    /// A removed hash can be a chain root in one shard and an interior
    /// block of a chain rooted elsewhere in another, so this fans out
    /// across all shards. A shard that carries none of the hashes is
    /// skipped under a READ lock: `BlockRemoved` is roughly as frequent as
    /// `BlockStored`, and write-locking all [`N_SHARDS`] on every one of
    /// them would block every concurrent routing match — the exact cost the
    /// sharding exists to remove. Safe because [`Self::writer`] is held, so
    /// no other writer can add a carrier between the check and the write.
    pub fn remove_tiered(&self, worker: &KvWorkerId, block_hashes: &[i64], tiers: Tiers) {
        if block_hashes.is_empty() {
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

    /// Apply an `AllBlocksCleared` event for `worker`.
    ///
    /// A worker can hold chains in many shards, so this fans out across
    /// all shards. Also the scale-down path
    /// (`KvEventIndex::remove_worker`), which runs on a different task than
    /// the KV-event pump — hence [`Self::writer`].
    pub fn clear_worker(&self, worker: &KvWorkerId) {
        let _writer = self.writer.lock();
        for shard in &self.shards {
            shard.write().clear_worker(worker);
        }
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
    /// this method only needs a read lock on a single shard and many
    /// threads can match concurrently across shards.
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
        if block_hashes.is_empty() {
            return MatchResult::default();
        }
        let (idx, effective_parent) = self.route_match(parent_hash, block_hashes);
        self.shards[idx]
            .read()
            .match_prefix(effective_parent, block_hashes)
    }

    /// How many leading blocks of `block_hashes` each worker holds
    /// contiguously, in one descent under one shard's read lock.
    /// [`Self::match_prefix`] names only the deepest matched node's holders,
    /// so it cannot answer this. Absent = none.
    pub fn prefix_depths(
        &self,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
    ) -> HashMap<KvWorkerId, usize> {
        if block_hashes.is_empty() {
            return HashMap::new();
        }
        let (idx, effective_parent) = self.route_match(parent_hash, block_hashes);
        self.shards[idx]
            .read()
            .prefix_depths(effective_parent, block_hashes)
    }

    /// Number of non-root nodes across all shards (root sentinels are not
    /// counted), summed under a per-shard read lock. Exact under the
    /// single-writer pump (module docs); a point-in-time sum, not one
    /// consistent instant across shards. Useful for metrics and to decide
    /// when to call [`HashTree::evict_lru`].
    pub fn node_count(&self) -> usize {
        self.shards.iter().map(|s| s.read().node_count()).sum()
    }

    /// Number of distinct block-hash keys carried by the reverse index,
    /// summed across shards. A given hash value can appear in more than
    /// one shard (root of one chain, interior of another), and each
    /// occurrence is counted once per shard — consistent with the
    /// per-shard reverse indexes being independent.
    ///
    /// Exposed for invariant tests: when `node_count() == 0` this must
    /// also be 0. A nonzero value here with zero nodes means a `prune`
    /// path forgot to clean up `by_hash` and the index has leaked.
    pub fn reverse_index_size(&self) -> usize {
        self.shards.iter().map(|s| s.read().by_hash.len()).sum()
    }

    /// Evict least-recently-used nodes until `node_count() <= max_size`
    /// across the whole tree.
    ///
    /// Strategy:
    /// 1. Drop already-empty leaves (no workers, no children) in every
    ///    shard.
    /// 2. If still over cap, repeatedly evict the globally-oldest leaf —
    ///    found by comparing each shard's LRU leaf — force-clearing its
    ///    workers and cascade-pruning, until the global count is at the
    ///    cap.
    ///
    /// Returns the exact total number of nodes pruned, including any
    /// ancestors removed by cascade-pruning. Suitable for wiring into a
    /// metric counter.
    ///
    /// A per-shard quota would instead evict hot entries in a busy shard
    /// while idle shards sit under quota, changing which nodes survive; the
    /// global LRU keeps eviction order equivalent to the single-tree
    /// behavior. The cap check and LRU selection take per-shard locks one
    /// at a time, so [`Self::writer`] is held throughout — otherwise a
    /// concurrent inserter would make the cap a best-effort target rather
    /// than a hard postcondition. Not on the hot path (it runs
    /// periodically), so briefly read/write-locking each shard is
    /// acceptable.
    pub fn evict_lru(&self, max_size: usize) -> usize {
        let _writer = self.writer.lock();
        // Fast-path: already under cap.
        let mut remaining = self.node_count();
        if remaining <= max_size {
            return 0;
        }
        let mut pruned = 0usize;

        // Phase 1: free empty leaves everywhere. `remaining` tracks the
        // global count as we go, so the cap check between shards is free
        // rather than another all-shard `node_count()` scan.
        for shard in &self.shards {
            let dropped = shard.write().drop_empty_leaves();
            pruned += dropped;
            remaining -= dropped;
            if remaining <= max_size {
                return pruned;
            }
        }

        // Phase 2: evict the globally-oldest leaf one at a time. Bound the
        // loop by the total node count so a degenerate tree can't spin.
        let mut iters = 0usize;
        let max_iters = remaining.saturating_add(1);
        while remaining > max_size && iters < max_iters {
            iters += 1;
            // Pick the shard whose LRU leaf is globally oldest. Tie-break
            // by (timestamp, shard-local node id, shard index) so the
            // choice is deterministic.
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
                // The chosen leaf is no longer a leaf. Re-scan on the next
                // iteration rather than spin on a stale pick.
                continue;
            }
            pruned += dropped;
            remaining -= dropped;
        }
        pruned
    }
}

// ---------------------------------------------------------------------------
// Whitebox test helpers
//
// WHY these exist: the in-module tests assert on internal structure
// (reverse index membership, `parent_block_hash` chaining). State is now
// split across shards, so these helpers aggregate the per-shard layout
// while the tests' behavioral assertions stay identical in meaning.
// ---------------------------------------------------------------------------

#[cfg(test)]
impl HashTree {
    /// Whether every carrier on every node in every shard holds at least
    /// one tier. `remove` and `prune_cascade` rely on "no bits ⇒ no entry";
    /// a violation means a node can never be pruned and a worker never
    /// dropped.
    fn debug_no_empty_carrier(&self) -> bool {
        self.shards.iter().all(|s| {
            s.read()
                .nodes
                .values()
                .all(|n| n.workers.values().all(|t| !t.is_empty()))
        })
    }

    /// Whether every chain root (child of a shard's root sentinel) lives in
    /// `shard_of` its own block hash. This is the invariant
    /// `match_prefix(None, …)` rests on: it looks only in
    /// `shard_of(block_hashes[0])`, so a chain rooted anywhere else can
    /// never be matched again.
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

    /// Total number of distinct nodes carrying `hash`, summed across
    /// shards.
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

    /// `parent_block_hash` recorded on the (assumed unique) node carrying
    /// `hash`. Panics if `hash` is carried by zero or more than one node
    /// (the tests that use it construct unambiguous chains).
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    /// Two invariants that every mutation site has to preserve — exactly the
    /// shape that rots under a later refactor — asserted after EVERY step of
    /// a deterministic random walk over every operation and every medium:
    ///
    /// * "no bits ⇒ no entry", which `remove` and `prune_cascade` rely on to
    ///   know when a node is prunable.
    /// * every chain root sits in `shard_of` its own hash, which is the only
    ///   reason `match_prefix(None, ..)` can find it by looking in one shard.
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

    /// `AllBlocksCleared` is the pod-restart / scale-down path
    /// (`remove_worker` clears every rank), so it must drop a carrier
    /// regardless of which tiers it held — a hold on any lower tier is still
    /// a hold.
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

        // Reverse index for hash 5 has 2 distinct nodes (the two chains
        // have different roots, so they may live in different shards — the
        // count sums across shards).
        assert_eq!(tree.debug_hash_node_count(5), 2);

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

    /// Two chains whose ROOT hashes collide into the SAME shard must stay
    /// fully independent — distinct worker sets, independent match,
    /// independent remove. Every multi-root test above deliberately SPREADS
    /// roots across shards; this pins the colliding case the sharding rests
    /// on.
    #[test]
    fn colliding_roots_in_same_shard_stay_independent() {
        // Premise: roots 1 and 22 hash to the same shard. Guarded so the
        // test fails loudly (not silently no-ops) if N_SHARDS / SHARD_MIX
        // change.
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

    /// Regression: the unowned-ambiguous `parent_hash` fallback must
    /// attach the new chain at ROOT even when the new chain's first hash
    /// happens to route to a shard that locally carries `parent_hash` in
    /// exactly one node. The hashes here are chosen so that roots 1 and 2
    /// land on different shards (both carrying hash 5), while the
    /// continuation's first hash 1009 routes to the SAME shard as root 1 —
    /// the case where a naive per-shard resolve would wrongly attach 1009
    /// under that shard's node-5 instead of root.
    #[test]
    fn unowned_ambiguous_parent_force_roots_even_on_carrier_shard() {
        // Guard the premise so the test still pins the right case if the
        // shard count / mix ever changes (it would just need new constants).
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

        // C (owns neither node-5) extends parent_hash=5 with [1009].
        // Single-tree behavior: two carriers of 5, none C-owned → attach
        // at root → 1009 becomes a fresh root child.
        tree.insert(&c, Some(5), &[1009]);

        // 1009 must be a root child (matched=1), NOT hanging under 1->5.
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

    /// Two concurrent writers must not be able to orphan a chain.
    ///
    /// `insert` picks its target shard from a scan that takes and releases
    /// each shard's read lock in turn, then write-locks the one it chose.
    /// `KvEventIndex::remove_worker` calls `clear_worker` from the service-
    /// discovery task, NOT the KV-event pump, so a prune can land in that
    /// gap. Unsynchronised, the insert then finds its chosen parent gone and
    /// re-roots the chain under the CHOSEN SHARD's sentinel — which is not
    /// `shard_of(block_hashes[0])`, so `match_prefix(None, …)` can never
    /// reach it again and the nodes sit there forever.
    #[test]
    fn concurrent_writers_never_orphan_a_chain() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let tree = Arc::new(HashTree::new());
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        // `h0` must route to a different shard than the chain root `r`, so
        // a wrongly-rooted continuation is observable.
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
        // parent_block_hash = Some(20), 20 -> Some(10), 10 -> None.
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

    /// Distinct chain roots land on different shards (with high
    /// probability over 64 roots and 32 shards) yet `match_prefix` /
    /// `node_count` / eviction stay correct — the routing invariant the
    /// sharding relies on.
    #[test]
    fn distinct_roots_spread_across_shards() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        // Insert 64 independent two-block chains rooted at distinct hashes.
        for r in 0..64i64 {
            tree.insert(&a, None, &[r * 1000, r * 1000 + 1]);
        }
        assert_eq!(tree.node_count(), 128);

        // Confirm the roots actually used more than one shard (else the
        // test wouldn't be exercising cross-shard routing).
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
