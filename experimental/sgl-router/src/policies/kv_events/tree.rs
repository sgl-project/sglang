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
//! WHY this is sharded: the routing hot path (`match_prefix`) takes a read
//! lock while the ZMQ KV-event pump takes a write lock per event per worker
//! (`insert` / `remove` / `clear_worker`). With a single process-wide lock
//! every write blocks every concurrent routing match, and under load the
//! per-request routing overhead grows several-fold. We therefore split the
//! tree into [`N_SHARDS`] independent [`TreeState`]s, each behind its own
//! [`parking_lot::RwLock`], keyed by the chain's ROOT block hash.
//!
//! A radix chain is rooted at its first block hash and lives entirely inside
//! one shard, so an `insert(parent_hash = None, [h0, h1, …])` and a
//! `match_prefix(None, [h0, …])` both touch only `shard_of(h0)`. A write to
//! one chain blocks only readers walking a chain in the same shard; readers
//! on every other root proceed in parallel. The maps use [`FxHashMap`] /
//! [`FxHashSet`] (the keys are trusted block hashes and node ids, so the
//! DoS-resistant SipHash default is pure overhead).
//!
//! ## Routing operations across shards
//!
//! * `parent_hash == None` → the shard of `block_hashes[0]` (or, for
//!   `match_prefix`, the empty-input early return).
//! * `parent_hash == Some(p)` → the shard whose local reverse index carries
//!   `p`. `insert` continuations need this so a chain extending an existing
//!   one lands in the same shard and stays whole; if `p` is not in any shard
//!   (parent absent), we fall back exactly as the single-shard code did —
//!   `insert` roots the new chain at its own first hash, `match_prefix`
//!   matches from the root of `shard_of(block_hashes[0])`. In practice
//!   `match_prefix` is only ever called with `None` (the sole production
//!   caller routes from root); the
//!   `Some` scatter exists only for `insert` continuations and whitebox
//!   tests, neither on the hot path.
//! * `remove([h, …])` and `clear_worker(w)` may touch state in several
//!   shards (the same hash can be a chain root in one shard and an interior
//!   block in another; a worker can hold chains in many shards), so they
//!   fan out across all shards.
//! * `node_count` / `reverse_index_size` / `evict_lru` aggregate across all
//!   shards; `evict_lru` enforces a single global node cap (see its doc).
//!
//! ## One writer, many readers (why the cross-shard scans are safe)
//!
//! All mutation — `insert` / `remove` / `clear_worker` / `evict_lru` — runs on
//! the SINGLE KV-event pump task (`super::index` drains one `mpsc::Receiver`,
//! applying every worker's events serially). Only `match_prefix` runs
//! concurrently with it, and `match_prefix` never mutates. So although
//! `route_insert` / `route_match` scan shards by taking and releasing each
//! shard's read lock in turn — not one consistent snapshot — no other writer
//! can change the carrier set between that scan and the targeted write. This
//! single-writer property is what keeps cross-shard routing decisions
//! equivalent to the old single-lock tree; a second concurrent writer would
//! break it and would need explicit cross-shard synchronization.
//!
//! # Reverse index
//!
//! `BlockRemoved` events carry only `block_hashes` and no parent context,
//! so without an index from `block_hash → set of nodes carrying that hash`
//! we'd have to walk the whole tree. We maintain that reverse index per
//! shard as [`TreeState::by_hash`]. The same hash can legitimately appear at
//! multiple positions (e.g. as the last block of one chain and the second
//! block of another) within a shard, so each entry is a *set* of node ids.
//!
//! # Pruning
//!
//! When a worker is dropped from a node and the node has no remaining
//! workers AND no children, we detach it from its parent and remove it from
//! the reverse index. Pruning cascades upward iteratively (chains can be
//! deep — the recursive form would risk stack-overflow for pathological
//! inputs). Pruning is shard-local: a chain never crosses a shard boundary.
//!
//! # Storage tiers
//!
//! An engine running a hierarchical cache holds a block on device and, once
//! it has been backed up, on host pinned memory (or a storage backend) as
//! well. It publishes every tier transition as its own event, tagged with a
//! `medium`: a host-tier `BlockStored` when the backup lands, a device-tier
//! `BlockRemoved` when the device copy is evicted, a host-tier `BlockRemoved`
//! when the host copy goes. Under write-back the device eviction of a
//! backed-up block is therefore preceded by a host store for the same block.
//!
//! WHY the tree keeps the tiers apart instead of treating any `BlockRemoved`
//! as "the worker lost the block": a device eviction that leaves a host copy
//! behind does not end the worker's ability to serve the prefix — it loads
//! the copy back at memory speed instead of recomputing it. Dropping the
//! worker on that event makes every prefix unroutable after one device
//! turnover, even though the fleet still holds it for the whole host
//! retention horizon, and the repeat request lands on a random worker that
//! then prefills it cold. Ownership here is "any tier"; the device subset is
//! reported alongside so the policy can prefer it.
//!
//! Untagged events keep their pre-tiering meaning: an untagged store is a
//! device store, an untagged remove clears every tier. See [`Tiers`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

/// Number of independent tree shards. A power of two so shard selection is a
/// mask, not a modulo. Sized generously relative to typical fleet sizes so
/// distinct chains rarely collide on a shard while keeping per-shard
/// overhead (one `RwLock` + one arena) negligible.
const N_SHARDS: usize = 32;

// `shard_of` shifts by `64 - log2(N_SHARDS)` and indexes `shards[..N_SHARDS]`,
// both of which are only correct for a power of two ≥ 2. Make a bad value a
// compile error rather than a runtime panic / out-of-bounds.
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
/// set. Travels between replicas as raw bits ([`Self::bits`] /
/// [`Self::from_bits`]) inside [`SnapshotNode::tiers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Tiers(u8);

impl Tiers {
    /// Device HBM: `medium = "GPU"`, or an untagged event from a publisher that
    /// predates tiering.
    pub const DEVICE: Tiers = Tiers(1);
    /// Host pinned memory: `medium = "CPU_PINNED"`.
    pub const HOST: Tiers = Tiers(1 << 1);
    /// A storage backend the worker can load back from: `medium = "DISK"` or
    /// `"EXTERNAL"`.
    pub const STORAGE: Tiers = Tiers(1 << 2);
    /// A tier this build cannot rank: a `medium` string [`Self::known`] does
    /// not recognise, or a snapshot bit outside the tiers it knows. The
    /// worker is an owner (it can serve the prefix somehow) but is never
    /// preferred as a device owner for it.
    pub const OTHER: Tiers = Tiers(1 << 3);
    /// Every tier — what an untagged `BlockRemoved` clears.
    pub const ALL: Tiers = Tiers(Self::DEVICE.0 | Self::HOST.0 | Self::STORAGE.0 | Self::OTHER.0);
    /// Every tier with its metric label, in bit order. [`TierCounts`] is
    /// indexed by position in this table.
    pub const SLOTS: [(Tiers, &'static str); TIER_SLOT_COUNT] = [
        (Self::DEVICE, "device"),
        (Self::HOST, "host"),
        (Self::STORAGE, "storage"),
        (Self::OTHER, "other"),
    ];

    /// The tier a `BlockStored` tagged `medium` lands on. Untagged reads as
    /// device, which is what the event meant before tiers existed. An unknown
    /// tag reads as [`Self::OTHER`]: the store still makes the worker an
    /// owner, but a tier this tree cannot rank must not be the one the policy
    /// prefers — an engine adding a medium would otherwise turn every store
    /// on it into a device hit.
    pub fn for_store(medium: Option<&str>) -> Tiers {
        match medium {
            None => Self::DEVICE,
            Some(m) => Self::known(m).unwrap_or(Self::OTHER),
        }
    }

    /// The tiers a `BlockRemoved` tagged `medium` clears. Untagged clears
    /// everything — the pre-tiering meaning. An unknown tag ALSO clears
    /// everything, deliberately asymmetric with [`Self::for_store`]: clearing
    /// only a bit this tree never set would leave the worker owning the block
    /// for good, and a permanently stale owner is the one failure a routing
    /// index must never manufacture. Over-removal costs at most one cold
    /// prefill.
    pub fn for_remove(medium: Option<&str>) -> Tiers {
        medium.and_then(Self::known).unwrap_or(Self::ALL)
    }

    /// The `StorageMedium` strings SGLang puts on the wire
    /// (`python/sglang/srt/disaggregation/kv_events.py`) and the tier each
    /// lands on. The single source for both the tree's ranking and the event
    /// tally's medium labels, so a medium the tree ranks can never be one the
    /// tally reports as unknown.
    pub const WIRE_MEDIA: [(&'static str, Tiers); 4] = [
        ("GPU", Self::DEVICE),
        ("CPU_PINNED", Self::HOST),
        ("DISK", Self::STORAGE),
        ("EXTERNAL", Self::STORAGE),
    ];

    fn known(medium: &str) -> Option<Tiers> {
        Self::WIRE_MEDIA
            .iter()
            .find(|(name, _)| *name == medium)
            .map(|(_, tier)| *tier)
    }

    /// Position in [`Self::SLOTS`] of the best tier held, `None` if none.
    /// `SLOTS` order is preference order: a device copy is served in place, a
    /// host copy by load-back, storage slower still, `other` unranked.
    pub fn best_slot(self) -> Option<usize> {
        Self::SLOTS
            .iter()
            .position(|(tier, _)| self.contains(*tier))
    }

    /// Metric label of the best tier held (see [`Self::best_slot`]).
    pub fn best_label(self) -> Option<&'static str> {
        self.best_slot().map(|slot| Self::SLOTS[slot].1)
    }

    /// Bits as they travel in a snapshot. The inverse, [`Self::from_bits`],
    /// reads an all-zero entry as device (a producer that wrote no tier meant
    /// device) and folds bits outside the known tiers into [`Self::OTHER`], so
    /// a carrier is never restored as an owner of no tier, nor as a device
    /// owner on the strength of a tier this build does not know.
    pub fn bits(self) -> u8 {
        self.0
    }

    pub fn from_bits(bits: u8) -> Tiers {
        if bits == 0 {
            return Self::DEVICE;
        }
        let known = bits & Self::ALL.0;
        let unknown = bits & !Self::ALL.0;
        Tiers(known | if unknown != 0 { Self::OTHER.0 } else { 0 })
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every bit of `other` is set here.
    pub fn contains(self, other: Tiers) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Tiers) {
        self.0 |= other.0;
    }

    pub fn remove(&mut self, other: Tiers) {
        self.0 &= !other.0;
    }

    /// The bits set in both.
    pub fn intersection(self, other: Tiers) -> Tiers {
        Tiers(self.0 & other.0)
    }

    /// The bits set here and not in `other`.
    pub fn difference(self, other: Tiers) -> Tiers {
        Tiers(self.0 & !other.0)
    }
}

/// Number of entries in [`Tiers::SLOTS`].
pub const TIER_SLOT_COUNT: usize = 4;

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
/// leaves an entry with no bits, which `TreeState::remove` relies on.
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
    /// Workers holding the deepest matched node on ANY tier. Empty when
    /// `matched_blocks == 0`.
    pub workers: HashSet<KvWorkerId>,
    /// The tiers each of `workers` holds the deepest matched node on. A worker
    /// without the device bit serves the prefix by loading it back from a
    /// lower tier — cheaper than a cold prefill, dearer than serving in place
    /// — which is the ordering the policy prefers on ([`Tiers::best_slot`]).
    pub tiers: HashMap<KvWorkerId, Tiers>,
}

impl MatchResult {
    /// The subset of `workers` holding the deepest matched node on device.
    pub fn device_workers(&self) -> HashSet<KvWorkerId> {
        self.tiers
            .iter()
            .filter(|(_, tiers)| tiers.contains(Tiers::DEVICE))
            .map(|(w, _)| w.clone())
            .collect()
    }
}

/// One node of a tree snapshot, as produced by
/// [`HashTree::export_snapshot`] and consumed by
/// [`HashTree::restore_snapshot`].
///
/// Records are parent-linked by their **index in the snapshot's node list**,
/// not by block hash. That is deliberate: the same block hash legitimately
/// occupies several tree positions (see the reverse-index module docs), so a
/// hash-keyed replay would land in `resolve_parent`'s ambiguous branch and
/// could graft a chain under the wrong node. Indices make placement exact.
///
/// `workers` are indices into the snapshot's worker table, kept sorted so a
/// record's carrier list does not depend on hash-set iteration order. The
/// order of the records themselves follows the per-shard children maps and is
/// unspecified — only the backward-reference property is guaranteed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotNode {
    /// Index of this node's parent record, or `None` when the node hangs
    /// directly off a shard root. Always a backward reference: strictly less
    /// than this record's own index.
    pub parent: Option<u32>,
    pub block_hash: i64,
    pub workers: Vec<u32>,
    /// Tier bits of each carrier, parallel to `workers` (`tiers[k]` describes
    /// `workers[k]`), as [`Tiers::bits`]. Empty means every carrier holds the
    /// node on device.
    ///
    /// That empty-means-device reading is what keeps [`super::bootstrap::SNAPSHOT_FORMAT`]
    /// unchanged: a producer that predates tiers omits the field and its
    /// carriers restore exactly as they did before, and a consumer that
    /// predates tiers ignores the field and reads the old meaning. The JSON
    /// wire tolerates both directions; `serde(default)` fills the gap.
    #[serde(default)]
    pub tiers: Vec<u8>,
}

impl SnapshotNode {
    /// Keep the carriers `map` accepts, renumbering each to what it returns,
    /// and keep their tier entries in lockstep. The only correct way to
    /// filter a record's carriers: `workers` and `tiers` are parallel by
    /// index, so filtering one alone silently re-pairs every later carrier
    /// with another carrier's tiers.
    pub fn retain_carriers(&mut self, mut map: impl FnMut(u32) -> Option<u32>) {
        let tiered = !self.tiers.is_empty();
        let mut workers = Vec::with_capacity(self.workers.len());
        let mut tiers = Vec::with_capacity(if tiered { self.workers.len() } else { 0 });
        for (k, &w) in self.workers.iter().enumerate() {
            let Some(kept) = map(w) else {
                continue;
            };
            workers.push(kept);
            if tiered {
                tiers.push(self.tiers.get(k).copied().unwrap_or(Tiers::DEVICE.bits()));
            }
        }
        self.workers = workers;
        self.tiers = tiers;
    }
}

/// Why a [`HashTree::restore_snapshot`] was rejected.
///
/// Snapshots arrive over the network from a peer replica, so their shape is
/// validated rather than assumed.
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
            Self::TreeInvariant { index } => {
                write!(
                    f,
                    "snapshot node {index} could not be grafted: parent missing"
                )
            }
        }
    }
}

impl std::error::Error for RestoreError {}

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
/// Node ids are unique within this shard only; two shards may both mint id 1.
#[derive(Debug)]
struct TreeState {
    nodes: FxHashMap<NodeId, Node>,
    by_hash: FxHashMap<i64, FxHashSet<NodeId>>,
    next_id: NodeId,
    /// Nodes in this shard each carrier holds, per tier. Booked at every
    /// site where a carrier's tier bits on a node change (`account_add` /
    /// `account_remove`), so a scrape reads it without walking the tree. A
    /// carrier holding nothing in the shard has no row.
    occupancy: HashMap<KvWorkerId, TierCounts>,
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
            occupancy: HashMap::new(),
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Book one node's worth of `added` tier bits newly held by `worker`.
    fn account_add(&mut self, worker: &KvWorkerId, added: Tiers) {
        let mut delta = TierCounts::default();
        tally_tiers(&mut delta, added);
        self.account_add_counts(worker, delta);
    }

    /// Book `delta` nodes-per-tier newly held by `worker` — one row lookup for
    /// a whole chain, which is how `insert` uses it.
    fn account_add_counts(&mut self, worker: &KvWorkerId, delta: TierCounts) {
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

    /// Book one node's worth of `removed` tier bits `worker` no longer holds,
    /// dropping the carrier's row once it holds nothing in this shard.
    fn account_remove(&mut self, worker: &KvWorkerId, removed: Tiers) {
        if removed.is_empty() {
            return;
        }
        let Some(counts) = self.occupancy.get_mut(worker) else {
            error!(
                worker = %worker.url,
                dp_rank = worker.dp_rank,
                "tree invariant violation: releasing tiers for a carrier with no occupancy row",
            );
            return;
        };
        for (slot, (tier, _)) in Tiers::SLOTS.iter().enumerate() {
            if removed.contains(*tier) {
                counts[slot] = counts[slot].saturating_sub(1);
            }
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

    /// Pick the parent node id for an incoming `BlockStored` event, given
    /// that this shard is already known to own (or be the fallback for)
    /// the chain.
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
            let added = add_tiers(&mut child.workers, worker, tiers);
            child.last_used.store(now, Ordering::Relaxed);
            tally_tiers(&mut delta, added);
            current = child_id;
            prev_hash = Some(h);
        }
        self.account_add_counts(worker, delta);
    }

    /// Clear `tiers` from `worker`'s hold on every node in THIS shard carrying
    /// any hash in `block_hashes`. The worker leaves a node once no tier bit
    /// remains, and a node that becomes empty + childless is pruned.
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
            let (still_present, released) = match self.nodes.get_mut(&id) {
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
            if still_present {
                self.prune_cascade(id);
            }
        }
    }

    /// Drop `worker` from every node in THIS shard, pruning emptied nodes.
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
            let removed = self.nodes.get_mut(&id).and_then(|node| {
                node.workers
                    .remove(worker)
                    .map(|held| (held, node.workers.is_empty() && node.children.is_empty()))
            });
            if let Some((held, prunable)) = removed {
                self.account_remove(worker, held);
                if prunable {
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

    /// Read-only match path within this shard. Takes `&self` (not
    /// `&mut self`) so the public [`HashTree::match_prefix`] holds only a
    /// read lock on the shard — matching is the routing hot path and
    /// write-locking would serialise routing decisions. `last_used` is an
    /// [`AtomicU64`] specifically so the touch-on-descend can happen through
    /// a shared reference.
    ///
    /// Note the asymmetry with [`TreeState::resolve_parent`] (used by
    /// `insert`): that function disambiguates a multi-candidate
    /// `parent_hash` by preferring a worker-owned node. This function has
    /// no worker context to do the same, so multiple candidates fall back
    /// to root. The asymmetry is intentional; the public doc on
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
            workers: tiers.keys().cloned().collect(),
            tiers,
        }
    }

    /// Read-only per-URL match path within this shard. See
    /// [`HashTree::match_prefix_for_url`] for the semantics.
    ///
    /// Deliberately does NOT touch `last_used`. This runs only to fill a
    /// metric, and an observation should not move eviction state: whether the
    /// router happens to be metering must not decide which node
    /// [`Self::lru_leaf`] drops next. The touch would also be redundant —
    /// [`Self::match_prefix`] has just walked this same path and stamped it.
    /// (Redundant, not harmful: recency is a timestamp, so a second stamp
    /// microseconds later is the same value. The reason to skip it is the
    /// separation, not an arithmetic effect.)
    fn match_prefix_for_url(
        &self,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
        url: &str,
    ) -> usize {
        let start = match parent_hash {
            None => ROOT_ID,
            Some(p) => match self.by_hash.get(&p) {
                Some(set) if set.len() == 1 => *set.iter().next().unwrap(),
                _ => ROOT_ID,
            },
        };

        let mut current = start;
        let mut depth = 0usize;
        let mut owned_depth = 0usize;
        for &h in block_hashes {
            let Some(child_id) = self
                .nodes
                .get(&current)
                .and_then(|n| n.children.get(&h).copied())
            else {
                break;
            };
            current = child_id;
            depth += 1;
            if self
                .nodes
                .get(&child_id)
                .is_some_and(|n| n.workers.keys().any(|w| w.url == url))
            {
                owned_depth = depth;
            }
        }
        owned_depth
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
    /// is preserved; the per-millisecond victim among equal timestamps is an
    /// arbitrary-but-deterministic choice).
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

/// Public hash-keyed radix tree. Cheap to clone an [`Arc`] of; the
/// underlying state is `Send + Sync`.
///
/// WHY sharded: the routing hot path reads while the event pump writes, and
/// a single lock makes every write block every read. The tree is split into
/// [`N_SHARDS`] independent [`TreeState`]s (each its own `RwLock`) keyed by
/// chain root, so unrelated chains no longer contend. See the module docs.
#[derive(Debug)]
pub struct HashTree {
    shards: Vec<RwLock<TreeState>>,
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
        Self { shards }
    }

    /// Resolve an `insert`'s `parent_hash` GLOBALLY, replicating the
    /// single-tree `resolve_parent` decision across all shards, and return
    /// `(shard, effective_parent_hash)` for the local insert.
    ///
    /// WHY return an effective parent: the single-tree `resolve_parent`
    /// decides "attach under node X" or "attach at root" from the COMPLETE
    /// set of nodes carrying `parent_hash`. A per-shard `resolve_parent` only
    /// sees its own slice of that set, so when the global decision is
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
    /// 4. Multiple nodes carry it → a `worker`-owned carrier's shard keeping
    ///    `parent_hash`; if none is owned → root shard, parent `None`
    ///    (single-tree "attach to root" fallback).
    ///
    /// Only invoked on `insert`; the `match_prefix` hot path never scatters.
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
        // Gather, across shards, how many nodes carry `p` and which shard (if
        // any) holds a node `worker` already owns.
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

    /// Resolve a `match_prefix`'s start point GLOBALLY and return
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
        // The single-tree `match_prefix` only honors a UNIQUE carrier of `p`;
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
    /// A removed hash can be a chain root in one shard and an interior block
    /// of a chain rooted elsewhere in another, so this fans out across all
    /// shards. Each shard's local `by_hash` short-circuits shards that don't
    /// carry any of the hashes.
    pub fn remove_tiered(&self, worker: &KvWorkerId, block_hashes: &[i64], tiers: Tiers) {
        if block_hashes.is_empty() || tiers.is_empty() {
            return;
        }
        for shard in &self.shards {
            shard.write().remove(worker, block_hashes, tiers);
        }
    }

    /// Apply an `AllBlocksCleared` event for `worker`.
    ///
    /// A worker can hold chains in many shards, so this fans out across all
    /// shards.
    pub fn clear_worker(&self, worker: &KvWorkerId) {
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
    /// this method only needs a read lock on a single shard and many threads
    /// can match concurrently across shards.
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

    /// How many leading blocks of `block_hashes` the worker at `url` holds
    /// *itself*, as opposed to the fleet-wide depth [`Self::match_prefix`]
    /// reports.
    ///
    /// `match_prefix` answers "how deep does this chain go in the tree at
    /// all", which is the best prefix any worker could serve. This answers it
    /// for one destination. The two diverge exactly when the policy does not
    /// route to an owner of the deepest node — the queue gate diverting off a
    /// prefix owner, or a below-threshold match falling back to min-load — so
    /// the difference is the locality the routing decision gave up, separated
    /// from the locality that was never there.
    ///
    /// Keyed by URL rather than [`KvWorkerId`] because a URL carrying several
    /// `dp_rank`s is a single routing destination: any rank holding the prefix
    /// means a request sent to that URL can hit it. This matches how
    /// `match_prefix`'s callers collapse [`MatchResult::workers`] to URLs.
    ///
    /// Returns 0 for an empty `block_hashes`, a chain the worker does not
    /// hold, AND a URL the tree has never seen. That last one is not the same
    /// thing but is reported as though it were: a worker whose KV-event
    /// subscription never started serves traffic while owning nothing here, so
    /// a caller metering cache locality will book real zeros for it. Callers
    /// that need to tell "holds none of this" from "publishes nothing" must
    /// get that from registration state, not from this return value.
    ///
    /// Unlike [`Self::match_prefix`], this does NOT touch `last_used` — a
    /// metering read must not perturb eviction order.
    ///
    /// Ambiguous `parent_hash` falls back to matching from the root, exactly
    /// as in [`Self::match_prefix`].
    pub fn match_prefix_for_url(
        &self,
        parent_hash: Option<i64>,
        block_hashes: &[i64],
        url: &str,
    ) -> usize {
        if block_hashes.is_empty() {
            return 0;
        }
        let (idx, effective_parent) = self.route_match(parent_hash, block_hashes);
        self.shards[idx]
            .read()
            .match_prefix_for_url(effective_parent, block_hashes, url)
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
    /// summed across shards. A given hash value can appear in more than one
    /// shard (root of one chain, interior of another), and each occurrence
    /// is counted once per shard — consistent with the per-shard reverse
    /// indexes being independent.
    ///
    /// Exposed for invariant tests: when `node_count() == 0` this must also
    /// be 0. A nonzero value here with zero nodes means a `prune` path forgot
    /// to clean up `by_hash` and the index has leaked.
    pub fn reverse_index_size(&self) -> usize {
        self.shards.iter().map(|s| s.read().by_hash.len()).sum()
    }

    /// Nodes each carrier holds on each tier, summed across shards and sorted
    /// by carrier. Rendered as `sgl_router_kv_tree_blocks`; against the
    /// engine's own per-tier occupancy for the same pod it is the tree's
    /// coverage of that tier — the number that says whether a tier the engine
    /// holds is visible to routing at all (module docs, "Storage tiers").
    ///
    /// Read off the per-shard accounting rather than walked, so a scrape
    /// costs one read lock per shard plus a merge over carriers. Like
    /// [`Self::node_count`] it is a point-in-time aggregate across shards, not
    /// one consistent instant. A carrier holding nothing has no row.
    pub fn tier_occupancy(&self) -> Vec<(KvWorkerId, TierCounts)> {
        let mut total: HashMap<KvWorkerId, TierCounts> = HashMap::new();
        for shard in &self.shards {
            let st = shard.read();
            for (worker, counts) in &st.occupancy {
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

    /// Whether ANY node carries `hash`, regardless of its position in a chain.
    ///
    /// Answers the question [`Self::match_prefix`] cannot. That method returns
    /// `matched_blocks == 0` in two situations with OPPOSITE diagnoses: the
    /// fleet never stored the block at all, or the block is present but not
    /// reachable as a chain from the root. The first is an engine-side publish
    /// or eviction gap; the second is a parent-linkage problem in this tree.
    /// Nothing else separates them, and a fleet reading ~100% zero-overlap
    /// looks identical under both.
    ///
    /// Scans every shard rather than only `shard_of(hash)`, for the reason
    /// [`Self::reverse_index_size`] documents: a hash reached as an interior
    /// node of another chain is not confined to its own root shard.
    ///
    /// Does NOT touch `last_used` — a diagnostic read must not perturb
    /// eviction order, matching [`Self::match_prefix_for_url`].
    pub fn contains_hash(&self, hash: i64) -> bool {
        // The root shard is where a chain-leading block lives, so try it
        // alone before sweeping the rest: a present hash usually costs one
        // lock, and only the absent case pays the full scan.
        let root = shard_of(hash);
        if self.shards[root].read().by_hash.contains_key(&hash) {
            return true;
        }
        self.shards
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != root)
            .any(|(_, shard)| shard.read().by_hash.contains_key(&hash))
    }

    /// Evict least-recently-used nodes until `node_count() <= max_size`
    /// across the whole tree.
    ///
    /// Strategy:
    /// 1. Drop already-empty leaves (no workers, no children) in every shard.
    /// 2. If still over cap, repeatedly evict the globally-oldest leaf —
    ///    found by comparing each shard's LRU leaf — force-clearing its
    ///    workers and cascade-pruning, until the global count is at the cap.
    ///
    /// Returns the exact total number of nodes pruned, including any
    /// ancestors removed by cascade-pruning. Suitable for wiring into a
    /// metric counter.
    ///
    /// WHY a global pass rather than a per-shard cap: a per-shard quota
    /// would evict hot entries in a busy shard while idle shards sit under
    /// quota, changing which nodes survive. The global LRU keeps eviction
    /// order equivalent to the single-tree behavior. Like the other mutators,
    /// this MUST run on the single writer thread (the KV-event pump): the cap
    /// check and LRU selection take per-shard locks one at a time, so a
    /// concurrent inserter would make the cap a best-effort target rather than
    /// a hard postcondition. Not on the hot path (it runs periodically), so
    /// briefly read/write-locking each shard is acceptable.
    pub fn evict_lru(&self, max_size: usize) -> usize {
        // Fast-path: already under cap.
        if self.node_count() <= max_size {
            return 0;
        }
        let mut pruned = 0usize;

        // Phase 1: free empty leaves everywhere.
        for shard in &self.shards {
            pruned += shard.write().drop_empty_leaves();
            if self.node_count() <= max_size {
                return pruned;
            }
        }

        // Phase 2: evict the globally-oldest leaf one at a time. Bound the
        // loop by the total node count so a degenerate tree can't spin.
        let mut iters = 0usize;
        let max_iters = self.node_count().saturating_add(1);
        while self.node_count() > max_size && iters < max_iters {
            iters += 1;
            // Pick the shard whose LRU leaf is globally oldest. Tie-break
            // by (timestamp, shard-local node id, shard index) so the choice
            // is deterministic.
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
                // The chosen leaf raced away (e.g. concurrent prune). Re-scan
                // on the next iteration rather than spin on a stale pick.
                continue;
            }
            pruned += dropped;
        }
        pruned
    }

    /// Export every node as a flat, parent-linked list that
    /// [`HashTree::restore_snapshot`] can rebuild an identical tree from.
    ///
    /// Returns `(worker_table, nodes)`. `nodes` is in DFS pre-order within
    /// each shard, so every record's `parent` index is strictly less than its
    /// own — the precondition `restore_snapshot` validates.
    ///
    /// WHY not full root-to-node hash paths (the obvious alternative): that is
    /// quadratic in depth, and production chains run thousands of blocks deep.
    /// Parent-by-index is linear and unambiguous.
    ///
    /// Takes each shard's read lock in turn, never all at once, so exporting a
    /// large tree cannot stall the `match_prefix` hot path fleet-wide. Like
    /// [`HashTree::node_count`], the result is a point-in-time aggregate, not
    /// one consistent instant across shards; under the single-writer pump each
    /// shard is individually consistent, which is what a bootstrap needs.
    pub fn export_snapshot(&self) -> (Vec<KvWorkerId>, Vec<SnapshotNode>) {
        let mut worker_table: Vec<KvWorkerId> = Vec::new();
        let mut worker_index: FxHashMap<KvWorkerId, u32> = FxHashMap::default();
        let mut nodes: Vec<SnapshotNode> = Vec::new();

        for shard in &self.shards {
            let st = shard.read();
            // (node id, parent's index in `nodes`); `None` for shard-root children.
            let mut stack: Vec<(NodeId, Option<u32>)> = st
                .nodes
                .get(&ROOT_ID)
                .map(|root| root.children.values().map(|&id| (id, None)).collect())
                .unwrap_or_default();
            while let Some((id, parent_idx)) = stack.pop() {
                let Some(node) = st.nodes.get(&id) else {
                    continue;
                };
                let mut carriers: Vec<(u32, u8)> = node
                    .workers
                    .iter()
                    .map(|(w, tiers)| {
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
                    })
                    .collect();
                // Sorted by worker index so a record's carrier list does not
                // depend on hash-map iteration order; tiers ride along.
                carriers.sort_unstable();
                let (workers, tiers): (Vec<u32>, Vec<u8>) = carriers.into_iter().unzip();
                // Pushed before its children, so children get larger indices
                // and the backward-reference invariant holds by construction.
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
        (worker_table, nodes)
    }

    /// Rebuild tree state from a snapshot, typically one fetched from a warm
    /// peer replica at boot.
    ///
    /// Grafts each record under its recorded parent: creates the node when
    /// absent, unions the worker set when present, so restoring onto a
    /// non-empty tree is well defined. A record with `parent == None` is
    /// rooted in `shard_of(block_hash)` — exactly where [`HashTree::insert`]
    /// would have put it — so a restored tree routes identically to the tree
    /// it came from.
    ///
    /// Returns the number of records applied.
    ///
    /// # Untrusted input
    ///
    /// The node list arrives over the network, so its shape is validated up
    /// front rather than assumed: `parent` must be a backward reference, and
    /// every worker index must be in range. Validation happens before any
    /// mutation, so a rejected snapshot leaves the tree untouched.
    ///
    /// `worker_table` must hold ids resolved against the local worker
    /// registry, NOT ids deserialized straight off the wire — see the
    /// provenance note on [`KvWorkerId`]. This method trusts the ids it is
    /// handed, which is why it is module-internal: outside `kv_events` the only
    /// way in is `bootstrap::VettedSnapshot::graft_into`, and a `VettedSnapshot`
    /// can only be built by resolving wire identities against the live set.
    ///
    /// MUST run on the single writer (the KV-event pump), like every other
    /// mutator; see the single-writer property in the module docs.
    pub(super) fn restore_snapshot(
        &self,
        worker_table: &[KvWorkerId],
        nodes: &[SnapshotNode],
    ) -> Result<usize, RestoreError> {
        // Validate before mutating. The backward-reference check is what makes
        // the placement lookup below infallible; the bounds check keeps a
        // malformed peer from silently dropping cache carriers.
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

        // Record index -> where it landed. Filled in order, so a parent is
        // always placed before any child consults it.
        let mut placed: Vec<(usize, NodeId)> = Vec::with_capacity(nodes.len());
        let now = now_millis();
        for (i, rec) in nodes.iter().enumerate() {
            let (shard_idx, parent_id) = match rec.parent {
                None => (shard_of(rec.block_hash), ROOT_ID),
                Some(p) => placed[p as usize],
            };
            let parent_block_hash = rec.parent.map(|p| nodes[p as usize].block_hash);

            let mut st = self.shards[shard_idx].write();
            let existing = st
                .nodes
                .get(&parent_id)
                .and_then(|n| n.children.get(&rec.block_hash).copied());
            let id = match existing {
                Some(id) => id,
                None => st
                    .create_child(parent_id, rec.block_hash, parent_block_hash)
                    .ok_or(RestoreError::TreeInvariant { index: i })?,
            };
            // (worker-table index, bits newly held) per carrier, booked into
            // the shard's occupancy once the node borrow ends.
            let mut added: Vec<(usize, Tiers)> = Vec::with_capacity(rec.workers.len());
            if let Some(node) = st.nodes.get_mut(&id) {
                for (k, &w) in rec.workers.iter().enumerate() {
                    // Absent tiers (pre-tiering producer) read as device;
                    // `from_bits` also folds a zero entry to device so no
                    // carrier is restored owning no tier.
                    let tiers = rec
                        .tiers
                        .get(k)
                        .map(|&bits| Tiers::from_bits(bits))
                        .unwrap_or(Tiers::DEVICE);
                    let bits = add_tiers(&mut node.workers, &worker_table[w as usize], tiers);
                    if !bits.is_empty() {
                        added.push((w as usize, bits));
                    }
                }
                node.last_used.store(now, Ordering::Relaxed);
            }
            for (w, bits) in added {
                st.account_add(&worker_table[w], bits);
            }
            placed.push((shard_idx, id));
        }
        Ok(nodes.len())
    }
}

// ---------------------------------------------------------------------------
// Whitebox test helpers
//
// WHY these exist: the in-module tests assert on internal structure
// (reverse index membership, `parent_block_hash` chaining). State is now
// split across shards, so these helpers aggregate the per-shard layout while
// the tests' behavioral assertions stay identical in meaning.
// ---------------------------------------------------------------------------

#[cfg(test)]
impl HashTree {
    /// Recompute [`Self::tier_occupancy`] by walking every node — the oracle
    /// the incrementally maintained counters are checked against.
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
                    for (slot, (tier, _)) in Tiers::SLOTS.iter().enumerate() {
                        if held.contains(*tier) {
                            acc[slot] += 1;
                        }
                    }
                }
            }
        }
        let mut rows: Vec<(KvWorkerId, TierCounts)> = total.into_iter().collect();
        rows.sort_by(|a, b| (&a.0.url, a.0.dp_rank).cmp(&(&b.0.url, b.0.dp_rank)));
        rows
    }

    /// Total number of distinct nodes carrying `hash`, summed across shards.
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

    #[test]
    fn empty_match_returns_zero_no_workers() {
        let tree = HashTree::new();
        let m = tree.match_prefix(None, &[]);
        assert_eq!(m.matched_blocks, 0);
        assert!(m.workers.is_empty());

        let m2 = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m2.matched_blocks, 0);
        assert!(m2.workers.is_empty());
    }

    /// The whole point of `contains_hash`: `match_prefix` returns 0 both for a
    /// block the fleet never stored AND for one it stores at a position the
    /// root-anchored walk cannot reach. Those have opposite diagnoses, so the
    /// accessor must tell them apart — and must stop saying "in_tree" once the
    /// block is actually removed, or the counter would report stale presence.
    #[test]
    fn contains_hash_separates_absent_from_unreachable() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);

        // Reachable from the root: both signals agree it is there.
        assert_eq!(tree.match_prefix(None, &[1, 2]).matched_blocks, 2);
        assert!(tree.contains_hash(1));

        // Carried by an INTERIOR node. match_prefix reports 0 because the chain
        // does not start at the root, yet the hash is present. This is the case
        // the counter exists to separate; without it this is indistinguishable
        // from a block the engines never published.
        assert_eq!(tree.match_prefix(None, &[2, 3]).matched_blocks, 0);
        assert!(tree.contains_hash(2), "interior hash must still be visible");
        assert!(tree.contains_hash(3), "leaf hash must still be visible");

        // Never inserted: genuinely absent.
        assert_eq!(tree.match_prefix(None, &[99]).matched_blocks, 0);
        assert!(!tree.contains_hash(99));

        // After removal the reverse index must not keep reporting presence, or
        // an engine-side eviction would be misread as a router linkage fault.
        tree.remove(&a, &[1, 2, 3]);
        assert!(
            !tree.contains_hash(1),
            "removed chain must not read in_tree"
        );
        assert!(
            !tree.contains_hash(2),
            "removed chain must not read in_tree"
        );
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
        assert_eq!(m.workers, workers(&[&a]));
        assert_eq!(m.device_workers(), workers(&[&a]), "held on device too");

        tree.remove_tiered(&a, &[3], Tiers::for_remove(Some("GPU")));
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3, "host copy keeps the chain matchable");
        assert_eq!(
            m.workers,
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
        assert!(!tree.contains_hash(3));
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

    /// An unknown medium is asymmetric on purpose: a store on it makes the
    /// worker an owner on the unranked `OTHER` tier — never a device owner —
    /// while a removal tagged with it clears everything. The alternative for
    /// removal — clearing only a bit that was never set — leaves a stale
    /// owner forever.
    #[test]
    fn unknown_medium_stores_as_owner_and_removes_everything() {
        assert_eq!(Tiers::for_store(Some("NVLINK_PEER")), Tiers::OTHER);
        assert_eq!(Tiers::for_remove(Some("NVLINK_PEER")), Tiers::ALL);
        assert_eq!(Tiers::for_store(None), Tiers::DEVICE);
        assert_eq!(Tiers::for_remove(None), Tiers::ALL);
        assert_eq!(Tiers::for_store(Some("CPU_PINNED")), Tiers::HOST);
        assert_eq!(Tiers::for_remove(Some("DISK")), Tiers::STORAGE);
        assert_eq!(Tiers::for_remove(Some("EXTERNAL")), Tiers::STORAGE);

        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        tree.insert_tiered(&a, None, &[1], Tiers::DEVICE);
        tree.insert_tiered(&a, None, &[1], Tiers::HOST);
        tree.insert_tiered(&b, None, &[1], Tiers::for_store(Some("NVLINK_PEER")));
        let m = tree.match_prefix(None, &[1]);
        assert_eq!(
            m.workers,
            workers(&[&a, &b]),
            "unknown-tier holder is an owner"
        );
        assert_eq!(m.device_workers(), workers(&[&a]), "but not a device owner");
        tree.remove_tiered(&a, &[1], Tiers::for_remove(Some("NVLINK_PEER")));
        tree.remove_tiered(&b, &[1], Tiers::for_remove(Some("NVLINK_PEER")));
        assert_eq!(tree.match_prefix(None, &[1]).matched_blocks, 0);
    }

    /// Snapshot bits this build does not know fold into `OTHER`: the carrier
    /// stays an owner but is not promoted to device. A zero entry still reads
    /// as device.
    #[test]
    fn from_bits_folds_unknown_bits_into_other_not_device() {
        assert_eq!(Tiers::from_bits(0), Tiers::DEVICE);
        assert_eq!(Tiers::from_bits(0b1000_0000), Tiers::OTHER);
        let mut host_plus_unknown = Tiers::HOST;
        host_plus_unknown.insert(Tiers::OTHER);
        assert_eq!(
            Tiers::from_bits(Tiers::HOST.bits() | 0b1000_0000),
            host_plus_unknown
        );
        assert_eq!(Tiers::from_bits(Tiers::DEVICE.bits()), Tiers::DEVICE);
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
        assert_eq!(m.workers, workers(&[&a, &b]));
        assert_eq!(m.device_workers(), workers(&[&a]));

        tree.remove_tiered(&a, &[2], Tiers::DEVICE);
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.workers, workers(&[&b]), "a dropped, b untouched");
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
    /// `clear_worker`, LRU eviction, and a snapshot restore onto live state.
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
        tree.insert_tiered(&c, None, &[7], Tiers::for_store(Some("NVLINK_PEER")));
        for r in 0..40i64 {
            tree.insert(&c, None, &[r * 4096 + 11, r * 4096 + 12]);
        }
        assert_eq!(tree.tier_occupancy(), tree.debug_recount_occupancy());

        // Spot-check the shape: a holds 4 device nodes and 2 host nodes.
        let rows = tree.tier_occupancy();
        let (_, a_counts) = rows.iter().find(|(w, _)| *w == a).unwrap();
        assert_eq!(a_counts[0], 4, "device");
        assert_eq!(a_counts[1], 2, "host");
        assert_eq!(a_counts[2], 0, "storage");
        let (_, c_counts) = rows.iter().find(|(w, _)| *w == c).unwrap();
        assert_eq!(c_counts[3], 1, "unknown medium lands on other");

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

        // Restore a peer snapshot onto the live tree (unions carriers).
        let src = HashTree::new();
        src.insert_tiered(&b, None, &[1, 2], Tiers::HOST);
        src.insert_tiered(&a, None, &[1, 2, 3], Tiers::DEVICE);
        let (table, nodes) = src.export_snapshot();
        tree.restore_snapshot(&table, &nodes).unwrap();
        assert_eq!(tree.tier_occupancy(), tree.debug_recount_occupancy());
    }

    #[test]
    fn single_insert_and_match() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);

        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers, workers(&[&a]));

        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers, workers(&[&a]));

        // Diverges at depth 3 (input asks for 4, tree has 3).
        let m = tree.match_prefix(None, &[1, 2, 4]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers, workers(&[&a]));

        // No match at root.
        let m = tree.match_prefix(None, &[9, 9]);
        assert_eq!(m.matched_blocks, 0);
        assert!(m.workers.is_empty());
    }

    /// The per-URL depth is a *different question* from the tree depth: it
    /// must report what one worker holds, not what the deepest node holds.
    /// Getting these confused is what makes a router's own hit-rate metric
    /// read above the engine's.
    #[test]
    fn match_prefix_for_url_reports_that_worker_not_the_deepest_node() {
        let tree = HashTree::new();
        let deep = worker("http://deep", 0);
        let shallow = worker("http://shallow", 0);
        tree.insert(&deep, None, &[1, 2, 3, 4]);
        tree.insert(&shallow, None, &[1, 2]);

        assert_eq!(tree.match_prefix(None, &[1, 2, 3, 4]).matched_blocks, 4);
        assert_eq!(
            tree.match_prefix_for_url(None, &[1, 2, 3, 4], "http://deep"),
            4,
        );
        assert_eq!(
            tree.match_prefix_for_url(None, &[1, 2, 3, 4], "http://shallow"),
            2,
            "a worker holding only a prefix of the chain must report its own depth",
        );
        assert_eq!(
            tree.match_prefix_for_url(None, &[1, 2, 3, 4], "http://absent"),
            0,
            "an unknown URL holds nothing",
        );
        assert_eq!(tree.match_prefix_for_url(None, &[], "http://deep"), 0);
        assert_eq!(
            tree.match_prefix_for_url(None, &[9, 9], "http://deep"),
            0,
            "a chain that does not exist in the tree holds nothing",
        );
    }

    /// One URL with several `dp_rank`s is a single routing destination, so the
    /// deepest rank's holding is what a request sent to that URL can hit.
    #[test]
    fn match_prefix_for_url_takes_the_deepest_dp_rank() {
        let tree = HashTree::new();
        tree.insert(&worker("http://a", 0), None, &[1, 2]);
        tree.insert(&worker("http://a", 1), None, &[1, 2, 3, 4]);

        assert_eq!(
            tree.match_prefix_for_url(None, &[1, 2, 3, 4], "http://a"),
            4,
        );
    }

    /// A node kept alive by a descendant after its owner was removed is not
    /// owned by anyone at that depth. The per-URL walk must resume from the
    /// last depth the worker actually owns rather than stopping at the hole.
    #[test]
    fn match_prefix_for_url_ignores_depths_the_worker_no_longer_owns() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2, 3]);
        tree.remove(&a, &[2]);

        assert_eq!(
            tree.match_prefix_for_url(None, &[1, 2, 3], "http://a"),
            3,
            "ownership at depth 3 stands even with depth 2 vacated",
        );
        assert_eq!(
            tree.match_prefix_for_url(None, &[1, 2], "http://a"),
            1,
            "asked only for the first two blocks, the deepest owned is depth 1",
        );
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
        assert_eq!(m.workers, workers(&[&a, &b]));

        // Divergent leaf carries only the matching worker.
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers, workers(&[&a]));

        let m = tree.match_prefix(None, &[1, 2, 4]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers, workers(&[&b]));
    }

    #[test]
    fn continuation_insert_chains_via_parent_hash() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        tree.insert(&a, None, &[1, 2]);
        tree.insert(&a, Some(2), &[3]);

        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers, workers(&[&a]));
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
        assert!(m.workers.is_empty());

        // Match length 3 lands on node 3 (workers still has A).
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers, workers(&[&a]));

        // Reverse-index sanity for hash 2: still present (node holds it).
        assert!(tree.contains_hash(2));
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
        assert_eq!(m.workers, workers(&[&b]));

        // [1,2] now has only B (A was the only other holder of node 2;
        // wait — actually A held 1 and 2 too. But B also holds 1 and 2.)
        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers, workers(&[&b]));

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
        assert!(!tree.contains_hash(1));
        assert!(!tree.contains_hash(2));
        assert!(!tree.contains_hash(3));
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
        assert_eq!(m.workers, workers(&[&a]));

        let m = tree.match_prefix(None, &[2, 5]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers, workers(&[&a]));

        // Reverse index for hash 5 has 2 distinct nodes (the two chains have
        // different roots, so they may live in different shards — the count
        // sums across shards).
        assert_eq!(tree.debug_hash_node_count(5), 2);

        // BlockRemoved [5] should remove A from BOTH nodes-carrying-5.
        // Both nodes are leaves, so both prune.
        tree.remove(&a, &[5]);
        // Remaining nodes: 1 and 2 (still hold A).
        assert_eq!(tree.node_count(), 2);
        let m = tree.match_prefix(None, &[1, 5]);
        assert_eq!(m.matched_blocks, 1);
        assert_eq!(m.workers, workers(&[&a]));
        let m = tree.match_prefix(None, &[2, 5]);
        assert_eq!(m.matched_blocks, 1);
        assert_eq!(m.workers, workers(&[&a]));
    }

    /// Two chains whose ROOT hashes collide into the SAME shard must stay
    /// fully independent — distinct worker sets, independent match, independent
    /// remove. Every multi-root test above deliberately SPREADS roots across
    /// shards; this pins the colliding case the sharding rests on.
    #[test]
    fn colliding_roots_in_same_shard_stay_independent() {
        // Premise: roots 1 and 22 hash to the same shard. Guarded so the test
        // fails loudly (not silently no-ops) if N_SHARDS / SHARD_MIX change.
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
        assert_eq!(m.workers, workers(&[&a]));
        let m = tree.match_prefix(None, &[22, 901]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers, workers(&[&b]));

        // Removing A's chain leaves B's chain in the shared shard untouched.
        tree.remove(&a, &[1, 900]);
        assert_eq!(tree.match_prefix(None, &[1, 900]).matched_blocks, 0);
        let m = tree.match_prefix(None, &[22, 901]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers, workers(&[&b]));
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
        assert_eq!(m.workers, workers(&[&w0, &w1]));

        // Divergent leaves: each rank on its own.
        let m = tree.match_prefix(None, &[1, 2, 3]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers, workers(&[&w0]));

        let m = tree.match_prefix(None, &[1, 2, 4]);
        assert_eq!(m.matched_blocks, 3);
        assert_eq!(m.workers, workers(&[&w1]));
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
        assert_eq!(m.workers, workers(&[&a]));

        // The chain 2->5 should NOT have a 7-child (we routed to A's branch).
        let m = tree.match_prefix(None, &[2, 5, 7]);
        assert_eq!(m.matched_blocks, 2);
        assert_eq!(m.workers, workers(&[&b]));
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
        assert_eq!(m.workers, workers(&[&c]));
    }

    /// Regression: the unowned-ambiguous `parent_hash` fallback must attach
    /// the new chain at ROOT even when the new chain's first hash happens to
    /// route to a shard that locally carries `parent_hash` in exactly one
    /// node. The hashes here are chosen so that roots 1 and 2 land on
    /// different shards (both carrying hash 5), while the continuation's
    /// first hash 1009 routes to the SAME shard as root 1 — the case where a
    /// naive per-shard resolve would wrongly attach 1009 under that shard's
    /// node-5 instead of root.
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
        // Single-tree behavior: two carriers of 5, none C-owned → attach at
        // root → 1009 becomes a fresh root child.
        tree.insert(&c, Some(5), &[1009]);

        // 1009 must be a root child (matched=1), NOT hanging under 1->5.
        let m = tree.match_prefix(None, &[1009]);
        assert_eq!(
            m.matched_blocks, 1,
            "1009 must attach at root, reachable as a top-level child",
        );
        assert_eq!(m.workers, workers(&[&c]));

        // And 1->5 must NOT have grown a 1009 child.
        let m = tree.match_prefix(None, &[1, 5, 1009]);
        assert_eq!(
            m.matched_blocks, 2,
            "1009 must NOT be attached under the shard's node carrying 5",
        );
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
        assert_eq!(m.workers, workers(&[&a]));
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
        // Each leaf hangs directly off its shard's root, so cascade-pruning
        // never cascades past the leaf itself: count must equal exactly the
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
        assert_eq!(m.workers, workers(&[&a]));
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
        assert_eq!(m.workers, workers(&[&a]));

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
        assert_eq!(m.workers, workers(&[&b]));
    }

    /// Distinct chain roots land on different shards (with high probability
    /// over 64 roots and 32 shards) yet `match_prefix` / `node_count` /
    /// eviction stay correct — the routing invariant the sharding relies on.
    #[test]
    fn distinct_roots_spread_across_shards() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        // Insert 64 independent two-block chains rooted at distinct hashes.
        for r in 0..64i64 {
            tree.insert(&a, None, &[r * 1000, r * 1000 + 1]);
        }
        assert_eq!(tree.node_count(), 128);

        // Confirm the roots actually used more than one shard (else the test
        // wouldn't be exercising cross-shard routing).
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
            assert_eq!(m.workers, workers(&[&a]));
        }
    }

    // -----------------------------------------------------------------------
    // Snapshot export / restore
    // -----------------------------------------------------------------------

    /// Build a tree exercising the cases a real snapshot has to survive:
    /// multi-worker shared prefixes, divergent branches, chains spanning many
    /// shards, and the same block hash occupying more than one position.
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
        // Spread roots across shards.
        for r in 0..32i64 {
            tree.insert(&b, None, &[r * 4096 + 11, r * 4096 + 12]);
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
        for r in 0..32i64 {
            q.push(vec![r * 4096 + 11]);
            q.push(vec![r * 4096 + 11, r * 4096 + 12]);
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
                (got.matched_blocks, got.workers),
                (want.matched_blocks, want.workers),
                "match_prefix diverged for {q:?}",
            );
        }
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
        assert!(!src.match_prefix(None, &[10, 20]).workers.contains(&a));

        let (worker_table, nodes) = src.export_snapshot();
        let dst = HashTree::new();
        dst.restore_snapshot(&worker_table, &nodes).unwrap();

        assert_eq!(dst.node_count(), src.node_count());
        for q in [vec![10], vec![10, 20], vec![10, 20, 30]] {
            let want = src.match_prefix(None, &q);
            let got = dst.match_prefix(None, &q);
            assert_eq!(
                (got.matched_blocks, got.workers),
                (want.matched_blocks, want.workers),
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

    #[test]
    fn restore_rejects_forward_parent_reference() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let nodes = vec![
            SnapshotNode {
                parent: Some(1), // forward
                block_hash: 1,
                workers: vec![0],
                tiers: vec![],
            },
            SnapshotNode {
                parent: None,
                block_hash: 2,
                workers: vec![0],
                tiers: vec![],
            },
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
        let nodes = vec![SnapshotNode {
            parent: Some(0),
            block_hash: 1,
            workers: vec![0],
            tiers: vec![],
        }];
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
            SnapshotNode {
                parent: None,
                block_hash: 1,
                workers: vec![0],
                tiers: vec![],
            },
            SnapshotNode {
                parent: Some(0),
                block_hash: 2,
                workers: vec![7], // table has one entry
                tiers: vec![],
            },
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

    /// Tiers survive an export → restore round trip per carrier, so a replica
    /// bootstrapped from a warm peer ranks device and host owners the same way
    /// the peer did.
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
        assert_eq!(m.workers, workers(&[&a, &b]));
        assert_eq!(
            m.device_workers(),
            workers(&[&a]),
            "b was host-only on the peer"
        );
    }

    /// A snapshot from a producer that predates tiers carries no `tiers`; its
    /// carriers restore as device owners, which is what its `workers` list
    /// meant when it was written.
    #[test]
    fn legacy_snapshot_without_tiers_restores_as_device() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let nodes = vec![SnapshotNode {
            parent: None,
            block_hash: 1,
            workers: vec![0],
            tiers: vec![],
        }];
        tree.restore_snapshot(std::slice::from_ref(&a), &nodes)
            .unwrap();
        assert_eq!(
            tree.match_prefix(None, &[1]).device_workers(),
            workers(&[&a])
        );
    }

    #[test]
    fn restore_rejects_tiers_that_do_not_pair_with_workers() {
        let tree = HashTree::new();
        let a = worker("http://a", 0);
        let b = worker("http://b", 0);
        let nodes = vec![SnapshotNode {
            parent: None,
            block_hash: 1,
            workers: vec![0, 1],
            tiers: vec![Tiers::HOST.bits()], // one entry for two carriers
        }];
        assert_eq!(
            tree.restore_snapshot(&[a, b], &nodes),
            Err(RestoreError::TierTableMismatch { index: 0 }),
        );
        assert_eq!(tree.node_count(), 0);
    }

    /// `retain_carriers` is the only correct way to filter a record's
    /// carriers: workers and tiers are parallel by index, so dropping a worker
    /// must drop its tier entry, not shift a neighbour's onto it.
    #[test]
    fn retain_carriers_keeps_tiers_aligned() {
        let mut rec = SnapshotNode {
            parent: None,
            block_hash: 1,
            workers: vec![0, 1, 2],
            tiers: vec![Tiers::DEVICE.bits(), Tiers::HOST.bits(), Tiers::ALL.bits()],
        };
        // Drop worker 1, renumber 2 → 1.
        rec.retain_carriers(|w| match w {
            0 => Some(0),
            2 => Some(1),
            _ => None,
        });
        assert_eq!(rec.workers, vec![0, 1]);
        assert_eq!(rec.tiers, vec![Tiers::DEVICE.bits(), Tiers::ALL.bits()]);

        // A legacy record stays legacy: no tiers are invented.
        let mut legacy = SnapshotNode {
            parent: None,
            block_hash: 1,
            workers: vec![0, 1],
            tiers: vec![],
        };
        legacy.retain_carriers(|w| (w == 1).then_some(0));
        assert_eq!(legacy.workers, vec![0]);
        assert!(legacy.tiers.is_empty());
    }

    /// Restoring onto a tree that already holds live state must union
    /// carriers, not duplicate nodes — the steady-state case when a rank's
    /// buffered events land before its snapshot.
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
        assert_eq!(m.workers, workers(&[&a, &b]));
    }

    /// Snapshot fidelity must not depend on chains sharing a shard.
    #[test]
    fn export_restore_spans_shards() {
        let src = HashTree::new();
        let a = worker("http://a", 0);
        for r in 0..128i64 {
            src.insert(&a, None, &[r * 7919, r * 7919 + 1]);
        }
        let used = (0..128i64)
            .map(|r| shard_of(r * 7919))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(used > 1, "expected multiple shards, got {used}");

        let (worker_table, nodes) = src.export_snapshot();
        let dst = HashTree::new();
        dst.restore_snapshot(&worker_table, &nodes).unwrap();

        assert_eq!(dst.node_count(), src.node_count());
        for r in 0..128i64 {
            let m = dst.match_prefix(None, &[r * 7919, r * 7919 + 1]);
            assert_eq!(m.matched_blocks, 2, "chain {r} must survive the round trip");
            assert_eq!(m.workers, workers(&[&a]));
        }
    }
}
