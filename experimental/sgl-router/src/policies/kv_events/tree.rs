//! Hash-keyed radix tree for KV-cache event indexing.
//!
//! Each non-root node represents one block hash (`i64`). A node's children
//! are keyed by the *next* block hash in a chain, so a path from the root
//! down to depth `n` represents a chain of `n` block hashes. Every node
//! tracks the set of [`KvWorkerId`]s that hold the chain ending at that
//! node.
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

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
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

/// Result of [`HashTree::match_prefix`].
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Number of leading block hashes from the input slice that matched a
    /// path from the root.
    pub matched_blocks: usize,
    /// Workers holding the deepest matched node. Empty when
    /// `matched_blocks == 0`.
    pub workers: HashSet<KvWorkerId>,
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
    workers: HashSet<KvWorkerId>,
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
            workers: HashSet::new(),
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
                workers: HashSet::new(),
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
                .is_some_and(|n| n.workers.contains(worker))
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

    fn insert(&mut self, worker: &KvWorkerId, parent_hash: Option<i64>, block_hashes: &[i64]) {
        if block_hashes.is_empty() {
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
            child.workers.insert(worker.clone());
            child.last_used.store(now, Ordering::Relaxed);
            current = child_id;
            prev_hash = Some(h);
        }
    }

    /// Drop `worker` from every node in THIS shard carrying any hash in
    /// `block_hashes`, pruning nodes that become empty + childless.
    fn remove(&mut self, worker: &KvWorkerId, block_hashes: &[i64]) {
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
            let still_present = match self.nodes.get_mut(&id) {
                Some(node) => {
                    node.workers.remove(worker);
                    node.workers.is_empty() && node.children.is_empty()
                }
                None => false,
            };
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
            if let Some(node) = self.nodes.get_mut(&id) {
                if node.workers.remove(worker)
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
            return MatchResult {
                matched_blocks: 0,
                workers: HashSet::new(),
            };
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
        let workers = match last_match_node {
            Some(id) => self
                .nodes
                .get(&id)
                .map(|n| n.workers.clone())
                .unwrap_or_default(),
            None => HashSet::new(),
        };
        MatchResult {
            matched_blocks: matched,
            workers,
        }
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
        if let Some(node) = self.nodes.get_mut(&victim) {
            node.workers.clear();
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
                && ids
                    .iter()
                    .any(|id| st.nodes.get(id).is_some_and(|n| n.workers.contains(worker)))
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

    /// Apply a `BlockStored` event.
    ///
    /// Walks from `parent_hash`'s node (or root) and descends along
    /// `block_hashes`, marking every visited node as held by `worker`.
    /// Empty `block_hashes` is a no-op.
    pub fn insert(&self, worker: &KvWorkerId, parent_hash: Option<i64>, block_hashes: &[i64]) {
        if block_hashes.is_empty() {
            return;
        }
        let (idx, effective_parent) = self.route_insert(worker, parent_hash, block_hashes);
        self.shards[idx]
            .write()
            .insert(worker, effective_parent, block_hashes);
    }

    /// Apply a `BlockRemoved` event.
    ///
    /// For every node carrying any hash in `block_hashes`, drop `worker`
    /// from that node's worker set. Nodes that become empty AND childless
    /// are pruned (cascading upward).
    ///
    /// Removing the worker from a node does NOT remove the node if other
    /// workers still hold it.
    ///
    /// A removed hash can be a chain root in one shard and an interior block
    /// of a chain rooted elsewhere in another, so this fans out across all
    /// shards. Each shard's local `by_hash` short-circuits shards that don't
    /// carry any of the hashes.
    pub fn remove(&self, worker: &KvWorkerId, block_hashes: &[i64]) {
        if block_hashes.is_empty() {
            return;
        }
        for shard in &self.shards {
            shard.write().remove(worker, block_hashes);
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
            return MatchResult {
                matched_blocks: 0,
                workers: HashSet::new(),
            };
        }
        let (idx, effective_parent) = self.route_match(parent_hash, block_hashes);
        self.shards[idx]
            .read()
            .match_prefix(effective_parent, block_hashes)
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
    /// Whether any shard's reverse index carries `hash`.
    fn debug_has_hash(&self, hash: i64) -> bool {
        self.shards
            .iter()
            .any(|s| s.read().by_hash.contains_key(&hash))
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
}
