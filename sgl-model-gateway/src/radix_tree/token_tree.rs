//! Token-based radix tree for gRPC router with pre-tokenized input.
//!
//! This implementation uses token IDs (`u32`) instead of characters,
//! matching SGLang's Python scheduler which operates on token arrays.
//!
//! **Page-aligned design**: Following SGLang's radix cache, tokens are grouped
//! into pages (default 16 tokens). Only page-aligned prefixes are cached.
//! Sequences shorter than PAGE_SIZE get no cache benefit (matching engine behavior).
//!
//! Benefits:
//! - O(1) page-key comparisons vs O(n) single-token lookups
//! - Aligned with SGLang's internal KV cache page structure
//! - Reduced hash table overhead (1 lookup per PAGE_SIZE tokens)

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    sync::{
        atomic::{AtomicI32, AtomicU64, Ordering},
        Arc, Weak,
    },
};

use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock as ParkingLotRwLock;
use tracing::debug;

use super::{
    common::{MatchResult, TenantId},
    RadixTree,
};

/// Token ID type (matches SGLang's token representation)
pub type TokenId = u32;

/// Default page size for token grouping (matches SGLang's default radix cache page size).
/// SGLang supports: 1, 16, 32, 64, 128 depending on attention backend.
///
/// Note: This is a compile-time constant used for the `TokenPageKey` type.
/// The `TokenTree::with_config()` constructor accepts page_size as a parameter,
/// but currently only page_size=16 is supported. Future versions may support
/// other sizes via const generics or dynamic key types.
pub const PAGE_SIZE: usize = 16;

/// A page of tokens used as the children map key.
/// Fixed-size array enables efficient hashing and comparison.
pub type TokenPageKey = [TokenId; PAGE_SIZE];

type NodeRef = Arc<Node>;

/// Shard counts for DashMaps to balance concurrency vs allocation overhead.
/// Root node has more shards due to higher contention.
const ROOT_SHARD_COUNT: usize = 32;
/// Child nodes typically have few entries, minimize shard overhead.
const NODE_SHARD_COUNT: usize = 4;

/// Eviction policy matching SGLang's scheduler options.
/// The gateway should use the same policy as the backend worker to stay in sync.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used - evict oldest by last access time (default)
    #[default]
    Lru,
    /// Least Frequently Used - evict by (hit_count, last_access_time)
    Lfu,
    /// First In First Out - evict oldest by creation time
    Fifo,
    /// Most Recently Used - evict newest by last access time
    Mru,
    /// First In Last Out (stack) - evict newest by creation time
    Filo,
    /// Priority-based - evict by (priority, last_access_time), lower priority first
    Priority,
}

/// Align token count to page boundary (truncate to nearest page).
/// Matches SGLang's: `page_aligned_len = len(key) // page_size * page_size`
#[inline]
fn align_to_page(len: usize) -> usize {
    (len / PAGE_SIZE) * PAGE_SIZE
}

/// Extract page key from token slice (first PAGE_SIZE tokens).
/// Panics if tokens.len() < PAGE_SIZE.
#[inline]
fn make_page_key(tokens: &[TokenId]) -> TokenPageKey {
    debug_assert!(tokens.len() >= PAGE_SIZE);
    let mut key = [0u32; PAGE_SIZE];
    key.copy_from_slice(&tokens[..PAGE_SIZE]);
    key
}

/// A fast hasher for token page keys.
/// Uses FxHash-style multiplication mixing for excellent distribution.
#[derive(Default)]
struct TokenPageHasher(u64);

impl Hasher for TokenPageHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        // Process 4 bytes at a time (each token is u32)
        for chunk in bytes.chunks(4) {
            if chunk.len() == 4 {
                let val = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                // FxHash-style mixing: multiply by golden ratio prime
                self.0 = self
                    .0
                    .wrapping_add(val as u64)
                    .wrapping_mul(0x517cc1b727220a95);
            }
        }
    }
}

type TokenPageHasherBuilder = BuildHasherDefault<TokenPageHasher>;

/// Create a children DashMap with page-key lookup
#[inline]
fn new_children_map() -> DashMap<TokenPageKey, NodeRef, TokenPageHasherBuilder> {
    DashMap::with_hasher_and_shard_amount(TokenPageHasherBuilder::default(), NODE_SHARD_COUNT)
}

/// Create a tenant access time DashMap
#[inline]
fn new_tenant_map() -> DashMap<TenantId, u64> {
    DashMap::with_shard_amount(NODE_SHARD_COUNT)
}

/// Result of a prefix match operation with token counts.
#[derive(Debug, Clone)]
pub struct PrefixMatchResult {
    /// The tenant that owns the matched prefix
    pub tenant: TenantId,
    /// Number of tokens matched
    pub matched_token_count: usize,
    /// Total number of tokens in the input
    pub input_token_count: usize,
}

impl MatchResult for PrefixMatchResult {
    fn tenant(&self) -> &TenantId {
        &self.tenant
    }

    fn matched_count(&self) -> usize {
        self.matched_token_count
    }

    fn input_count(&self) -> usize {
        self.input_token_count
    }
}

/// Global tenant string intern pool to avoid repeated allocations.
/// Uses DashMap for concurrent access with minimal contention.
static TENANT_INTERN_POOL: Lazy<DashMap<Arc<str>, ()>> = Lazy::new(DashMap::new);

/// Intern tenant ID to avoid repeated allocations.
/// Returns cached Arc<str> if tenant was seen before.
fn intern_tenant(tenant: &str) -> TenantId {
    // Fast path: check if already interned
    if let Some(entry) = TENANT_INTERN_POOL.get(tenant) {
        return Arc::clone(entry.key());
    }

    // Slow path: intern new tenant
    let interned: Arc<str> = Arc::from(tenant);
    TENANT_INTERN_POOL.insert(Arc::clone(&interned), ());
    interned
}

/// Global timestamp counter for LRU ordering
static GLOBAL_TIMESTAMP: AtomicU64 = AtomicU64::new(0);

fn next_timestamp() -> u64 {
    GLOBAL_TIMESTAMP.fetch_add(1, Ordering::Relaxed)
}

/// Node in the token-based radix tree.
/// Uses parking_lot RwLock for better performance (no poisoning, smaller size).
/// Parent pointers enable O(d) ancestor cleanup during eviction.
struct Node {
    /// Token sequence stored at this node (always page-aligned length, multiple of PAGE_SIZE)
    tokens: ParkingLotRwLock<Vec<TokenId>>,
    /// Children nodes keyed by first PAGE_SIZE tokens (page key)
    children: DashMap<TokenPageKey, NodeRef, TokenPageHasherBuilder>,
    /// Tenants that own this node with last access timestamps
    tenant_last_access_time: DashMap<TenantId, u64>,
    /// Cached last tenant for fast access (probabilistic update)
    last_tenant: ParkingLotRwLock<Option<TenantId>>,
    /// Parent node (Weak to avoid reference cycles). None for root.
    parent: ParkingLotRwLock<Weak<Node>>,
    /// This node's key in parent's children map (for O(1) removal)
    page_key: ParkingLotRwLock<Option<TokenPageKey>>,
    /// Hit count for LFU eviction policy (incremented on each access)
    hit_count: AtomicU64,
    /// Creation timestamp for FIFO/FILO eviction policies
    creation_time: u64,
    /// Priority for priority-based eviction (higher = less likely to evict)
    priority: AtomicI32,
}

impl Node {
    fn new(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens: ParkingLotRwLock::new(tokens),
            children: new_children_map(),
            tenant_last_access_time: new_tenant_map(),
            last_tenant: ParkingLotRwLock::new(None),
            parent: ParkingLotRwLock::new(Weak::new()),
            page_key: ParkingLotRwLock::new(None),
            hit_count: AtomicU64::new(0),
            creation_time: next_timestamp(),
            priority: AtomicI32::new(0),
        }
    }

    #[allow(dead_code)] // Reserved for priority-based eviction policy support
    fn new_with_priority(tokens: Vec<TokenId>, priority: i32) -> Self {
        Self {
            tokens: ParkingLotRwLock::new(tokens),
            children: new_children_map(),
            tenant_last_access_time: new_tenant_map(),
            last_tenant: ParkingLotRwLock::new(None),
            parent: ParkingLotRwLock::new(Weak::new()),
            page_key: ParkingLotRwLock::new(None),
            hit_count: AtomicU64::new(0),
            creation_time: next_timestamp(),
            priority: AtomicI32::new(priority),
        }
    }

    fn new_root() -> Self {
        Self {
            tokens: ParkingLotRwLock::new(Vec::new()),
            children: DashMap::with_hasher_and_shard_amount(
                TokenPageHasherBuilder::default(),
                ROOT_SHARD_COUNT,
            ),
            tenant_last_access_time: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
            last_tenant: ParkingLotRwLock::new(None),
            parent: ParkingLotRwLock::new(Weak::new()),
            page_key: ParkingLotRwLock::new(None),
            hit_count: AtomicU64::new(0),
            creation_time: 0,                   // Root is always oldest
            priority: AtomicI32::new(i32::MIN), // Root has minimum priority (matches SGLang)
        }
    }

    /// Set parent pointer and page key for this node
    fn set_parent(&self, parent: &NodeRef, key: TokenPageKey) {
        *self.parent.write() = Arc::downgrade(parent);
        *self.page_key.write() = Some(key);
    }

    /// Check if this node is empty (no tenants and no children)
    fn is_empty(&self) -> bool {
        self.tenant_last_access_time.is_empty() && self.children.is_empty()
    }

    /// Get any tenant that owns this node (for match results)
    fn get_any_tenant(&self) -> Option<TenantId> {
        // Fast path: check cached tenant (parking_lot has no poisoning)
        let guard = self.last_tenant.read();
        if let Some(ref tenant) = *guard {
            // Use borrowed lookup to avoid Arc clone for validation
            if self.tenant_last_access_time.contains_key(tenant.as_ref()) {
                return Some(Arc::clone(tenant));
            }
        }
        drop(guard);

        // Slow path: iterate to find any tenant
        self.tenant_last_access_time
            .iter()
            .next()
            .map(|entry| Arc::clone(entry.key()))
    }

    /// Update tenant access and cache (with probabilistic update to reduce contention).
    ///
    /// # Arguments
    /// * `tenant` - The tenant to touch
    /// * `track_lfu` - If true, increment hit_count for LFU policy (only needed when eviction_policy == LFU)
    fn touch_tenant(&self, tenant: &TenantId, track_lfu: bool) {
        let ts = next_timestamp();

        // Conditionally increment hit count (only for LFU policy to reduce contention)
        if track_lfu {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
        }

        // Fast path: try to update existing entry without Arc clone
        // DashMap supports Borrow<str> lookups, avoiding allocation
        if let Some(mut entry) = self.tenant_last_access_time.get_mut(tenant.as_ref()) {
            *entry = ts;
        } else {
            // Slow path: insert new entry (requires Arc clone)
            self.tenant_last_access_time.insert(Arc::clone(tenant), ts);
        }

        // Probabilistic cache update (1/16 chance) to reduce write contention
        if ts & 0xF == 0 {
            if let Some(mut guard) = self.last_tenant.try_write() {
                *guard = Some(Arc::clone(tenant));
            }
        }
    }

    /// Update priority (take max to propagate higher priority, matching SGLang)
    #[allow(dead_code)] // Reserved for priority-based eviction policy support
    fn update_priority(&self, new_priority: i32) {
        // Use fetch_max for correct concurrent updates (avoids CAS race condition)
        self.priority.fetch_max(new_priority, Ordering::Relaxed);
    }
}

/// Token-based radix tree for cache-aware routing.
pub struct TokenTree {
    root: NodeRef,
    /// Track total tokens per tenant for eviction decisions
    tenant_token_count: DashMap<TenantId, usize>,
    /// Eviction policy (should match the backend worker's policy)
    eviction_policy: EvictionPolicy,
    /// Page size for token grouping (should match the backend worker's page size)
    page_size: usize,
}

impl Default for TokenTree {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenTree {
    /// Create a new TokenTree with default settings (page_size=16, eviction_policy=LRU).
    pub fn new() -> Self {
        Self::with_config(PAGE_SIZE, EvictionPolicy::default())
    }

    /// Create a new TokenTree with a specific eviction policy (uses default page_size=16).
    /// The policy should match the backend worker's policy to stay in sync.
    pub fn with_policy(policy: EvictionPolicy) -> Self {
        Self::with_config(PAGE_SIZE, policy)
    }

    /// Create a new TokenTree with specific page size and eviction policy.
    ///
    /// # Arguments
    /// * `page_size` - Token page size for grouping (must match backend worker's page size).
    ///   SGLang supports: 1, 16, 32, 64, 128 depending on attention backend.
    ///   **Note**: Currently only page_size=16 is supported at compile time.
    /// * `policy` - Eviction policy (should match the backend worker's policy).
    ///
    /// # Panics
    /// Panics if `page_size` != 16 (compile-time limitation, future versions may support other sizes).
    pub fn with_config(page_size: usize, policy: EvictionPolicy) -> Self {
        assert_eq!(
            page_size, PAGE_SIZE,
            "TokenTree currently only supports page_size={} (compile-time limitation). \
             Got page_size={}. Future versions may support configurable page sizes.",
            PAGE_SIZE, page_size
        );
        Self {
            root: Arc::new(Node::new_root()),
            tenant_token_count: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
            eviction_policy: policy,
            page_size,
        }
    }

    /// Get the current eviction policy.
    pub fn eviction_policy(&self) -> EvictionPolicy {
        self.eviction_policy
    }

    /// Get the page size used for token grouping.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Insert a token sequence with associated tenant.
    ///
    /// **Page-aligned**: Input is aligned to PAGE_SIZE boundary.
    /// Sequences shorter than PAGE_SIZE are skipped (no cache benefit).
    pub fn insert_tokens(&self, tokens: &[TokenId], tenant: &str) {
        // Align to page boundary (truncate to nearest page)
        let aligned_len = align_to_page(tokens.len());
        if aligned_len == 0 {
            // Sequence too short for cache benefit (matches SGLang behavior)
            return;
        }
        let tokens = &tokens[..aligned_len];

        let tenant_id = intern_tenant(tenant);

        // Ensure tenant exists at root
        self.root
            .tenant_last_access_time
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        self.tenant_token_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        let mut remaining = tokens;
        let mut current = Arc::clone(&self.root);
        let mut tokens_added = 0usize;

        // Compute once: only track hit counts for LFU policy
        let track_lfu = self.eviction_policy == EvictionPolicy::Lfu;

        // Result type to carry state out of the match block
        // This allows the entry guard to be dropped before we update current
        enum InsertStep {
            Done(usize),
            Continue { next: NodeRef, advance: usize },
        }

        while remaining.len() >= PAGE_SIZE {
            // Use first PAGE_SIZE tokens as key for children lookup
            let page_key = make_page_key(remaining);

            let step = match current.children.entry(page_key) {
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    // No child with this page key - create new node
                    let new_node = Arc::new(Node::new(remaining.to_vec()));
                    new_node.set_parent(&current, page_key);
                    new_node.touch_tenant(&tenant_id, track_lfu);
                    entry.insert(new_node);
                    InsertStep::Done(remaining.len())
                }
                dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                    let child = Arc::clone(entry.get());
                    let child_tokens = child.tokens.read();
                    let child_len = child_tokens.len();

                    // Find common prefix length (page-aligned)
                    let common_len = remaining
                        .iter()
                        .zip(child_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    // Align common length to page boundary
                    let common_len = align_to_page(common_len);

                    if common_len == 0 {
                        // No page-aligned match despite same page key (shouldn't happen)
                        drop(child_tokens);
                        InsertStep::Done(0)
                    } else if common_len == child_len {
                        // Full match with child - continue traversal
                        drop(child_tokens);
                        child.touch_tenant(&tenant_id, track_lfu);
                        InsertStep::Continue {
                            next: child,
                            advance: common_len,
                        }
                    } else if common_len >= remaining.len() {
                        // Input is prefix of child - split child at page boundary
                        // Strategy: Create NEW intermediate node with prefix tokens,
                        // keep original child as suffix (preserving its children/tenants)
                        let common_len = align_to_page(remaining.len());
                        let prefix_tokens: Vec<TokenId> = child_tokens[..common_len].to_vec();
                        let suffix_page_key = make_page_key(&child_tokens[common_len..]);

                        // Check if tenant already owned the child (tokens already counted)
                        let tenant_already_owned = child
                            .tenant_last_access_time
                            .contains_key(tenant_id.as_ref());
                        drop(child_tokens);

                        // Modify original child to hold only suffix tokens
                        let mut child_tokens_write = child.tokens.write();
                        let suffix_tokens: Vec<TokenId> = child_tokens_write[common_len..].to_vec();
                        *child_tokens_write = suffix_tokens;
                        drop(child_tokens_write);

                        // Create intermediate node with prefix - clone tenant map (O(1))
                        // Intermediate inherits metadata from child (represents same prefix)
                        let intermediate_node = Arc::new(Node {
                            tokens: ParkingLotRwLock::new(prefix_tokens),
                            children: new_children_map(),
                            tenant_last_access_time: child.tenant_last_access_time.clone(),
                            last_tenant: ParkingLotRwLock::new(child.last_tenant.read().clone()),
                            parent: ParkingLotRwLock::new(Arc::downgrade(&current)),
                            page_key: ParkingLotRwLock::new(Some(page_key)),
                            hit_count: AtomicU64::new(child.hit_count.load(Ordering::Relaxed)),
                            creation_time: child.creation_time,
                            priority: AtomicI32::new(child.priority.load(Ordering::Relaxed)),
                        });

                        // Add original child (now suffix) as child of intermediate
                        // Update child's parent to point to intermediate
                        child.set_parent(&intermediate_node, suffix_page_key);
                        intermediate_node
                            .children
                            .insert(suffix_page_key, Arc::clone(&child));

                        // Replace entry with intermediate node
                        entry.insert(intermediate_node.clone());

                        intermediate_node.touch_tenant(&tenant_id, track_lfu);

                        // Only count new tokens if tenant didn't already own this path
                        let new_tokens = if tenant_already_owned { 0 } else { common_len };
                        InsertStep::Done(new_tokens)
                    } else {
                        // Partial match - need to split and add new branch at page boundary
                        // Strategy: Create NEW intermediate node with common prefix,
                        // keep original child as one suffix, create new node for other suffix
                        let prefix_tokens: Vec<TokenId> = child_tokens[..common_len].to_vec();
                        let child_suffix_page_key = make_page_key(&child_tokens[common_len..]);

                        // Check if tenant already owned the child (common prefix already counted)
                        let tenant_already_owned = child
                            .tenant_last_access_time
                            .contains_key(tenant_id.as_ref());
                        drop(child_tokens);

                        // Modify original child to hold only its suffix tokens
                        let mut child_tokens_write = child.tokens.write();
                        let child_suffix: Vec<TokenId> = child_tokens_write[common_len..].to_vec();
                        *child_tokens_write = child_suffix;
                        drop(child_tokens_write);

                        // Create intermediate node with common prefix - clone tenant map (O(1))
                        // Intermediate inherits metadata from child (represents same prefix)
                        let intermediate_node = Arc::new(Node {
                            tokens: ParkingLotRwLock::new(prefix_tokens),
                            children: new_children_map(),
                            tenant_last_access_time: child.tenant_last_access_time.clone(),
                            last_tenant: ParkingLotRwLock::new(child.last_tenant.read().clone()),
                            parent: ParkingLotRwLock::new(Arc::downgrade(&current)),
                            page_key: ParkingLotRwLock::new(Some(page_key)),
                            hit_count: AtomicU64::new(child.hit_count.load(Ordering::Relaxed)),
                            creation_time: child.creation_time,
                            priority: AtomicI32::new(child.priority.load(Ordering::Relaxed)),
                        });

                        // Add original child (now suffix) as child of intermediate
                        // Update child's parent to point to intermediate
                        child.set_parent(&intermediate_node, child_suffix_page_key);
                        intermediate_node
                            .children
                            .insert(child_suffix_page_key, Arc::clone(&child));

                        // Create new node for the remaining input suffix
                        let new_remaining = &remaining[common_len..];
                        let new_branch_tokens = if new_remaining.len() >= PAGE_SIZE {
                            let new_node = Arc::new(Node::new(new_remaining.to_vec()));
                            let new_page_key = make_page_key(new_remaining);
                            new_node.set_parent(&intermediate_node, new_page_key);
                            new_node.touch_tenant(&tenant_id, track_lfu);
                            intermediate_node.children.insert(new_page_key, new_node);
                            new_remaining.len()
                        } else {
                            0
                        };

                        // Replace entry with intermediate node
                        entry.insert(intermediate_node.clone());

                        intermediate_node.touch_tenant(&tenant_id, track_lfu);

                        // Count: new branch tokens + common prefix only if tenant is new
                        let common_tokens = if tenant_already_owned { 0 } else { common_len };
                        InsertStep::Done(new_branch_tokens + common_tokens)
                    }
                }
            };

            match step {
                InsertStep::Done(added) => {
                    tokens_added += added;
                    break;
                }
                InsertStep::Continue { next, advance } => {
                    tokens_added += advance;
                    remaining = &remaining[advance..];
                    current = next;
                }
            }
        }

        // Update tenant token count
        if tokens_added > 0 {
            self.tenant_token_count
                .entry(tenant_id)
                .and_modify(|c| *c += tokens_added)
                .or_insert(tokens_added);
        }
    }

    /// Find longest matching prefix with detailed counts.
    ///
    /// **Page-aligned**: Input is aligned to PAGE_SIZE boundary before lookup.
    /// Sequences shorter than PAGE_SIZE return 0 matched tokens.
    pub fn match_prefix_with_counts(&self, tokens: &[TokenId]) -> PrefixMatchResult {
        let input_token_count = tokens.len();

        // Align to page boundary (truncate to nearest page)
        let aligned_len = align_to_page(tokens.len());
        if aligned_len == 0 {
            // Sequence too short for cache lookup (matches SGLang behavior)
            return PrefixMatchResult {
                tenant: self
                    .root
                    .get_any_tenant()
                    .unwrap_or_else(|| Arc::from("empty")),
                matched_token_count: 0,
                input_token_count,
            };
        }
        let tokens = &tokens[..aligned_len];

        let mut matched_tokens = 0;
        let mut last_tenant: Option<TenantId> = None;
        let mut remaining = tokens;
        let mut current = Arc::clone(&self.root);

        // Compute once: only track hit counts for LFU policy
        let track_lfu = self.eviction_policy == EvictionPolicy::Lfu;

        enum MatchStep {
            Done,
            Continue {
                next: NodeRef,
                advance: usize,
                tenant: Option<TenantId>,
            },
            PartialMatch {
                matched: usize,
                tenant: Option<TenantId>,
            },
        }

        while remaining.len() >= PAGE_SIZE {
            // Use first PAGE_SIZE tokens as key for children lookup
            let page_key = make_page_key(remaining);

            let step = match current.children.get(&page_key) {
                None => MatchStep::Done,
                Some(child_ref) => {
                    let child = Arc::clone(child_ref.value());
                    drop(child_ref);

                    let child_tokens = child.tokens.read();

                    // Count matching tokens
                    let match_len = remaining
                        .iter()
                        .zip(child_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    // Align match length to page boundary
                    let match_len = align_to_page(match_len);

                    if match_len == 0 {
                        MatchStep::Done
                    } else {
                        let tenant = child.get_any_tenant();

                        // If node has no tenants (all evicted), stop matching here
                        // This ensures we only route to workers that still have the prefix cached
                        if tenant.is_none() {
                            MatchStep::Done
                        } else {
                            // Update timestamp on match to keep LRU in sync with backend
                            // SGLang does: child.last_access_time = access_time
                            if let Some(ref t) = tenant {
                                child.touch_tenant(t, track_lfu);
                            }

                            if match_len < child_tokens.len() {
                                // Partial match within node (at page boundary)
                                MatchStep::PartialMatch {
                                    matched: match_len,
                                    tenant,
                                }
                            } else {
                                // Full match - continue
                                drop(child_tokens);
                                MatchStep::Continue {
                                    next: child,
                                    advance: match_len,
                                    tenant,
                                }
                            }
                        }
                    }
                }
            };

            match step {
                MatchStep::Done => break,
                MatchStep::PartialMatch { matched, tenant } => {
                    matched_tokens += matched;
                    if let Some(t) = tenant {
                        last_tenant = Some(t);
                    }
                    break;
                }
                MatchStep::Continue {
                    next,
                    advance,
                    tenant,
                } => {
                    matched_tokens += advance;
                    if let Some(t) = tenant {
                        last_tenant = Some(t);
                    }
                    remaining = &remaining[advance..];
                    current = next;
                }
            }
        }

        PrefixMatchResult {
            tenant: last_tenant.unwrap_or_else(|| Arc::from("empty")),
            matched_token_count: matched_tokens,
            input_token_count,
        }
    }

    /// Legacy prefix_match API returning (matched_tokens, tenant_string).
    pub fn prefix_match_legacy(&self, tokens: &[TokenId]) -> (Vec<TokenId>, String) {
        let result = self.match_prefix_with_counts(tokens);
        let matched: Vec<TokenId> = tokens[..result.matched_token_count].to_vec();
        (matched, result.tenant.to_string())
    }

    /// Get token counts per tenant.
    #[allow(dead_code)]
    pub fn get_tenant_token_counts(&self) -> HashMap<String, usize> {
        self.tenant_token_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    /// Compute eviction priority for a node based on the configured policy.
    /// Returns (primary_key, tiebreaker) where lower values are evicted first.
    ///
    /// Uses wrapping arithmetic for u64→i64 conversion. This preserves relative ordering
    /// for values within i64::MAX of each other, which is sufficient since our timestamps
    /// are monotonic counters starting from 0.
    fn compute_eviction_priority(&self, node: &Node, last_access_time: u64) -> (i64, u64) {
        match self.eviction_policy {
            EvictionPolicy::Lru => {
                // Lower last_access_time = older = evict first
                // Wrapping cast preserves ordering for nearby values
                (last_access_time as i64, 0)
            }
            EvictionPolicy::Lfu => {
                // Lower hit_count = less used = evict first, tiebreak by last_access_time
                let hit_count = node.hit_count.load(Ordering::Relaxed);
                (hit_count as i64, last_access_time)
            }
            EvictionPolicy::Fifo => {
                // Lower creation_time = older = evict first
                (node.creation_time as i64, 0)
            }
            EvictionPolicy::Mru => {
                // Higher last_access_time = newer = evict first (negate for min-heap)
                // wrapping_neg handles i64::MIN correctly (wraps to itself)
                ((last_access_time as i64).wrapping_neg(), 0)
            }
            EvictionPolicy::Filo => {
                // Higher creation_time = newer = evict first (negate for min-heap)
                ((node.creation_time as i64).wrapping_neg(), 0)
            }
            EvictionPolicy::Priority => {
                // Lower priority = less important = evict first, tiebreak by last_access_time
                let priority = node.priority.load(Ordering::Relaxed);
                (priority as i64, last_access_time)
            }
        }
    }

    /// Evict entries for a tenant to reduce to max_tokens.
    /// Uses leaf-first eviction with the configured policy (matching SGLang's behavior).
    pub fn evict_tenant(&self, tenant: &TenantId, max_tokens: usize) {
        use std::{cmp::Reverse, collections::BinaryHeap};

        let current_count = self
            .tenant_token_count
            .get(tenant.as_ref())
            .map(|v| *v)
            .unwrap_or(0);

        if current_count <= max_tokens {
            return;
        }

        let to_evict = current_count - max_tokens;
        let mut evicted = 0;

        let mut leaves: Vec<(NodeRef, u64)> = Vec::new();
        self.collect_tenant_leaves(&self.root, tenant, &mut leaves);

        // Min-heap by eviction priority (policy-dependent)
        let mut heap: BinaryHeap<Reverse<((i64, u64), usize)>> = BinaryHeap::new();

        // Pre-allocate for initial leaves. Additional capacity for leaf promotions
        // (when a parent becomes a leaf after child eviction) will be handled by
        // Vec's amortized growth strategy.
        let mut leaf_data: Vec<NodeRef> = Vec::with_capacity(leaves.len());

        for (node, ts) in leaves.drain(..) {
            let priority = self.compute_eviction_priority(&node, ts);
            let idx = leaf_data.len();
            leaf_data.push(node);
            heap.push(Reverse((priority, idx)));
        }

        while evicted < to_evict {
            let Some(Reverse((_, idx))) = heap.pop() else {
                break;
            };

            let node = &leaf_data[idx];
            let (node_tokens, parent_became_leaf) = self.remove_tenant_and_cleanup(node, tenant);
            if node_tokens > 0 {
                evicted += node_tokens;

                // Incremental leaf promotion (matching SGLang's approach)
                if let Some((parent_node, parent_ts)) = parent_became_leaf {
                    let priority = self.compute_eviction_priority(&parent_node, parent_ts);
                    let new_idx = leaf_data.len();
                    leaf_data.push(parent_node);
                    heap.push(Reverse((priority, new_idx)));
                }
            }
        }

        if let Some(mut count) = self.tenant_token_count.get_mut(tenant.as_ref()) {
            *count = count.saturating_sub(evicted);
        }

        debug!(
            tenant = %tenant.as_ref(),
            evicted = evicted,
            remaining = current_count.saturating_sub(evicted),
            policy = ?self.eviction_policy,
            "Evicted tokens from tenant (leaf-first)"
        );
    }

    /// Collect tenant-specific leaves (nodes where tenant exists but no children have tenant).
    fn collect_tenant_leaves(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
        result: &mut Vec<(NodeRef, u64)>,
    ) {
        let is_root = Arc::ptr_eq(node, &self.root);
        let node_has_tenant = !is_root
            && node
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref());

        let mut any_child_has_tenant = false;
        for child_entry in node.children.iter() {
            let child = child_entry.value();
            if child
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
            {
                any_child_has_tenant = true;
            }
            self.collect_tenant_leaves(child, tenant_id, result);
        }

        if node_has_tenant && !any_child_has_tenant {
            if let Some(ts) = node.tenant_last_access_time.get(tenant_id.as_ref()) {
                result.push((Arc::clone(node), *ts));
            }
        }
    }

    fn is_tenant_leaf(&self, node: &NodeRef, tenant_id: &TenantId) -> bool {
        if !node
            .tenant_last_access_time
            .contains_key(tenant_id.as_ref())
        {
            return false;
        }
        for child_entry in node.children.iter() {
            if child_entry
                .value()
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
            {
                return false;
            }
        }
        true
    }

    /// Remove tenant from node and clean up empty ancestors.
    /// Returns (tokens_removed, Option<(parent_node, ts)> if parent became a leaf).
    fn remove_tenant_and_cleanup(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
    ) -> (usize, Option<(NodeRef, u64)>) {
        if node
            .tenant_last_access_time
            .remove(tenant_id.as_ref())
            .is_none()
        {
            return (0, None);
        }

        let node_tokens = node.tokens.read().len();
        let parent_leaf_info = self.cleanup_empty_ancestors_with_parent(node, tenant_id);

        (node_tokens, parent_leaf_info)
    }

    /// Remove empty nodes walking up via parent pointers.
    /// Returns Some((parent_node, ts)) if a parent became a tenant-leaf.
    fn cleanup_empty_ancestors_with_parent(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
    ) -> Option<(NodeRef, u64)> {
        let mut current = Arc::clone(node);
        let mut parent_leaf_info: Option<(NodeRef, u64)> = None;

        loop {
            if !current.is_empty() {
                if parent_leaf_info.is_none() && self.is_tenant_leaf(&current, tenant_id) {
                    if let Some(ts) = current.tenant_last_access_time.get(tenant_id.as_ref()) {
                        parent_leaf_info = Some((Arc::clone(&current), *ts));
                    }
                }
                break;
            }

            let parent_weak = current.parent.read();
            let Some(parent) = parent_weak.upgrade() else {
                break;
            };
            drop(parent_weak);

            let page_key_guard = current.page_key.read();
            let Some(page_key) = *page_key_guard else {
                break;
            };
            drop(page_key_guard);

            parent.children.remove(&page_key);

            if parent_leaf_info.is_none() && self.is_tenant_leaf(&parent, tenant_id) {
                if let Some(ts) = parent.tenant_last_access_time.get(tenant_id.as_ref()) {
                    parent_leaf_info = Some((Arc::clone(&parent), *ts));
                }
            }

            current = parent;
        }

        parent_leaf_info
    }

    /// Get the token count for a specific tenant.
    pub fn tenant_token_size(&self, tenant: &TenantId) -> usize {
        // Use borrowed lookup to avoid Arc hash overhead
        self.tenant_token_count
            .get(tenant.as_ref())
            .map(|v| *v)
            .unwrap_or(0)
    }

    /// Clear the tree to empty state.
    pub fn clear(&self) {
        self.root.children.clear();
        self.root.tenant_last_access_time.clear();
        self.tenant_token_count.clear();
    }
}

impl RadixTree for TokenTree {
    type Key = [TokenId];
    type MatchResult = PrefixMatchResult;

    fn insert(&self, key: &Self::Key, tenant: &str) {
        self.insert_tokens(key, tenant);
    }

    fn prefix_match(&self, key: &Self::Key) -> Option<TenantId> {
        let result = self.match_prefix_with_counts(key);
        if result.matched_token_count > 0 {
            Some(result.tenant)
        } else {
            None
        }
    }

    fn prefix_match_with_counts(&self, key: &Self::Key) -> Self::MatchResult {
        self.match_prefix_with_counts(key)
    }

    fn evict(&self, tenant: &TenantId, max_units: usize) {
        self.evict_tenant(tenant, max_units);
    }

    fn tenant_size(&self, tenant: &TenantId) -> usize {
        self.tenant_token_size(tenant)
    }

    fn reset(&self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use super::*;

    /// Helper to create a page-aligned token sequence starting from `base`.
    /// Creates `pages` full pages of tokens.
    fn make_tokens(base: u32, pages: usize) -> Vec<TokenId> {
        (0..(pages * PAGE_SIZE)).map(|i| base + i as u32).collect()
    }

    #[test]
    fn test_basic_insert_match() {
        let tree = TokenTree::new();

        // Insert 2 pages (32 tokens)
        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");

        // Exact match
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 32);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        // Match first page only
        let first_page = make_tokens(1, 1);
        let result = tree.match_prefix_with_counts(&first_page);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        // Match with extra tokens (truncated to page boundary)
        let mut extended = tokens.clone();
        extended.extend([100, 101, 102, 103, 104]);
        let result = tree.match_prefix_with_counts(&extended);
        assert_eq!(result.matched_token_count, 32);
    }

    #[test]
    fn test_short_sequences_skipped() {
        let tree = TokenTree::new();

        // Sequences shorter than PAGE_SIZE are skipped
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");

        // Should have no entries (too short)
        let counts = tree.get_tenant_token_counts();
        assert!(counts.is_empty() || counts.get("tenant1").copied().unwrap_or(0) == 0);

        // Lookup also returns 0 for short sequences
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, 5);
    }

    #[test]
    fn test_multiple_tenants() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 1);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        // Either tenant is valid
        assert!(result.tenant.as_ref() == "tenant1" || result.tenant.as_ref() == "tenant2");
    }

    #[test]
    fn test_prefix_split() {
        let tree = TokenTree::new();

        // Insert 3 pages first
        let long_tokens = make_tokens(1, 3);
        tree.insert_tokens(&long_tokens, "tenant1");

        // Insert 1 page (causes split)
        let short_tokens = make_tokens(1, 1);
        tree.insert_tokens(&short_tokens, "tenant2");

        // Short match
        let result = tree.match_prefix_with_counts(&short_tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        // Long match
        let result = tree.match_prefix_with_counts(&long_tokens);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");
    }

    #[test]
    fn test_empty_input() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 1);
        tree.insert_tokens(&tokens, "tenant1");

        let result = tree.match_prefix_with_counts(&[]);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, 0);
    }

    #[test]
    fn test_no_match() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 1);
        tree.insert_tokens(&tokens, "tenant1");

        // Different page key
        let other = make_tokens(1000, 1);
        let result = tree.match_prefix_with_counts(&other);
        assert_eq!(result.matched_token_count, 0);
    }

    #[test]
    fn test_eviction() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1, 3);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");

        let counts = tree.get_tenant_token_counts();
        assert!(counts.get("tenant1").unwrap() > &0);

        tree.evict(&TenantId::from("tenant1"), 0);

        // After eviction, counts should be reduced
        let new_counts = tree.get_tenant_token_counts();
        let new_count = new_counts.get("tenant1").copied().unwrap_or(0);
        assert!(new_count < *counts.get("tenant1").unwrap());
    }

    #[test]
    fn test_concurrent_insert_match() {
        let tree = Arc::new(TokenTree::new());
        let mut handles = vec![];

        // Spawn inserters - use page-aligned sequences
        for i in 0..4 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let base = (i * 1000000 + j * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("tenant{}", i));
                }
            }));
        }

        // Spawn matchers
        for i in 0..4 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let base = (i * 1000000 + j * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    let _ = tree.match_prefix_with_counts(&tokens);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_prefix_match_with_counts() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");

        // Exact match
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 2 * PAGE_SIZE);

        // Match first page
        let first_page = make_tokens(1, 1);
        let result = tree.match_prefix_with_counts(&first_page);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.input_token_count, PAGE_SIZE);

        // Extended input (aligned)
        let extended = make_tokens(1, 3);
        let result = tree.match_prefix_with_counts(&extended);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 3 * PAGE_SIZE);
    }

    #[test]
    fn test_disjoint_paths() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(1000, 1);
        let tokens3 = make_tokens(2000, 1);

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");
        tree.insert_tokens(&tokens3, "tenant3");

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");

        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant3");
    }

    #[test]
    fn test_branching_paths() {
        let tree = TokenTree::new();

        // Common first page, different second page
        let mut tokens1 = make_tokens(1, 1);
        tokens1.extend(make_tokens(100, 1));

        let mut tokens2 = make_tokens(1, 1);
        tokens2.extend(make_tokens(200, 1));

        let mut tokens3 = make_tokens(1, 1);
        tokens3.extend(make_tokens(300, 1));

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");
        tree.insert_tokens(&tokens3, "tenant3");

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");

        // Partial match at branch point
        let first_page = make_tokens(1, 1);
        let result = tree.match_prefix_with_counts(&first_page);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
    }

    #[test]
    fn test_radix_tree_trait() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);
        RadixTree::insert(&tree, &tokens, "tenant1");

        let tenant = RadixTree::prefix_match(&tree, &tokens);
        assert!(tenant.is_some());
        assert_eq!(tenant.unwrap().as_ref(), "tenant1");

        // Extended input - should match 2 pages (short sequences get 0)
        let extended = make_tokens(1, 3);
        let result = RadixTree::prefix_match_with_counts(&tree, &extended);
        assert_eq!(result.matched_count(), 2 * PAGE_SIZE);
        assert_eq!(result.input_count(), 3 * PAGE_SIZE);

        assert!(RadixTree::tenant_size(&tree, &TenantId::from("tenant1")) > 0);
    }

    #[test]
    fn test_clear() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(1000, 1);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");

        assert!(!tree.get_tenant_token_counts().is_empty());

        tree.clear();

        assert!(tree.get_tenant_token_counts().is_empty());
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, 0);
    }

    #[test]
    fn test_tenant_token_count() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1, 3);
        let tokens3 = make_tokens(1000, 1);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");
        let tenant2_id: TenantId = Arc::from("tenant2");

        assert!(tree.tenant_token_size(&tenant1_id) >= PAGE_SIZE);
        assert!(tree.tenant_token_size(&tenant2_id) >= PAGE_SIZE);

        let counts = tree.get_tenant_token_counts();
        assert!(counts.contains_key("tenant1"));
        assert!(counts.contains_key("tenant2"));
    }

    #[test]
    fn test_cold_start() {
        let tree = TokenTree::new();
        // Short sequences return 0 (no cache benefit)
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, 5);

        // Page-sized sequences also return 0 on empty tree
        let tokens = make_tokens(1, 1);
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, PAGE_SIZE);
    }

    #[test]
    fn test_exact_match_seq() {
        let tree = TokenTree::new();

        for i in 0..100 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            tree.insert_tokens(&tokens, &format!("tenant{}", i));
        }

        for i in 0..100 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            let result = tree.match_prefix_with_counts(&tokens);
            assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
            assert_eq!(result.tenant.as_ref(), &format!("tenant{}", i));
        }
    }

    #[test]
    fn test_exact_match_concurrent() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 8;
        let entries_per_thread = 100;

        // Insert phase
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let base = (t * 1000000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Match phase
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let base = (t * 1000000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    let result = tree.match_prefix_with_counts(&tokens);
                    assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
                    assert_eq!(result.tenant.as_ref(), &format!("tenant{}", t));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_partial_match_concurrent() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 8;
        let entries_per_thread = 100;

        // Insert full sequences (3 pages)
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let base = (t * 1000000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 3);
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Match with partial (1 page)
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let base = (t * 1000000 + i * 1000) as u32;
                    let partial = make_tokens(base, 1);
                    let result = tree.match_prefix_with_counts(&partial);
                    assert_eq!(result.matched_token_count, PAGE_SIZE);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_group_prefix_insert_match_concurrent() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 8;

        // All threads share the same prefix (1 page)
        let common_prefix = make_tokens(100, 1);

        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            let prefix = common_prefix.clone();
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let mut tokens = prefix.clone();
                    let suffix = make_tokens((t * 10000 + i * 100) as u32, 1);
                    tokens.extend(suffix);
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify prefix matching works
        let result = tree.match_prefix_with_counts(&common_prefix);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
    }

    #[test]
    fn test_mixed_concurrent_insert_match() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 4;

        // Pre-populate some data
        for i in 0..100 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            tree.insert_tokens(&tokens, &format!("initial{}", i));
        }

        let mut handles = vec![];

        // Concurrent inserters
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let base = (10000000 + t * 100000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("new_tenant{}", t));
                }
            }));
        }

        // Concurrent matchers (matching existing data)
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let base = ((t * 10) * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    let result = tree.match_prefix_with_counts(&tokens);
                    assert!(result.matched_token_count > 0);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_simple_eviction() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1000, 2);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // tenant2 should still work
        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_advanced_eviction() {
        let tree = TokenTree::new();

        // Insert multiple paths for tenant1
        let mut tokens1 = make_tokens(1, 1);
        tokens1.extend(make_tokens(100, 1));
        let mut tokens2 = make_tokens(1, 1);
        tokens2.extend(make_tokens(200, 1));
        let mut tokens3 = make_tokens(1, 1);
        tokens3.extend(make_tokens(300, 1));

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1");

        let tenant1_id: TenantId = Arc::from("tenant1");

        // Partial eviction
        let initial_size = tree.tenant_token_size(&tenant1_id);
        tree.evict_tenant(&tenant1_id, initial_size / 2);
        let after_size = tree.tenant_token_size(&tenant1_id);

        assert!(after_size <= initial_size);
    }

    #[test]
    fn test_concurrent_operations_with_eviction() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 4;

        // Pre-populate
        for i in 0..100 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            tree.insert_tokens(&tokens, &format!("tenant{}", i % 4));
        }

        let mut handles = vec![];

        // Inserters
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let base = (10000000 + t * 100000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }

        // Evictors
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                let tenant_id: TenantId = Arc::from(format!("tenant{}", t));
                for _ in 0..10 {
                    tree.evict_tenant(&tenant_id, 50);
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            }));
        }

        // Matchers
        for _ in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let base = (i * 1000) as u32;
                    let tokens = make_tokens(base, 1);
                    let _ = tree.match_prefix_with_counts(&tokens);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_get_used_size_per_tenant() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1, 3);
        let tokens3 = make_tokens(1000, 1);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant2");

        let counts = tree.get_tenant_token_counts();

        assert!(counts.contains_key("tenant1"));
        assert!(counts.contains_key("tenant2"));
        assert!(*counts.get("tenant1").unwrap() >= PAGE_SIZE);
        assert!(*counts.get("tenant2").unwrap() >= PAGE_SIZE);
    }

    #[test]
    fn test_prefix_match_tenant() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        // Both tenants should have access time updated
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        // tenant should be either tenant1 or tenant2 (last_tenant cache)
        assert!(result.tenant.as_ref() == "tenant1" || result.tenant.as_ref() == "tenant2");
    }

    #[test]
    fn test_simple_tenant_eviction() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1000, 2);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // tenant2 should be unaffected
        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_complex_tenant_eviction() {
        let tree = TokenTree::new();

        // Create overlapping paths (common first page, different second pages)
        let mut tokens1 = make_tokens(1, 1);
        tokens1.extend(make_tokens(100, 1));
        let mut tokens2 = make_tokens(1, 1);
        tokens2.extend(make_tokens(200, 1));
        let mut tokens3 = make_tokens(1, 1);
        tokens3.extend(make_tokens(300, 1));

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");
        tree.insert_tokens(&tokens3, "tenant1");

        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // tenant2's path should still work
        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_empty_node_cleanup_after_eviction() {
        // Test that empty nodes are removed from the tree after tenant eviction
        // This prevents memory bloat from orphaned nodes
        let tree = TokenTree::new();

        // Insert a path only owned by tenant1
        let tokens1 = make_tokens(500, 2); // Unique path
        tree.insert_tokens(&tokens1, "tenant1");

        // Verify tree has children before eviction
        assert!(
            !tree.root.children.is_empty(),
            "Tree should have children after insert"
        );

        // Evict tenant1 completely
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // Verify empty nodes were cleaned up
        assert!(
            tree.root.children.is_empty(),
            "Empty nodes should be removed after tenant eviction"
        );

        // Verify tenant count is zero
        assert_eq!(tree.tenant_token_size(&tenant1_id), 0);
    }

    #[test]
    fn test_partial_cleanup_shared_nodes() {
        // Test that shared nodes are NOT cleaned up when still owned by other tenants
        let tree = TokenTree::new();

        // Both tenants share the same path
        let tokens = make_tokens(100, 2);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        // Evict tenant1
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // Tree should still have children (owned by tenant2)
        assert!(
            !tree.root.children.is_empty(),
            "Shared nodes should NOT be removed when other tenants exist"
        );

        // tenant2 should still work
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_leaf_first_eviction() {
        // Test that eviction follows leaf-first LRU like SGLang's radix cache
        //
        // Tree structure for tenant1:
        //   root -> [A: page1] -> [B: page2] -> [C: page3]
        //
        // When we evict, C (leaf) should be evicted first, then B, then A.
        // This ensures shared prefixes (like system prompts near root) survive longer.

        let tree = TokenTree::new();

        // Insert a 3-page path: A -> B -> C
        let path_abc = make_tokens(1, 3); // 3 pages
        tree.insert_tokens(&path_abc, "tenant1");

        // Also insert just the first page (shared prefix)
        let path_a = make_tokens(1, 1); // 1 page
        tree.insert_tokens(&path_a, "tenant1");

        // Initial count should be 3 pages (A, B, C share with the single A)
        let initial_count = tree.tenant_token_size(&Arc::from("tenant1"));
        assert_eq!(initial_count, 3 * PAGE_SIZE);

        // Evict to keep only 2 pages - should evict C (the leaf)
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 2 * PAGE_SIZE);

        // After evicting C, we should still match A and B
        let result = tree.match_prefix_with_counts(&path_abc);
        // Should match 2 pages (A and B), not 3
        assert!(
            result.matched_token_count <= 2 * PAGE_SIZE,
            "Expected at most 2 pages matched after evicting leaf, got {}",
            result.matched_token_count / PAGE_SIZE
        );

        // The prefix path should still work
        let result = tree.match_prefix_with_counts(&path_a);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "Shared prefix A should still be cached"
        );

        // Evict more - should evict B (now the new leaf)
        tree.evict_tenant(&tenant1_id, PAGE_SIZE);

        // Now only A should remain
        let result = tree.match_prefix_with_counts(&path_abc);
        assert!(
            result.matched_token_count <= PAGE_SIZE,
            "Expected at most 1 page matched after evicting B"
        );

        // A should still work
        let result = tree.match_prefix_with_counts(&path_a);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "Root prefix A should survive longest"
        );
    }

    #[test]
    fn test_single_page_operations() {
        let tree = TokenTree::new();

        // Single page operations
        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(1000, 1);
        let tokens3 = make_tokens(2000, 1);

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");
        tree.insert_tokens(&tokens3, "tenant3");

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");

        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant3");
    }

    #[test]
    fn test_prefix_is_subset_of_existing() {
        let tree = TokenTree::new();

        // Insert longer sequence first (3 pages)
        let long_tokens = make_tokens(1, 3);
        tree.insert_tokens(&long_tokens, "tenant1");

        // Insert prefix (1 page)
        let short_tokens = make_tokens(1, 1);
        tree.insert_tokens(&short_tokens, "tenant2");

        // Short match
        let result = tree.match_prefix_with_counts(&short_tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        // Long match
        let result = tree.match_prefix_with_counts(&long_tokens);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");
    }

    #[test]
    fn test_existing_is_prefix_of_new() {
        let tree = TokenTree::new();

        // Insert shorter first (1 page)
        let short_tokens = make_tokens(1, 1);
        tree.insert_tokens(&short_tokens, "tenant1");

        // Insert longer (3 pages)
        let long_tokens = make_tokens(1, 3);
        tree.insert_tokens(&long_tokens, "tenant2");

        // Short match
        let result = tree.match_prefix_with_counts(&short_tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        // Long match
        let result = tree.match_prefix_with_counts(&long_tokens);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_prefix_match_with_counts_accuracy() {
        let tree = TokenTree::new();

        // Insert 4 pages
        let tokens = make_tokens(1, 4);
        tree.insert_tokens(&tokens, "tenant1");

        // Exact match
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 4 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 4 * PAGE_SIZE);

        // Partial match (2 pages)
        let partial = make_tokens(1, 2);
        let result = tree.match_prefix_with_counts(&partial);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 2 * PAGE_SIZE);

        // Extended match (input longer than inserted)
        let extended = make_tokens(1, 6);
        let result = tree.match_prefix_with_counts(&extended);
        assert_eq!(result.matched_token_count, 4 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 6 * PAGE_SIZE);
    }

    #[test]
    fn test_split_at_page_boundary() {
        let tree = TokenTree::new();

        // Insert 3 pages
        let long_tokens = make_tokens(1, 3);
        tree.insert_tokens(&long_tokens, "tenant1");

        // Insert 1 page (causes split at page boundary)
        let short_tokens = make_tokens(1, 1);
        tree.insert_tokens(&short_tokens, "tenant2");

        // 1 page match
        let result = tree.match_prefix_with_counts(&short_tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        // 3 pages match
        let result = tree.match_prefix_with_counts(&long_tokens);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");
    }

    #[test]
    fn test_multiple_splits_same_path() {
        let tree = TokenTree::new();

        // Insert 4 pages first
        let tokens4 = make_tokens(1, 4);
        tree.insert_tokens(&tokens4, "tenant1");

        // Insert 3 pages
        let tokens3 = make_tokens(1, 3);
        tree.insert_tokens(&tokens3, "tenant2");

        // Insert 2 pages
        let tokens2 = make_tokens(1, 2);
        tree.insert_tokens(&tokens2, "tenant3");

        // Insert 1 page
        let tokens1 = make_tokens(1, 1);
        tree.insert_tokens(&tokens1, "tenant4");

        // All should match correctly
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);

        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);

        let result = tree.match_prefix_with_counts(&tokens4);
        assert_eq!(result.matched_token_count, 4 * PAGE_SIZE);
    }

    #[test]
    fn test_high_contention_same_prefix() {
        let tree = Arc::new(TokenTree::new());
        let prefix = make_tokens(100, 1);
        let num_threads = 16;

        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            let p = prefix.clone();
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let mut tokens = p.clone();
                    let suffix = make_tokens((t * 10000 + i * 100) as u32, 1);
                    tokens.extend(suffix);
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify prefix matching
        let result = tree.match_prefix_with_counts(&prefix);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
    }

    #[test]
    fn test_rapid_insert_remove_cycles() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 4;

        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                let tenant_id: TenantId = Arc::from(format!("tenant{}", t));
                for cycle in 0..10 {
                    // Insert
                    for i in 0..20 {
                        let base = (t * 10000000 + cycle * 100000 + i * 1000) as u32;
                        let tokens = make_tokens(base, 2);
                        tree.insert_tokens(&tokens, &format!("tenant{}", t));
                    }
                    // Evict
                    tree.evict_tenant(&tenant_id, 10);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_eviction_empty_tree() {
        let tree = TokenTree::new();
        let tenant_id: TenantId = Arc::from("nonexistent");

        // Should not panic
        tree.evict_tenant(&tenant_id, 0);
        tree.evict_tenant(&tenant_id, 100);
    }

    #[test]
    fn test_eviction_zero_max_size() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1000, 2);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");

        let tenant_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant_id, 0);

        // Eviction with max_size=0 should remove entries
        let size = tree.tenant_token_size(&tenant_id);
        assert!(size < 4 * PAGE_SIZE);
    }

    #[test]
    fn test_eviction_single_tenant_all_entries() {
        let tree = TokenTree::new();

        // Insert multiple entries for one tenant
        for i in 0..10 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            tree.insert_tokens(&tokens, "tenant1");
        }

        let tenant_id: TenantId = Arc::from("tenant1");
        let initial_size = tree.tenant_token_size(&tenant_id);
        assert!(initial_size > 0);

        tree.evict_tenant(&tenant_id, 0);

        let final_size = tree.tenant_token_size(&tenant_id);
        assert!(final_size < initial_size);
    }

    #[test]
    fn test_last_tenant_cache_update() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 1);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        // First match
        let result1 = tree.match_prefix_with_counts(&tokens);
        let first_tenant = result1.tenant.clone();

        // Match again - should get cached tenant
        let result2 = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result2.tenant, first_tenant);
    }

    #[test]
    fn test_stale_cache_after_tenant_removal() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        // Match to populate cache
        let _ = tree.match_prefix_with_counts(&tokens);

        // Evict one tenant
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // Match should still work (tenant2 or cache still valid)
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
    }

    #[test]
    fn test_token_count_consistency_after_operations() {
        let tree = TokenTree::new();

        // Insert
        let tokens1 = make_tokens(1, 3);
        let tokens2 = make_tokens(1, 5);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");

        let tenant1_id: TenantId = Arc::from("tenant1");
        let count1 = tree.tenant_token_size(&tenant1_id);
        assert!(count1 >= PAGE_SIZE);

        // Partial eviction
        tree.evict_tenant(&tenant1_id, count1 / 2);
        let count2 = tree.tenant_token_size(&tenant1_id);
        assert!(count2 <= count1);

        // Insert more
        let tokens3 = make_tokens(2000, 2);
        tree.insert_tokens(&tokens3, "tenant1");
        let count3 = tree.tenant_token_size(&tenant1_id);
        assert!(count3 >= count2);
    }

    #[test]
    fn test_tree_structure_integrity_after_stress() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 8;

        let mut handles = vec![];

        // Stress insert
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..200 {
                    let base = (t * 10000000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify structure by matching
        for t in 0..num_threads {
            for i in 0..10 {
                let base = (t * 10000000 + i * 1000) as u32;
                let tokens = make_tokens(base, 2);
                let result = tree.match_prefix_with_counts(&tokens);
                assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
            }
        }
    }

    #[test]
    fn test_very_long_sequences() {
        let tree = TokenTree::new();

        // Insert a very long sequence (64 pages = 1024 tokens)
        let long_seq = make_tokens(1, 64);
        tree.insert_tokens(&long_seq, "tenant1");

        // Match the full sequence
        let result = tree.match_prefix_with_counts(&long_seq);
        assert_eq!(result.matched_token_count, 64 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        // Match a prefix (32 pages)
        let prefix = make_tokens(1, 32);
        let result = tree.match_prefix_with_counts(&prefix);
        assert_eq!(result.matched_token_count, 32 * PAGE_SIZE);
    }

    #[test]
    fn test_many_tenants_same_path() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);

        for i in 0..100 {
            tree.insert_tokens(&tokens, &format!("tenant{}", i));
        }

        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);

        // Some tenants should have registered
        let counts = tree.get_tenant_token_counts();
        assert!(!counts.is_empty()); // At least some tenants tracked
    }

    #[test]
    fn test_token_id_edge_values() {
        let tree = TokenTree::new();

        // Test with edge case token IDs in page-aligned sequences
        // Create page starting with 0
        let mut zeros_page: Vec<TokenId> = (0..PAGE_SIZE as u32).collect();
        zeros_page[0] = 0;
        tree.insert_tokens(&zeros_page, "tenant1");

        // Create page starting with u32::MAX
        let mut max_page: Vec<TokenId> = (0..PAGE_SIZE as u32).collect();
        max_page[0] = u32::MAX;
        tree.insert_tokens(&max_page, "tenant2");

        // Create page with mixed edge values
        let mut mixed_page: Vec<TokenId> = (0..PAGE_SIZE as u32).collect();
        mixed_page[0] = 0;
        mixed_page[1] = u32::MAX;
        tree.insert_tokens(&mixed_page, "tenant3");

        let (matched, tenant) = tree.prefix_match_legacy(&zeros_page);
        assert_eq!(matched.len(), PAGE_SIZE);
        assert!(tenant == "tenant1" || tenant == "tenant3");

        let (matched, tenant) = tree.prefix_match_legacy(&max_page);
        assert_eq!(matched.len(), PAGE_SIZE);
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_hit_ratio_calculation() {
        use crate::radix_tree::MatchResult;

        let tree = TokenTree::new();
        let one_page = make_tokens(1, 1); // 1 page = PAGE_SIZE tokens
        tree.insert_tokens(&one_page, "tenant1");

        // 100% hit ratio - query exactly the inserted sequence
        let result = tree.match_prefix_with_counts(&one_page);
        assert!((result.hit_ratio() - 1.0).abs() < 0.001);

        // 50% hit ratio - query 2 pages, only 1 page cached
        let two_pages = make_tokens(1, 2);
        let result = tree.match_prefix_with_counts(&two_pages);
        assert!((result.hit_ratio() - 0.5).abs() < 0.001);

        // 0% hit ratio - query non-existent tokens (but page-aligned)
        let no_match = make_tokens(1000, 1);
        let result = tree.match_prefix_with_counts(&no_match);
        assert!(result.hit_ratio() == 0.0);
    }

    #[test]
    fn test_reset_via_trait() {
        use crate::radix_tree::RadixTree;

        let tree = TokenTree::new();
        tree.insert_tokens(&make_tokens(1, 1), "tenant1");
        tree.insert_tokens(&make_tokens(100, 1), "tenant2");

        assert!(!tree.get_tenant_token_counts().is_empty());

        RadixTree::reset(&tree);

        assert!(tree.get_tenant_token_counts().is_empty());
    }

    #[test]
    fn test_eviction_policy_lru() {
        // LRU: Evict oldest by last access time
        let tree = TokenTree::with_policy(EvictionPolicy::Lru);

        // Insert 3 paths, access them in order (oldest first)
        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1"); // Oldest
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1"); // Newest

        // Access tokens2 to make it newer
        let _ = tree.match_prefix_with_counts(&tokens2);

        // Evict 1 page - should evict tokens1 (oldest by last_access_time)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens1 should be evicted, tokens2 and tokens3 should remain
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens1 should be evicted (oldest)"
        );

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens2 should remain"
        );

        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens3 should remain"
        );
    }

    #[test]
    fn test_eviction_policy_mru() {
        // MRU: Evict newest by last access time (opposite of LRU)
        let tree = TokenTree::with_policy(EvictionPolicy::Mru);

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1"); // Oldest
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1"); // Newest

        // Evict 1 page - should evict tokens3 (newest by last_access_time)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens3 should be evicted (newest), tokens1 and tokens2 should remain
        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens3 should be evicted (newest)"
        );

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens1 should remain"
        );
    }

    #[test]
    fn test_eviction_policy_fifo() {
        // FIFO: Evict oldest by creation time (not affected by access)
        let tree = TokenTree::with_policy(EvictionPolicy::Fifo);

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1"); // First created
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1"); // Last created

        // Access tokens1 multiple times (shouldn't affect FIFO)
        for _ in 0..10 {
            let _ = tree.match_prefix_with_counts(&tokens1);
        }

        // Evict 1 page - should evict tokens1 (oldest by creation_time)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens1 should be evicted despite being accessed most recently
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens1 should be evicted (oldest creation)"
        );
    }

    #[test]
    fn test_eviction_policy_filo() {
        // FILO: Evict newest by creation time (stack-like)
        let tree = TokenTree::with_policy(EvictionPolicy::Filo);

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1"); // First created
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1"); // Last created

        // Evict 1 page - should evict tokens3 (newest by creation_time)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens3 should be evicted (newest creation)
        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens3 should be evicted (newest creation)"
        );

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens1 should remain"
        );
    }

    #[test]
    fn test_eviction_policy_lfu() {
        // LFU: Evict least frequently used (by hit_count)
        let tree = TokenTree::with_policy(EvictionPolicy::Lfu);

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1");

        // Access tokens1 many times to increase its hit_count
        for _ in 0..20 {
            let _ = tree.match_prefix_with_counts(&tokens1);
        }

        // Access tokens3 a few times
        for _ in 0..5 {
            let _ = tree.match_prefix_with_counts(&tokens3);
        }

        // tokens2 has the lowest hit_count (only 1 from insert)

        // Evict 1 page - should evict tokens2 (lowest hit_count)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens2 should be evicted (least frequently used)
        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens2 should be evicted (lowest hit count)"
        );

        // tokens1 should remain (highest hit count)
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens1 should remain (high hit count)"
        );
    }

    #[test]
    fn test_eviction_policy_enum_default() {
        // Default should be LRU
        assert_eq!(EvictionPolicy::default(), EvictionPolicy::Lru);
    }

    #[test]
    fn test_tree_with_policy_getter() {
        let tree_lru = TokenTree::with_policy(EvictionPolicy::Lru);
        assert_eq!(tree_lru.eviction_policy(), EvictionPolicy::Lru);

        let tree_lfu = TokenTree::with_policy(EvictionPolicy::Lfu);
        assert_eq!(tree_lfu.eviction_policy(), EvictionPolicy::Lfu);

        let tree_fifo = TokenTree::with_policy(EvictionPolicy::Fifo);
        assert_eq!(tree_fifo.eviction_policy(), EvictionPolicy::Fifo);
    }

    #[test]
    fn test_tree_with_config() {
        // Test with_config constructor with valid page_size
        let tree = TokenTree::with_config(PAGE_SIZE, EvictionPolicy::Lfu);
        assert_eq!(tree.page_size(), PAGE_SIZE);
        assert_eq!(tree.eviction_policy(), EvictionPolicy::Lfu);

        // Verify defaults: new() should use PAGE_SIZE=16 and LRU
        let tree_default = TokenTree::new();
        assert_eq!(tree_default.page_size(), 16);
        assert_eq!(tree_default.eviction_policy(), EvictionPolicy::Lru);

        // with_policy should use default page_size
        let tree_policy = TokenTree::with_policy(EvictionPolicy::Fifo);
        assert_eq!(tree_policy.page_size(), PAGE_SIZE);
        assert_eq!(tree_policy.eviction_policy(), EvictionPolicy::Fifo);
    }

    #[test]
    #[should_panic(expected = "TokenTree currently only supports page_size=16")]
    fn test_tree_with_config_invalid_page_size() {
        // Test that invalid page_size panics
        let _tree = TokenTree::with_config(32, EvictionPolicy::Lru);
    }
}
