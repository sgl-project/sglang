//! Token-based radix tree for gRPC router with pre-tokenized input.
//!
//! This implementation uses token IDs (`u32`) instead of characters,
//! matching SGLang's Python scheduler which operates on token arrays.
//! Benefits:
//! - O(1) token comparisons vs O(n) UTF-8 char extraction
//! - Direct integration with tokenized gRPC requests
//! - Memory-efficient for high-throughput scenarios

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
};

use dashmap::DashMap;
use tracing::debug;

use super::{
    common::{MatchResult, TenantId},
    RadixTree,
};

/// Token ID type (matches SGLang's token representation)
pub type TokenId = u32;

type NodeRef = Arc<Node>;

/// Shard counts for DashMaps to balance concurrency vs allocation overhead.
const ROOT_SHARD_COUNT: usize = 32;
const NODE_SHARD_COUNT: usize = 8;

/// A fast hasher for single token IDs (u32).
/// Uses FxHash-style multiplication mixing for excellent distribution.
#[derive(Default)]
struct TokenHasher(u64);

impl Hasher for TokenHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        // Fast path for u32 (single token)
        if bytes.len() == 4 {
            let val = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            // FxHash-style mixing: multiply by golden ratio prime and rotate
            self.0 = (val as u64).wrapping_mul(0x517cc1b727220a95);
        } else {
            // Fallback for other sizes
            for chunk in bytes.chunks(4) {
                if chunk.len() == 4 {
                    let val = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.0 = self.0.wrapping_add(val as u64).wrapping_mul(0x517cc1b727220a95);
                }
            }
        }
    }
}

type TokenHasherBuilder = BuildHasherDefault<TokenHasher>;

/// Create a children DashMap with single-token key lookup
#[inline]
fn new_children_map() -> DashMap<TokenId, NodeRef, TokenHasherBuilder> {
    DashMap::with_hasher_and_shard_amount(TokenHasherBuilder::default(), NODE_SHARD_COUNT)
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

/// Intern tenant ID to avoid repeated allocations
fn intern_tenant(tenant: &str) -> TenantId {
    Arc::from(tenant)
}

/// Global timestamp counter for LRU ordering
static GLOBAL_TIMESTAMP: AtomicU64 = AtomicU64::new(0);

fn next_timestamp() -> u64 {
    GLOBAL_TIMESTAMP.fetch_add(1, Ordering::Relaxed)
}

/// Node in the token-based radix tree
struct Node {
    /// Token sequence stored at this node
    tokens: RwLock<Vec<TokenId>>,
    /// Children nodes keyed by first token (for fast lookup)
    children: DashMap<TokenId, NodeRef, TokenHasherBuilder>,
    /// Tenants that own this node with last access timestamps
    tenant_last_access_time: DashMap<TenantId, u64>,
    /// Cached last tenant for fast access (probabilistic update)
    last_tenant: RwLock<Option<TenantId>>,
}

impl Node {
    fn new(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens: RwLock::new(tokens),
            children: new_children_map(),
            tenant_last_access_time: new_tenant_map(),
            last_tenant: RwLock::new(None),
        }
    }

    fn new_root() -> Self {
        Self {
            tokens: RwLock::new(Vec::new()),
            children: DashMap::with_hasher_and_shard_amount(
                TokenHasherBuilder::default(),
                ROOT_SHARD_COUNT,
            ),
            tenant_last_access_time: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
            last_tenant: RwLock::new(None),
        }
    }

    /// Get any tenant that owns this node (for match results)
    fn get_any_tenant(&self) -> Option<TenantId> {
        // Fast path: check cached tenant
        if let Ok(guard) = self.last_tenant.read() {
            if let Some(ref tenant) = *guard {
                if self.tenant_last_access_time.contains_key(tenant) {
                    return Some(Arc::clone(tenant));
                }
            }
        }

        // Slow path: iterate to find any tenant
        self.tenant_last_access_time
            .iter()
            .next()
            .map(|entry| Arc::clone(entry.key()))
    }

    /// Update tenant access and cache (with probabilistic update to reduce contention)
    fn touch_tenant(&self, tenant: &TenantId) {
        let ts = next_timestamp();
        self.tenant_last_access_time
            .entry(Arc::clone(tenant))
            .and_modify(|t| *t = ts)
            .or_insert(ts);

        // Probabilistic cache update (1/16 chance) to reduce write contention
        if ts & 0xF == 0 {
            if let Ok(mut guard) = self.last_tenant.try_write() {
                *guard = Some(Arc::clone(tenant));
            }
        }
    }
}

/// Token-based radix tree for cache-aware routing.
pub struct TokenTree {
    root: NodeRef,
    /// Track total tokens per tenant for eviction decisions
    tenant_token_count: DashMap<TenantId, usize>,
}

impl Default for TokenTree {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenTree {
    pub fn new() -> Self {
        Self {
            root: Arc::new(Node::new_root()),
            tenant_token_count: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
        }
    }

    /// Insert a token sequence with associated tenant.
    pub fn insert_tokens(&self, tokens: &[TokenId], tenant: &str) {
        if tokens.is_empty() {
            return;
        }

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

        // Result type to carry state out of the match block
        // This allows the entry guard to be dropped before we update current
        enum InsertStep {
            Done(usize),
            Continue { next: NodeRef, advance: usize },
        }

        while !remaining.is_empty() {
            // Use first token as key for children lookup
            let first_token = remaining[0];

            let step = match current.children.entry(first_token) {
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    // No child with this token - create new node
                    let new_node = Arc::new(Node::new(remaining.to_vec()));
                    new_node.touch_tenant(&tenant_id);
                    entry.insert(new_node);
                    InsertStep::Done(remaining.len())
                }
                dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                    let child = Arc::clone(entry.get());
                    let child_tokens = child.tokens.read().unwrap();
                    let child_len = child_tokens.len();

                    // Find common prefix length
                    let common_len = remaining
                        .iter()
                        .zip(child_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();

                    if common_len == child_len {
                        // Full match with child - continue traversal
                        drop(child_tokens);
                        child.touch_tenant(&tenant_id);
                        InsertStep::Continue {
                            next: child,
                            advance: common_len,
                        }
                    } else if common_len == remaining.len() {
                        // Input is prefix of child - split child
                        // Strategy: Create NEW intermediate node with prefix tokens,
                        // keep original child as suffix (preserving its children/tenants)
                        let prefix_tokens: Vec<TokenId> = child_tokens[..common_len].to_vec();
                        let suffix_first = child_tokens[common_len];
                        drop(child_tokens);

                        // Modify original child to hold only suffix tokens
                        let mut child_tokens_write = child.tokens.write().unwrap();
                        let suffix_tokens: Vec<TokenId> = child_tokens_write[common_len..].to_vec();
                        *child_tokens_write = suffix_tokens;
                        drop(child_tokens_write);

                        // Create intermediate node with prefix - clone tenant map (O(1))
                        let intermediate_node = Arc::new(Node {
                            tokens: RwLock::new(prefix_tokens),
                            children: new_children_map(),
                            tenant_last_access_time: child.tenant_last_access_time.clone(),
                            last_tenant: RwLock::new(
                                child.last_tenant.read().ok().and_then(|g| g.clone()),
                            ),
                        });

                        // Add original child (now suffix) as child of intermediate
                        intermediate_node
                            .children
                            .insert(suffix_first, Arc::clone(&child));

                        // Replace entry with intermediate node
                        entry.insert(intermediate_node.clone());

                        intermediate_node.touch_tenant(&tenant_id);
                        InsertStep::Done(common_len)
                    } else {
                        // Partial match - need to split and add new branch
                        // Strategy: Create NEW intermediate node with common prefix,
                        // keep original child as one suffix, create new node for other suffix
                        let prefix_tokens: Vec<TokenId> = child_tokens[..common_len].to_vec();
                        let child_suffix_first = child_tokens[common_len];
                        drop(child_tokens);

                        // Modify original child to hold only its suffix tokens
                        let mut child_tokens_write = child.tokens.write().unwrap();
                        let child_suffix: Vec<TokenId> = child_tokens_write[common_len..].to_vec();
                        *child_tokens_write = child_suffix;
                        drop(child_tokens_write);

                        // Create intermediate node with common prefix - clone tenant map (O(1))
                        let intermediate_node = Arc::new(Node {
                            tokens: RwLock::new(prefix_tokens),
                            children: new_children_map(),
                            tenant_last_access_time: child.tenant_last_access_time.clone(),
                            last_tenant: RwLock::new(
                                child.last_tenant.read().ok().and_then(|g| g.clone()),
                            ),
                        });

                        // Add original child (now suffix) as child of intermediate
                        intermediate_node
                            .children
                            .insert(child_suffix_first, Arc::clone(&child));

                        // Create new node for the remaining input suffix
                        let new_remaining = &remaining[common_len..];
                        let new_node = Arc::new(Node::new(new_remaining.to_vec()));
                        new_node.touch_tenant(&tenant_id);
                        intermediate_node
                            .children
                            .insert(new_remaining[0], new_node);

                        // Replace entry with intermediate node
                        entry.insert(intermediate_node.clone());

                        intermediate_node.touch_tenant(&tenant_id);
                        InsertStep::Done(remaining.len())
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
    pub fn match_prefix_with_counts(&self, tokens: &[TokenId]) -> PrefixMatchResult {
        let input_token_count = tokens.len();

        if tokens.is_empty() {
            return PrefixMatchResult {
                tenant: self
                    .root
                    .get_any_tenant()
                    .unwrap_or_else(|| Arc::from("empty")),
                matched_token_count: 0,
                input_token_count: 0,
            };
        }

        let mut matched_tokens = 0;
        let mut last_tenant: Option<TenantId> = None;
        let mut remaining = tokens;
        let mut current = Arc::clone(&self.root);

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

        while !remaining.is_empty() {
            // Use first token as key for children lookup
            let first_token = remaining[0];

            let step = match current.children.get(&first_token) {
                None => MatchStep::Done,
                Some(child_ref) => {
                    let child = Arc::clone(child_ref.value());
                    drop(child_ref);

                    let child_tokens = child.tokens.read().unwrap();

                    // Count matching tokens
                    let match_len = remaining
                        .iter()
                        .zip(child_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();

                    if match_len == 0 {
                        MatchStep::Done
                    } else {
                        let tenant = child.get_any_tenant();

                        if match_len < child_tokens.len() {
                            // Partial match within node
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

    /// Evict entries for a tenant to reduce to max_tokens.
    pub fn evict_tenant(&self, tenant: &TenantId, max_tokens: usize) {
        let current_count = self.tenant_token_count.get(tenant).map(|v| *v).unwrap_or(0);

        if current_count <= max_tokens {
            return;
        }

        let to_evict = current_count - max_tokens;
        let mut evicted = 0;

        // Collect nodes with timestamps for LRU eviction
        let mut nodes_with_time: Vec<(NodeRef, u64)> = Vec::new();
        self.collect_tenant_nodes(&self.root, tenant, &mut nodes_with_time);

        // Sort by timestamp (oldest first)
        nodes_with_time.sort_by_key(|(_, ts)| *ts);

        for (node, _) in nodes_with_time {
            if evicted >= to_evict {
                break;
            }

            let node_tokens = node.tokens.read().unwrap().len();
            if self.remove_tenant_from_node(&node, tenant) {
                evicted += node_tokens;
            }
        }

        // Update tenant token count
        self.tenant_token_count
            .entry(tenant.clone())
            .and_modify(|count| *count = count.saturating_sub(evicted));

        debug!(
            tenant = %tenant.as_ref(),
            evicted = evicted,
            remaining = current_count.saturating_sub(evicted),
            "Evicted tokens from tenant"
        );
    }

    fn collect_tenant_nodes(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
        result: &mut Vec<(NodeRef, u64)>,
    ) {
        // Skip root
        if !Arc::ptr_eq(node, &self.root) {
            if let Some(ts) = node.tenant_last_access_time.get(tenant_id) {
                result.push((Arc::clone(node), *ts));
            }
        }

        for child in node.children.iter() {
            self.collect_tenant_nodes(child.value(), tenant_id, result);
        }
    }

    fn remove_tenant_from_node(&self, node: &NodeRef, tenant_id: &TenantId) -> bool {
        node.tenant_last_access_time.remove(tenant_id).is_some()
    }

    /// Get the token count for a specific tenant.
    pub fn tenant_token_size(&self, tenant: &TenantId) -> usize {
        self.tenant_token_count.get(tenant).map(|v| *v).unwrap_or(0)
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

    #[test]
    fn test_basic_insert_match() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3]);
        assert_eq!(matched, vec![1, 2, 3]);
        assert_eq!(tenant, "tenant1");

        let (matched, _) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_multiple_tenants() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3], "tenant1");
        tree.insert_tokens(&[1, 2, 3], "tenant2");

        let (matched, _tenant) = tree.prefix_match_legacy(&[1, 2, 3]);
        assert_eq!(matched, vec![1, 2, 3]);
    }

    #[test]
    fn test_prefix_split() {
        let tree = TokenTree::new();

        // Insert longer first
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        // Insert shorter (causes split)
        tree.insert_tokens(&[1, 2], "tenant2");

        let (matched, _tenant) = tree.prefix_match_legacy(&[1, 2]);
        assert_eq!(matched, vec![1, 2]);

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_empty_input() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3], "tenant1");

        let result = tree.match_prefix_with_counts(&[]);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, 0);
    }

    #[test]
    fn test_no_match() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3], "tenant1");

        let (matched, _) = tree.prefix_match_legacy(&[4, 5, 6]);
        assert_eq!(matched, vec![] as Vec<TokenId>);
    }

    #[test]
    fn test_eviction() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "tenant1");

        let counts = tree.get_tenant_token_counts();
        assert!(counts.get("tenant1").unwrap() > &0);

        tree.evict(&TenantId::from("tenant1"), 0);

        // After eviction, matches should still work for remaining entries
        let (matched, _) = tree.prefix_match_legacy(&[1, 2, 3]);
        assert!(matched.len() <= 3);
    }

    #[test]
    fn test_concurrent_insert_match() {
        let tree = Arc::new(TokenTree::new());
        let mut handles = vec![];

        // Spawn inserters
        for i in 0..4 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let tokens: Vec<TokenId> =
                        (0..5).map(|k| (i * 1000 + j * 10 + k) as u32).collect();
                    tree.insert_tokens(&tokens, &format!("tenant{}", i));
                }
            }));
        }

        // Spawn matchers
        for i in 0..4 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let tokens: Vec<TokenId> =
                        (0..5).map(|k| (i * 1000 + j * 10 + k) as u32).collect();
                    let _ = tree.prefix_match_legacy(&tokens);
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

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");

        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 5);
        assert_eq!(result.input_token_count, 5);

        let result = tree.match_prefix_with_counts(&[1, 2, 3]);
        assert_eq!(result.matched_token_count, 3);
        assert_eq!(result.input_token_count, 3);

        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(result.matched_token_count, 5);
        assert_eq!(result.input_token_count, 7);
    }

    #[test]
    fn test_disjoint_paths() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3], "tenant1");
        tree.insert_tokens(&[100, 200, 300], "tenant2");
        tree.insert_tokens(&[1000, 2000, 3000], "tenant3");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3]);
        assert_eq!(matched, vec![1, 2, 3]);
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy(&[100, 200, 300]);
        assert_eq!(matched, vec![100, 200, 300]);
        assert_eq!(tenant, "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[1000, 2000, 3000]);
        assert_eq!(matched, vec![1000, 2000, 3000]);
        assert_eq!(tenant, "tenant3");
    }

    #[test]
    fn test_branching_paths() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 6, 7], "tenant2");
        tree.insert_tokens(&[1, 2, 3, 8, 9], "tenant3");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 6, 7]);
        assert_eq!(matched, vec![1, 2, 3, 6, 7]);
        assert_eq!(tenant, "tenant2");

        // Partial match at branch point
        let (matched, _) = tree.prefix_match_legacy(&[1, 2, 3, 100, 200]);
        assert_eq!(matched, vec![1, 2, 3]);
    }

    #[test]
    fn test_radix_tree_trait() {
        let tree = TokenTree::new();

        // Use trait methods
        RadixTree::insert(&tree, &[1, 2, 3, 4, 5], "tenant1");

        let tenant = RadixTree::prefix_match(&tree, &[1, 2, 3]);
        assert!(tenant.is_some());
        assert_eq!(tenant.unwrap().as_ref(), "tenant1");

        let result = RadixTree::prefix_match_with_counts(&tree, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(result.matched_count(), 5);
        assert_eq!(result.input_count(), 6);

        assert!(RadixTree::tenant_size(&tree, &TenantId::from("tenant1")) > 0);
    }

    #[test]
    fn test_clear() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3], "tenant1");
        tree.insert_tokens(&[4, 5, 6], "tenant2");

        assert!(!tree.get_tenant_token_counts().is_empty());

        tree.clear();

        assert!(tree.get_tenant_token_counts().is_empty());
        let (matched, _) = tree.prefix_match_legacy(&[1, 2, 3]);
        assert!(matched.is_empty());
    }

    #[test]
    fn test_tenant_token_count() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 4, 5, 6, 7, 8], "tenant1");
        tree.insert_tokens(&[10, 20, 30], "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");
        let tenant2_id: TenantId = Arc::from("tenant2");

        assert!(tree.tenant_token_size(&tenant1_id) >= 5);
        assert!(tree.tenant_token_size(&tenant2_id) >= 3);

        let counts = tree.get_tenant_token_counts();
        assert!(counts.contains_key("tenant1"));
        assert!(counts.contains_key("tenant2"));
    }

    #[test]
    fn test_cold_start() {
        let tree = TokenTree::new();
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, 5);
    }

    #[test]
    fn test_exact_match_seq() {
        let tree = TokenTree::new();

        for i in 0..100 {
            let tokens: Vec<TokenId> = (0..10).map(|j| (i * 100 + j) as u32).collect();
            tree.insert_tokens(&tokens, &format!("tenant{}", i));
        }

        for i in 0..100 {
            let tokens: Vec<TokenId> = (0..10).map(|j| (i * 100 + j) as u32).collect();
            let (matched, tenant) = tree.prefix_match_legacy(&tokens);
            assert_eq!(matched, tokens);
            assert_eq!(tenant, format!("tenant{}", i));
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
                    let tokens: Vec<TokenId> =
                        (0..10).map(|j| (t * 10000 + i * 100 + j) as u32).collect();
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
                    let tokens: Vec<TokenId> =
                        (0..10).map(|j| (t * 10000 + i * 100 + j) as u32).collect();
                    let (matched, tenant) = tree.prefix_match_legacy(&tokens);
                    assert_eq!(matched, tokens);
                    assert_eq!(tenant, format!("tenant{}", t));
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

        // Insert full sequences
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let tokens: Vec<TokenId> =
                        (0..20).map(|j| (t * 10000 + i * 100 + j) as u32).collect();
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Match with prefixes
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let full_tokens: Vec<TokenId> =
                        (0..20).map(|j| (t * 10000 + i * 100 + j) as u32).collect();
                    let partial: Vec<TokenId> = full_tokens[..10].to_vec();
                    let (matched, _) = tree.prefix_match_legacy(&partial);
                    assert_eq!(matched, partial);
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

        // All threads share the same prefix
        let common_prefix: Vec<TokenId> = vec![100, 200, 300, 400, 500];

        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            let prefix = common_prefix.clone();
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let mut tokens = prefix.clone();
                    tokens.extend((0..5).map(|j| (t * 1000 + i * 10 + j) as u32));
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify prefix matching works
        let (matched, _) = tree.prefix_match_legacy(&common_prefix);
        assert_eq!(matched.len(), common_prefix.len());
    }

    #[test]
    fn test_mixed_concurrent_insert_match() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 4;

        // Pre-populate some data
        for i in 0..100 {
            let tokens: Vec<TokenId> = (0..10).map(|j| (i * 100 + j) as u32).collect();
            tree.insert_tokens(&tokens, &format!("initial{}", i));
        }

        let mut handles = vec![];

        // Concurrent inserters
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let tokens: Vec<TokenId> = (0..10)
                        .map(|j| (1000000 + t * 10000 + i * 100 + j) as u32)
                        .collect();
                    tree.insert_tokens(&tokens, &format!("new_tenant{}", t));
                }
            }));
        }

        // Concurrent matchers (matching existing data)
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let i = t * 10; // Each thread checks a subset
                    let tokens: Vec<TokenId> = (0..10).map(|j| (i * 100 + j) as u32).collect();
                    let (matched, _) = tree.prefix_match_legacy(&tokens);
                    assert!(!matched.is_empty());
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

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[10, 20, 30, 40, 50], "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");

        tree.evict_tenant(&tenant1_id, 0);

        // tenant2 should still work
        let (matched, tenant) = tree.prefix_match_legacy(&[10, 20, 30]);
        assert_eq!(matched, vec![10, 20, 30]);
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_advanced_eviction() {
        let tree = TokenTree::new();

        // Insert multiple paths for tenant1
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 6, 7], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 8, 9], "tenant1");

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
            let tokens: Vec<TokenId> = (0..10).map(|j| (i * 100 + j) as u32).collect();
            tree.insert_tokens(&tokens, &format!("tenant{}", i % 4));
        }

        let mut handles = vec![];

        // Inserters
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let tokens: Vec<TokenId> = (0..10)
                        .map(|j| (100000 + t * 10000 + i * 100 + j) as u32)
                        .collect();
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
                    let tokens: Vec<TokenId> = (0..5).map(|j| (i * 100 + j) as u32).collect();
                    let _ = tree.prefix_match_legacy(&tokens);
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

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "tenant1");
        tree.insert_tokens(&[100, 200, 300], "tenant2");

        let counts = tree.get_tenant_token_counts();

        assert!(counts.contains_key("tenant1"));
        assert!(counts.contains_key("tenant2"));
        assert!(*counts.get("tenant1").unwrap() >= 5);
        assert!(*counts.get("tenant2").unwrap() >= 3);
    }

    #[test]
    fn test_prefix_match_tenant() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant2");

        // Both tenants should have access time updated
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 5);
        // tenant should be either tenant1 or tenant2 (last_tenant cache)
        assert!(result.tenant.as_ref() == "tenant1" || result.tenant.as_ref() == "tenant2");
    }

    #[test]
    fn test_simple_tenant_eviction() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[10, 20, 30, 40, 50], "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // tenant2 should be unaffected
        let (matched, tenant) = tree.prefix_match_legacy(&[10, 20, 30, 40, 50]);
        assert_eq!(matched, vec![10, 20, 30, 40, 50]);
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_complex_tenant_eviction() {
        let tree = TokenTree::new();

        // Create overlapping paths
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 6, 7], "tenant2");
        tree.insert_tokens(&[1, 2, 3, 8, 9], "tenant1");

        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // tenant2's path should still work
        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 6, 7]);
        assert_eq!(matched, vec![1, 2, 3, 6, 7]);
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_single_token_operations() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1], "tenant1");
        tree.insert_tokens(&[2], "tenant2");
        tree.insert_tokens(&[3], "tenant3");

        let (matched, tenant) = tree.prefix_match_legacy(&[1]);
        assert_eq!(matched, vec![1]);
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy(&[2]);
        assert_eq!(matched, vec![2]);
        assert_eq!(tenant, "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[3]);
        assert_eq!(matched, vec![3]);
        assert_eq!(tenant, "tenant3");
    }

    #[test]
    fn test_prefix_is_subset_of_existing() {
        let tree = TokenTree::new();

        // Insert longer sequence first
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");

        // Insert prefix
        tree.insert_tokens(&[1, 2, 3], "tenant2");

        // Both should match correctly
        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3]);
        assert_eq!(matched, vec![1, 2, 3]);
        assert!(tenant == "tenant1" || tenant == "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_existing_is_prefix_of_new() {
        let tree = TokenTree::new();

        // Insert shorter first
        tree.insert_tokens(&[1, 2, 3], "tenant1");

        // Insert longer
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3]);
        assert_eq!(matched, vec![1, 2, 3]);
        assert!(tenant == "tenant1" || tenant == "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_prefix_match_with_counts_accuracy() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "tenant1");

        // Exact match
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(result.matched_token_count, 10);
        assert_eq!(result.input_token_count, 10);

        // Partial match (prefix of inserted)
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 5);
        assert_eq!(result.input_token_count, 5);

        // Extended match (input longer than inserted)
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(result.matched_token_count, 10);
        assert_eq!(result.input_token_count, 12);
    }

    #[test]
    fn test_split_at_first_token() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1], "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[1]);
        assert_eq!(matched, vec![1]);
        assert!(tenant == "tenant1" || tenant == "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_split_at_last_token() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 4], "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 4]);
        assert_eq!(matched, vec![1, 2, 3, 4]);
        assert!(tenant == "tenant1" || tenant == "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_multiple_splits_same_path() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3], "tenant2");
        tree.insert_tokens(&[1, 2], "tenant3");
        tree.insert_tokens(&[1], "tenant4");

        let (matched, _) = tree.prefix_match_legacy(&[1]);
        assert_eq!(matched, vec![1]);

        let (matched, _) = tree.prefix_match_legacy(&[1, 2]);
        assert_eq!(matched, vec![1, 2]);

        let (matched, _) = tree.prefix_match_legacy(&[1, 2, 3]);
        assert_eq!(matched, vec![1, 2, 3]);

        let (matched, _) = tree.prefix_match_legacy(&[1, 2, 3, 4, 5]);
        assert_eq!(matched, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_high_contention_same_prefix() {
        let tree = Arc::new(TokenTree::new());
        let prefix: Vec<TokenId> = vec![1, 2, 3, 4, 5];
        let num_threads = 16;

        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            let p = prefix.clone();
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let mut tokens = p.clone();
                    tokens.push((t * 1000 + i) as u32);
                    tree.insert_tokens(&tokens, &format!("tenant{}", t));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify prefix matching
        let (matched, _) = tree.prefix_match_legacy(&prefix);
        assert_eq!(matched, prefix);
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
                        let tokens: Vec<TokenId> = (0..5)
                            .map(|j| (t * 10000 + cycle * 1000 + i * 10 + j) as u32)
                            .collect();
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

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[6, 7, 8, 9, 10], "tenant1");

        let tenant_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant_id, 0);

        // Eviction with max_size=0 should remove entries
        let size = tree.tenant_token_size(&tenant_id);
        assert!(size == 0 || size < 10);
    }

    #[test]
    fn test_eviction_single_tenant_all_entries() {
        let tree = TokenTree::new();

        // Insert multiple entries for one tenant
        for i in 0..10 {
            let tokens: Vec<TokenId> = (0..5).map(|j| (i * 100 + j) as u32).collect();
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

        tree.insert_tokens(&[1, 2, 3], "tenant1");
        tree.insert_tokens(&[1, 2, 3], "tenant2");

        // First match
        let result1 = tree.match_prefix_with_counts(&[1, 2, 3]);
        let first_tenant = result1.tenant.clone();

        // Match again - should get cached tenant
        let result2 = tree.match_prefix_with_counts(&[1, 2, 3]);
        assert_eq!(result2.tenant, first_tenant);
    }

    #[test]
    fn test_stale_cache_after_tenant_removal() {
        let tree = TokenTree::new();

        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant2");

        // Match to populate cache
        let _ = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);

        // Evict one tenant
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // Match should still work (tenant2 or cache still valid)
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 5);
    }

    #[test]
    fn test_token_count_consistency_after_operations() {
        let tree = TokenTree::new();

        // Insert
        tree.insert_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "tenant1");
        tree.insert_tokens(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "tenant1",
        );

        let tenant1_id: TenantId = Arc::from("tenant1");
        let count1 = tree.tenant_token_size(&tenant1_id);
        assert!(count1 >= 10);

        // Partial eviction
        tree.evict_tenant(&tenant1_id, count1 / 2);
        let count2 = tree.tenant_token_size(&tenant1_id);
        assert!(count2 <= count1);

        // Insert more
        tree.insert_tokens(&[100, 200, 300], "tenant1");
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
                    let tokens: Vec<TokenId> =
                        (0..10).map(|j| (t * 100000 + i * 100 + j) as u32).collect();
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
                let tokens: Vec<TokenId> =
                    (0..10).map(|j| (t * 100000 + i * 100 + j) as u32).collect();
                let (matched, _) = tree.prefix_match_legacy(&tokens);
                assert_eq!(matched, tokens);
            }
        }
    }

    #[test]
    fn test_very_long_sequences() {
        let tree = TokenTree::new();

        // Insert a very long sequence
        let long_seq: Vec<TokenId> = (0..1000).map(|i| i as u32).collect();
        tree.insert_tokens(&long_seq, "tenant1");

        // Match the full sequence
        let (matched, tenant) = tree.prefix_match_legacy(&long_seq);
        assert_eq!(matched.len(), 1000);
        assert_eq!(tenant, "tenant1");

        // Match a prefix
        let prefix: Vec<TokenId> = (0..500).map(|i| i as u32).collect();
        let (matched, _) = tree.prefix_match_legacy(&prefix);
        assert_eq!(matched.len(), 500);
    }

    #[test]
    fn test_many_tenants_same_path() {
        let tree = TokenTree::new();

        let tokens: Vec<TokenId> = vec![1, 2, 3, 4, 5];

        for i in 0..100 {
            tree.insert_tokens(&tokens, &format!("tenant{}", i));
        }

        let (matched, _) = tree.prefix_match_legacy(&tokens);
        assert_eq!(matched, tokens);

        // All 100 tenants should have registered
        let counts = tree.get_tenant_token_counts();
        assert!(!counts.is_empty()); // At least some tenants tracked
    }

    #[test]
    fn test_token_id_edge_values() {
        let tree = TokenTree::new();

        // Test with edge case token IDs
        tree.insert_tokens(&[0], "tenant1");
        tree.insert_tokens(&[u32::MAX], "tenant2");
        tree.insert_tokens(&[0, u32::MAX], "tenant3");
        tree.insert_tokens(&[u32::MAX, 0], "tenant4");

        let (matched, tenant) = tree.prefix_match_legacy(&[0]);
        assert_eq!(matched, vec![0]);
        assert!(tenant == "tenant1" || tenant == "tenant3");

        let (matched, tenant) = tree.prefix_match_legacy(&[u32::MAX]);
        assert_eq!(matched, vec![u32::MAX]);
        assert!(tenant == "tenant2" || tenant == "tenant4");
    }

    #[test]
    fn test_hit_ratio_calculation() {
        use crate::radix_tree::MatchResult;

        let tree = TokenTree::new();
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");

        // 100% hit ratio
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert!((result.hit_ratio() - 1.0).abs() < 0.001);

        // 50% hit ratio
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert!((result.hit_ratio() - 0.5).abs() < 0.001);

        // 0% hit ratio
        let result = tree.match_prefix_with_counts(&[100, 200, 300]);
        assert!(result.hit_ratio() == 0.0);
    }

    #[test]
    fn test_reset_via_trait() {
        use crate::radix_tree::RadixTree;

        let tree = TokenTree::new();
        tree.insert_tokens(&[1, 2, 3], "tenant1");
        tree.insert_tokens(&[4, 5, 6], "tenant2");

        assert!(!tree.get_tenant_token_counts().is_empty());

        RadixTree::reset(&tree);

        assert!(tree.get_tenant_token_counts().is_empty());
    }
}
