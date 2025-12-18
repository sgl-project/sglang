use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, VecDeque},
    hash::{BuildHasherDefault, Hasher},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use dashmap::{mapref::entry::Entry, DashMap};
use tracing::info;

type NodeRef = Arc<Node>;

/// Interned tenant ID to avoid repeated string allocations.
/// Using Arc<str> allows cheap cloning and comparison.
pub type TenantId = Arc<str>;

/// A fast identity hasher for single-character keys (used in children DashMap).
/// Since chars have good distribution already, we use identity hashing with mixing.
#[derive(Default)]
struct CharHasher(u64);

impl Hasher for CharHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        // Fast path for 4-byte (char) writes - avoid loop
        if bytes.len() == 4 {
            let val = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            // Mix with golden ratio for better distribution
            self.0 = (val as u64).wrapping_mul(0x9E3779B97F4A7C15);
            return;
        }
        // Fallback for other sizes (shouldn't happen for char keys)
        for &byte in bytes {
            self.0 = self.0.wrapping_mul(0x100000001b3).wrapping_add(byte as u64);
        }
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        // Chars are u32 - use golden ratio multiplication for distribution
        self.0 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
}

type CharHasherBuilder = BuildHasherDefault<CharHasher>;

/// Pre-indexed text for efficient character access.
/// Converts UTF-8 string to Vec<char> once to enable O(1) indexing.
struct CharIndexedText {
    chars: Vec<char>,
}

impl CharIndexedText {
    #[inline]
    fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.chars.len()
    }

    #[inline]
    fn get(&self, idx: usize) -> Option<char> {
        self.chars.get(idx).copied()
    }

    #[inline]
    fn slice_to_string(&self, start: usize, end: usize) -> String {
        self.chars[start..end].iter().collect()
    }
}

/// Node text with cached character count to avoid repeated O(n) chars().count() calls.
#[derive(Debug)]
struct NodeText {
    /// The actual text stored in this node
    text: String,
    /// Cached character count (UTF-8 chars, not bytes)
    char_count: usize,
}

impl NodeText {
    #[inline]
    fn new(text: String) -> Self {
        let char_count = text.chars().count();
        Self { text, char_count }
    }

    #[inline]
    fn empty() -> Self {
        Self {
            text: String::new(),
            char_count: 0,
        }
    }

    #[inline]
    fn char_count(&self) -> usize {
        self.char_count
    }

    #[inline]
    fn as_str(&self) -> &str {
        &self.text
    }

    #[inline]
    fn first_char(&self) -> Option<char> {
        self.text.chars().next()
    }

    /// Split the text at a character boundary, returning the prefix and suffix.
    /// This is more efficient than slice_by_chars as it computes both at once.
    #[inline]
    fn split_at_char(&self, char_idx: usize) -> (NodeText, NodeText) {
        if char_idx == 0 {
            return (NodeText::empty(), self.clone_text());
        }
        if char_idx >= self.char_count {
            return (self.clone_text(), NodeText::empty());
        }

        // Find byte index for the character boundary
        let byte_idx = self
            .text
            .char_indices()
            .nth(char_idx)
            .map(|(i, _)| i)
            .unwrap_or(self.text.len());

        let prefix = NodeText {
            text: self.text[..byte_idx].to_string(),
            char_count: char_idx,
        };
        let suffix = NodeText {
            text: self.text[byte_idx..].to_string(),
            char_count: self.char_count - char_idx,
        };
        (prefix, suffix)
    }

    #[inline]
    fn clone_text(&self) -> NodeText {
        NodeText {
            text: self.text.clone(),
            char_count: self.char_count,
        }
    }
}

impl Clone for NodeText {
    fn clone(&self) -> Self {
        self.clone_text()
    }
}

/// Global timestamp that gets updated periodically to reduce syscalls.
/// Uses milliseconds since epoch.
static CURRENT_TIMESTAMP_MS: AtomicU64 = AtomicU64::new(0);

/// Staleness threshold in milliseconds for forced refresh.
/// If cached timestamp is older than this, always get fresh time.
const TIMESTAMP_STALENESS_MS: u64 = 5;

/// Get current timestamp in milliseconds, using cached value when possible.
/// Refreshes if the cached value is stale (>TIMESTAMP_STALENESS_MS).
/// This provides ~99% syscall reduction under high load while maintaining accuracy.
#[inline]
fn get_timestamp_ms() -> u128 {
    let cached = CURRENT_TIMESTAMP_MS.load(Ordering::Relaxed);

    // Always need syscall to check staleness, but it's cheap and necessary for correctness
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Fast path: return cached if still fresh (within TIMESTAMP_STALENESS_MS)
    if cached != 0 && now.saturating_sub(cached) < TIMESTAMP_STALENESS_MS {
        return cached as u128;
    }

    // Update cached value
    CURRENT_TIMESTAMP_MS.store(now, Ordering::Relaxed);
    now as u128
}

#[derive(Debug)]
struct Node {
    /// Children nodes indexed by first character.
    /// Using custom hasher optimized for char keys.
    children: DashMap<char, NodeRef, CharHasherBuilder>,
    /// Node text with cached character count
    text: RwLock<NodeText>,
    /// Per-tenant last access timestamps. Using TenantId (Arc<str>) for cheap cloning.
    tenant_last_access_time: DashMap<TenantId, u128>,
    /// Parent pointer for upward traversal during timestamp updates
    parent: RwLock<Option<NodeRef>>,
}

#[derive(Debug)]
pub struct Tree {
    root: NodeRef,
    /// Per-tenant character count for size tracking. Using TenantId for consistency.
    pub tenant_char_count: DashMap<TenantId, usize>,
}

// For the heap

struct EvictionEntry {
    timestamp: u128,
    tenant: TenantId,
    node: NodeRef,
}

impl Eq for EvictionEntry {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for EvictionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.timestamp.cmp(&other.timestamp))
    }
}

impl Ord for EvictionEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

impl PartialEq for EvictionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
    }
}

// For char operations
// Note that in rust, `.len()` or slice is operated on the "byte" level. It causes issues for UTF-8 characters because one character might use multiple bytes.
// https://en.wikipedia.org/wiki/UTF-8

/// Efficient shared prefix count using pre-indexed chars for O(1) access.
/// Returns the number of characters that match between `a` (starting at `a_start`) and `b`.
#[inline]
fn shared_prefix_count_indexed(a: &CharIndexedText, a_start: usize, b: &str) -> usize {
    let mut i = 0;
    let mut b_iter = b.chars();

    while a_start + i < a.len() {
        match (a.get(a_start + i), b_iter.next()) {
            (Some(a_char), Some(b_char)) if a_char == b_char => {
                i += 1;
            }
            _ => break,
        }
    }

    i
}

/// Intern a tenant string into an Arc<str> for efficient storage and comparison.
#[inline]
fn intern_tenant(tenant: &str) -> TenantId {
    Arc::from(tenant)
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    /*
    Thread-safe multi tenant radix tree

    1. Storing data for multiple tenants (the overlap of multiple radix tree)
    2. Node-level lock to enable concurrent access on nodes
    3. Leaf LRU eviction based on tenant access time

    Optimizations:
    - Cached character counts in NodeText to avoid O(n) chars().count() calls
    - Interned tenant IDs (Arc<str>) for cheap cloning and comparison
    - Batched timestamp updates to reduce syscalls
    - Custom hasher for char keys in children DashMap
    */

    pub fn new() -> Self {
        Tree {
            root: Arc::new(Node {
                children: DashMap::with_hasher(CharHasherBuilder::default()),
                text: RwLock::new(NodeText::empty()),
                tenant_last_access_time: DashMap::new(),
                parent: RwLock::new(None),
            }),
            tenant_char_count: DashMap::new(),
        }
    }

    pub fn insert(&self, text: &str, tenant: &str) {
        // Insert text into tree with given tenant
        // Pre-index text once for O(1) character access (avoids O(n²) chars().nth() calls)
        let indexed_text = CharIndexedText::new(text);
        let text_count = indexed_text.len();

        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        // Use cached timestamp to reduce syscalls
        let timestamp_ms = get_timestamp_ms();

        // Intern the tenant ID once for reuse
        let tenant_id = intern_tenant(tenant);

        curr.tenant_last_access_time
            .insert(Arc::clone(&tenant_id), timestamp_ms);

        self.tenant_char_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        let mut prev = Arc::clone(&self.root);

        while curr_idx < text_count {
            // O(1) character access instead of O(n) chars().nth()
            let first_char = indexed_text.get(curr_idx).unwrap();

            curr = prev;

            // dashmap.entry locks the entry until the op is done
            // if using contains_key + insert, there will be an issue that
            // 1. "apple" and "app" entered at the same time
            // 2. and get inserted to the dashmap concurrently, so only one is inserted

            match curr.children.entry(first_char) {
                Entry::Vacant(entry) => {
                    /*
                       no matched
                       [curr]
                       becomes
                       [curr] => [new node]
                    */

                    // Use indexed slice for efficient string extraction
                    let curr_text = indexed_text.slice_to_string(curr_idx, text_count);
                    let curr_text_count = text_count - curr_idx;
                    let new_node = Arc::new(Node {
                        children: DashMap::with_hasher(CharHasherBuilder::default()),
                        text: RwLock::new(NodeText::new(curr_text)),
                        tenant_last_access_time: DashMap::new(),
                        parent: RwLock::new(Some(Arc::clone(&curr))),
                    });

                    // Attach tenant to the new node (map is empty here) and increment count once
                    self.tenant_char_count
                        .entry(Arc::clone(&tenant_id))
                        .and_modify(|count| *count += curr_text_count)
                        .or_insert(curr_text_count);
                    new_node
                        .tenant_last_access_time
                        .insert(Arc::clone(&tenant_id), timestamp_ms);

                    entry.insert(Arc::clone(&new_node));

                    prev = Arc::clone(&new_node);
                    curr_idx = text_count;
                }

                Entry::Occupied(mut entry) => {
                    // matched
                    let matched_node = entry.get().clone();

                    let matched_node_text = matched_node.text.read().unwrap();
                    // Use cached char count instead of chars().count()
                    let matched_node_text_count = matched_node_text.char_count();

                    // Use indexed comparison to avoid creating intermediate string
                    let shared_count = shared_prefix_count_indexed(
                        &indexed_text,
                        curr_idx,
                        matched_node_text.as_str(),
                    );

                    if shared_count < matched_node_text_count {
                        /*
                           split the matched node
                           [curr] -> [matched_node] =>
                           becomes
                           [curr] -> [new_node] -> [contracted_matched_node]
                        */

                        // Use split_at_char for efficient splitting with cached counts
                        let (matched_text, contracted_text) =
                            matched_node_text.split_at_char(shared_count);
                        let matched_text_count = shared_count;

                        // Drop read lock before creating new node
                        drop(matched_node_text);

                        let new_node = Arc::new(Node {
                            text: RwLock::new(matched_text),
                            children: DashMap::with_hasher(CharHasherBuilder::default()),
                            parent: RwLock::new(Some(Arc::clone(&curr))),
                            tenant_last_access_time: matched_node.tenant_last_access_time.clone(),
                        });

                        let first_new_char = contracted_text.first_char().unwrap();
                        new_node
                            .children
                            .insert(first_new_char, Arc::clone(&matched_node));

                        entry.insert(Arc::clone(&new_node));

                        *matched_node.text.write().unwrap() = contracted_text;
                        *matched_node.parent.write().unwrap() = Some(Arc::clone(&new_node));

                        prev = Arc::clone(&new_node);

                        // Atomically attach tenant to the new split node and increment count once
                        match prev.tenant_last_access_time.entry(Arc::clone(&tenant_id)) {
                            Entry::Vacant(v) => {
                                self.tenant_char_count
                                    .entry(Arc::clone(&tenant_id))
                                    .and_modify(|count| *count += matched_text_count)
                                    .or_insert(matched_text_count);
                                v.insert(timestamp_ms);
                            }
                            Entry::Occupied(mut o) => {
                                o.insert(timestamp_ms);
                            }
                        }

                        curr_idx += shared_count;
                    } else {
                        // move to next node
                        // Drop read lock before continuing
                        drop(matched_node_text);

                        prev = Arc::clone(&matched_node);

                        // Atomically attach tenant to existing node and increment count once
                        match prev.tenant_last_access_time.entry(Arc::clone(&tenant_id)) {
                            Entry::Vacant(v) => {
                                self.tenant_char_count
                                    .entry(Arc::clone(&tenant_id))
                                    .and_modify(|count| *count += matched_node_text_count)
                                    .or_insert(matched_node_text_count);
                                v.insert(timestamp_ms);
                            }
                            Entry::Occupied(mut o) => {
                                o.insert(timestamp_ms);
                            }
                        }
                        curr_idx += shared_count;
                    }
                }
            }
        }
    }

    #[allow(unused_assignments)]
    pub fn prefix_match(&self, text: &str) -> (String, String) {
        // Pre-index text once for O(1) character access
        let indexed_text = CharIndexedText::new(text);
        let text_count = indexed_text.len();

        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        let mut prev = Arc::clone(&self.root);

        while curr_idx < text_count {
            // O(1) character access instead of O(n) chars().nth()
            let first_char = indexed_text.get(curr_idx).unwrap();

            curr = prev.clone();

            if let Some(entry) = curr.children.get(&first_char) {
                let matched_node = entry.value().clone();
                let matched_text_guard = matched_node.text.read().unwrap();
                // Use indexed comparison to avoid creating intermediate string
                let shared_count = shared_prefix_count_indexed(
                    &indexed_text,
                    curr_idx,
                    matched_text_guard.as_str(),
                );
                // Use cached char count instead of chars().count()
                let matched_node_text_count = matched_text_guard.char_count();
                drop(matched_text_guard);

                if shared_count == matched_node_text_count {
                    // Full match with current node's text, continue to next node
                    curr_idx += shared_count;
                    prev = Arc::clone(&matched_node);
                } else {
                    // Partial match, stop here
                    curr_idx += shared_count;
                    prev = Arc::clone(&matched_node);
                    break;
                }
            } else {
                // No match found, stop here
                break;
            }
        }

        curr = prev.clone();

        // Select the first tenant (key in the map) - use Arc<str> directly
        let tenant: Option<TenantId> = curr
            .tenant_last_access_time
            .iter()
            .next()
            .map(|kv| Arc::clone(kv.key()));

        // Use cached timestamp to reduce syscalls
        let timestamp_ms = get_timestamp_ms();

        // Traverse from the curr node to the root and update the timestamp
        if let Some(ref tenant_id) = tenant {
            let mut current_node = Some(curr);
            while let Some(node) = current_node {
                node.tenant_last_access_time
                    .insert(Arc::clone(tenant_id), timestamp_ms);
                current_node = node.parent.read().unwrap().clone();
            }
        }

        // Use indexed slice for result
        let ret_text = indexed_text.slice_to_string(0, curr_idx);
        let tenant_str = tenant
            .map(|t| t.to_string())
            .unwrap_or_else(|| "empty".to_string());
        (ret_text, tenant_str)
    }

    #[allow(unused_assignments, dead_code)]
    pub fn prefix_match_tenant(&self, text: &str, tenant: &str) -> String {
        // Pre-index text once for O(1) character access
        let indexed_text = CharIndexedText::new(text);
        let text_count = indexed_text.len();

        // Intern tenant ID once for efficient lookups
        let tenant_id = intern_tenant(tenant);

        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        let mut prev = Arc::clone(&self.root);

        while curr_idx < text_count {
            // O(1) character access instead of O(n) chars().nth()
            let first_char = indexed_text.get(curr_idx).unwrap();

            curr = prev.clone();

            if let Some(entry) = curr.children.get(&first_char) {
                let matched_node = entry.value().clone();

                // Only continue matching if this node belongs to the specified tenant
                // Note: contains_key with &str works because Arc<str> implements Borrow<str>
                if !matched_node
                    .tenant_last_access_time
                    .contains_key(tenant_id.as_ref())
                {
                    break;
                }

                let matched_text_guard = matched_node.text.read().unwrap();
                // Use indexed comparison to avoid creating intermediate string
                let shared_count = shared_prefix_count_indexed(
                    &indexed_text,
                    curr_idx,
                    matched_text_guard.as_str(),
                );
                // Use cached char count instead of chars().count()
                let matched_node_text_count = matched_text_guard.char_count();
                drop(matched_text_guard);

                if shared_count == matched_node_text_count {
                    // Full match with current node's text, continue to next node
                    curr_idx += shared_count;
                    prev = Arc::clone(&matched_node);
                } else {
                    // Partial match, stop here
                    curr_idx += shared_count;
                    prev = Arc::clone(&matched_node);
                    break;
                }
            } else {
                // No match found, stop here
                break;
            }
        }

        curr = prev.clone();

        // Only update timestamp if we found a match for the specified tenant
        if curr
            .tenant_last_access_time
            .contains_key(tenant_id.as_ref())
        {
            // Use cached timestamp to reduce syscalls
            let timestamp_ms = get_timestamp_ms();

            let mut current_node = Some(curr);
            while let Some(node) = current_node {
                node.tenant_last_access_time
                    .insert(Arc::clone(&tenant_id), timestamp_ms);
                current_node = node.parent.read().unwrap().clone();
            }
        }

        // Use indexed slice for result
        indexed_text.slice_to_string(0, curr_idx)
    }

    fn leaf_of(node: &NodeRef) -> Vec<TenantId> {
        /*
        Return the list of tenants if it's a leaf for the tenant.
        A tenant is a "leaf" at this node if this node has the tenant but none of its children do.
         */
        let mut candidates: HashMap<TenantId, bool> = node
            .tenant_last_access_time
            .iter()
            .map(|entry| (Arc::clone(entry.key()), true))
            .collect();

        for child in node.children.iter() {
            for tenant in child.value().tenant_last_access_time.iter() {
                // Mark as non-leaf if any child has this tenant
                candidates.insert(Arc::clone(tenant.key()), false);
            }
        }

        candidates
            .into_iter()
            .filter(|(_, is_leaf)| *is_leaf)
            .map(|(tenant, _)| tenant)
            .collect()
    }

    pub fn evict_tenant_by_size(&self, max_size: usize) {
        // Calculate used size and collect leaves
        let mut stack = vec![Arc::clone(&self.root)];
        let mut pq = BinaryHeap::new();

        while let Some(curr) = stack.pop() {
            for child in curr.children.iter() {
                stack.push(Arc::clone(child.value()));
            }

            // Add leaves to priority queue
            for tenant in Tree::leaf_of(&curr) {
                if let Some(timestamp) = curr.tenant_last_access_time.get(tenant.as_ref()) {
                    pq.push(Reverse(EvictionEntry {
                        timestamp: *timestamp,
                        tenant: Arc::clone(&tenant),
                        node: Arc::clone(&curr),
                    }));
                }
            }
        }

        info!("Before eviction - Used size per tenant:");
        for entry in self.tenant_char_count.iter() {
            info!("Tenant: {}, Size: {}", entry.key(), entry.value());
        }

        // Process eviction
        while let Some(Reverse(entry)) = pq.pop() {
            let EvictionEntry { tenant, node, .. } = entry;

            if let Some(used_size) = self.tenant_char_count.get(tenant.as_ref()) {
                if *used_size <= max_size {
                    continue;
                }
            }

            // Decrement when removing tenant from node
            if node.tenant_last_access_time.contains_key(tenant.as_ref()) {
                // Use cached char count instead of chars().count()
                let node_len = node.text.read().unwrap().char_count();
                self.tenant_char_count
                    .entry(Arc::clone(&tenant))
                    .and_modify(|count| {
                        *count = count.saturating_sub(node_len);
                    });
            }

            // Remove tenant from node
            node.tenant_last_access_time.remove(tenant.as_ref());

            // Remove empty nodes
            if node.children.is_empty() && node.tenant_last_access_time.is_empty() {
                if let Some(parent) = node.parent.read().unwrap().as_ref() {
                    let text_guard = node.text.read().unwrap();
                    if let Some(first_char) = text_guard.first_char() {
                        parent.children.remove(&first_char);
                    }
                }
            }

            // Add parent to queue if it becomes a leaf
            if let Some(parent) = node.parent.read().unwrap().as_ref() {
                let parent_leaves = Tree::leaf_of(parent);
                if parent_leaves.iter().any(|t| t.as_ref() == tenant.as_ref()) {
                    if let Some(timestamp) = parent.tenant_last_access_time.get(tenant.as_ref()) {
                        pq.push(Reverse(EvictionEntry {
                            timestamp: *timestamp,
                            tenant: Arc::clone(&tenant),
                            node: Arc::clone(parent),
                        }));
                    }
                }
            };
        }

        info!("After eviction - Used size per tenant:");
        for entry in self.tenant_char_count.iter() {
            info!("Tenant: {}, Size: {}", entry.key(), entry.value());
        }
    }

    pub fn remove_tenant(&self, tenant: &str) {
        // Intern tenant ID once for efficient lookups
        let tenant_id = intern_tenant(tenant);

        // 1. Find all the leaves for the tenant
        let mut stack = vec![Arc::clone(&self.root)];
        let mut queue = VecDeque::new();

        while let Some(curr) = stack.pop() {
            for child in curr.children.iter() {
                stack.push(Arc::clone(child.value()));
            }

            let leaves = Tree::leaf_of(&curr);
            if leaves.iter().any(|t| t.as_ref() == tenant_id.as_ref()) {
                queue.push_back(Arc::clone(&curr));
            }
        }

        // 2. Start from the leaves and traverse up to the root, removing the tenant from each node
        while let Some(curr) = queue.pop_front() {
            // remove tenant from node
            curr.tenant_last_access_time.remove(tenant_id.as_ref());

            // remove empty nodes
            if curr.children.is_empty() && curr.tenant_last_access_time.is_empty() {
                if let Some(parent) = curr.parent.read().unwrap().as_ref() {
                    let text_guard = curr.text.read().unwrap();
                    if let Some(first_char) = text_guard.first_char() {
                        parent.children.remove(&first_char);
                    }
                }
            }

            // add parent to queue if it becomes a leaf
            if let Some(parent) = curr.parent.read().unwrap().as_ref() {
                let parent_leaves = Tree::leaf_of(parent);
                if parent_leaves
                    .iter()
                    .any(|t| t.as_ref() == tenant_id.as_ref())
                {
                    queue.push_back(Arc::clone(parent));
                }
            }
        }

        // 3. Remove the tenant from the tenant_char_count map
        self.tenant_char_count.remove(tenant_id.as_ref());
    }

    #[allow(dead_code)]
    pub fn get_tenant_char_count(&self) -> HashMap<String, usize> {
        self.tenant_char_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    #[allow(dead_code)]
    pub fn get_used_size_per_tenant(&self) -> HashMap<String, usize> {
        // perform a DFS to traverse all nodes and calculate the total size used by each tenant

        let mut used_size_per_tenant: HashMap<String, usize> = HashMap::new();
        let mut stack = vec![Arc::clone(&self.root)];

        while let Some(curr) = stack.pop() {
            // Use cached char count instead of chars().count()
            let text_count = curr.text.read().unwrap().char_count();

            for tenant in curr.tenant_last_access_time.iter() {
                let size = used_size_per_tenant
                    .entry(tenant.key().to_string())
                    .or_insert(0);
                *size += text_count;
            }

            for child in curr.children.iter() {
                stack.push(Arc::clone(child.value()));
            }
        }

        used_size_per_tenant
    }

    #[allow(dead_code)]
    fn node_to_string(node: &NodeRef, prefix: &str, is_last: bool) -> String {
        use std::time::Duration;

        let mut result = String::new();

        // Add prefix and branch character
        result.push_str(prefix);
        result.push_str(if is_last { "└── " } else { "├── " });

        // Add node text
        let node_text = node.text.read().unwrap();
        result.push_str(&format!("'{}' [", node_text.as_str()));

        // Add tenant information with timestamps
        let mut tenant_info = Vec::new();
        for entry in node.tenant_last_access_time.iter() {
            let tenant_id = entry.key();
            let timestamp_ms = entry.value();

            // Convert milliseconds to seconds and remaining milliseconds
            let seconds = (timestamp_ms / 1000) as u64;
            let millis = (timestamp_ms % 1000) as u32;

            // Create SystemTime from Unix timestamp
            let system_time = UNIX_EPOCH + Duration::from_secs(seconds);

            // Format time as HH:MM:SS.mmm
            let datetime = system_time.duration_since(UNIX_EPOCH).unwrap();
            let hours = (datetime.as_secs() % 86400) / 3600;
            let minutes = (datetime.as_secs() % 3600) / 60;
            let seconds = datetime.as_secs() % 60;

            tenant_info.push(format!(
                "{} | {:02}:{:02}:{:02}.{:03}",
                tenant_id, hours, minutes, seconds, millis
            ));
        }

        result.push_str(&tenant_info.join(", "));
        result.push_str("]\n");

        // Process children
        let children: Vec<_> = node.children.iter().collect();
        let child_count = children.len();

        for (i, entry) in children.iter().enumerate() {
            let is_last_child = i == child_count - 1;
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            result.push_str(&Tree::node_to_string(
                entry.value(),
                &new_prefix,
                is_last_child,
            ));
        }

        result
    }

    #[allow(dead_code)]
    pub fn pretty_print(&self) {
        if self.root.children.is_empty() {
            return;
        }

        let mut result = String::new();
        let children: Vec<_> = self.root.children.iter().collect();
        let child_count = children.len();

        for (i, entry) in children.iter().enumerate() {
            let is_last = i == child_count - 1;
            result.push_str(&Tree::node_to_string(entry.value(), "", is_last));
        }

        println!("{result}");
    }
}

//  Unit tests
#[cfg(test)]
mod tests {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use rand::{
        distr::{Alphanumeric, SampleString},
        rng as thread_rng, Rng,
    };

    use super::*;

    /// Helper to convert tenant_char_count to HashMap<String, usize> for comparison
    fn get_maintained_counts(tree: &Tree) -> HashMap<String, usize> {
        tree.tenant_char_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    #[test]
    fn test_tenant_char_count() {
        let tree = Tree::new();

        tree.insert("apple", "tenant1");
        tree.insert("apricot", "tenant1");
        tree.insert("banana", "tenant1");
        tree.insert("amplify", "tenant2");
        tree.insert("application", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 1 - Maintained vs Computed counts:");
        println!(
            "Maintained: {:?}\nComputed: {:?}",
            maintained_counts, computed_sizes
        );
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 1: Initial insertions"
        );

        tree.insert("apartment", "tenant1");
        tree.insert("appetite", "tenant2");
        tree.insert("ball", "tenant1");
        tree.insert("box", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 2 - Maintained vs Computed counts:");
        println!(
            "Maintained: {:?}\nComputed: {:?}",
            maintained_counts, computed_sizes
        );
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 2: Additional insertions"
        );

        tree.insert("zebra", "tenant1");
        tree.insert("zebra", "tenant2");
        tree.insert("zero", "tenant1");
        tree.insert("zero", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 3 - Maintained vs Computed counts:");
        println!(
            "Maintained: {:?}\nComputed: {:?}",
            maintained_counts, computed_sizes
        );
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 3: Overlapping insertions"
        );

        tree.evict_tenant_by_size(10);

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 4 - Maintained vs Computed counts:");
        println!(
            "Maintained: {:?}\nComputed: {:?}",
            maintained_counts, computed_sizes
        );
        assert_eq!(maintained_counts, computed_sizes, "Phase 4: After eviction");
    }

    fn random_string(len: usize) -> String {
        Alphanumeric.sample_string(&mut thread_rng(), len)
    }

    #[test]
    fn test_cold_start() {
        let tree = Tree::new();

        let (matched_text, tenant) = tree.prefix_match("hello");

        assert_eq!(matched_text, "");
        assert_eq!(tenant, "empty");
    }

    #[test]
    fn test_exact_match_seq() {
        let tree = Tree::new();
        tree.insert("hello", "tenant1");
        tree.pretty_print();
        tree.insert("apple", "tenant2");
        tree.pretty_print();
        tree.insert("banana", "tenant3");
        tree.pretty_print();

        let (matched_text, tenant) = tree.prefix_match("hello");
        assert_eq!(matched_text, "hello");
        assert_eq!(tenant, "tenant1");

        let (matched_text, tenant) = tree.prefix_match("apple");
        assert_eq!(matched_text, "apple");
        assert_eq!(tenant, "tenant2");

        let (matched_text, tenant) = tree.prefix_match("banana");
        assert_eq!(matched_text, "banana");
        assert_eq!(tenant, "tenant3");
    }

    #[test]
    fn test_exact_match_concurrent() {
        let tree = Arc::new(Tree::new());

        // spawn 3 threads for insert
        let tree_clone = Arc::clone(&tree);

        let texts = ["hello", "apple", "banana"];
        let tenants = ["tenant1", "tenant2", "tenant3"];

        let mut handles = vec![];

        for i in 0..3 {
            let tree_clone = Arc::clone(&tree_clone);
            let text = texts[i];
            let tenant = tenants[i];

            let handle = thread::spawn(move || {
                tree_clone.insert(text, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }

        // spawn 3 threads for match
        let mut handles = vec![];

        let tree_clone = Arc::clone(&tree);

        for i in 0..3 {
            let tree_clone = Arc::clone(&tree_clone);
            let text = texts[i];
            let tenant = tenants[i];

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
                assert_eq!(matched_text, text);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_partial_match_concurrent() {
        let tree = Arc::new(Tree::new());

        // spawn 3 threads for insert
        let tree_clone = Arc::clone(&tree);

        static TEXTS: [&str; 3] = ["apple", "apabc", "acbdeds"];

        let mut handles = vec![];

        for text in TEXTS.iter() {
            let tree_clone = Arc::clone(&tree_clone);
            let tenant = "tenant0";

            let handle = thread::spawn(move || {
                tree_clone.insert(text, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }

        // spawn 3 threads for match
        let mut handles = vec![];

        let tree_clone = Arc::clone(&tree);

        for text in TEXTS.iter() {
            let tree_clone = Arc::clone(&tree_clone);
            let tenant = "tenant0";

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
                assert_eq!(matched_text, *text);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_group_prefix_insert_match_concurrent() {
        static PREFIXES: [&str; 4] = [
            "Clock strikes midnight, I'm still wide awake",
            "Got dreams bigger than these city lights",
            "Time waits for no one, gotta make my move",
            "Started from the bottom, that's no metaphor",
        ];
        let suffixes = [
            "Got too much to prove, ain't got time to lose",
            "History in the making, yeah, you can't erase this",
        ];
        let tree = Arc::new(Tree::new());

        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            for suffix in suffixes.iter() {
                let tree_clone = Arc::clone(&tree);
                let text = format!("{} {}", prefix, suffix);
                let tenant = format!("tenant{}", i);

                let handle = thread::spawn(move || {
                    tree_clone.insert(&text, &tenant);
                });

                handles.push(handle);
            }
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }

        tree.pretty_print();

        // check matching using multi threads
        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match(prefix);
                let tenant = format!("tenant{}", i);
                assert_eq!(matched_text, *prefix);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_mixed_concurrent_insert_match() {
        // ensure it does not deadlock instead of doing correctness check

        static PREFIXES: [&str; 4] = [
            "Clock strikes midnight, I'm still wide awake",
            "Got dreams bigger than these city lights",
            "Time waits for no one, gotta make my move",
            "Started from the bottom, that's no metaphor",
        ];
        let suffixes = [
            "Got too much to prove, ain't got time to lose",
            "History in the making, yeah, you can't erase this",
        ];
        let tree = Arc::new(Tree::new());

        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            for suffix in suffixes.iter() {
                let tree_clone = Arc::clone(&tree);
                let text = format!("{} {}", prefix, suffix);
                let tenant = format!("tenant{}", i);

                let handle = thread::spawn(move || {
                    tree_clone.insert(&text, &tenant);
                });

                handles.push(handle);
            }
        }

        // check matching using multi threads
        for prefix in PREFIXES.iter() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (_matched_text, _matched_tenant) = tree_clone.prefix_match(prefix);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_utf8_split_seq() {
        // The string should be indexed and split by a utf-8 value basis instead of byte basis
        // use .chars() to get the iterator of the utf-8 value
        let tree = Arc::new(Tree::new());

        static TEST_PAIRS: [(&str, &str); 3] = [
            ("你好嗎", "tenant1"),
            ("你好喔", "tenant2"),
            ("你心情好嗎", "tenant3"),
        ];

        // Insert sequentially
        for (text, tenant) in TEST_PAIRS.iter() {
            tree.insert(text, tenant);
        }

        tree.pretty_print();

        for (text, tenant) in TEST_PAIRS.iter() {
            let (matched_text, matched_tenant) = tree.prefix_match(text);
            assert_eq!(matched_text, *text);
            assert_eq!(matched_tenant, *tenant);
        }
    }

    #[test]
    fn test_utf8_split_concurrent() {
        let tree = Arc::new(Tree::new());

        static TEST_PAIRS: [(&str, &str); 3] = [
            ("你好嗎", "tenant1"),
            ("你好喔", "tenant2"),
            ("你心情好嗎", "tenant3"),
        ];

        // Create multiple threads for insertion
        let mut handles = vec![];

        for (text, tenant) in TEST_PAIRS.iter() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                tree_clone.insert(text, tenant);
            });

            handles.push(handle);
        }

        // Wait for all insertions to complete
        for handle in handles {
            handle.join().unwrap();
        }

        tree.pretty_print();

        // Create multiple threads for matching
        let mut handles = vec![];

        for (text, tenant) in TEST_PAIRS.iter() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
                assert_eq!(matched_text, *text);
                assert_eq!(matched_tenant, *tenant);
            });

            handles.push(handle);
        }

        // Wait for all matches to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_simple_eviction() {
        let tree = Tree::new();
        let max_size = 5;

        // Insert strings for both tenants
        tree.insert("hello", "tenant1"); // size 5

        tree.insert("hello", "tenant2"); // size 5
        thread::sleep(Duration::from_millis(10));
        tree.insert("world", "tenant2"); // size 5, total for tenant2 = 10

        tree.pretty_print();

        let sizes_before = tree.get_used_size_per_tenant();
        assert_eq!(sizes_before.get("tenant1").unwrap(), &5); // "hello" = 5
        assert_eq!(sizes_before.get("tenant2").unwrap(), &10); // "hello" + "world" = 10

        // Evict - should remove "hello" from tenant2 as it's the oldest
        tree.evict_tenant_by_size(max_size);

        tree.pretty_print();

        let sizes_after = tree.get_used_size_per_tenant();
        assert_eq!(sizes_after.get("tenant1").unwrap(), &5); // Should be unchanged
        assert_eq!(sizes_after.get("tenant2").unwrap(), &5); // Only "world" remains

        let (matched, tenant) = tree.prefix_match("world");
        assert_eq!(matched, "world");
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_advanced_eviction() {
        let tree = Tree::new();

        // Set limits for each tenant
        let max_size: usize = 100;

        // Define prefixes
        let prefixes = ["aqwefcisdf", "iajsdfkmade", "kjnzxcvewqe", "iejksduqasd"];

        // Insert strings with shared prefixes
        for _i in 0..100 {
            for (j, prefix) in prefixes.iter().enumerate() {
                let random_suffix = random_string(10);
                let text = format!("{}{}", prefix, random_suffix);
                let tenant = format!("tenant{}", j + 1);
                tree.insert(&text, &tenant);
            }
        }

        // Perform eviction
        tree.evict_tenant_by_size(max_size);

        // Check sizes after eviction
        let sizes_after = tree.get_used_size_per_tenant();
        for (tenant, &size) in sizes_after.iter() {
            assert!(
                size <= max_size,
                "Tenant {} exceeds size limit. Current size: {}, Limit: {}",
                tenant,
                size,
                max_size
            );
        }
    }

    #[test]
    fn test_concurrent_operations_with_eviction() {
        // Ensure eviction works fine with concurrent insert and match operations for a given period

        let tree = Arc::new(Tree::new());
        let mut handles = vec![];
        let test_duration = Duration::from_secs(10);
        let start_time = Instant::now();
        let max_size = 100; // Single max size for all tenants

        // Spawn eviction thread
        {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                while start_time.elapsed() < test_duration {
                    // Run eviction
                    tree.evict_tenant_by_size(max_size);

                    // Sleep for 5 seconds
                    thread::sleep(Duration::from_secs(5));
                }
            });
            handles.push(handle);
        }

        // Spawn 4 worker threads
        for thread_id in 0..4 {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                let mut rng = rand::rng();
                let tenant = format!("tenant{}", thread_id + 1);
                let prefix = format!("prefix{}", thread_id);

                while start_time.elapsed() < test_duration {
                    // Random decision: match or insert (70% match, 30% insert)
                    if rng.random_bool(0.7) {
                        // Perform match operation
                        let random_len = rng.random_range(3..10);
                        let search_str = format!("{}{}", prefix, random_string(random_len));
                        let (_matched, _) = tree.prefix_match(&search_str);
                    } else {
                        // Perform insert operation
                        let random_len = rng.random_range(5..15);
                        let insert_str = format!("{}{}", prefix, random_string(random_len));
                        tree.insert(&insert_str, &tenant);
                        // println!("Thread {} inserted: {}", thread_id, insert_str);
                    }

                    // Small random sleep to vary timing
                    thread::sleep(Duration::from_millis(rng.random_range(10..100)));
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // final eviction
        tree.evict_tenant_by_size(max_size);

        // Final size check
        let final_sizes = tree.get_used_size_per_tenant();
        println!("Final sizes after test completion: {:?}", final_sizes);

        for (_, &size) in final_sizes.iter() {
            assert!(
                size <= max_size,
                "Tenant exceeds size limit. Final size: {}, Limit: {}",
                size,
                max_size
            );
        }
    }

    #[test]
    fn test_leaf_of() {
        let tree = Tree::new();

        // Helper to convert leaves to strings for easier assertion
        let leaves_as_strings =
            |leaves: &[TenantId]| -> Vec<String> { leaves.iter().map(|t| t.to_string()).collect() };

        // Single node
        tree.insert("hello", "tenant1");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        assert_eq!(leaves_as_strings(&leaves), vec!["tenant1"]);

        // Node with multiple tenants
        tree.insert("hello", "tenant2");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        let leaves_str = leaves_as_strings(&leaves);
        assert_eq!(leaves_str.len(), 2);
        assert!(leaves_str.contains(&"tenant1".to_string()));
        assert!(leaves_str.contains(&"tenant2".to_string()));

        // Non-leaf node
        tree.insert("hi", "tenant1");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        assert!(leaves.is_empty());
    }

    #[test]
    fn test_get_used_size_per_tenant() {
        let tree = Tree::new();

        // Single tenant
        tree.insert("hello", "tenant1");
        tree.insert("world", "tenant1");
        let sizes = tree.get_used_size_per_tenant();

        tree.pretty_print();
        println!("{:?}", sizes);
        assert_eq!(sizes.get("tenant1").unwrap(), &10); // "hello" + "world"

        // Multiple tenants sharing nodes
        tree.insert("hello", "tenant2");
        tree.insert("help", "tenant2");
        let sizes = tree.get_used_size_per_tenant();

        tree.pretty_print();
        println!("{:?}", sizes);
        assert_eq!(sizes.get("tenant1").unwrap(), &10);
        assert_eq!(sizes.get("tenant2").unwrap(), &6); // "hello" + "p"

        // UTF-8 characters
        tree.insert("你好", "tenant3");
        let sizes = tree.get_used_size_per_tenant();
        tree.pretty_print();
        println!("{:?}", sizes);
        assert_eq!(sizes.get("tenant3").unwrap(), &2); // 2 Chinese characters

        tree.pretty_print();
    }

    #[test]
    fn test_prefix_match_tenant() {
        let tree = Tree::new();

        // Insert overlapping prefixes for different tenants
        tree.insert("hello", "tenant1"); // tenant1: hello
        tree.insert("hello", "tenant2"); // tenant2: hello
        tree.insert("hello world", "tenant2"); // tenant2: hello -> world
        tree.insert("help", "tenant1"); // tenant1: hel -> p
        tree.insert("helicopter", "tenant2"); // tenant2: hel -> icopter

        assert_eq!(tree.prefix_match_tenant("hello", "tenant1"), "hello"); // Full match for tenant1
        assert_eq!(tree.prefix_match_tenant("help", "tenant1"), "help"); // Exclusive to tenant1
        assert_eq!(tree.prefix_match_tenant("hel", "tenant1"), "hel"); // Shared prefix
        assert_eq!(tree.prefix_match_tenant("hello world", "tenant1"), "hello"); // Should stop at tenant1's boundary
        assert_eq!(tree.prefix_match_tenant("helicopter", "tenant1"), "hel"); // Should stop at tenant1's boundary

        assert_eq!(tree.prefix_match_tenant("hello", "tenant2"), "hello"); // Full match for tenant2
        assert_eq!(
            tree.prefix_match_tenant("hello world", "tenant2"),
            "hello world"
        ); // Exclusive to tenant2
        assert_eq!(
            tree.prefix_match_tenant("helicopter", "tenant2"),
            "helicopter"
        ); // Exclusive to tenant2
        assert_eq!(tree.prefix_match_tenant("hel", "tenant2"), "hel"); // Shared prefix
        assert_eq!(tree.prefix_match_tenant("help", "tenant2"), "hel"); // Should stop at tenant2's boundary

        assert_eq!(tree.prefix_match_tenant("hello", "tenant3"), ""); // Non-existent tenant
        assert_eq!(tree.prefix_match_tenant("help", "tenant3"), ""); // Non-existent tenant
    }

    #[test]
    fn test_simple_tenant_eviction() {
        let tree = Tree::new();

        // Insert data for multiple tenants
        tree.insert("hello", "tenant1");
        tree.insert("world", "tenant1");
        tree.insert("hello", "tenant2");
        tree.insert("help", "tenant2");

        let initial_sizes = tree.get_used_size_per_tenant();
        assert_eq!(initial_sizes.get("tenant1").unwrap(), &10); // "hello" + "world"
        assert_eq!(initial_sizes.get("tenant2").unwrap(), &6); // "hello" + "p"

        // Evict tenant1
        tree.remove_tenant("tenant1");

        let final_sizes = tree.get_used_size_per_tenant();
        assert!(
            !final_sizes.contains_key("tenant1"),
            "tenant1 should be completely removed"
        );
        assert_eq!(
            final_sizes.get("tenant2").unwrap(),
            &6,
            "tenant2 should be unaffected"
        );

        assert_eq!(tree.prefix_match_tenant("hello", "tenant1"), "");
        assert_eq!(tree.prefix_match_tenant("world", "tenant1"), "");

        assert_eq!(tree.prefix_match_tenant("hello", "tenant2"), "hello");
        assert_eq!(tree.prefix_match_tenant("help", "tenant2"), "help");
    }

    #[test]
    fn test_complex_tenant_eviction() {
        let tree = Tree::new();

        // Create a more complex tree structure with shared prefixes
        tree.insert("apple", "tenant1");
        tree.insert("application", "tenant1");
        tree.insert("apple", "tenant2");
        tree.insert("appetite", "tenant2");
        tree.insert("banana", "tenant1");
        tree.insert("banana", "tenant2");
        tree.insert("ball", "tenant2");

        let initial_sizes = tree.get_used_size_per_tenant();
        println!("Initial sizes: {:?}", initial_sizes);
        tree.pretty_print();

        // Evict tenant1
        tree.remove_tenant("tenant1");

        let final_sizes = tree.get_used_size_per_tenant();
        println!("Final sizes: {:?}", final_sizes);
        tree.pretty_print();

        assert!(
            !final_sizes.contains_key("tenant1"),
            "tenant1 should be completely removed"
        );

        assert_eq!(tree.prefix_match_tenant("apple", "tenant1"), "");
        assert_eq!(tree.prefix_match_tenant("application", "tenant1"), "");
        assert_eq!(tree.prefix_match_tenant("banana", "tenant1"), "");

        assert_eq!(tree.prefix_match_tenant("apple", "tenant2"), "apple");
        assert_eq!(tree.prefix_match_tenant("appetite", "tenant2"), "appetite");
        assert_eq!(tree.prefix_match_tenant("banana", "tenant2"), "banana");
        assert_eq!(tree.prefix_match_tenant("ball", "tenant2"), "ball");

        let tenant2_size = final_sizes.get("tenant2").unwrap();
        assert_eq!(tenant2_size, &(5 + 5 + 6 + 2)); // "apple" + "etite" + "banana" + "ll"
    }
}
