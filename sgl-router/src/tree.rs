use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use dashmap::{mapref::entry::Entry, DashMap};
use tracing::info;

type NodeRef = Arc<Node>;

#[derive(Debug)]
struct Node {
    children: DashMap<char, NodeRef>,
    text: RwLock<String>,
    tenant_last_access_time: DashMap<String, u128>,
    parent: RwLock<Option<NodeRef>>,
}

#[derive(Debug)]
pub struct Tree {
    root: NodeRef,
    pub tenant_char_count: DashMap<String, usize>,
}

// For the heap

struct EvictionEntry {
    timestamp: u128,
    tenant: String,
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

fn shared_prefix_count(a: &str, b: &str) -> usize {
    let mut i = 0;
    let mut a_iter = a.chars();
    let mut b_iter = b.chars();

    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(a_char), Some(b_char)) if a_char == b_char => {
                i += 1;
            }
            _ => break,
        }
    }

    i
}

fn slice_by_chars(s: &str, start: usize, end: usize) -> String {
    s.chars().skip(start).take(end - start).collect()
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
    */

    pub fn new() -> Self {
        Tree {
            root: Arc::new(Node {
                children: DashMap::new(),
                text: RwLock::new("".to_string()),
                tenant_last_access_time: DashMap::new(),
                parent: RwLock::new(None),
            }),
            tenant_char_count: DashMap::new(),
        }
    }

    pub fn insert(&self, text: &str, tenant: &str) {
        // Insert text into tree with given tenant

        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        curr.tenant_last_access_time
            .insert(tenant.to_string(), timestamp_ms);

        self.tenant_char_count
            .entry(tenant.to_string())
            .or_insert(0);

        let mut prev = Arc::clone(&self.root);

        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();

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

                    let curr_text = slice_by_chars(text, curr_idx, text_count);
                    let curr_text_count = curr_text.chars().count();
                    let new_node = Arc::new(Node {
                        children: DashMap::new(),
                        text: RwLock::new(curr_text),
                        tenant_last_access_time: DashMap::new(),
                        parent: RwLock::new(Some(Arc::clone(&curr))),
                    });

                    // Attach tenant to the new node (map is empty here) and increment count once
                    self.tenant_char_count
                        .entry(tenant.to_string())
                        .and_modify(|count| *count += curr_text_count)
                        .or_insert(curr_text_count);
                    new_node
                        .tenant_last_access_time
                        .insert(tenant.to_string(), timestamp_ms);

                    entry.insert(Arc::clone(&new_node));

                    prev = Arc::clone(&new_node);
                    curr_idx = text_count;
                }

                Entry::Occupied(mut entry) => {
                    // matched
                    let matched_node = entry.get().clone();

                    let matched_node_text = matched_node.text.read().unwrap().to_owned();
                    let matched_node_text_count = matched_node_text.chars().count();

                    let curr_text = slice_by_chars(text, curr_idx, text_count);
                    let shared_count = shared_prefix_count(&matched_node_text, &curr_text);

                    if shared_count < matched_node_text_count {
                        /*
                           split the matched node
                           [curr] -> [matched_node] =>
                           becomes
                           [curr] -> [new_node] -> [contracted_matched_node]
                        */

                        let matched_text = slice_by_chars(&matched_node_text, 0, shared_count);
                        let contracted_text = slice_by_chars(
                            &matched_node_text,
                            shared_count,
                            matched_node_text_count,
                        );
                        let matched_text_count = matched_text.chars().count();

                        let new_node = Arc::new(Node {
                            text: RwLock::new(matched_text),
                            children: DashMap::new(),
                            parent: RwLock::new(Some(Arc::clone(&curr))),
                            tenant_last_access_time: matched_node.tenant_last_access_time.clone(),
                        });

                        let first_new_char = contracted_text.chars().nth(0).unwrap();
                        new_node
                            .children
                            .insert(first_new_char, Arc::clone(&matched_node));

                        entry.insert(Arc::clone(&new_node));

                        *matched_node.text.write().unwrap() = contracted_text;
                        *matched_node.parent.write().unwrap() = Some(Arc::clone(&new_node));

                        prev = Arc::clone(&new_node);

                        // Atomically attach tenant to the new split node and increment count once
                        match prev.tenant_last_access_time.entry(tenant.to_string()) {
                            Entry::Vacant(v) => {
                                self.tenant_char_count
                                    .entry(tenant.to_string())
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
                        prev = Arc::clone(&matched_node);

                        // Atomically attach tenant to existing node and increment count once
                        match prev.tenant_last_access_time.entry(tenant.to_string()) {
                            Entry::Vacant(v) => {
                                self.tenant_char_count
                                    .entry(tenant.to_string())
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
        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        let mut prev = Arc::clone(&self.root);
        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();
            let curr_text = slice_by_chars(text, curr_idx, text_count);

            curr = prev.clone();

            if let Some(entry) = curr.children.get(&first_char) {
                let matched_node = entry.value().clone();
                let matched_text_guard = matched_node.text.read().unwrap();
                let shared_count = shared_prefix_count(&matched_text_guard, &curr_text);
                let matched_node_text_count = matched_text_guard.chars().count();
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

        // Select the first tenant (key in the map)
        let tenant = curr
            .tenant_last_access_time
            .iter()
            .next()
            .map(|kv| kv.key().to_owned())
            .unwrap_or("empty".to_string());

        // Traverse from the curr node to the root and update the timestamp

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        if !tenant.eq("empty") {
            let mut current_node = Some(curr);
            while let Some(node) = current_node {
                node.tenant_last_access_time
                    .insert(tenant.clone(), timestamp_ms);
                current_node = node.parent.read().unwrap().clone();
            }
        }

        let ret_text = slice_by_chars(text, 0, curr_idx);
        (ret_text, tenant)
    }

    #[allow(unused_assignments)]
    pub fn prefix_match_tenant(&self, text: &str, tenant: &str) -> String {
        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;

        let mut prev = Arc::clone(&self.root);
        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();
            let curr_text = slice_by_chars(text, curr_idx, text_count);

            curr = prev.clone();

            if let Some(entry) = curr.children.get(&first_char) {
                let matched_node = entry.value().clone();

                // Only continue matching if this node belongs to the specified tenant
                if !matched_node.tenant_last_access_time.contains_key(tenant) {
                    break;
                }

                let matched_text_guard = matched_node.text.read().unwrap();
                let shared_count = shared_prefix_count(&matched_text_guard, &curr_text);
                let matched_node_text_count = matched_text_guard.chars().count();
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
        if curr.tenant_last_access_time.contains_key(tenant) {
            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();

            let mut current_node = Some(curr);
            while let Some(node) = current_node {
                node.tenant_last_access_time
                    .insert(tenant.to_string(), timestamp_ms);
                current_node = node.parent.read().unwrap().clone();
            }
        }

        slice_by_chars(text, 0, curr_idx)
    }

    fn leaf_of(node: &NodeRef) -> Vec<String> {
        /*
        Return the list of tenants if it's a leaf for the tenant
         */
        let mut candidates: HashMap<String, bool> = node
            .tenant_last_access_time
            .iter()
            .map(|entry| (entry.key().clone(), true))
            .collect();

        for child in node.children.iter() {
            for tenant in child.value().tenant_last_access_time.iter() {
                candidates.insert(tenant.key().clone(), false);
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
                if let Some(timestamp) = curr.tenant_last_access_time.get(&tenant) {
                    pq.push(Reverse(EvictionEntry {
                        timestamp: *timestamp,
                        tenant: tenant.clone(),
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

            if let Some(used_size) = self.tenant_char_count.get(&tenant) {
                if *used_size <= max_size {
                    continue;
                }
            }

            // Decrement when removing tenant from node
            if node.tenant_last_access_time.contains_key(&tenant) {
                let node_len = node.text.read().unwrap().chars().count();
                self.tenant_char_count
                    .entry(tenant.clone())
                    .and_modify(|count| {
                        *count = count.saturating_sub(node_len);
                    });
            }

            // Remove tenant from node
            node.tenant_last_access_time.remove(&tenant);

            // Remove empty nodes
            if node.children.is_empty() && node.tenant_last_access_time.is_empty() {
                if let Some(parent) = node.parent.read().unwrap().as_ref() {
                    let text_guard = node.text.read().unwrap();
                    if let Some(first_char) = text_guard.chars().next() {
                        parent.children.remove(&first_char);
                    }
                }
            }

            // Add parent to queue if it becomes a leaf
            if let Some(parent) = node.parent.read().unwrap().as_ref() {
                if Tree::leaf_of(parent).contains(&tenant) {
                    if let Some(timestamp) = parent.tenant_last_access_time.get(&tenant) {
                        pq.push(Reverse(EvictionEntry {
                            timestamp: *timestamp,
                            tenant: tenant.clone(),
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
        // 1. Find all the leaves for the tenant
        let mut stack = vec![Arc::clone(&self.root)];
        let mut queue = VecDeque::new();

        while let Some(curr) = stack.pop() {
            for child in curr.children.iter() {
                stack.push(Arc::clone(child.value()));
            }

            if Tree::leaf_of(&curr).contains(&tenant.to_string()) {
                queue.push_back(Arc::clone(&curr));
            }
        }

        // 2. Start from the leaves and traverse up to the root, removing the tenant from each node
        while let Some(curr) = queue.pop_front() {
            // remove tenant from node
            curr.tenant_last_access_time.remove(&tenant.to_string());

            // remove empty nodes
            if curr.children.is_empty() && curr.tenant_last_access_time.is_empty() {
                if let Some(parent) = curr.parent.read().unwrap().as_ref() {
                    let text_guard = curr.text.read().unwrap();
                    if let Some(first_char) = text_guard.chars().next() {
                        parent.children.remove(&first_char);
                    }
                }
            }

            // add parent to queue if it becomes a leaf
            if let Some(parent) = curr.parent.read().unwrap().as_ref() {
                if Tree::leaf_of(parent).contains(&tenant.to_string()) {
                    queue.push_back(Arc::clone(parent));
                }
            }
        }

        // 3. Remove the tenant from the tenant_char_count map
        self.tenant_char_count.remove(&tenant.to_string());
    }

    pub fn get_tenant_char_count(&self) -> HashMap<String, usize> {
        self.tenant_char_count
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    pub fn get_smallest_tenant(&self) -> String {
        // Return a placeholder if there are no tenants
        if self.tenant_char_count.is_empty() {
            return "empty".to_string();
        }

        // Find the tenant with minimum char count
        let mut min_tenant = None;
        let mut min_count = usize::MAX;

        for entry in self.tenant_char_count.iter() {
            let tenant = entry.key();
            let count = *entry.value();

            if count < min_count {
                min_count = count;
                min_tenant = Some(tenant.clone());
            }
        }

        // Return the found tenant or "empty" if somehow none was found
        min_tenant.unwrap_or_else(|| "empty".to_string())
    }

    pub fn get_used_size_per_tenant(&self) -> HashMap<String, usize> {
        // perform a DFS to traverse all nodes and calculate the total size used by each tenant

        let mut used_size_per_tenant: HashMap<String, usize> = HashMap::new();
        let mut stack = vec![Arc::clone(&self.root)];

        while let Some(curr) = stack.pop() {
            let text_count = curr.text.read().unwrap().chars().count();

            for tenant in curr.tenant_last_access_time.iter() {
                let size = used_size_per_tenant
                    .entry(tenant.key().clone())
                    .or_insert(0);
                *size += text_count;
            }

            for child in curr.children.iter() {
                stack.push(Arc::clone(child.value()));
            }
        }

        used_size_per_tenant
    }

    fn node_to_string(node: &NodeRef, prefix: &str, is_last: bool) -> String {
        let mut result = String::new();

        // Add prefix and branch character
        result.push_str(prefix);
        result.push_str(if is_last { "└── " } else { "├── " });

        // Add node text
        let node_text = node.text.read().unwrap();
        result.push_str(&format!("'{}' [", node_text));

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
    use std::{thread, time::Instant};

    use rand::{
        distr::{Alphanumeric, SampleString},
        rng as thread_rng, Rng,
    };

    use super::*;

    #[test]
    fn test_get_smallest_tenant() {
        let tree = Tree::new();

        assert_eq!(tree.get_smallest_tenant(), "empty");

        // Insert data for tenant1 - "ap" + "icot" = 6 chars
        tree.insert("ap", "tenant1");
        tree.insert("icot", "tenant1");

        // Insert data for tenant2 - "cat" = 3 chars
        tree.insert("cat", "tenant2");

        assert_eq!(
            tree.get_smallest_tenant(),
            "tenant2",
            "Expected tenant2 to be smallest with 3 characters."
        );

        // Insert overlapping data for tenant3 and tenant4 to test equal counts
        // tenant3: "do" = 2 chars
        // tenant4: "hi" = 2 chars
        tree.insert("do", "tenant3");
        tree.insert("hi", "tenant4");

        let smallest = tree.get_smallest_tenant();
        assert!(
            smallest == "tenant3" || smallest == "tenant4",
            "Expected either tenant3 or tenant4 (both have 2 characters), got {}",
            smallest
        );

        // Add more text to tenant4 to make it larger
        tree.insert("hello", "tenant4"); // Now tenant4 has "hi" + "hello" = 6 chars

        // Now tenant3 should be smallest (2 chars vs 6 chars for tenant4)
        assert_eq!(
            tree.get_smallest_tenant(),
            "tenant3",
            "Expected tenant3 to be smallest with 2 characters"
        );

        tree.evict_tenant_by_size(3); // This should evict tenants with more than 3 chars

        let post_eviction_smallest = tree.get_smallest_tenant();
        println!("Smallest tenant after eviction: {}", post_eviction_smallest);
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
        let maintained_counts: HashMap<String, usize> = tree
            .tenant_char_count
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

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
        let maintained_counts: HashMap<String, usize> = tree
            .tenant_char_count
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

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
        let maintained_counts: HashMap<String, usize> = tree
            .tenant_char_count
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

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
        let maintained_counts: HashMap<String, usize> = tree
            .tenant_char_count
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

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

        // Single node
        tree.insert("hello", "tenant1");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        assert_eq!(leaves, vec!["tenant1"]);

        // Node with multiple tenants
        tree.insert("hello", "tenant2");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        assert_eq!(leaves.len(), 2);
        assert!(leaves.contains(&"tenant1".to_string()));
        assert!(leaves.contains(&"tenant2".to_string()));

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
