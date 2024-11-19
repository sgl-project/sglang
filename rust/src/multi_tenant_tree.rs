use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use dashmap::DashMap;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use dashmap::mapref::entry::Entry;
use rand::distributions::{Alphanumeric, DistString};
use rand::thread_rng;

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
}


fn shared_prefix_length(a: &str, b: &str) -> usize {
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

    return i;
}

// Method 1: Using chars().take() and skip()
fn slice_by_chars(s: &str, start: usize, end: usize) -> String {
    s.chars()
        .skip(start)
        .take(end - start)
        .collect()
}

struct EvictionEntry {
    timestamp: u128,
    tenant: String,
    node: NodeRef
}

impl Eq for EvictionEntry {}

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

impl Tree {
    pub fn new() -> Self {
        Tree {
            root: Arc::new(Node {
                children: DashMap::new(),
                text: RwLock::new("".to_string()),
                tenant_last_access_time: DashMap::new(),
                parent: RwLock::new(None),
            })
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
        curr.tenant_last_access_time.insert(tenant.to_string(), timestamp_ms);

        let mut prev = Arc::clone(&self.root);

        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();

            curr = prev;

            // dashmap.entry locks the entry until the op is done
            // if using contains_key + insert, there will be an issue that
            // 1. "apple" and "app" entered at the same time
            // 2. and get inserted to the dashmap concurrently, so only one is inserted
            // we want to ensure when the are visiting a key in the dashmap, the entry is locked

            match curr.children.entry(first_char) {
                Entry::Vacant(entry) => {
                    // no matched
                    // [curr]
                    // [curr] => [new node]
                    let curr_text = slice_by_chars(text, curr_idx, text_count);

                    let new_node = Arc::new(Node {
                        children: DashMap::new(),
                        text: RwLock::new(curr_text),
                        tenant_last_access_time: DashMap::new(),
                        parent: RwLock::new(Some(Arc::clone(&curr))),
                    });
                    new_node.tenant_last_access_time.insert(tenant.to_string(), timestamp_ms);

                    entry.insert(Arc::clone(&new_node));

                    prev = Arc::clone(&new_node);
                    // prev.tenant_last_access_time.insert(tenant.to_string(), timestamp_ms);
                    curr_idx = text_count;
                }

                Entry::Occupied(mut entry) => {
                    // matched
                    let matched_node = entry.get().clone();

                    // this can cause deadlock because maybe matched_node is holding the read lock of curr.children, so we can not acquire the write lock on it
                    // let mut matched_node = curr.children.get_mut(&first_char).unwrap();

                    let matched_node_text = matched_node.text.read().unwrap().to_owned();
                    let matched_node_text_count = matched_node_text.chars().count();

                    let curr_text = slice_by_chars(text, curr_idx, text_count);

                    let shared_len = shared_prefix_length(
                        &matched_node_text,
                        &curr_text,
                    );

                    if shared_len < matched_node_text_count {
                        // split the matched node
                        // Split structure: [curr] -> [matched_node] =>
                        //                  [curr] -> [new_node] -> [contracted_matched_node]

                        let matched_text = slice_by_chars(&matched_node_text, 0, shared_len);// matched_node_text[..shared_len].to_string();
                        let new_text = slice_by_chars(&matched_node_text, shared_len, matched_node_text_count); // matched_node_text[shared_len..].to_string();

                        let new_node = Arc::new(Node {
                            text: RwLock::new(matched_text),
                            children: DashMap::new(),
                            parent: RwLock::new(Some(Arc::clone(&curr))),
                            tenant_last_access_time: matched_node.tenant_last_access_time.clone(),
                        });


                        let first_new_char = new_text.chars().nth(0).unwrap();
                        new_node.children.insert(first_new_char, Arc::clone(&matched_node));


                        // println!("before deadlock");
                        // // deadlock here!?
                        entry.insert(Arc::clone(&new_node));
                        // println!("after deadlock");

                        let mut matched_node_text = matched_node.text.write().unwrap();
                        *matched_node_text = new_text;
                        // To modify data inside an Arc, you need synchronization primitives like Mutex or RwLock
                        let mut matched_node_parent = matched_node.parent.write().unwrap();
                        *matched_node_parent = Some(Arc::clone(&new_node));

                        prev = Arc::clone(&new_node);
                        prev.tenant_last_access_time.insert(tenant.to_string(), timestamp_ms);
                        curr_idx += shared_len;
                    } else {
                        // move to next node
                        prev = Arc::clone(&matched_node);
                        prev.tenant_last_access_time.insert(tenant.to_string(), timestamp_ms);
                        curr_idx += shared_len;
                    }
                }
            }
        }


    }

    pub fn prefix_match(&self, text: &str) -> (String, String) {
        let mut curr = Arc::clone(&self.root);
        let mut curr_idx = 0;
        let mut prev = Arc::clone(&self.root);
        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();
            curr = prev.clone();
            let curr_text = slice_by_chars(text, curr_idx, text_count);

            match curr.children.entry(first_char) {
                Entry::Occupied(entry) => {
                    let matched_node = entry.get().clone();
                    let shared_len = shared_prefix_length(
                        &matched_node.text.read().unwrap(),
                        &curr_text,
                    );

                    let matched_node_text_count = matched_node.text.read().unwrap().chars().count();

                    if shared_len == matched_node_text_count {
                        // Full match with current node's text, continue to next node
                        curr_idx += shared_len;
                        prev = Arc::clone(&matched_node);
                    } else {
                        // Partial match, stop here
                        curr_idx += shared_len;
                        prev = Arc::clone(&matched_node);
                        break;
                    }
                }
                Entry::Vacant(_) => {
                    // No match found, stop here
                    break;
                }
            }
        }

        curr = prev.clone();

        // Randomly select the first tenant (key in the map)
        let tenant = curr.tenant_last_access_time
            .iter()
            .next()
            .map(|kv| kv.key().to_owned())
            .unwrap_or("empty".to_string());

        // TODO traverse from the curr node to the root and update the timestamp

        // Update timestamps from current node to root

        // Get current timestamp
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        if !tenant.eq("empty") {
            let mut current_node = Some(curr);
            while let Some(node) = current_node {
                // Update timestamp for the tenant in current node
                node.tenant_last_access_time.insert(tenant.clone(), timestamp_ms);

                // Move to parent node
                current_node = node.parent.read().unwrap().clone();
            }
        }


        let ret_text = slice_by_chars(text, 0, curr_idx);
        (ret_text, tenant)
    }

    fn leaf_of(node: &NodeRef) -> Vec<String> {
        let mut candidates: HashMap<String, bool> = node.tenant_last_access_time
            .iter()
            .map(|entry| (entry.key().clone(), true))
            .collect();

        for child in node.children.iter() {
            for tenant in child.value().tenant_last_access_time.iter() {
                candidates.insert(tenant.key().clone(), false);
            }
        }

        candidates.into_iter()
            .filter(|(_, is_leaf)| *is_leaf)
            .map(|(tenant, _)| tenant)
            .collect()
    }

    pub fn evict_tenant_data(&self, max_size: usize) {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        // Calculate used size and collect leaves
        let mut stack = vec![Arc::clone(&self.root)];
        let mut used_size_per_tenant: HashMap<String, usize> = HashMap::new();
        let mut pq = BinaryHeap::new();

        while let Some(curr) = stack.pop() {
            for tenant in curr.tenant_last_access_time.iter() {
                let size = used_size_per_tenant
                    .entry(tenant.key().clone())
                    .or_insert(0);
                *size += curr.text.read().unwrap().chars().count();
            }

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

        println!("Before eviction - Used size per tenant:");
        for (tenant, size) in &used_size_per_tenant {
            println!("Tenant: {}, Size: {}", tenant, size);
        }

        // Process eviction
        while let Some(Reverse(entry)) = pq.pop() {
            let EvictionEntry { timestamp, tenant, node } = entry;

            if let Some(&used_size) = used_size_per_tenant.get(&tenant) {
                if used_size <= max_size {
                    continue;
                }

                // Update used size
                if let Some(size) = used_size_per_tenant.get_mut(&tenant) {
                    *size -= node.text.read().unwrap().chars().count();
                }

                let node_text = node.text.read().unwrap().clone();
                // println!("Evicting - Node: '{}', Tenant: {}, Size: {}",
                //     node_text, tenant, node_text.chars().count());

                // Remove tenant from node
                node.tenant_last_access_time.remove(&tenant);

                // Remove empty nodes
                if node.children.is_empty() && node.tenant_last_access_time.is_empty() {
                    if let Some(parent) = node.parent.write().unwrap().as_ref() {
                        let first_char = node.text.read().unwrap().chars().next().unwrap();
                        parent.children.remove(&first_char);
                        // println!("Removing empty node: '{}'", node_text);
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
                }
            }
        }

        println!("\nAfter eviction - Used size per tenant:");
        for (tenant, size) in &used_size_per_tenant {
            println!("Tenant: {}, Size: {}", tenant, size);
        }
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

            tenant_info.push(format!("{} | {:02}:{:02}:{:02}.{:03}",
                tenant_id, hours, minutes, seconds, millis));
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
                is_last_child
            ));
        }

        result
    }

    pub fn pretty_print(&self) {



        if self.root.children.is_empty() {
            return
        }

        let mut result = String::new();
        let children: Vec<_> = self.root.children.iter().collect();
        let child_count = children.len();

        for (i, entry) in children.iter().enumerate() {
            let is_last = i == child_count - 1;
            result.push_str(&Tree::node_to_string(
                entry.value(),
                "",
                is_last
            ));
        }


        println!("{result}");

        return
    }
}



#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::Rng;

    use super::*;

    // #[test]
    // fn test_simple_eviction() {
    //     let tree = Tree::new();
    //     let max_size = 5;

    //     // Insert strings for both tenants
    //     tree.insert("hello", "tenant1");  // size 5
    //     tree.insert("hello", "tenant2");  // size 5
    //     tree.insert("world", "tenant2");  // size 5, total for tenant2 = 10

    //     tree.pretty_print();

    //     // Verify initial sizes
    //     let sizes_before = tree.get_used_size_per_tenant();
    //     assert_eq!(sizes_before.get("tenant1").unwrap(), &5);  // "hello" = 5
    //     assert_eq!(sizes_before.get("tenant2").unwrap(), &10); // "hello" + "world" = 10

    //     // Evict - should remove "hello" from tenant2 as it's the oldest
    //     tree.evict_tenant_data(max_size);

    //     tree.pretty_print();

    //     // Verify sizes after eviction
    //     let sizes_after = tree.get_used_size_per_tenant();
    //     assert_eq!(sizes_after.get("tenant1").unwrap(), &5);  // Should be unchanged
    //     assert_eq!(sizes_after.get("tenant2").unwrap(), &5);  // Only "world" remains

    //     // Verify "world" remains for tenant2
    //     let (matched, tenant) = tree.prefix_match("world");
    //     assert_eq!(matched, "world");
    //     assert_eq!(tenant, "tenant2");
    // }

    fn random_string(len: usize) -> String {
        // Method 1: Using Alphanumeric distribution's generate method
        Alphanumeric.sample_string(&mut thread_rng(), len)
    }

    // #[test]
    // fn test_advanced_eviction() {
    //     let tree = Tree::new();

    //     // Set limits for each tenant
    //     let max_size: usize = 100;

    //     // Define prefixes
    //     let prefixes = vec![
    //         "aqwefcisdf",
    //         "iajsdfkmade",
    //         "kjnzxcvewqe",
    //         "iejksduqasd"
    //     ];

    //     // Insert strings with shared prefixes
    //     for i in 0..100 {
    //         for (j, prefix) in prefixes.iter().enumerate() {
    //             let random_suffix = random_string(10);
    //             let text = format!("{}{}", prefix, random_suffix);
    //             let tenant = format!("tenant{}", j + 1);
    //             tree.insert(&text, &tenant);
    //         }
    //     }

    //     // Perform eviction
    //     tree.evict_tenant_data(max_size);

    //     // Check sizes after eviction
    //     let sizes_after = tree.get_used_size_per_tenant();
    //     // Verify all tenants are under their size limits
    //     for (tenant, &size) in sizes_after.iter() {
    //         assert!(
    //             size <= max_size,
    //             "Tenant {} exceeds size limit. Current size: {}, Limit: {}",
    //             tenant,
    //             size,
    //             max_size
    //         );
    //     }
    // }

    #[test]
    fn test_concurrent_operations_with_eviction() {
        let tree = Arc::new(Tree::new());
        let mut handles = vec![];
        let test_duration = Duration::from_secs(60);
        let start_time = Instant::now();
        let max_size = 100;  // Single max size for all tenants

        // Spawn eviction thread
        {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                while start_time.elapsed() < test_duration {
                    // Run eviction
                    tree.evict_tenant_data(max_size);

                    // Sleep for 5 seconds
                    thread::sleep(Duration::from_secs(5));

                    // Print current sizes
                    let sizes = tree.get_used_size_per_tenant();
                    // println!("Current sizes: {:?}", sizes);
                    // tree.pretty_print();
                }
            });
            handles.push(handle);
        }

        // Spawn 4 worker threads
        for thread_id in 0..4 {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();
                let tenant = format!("tenant{}", thread_id + 1);
                let prefix = format!("prefix{}", thread_id);

                while start_time.elapsed() < test_duration {
                    // Random decision: match or insert (70% match, 30% insert)
                    if rng.gen_bool(0.7) {
                        // Perform match operation
                        let random_len = rng.gen_range(3..10);
                        let search_str = format!("{}{}", prefix, random_string(random_len));
                        let (matched, _) = tree.prefix_match(&search_str);
                        // println!("Thread {} matched: {}", thread_id, matched);
                    } else {
                        // Perform insert operation
                        let random_len = rng.gen_range(5..15);
                        let insert_str = format!("{}{}", prefix, random_string(random_len));
                        tree.insert(&insert_str, &tenant);
                        // println!("Thread {} inserted: {}", thread_id, insert_str);
                    }

                    // Small random sleep to vary timing
                    thread::sleep(Duration::from_millis(rng.gen_range(10..100)));
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // final eviction
        tree.evict_tenant_data(max_size);

        // Final size check
        let final_sizes = tree.get_used_size_per_tenant();
        println!("Final sizes after test completion: {:?}", final_sizes);

        // Verify all tenants are under limit
        for (_, &size) in final_sizes.iter() {
            assert!(
                size <= max_size,
                "Tenant exceeds size limit. Final size: {}, Limit: {}",
                size,
                max_size
            );
        }
    }

    // #[test]
    // fn test_crash_ram_primitive() {
    //     println!("Starting memory consumption...");

    //     let mut vector = Vec::new();
    //     loop {
    //         // Allocate a full gigabyte in each iteration
    //         vector.push(vec![0u8; 1024 * 1024 * 1024]);  // 1 GB
    //         println!("Memory allocated: {} GB", vector.len());
    //         // Add a delay to slow down allocation and reduce CPU usage
    //         // thread::sleep(Duration::from_secs(1));
    //     }
    // }

    // #[test]
    // fn test_crash_tree() {
    //     let tree = Arc::new(Tree::new());
    //     let mut handles = vec![];

    //     for _ in 0..16 {
    //         let tree_clone = Arc::clone(&tree);

    //         let handle = thread::spawn(move || {
    //             let mut idx = 0;
    //             loop {
    //                 let text = random_string(1024 * 1024);  // 1MB
    //                 let tenant = "main_tenant";

    //                 tree_clone.insert(&text, tenant);

    //                 if idx % 100 == 0 {
    //                     let used_size_map = tree_clone.get_used_size_per_tenant();
    //                     println!("Used size: {:?}", used_size_map);
    //                 }
    //                 idx += 1;
    //                 thread::sleep(Duration::from_millis(1));
    //             }


    //         });
    //         handles.push(handle);
    //     }

    //     for handle in handles {
    //         handle.join().unwrap();
    //     }
    // }

    // #[test]
    // fn test_leaf_of() {
    //     let tree = Tree::new();

    //     // Single node
    //     tree.insert("hello", "tenant1");
    //     let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
    //     assert_eq!(leaves, vec!["tenant1"]);

    //     // Node with multiple tenants
    //     tree.insert("hello", "tenant2");
    //     let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
    //     assert_eq!(leaves.len(), 2);
    //     assert!(leaves.contains(&"tenant1".to_string()));
    //     assert!(leaves.contains(&"tenant2".to_string()));

    //     // Non-leaf node
    //     tree.insert("hi", "tenant1");
    //     let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
    //     assert!(leaves.is_empty());
    // }

    // #[test]
    // fn test_get_used_size_per_tenant() {
    //     let tree = Tree::new();

    //     // Single tenant
    //     tree.insert("hello", "tenant1");
    //     tree.insert("world", "tenant1");
    //     let sizes = tree.get_used_size_per_tenant();

    //     tree.pretty_print();
    //     println!("{:?}", sizes);
    //     assert_eq!(sizes.get("tenant1").unwrap(), &10); // "hello" + "world"

    //     // Multiple tenants sharing nodes
    //     tree.insert("hello", "tenant2");
    //     tree.insert("help", "tenant2");
    //     let sizes = tree.get_used_size_per_tenant();

    //     tree.pretty_print();
    //     println!("{:?}", sizes);
    //     assert_eq!(sizes.get("tenant1").unwrap(), &10);
    //     assert_eq!(sizes.get("tenant2").unwrap(), &6); // "hello" + "p"

    //     // UTF-8 characters
    //     tree.insert("你好", "tenant3");
    //     let sizes = tree.get_used_size_per_tenant();
    //     tree.pretty_print();
    //     println!("{:?}", sizes);
    //     assert_eq!(sizes.get("tenant3").unwrap(), &2); // 2 Chinese characters

    //     tree.pretty_print();
    // }

    // #[test]
    // fn test_cold_start() {
    //     let tree = Tree::new();


    //     let (matched_text, tenant) = tree.prefix_match("hello");

    //     assert_eq!(matched_text, "");
    //     assert_eq!(tenant, "empty");
    // }


    // #[test]
    // fn test_exact_match_seq() {
    //     let tree = Tree::new();
    //     tree.insert("hello", "tenant1");
    //     tree.pretty_print();
    //     tree.insert("apple", "tenant2");
    //     tree.pretty_print();
    //     tree.insert("banana", "tenant3");
    //     tree.pretty_print();

    //     let (matched_text, tenant) = tree.prefix_match("hello");
    //     assert_eq!(matched_text, "hello");
    //     assert_eq!(tenant, "tenant1");

    //     let (matched_text, tenant) = tree.prefix_match("apple");
    //     assert_eq!(matched_text, "apple");
    //     assert_eq!(tenant, "tenant2");

    //     let (matched_text, tenant) = tree.prefix_match("banana");
    //     assert_eq!(matched_text, "banana");
    //     assert_eq!(tenant, "tenant3");
    // }

    // #[test]
    // fn test_exact_match_concurrent() {

    //     let tree = Arc::new(Tree::new());

    //     // spawn 3 threads for insert
    //     let tree_clone = Arc::clone(&tree);

    //     let texts = vec!["hello", "apple", "banana"];
    //     let tenants = vec!["tenant1", "tenant2", "tenant3"];

    //     let mut handles = vec![];

    //     for i in 0..3 {
    //         let tree_clone = Arc::clone(&tree_clone);
    //         let text = texts[i];
    //         let tenant = tenants[i];

    //         let handle = thread::spawn(move || {
    //             tree_clone.insert(text, tenant);
    //         });

    //         handles.push(handle);
    //     }

    //     // wait
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }

    //     // spawn 3 threads for match
    //     let mut handles = vec![];

    //     let tree_clone = Arc::clone(&tree);

    //     for i in 0..3 {
    //         let tree_clone = Arc::clone(&tree_clone);
    //         let text = texts[i];
    //         let tenant = tenants[i];

    //         let handle = thread::spawn(move || {
    //             let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
    //             assert_eq!(matched_text, text);
    //             assert_eq!(matched_tenant, tenant);
    //         });

    //         handles.push(handle);
    //     }

    //     // wait
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }

    // }

    // #[test]
    // fn test_partial_match_concurrent() {

    //     let tree = Arc::new(Tree::new());

    //     // spawn 3 threads for insert
    //     let tree_clone = Arc::clone(&tree);

    //     let texts = vec!["apple", "apabc", "acbdeds"];

    //     let mut handles = vec![];

    //     for i in 0..3 {
    //         let tree_clone = Arc::clone(&tree_clone);
    //         let text = texts[i];
    //         let tenant = "tenant0";

    //         let handle = thread::spawn(move || {
    //             tree_clone.insert(text, tenant);
    //         });

    //         handles.push(handle);
    //     }

    //     // wait
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }

    //     // spawn 3 threads for match
    //     let mut handles = vec![];

    //     let tree_clone = Arc::clone(&tree);

    //     for i in 0..3 {
    //         let tree_clone = Arc::clone(&tree_clone);
    //         let text = texts[i];
    //         let tenant = "tenant0";

    //         let handle = thread::spawn(move || {
    //             let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
    //             assert_eq!(matched_text, text);
    //             assert_eq!(matched_tenant, tenant);
    //         });

    //         handles.push(handle);
    //     }

    //     // wait
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }

    // }

    // #[test]
    // fn test_group_prefix_insert_match_concurrent() {
    //     let prefix = vec!["Clock strikes midnight, I'm still wide awake", "Got dreams bigger than these city lights", "Time waits for no one, gotta make my move", "Started from the bottom, that's no metaphor"];
    //     let suffix = vec!["Got too much to prove, ain't got time to lose", "History in the making, yeah, you can't erase this"];
    //     let tree = Arc::new(Tree::new());

    //     let mut handles = vec![];

    //     for i in 0..prefix.len() {

    //         for j in 0..suffix.len() {
    //             let tree_clone = Arc::clone(&tree);
    //             let text = format!("{} {}", prefix[i], suffix[j]);
    //             let tenant = format!("tenant{}", i);

    //             let handle = thread::spawn(move || {
    //                 tree_clone.insert(&text, &tenant);
    //             });

    //             handles.push(handle);
    //         }
    //     }

    //     // wait
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }


    //     tree.pretty_print();

    //     // check matching using multi threads

    //     let mut handles = vec![];

    //     for i in 0..prefix.len() {
    //         let tree_clone = Arc::clone(&tree);
    //         let text = prefix[i];

    //         let handle = thread::spawn(move || {
    //             let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
    //             let tenant = format!("tenant{}", i);
    //             assert_eq!(matched_text, text);
    //             assert_eq!(matched_tenant, tenant);
    //         });

    //         handles.push(handle);
    //     }

    //     // wait
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }
    // }


    // #[test]
    // fn test_mixed_concurrent_insert_match() {
    //     // Do not do correctness check but just to ensure it does not deadlock
    //     let prefix = vec!["Clock strikes midnight, I'm still wide awake", "Got dreams bigger than these city lights", "Time waits for no one, gotta make my move", "Started from the bottom, that's no metaphor"];
    //     let suffix = vec!["Got too much to prove, ain't got time to lose", "History in the making, yeah, you can't erase this"];
    //     let tree = Arc::new(Tree::new());

    //     let mut handles = vec![];

    //     for i in 0..prefix.len() {

    //         for j in 0..suffix.len() {
    //             let tree_clone = Arc::clone(&tree);
    //             let text = format!("{} {}", prefix[i], suffix[j]);
    //             let tenant = format!("tenant{}", i);

    //             let handle = thread::spawn(move || {
    //                 tree_clone.insert(&text, &tenant);
    //             });

    //             handles.push(handle);
    //         }
    //     }

    //     // check matching using multi threads

    //     for i in 0..prefix.len() {
    //         let tree_clone = Arc::clone(&tree);
    //         let text = prefix[i];

    //         let handle = thread::spawn(move || {
    //             let (matched_text, matched_tenant) = tree_clone.prefix_match(text);
    //         });

    //         handles.push(handle);
    //     }

    //     // wait
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }
    // }

    // #[test]
    // fn test_utf8_split_seq() {
    //     // The string should be indexed and splitted by a utf-8 value basis instead of byte basis
    //     // use .chars() to get the iterator of the utf-8 value
    //     let tree = Arc::new(Tree::new());


    //     let test_pairs = vec![
    //         ("你好嗎", "tenant1"),
    //         ("你好喔", "tenant2"),
    //         ("你心情好嗎", "tenant3"),
    //     ];

    //     // Insert sequentially
    //     for i in 0..test_pairs.len() {
    //         let text = test_pairs[i].0;
    //         let tenant = test_pairs[i].1;
    //         tree.insert(text, tenant);
    //     }

    //     tree.pretty_print();

    //     // Test sequentially

    //     for i in 0..test_pairs.len() {
    //         let (matched_text, matched_tenant) = tree.prefix_match(test_pairs[i].0);
    //         assert_eq!(matched_text, test_pairs[i].0);
    //         assert_eq!(matched_tenant, test_pairs[i].1);
    //     }

    // }

    // #[test]
    // fn test_utf8_split_concurrent() {
    //     let tree = Arc::new(Tree::new());

    //     let test_pairs = vec![
    //         ("你好嗎", "tenant1"),
    //         ("你好喔", "tenant2"),
    //         ("你心情好嗎", "tenant3"),
    //     ];

    //     // Create multiple threads for insertion
    //     let mut handles = vec![];

    //     for i in 0..test_pairs.len() {
    //         let tree_clone = Arc::clone(&tree);
    //         let text = test_pairs[i].0.to_string();
    //         let tenant = test_pairs[i].1.to_string();

    //         let handle = thread::spawn(move || {
    //             tree_clone.insert(&text, &tenant);
    //         });

    //         handles.push(handle);
    //     }

    //     // Wait for all insertions to complete
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }

    //     tree.pretty_print();

    //     // Create multiple threads for matching
    //     let mut handles = vec![];

    //     for i in 0..test_pairs.len() {
    //         let tree_clone = Arc::clone(&tree);
    //         let text = test_pairs[i].0.to_string();
    //         let tenant = test_pairs[i].1.to_string();

    //         let handle = thread::spawn(move || {
    //             let (matched_text, matched_tenant) = tree_clone.prefix_match(&text);
    //             assert_eq!(matched_text, text);
    //             assert_eq!(matched_tenant, tenant);
    //         });

    //         handles.push(handle);
    //     }

    //     // Wait for all matches to complete
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }
    // }
}
