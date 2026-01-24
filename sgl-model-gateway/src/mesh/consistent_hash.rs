//! Consistent hashing for rate-limit ownership
//!
//! Implements consistent hashing ring to determine K owners (K=1-3) for each rate-limit key.
//! Supports ownership transfer on node failures.

use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, HashSet},
    hash::{Hash, Hasher},
};

/// Number of virtual nodes per physical node (for better distribution)
const VIRTUAL_NODES_PER_NODE: usize = 150;

/// Number of owners (K) for each key
const NUM_OWNERS: usize = 3;

/// Consistent hash ring
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    /// Ring: hash -> node_name
    ring: BTreeMap<u64, String>,
    /// Node -> set of virtual node hashes
    node_hashes: BTreeMap<String, HashSet<u64>>,
}

impl ConsistentHashRing {
    pub fn new() -> Self {
        Self {
            ring: BTreeMap::new(),
            node_hashes: BTreeMap::new(),
        }
    }

    /// Add a node to the ring
    pub fn add_node(&mut self, node_name: &str) {
        if self.node_hashes.contains_key(node_name) {
            // Node already exists
            return;
        }

        let mut hashes = HashSet::new();
        for i in 0..VIRTUAL_NODES_PER_NODE {
            let virtual_node = format!("{}:{}", node_name, i);
            let hash = Self::hash(&virtual_node);
            self.ring.insert(hash, node_name.to_string());
            hashes.insert(hash);
        }
        self.node_hashes.insert(node_name.to_string(), hashes);
    }

    /// Remove a node from the ring
    pub fn remove_node(&mut self, node_name: &str) {
        if let Some(hashes) = self.node_hashes.remove(node_name) {
            for hash in hashes {
                self.ring.remove(&hash);
            }
        }
    }

    /// Get K owners for a key
    pub fn get_owners(&self, key: &str) -> Vec<String> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let key_hash = Self::hash(key);
        let mut owners = Vec::new();
        let mut seen_nodes = HashSet::new();
        let total_unique_nodes = self.node_hashes.len();

        // Find the first node >= key_hash (clockwise)
        let mut iter = self.ring.range(key_hash..);
        while owners.len() < NUM_OWNERS && seen_nodes.len() < total_unique_nodes {
            if let Some((_, node)) = iter.next() {
                if !seen_nodes.contains(node) {
                    owners.push(node.clone());
                    seen_nodes.insert(node.clone());
                }
            } else {
                // Wrap around to the beginning
                iter = self.ring.range(..);
            }
        }

        owners
    }

    /// Check if a node is an owner of a key
    pub fn is_owner(&self, key: &str, node_name: &str) -> bool {
        self.get_owners(key).contains(&node_name.to_string())
    }

    /// Get all nodes in the ring
    pub fn get_nodes(&self) -> Vec<String> {
        self.node_hashes.keys().cloned().collect()
    }

    /// Check if a node exists in the ring
    pub fn has_node(&self, node_name: &str) -> bool {
        self.node_hashes.contains_key(node_name)
    }

    /// Hash a string to u64
    fn hash(s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Update ring with current membership
    pub fn update_membership(&mut self, nodes: &[String]) {
        let current_nodes: HashSet<String> = self.node_hashes.keys().cloned().collect();
        let new_nodes: HashSet<String> = nodes.iter().cloned().collect();

        // Remove nodes that are no longer present
        for node in current_nodes.difference(&new_nodes) {
            self.remove_node(node);
        }

        // Add new nodes
        for node in new_nodes.difference(&current_nodes) {
            self.add_node(node);
        }
    }
}

impl Default for ConsistentHashRing {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_remove_node() {
        let mut ring = ConsistentHashRing::new();
        ring.add_node("node1");
        assert!(ring.has_node("node1"));
        assert_eq!(ring.get_nodes().len(), 1);

        ring.add_node("node2");
        assert_eq!(ring.get_nodes().len(), 2);

        ring.remove_node("node1");
        assert!(!ring.has_node("node1"));
        assert_eq!(ring.get_nodes().len(), 1);
    }

    #[test]
    fn test_get_owners() {
        let mut ring = ConsistentHashRing::new();
        ring.add_node("node1");
        ring.add_node("node2");
        ring.add_node("node3");

        let owners = ring.get_owners("test_key");
        assert_eq!(owners.len(), NUM_OWNERS);
        assert!(owners.iter().all(|n| ring.has_node(n)));
    }

    #[test]
    fn test_is_owner() {
        let mut ring = ConsistentHashRing::new();
        ring.add_node("node1");
        ring.add_node("node2");
        ring.add_node("node3");

        let owners = ring.get_owners("test_key");
        for owner in &owners {
            assert!(ring.is_owner("test_key", owner));
        }
    }

    #[test]
    fn test_update_membership() {
        let mut ring = ConsistentHashRing::new();
        ring.add_node("node1");
        ring.add_node("node2");

        ring.update_membership(&["node2".to_string(), "node3".to_string()]);
        assert!(!ring.has_node("node1"));
        assert!(ring.has_node("node2"));
        assert!(ring.has_node("node3"));
    }

    #[test]
    fn test_get_owners_with_fewer_nodes_than_owners() {
        // Test that the loop terminates correctly when there are fewer nodes than NUM_OWNERS
        let mut ring = ConsistentHashRing::new();
        ring.add_node("node1");
        ring.add_node("node2");
        // Only 2 nodes, but NUM_OWNERS is 3

        let owners = ring.get_owners("test_key");
        // Should return all available nodes (2) without infinite loop
        assert_eq!(owners.len(), 2);
        assert!(owners.contains(&"node1".to_string()));
        assert!(owners.contains(&"node2".to_string()));
    }

    #[test]
    fn test_get_owners_with_single_node() {
        // Test with only one node
        let mut ring = ConsistentHashRing::new();
        ring.add_node("node1");

        let owners = ring.get_owners("test_key");
        // Should return the single node without infinite loop
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0], "node1");
    }
}
