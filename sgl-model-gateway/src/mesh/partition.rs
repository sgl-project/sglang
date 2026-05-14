//! Partition detection and handling
//!
//! Detects network partitions and handles state isolation and recovery

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use tracing::warn;

use super::gossip::{NodeState, NodeStatus};

/// Partition detection configuration
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Timeout for considering a node unreachable (seconds)
    pub unreachable_timeout: Duration,
    /// Minimum cluster size to consider a partition
    pub min_cluster_size: usize,
    /// Quorum threshold (minimum nodes needed for quorum)
    pub quorum_threshold: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            unreachable_timeout: Duration::from_secs(30),
            min_cluster_size: 3,
            quorum_threshold: 2,
        }
    }
}

/// Partition state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionState {
    /// Normal operation, no partition detected
    Normal,
    /// Partition detected, but we have quorum
    PartitionedWithQuorum,
    /// Partition detected, we don't have quorum
    PartitionedWithoutQuorum,
}

/// Partition detector
#[derive(Debug)]
pub struct PartitionDetector {
    config: PartitionConfig,
    last_seen: Arc<RwLock<BTreeMap<String, Instant>>>,
    current_state: Arc<RwLock<PartitionState>>,
}

impl PartitionDetector {
    pub fn new(config: PartitionConfig) -> Self {
        Self {
            config,
            last_seen: Arc::new(RwLock::new(BTreeMap::new())),
            current_state: Arc::new(RwLock::new(PartitionState::Normal)),
        }
    }

    /// Update last seen time for a node
    pub fn update_last_seen(&self, node_name: &str) {
        let mut last_seen = self.last_seen.write();
        last_seen.insert(node_name.to_string(), Instant::now());
    }

    /// Detect partition based on current cluster state
    pub fn detect_partition(&self, cluster_state: &BTreeMap<String, NodeState>) -> PartitionState {
        let now = Instant::now();
        let last_seen = self.last_seen.read();

        // Count alive nodes and unreachable nodes
        let mut alive_count = 0;
        let mut unreachable_count = 0;
        let mut reachable_nodes = HashSet::new();

        for (name, node) in cluster_state.iter() {
            if node.status == NodeStatus::Alive as i32 {
                alive_count += 1;

                // Check if we've seen this node recently
                if let Some(last_seen_time) = last_seen.get(name) {
                    if now.duration_since(*last_seen_time) < self.config.unreachable_timeout {
                        reachable_nodes.insert(name.clone());
                    } else {
                        unreachable_count += 1;
                        warn!(
                            "Node {} unreachable for {:?}",
                            name,
                            now.duration_since(*last_seen_time)
                        );
                    }
                } else {
                    // New node, consider it reachable for now
                    reachable_nodes.insert(name.clone());
                }
            }
        }

        let reachable_count = reachable_nodes.len();

        // Determine partition state
        let state = if unreachable_count == 0 {
            PartitionState::Normal
        } else if reachable_count >= self.config.quorum_threshold {
            PartitionState::PartitionedWithQuorum
        } else {
            PartitionState::PartitionedWithoutQuorum
        };

        // Update current state
        *self.current_state.write() = state.clone();

        if state != PartitionState::Normal {
            warn!(
                "Partition detected: state={:?}, reachable={}, unreachable={}, total_alive={}",
                state, reachable_count, unreachable_count, alive_count
            );
        }

        state
    }

    /// Get current partition state
    pub fn current_state(&self) -> PartitionState {
        self.current_state.read().clone()
    }

    /// Check if we have quorum
    pub fn has_quorum(&self, reachable_count: usize) -> bool {
        reachable_count >= self.config.quorum_threshold
    }

    /// Get unreachable nodes
    pub fn get_unreachable_nodes(
        &self,
        cluster_state: &BTreeMap<String, NodeState>,
    ) -> Vec<String> {
        let now = Instant::now();
        let last_seen = self.last_seen.read();
        let mut unreachable = Vec::new();

        for (name, node) in cluster_state.iter() {
            if node.status == NodeStatus::Alive as i32 {
                if let Some(last_seen_time) = last_seen.get(name) {
                    if now.duration_since(*last_seen_time) >= self.config.unreachable_timeout {
                        unreachable.push(name.clone());
                    }
                }
            }
        }

        unreachable
    }

    /// Check if we should continue serving (have quorum)
    pub fn should_serve(&self) -> bool {
        let state = self.current_state.read();
        matches!(
            *state,
            PartitionState::Normal | PartitionState::PartitionedWithQuorum
        )
    }
}

impl Default for PartitionDetector {
    fn default() -> Self {
        Self::new(PartitionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, time::Duration};

    use super::*;
    // Import NodeState and NodeStatus from gossip module
    use crate::mesh::service::gossip::{NodeState, NodeStatus};

    fn create_test_config() -> PartitionConfig {
        PartitionConfig {
            unreachable_timeout: Duration::from_millis(100),
            min_cluster_size: 3,
            quorum_threshold: 2,
        }
    }

    fn create_node_state(name: &str, address: &str, status: NodeStatus) -> NodeState {
        NodeState {
            name: name.to_string(),
            address: address.to_string(),
            status: status as i32,
            version: 1,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_partition_config_default() {
        let config = PartitionConfig::default();
        assert_eq!(config.unreachable_timeout, Duration::from_secs(30));
        assert_eq!(config.min_cluster_size, 3);
        assert_eq!(config.quorum_threshold, 2);
    }

    #[test]
    fn test_partition_detector_initial_state() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        assert_eq!(detector.current_state(), PartitionState::Normal);
        assert!(detector.should_serve());
    }

    #[test]
    fn test_update_last_seen() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        detector.update_last_seen("node1");
        detector.update_last_seen("node2");

        // Verify nodes are tracked
        let cluster_state = BTreeMap::new();
        let state = detector.detect_partition(&cluster_state);
        assert_eq!(state, PartitionState::Normal);
    }

    #[test]
    fn test_detect_partition_normal() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        let mut cluster_state = BTreeMap::new();
        cluster_state.insert(
            "node1".to_string(),
            create_node_state("node1", "127.0.0.1:8080", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node2".to_string(),
            create_node_state("node2", "127.0.0.1:8081", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node3".to_string(),
            create_node_state("node3", "127.0.0.1:8082", NodeStatus::Alive),
        );

        // Update last seen for all nodes
        detector.update_last_seen("node1");
        detector.update_last_seen("node2");
        detector.update_last_seen("node3");

        let state = detector.detect_partition(&cluster_state);
        assert_eq!(state, PartitionState::Normal);
        assert!(detector.should_serve());
    }

    #[test]
    fn test_detect_partition_with_quorum() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        let mut cluster_state = BTreeMap::new();
        cluster_state.insert(
            "node1".to_string(),
            create_node_state("node1", "127.0.0.1:8080", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node2".to_string(),
            create_node_state("node2", "127.0.0.1:8081", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node3".to_string(),
            create_node_state("node3", "127.0.0.1:8082", NodeStatus::Alive),
        );

        // Update last seen for node1 and node2 (quorum)
        detector.update_last_seen("node1");
        detector.update_last_seen("node2");

        // Don't update node3, but wait for it to be considered unreachable
        // Since node3 is new, it's initially considered reachable
        // We need to update it first, then wait for timeout
        detector.update_last_seen("node3");
        std::thread::sleep(Duration::from_millis(150));

        // Update node1 and node2 again to keep them reachable
        detector.update_last_seen("node1");
        detector.update_last_seen("node2");

        let state = detector.detect_partition(&cluster_state);
        // node1 and node2 are still reachable (quorum of 2), node3 is unreachable
        assert_eq!(state, PartitionState::PartitionedWithQuorum);
        assert!(detector.should_serve());
    }

    #[test]
    fn test_detect_partition_without_quorum() {
        let mut config = create_test_config();
        config.quorum_threshold = 2;
        let detector = PartitionDetector::new(config);

        let mut cluster_state = BTreeMap::new();
        cluster_state.insert(
            "node1".to_string(),
            create_node_state("node1", "127.0.0.1:8080", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node2".to_string(),
            create_node_state("node2", "127.0.0.1:8081", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node3".to_string(),
            create_node_state("node3", "127.0.0.1:8082", NodeStatus::Alive),
        );

        // Update last seen for all nodes first
        detector.update_last_seen("node1");
        detector.update_last_seen("node2");
        detector.update_last_seen("node3");

        // Wait for node2 and node3 to become unreachable
        std::thread::sleep(Duration::from_millis(150));

        // Only update node1 again to keep it reachable
        detector.update_last_seen("node1");

        let state = detector.detect_partition(&cluster_state);
        // Only node1 is reachable (below quorum of 2)
        assert_eq!(state, PartitionState::PartitionedWithoutQuorum);
        assert!(!detector.should_serve());
    }

    #[test]
    fn test_has_quorum() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        assert!(detector.has_quorum(2));
        assert!(detector.has_quorum(3));
        assert!(!detector.has_quorum(1));
        assert!(!detector.has_quorum(0));
    }

    #[test]
    fn test_get_unreachable_nodes() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        let mut cluster_state = BTreeMap::new();
        cluster_state.insert(
            "node1".to_string(),
            create_node_state("node1", "127.0.0.1:8080", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node2".to_string(),
            create_node_state("node2", "127.0.0.1:8081", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node3".to_string(),
            create_node_state("node3", "127.0.0.1:8082", NodeStatus::Alive),
        );

        // Update last seen for all nodes
        detector.update_last_seen("node1");
        detector.update_last_seen("node2");
        detector.update_last_seen("node3");

        // Initially no unreachable nodes
        let unreachable = detector.get_unreachable_nodes(&cluster_state);
        assert!(unreachable.is_empty());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));

        // All nodes should be unreachable now
        let unreachable = detector.get_unreachable_nodes(&cluster_state);
        assert_eq!(unreachable.len(), 3);
        assert!(unreachable.contains(&"node1".to_string()));
        assert!(unreachable.contains(&"node2".to_string()));
        assert!(unreachable.contains(&"node3".to_string()));
    }

    #[test]
    fn test_get_unreachable_nodes_with_recent_updates() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        let mut cluster_state = BTreeMap::new();
        cluster_state.insert(
            "node1".to_string(),
            create_node_state("node1", "127.0.0.1:8080", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node2".to_string(),
            create_node_state("node2", "127.0.0.1:8081", NodeStatus::Alive),
        );

        // Update node1 first (old)
        detector.update_last_seen("node1");
        std::thread::sleep(Duration::from_millis(50));

        // Update node2 later (more recent)
        detector.update_last_seen("node2");

        // Wait for node1 to timeout but node2 should still be reachable
        std::thread::sleep(Duration::from_millis(60));

        let unreachable = detector.get_unreachable_nodes(&cluster_state);
        // node1 should be unreachable (updated 110ms ago), node2 should still be reachable (updated 60ms ago)
        assert!(unreachable.contains(&"node1".to_string()));
        assert!(!unreachable.contains(&"node2".to_string()));
    }

    #[test]
    fn test_detect_partition_ignores_non_alive_nodes() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        let mut cluster_state = BTreeMap::new();
        cluster_state.insert(
            "node1".to_string(),
            create_node_state("node1", "127.0.0.1:8080", NodeStatus::Alive),
        );
        cluster_state.insert(
            "node2".to_string(),
            create_node_state("node2", "127.0.0.1:8081", NodeStatus::Down),
        );
        cluster_state.insert(
            "node3".to_string(),
            create_node_state("node3", "127.0.0.1:8082", NodeStatus::Suspected),
        );

        detector.update_last_seen("node1");

        let state = detector.detect_partition(&cluster_state);
        // Only node1 is alive and reachable
        // Since node2 and node3 are not alive, they don't count as unreachable
        // If all alive nodes are reachable (unreachable_count == 0), state is Normal
        assert_eq!(state, PartitionState::Normal);
    }

    #[test]
    fn test_new_node_considered_reachable() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        let mut cluster_state = BTreeMap::new();
        cluster_state.insert(
            "node1".to_string(),
            create_node_state("node1", "127.0.0.1:8080", NodeStatus::Alive),
        );
        cluster_state.insert(
            "new_node".to_string(),
            create_node_state("new_node", "127.0.0.1:8083", NodeStatus::Alive),
        );

        // Don't update last_seen for new_node, it should be considered reachable
        detector.update_last_seen("node1");

        let state = detector.detect_partition(&cluster_state);
        // Both nodes should be considered reachable (node1 explicitly, new_node by default)
        assert_eq!(state, PartitionState::Normal);
    }

    #[test]
    fn test_should_serve() {
        let config = create_test_config();
        let detector = PartitionDetector::new(config);

        // Normal state should serve
        *detector.current_state.write() = PartitionState::Normal;
        assert!(detector.should_serve());

        // Partitioned with quorum should serve
        *detector.current_state.write() = PartitionState::PartitionedWithQuorum;
        assert!(detector.should_serve());

        // Partitioned without quorum should not serve
        *detector.current_state.write() = PartitionState::PartitionedWithoutQuorum;
        assert!(!detector.should_serve());
    }

    #[test]
    fn test_default_implementation() {
        let detector = PartitionDetector::default();
        assert_eq!(detector.current_state(), PartitionState::Normal);
        assert!(detector.should_serve());
    }

    #[test]
    fn test_partition_state_equality() {
        assert_eq!(PartitionState::Normal, PartitionState::Normal);
        assert_ne!(
            PartitionState::Normal,
            PartitionState::PartitionedWithQuorum
        );
        assert_ne!(
            PartitionState::PartitionedWithQuorum,
            PartitionState::PartitionedWithoutQuorum
        );
    }
}
