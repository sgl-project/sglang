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
