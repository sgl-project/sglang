//! Topology management for mesh cluster
//!
//! Supports:
//! - Full mesh for small/medium clusters
//! - Sparse mesh for large clusters (by region/AZ)

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use parking_lot::RwLock;
use tracing::debug;

use super::{service::ClusterState, stores::MembershipState};

/// Topology configuration
#[derive(Debug, Clone)]
pub struct TopologyConfig {
    /// Maximum nodes for full mesh (beyond this, use sparse)
    pub full_mesh_threshold: usize,
    /// Region identifier (for sparse mesh)
    pub region: Option<String>,
    /// Availability zone identifier (for sparse mesh)
    pub availability_zone: Option<String>,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            full_mesh_threshold: 10,
            region: None,
            availability_zone: None,
        }
    }
}

/// Topology manager
pub struct TopologyManager {
    config: TopologyConfig,
    state: ClusterState,
    self_name: String,
    /// Active peer connections (for sparse mesh)
    active_peers: Arc<RwLock<HashSet<String>>>,
}

impl TopologyManager {
    pub fn new(config: TopologyConfig, state: ClusterState, self_name: String) -> Self {
        Self {
            config,
            state,
            self_name,
            active_peers: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Get peers to connect to based on topology
    pub fn get_peers(&self, count: usize) -> Vec<MembershipState> {
        let state = self.state.read();
        let total_nodes = state.len();

        if total_nodes <= self.config.full_mesh_threshold {
            // Full mesh: connect to all nodes
            self.get_full_mesh_peers(&state, count)
        } else {
            // Sparse mesh: connect based on region/AZ
            self.get_sparse_mesh_peers(&state, count)
        }
    }

    /// Get peers for full mesh topology
    fn get_full_mesh_peers(
        &self,
        state: &BTreeMap<String, super::gossip::NodeState>,
        count: usize,
    ) -> Vec<MembershipState> {
        let mut peers = Vec::new();
        let active = self.active_peers.read();

        for (name, node) in state.iter() {
            if name != &self.self_name
                && node.status == super::gossip::NodeStatus::Alive as i32
                && !active.contains(name)
            {
                let metadata: BTreeMap<String, Vec<u8>> = node
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<BTreeMap<_, _>>();
                peers.push(MembershipState {
                    name: node.name.clone(),
                    address: node.address.clone(),
                    status: node.status,
                    version: node.version,
                    metadata,
                });
                if peers.len() >= count {
                    break;
                }
            }
        }

        peers
    }

    /// Get peers for sparse mesh topology (by region/AZ)
    fn get_sparse_mesh_peers(
        &self,
        state: &BTreeMap<String, super::gossip::NodeState>,
        count: usize,
    ) -> Vec<MembershipState> {
        let mut peers = Vec::new();
        let active = self.active_peers.read();

        // First, try to connect to nodes in same region/AZ
        if let (Some(ref region), Some(ref az)) =
            (&self.config.region, &self.config.availability_zone)
        {
            for (name, node) in state.iter() {
                if name != &self.self_name
                    && node.status == super::gossip::NodeStatus::Alive as i32
                    && !active.contains(name)
                {
                    // Check if node is in same region/AZ (from metadata)
                    let node_region = node
                        .metadata
                        .get("region")
                        .and_then(|v| String::from_utf8(v.clone()).ok());
                    let node_az = node
                        .metadata
                        .get("availability_zone")
                        .and_then(|v| String::from_utf8(v.clone()).ok());

                    if node_region.as_ref() == Some(region) && node_az.as_ref() == Some(az) {
                        let metadata: BTreeMap<String, Vec<u8>> = node
                            .metadata
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        peers.push(MembershipState {
                            name: node.name.clone(),
                            address: node.address.clone(),
                            status: node.status,
                            version: node.version,
                            metadata,
                        });
                        if peers.len() >= count {
                            break;
                        }
                    }
                }
            }
        }

        // If not enough peers, add from other regions
        if peers.len() < count {
            for (name, node) in state.iter() {
                if name != &self.self_name
                    && node.status == super::gossip::NodeStatus::Alive as i32
                    && !active.contains(name)
                    && !peers.iter().any(|p| p.name == node.name)
                {
                    let metadata: BTreeMap<String, Vec<u8>> = node
                        .metadata
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    peers.push(MembershipState {
                        name: node.name.clone(),
                        address: node.address.clone(),
                        status: node.status,
                        version: node.version,
                        metadata,
                    });
                    if peers.len() >= count {
                        break;
                    }
                }
            }
        }

        peers
    }

    /// Mark peer as active
    pub fn mark_peer_active(&self, peer_name: &str) {
        self.active_peers.write().insert(peer_name.to_string());
        debug!("Marked peer {} as active", peer_name);
    }

    /// Mark peer as inactive
    pub fn mark_peer_inactive(&self, peer_name: &str) {
        self.active_peers.write().remove(peer_name);
        debug!("Marked peer {} as inactive", peer_name);
    }

    /// Get number of active peers
    pub fn active_peer_count(&self) -> usize {
        self.active_peers.read().len()
    }

    /// Check if we should use full mesh
    pub fn is_full_mesh(&self) -> bool {
        let state = self.state.read();
        state.len() <= self.config.full_mesh_threshold
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::mesh::service::gossip::{NodeState, NodeStatus};

    fn create_test_cluster_state(nodes: Vec<(String, String, i32)>) -> ClusterState {
        let mut state = BTreeMap::new();
        for (name, address, status) in nodes {
            state.insert(
                name.clone(),
                NodeState {
                    name: name.clone(),
                    address,
                    status,
                    version: 1,
                    metadata: std::collections::HashMap::new(),
                },
            );
        }
        Arc::new(RwLock::new(state))
    }

    #[test]
    fn test_full_mesh_topology() {
        let state = create_test_cluster_state(vec![
            (
                "node1".to_string(),
                "127.0.0.1:8000".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node2".to_string(),
                "127.0.0.1:8001".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node3".to_string(),
                "127.0.0.1:8002".to_string(),
                NodeStatus::Alive as i32,
            ),
        ]);

        let config = TopologyConfig {
            full_mesh_threshold: 10,
            region: None,
            availability_zone: None,
        };

        let manager = TopologyManager::new(config, state, "node1".to_string());

        let peers = manager.get_peers(5);
        // Should return all available peers (node2 and node3)
        assert_eq!(peers.len(), 2);
        assert!(peers.iter().any(|p| p.name == "node2"));
        assert!(peers.iter().any(|p| p.name == "node3"));
    }

    #[test]
    fn test_full_mesh_topology_excludes_self() {
        let state = create_test_cluster_state(vec![
            (
                "node1".to_string(),
                "127.0.0.1:8000".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node2".to_string(),
                "127.0.0.1:8001".to_string(),
                NodeStatus::Alive as i32,
            ),
        ]);

        let config = TopologyConfig {
            full_mesh_threshold: 10,
            region: None,
            availability_zone: None,
        };

        let manager = TopologyManager::new(config, state, "node1".to_string());

        let peers = manager.get_peers(5);
        // Should not include self (node1)
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].name, "node2");
    }

    #[test]
    fn test_full_mesh_topology_filters_down_nodes() {
        let state = create_test_cluster_state(vec![
            (
                "node1".to_string(),
                "127.0.0.1:8000".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node2".to_string(),
                "127.0.0.1:8001".to_string(),
                NodeStatus::Down as i32,
            ),
            (
                "node3".to_string(),
                "127.0.0.1:8002".to_string(),
                NodeStatus::Alive as i32,
            ),
        ]);

        let config = TopologyConfig {
            full_mesh_threshold: 10,
            region: None,
            availability_zone: None,
        };

        let manager = TopologyManager::new(config, state, "node1".to_string());

        let peers = manager.get_peers(5);
        // Should only return alive nodes (node3)
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].name, "node3");
    }

    #[test]
    fn test_sparse_mesh_topology() {
        let state = create_test_cluster_state(vec![
            (
                "node1".to_string(),
                "127.0.0.1:8000".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node2".to_string(),
                "127.0.0.1:8001".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node3".to_string(),
                "127.0.0.1:8002".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node4".to_string(),
                "127.0.0.1:8003".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node5".to_string(),
                "127.0.0.1:8004".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node6".to_string(),
                "127.0.0.1:8005".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node7".to_string(),
                "127.0.0.1:8006".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node8".to_string(),
                "127.0.0.1:8007".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node9".to_string(),
                "127.0.0.1:8008".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node10".to_string(),
                "127.0.0.1:8009".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node11".to_string(),
                "127.0.0.1:8010".to_string(),
                NodeStatus::Alive as i32,
            ),
        ]);

        let config = TopologyConfig {
            full_mesh_threshold: 10, // 11 nodes > 10, should use sparse
            region: None,
            availability_zone: None,
        };

        let manager = TopologyManager::new(config, state, "node1".to_string());

        let peers = manager.get_peers(5);
        // Should return peers (sparse mesh mode)
        assert!(!peers.is_empty());
        assert!(peers.len() <= 5);
    }

    #[test]
    fn test_sparse_mesh_with_region_az() {
        let mut state_map = BTreeMap::new();

        // Create nodes with region/AZ metadata
        let mut node1_metadata = std::collections::HashMap::new();
        node1_metadata.insert("region".to_string(), b"us-west".to_vec());
        node1_metadata.insert("availability_zone".to_string(), b"us-west-1a".to_vec());
        state_map.insert(
            "node1".to_string(),
            NodeState {
                name: "node1".to_string(),
                address: "127.0.0.1:8000".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: node1_metadata.clone(),
            },
        );

        let mut node2_metadata = std::collections::HashMap::new();
        node2_metadata.insert("region".to_string(), b"us-west".to_vec());
        node2_metadata.insert("availability_zone".to_string(), b"us-west-1a".to_vec());
        state_map.insert(
            "node2".to_string(),
            NodeState {
                name: "node2".to_string(),
                address: "127.0.0.1:8001".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: node2_metadata,
            },
        );

        let mut node3_metadata = std::collections::HashMap::new();
        node3_metadata.insert("region".to_string(), b"us-east".to_vec());
        node3_metadata.insert("availability_zone".to_string(), b"us-east-1a".to_vec());
        state_map.insert(
            "node3".to_string(),
            NodeState {
                name: "node3".to_string(),
                address: "127.0.0.1:8002".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: node3_metadata,
            },
        );

        let state = Arc::new(RwLock::new(state_map));

        let config = TopologyConfig {
            full_mesh_threshold: 2,
            region: Some("us-west".to_string()),
            availability_zone: Some("us-west-1a".to_string()),
        };

        let manager = TopologyManager::new(config, state, "node1".to_string());

        let peers = manager.get_peers(5);
        // Should prefer nodes in same region/AZ (node2)
        assert!(!peers.is_empty());
        // node2 should be in the list (same region/AZ)
        assert!(peers.iter().any(|p| p.name == "node2"));
    }

    #[test]
    fn test_mark_peer_active_inactive() {
        let state = create_test_cluster_state(vec![
            (
                "node1".to_string(),
                "127.0.0.1:8000".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node2".to_string(),
                "127.0.0.1:8001".to_string(),
                NodeStatus::Alive as i32,
            ),
        ]);

        let config = TopologyConfig {
            full_mesh_threshold: 10,
            region: None,
            availability_zone: None,
        };

        let manager = TopologyManager::new(config, state, "node1".to_string());

        assert_eq!(manager.active_peer_count(), 0);

        manager.mark_peer_active("node2");
        assert_eq!(manager.active_peer_count(), 1);

        manager.mark_peer_inactive("node2");
        assert_eq!(manager.active_peer_count(), 0);
    }

    #[test]
    fn test_get_peers_excludes_active_peers() {
        let state = create_test_cluster_state(vec![
            (
                "node1".to_string(),
                "127.0.0.1:8000".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node2".to_string(),
                "127.0.0.1:8001".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node3".to_string(),
                "127.0.0.1:8002".to_string(),
                NodeStatus::Alive as i32,
            ),
        ]);

        let config = TopologyConfig {
            full_mesh_threshold: 10,
            region: None,
            availability_zone: None,
        };

        let manager = TopologyManager::new(config, state, "node1".to_string());

        manager.mark_peer_active("node2");

        let peers = manager.get_peers(5);
        // Should exclude node2 (already active)
        assert!(!peers.iter().any(|p| p.name == "node2"));
        // Should include node3
        assert!(peers.iter().any(|p| p.name == "node3"));
    }

    #[test]
    fn test_is_full_mesh() {
        let state = create_test_cluster_state(vec![
            (
                "node1".to_string(),
                "127.0.0.1:8000".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node2".to_string(),
                "127.0.0.1:8001".to_string(),
                NodeStatus::Alive as i32,
            ),
        ]);

        let config = TopologyConfig {
            full_mesh_threshold: 10,
            region: None,
            availability_zone: None,
        };

        let manager = TopologyManager::new(config, state, "node1".to_string());
        assert!(manager.is_full_mesh());

        let state2 = create_test_cluster_state(vec![
            (
                "node1".to_string(),
                "127.0.0.1:8000".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node2".to_string(),
                "127.0.0.1:8001".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node3".to_string(),
                "127.0.0.1:8002".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node4".to_string(),
                "127.0.0.1:8003".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node5".to_string(),
                "127.0.0.1:8004".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node6".to_string(),
                "127.0.0.1:8005".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node7".to_string(),
                "127.0.0.1:8006".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node8".to_string(),
                "127.0.0.1:8007".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node9".to_string(),
                "127.0.0.1:8008".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node10".to_string(),
                "127.0.0.1:8009".to_string(),
                NodeStatus::Alive as i32,
            ),
            (
                "node11".to_string(),
                "127.0.0.1:8010".to_string(),
                NodeStatus::Alive as i32,
            ),
        ]);

        let config2 = TopologyConfig {
            full_mesh_threshold: 10,
            region: None,
            availability_zone: None,
        };

        let manager2 = TopologyManager::new(config2, state2, "node1".to_string());
        assert!(!manager2.is_full_mesh());
    }
}
