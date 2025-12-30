//! Topology management for HA cluster
//!
//! Supports:
//! - Full mesh for small/medium clusters
//! - Sparse mesh for large clusters (by region/AZ)

use std::{
    collections::{BTreeMap, HashSet},
    net::SocketAddr,
    sync::Arc,
};

use parking_lot::RwLock;
use tracing::{debug, info, warn};

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
