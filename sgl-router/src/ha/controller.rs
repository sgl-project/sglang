use anyhow::Result;

use rand::seq::{IndexedRandom, SliceRandom};
use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;

use std::time::Duration;

use tracing as log;
use tracing::instrument;

use super::gossip::{gossip_message, NodeState, NodeStatus, Ping, PingReq, StateSync};

use super::service::{broadcast_node_states, try_ping};
use super::ClusterState;

pub struct HAController {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
}

impl HAController {
    pub fn new(
        state: ClusterState,
        self_addr: SocketAddr,
        self_name: &str,
        init_peer: Option<SocketAddr>,
    ) -> Self {
        Self {
            state,
            self_name: self_name.to_string(),
            self_addr,
            init_peer: init_peer,
        }
    }

    #[instrument(fields(name = %self.self_name), skip(self, signal))]
    pub async fn event_loop(self, mut signal: tokio::sync::watch::Receiver<()>) -> Result<()> {
        let init_state = self.state.clone();
        let read_state = self.state.clone();
        let mut cnt: u64 = 0;

        loop {
            log::info!("Round {} Status:{:?}", cnt, read_state.read());
            let peer = if cnt == 0 {
                if let Some(init_peer) = self.init_peer {
                    Some(NodeState {
                        name: "init_peer".to_string(),
                        address: init_peer.to_string(),
                        status: NodeStatus::Suspected as i32,
                        version: 1,
                        metadata: HashMap::new(),
                    })
                } else {
                    None
                }
            } else {
                let mut map = init_state.read().clone();
                map.retain(|k, v| {
                    k.ne(&self.self_name.to_string())
                        && v.status != NodeStatus::Down as i32
                        && v.status != NodeStatus::Leaving as i32
                });
                let random_nodes = get_random_values_refs(&map, 1);
                random_nodes.first().map(|&node| node.clone())
            };
            cnt += 1;

            tokio::select! {

                _ = signal.changed() => {
                    log::info!("Gossip app_server {} at {} is shutting down", self.self_name, self.self_addr);
                    break;
                }

                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    if let Some(peer) = peer {
                        if let Err(e) = self.connect_to_peer(peer.clone()).await {
                            log::warn!("Error connecting to peer {}: {}", peer.name, e);
                        }
                    } else {
                        log::info!("No peer address available to connect");
                    }
                }
            }
        }
        Ok(())
    }

    async fn connect_to_peer(&self, peer: NodeState) -> Result<()> {
        log::info!("Connecting to peer {} at {}", peer.name, peer.address);

        let read_state = self.state.clone();

        // TODO: Maybe we don't need to send the whole state.
        let state_sync = StateSync {
            nodes: read_state.read().values().cloned().collect(),
        };
        let peer_addr = peer.address.parse::<SocketAddr>()?;
        let peer_name = peer.name.clone();
        match try_ping(
            &peer,
            Some(gossip_message::Payload::Ping(Ping {
                state_sync: Some(state_sync),
            })),
        )
        .await
        {
            Ok(node_update) => {
                log::info!("Received NodeUpdate from peer: {:?}", node_update);
                // Update state for Alive or Leaving status
                if node_update.status == NodeStatus::Alive as i32
                    || node_update.status == NodeStatus::Leaving as i32
                {
                    let mut s = read_state.write();
                    s.entry(node_update.name.clone())
                        .and_modify(|e| e.status = node_update.status)
                        .or_insert(NodeState {
                            name: node_update.name,
                            address: node_update.address,
                            status: node_update.status,
                            version: 1,
                            metadata: HashMap::new(),
                        });
                }
            }
            Err(e) => {
                log::info!("Failed to connect to peer: {}, now try ping-req", e);
                let mut map = read_state.read().clone();
                map.retain(|k, v| {
                    k.ne(&self.self_name)
                        && k.ne(&peer_name)
                        && v.status == NodeStatus::Alive as i32
                        && v.status != NodeStatus::Leaving as i32
                });
                let random_nodes = get_random_values_refs(&map, 3);
                let mut reachable = false;
                for node in random_nodes {
                    log::info!(
                        "Trying to ping-req node {}, req target: {}",
                        node.address,
                        peer_addr
                    );
                    if try_ping(
                        &node,
                        Some(gossip_message::Payload::PingReq(PingReq {
                            node: Some(peer.clone()),
                        })),
                    )
                    .await
                    .is_ok()
                    {
                        reachable = true;
                        break;
                    }
                }
                if !reachable {
                    let mut target = read_state.read().clone();

                    // Broadcast only the unreachable node's status is enough.
                    if let Some(mut unreachable_node) = target.remove(&peer_name) {
                        if unreachable_node.status == NodeStatus::Suspected as i32 {
                            unreachable_node.status = NodeStatus::Down as i32
                        } else {
                            unreachable_node.status = NodeStatus::Suspected as i32
                        }
                        unreachable_node.version += 1;

                        // Broadcast target nodes should include self.
                        let target_nodes: Vec<NodeState> = target
                            .values()
                            .filter(|v| {
                                v.name.ne(&peer_name)
                                    && v.status == NodeStatus::Alive as i32
                                    && v.status != NodeStatus::Leaving as i32
                            })
                            .cloned()
                            .collect();

                        log::info!(
                            "Broadcasting node status to {} alive nodes, new_state: {:?}",
                            target_nodes.len(),
                            unreachable_node
                        );

                        let (success_count, total_count) = broadcast_node_states(
                            vec![unreachable_node],
                            target_nodes,
                            None, // Use default timeout
                        )
                        .await;

                        log::info!(
                            "Broadcast node status: {}/{} successful",
                            success_count,
                            total_count
                        );
                    }
                }
            }
        }

        log::info!("Successfully connected to peer {}", peer_addr);

        Ok(())
    }
}

// TODO: Support weighted random selection. e.g. nodes in INIT state should be more likely to be selected.
fn get_random_values_refs<K, V>(map: &BTreeMap<K, V>, k: usize) -> Vec<&V> {
    let values: Vec<&V> = map.values().collect();

    if k >= values.len() {
        let mut all_values = values;
        all_values.shuffle(&mut rand::rng());
        return all_values;
    }

    let mut rng = rand::rng();

    values.choose_multiple(&mut rng, k).cloned().collect()
}
