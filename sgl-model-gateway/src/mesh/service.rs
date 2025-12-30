use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use parking_lot::RwLock;
use tonic::Request;
use tracing as log;

pub mod gossip {
    #![allow(unused_qualifications)]
    tonic::include_proto!("sglang.mesh.gossip");
}
use gossip::{
    gossip_client, gossip_message, GossipMessage, NodeState, NodeStatus, NodeUpdate, Ping,
    StateSync,
};

use crate::mesh::{
    controller::MeshController,
    node_state_machine::{ConvergenceConfig, NodeStateMachine},
    partition::PartitionDetector,
    ping_server::GossipService,
};

pub type ClusterState = Arc<RwLock<BTreeMap<String, NodeState>>>;

pub struct MeshServerConfig {
    pub self_name: String,
    pub self_addr: SocketAddr,
    pub init_peer: Option<SocketAddr>,
}

/// MeshServerHandler
/// It is the handler for the mesh server, which is responsible for the node management.
/// Includes some basic node management logic, like shutdown,
/// node discovery(TODO), node status update(TODO), etc.
pub struct MeshServerHandler {
    pub state: ClusterState,
    pub self_name: String,
    _self_addr: SocketAddr,
    signal_tx: tokio::sync::watch::Sender<()>,
    partition_detector: Option<Arc<PartitionDetector>>,
    state_machine: Option<Arc<NodeStateMachine>>,
}

impl MeshServerHandler {
    pub fn new(
        state: ClusterState,
        self_name: &str,
        self_addr: SocketAddr,
        signal_tx: tokio::sync::watch::Sender<()>,
    ) -> Self {
        Self {
            state,
            self_name: self_name.to_string(),
            _self_addr: self_addr,
            signal_tx,
            partition_detector: None,
            state_machine: None,
        }
    }

    /// Create with partition detector and state machine
    pub fn with_partition_and_state_machine(
        state: ClusterState,
        self_name: &str,
        self_addr: SocketAddr,
        signal_tx: tokio::sync::watch::Sender<()>,
        stores: Option<Arc<super::stores::StateStores>>,
    ) -> Self {
        let partition_detector = Some(Arc::new(PartitionDetector::default()));
        let state_machine =
            stores.map(|s| Arc::new(NodeStateMachine::new(s, ConvergenceConfig::default())));

        Self {
            state,
            self_name: self_name.to_string(),
            _self_addr: self_addr,
            signal_tx,
            partition_detector,
            state_machine,
        }
    }

    /// Get partition detector
    pub fn partition_detector(&self) -> Option<&Arc<PartitionDetector>> {
        self.partition_detector.as_ref()
    }

    /// Get state machine
    pub fn state_machine(&self) -> Option<&Arc<NodeStateMachine>> {
        self.state_machine.as_ref()
    }

    /// Check if node is ready
    pub fn is_ready(&self) -> bool {
        self.state_machine
            .as_ref()
            .map(|sm| sm.is_ready())
            .unwrap_or(true) // If no state machine, consider ready
    }

    /// Check if we should serve (have quorum)
    pub fn should_serve(&self) -> bool {
        self.partition_detector
            .as_ref()
            .map(|pd| pd.should_serve())
            .unwrap_or(true) // If no partition detector, consider should serve
    }

    /// Shutdown immediately without graceful shutdown
    pub fn shutdown(&self) {
        self.signal_tx.send(()).ok();
    }

    /// Graceful shutdown: broadcast LEAVING status to all alive nodes,
    /// wait for propagation, then shutdown
    pub async fn graceful_shutdown(&self) -> Result<()> {
        log::info!("Starting graceful shutdown for node {}", self.self_name);

        let (leaving_node, alive_nodes) = {
            let state = self.state.read();

            let mut self_node = if let Some(self_node) = state.get(&self.self_name) {
                self_node.clone()
            } else {
                self.signal_tx.send(()).ok();
                return Ok(());
            };

            if self_node.status != NodeStatus::Leaving as i32 {
                self_node.status = NodeStatus::Leaving as i32;
                self_node.version += 1;

                let alive_nodes = state
                    .values()
                    .filter(|node| {
                        node.status == NodeStatus::Alive as i32 // include self
                    })
                    .cloned()
                    .collect::<Vec<NodeState>>();
                (self_node.clone(), alive_nodes)
            } else {
                self.signal_tx.send(()).ok();
                return Ok(());
            }
        };

        log::info!(
            "Broadcasting LEAVING status to {} alive nodes",
            alive_nodes.len()
        );

        // Broadcast LEAVING status to all alive nodes
        let (success_count, total_count) = broadcast_node_states(
            vec![leaving_node],
            alive_nodes,
            Some(Duration::from_secs(3)),
        )
        .await;

        log::info!(
            "Broadcast LEAVING status: {}/{} successful",
            success_count,
            total_count
        );

        // Wait a bit more for state propagation
        let propagation_delay = Duration::from_secs(1);
        log::info!(
            "Waiting {} seconds for LEAVING status propagation",
            propagation_delay.as_secs()
        );
        tokio::time::sleep(propagation_delay).await;

        log::info!("Sending shutdown signal");
        self.signal_tx.send(()).ok();
        Ok(())
    }

    pub fn write_data(&self, key: String, value: Vec<u8>) {
        let mut state = self.state.write();
        state.entry(self.self_name.clone()).and_modify(|e| {
            e.version += 1;
            e.metadata.insert(key, value);
        });
    }

    pub fn read_data(&self, key: String) -> Option<Vec<u8>> {
        let state = self.state.read();
        state
            .get(&self.self_name)
            .and_then(|e| e.metadata.get(&key).cloned())
    }
}

pub struct MeshServerBuilder {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
}

impl MeshServerBuilder {
    pub fn new(self_name: String, self_addr: SocketAddr, init_peer: Option<SocketAddr>) -> Self {
        let state = Arc::new(RwLock::new(BTreeMap::from([(
            self_name.clone(),
            NodeState {
                name: self_name.clone(),
                address: self_addr.to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: HashMap::new(),
            },
        )])));
        Self {
            state,
            self_name,
            self_addr,
            init_peer,
        }
    }

    pub fn build(&self) -> (MeshServer, MeshServerHandler) {
        self.build_with_stores(None)
    }

    pub fn build_with_stores(
        &self,
        stores: Option<Arc<super::stores::StateStores>>,
    ) -> (MeshServer, MeshServerHandler) {
        let (signal_tx, signal_rx) = tokio::sync::watch::channel(());
        (
            MeshServer::new(
                self.state.clone(),
                &self.self_name,
                self.self_addr,
                self.init_peer,
                signal_rx,
            ),
            MeshServerHandler::with_partition_and_state_machine(
                self.state.clone(),
                &self.self_name,
                self.self_addr,
                signal_tx,
                stores,
            ),
        )
    }
}

pub struct MeshServer {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    signal_rx: tokio::sync::watch::Receiver<()>,
}

impl MeshServer {
    pub fn new(
        state: ClusterState,
        self_name: &str,
        self_addr: SocketAddr,
        init_peer: Option<SocketAddr>,
        signal_rx: tokio::sync::watch::Receiver<()>,
    ) -> Self {
        MeshServer {
            state,
            self_name: self_name.to_string(),
            self_addr,
            init_peer,
            signal_rx,
        }
    }

    pub fn build_ping_server(&self) -> GossipService {
        GossipService::new(self.state.clone(), self.self_addr, &self.self_name)
    }

    pub fn build_controller(&self) -> MeshController {
        MeshController::new(
            self.state.clone(),
            self.self_addr,
            &self.self_name,
            self.init_peer,
        )
    }

    pub async fn start_serve(self) -> Result<()> {
        log::info!("Mesh server listening on {}", self.self_addr);
        let self_name = self.self_name.clone();
        let self_address = self.self_addr;

        let service = self.build_ping_server();
        let controller = self.build_controller();

        let mut service_shutdown = self.signal_rx.clone();

        let listener = tokio::spawn(service.serve_ping_with_shutdown(async move {
            _ = service_shutdown.changed().await;
        }));
        tokio::time::sleep(Duration::from_secs(1)).await;
        let app_handle = tokio::spawn(controller.event_loop(self.signal_rx.clone()));

        tokio::select! {
            res = listener => res??,
            res = app_handle => res??,
        }

        log::info!(
            "Mesh server {} at {} is shutting down",
            self_name,
            self_address
        );

        Ok(())
    }

    pub async fn start_serve_with_stores(
        self,
        stores: Option<Arc<super::stores::StateStores>>,
        sync_manager: Option<Arc<super::sync::MeshSyncManager>>,
        partition_detector: Option<Arc<PartitionDetector>>,
    ) -> Result<()> {
        log::info!("Mesh server listening on {}", self.self_addr);
        let self_name = self.self_name.clone();
        let self_address = self.self_addr;

        let mut service = self.build_ping_server();
        if let Some(stores) = stores {
            service = service.with_stores(stores);
        }
        if let Some(sync_manager) = sync_manager {
            service = service.with_sync_manager(sync_manager);
        }
        if let Some(partition_detector) = partition_detector {
            service = service.with_partition_detector(partition_detector);
        }
        let controller = self.build_controller();

        let mut service_shutdown = self.signal_rx.clone();

        let listener = tokio::spawn(service.serve_ping_with_shutdown(async move {
            _ = service_shutdown.changed().await;
        }));
        tokio::time::sleep(Duration::from_secs(1)).await;
        let app_handle = tokio::spawn(controller.event_loop(self.signal_rx.clone()));

        tokio::select! {
            res = listener => res??,
            res = app_handle => res??,
        }

        log::info!(
            "Mesh server {} at {} is shutting down",
            self_name,
            self_address
        );
        Ok(())
    }
}

/// Broadcast node state updates to target nodes
/// Returns (success_count, total_count)
pub async fn broadcast_node_states(
    nodes_to_broadcast: Vec<NodeState>,
    target_nodes: Vec<NodeState>,
    timeout: Option<Duration>,
) -> (usize, usize) {
    if nodes_to_broadcast.is_empty() || target_nodes.is_empty() {
        log::debug!(
            "Nothing to broadcast: nodes_to_broadcast={}, target_nodes={}",
            nodes_to_broadcast.len(),
            target_nodes.len()
        );
        return (0, target_nodes.len());
    }

    let mut broadcast_tasks = Vec::new();
    for target_node in &target_nodes {
        let target_node_clone = target_node.clone();
        let nodes_for_task = nodes_to_broadcast.clone();
        let task = tokio::spawn(async move {
            let state_sync = StateSync {
                nodes: nodes_for_task,
            };
            let ping_payload = gossip_message::Payload::Ping(Ping {
                state_sync: Some(state_sync),
            });
            match try_ping(&target_node_clone, Some(ping_payload)).await {
                Ok(_) => {
                    log::debug!("Successfully broadcasted to {}", target_node_clone.name);
                    Ok(())
                }
                Err(e) => {
                    log::warn!("Failed to broadcast to {}: {}", target_node_clone.name, e);
                    Err(e)
                }
            }
        });
        broadcast_tasks.push(task);
    }

    let timeout_duration = timeout.unwrap_or(Duration::from_secs(3));
    let broadcast_result = tokio::time::timeout(timeout_duration, async {
        futures::future::join_all(broadcast_tasks).await
    })
    .await;

    match broadcast_result {
        Ok(results) => {
            let success_count = results.iter().filter(|r| r.is_ok()).count();
            let total_count = target_nodes.len();
            log::info!(
                "Broadcast completed: {}/{} successful",
                success_count,
                total_count
            );
            (success_count, total_count)
        }
        Err(_) => {
            log::warn!(
                "Broadcast timeout after {} seconds",
                timeout_duration.as_secs()
            );
            (0, target_nodes.len())
        }
    }
}

pub async fn try_ping(
    peer_node: &NodeState,
    payload: Option<gossip_message::Payload>,
) -> Result<NodeUpdate, tonic::Status> {
    let peer_name = peer_node.name.clone();

    let peer_addr = SocketAddr::from_str(&peer_node.address).map_err(|e| {
        tonic::Status::invalid_argument(format!(
            "Invalid address for node {}: {}, {}",
            peer_name, peer_node.address, e
        ))
    })?;
    let mut client = gossip_client::GossipClient::connect(format!("http://{}", peer_addr))
        .await
        .map_err(|e| {
            log::warn!(
                "Failed to connect to peer {} {}: {}.",
                peer_name,
                peer_addr,
                e
            );
            tonic::Status::unavailable("Failed to connect to peer")
        })?;

    let ping_message = GossipMessage { payload };
    let response = client.ping_server(Request::new(ping_message)).await?;

    Ok(response.into_inner())
}

#[macro_export]
macro_rules! mesh_run {
    ($addr:expr, $init_peer:expr) => {{
        mesh_run!($addr.to_string(), $addr, $init_peer)
    }};

    ($name:expr, $addr:expr, $init_peer:expr) => {{
        tracing::info!("Starting mesh server : {}", $addr);
        let (server, handler) =
            $crate::mesh::service::MeshServerBuilder::new($name.to_string(), $addr, $init_peer)
                .build();
        tokio::spawn(async move {
            if let Err(e) = server.start_serve().await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });
        handler
    }};
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use tokio::net::TcpListener;
    use tracing as log;
    use tracing_subscriber::{
        filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
    };

    use super::*;
    static INIT: Once = Once::new();
    fn init() {
        INIT.call_once(|| {
            let _ = tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer())
                .with(
                    EnvFilter::builder()
                        .with_default_directive(LevelFilter::INFO.into())
                        .from_env_lossy(),
                )
                .try_init();
        });
    }
    async fn find_free_port() -> (TcpListener, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        log::info!("Found free port: {}", port);
        (listener, port)
    }

    async fn get_node() -> SocketAddr {
        let (_listener, port) = find_free_port().await;
        format!("127.0.0.1:{}", port).parse().unwrap()
    }

    fn print_state(handler: &MeshServerHandler) -> String {
        let state = handler.state.read();
        let mut res = vec![];
        for (k, v) in state.iter() {
            res.push(format!(
                "{}: {:?} - {:?}",
                k,
                NodeStatus::try_from(v.status).unwrap(),
                v.metadata
            ));
        }
        res.join(", ")
    }

    #[tokio::test]
    async fn test_state_synchronization() {
        init();
        log::info!("Starting test_state_synchronization");

        // 1. setup node A and B for initial cluster
        let addr_a = get_node().await;
        let handler_a = mesh_run!("A", addr_a, None);
        let addr_b = get_node().await;
        let handler_b = mesh_run!("B", addr_b, Some(addr_a));

        // 2. wait for node A and B to sync and write some data
        tokio::time::sleep(Duration::from_secs(2)).await;
        handler_a.write_data("hello".into(), "world".into());
        log::info!("================================================");

        // 3. add node C and D and wait for them to sync
        let addr_c = get_node().await;
        let handler_c = mesh_run!("C", addr_c, Some(addr_a));
        let addr_d = get_node().await;
        let handler_d = mesh_run!("D", addr_d, Some(addr_c));
        tokio::time::sleep(Duration::from_secs(2)).await;
        log::info!("================================================");

        // 4. add node E and wait for it to sync and kill it
        {
            let addr_e = get_node().await;
            let handler_e = mesh_run!("E", addr_e, Some(addr_d));
            tokio::time::sleep(Duration::from_secs(3)).await;
            log::info!("State E: {:?}", print_state(&handler_e));
            // killing_button.send(()).unwrap();
            handler_e.shutdown();
        }

        handler_d.graceful_shutdown().await.unwrap();
        tokio::time::sleep(Duration::from_secs(2)).await;
        log::info!("================================================");

        // 5. wait for node status to sync
        tokio::time::sleep(Duration::from_secs(8)).await;
        log::info!("================================================");

        // 6. verify node status, status of all nodes should be same, and node E should be down
        let final_state = String::from("A: Alive - {\"hello\": [119, 111, 114, 108, 100]}, B: Alive - {}, C: Alive - {}, D: Leaving - {}, E: Down - {}");
        assert_eq!(
            print_state(&handler_a),
            final_state,
            "State A: {:?}",
            print_state(&handler_a)
        );
        assert_eq!(
            print_state(&handler_b),
            final_state,
            "State B: {:?}",
            print_state(&handler_b)
        );
        assert_eq!(
            print_state(&handler_c),
            final_state,
            "State C: {:?}",
            print_state(&handler_c)
        );
    }
}
