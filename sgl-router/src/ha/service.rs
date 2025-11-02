use anyhow::Result;
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use tonic::Request;
use tracing as log;

pub mod gossip {
    #![allow(unused_qualifications)]
    tonic::include_proto!("sglang.ha.gossip");
}
use gossip::{gossip_client, gossip_message, GossipMessage, NodeState, NodeStatus, NodeUpdate};

use crate::ha::controller::HAController;
use crate::ha::ping_server::GossipService;

pub type ClusterState = Arc<RwLock<BTreeMap<String, NodeState>>>;

pub struct HAServerConfig {
    pub self_name: String,
    pub self_addr: SocketAddr,
    pub init_peer: Option<SocketAddr>,
}

/// HAServerHandler
/// It is the handler for the HA server, which is responsible for the node management.
/// Includes some basic node management logic, like shutdown,
/// node discovery(TODO), node status update(TODO), etc.
pub struct HAServerHandler {
    pub state: ClusterState,
    self_name: String,
    _self_addr: SocketAddr,
    signal_tx: tokio::sync::watch::Sender<()>,
}

impl HAServerHandler {
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
            signal_tx: signal_tx,
        }
    }
    pub fn shutdown(&mut self) {
        self.signal_tx.send(()).ok();
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

pub struct HAServerBuilder {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
}

impl HAServerBuilder {
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

    pub fn build(&self) -> (HAServer, HAServerHandler) {
        let (signal_tx, signal_rx) = tokio::sync::watch::channel(());
        (
            HAServer::new(
                self.state.clone(),
                &self.self_name,
                self.self_addr,
                self.init_peer.clone(),
                signal_rx,
            ),
            HAServerHandler::new(
                self.state.clone(),
                &self.self_name,
                self.self_addr,
                signal_tx,
            ),
        )
    }
}

pub struct HAServer {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    signal_rx: tokio::sync::watch::Receiver<()>,
}

impl HAServer {
    pub fn new(
        state: ClusterState,
        self_name: &str,
        self_addr: SocketAddr,
        init_peer: Option<SocketAddr>,
        signal_rx: tokio::sync::watch::Receiver<()>,
    ) -> Self {
        HAServer {
            state: state,
            self_name: self_name.to_string(),
            self_addr: self_addr,
            init_peer: init_peer,
            signal_rx: signal_rx,
        }
    }

    pub fn build_ping_server(&self) -> GossipService {
        GossipService::new(self.state.clone(), self.self_addr, &self.self_name)
    }

    pub fn build_controller(&self) -> HAController {
        HAController::new(
            self.state.clone(),
            self.self_addr,
            &self.self_name,
            self.init_peer,
        )
    }

    pub async fn start_serve(self) -> Result<()> {
        log::info!("HA server listening on {}", self.self_addr);
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
            "HA server {} at {} is shutting down",
            self_name,
            self_address
        );

        Ok(())
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

    let ping_message = GossipMessage { payload: payload };
    let response = client.ping_server(Request::new(ping_message)).await?;

    Ok(response.into_inner())
}

#[macro_export]
macro_rules! ha_run {
    ($addr:expr, $init_peer:expr) => {{
        ha_run!($addr.to_string(), $addr, $init_peer)
    }};

    ($name:expr, $addr:expr, $init_peer:expr) => {{
        tracing::info!("Starting HA server : {}", $addr);
        let (server, handler) =
            $crate::ha::service::HAServerBuilder::new($name.to_string(), $addr, $init_peer)
                .build();
        tokio::spawn(async move {
            if let Err(e) = server.start_serve().await {
                tracing::error!("HA server failed: {}", e);
            }
        });
        handler
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;
    use tokio::net::TcpListener;
    use tracing as log;
    use tracing_subscriber::{
        filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
    };
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

    fn print_state(handler: &HAServerHandler) -> String {
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
        let handler_a = ha_run!("A", addr_a, None);
        let addr_b = get_node().await;
        let handler_b = ha_run!("B", addr_b, Some(addr_a));

        // 2. wait for node A and B to sync and write some data
        tokio::time::sleep(Duration::from_secs(2)).await;
        handler_a.write_data("hello".into(), "world".into());
        log::info!("================================================");

        // 3. add node C and D and wait for them to sync
        let addr_c = get_node().await;
        let handler_c = ha_run!("C", addr_c, Some(addr_a));
        let addr_d = get_node().await;
        let handler_d = ha_run!("D", addr_d, Some(addr_c));
        tokio::time::sleep(Duration::from_secs(2)).await;
        log::info!("================================================");

        // 4. add node E and wait for it to sync and kill it
        {
            let addr_e = get_node().await;
            let mut handler_e = ha_run!("E", addr_e, Some(addr_d));
            tokio::time::sleep(Duration::from_secs(3)).await;
            log::info!("State E: {:?}", print_state(&handler_e));
            // killing_button.send(()).unwrap();
            handler_e.shutdown();
        }

        // 5. wait for node status to sync
        tokio::time::sleep(Duration::from_secs(8)).await;
        log::info!("================================================");

        // 6. verify node status, status of all nodes should be same, and node E should be down
        let final_state = String::from("A: Alive - {\"hello\": [119, 111, 114, 108, 100]}, B: Alive - {}, C: Alive - {}, D: Alive - {}, E: Down - {}");
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
        assert_eq!(
            print_state(&handler_d),
            final_state,
            "State D: {:?}",
            print_state(&handler_d)
        );
    }
}
