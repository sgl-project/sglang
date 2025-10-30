use anyhow::Result;

use std::net::SocketAddr;
use tonic::transport::Server;
use tonic::{Response, Status};
use tracing as log;
use tracing::instrument;

use super::gossip::{
    self,
    gossip_server::{Gossip, GossipServer},
    GossipMessage, NodeState, NodeStatus, NodeUpdate, PingReq,
};

use super::try_ping;
use super::ClusterState;

#[derive(Debug)]
pub struct GossipService {
    state: ClusterState,
    self_addr: SocketAddr,
    self_name: String,
}

impl GossipService {
    pub fn new(
        state: ClusterState,
        self_addr: SocketAddr,
        self_name: &str,
    ) -> Self {
        Self {
            state,
            self_addr: self_addr,
            self_name: self_name.to_string(),
        }
    }

    pub async fn serve_ping_with_shutdown<F: std::future::Future<Output = ()>>(
        self,
        signal: F,
    ) -> Result<()> {
        let listen_addr = self.self_addr;
        let service = GossipServer::new(self);
        Server::builder()
            .add_service(service)
            .serve_with_shutdown(listen_addr, signal)
            .await?;
        Ok(())
    }

    async fn merge_state(&self, incoming_nodes: Vec<NodeState>) -> bool {
        let mut state = self.state.write();
        let mut updated = false;
        for node in incoming_nodes {
            state
                .entry(node.name.clone())
                .and_modify(|entry| {
                    if node.version > entry.version {
                        *entry = node.clone();
                        updated = true;
                    }
                })
                .or_insert_with(|| {
                    updated = true;
                    node
                });
        }
        if updated {
            log::info!("Cluster state updated. Current nodes: {}", state.len());
        }
        updated
    }
}

#[tonic::async_trait]
impl Gossip for GossipService {
    #[instrument(fields(name = %self.self_name), skip(self, request))]
    async fn ping_server(
        &self,
        request: tonic::Request<GossipMessage>,
    ) -> std::result::Result<Response<NodeUpdate>, Status> {
        let message = request.into_inner();
        match message.payload {
            Some(gossip::gossip_message::Payload::Ping(ping)) => {
                log::info!("Received {:?}", ping);
                if let Some(stat_sync) = ping.state_sync {
                    log::info!("Merging state from Ping: {} nodes", stat_sync.nodes.len());
                    self.merge_state(stat_sync.nodes).await;
                }
                Ok(Response::new(NodeUpdate {
                    name: self.self_name.clone(),
                    address: self.self_addr.to_string(),
                    status: NodeStatus::Alive as i32,
                }))
            }
            Some(gossip::gossip_message::Payload::PingReq(PingReq { node: Some(node) })) => {
                log::info!("PingReq to node {} addr:{}", node.name, node.address);
                let res = try_ping(&node, None).await?;
                Ok(Response::new(res))
            }
            _ => Err(Status::invalid_argument("Invalid message payload")),
        }
    }
}
