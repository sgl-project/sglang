use std::{net::SocketAddr, sync::Arc};

use anyhow::Result;
use tonic::{transport::Server, Response, Status};
use tracing as log;
use tracing::instrument;

use super::{
    gossip::{
        self,
        gossip_server::{Gossip, GossipServer},
        GossipMessage, NodeState, NodeStatus, NodeUpdate, PingReq,
        StreamMessage, StreamMessageType, IncrementalUpdate, SnapshotRequest, SnapshotChunk, StreamAck, StoreType,
        StateUpdate,
    },
    try_ping, ClusterState,
    stores::{StateStores, MembershipStore, AppStore, WorkerStore, PolicyStore},
    crdt::SKey,
    sync::HASyncManager,
};
use futures::Stream;
use std::pin::Pin;
use tokio_stream::StreamExt;
use serde::{Serialize, Deserialize};

#[derive(Debug)]
pub struct GossipService {
    state: ClusterState,
    self_addr: SocketAddr,
    self_name: String,
    stores: Option<StateStores>, // Optional state stores for CRDT-based sync
    sync_manager: Option<Arc<HASyncManager>>, // Optional sync manager for applying remote updates
}

impl GossipService {
    /// Create snapshot chunks for a store
    async fn create_snapshot_chunks(
        &self,
        store_type: StoreType,
        chunk_size: usize,
    ) -> Vec<SnapshotChunk> {
        // TODO: Implement actual snapshot generation from state stores
        // For now, return empty chunks
        vec![]
    }
}

impl GossipService {
    pub fn new(state: ClusterState, self_addr: SocketAddr, self_name: &str) -> Self {
        Self {
            state,
            self_addr,
            self_name: self_name.to_string(),
            stores: None,
            sync_manager: None,
        }
    }

    pub fn with_stores(mut self, stores: StateStores) -> Self {
        self.stores = Some(stores);
        self
    }

    pub fn with_sync_manager(mut self, sync_manager: Arc<HASyncManager>) -> Self {
        self.sync_manager = Some(sync_manager);
        self
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
    type SyncStreamStream = Pin<Box<dyn Stream<Item = Result<StreamMessage, Status>> + Send + 'static>>;

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
                // Return current status of self node (could be Alive or Leaving)
                let current_status = {
                    let state = self.state.read();
                    state
                        .get(&self.self_name)
                        .map(|n| n.status)
                        .unwrap_or(NodeStatus::Alive as i32)
                };
                Ok(Response::new(NodeUpdate {
                    name: self.self_name.clone(),
                    address: self.self_addr.to_string(),
                    status: current_status,
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

    #[instrument(fields(name = %self.self_name), skip(self, request))]
    async fn sync_stream(
        &self,
        request: tonic::Request<tonic::Streaming<StreamMessage>>,
    ) -> Result<Response<Self::SyncStreamStream>, Status> {
        let mut incoming = request.into_inner();
        let self_name = self.self_name.clone();
        let state = self.state.clone();
        let stores = self.stores.clone();
        let sync_manager = self.sync_manager.clone();

        // Create output stream
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamMessage, Status>>(128);

        // Spawn task to handle incoming messages
        let mut sequence: u64 = 0;
        tokio::spawn(async move {
            let mut peer_id = String::new();
            while let Some(msg_result) = incoming.next().await {
                match msg_result {
                    Ok(msg) => {
                        sequence += 1;
                        peer_id = msg.peer_id.clone();
                        
                        match msg.message_type() {
                            StreamMessageType::IncrementalUpdate => {
                                if let Some(gossip::stream_message::Payload::Incremental(update)) = &msg.payload {
                                    let store_type = StoreType::try_from(update.store).unwrap_or(StoreType::Membership);
                                    log::info!("Received incremental update from {}: store={:?}, {} updates", 
                                        peer_id, store_type, update.updates.len());
                                    
                                    // Apply incremental updates to state stores
                                    // This will be handled by the sync manager if available
                                    // For now, we acknowledge and the sync manager will handle it
                                    if let Some(ref sync_manager) = sync_manager {
                                        for state_update in &update.updates {
                                            match store_type {
                                                StoreType::Worker => {
                                                    // Deserialize and apply worker state
                                                    if let Ok(worker_state) = serde_json::from_slice::<super::stores::WorkerState>(&state_update.value) {
                                                        sync_manager.apply_remote_worker_state(worker_state);
                                                    }
                                                }
                                                StoreType::Policy => {
                                                    // Deserialize and apply policy state
                                                    if let Ok(policy_state) = serde_json::from_slice::<super::stores::PolicyState>(&state_update.value) {
                                                        sync_manager.apply_remote_policy_state(policy_state);
                                                    }
                                                }
                                                _ => {
                                                    // Other store types handled elsewhere
                                                }
                                            }
                                        }
                                    }
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(super::gossip::stream_message::Payload::Ack(StreamAck {
                                            sequence: msg.sequence,
                                            success: true,
                                            error_message: String::new(),
                                        })),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::SnapshotRequest => {
                                if let Some(gossip::stream_message::Payload::SnapshotRequest(req)) = &msg.payload {
                                    let store_type = StoreType::try_from(req.store).unwrap_or(StoreType::Membership);
                                    log::info!("Received snapshot request from {}: store={:?}, from_version={}", 
                                        peer_id, store_type, req.from_version);
                                    
                                    // Generate and send snapshot chunks
                                    // TODO: Implement actual snapshot generation
                                    // For now, send empty chunks
                                    let chunks = vec![];
                                    let total_chunks = chunks.len() as u64;
                                    
                                    for (idx, chunk) in chunks.into_iter().enumerate() {
                                        let mut chunk_msg = StreamMessage {
                                            message_type: StreamMessageType::SnapshotChunk as i32,
                                            payload: Some(gossip::stream_message::Payload::SnapshotChunk(chunk)),
                                            sequence: sequence + idx as u64 + 1,
                                            peer_id: self_name.clone(),
                                        };
                                        // Update chunk metadata
                                        if let Some(gossip::stream_message::Payload::SnapshotChunk(ref mut c)) = chunk_msg.payload {
                                            c.chunk_index = idx as u64;
                                            c.total_chunks = total_chunks;
                                        }
                                        
                                        if tx.send(Ok(chunk_msg)).await.is_err() {
                                            break;
                                        }
                                    }
                                    
                                    // Send snapshot complete message
                                    let complete = StreamMessage {
                                        message_type: StreamMessageType::SnapshotComplete as i32,
                                        payload: None,
                                        sequence: sequence + total_chunks + 1,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(Ok(complete)).await.is_err() {
                                        break;
                                    }
                                    
                                    // Send ACK
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(super::gossip::stream_message::Payload::Ack(StreamAck {
                                            sequence: msg.sequence,
                                            success: true,
                                            error_message: String::new(),
                                        })),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::SnapshotChunk => {
                                if let Some(gossip::stream_message::Payload::SnapshotChunk(chunk)) = &msg.payload {
                                    log::info!("Received snapshot chunk from {}: store={:?}, chunk={}/{}", 
                                        peer_id,
                                        StoreType::try_from(chunk.store).unwrap_or(StoreType::Membership),
                                        chunk.chunk_index, chunk.total_chunks);
                                    // TODO: Apply snapshot chunks
                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(super::gossip::stream_message::Payload::Ack(StreamAck {
                                            sequence: msg.sequence,
                                            success: true,
                                            error_message: String::new(),
                                        })),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::Ack => {
                                log::debug!("Received ACK from {}: sequence={}", peer_id, msg.sequence);
                            }
                            StreamMessageType::Heartbeat => {
                                // Send heartbeat back
                                let heartbeat = StreamMessage {
                                    message_type: StreamMessageType::Heartbeat as i32,
                                    payload: None,
                                    sequence,
                                    peer_id: self_name.clone(),
                                };
                                if tx.send(Ok(heartbeat)).await.is_err() {
                                    break;
                                }
                            }
                            _ => {
                                log::warn!("Unknown message type from {}: {:?}", peer_id, msg.message_type);
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Error receiving stream message: {}", e);
                        break;
                    }
                }
            }
            log::info!("Stream from {} closed", peer_id);
        });

        // Convert receiver to stream
        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream) as Self::SyncStreamStream))
    }
}
