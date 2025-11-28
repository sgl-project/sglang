use std::{
    net::SocketAddr,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use futures::Stream;
use tokio_stream::StreamExt;
use tonic::{transport::Server, Response, Status};
use tracing as log;
use tracing::instrument;

use super::{
    crdt::SKey,
    flow_control::MessageSizeValidator,
    gossip::{
        self,
        gossip_server::{Gossip, GossipServer},
        GossipMessage, IncrementalUpdate, NodeState, NodeStatus, NodeUpdate, PingReq,
        SnapshotChunk, SnapshotRequest, StateUpdate, StreamAck, StreamMessage, StreamMessageType,
    },
    incremental::IncrementalUpdateCollector,
    metrics::{
        record_ack, record_batch_sent, record_nack, record_peer_reconnect, record_snapshot_bytes,
        record_snapshot_duration, record_snapshot_trigger, update_peer_connections,
        ConvergenceTracker,
    },
    node_state_machine::NodeStateMachine,
    partition::PartitionDetector,
    stores::{StateStores, StoreType as LocalStoreType},
    sync::MeshSyncManager,
    try_ping, ClusterState,
};

#[derive(Debug)]
pub struct GossipService {
    state: ClusterState,
    self_addr: SocketAddr,
    self_name: String,
    stores: Option<Arc<StateStores>>, // Optional state stores for CRDT-based sync
    sync_manager: Option<Arc<MeshSyncManager>>, // Optional sync manager for applying remote updates
    state_machine: Option<Arc<NodeStateMachine>>,
    partition_detector: Option<Arc<PartitionDetector>>,
}

impl GossipService {
    /// Create snapshot chunks for a store
    async fn create_snapshot_chunks(
        &self,
        store_type: LocalStoreType,
        chunk_size: usize,
    ) -> Vec<SnapshotChunk> {
        let stores = match self.stores.as_ref() {
            Some(s) => s,
            None => {
                log::warn!("State stores not available for snapshot generation");
                return vec![];
            }
        };

        let proto_store_type = match store_type {
            LocalStoreType::Membership => gossip::StoreType::Membership as i32,
            LocalStoreType::App => gossip::StoreType::App as i32,
            LocalStoreType::Worker => gossip::StoreType::Worker as i32,
            LocalStoreType::Policy => gossip::StoreType::Policy as i32,
            LocalStoreType::RateLimit => gossip::StoreType::RateLimit as i32,
        };

        // Get all entries from the store
        let entries: Vec<(SKey, Vec<u8>)> = match store_type {
            LocalStoreType::Membership => stores
                .membership
                .all()
                .into_iter()
                .map(|(k, v)| {
                    let serialized = serde_json::to_vec(&v).unwrap_or_else(|e| {
                        log::error!("Failed to serialize membership state: {}", e);
                        vec![]
                    });
                    (k, serialized)
                })
                .collect(),
            LocalStoreType::App => stores
                .app
                .all()
                .into_iter()
                .map(|(k, v)| {
                    let serialized = serde_json::to_vec(&v).unwrap_or_else(|e| {
                        log::error!("Failed to serialize app state: {}", e);
                        vec![]
                    });
                    (k, serialized)
                })
                .collect(),
            LocalStoreType::Worker => stores
                .worker
                .all()
                .into_iter()
                .map(|(k, v)| {
                    let serialized = serde_json::to_vec(&v).unwrap_or_else(|e| {
                        log::error!("Failed to serialize worker state: {}", e);
                        vec![]
                    });
                    (k, serialized)
                })
                .collect(),
            LocalStoreType::Policy => stores
                .policy
                .all()
                .into_iter()
                .map(|(k, v)| {
                    let serialized = serde_json::to_vec(&v).unwrap_or_else(|e| {
                        log::error!("Failed to serialize policy state: {}", e);
                        vec![]
                    });
                    (k, serialized)
                })
                .collect(),
            LocalStoreType::RateLimit => {
                // For rate limit, serialize all counters from owners
                stores
                    .rate_limit
                    .keys()
                    .into_iter()
                    .filter_map(|key| {
                        if stores.rate_limit.is_owner(&key) {
                            stores.rate_limit.get_counter(&key).map(|counter| {
                                let serialized = serde_json::to_vec(&counter.snapshot())
                                    .unwrap_or_else(|e| {
                                        log::error!(
                                            "Failed to serialize rate limit counter: {}",
                                            e
                                        );
                                        vec![]
                                    });
                                (SKey::new(key.clone()), serialized)
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            }
        };

        if entries.is_empty() {
            return vec![];
        }

        // Split entries into chunks
        let mut chunks = Vec::new();
        let total_chunks = entries.len().div_ceil(chunk_size);

        for (chunk_idx, chunk_entries) in entries.chunks(chunk_size).enumerate() {
            let state_updates: Vec<StateUpdate> = chunk_entries
                .iter()
                .map(|(key, value)| {
                    // Get actual version from CRDT metadata
                    let version = match store_type {
                        LocalStoreType::Membership => stores
                            .membership
                            .get_metadata(key)
                            .map(|(v, _)| v)
                            .unwrap_or(1),
                        LocalStoreType::App => {
                            stores.app.get_metadata(key).map(|(v, _)| v).unwrap_or(1)
                        }
                        LocalStoreType::Worker => {
                            stores.worker.get_metadata(key).map(|(v, _)| v).unwrap_or(1)
                        }
                        LocalStoreType::Policy => {
                            stores.policy.get_metadata(key).map(|(v, _)| v).unwrap_or(1)
                        }
                        LocalStoreType::RateLimit => {
                            // For rate limit, use timestamp as version
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as u64
                        }
                    };

                    StateUpdate {
                        key: key.as_str().to_string(),
                        value: value.clone(),
                        version,
                        actor: self.self_name.clone(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64,
                    }
                })
                .collect();

            // Calculate checksum for integrity verification
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for update in &state_updates {
                update.key.hash(&mut hasher);
                update.value.hash(&mut hasher);
            }
            let checksum = hasher.finish().to_le_bytes().to_vec();

            chunks.push(SnapshotChunk {
                store: proto_store_type,
                chunk_index: chunk_idx as u64,
                total_chunks: total_chunks as u64,
                entries: state_updates,
                checksum,
            });
        }

        log::info!(
            "Generated {} snapshot chunks for store {:?}",
            chunks.len(),
            store_type
        );
        chunks
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
            state_machine: None,
            partition_detector: None,
        }
    }

    pub fn with_stores(mut self, stores: Arc<StateStores>) -> Self {
        self.stores = Some(stores.clone());
        // Create state machine if stores are provided
        if self.state_machine.is_none() {
            use super::node_state_machine::ConvergenceConfig;
            self.state_machine = Some(Arc::new(NodeStateMachine::new(
                stores,
                ConvergenceConfig::default(),
            )));
        }
        self
    }

    pub fn with_sync_manager(mut self, sync_manager: Arc<MeshSyncManager>) -> Self {
        self.sync_manager = Some(sync_manager);
        self
    }

    pub fn with_partition_detector(mut self, partition_detector: Arc<PartitionDetector>) -> Self {
        self.partition_detector = Some(partition_detector);
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
    type SyncStreamStream =
        Pin<Box<dyn Stream<Item = Result<StreamMessage, Status>> + Send + 'static>>;

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

        // Create output stream with flow control
        const CHANNEL_CAPACITY: usize = 128;
        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<StreamMessage, Status>>(CHANNEL_CAPACITY);
        let size_validator = MessageSizeValidator::default();

        // Create incremental update collector if stores are available
        let collector = stores.as_ref().map(|stores| {
            Arc::new(IncrementalUpdateCollector::new(
                stores.clone(),
                self_name.clone(),
            ))
        });

        // Spawn task to periodically send incremental updates
        if let Some(collector) = collector {
            let tx_incremental = tx.clone();
            let self_name_incremental = self_name.clone();
            let size_validator_clone = size_validator.clone();
            tokio::spawn(async move {
                // Use 1 second interval for rate limit counter sync (faster than other stores)
                let mut interval = tokio::time::interval(Duration::from_secs(1)); // Send every 1 second
                let mut sequence_counter: u64 = 0;

                loop {
                    interval.tick().await;

                    // Collect all incremental updates
                    let all_updates = collector.collect_all_updates();

                    if !all_updates.is_empty() {
                        for (store_type, updates) in all_updates {
                            let proto_store_type = match store_type {
                                LocalStoreType::Membership => gossip::StoreType::Membership as i32,
                                LocalStoreType::App => gossip::StoreType::App as i32,
                                LocalStoreType::Worker => gossip::StoreType::Worker as i32,
                                LocalStoreType::Policy => gossip::StoreType::Policy as i32,
                                LocalStoreType::RateLimit => gossip::StoreType::RateLimit as i32,
                            };

                            sequence_counter += 1;
                            let batch_size: usize = updates.iter().map(|u| u.value.len()).sum();

                            // Validate message size
                            if let Err(e) = size_validator_clone.validate(batch_size) {
                                log::warn!(
                                    "Incremental update too large, skipping: {} (max: {} bytes)",
                                    e,
                                    size_validator_clone.max_size()
                                );
                                continue;
                            }

                            let incremental_update = StreamMessage {
                                message_type: StreamMessageType::IncrementalUpdate as i32,
                                payload: Some(gossip::stream_message::Payload::Incremental(
                                    IncrementalUpdate {
                                        store: proto_store_type,
                                        updates: updates.clone(),
                                        version: 0, // Version is tracked per key in StateUpdate
                                    },
                                )),
                                sequence: sequence_counter,
                                peer_id: self_name_incremental.clone(),
                            };

                            // Check backpressure using try_send (mpsc::Sender doesn't have len())
                            match tx_incremental.try_send(Ok(incremental_update)) {
                                Ok(_) => {
                                    // Successfully queued
                                    // Record metrics
                                    record_batch_sent(&self_name_incremental, batch_size);
                                    // Mark as sent after successful transmission
                                    collector.mark_sent(store_type, &updates);
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                                    log::debug!(
                                        "Backpressure: channel full, skipping send (will retry next interval)"
                                    );
                                    // Don't mark as sent, will retry next interval
                                    continue;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                    log::warn!(
                                        "Channel closed, stopping incremental update sender"
                                    );
                                    break;
                                }
                            }

                            log::debug!(
                                "Sent incremental update: store={:?}, {} updates",
                                store_type,
                                updates.len()
                            );
                        }
                    }
                }
            });
        }

        // Spawn task to handle incoming messages
        let mut sequence: u64 = 0;
        let _convergence_tracker = ConvergenceTracker::new();

        // Track snapshot reception state: (store_type, total_chunks) -> received_chunks
        use std::collections::HashMap;
        let mut snapshot_state: HashMap<(LocalStoreType, u64), Vec<SnapshotChunk>> = HashMap::new();

        tokio::spawn(async move {
            let mut peer_id = String::new();
            update_peer_connections(&peer_id, true);

            // Check if we need to request snapshots on connection
            // This happens when:
            // 1. We're a new node joining (stores are empty or very small)
            // 2. We detect a version gap
            if let Some(ref stores) = stores {
                for store_type in [
                    LocalStoreType::Membership,
                    LocalStoreType::App,
                    LocalStoreType::Worker,
                    LocalStoreType::Policy,
                    LocalStoreType::RateLimit,
                ] {
                    let store_len = match store_type {
                        LocalStoreType::Membership => stores.membership.len(),
                        LocalStoreType::App => stores.app.len(),
                        LocalStoreType::Worker => stores.worker.len(),
                        LocalStoreType::Policy => stores.policy.len(),
                        LocalStoreType::RateLimit => stores.rate_limit.keys().len(),
                    };

                    // If store is empty or very small, request snapshot
                    if store_len == 0 {
                        log::info!(
                            "Store {:?} is empty, requesting snapshot from {}",
                            store_type,
                            peer_id
                        );
                        let proto_store_type = match store_type {
                            LocalStoreType::Membership => gossip::StoreType::Membership as i32,
                            LocalStoreType::App => gossip::StoreType::App as i32,
                            LocalStoreType::Worker => gossip::StoreType::Worker as i32,
                            LocalStoreType::Policy => gossip::StoreType::Policy as i32,
                            LocalStoreType::RateLimit => gossip::StoreType::RateLimit as i32,
                        };

                        let snapshot_request = StreamMessage {
                            message_type: StreamMessageType::SnapshotRequest as i32,
                            payload: Some(gossip::stream_message::Payload::SnapshotRequest(
                                SnapshotRequest {
                                    store: proto_store_type,
                                    from_version: 0, // Request from beginning
                                },
                            )),
                            sequence: 0,
                            peer_id: self_name.clone(),
                        };

                        if tx.send(Ok(snapshot_request)).await.is_err() {
                            log::warn!("Failed to send snapshot request");
                        }
                    }
                }
            }

            while let Some(msg_result) = incoming.next().await {
                match msg_result {
                    Ok(msg) => {
                        sequence += 1;
                        peer_id = msg.peer_id.clone();

                        match msg.message_type() {
                            StreamMessageType::IncrementalUpdate => {
                                if let Some(gossip::stream_message::Payload::Incremental(update)) =
                                    &msg.payload
                                {
                                    // Validate message size
                                    let msg_size: usize =
                                        update.updates.iter().map(|u| u.value.len()).sum();
                                    if let Err(e) = size_validator.validate(msg_size) {
                                        log::warn!(
                                            "Received oversized incremental update from {}: {} (max: {} bytes), rejecting",
                                            peer_id, e, size_validator.max_size()
                                        );
                                        let nack = StreamMessage {
                                            message_type: StreamMessageType::Nack as i32,
                                            payload: Some(gossip::stream_message::Payload::Ack(
                                                StreamAck {
                                                    sequence: msg.sequence,
                                                    success: false,
                                                    error_message: format!(
                                                        "Message too large: {}",
                                                        e
                                                    ),
                                                },
                                            )),
                                            sequence,
                                            peer_id: self_name.clone(),
                                        };
                                        if tx.send(Ok(nack)).await.is_err() {
                                            break;
                                        }
                                        record_nack(&peer_id);
                                        continue;
                                    }

                                    let store_type = LocalStoreType::from_proto(update.store);
                                    log::info!("Received incremental update from {}: store={:?}, {} updates",
                                        peer_id, store_type, update.updates.len());

                                    // Apply incremental updates to state stores
                                    // This will be handled by the sync manager if available
                                    // For now, we acknowledge and the sync manager will handle it
                                    if let Some(ref sync_manager) = sync_manager {
                                        for state_update in &update.updates {
                                            match store_type {
                                                LocalStoreType::Worker => {
                                                    // Deserialize and apply worker state
                                                    if let Ok(worker_state) = serde_json::from_slice::<
                                                        super::stores::WorkerState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        // Extract actor from StateUpdate
                                                        let actor =
                                                            Some(state_update.actor.clone());
                                                        sync_manager.apply_remote_worker_state(
                                                            worker_state,
                                                            actor,
                                                        );
                                                    }
                                                }
                                                LocalStoreType::Policy => {
                                                    // Deserialize and apply policy state
                                                    if let Ok(policy_state) = serde_json::from_slice::<
                                                        super::stores::PolicyState,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        // Extract actor from StateUpdate
                                                        let actor =
                                                            Some(state_update.actor.clone());

                                                        // Check if this is a tree state update
                                                        if policy_state.policy_type == "tree_state"
                                                        {
                                                            // Deserialize tree state
                                                            if let Ok(tree_state) =
                                                                serde_json::from_slice::<
                                                                    super::tree_ops::TreeState,
                                                                >(
                                                                    &policy_state.config
                                                                )
                                                            {
                                                                sync_manager
                                                                    .apply_remote_tree_operation(
                                                                        policy_state
                                                                            .model_id
                                                                            .clone(),
                                                                        tree_state,
                                                                        actor,
                                                                    );
                                                            }
                                                        } else {
                                                            // Regular policy state update
                                                            sync_manager.apply_remote_policy_state(
                                                                policy_state,
                                                                actor,
                                                            );
                                                        }
                                                    }
                                                }
                                                LocalStoreType::RateLimit => {
                                                    // Deserialize and apply rate limit counter
                                                    if let Ok(counter) = serde_json::from_slice::<
                                                        super::crdt::CRDTPNCounter,
                                                    >(
                                                        &state_update.value
                                                    ) {
                                                        // Convert CRDTPNCounter to SyncPNCounter for merging
                                                        let sync_counter =
                                                            super::crdt::SyncPNCounter::new();
                                                        sync_counter.merge(&counter);
                                                        sync_manager
                                                            .apply_remote_rate_limit_counter(
                                                                state_update.key.clone(),
                                                                &sync_counter,
                                                            );
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
                                        payload: Some(gossip::stream_message::Payload::Ack(
                                            StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::SnapshotRequest => {
                                if let Some(gossip::stream_message::Payload::SnapshotRequest(req)) =
                                    &msg.payload
                                {
                                    let store_type = LocalStoreType::from_proto(req.store);
                                    let store_name = store_type.as_str();
                                    log::info!("Received snapshot request from {}: store={:?}, from_version={}",
                                        peer_id, store_type, req.from_version);

                                    record_snapshot_trigger(store_name, "request");
                                    let snapshot_start = Instant::now();

                                    // Generate and send snapshot chunks
                                    let service = GossipService {
                                        state: state.clone(),
                                        self_addr: SocketAddr::from(([0, 0, 0, 0], 0)), // Not used in snapshot generation
                                        self_name: self_name.clone(),
                                        stores: stores.clone(),
                                        sync_manager: sync_manager.clone(),
                                        state_machine: None,
                                        partition_detector: None,
                                    };
                                    let chunks =
                                        service.create_snapshot_chunks(store_type, 100).await; // chunk_size = 100 entries
                                    let total_chunks = chunks.len() as u64;
                                    let mut total_bytes = 0;

                                    for (idx, chunk) in chunks.into_iter().enumerate() {
                                        let chunk_bytes = chunk
                                            .entries
                                            .iter()
                                            .map(|e| e.value.len())
                                            .sum::<usize>();
                                        total_bytes += chunk_bytes;

                                        let mut chunk_msg = StreamMessage {
                                            message_type: StreamMessageType::SnapshotChunk as i32,
                                            payload: Some(
                                                gossip::stream_message::Payload::SnapshotChunk(
                                                    chunk,
                                                ),
                                            ),
                                            sequence: sequence + idx as u64 + 1,
                                            peer_id: self_name.clone(),
                                        };
                                        // Update chunk metadata
                                        if let Some(
                                            gossip::stream_message::Payload::SnapshotChunk(
                                                ref mut c,
                                            ),
                                        ) = chunk_msg.payload
                                        {
                                            c.chunk_index = idx as u64;
                                            c.total_chunks = total_chunks;
                                        }

                                        // Check backpressure using try_send
                                        match tx.try_send(Ok(chunk_msg)) {
                                            Ok(_) => {
                                                // Successfully queued
                                            }
                                            Err(tokio::sync::mpsc::error::TrySendError::Full(
                                                msg,
                                            )) => {
                                                log::debug!(
                                                    "Backpressure: channel full, waiting for drain"
                                                );
                                                // Wait a bit for channel to drain, then use blocking send
                                                tokio::time::sleep(Duration::from_millis(100))
                                                    .await;
                                                if tx.send(msg).await.is_err() {
                                                    log::warn!("Backpressure: channel closed, stopping snapshot");
                                                    break;
                                                }
                                            }
                                            Err(
                                                tokio::sync::mpsc::error::TrySendError::Closed(_),
                                            ) => {
                                                log::warn!("Channel closed, stopping snapshot");
                                                break;
                                            }
                                        }
                                    }

                                    record_snapshot_duration(store_name, snapshot_start.elapsed());
                                    record_snapshot_bytes(store_name, "sent", total_bytes);

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
                                        payload: Some(gossip::stream_message::Payload::Ack(
                                            StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    record_ack(&peer_id, true);
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::SnapshotChunk => {
                                if let Some(gossip::stream_message::Payload::SnapshotChunk(chunk)) =
                                    &msg.payload
                                {
                                    let store_type = LocalStoreType::from_proto(chunk.store);
                                    let store_name = store_type.as_str();
                                    log::info!(
                                        "Received snapshot chunk from {}: store={:?}, chunk={}/{}",
                                        peer_id,
                                        store_type,
                                        chunk.chunk_index,
                                        chunk.total_chunks
                                    );

                                    // Record metrics
                                    let chunk_bytes: usize =
                                        chunk.entries.iter().map(|e| e.value.len()).sum();
                                    record_snapshot_bytes(store_name, "received", chunk_bytes);

                                    // Store chunk for later application
                                    let chunk_key = (store_type, chunk.total_chunks);
                                    snapshot_state
                                        .entry(chunk_key)
                                        .or_default()
                                        .push(chunk.clone());

                                    // Check if we've received all chunks
                                    if let Some(received_chunks) = snapshot_state.get(&chunk_key) {
                                        if received_chunks.len() as u64 == chunk.total_chunks {
                                            // All chunks received, apply snapshot
                                            log::info!("All {} chunks received for store {:?}, applying snapshot",
                                                chunk.total_chunks, store_type);

                                            if let Some(ref stores) = stores {
                                                // Sort chunks by index
                                                let mut sorted_chunks = received_chunks.clone();
                                                sorted_chunks.sort_by_key(|c| c.chunk_index);

                                                // Apply all entries from chunks
                                                for chunk in &sorted_chunks {
                                                    for entry in &chunk.entries {
                                                        let key = SKey::new(entry.key.clone());

                                                        match store_type {
                                                            LocalStoreType::Membership => {
                                                                if let Ok(membership_state) = serde_json::from_slice::<super::stores::MembershipState>(&entry.value) {
                                                                    stores.membership.insert(key, membership_state, entry.actor.clone());
                                                                }
                                                            }
                                                            LocalStoreType::App => {
                                                                if let Ok(app_state) = serde_json::from_slice::<super::stores::AppState>(&entry.value) {
                                                                    stores.app.insert(key, app_state, entry.actor.clone());
                                                                }
                                                            }
                                                            LocalStoreType::Worker => {
                                                                if let Ok(worker_state) = serde_json::from_slice::<super::stores::WorkerState>(&entry.value) {
                                                                    stores.worker.insert(key, worker_state.clone(), entry.actor.clone());
                                                                    // Also update sync manager if available
                                                                    if let Some(ref sync_manager) = sync_manager {
                                                                        sync_manager.apply_remote_worker_state(worker_state, Some(entry.actor.clone()));
                                                                    }
                                                                }
                                                            }
                                                            LocalStoreType::Policy => {
                                                                if let Ok(policy_state) = serde_json::from_slice::<super::stores::PolicyState>(&entry.value) {
                                                                    stores.policy.insert(key, policy_state.clone(), entry.actor.clone());
                                                                    // Also update sync manager if available
                                                                    if let Some(ref sync_manager) = sync_manager {
                                                                        // Check if this is a tree state update
                                                                        if policy_state.policy_type == "tree_state" {
                                                                            // Deserialize tree state
                                                                            if let Ok(tree_state) = serde_json::from_slice::<
                                                                                super::tree_ops::TreeState,
                                                                            >(
                                                                                &policy_state.config
                                                                            ) {
                                                                                sync_manager.apply_remote_tree_operation(
                                                                                    policy_state.model_id.clone(),
                                                                                    tree_state,
                                                                                    Some(entry.actor.clone()),
                                                                                );
                                                                            }
                                                                        } else {
                                                                            sync_manager.apply_remote_policy_state(policy_state, Some(entry.actor.clone()));
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            LocalStoreType::RateLimit => {
                                                                // For rate limit counters, deserialize and merge
                                                                if let Ok(counter) = serde_json::from_slice::<super::crdt::CRDTPNCounter>(&entry.value) {
                                                                    if let Some(ref sync_manager) = sync_manager {
                                                                        let sync_counter = super::crdt::SyncPNCounter::new();
                                                                        sync_counter.merge(&counter);
                                                                        sync_manager.apply_remote_rate_limit_counter(entry.key.clone(), &sync_counter);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }

                                                // Clear snapshot state
                                                snapshot_state.remove(&chunk_key);
                                                log::info!(
                                                    "Snapshot applied successfully for store {:?}",
                                                    store_type
                                                );
                                            }
                                        }
                                    }

                                    let ack = StreamMessage {
                                        message_type: StreamMessageType::Ack as i32,
                                        payload: Some(gossip::stream_message::Payload::Ack(
                                            StreamAck {
                                                sequence: msg.sequence,
                                                success: true,
                                                error_message: String::new(),
                                            },
                                        )),
                                        sequence,
                                        peer_id: self_name.clone(),
                                    };
                                    record_ack(&peer_id, true);
                                    if tx.send(Ok(ack)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            StreamMessageType::Ack => {
                                log::debug!(
                                    "Received ACK from {}: sequence={}",
                                    peer_id,
                                    msg.sequence
                                );
                                if let Some(gossip::stream_message::Payload::Ack(ack)) =
                                    &msg.payload
                                {
                                    record_ack(&peer_id, ack.success);
                                }
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
                                log::warn!(
                                    "Unknown message type from {}: {:?}",
                                    peer_id,
                                    msg.message_type
                                );
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Error receiving stream message: {}", e);
                        record_nack(&peer_id);
                        update_peer_connections(&peer_id, false);
                        record_peer_reconnect(&peer_id);
                        break;
                    }
                }
            }
            log::info!("Stream from {} closed", peer_id);
            update_peer_connections(&peer_id, false);
        });

        // Convert receiver to stream
        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(
            Box::pin(output_stream) as Self::SyncStreamStream
        ))
    }
}
