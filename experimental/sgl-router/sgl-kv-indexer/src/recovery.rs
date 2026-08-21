// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Multi-worker Bridge recovery: subscribe Live first, install Snapshot at its
//! barrier, then apply a fenced contiguous stream to the paired Indexer.

use std::collections::HashSet;
use std::time::Duration;

use serde::Deserialize;
use tonic::transport::{Channel, Endpoint};
use tracing::{info, warn};
use zeromq::{Socket, SocketRecv, SubSocket};

use crate::bridge::{build_apply_request, decode_event_batch, BridgeConfig, BridgeError};
use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::{
    ConfigureExpectedWorkersRequest, ExpectedWorker, InvalidateWorkerRequest,
    ReplaceExternalKvSnapshotRequest, TierHashes, TierType,
};
use crate::service::{MAX_ACTIONS_PER_BATCH, MAX_HASHES_PER_REQUEST};
use crate::snapshot::fetch_snapshot;

const RECONNECT_BASE: Duration = Duration::from_millis(100);
const RECONNECT_CAP: Duration = Duration::from_secs(2);
const SUBSCRIPTION_SETTLE: Duration = Duration::from_millis(50);
const BARRIER_TIMEOUT: Duration = Duration::from_secs(5);
const INDEXER_EPOCH_CHECK_INTERVAL: Duration = Duration::from_secs(1);
const EPOCH_MARKER: &[u8] = b"\x00sgl-kv-epoch=";
const BARRIER_MARKER: &[u8] = b"\x00sgl-kv-snapshot=";

#[derive(Debug, Clone, Deserialize)]
pub struct BridgeWorkerConfig {
    pub worker_id: String,
    pub worker_address: String,
    pub event_endpoint: String,
    pub snapshot_endpoint: String,
    #[serde(default)]
    pub event_topic: String,
    #[serde(default)]
    pub dp_rank: u32,
}

#[derive(Debug, Clone)]
pub struct BridgeFleetConfig {
    pub indexer_endpoint: String,
    pub workers: Vec<BridgeWorkerConfig>,
}

impl BridgeFleetConfig {
    /// Returns `None` when fleet mode is not configured, allowing the legacy
    /// single-worker environment contract to remain backward compatible.
    pub fn from_env() -> Result<Option<Self>, BridgeError> {
        let raw = match std::env::var("KV_INDEXER_WORKERS_JSON") {
            Ok(raw) if !raw.trim().is_empty() => raw,
            _ => return Ok(None),
        };
        let workers: Vec<BridgeWorkerConfig> = serde_json::from_str(&raw).map_err(|error| {
            BridgeError::Config(format!("invalid KV_INDEXER_WORKERS_JSON: {error}"))
        })?;
        if workers.is_empty() {
            return Err(BridgeError::Config(
                "KV_INDEXER_WORKERS_JSON must contain at least one worker".into(),
            ));
        }
        let mut ids = HashSet::new();
        for worker in &workers {
            if worker.worker_id.is_empty()
                || worker.worker_address.is_empty()
                || worker.event_endpoint.is_empty()
                || worker.snapshot_endpoint.is_empty()
            {
                return Err(BridgeError::Config(
                    "worker_id, worker_address, event_endpoint and snapshot_endpoint are required"
                        .into(),
                ));
            }
            if !ids.insert(worker.worker_id.as_str()) {
                return Err(BridgeError::Config(format!(
                    "duplicate worker_id {}",
                    worker.worker_id
                )));
            }
        }
        Ok(Some(Self {
            indexer_endpoint: std::env::var("KV_INDEXER_ENDPOINT")
                .unwrap_or_else(|_| "http://[::1]:50051".into()),
            workers,
        }))
    }
}

pub async fn run_recoverable_bridge_fleet_until<F>(
    config: BridgeFleetConfig,
    shutdown: F,
) -> Result<(), BridgeError>
where
    F: std::future::Future<Output = ()>,
{
    configure_workers(&config).await?;
    let mut tasks = tokio::task::JoinSet::new();
    for worker in config.workers.clone() {
        let indexer = config.indexer_endpoint.clone();
        let expected = config.workers.clone();
        tasks.spawn(async move { supervise_worker(indexer, worker, expected).await });
    }
    tokio::pin!(shutdown);
    tokio::select! {
        _ = &mut shutdown => {
            tasks.abort_all();
            while tasks.join_next().await.is_some() {}
            Ok(())
        }
        result = tasks.join_next() => match result {
            Some(Ok(result)) => result,
            Some(Err(error)) => Err(BridgeError::Config(format!("worker recovery task failed: {error}"))),
            None => Err(BridgeError::Config("worker recovery fleet exited".into())),
        }
    }
}

async fn configure_workers(config: &BridgeFleetConfig) -> Result<(), BridgeError> {
    configure_worker_list(&config.indexer_endpoint, &config.workers)
        .await
        .map(|_| ())
}

async fn supervise_worker(
    indexer_endpoint: String,
    worker: BridgeWorkerConfig,
    expected_workers: Vec<BridgeWorkerConfig>,
) -> Result<(), BridgeError> {
    let mut delay = RECONNECT_BASE;
    loop {
        // Re-establish the complete desired Worker set after an Indexer restart.
        let indexer_epoch = match configure_worker_list(&indexer_endpoint, &expected_workers).await
        {
            Ok(epoch) => epoch,
            Err(error) => {
                if error.is_permanent() {
                    return Err(error);
                }
                warn!(worker_id = %worker.worker_id, %error, retry_in = ?delay, "failed to configure restarted Indexer");
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(RECONNECT_CAP);
                continue;
            }
        };
        if let Err(error) = invalidate_worker(&indexer_endpoint, &worker.worker_id).await {
            if error.is_permanent() {
                return Err(error);
            }
            warn!(worker_id = %worker.worker_id, %error, retry_in = ?delay, "failed to invalidate Worker before recovery");
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(RECONNECT_CAP);
            continue;
        }
        match recover_and_stream(
            &indexer_endpoint,
            &worker,
            &expected_workers,
            &indexer_epoch,
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(error) if error.is_permanent() => return Err(error),
            Err(error) => {
                warn!(worker_id = %worker.worker_id, %error, retry_in = ?delay, "worker stream lost; rebuilding from snapshot");
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(RECONNECT_CAP);
            }
        }
    }
}

async fn invalidate_worker(indexer_endpoint: &str, worker_id: &str) -> Result<(), BridgeError> {
    let mut client = connect_indexer(indexer_endpoint).await?;
    client
        .invalidate_worker(InvalidateWorkerRequest {
            worker_id: worker_id.to_owned(),
        })
        .await
        .map_err(super::bridge::classify_rpc)?;
    Ok(())
}

async fn configure_worker_list(
    indexer_endpoint: &str,
    workers: &[BridgeWorkerConfig],
) -> Result<String, BridgeError> {
    let mut client = connect_indexer(indexer_endpoint).await?;
    let response = client
        .configure_expected_workers(ConfigureExpectedWorkersRequest {
            workers: workers
                .iter()
                .map(|worker| ExpectedWorker {
                    worker_id: worker.worker_id.clone(),
                    worker_address: worker.worker_address.clone(),
                    cache_spec: None,
                })
                .collect(),
        })
        .await
        .map_err(super::bridge::classify_rpc)?;
    Ok(response.into_inner().indexer_epoch)
}

async fn recover_and_stream(
    indexer_endpoint: &str,
    worker: &BridgeWorkerConfig,
    expected_workers: &[BridgeWorkerConfig],
    indexer_epoch: &str,
) -> Result<(), BridgeError> {
    let mut client = connect_indexer(indexer_endpoint).await?;
    let mut subscriber = SubSocket::new();
    subscriber.subscribe(&worker.event_topic).await?;
    subscriber.connect(&worker.event_endpoint).await?;
    // ZeroMQ has no subscription acknowledgement. Give the TCP subscription
    // handshake a bounded head start before asking the worker to emit a barrier.
    tokio::time::sleep(SUBSCRIPTION_SETTLE).await;

    let snapshot = fetch_snapshot(&worker.snapshot_endpoint, worker.dp_rank)
        .await
        .map_err(|error| BridgeError::Decode(error.to_string()))?;
    let epoch = snapshot.header.epoch.clone();
    let barrier_seq = snapshot.header.barrier_seq as u64;
    let barrier_id = snapshot.header.barrier_id.clone();

    // Ignore pre-cut live data and wait until the exact snapshot barrier is
    // observed on the already-connected SUB stream.
    tokio::time::timeout(BARRIER_TIMEOUT, async {
        loop {
            let message = subscriber.recv().await?;
            let frame = parse_live_message(&message.into_vec())?;
            if frame.epoch.as_deref() != Some(epoch.as_str()) {
                continue;
            }
            if frame.seq == barrier_seq && frame.barrier_id.as_deref() == Some(barrier_id.as_str())
            {
                return Ok::<(), BridgeError>(());
            }
            if frame.seq > barrier_seq {
                return Err(BridgeError::Decode(
                    "snapshot barrier was missed on the live stream".into(),
                ));
            }
        }
    })
    .await
    .map_err(|_| BridgeError::Decode("timed out waiting for snapshot barrier".into()))??;

    let mut hashes = Vec::with_capacity(snapshot.blocks.len());
    for block in snapshot.blocks {
        hashes.extend(block.block_hashes);
    }
    client
        .replace_external_kv_snapshot(ReplaceExternalKvSnapshotRequest {
            worker_id: worker.worker_id.clone(),
            worker_address: worker.worker_address.clone(),
            worker_epoch: epoch.clone(),
            applied_seq: barrier_seq,
            hashes_by_tier: if hashes.is_empty() {
                Vec::new()
            } else {
                vec![TierHashes {
                    tier: TierType::TierHbm as i32,
                    // snapshot-v1 carries no component metadata; an empty side
                    // array preserves the legacy whole-block semantics.
                    component_masks: Vec::new(),
                    block_sizes: Vec::new(),
                    hashes,
                }]
            },
            cache_spec: None,
        })
        .await
        .map_err(super::bridge::classify_rpc)?;
    info!(worker_id = %worker.worker_id, epoch = %epoch, barrier_seq, "worker snapshot installed; stream READY");

    let bridge_config = BridgeConfig {
        worker_id: worker.worker_id.clone(),
        worker_address: worker.worker_address.clone(),
        event_endpoint: worker.event_endpoint.clone(),
        event_topic: worker.event_topic.clone(),
        indexer_endpoint: indexer_endpoint.to_owned(),
        clear_tiers: vec![
            TierType::TierHbm as i32,
            TierType::TierDram as i32,
            TierType::TierSsd as i32,
        ],
        cache_spec: None,
    };
    let mut expected = barrier_seq.saturating_add(1);
    let mut epoch_check = tokio::time::interval(INDEXER_EPOCH_CHECK_INTERVAL);
    epoch_check.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    epoch_check.tick().await;
    loop {
        let message = tokio::select! {
            message = subscriber.recv() => message?,
            _ = epoch_check.tick() => {
                let current = configure_worker_list(indexer_endpoint, expected_workers).await?;
                if current != indexer_epoch {
                    return Err(BridgeError::Decode("paired Indexer process restarted".into()));
                }
                continue;
            }
        };
        let frame = parse_live_message(&message.into_vec())?;
        if frame.epoch.as_deref() != Some(epoch.as_str()) {
            return Err(BridgeError::Decode("worker epoch changed".into()));
        }
        if frame.seq != expected {
            return Err(BridgeError::Decode(format!(
                "event sequence gap: expected {expected}, got {}",
                frame.seq
            )));
        }
        let actions = decode_event_batch(&frame.payload)?;
        let mut request = build_apply_request(&bridge_config, frame.seq, actions);
        let total_hashes: usize = request
            .actions
            .iter()
            .map(|action| action.hashes.len())
            .sum();
        if request.actions.len() > MAX_ACTIONS_PER_BATCH || total_hashes > MAX_HASHES_PER_REQUEST {
            return Err(BridgeError::Decode(
                "recovery-aware event batch exceeds one atomic apply request".into(),
            ));
        }
        request.worker_epoch = epoch.clone();
        request.enforce_sequence = true;
        client
            .apply_external_kv_batch(request)
            .await
            .map_err(super::bridge::classify_rpc)?;
        expected = expected.saturating_add(1);
    }
}

async fn connect_indexer(endpoint: &str) -> Result<KvIndexerClient<Channel>, BridgeError> {
    let channel = Endpoint::from_shared(endpoint.to_owned())?
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .connect()
        .await?;
    Ok(KvIndexerClient::new(channel))
}

struct LiveFrame {
    seq: u64,
    epoch: Option<String>,
    barrier_id: Option<String>,
    payload: Vec<u8>,
}

fn parse_live_message(frames: &[bytes::Bytes]) -> Result<LiveFrame, BridgeError> {
    if frames.len() != 3 {
        return Err(BridgeError::Decode(format!(
            "expected 3 recovery frames, got {}",
            frames.len()
        )));
    }
    let seq = u64::from_be_bytes(
        frames[1]
            .as_ref()
            .try_into()
            .map_err(|_| BridgeError::Decode("sequence frame must be 8 bytes".into()))?,
    );
    Ok(LiveFrame {
        seq,
        epoch: topic_metadata(&frames[0], EPOCH_MARKER),
        barrier_id: topic_metadata(&frames[0], BARRIER_MARKER),
        payload: frames[2].to_vec(),
    })
}

fn topic_metadata(topic: &[u8], marker: &[u8]) -> Option<String> {
    let start = topic
        .windows(marker.len())
        .position(|window| window == marker)?
        + marker.len();
    let tail = &topic[start..];
    let end = tail
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(tail.len());
    (end > 0)
        .then(|| std::str::from_utf8(&tail[..end]).ok().map(str::to_owned))
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use bytes::Bytes;
    use rmpv::Value;
    use zeromq::{Endpoint, PubSocket, RouterSocket, SocketSend, ZmqMessage};

    use crate::pb::kv_indexer_client::KvIndexerClient;
    use crate::pb::MatchExternalKvPrefixRequest;
    use crate::{server_builder, InMemoryKvIndexerBackend, KvIndexerBackend, KvIndexerService};

    #[test]
    fn parses_snapshot_topic_metadata() {
        let topic = b"kv\0sgl-kv-epoch=e1\0sgl-kv-snapshot=b1";
        assert_eq!(topic_metadata(topic, EPOCH_MARKER).as_deref(), Some("e1"));
        assert_eq!(topic_metadata(topic, BARRIER_MARKER).as_deref(), Some("b1"));
    }

    fn encode_value(value: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, value).unwrap();
        out
    }

    fn stored_batch(hash: i64) -> Vec<u8> {
        encode_value(&Value::Array(vec![
            Value::from(1.0_f64),
            Value::Array(vec![Value::Array(vec![
                Value::String("BlockStored".into()),
                Value::Array(vec![Value::from(hash)]),
                Value::Nil,
                Value::Array(Vec::new()),
                Value::from(1),
                Value::Nil,
                Value::String("GPU".into()),
            ])]),
            Value::from(0),
        ]))
    }

    async fn send_pub(pub_socket: &mut PubSocket, topic: &[u8], seq: u64, payload: Vec<u8>) {
        let mut message = ZmqMessage::from(Bytes::copy_from_slice(topic));
        message.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
        message.push_back(Bytes::from(payload));
        pub_socket.send(message).await.unwrap();
    }

    async fn send_snapshot_reply(
        router: &mut RouterSocket,
        identity: Bytes,
        kind: &'static [u8],
        payload: Vec<u8>,
    ) {
        let mut reply = ZmqMessage::from(identity);
        reply.push_back(Bytes::new());
        reply.push_back(Bytes::from_static(kind));
        reply.push_back(Bytes::from(payload));
        router.send(reply).await.unwrap();
    }

    fn start_indexer_server(addr: std::net::SocketAddr) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let backend: Arc<dyn KvIndexerBackend> = Arc::new(InMemoryKvIndexerBackend::new());
            server_builder()
                .add_service(KvIndexerService::new(backend).into_server())
                .serve(addr)
                .await
                .unwrap();
        })
    }

    async fn connect_indexer_client(addr: std::net::SocketAddr) -> KvIndexerClient<Channel> {
        let endpoint = format!("http://{addr}");
        for _ in 0..100 {
            if let Ok(client) = KvIndexerClient::connect(endpoint.clone()).await {
                return client;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("failed to connect to Indexer at {endpoint}");
    }

    async fn wait_for_prefix(client: &mut KvIndexerClient<Channel>, hashes: Vec<i64>) {
        for _ in 0..150 {
            let response = client
                .match_external_kv_prefix(MatchExternalKvPrefixRequest {
                    hashes: hashes.clone(),
                    max_blocks: 0,
                })
                .await
                .unwrap()
                .into_inner();
            if response.best_prefix_blocks as usize == hashes.len() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("snapshot plus live events did not restore the expected prefix");
    }

    #[tokio::test]
    async fn snapshot_then_live_event_restores_indexer_state() {
        let mut publisher = PubSocket::new();
        let pub_endpoint = publisher.bind("tcp://127.0.0.1:0").await.unwrap();
        let pub_port = match pub_endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };

        let mut snapshot_router = RouterSocket::new();
        let snapshot_endpoint = snapshot_router.bind("tcp://127.0.0.1:0").await.unwrap();
        let snapshot_port = match snapshot_endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let indexer_addr = listener.local_addr().unwrap();
        drop(listener);
        let indexer_server = start_indexer_server(indexer_addr);
        tokio::time::sleep(Duration::from_millis(30)).await;

        let worker = tokio::spawn(async move {
            let request = snapshot_router.recv().await.unwrap();
            let identity = request.get(0).unwrap().clone();
            send_snapshot_reply(
                &mut snapshot_router,
                identity.clone(),
                b"header",
                rmp_serde::to_vec(&(1_u32, "e1", 0_u32, 1_i64, 0_i64, "b1", 1_usize)).unwrap(),
            )
            .await;
            send_snapshot_reply(
                &mut snapshot_router,
                identity.clone(),
                b"chunk",
                rmp_serde::to_vec(&vec![(Option::<i64>::None, vec![1_i64])]).unwrap(),
            )
            .await;
            send_snapshot_reply(&mut snapshot_router, identity, b"end", Vec::new()).await;

            let barrier_topic = b"kv\0sgl-kv-epoch=e1\0sgl-kv-snapshot=b1";
            send_pub(&mut publisher, barrier_topic, 0, Vec::new()).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
            send_pub(&mut publisher, b"kv\0sgl-kv-epoch=e1", 1, stored_batch(2)).await;
        });

        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
        let fleet = tokio::spawn(run_recoverable_bridge_fleet_until(
            BridgeFleetConfig {
                indexer_endpoint: format!("http://{indexer_addr}"),
                workers: vec![BridgeWorkerConfig {
                    worker_id: "w1".into(),
                    worker_address: "http://w1".into(),
                    event_endpoint: format!("tcp://127.0.0.1:{pub_port}"),
                    snapshot_endpoint: format!("tcp://127.0.0.1:{snapshot_port}"),
                    event_topic: "kv".into(),
                    dp_rank: 0,
                }],
            },
            async {
                let _ = stop_rx.await;
            },
        ));

        let mut client = connect_indexer_client(indexer_addr).await;
        wait_for_prefix(&mut client, vec![1, 2]).await;

        let _ = stop_tx.send(());
        fleet.await.unwrap().unwrap();
        worker.await.unwrap();
        indexer_server.abort();
    }

    #[tokio::test]
    async fn idle_indexer_restart_is_detected_and_restored_from_a_new_snapshot() {
        let mut publisher = PubSocket::new();
        let pub_endpoint = publisher.bind("tcp://127.0.0.1:0").await.unwrap();
        let pub_port = match pub_endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };

        let mut snapshot_router = RouterSocket::new();
        let snapshot_endpoint = snapshot_router.bind("tcp://127.0.0.1:0").await.unwrap();
        let snapshot_port = match snapshot_endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let indexer_addr = listener.local_addr().unwrap();
        drop(listener);
        let mut indexer_server = start_indexer_server(indexer_addr);
        let mut client = connect_indexer_client(indexer_addr).await;

        let worker = tokio::spawn(async move {
            for cycle in 0..2 {
                let request = snapshot_router.recv().await.unwrap();
                let identity = request.get(0).unwrap().clone();
                let barrier_seq = if cycle == 0 { 0_i64 } else { 2_i64 };
                let blocks = if cycle == 0 {
                    vec![(Option::<i64>::None, vec![1_i64])]
                } else {
                    vec![
                        (Option::<i64>::None, vec![1_i64]),
                        (Some(1_i64), vec![2_i64]),
                    ]
                };
                let barrier_id = format!("b{cycle}");
                send_snapshot_reply(
                    &mut snapshot_router,
                    identity.clone(),
                    b"header",
                    rmp_serde::to_vec(&(
                        1_u32,
                        "e1",
                        0_u32,
                        barrier_seq + 1,
                        barrier_seq,
                        barrier_id.as_str(),
                        blocks.len(),
                    ))
                    .unwrap(),
                )
                .await;
                send_snapshot_reply(
                    &mut snapshot_router,
                    identity.clone(),
                    b"chunk",
                    rmp_serde::to_vec(&blocks).unwrap(),
                )
                .await;
                send_snapshot_reply(&mut snapshot_router, identity, b"end", Vec::new()).await;

                let barrier_topic = format!("kv\0sgl-kv-epoch=e1\0sgl-kv-snapshot={barrier_id}");
                send_pub(
                    &mut publisher,
                    barrier_topic.as_bytes(),
                    barrier_seq as u64,
                    Vec::new(),
                )
                .await;
                if cycle == 0 {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    send_pub(&mut publisher, b"kv\0sgl-kv-epoch=e1", 1, stored_batch(2)).await;
                }
            }
        });

        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
        let fleet = tokio::spawn(run_recoverable_bridge_fleet_until(
            BridgeFleetConfig {
                indexer_endpoint: format!("http://{indexer_addr}"),
                workers: vec![BridgeWorkerConfig {
                    worker_id: "w1".into(),
                    worker_address: "http://w1".into(),
                    event_endpoint: format!("tcp://127.0.0.1:{pub_port}"),
                    snapshot_endpoint: format!("tcp://127.0.0.1:{snapshot_port}"),
                    event_topic: "kv".into(),
                    dp_rank: 0,
                }],
            },
            async {
                let _ = stop_rx.await;
            },
        ));

        wait_for_prefix(&mut client, vec![1, 2]).await;
        indexer_server.abort();
        let _ = indexer_server.await;

        indexer_server = start_indexer_server(indexer_addr);
        client = connect_indexer_client(indexer_addr).await;
        wait_for_prefix(&mut client, vec![1, 2]).await;

        let _ = stop_tx.send(());
        fleet.await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(2), worker)
            .await
            .expect("Bridge should request a second snapshot")
            .unwrap();
        indexer_server.abort();
    }
}
