//! One Bridge per Indexer. Worker discovery is independent of Router lifetime.
//! The Indexer owns all recovery decisions; this process forwards Worker cuts,
//! live barriers, replay envelopes and ordered mutations without changing IDs.

use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use serde::Deserialize;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tonic::transport::{Channel, Endpoint};
use tracing::{info, warn};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

use crate::bridge::decode_recoverable_batch;
use crate::pb::kv_replica_client::KvReplicaClient;
use crate::pb::*;

const IO_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_FRAME: usize = 8 * 1024 * 1024;
const BUFFER_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicaBridgeConfig {
    pub indexer_endpoint: String,
    pub worker_urls: Vec<String>,
    #[serde(default = "default_queue")]
    pub queue_capacity: usize,
    #[serde(default = "default_concurrency")]
    pub snapshot_concurrency: usize,
}
fn default_queue() -> usize {
    256
}
fn default_concurrency() -> usize {
    4
}

impl ReplicaBridgeConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.queue_capacity > 0 && self.queue_capacity <= 65536,
            "invalid queue capacity"
        );
        ensure!(
            self.snapshot_concurrency > 0 && self.snapshot_concurrency <= 64,
            "invalid snapshot concurrency"
        );
        ensure!(self.worker_urls.len() <= 4096, "too many workers");
        Endpoint::from_shared(self.indexer_endpoint.clone())?;
        for url in &self.worker_urls {
            reqwest::Url::parse(url)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveredStream {
    pub descriptor: StreamDescriptor,
    pub event_endpoint: String,
    pub snapshot_endpoint: String,
    pub replay_endpoint: String,
    pub topic: String,
}

pub async fn discover(client: &reqwest::Client, url: &str) -> Result<Vec<DiscoveredStream>> {
    let body: serde_json::Value = client
        .get(format!("{}/server_info", url.trim_end_matches('/')))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let kv = &body["kv_events"];
    let host = reqwest::Url::parse(url)?
        .host_str()
        .context("worker host missing")?
        .to_owned();
    let string = |field: &str| -> Result<String> {
        Ok(kv[field]
            .as_str()
            .with_context(|| format!("missing {field}"))?
            .to_owned())
    };
    let number = |field: &str| -> Result<u32> {
        Ok(u32::try_from(
            kv[field]
                .as_u64()
                .with_context(|| format!("missing {field}"))?,
        )?)
    };
    let endpoint = |prefix: &str, rank: u32| -> Result<String> {
        let advertised = string(&format!("{prefix}endpoint_host"))?;
        let resolved = if ["*", "0.0.0.0", "::", "[::]"].contains(&advertised.as_str()) {
            &host
        } else {
            &advertised
        };
        let resolved = if resolved.contains(':') && !resolved.starts_with('[') {
            format!("[{resolved}]")
        } else {
            resolved.to_owned()
        };
        let port = number(&format!("{prefix}endpoint_port_base"))?
            .checked_add(rank)
            .context("port overflow")?;
        ensure!((1..=65535).contains(&port), "invalid stream port");
        Ok(format!("tcp://{resolved}:{port}"))
    };
    let ranks = number("dp_size")?;
    ensure!((1..=4096).contains(&ranks), "invalid dp_size");
    ensure!(
        kv["snapshot_versions"]
            .as_array()
            .is_some_and(|v| v.contains(&serde_json::json!(2))),
        "Worker does not advertise snapshot v2"
    );
    let worker_id = string("worker_id")?;
    ensure!(
        !worker_id.is_empty(),
        "configure worker_id in Worker kv-events-config"
    );
    let mut streams = Vec::new();
    for rank in 0..ranks {
        streams.push(DiscoveredStream {
            descriptor: StreamDescriptor {
                key: Some(StreamKey {
                    namespace: string("namespace")?,
                    worker_id: worker_id.clone(),
                    dp_rank: rank,
                }),
                worker_address: url.into(),
                model: string("model")?,
                hash_schema_version: number("hash_schema_version")?,
                page_size: number("block_size")?,
                is_bigram: kv["is_bigram"].as_bool().context("missing is_bigram")?,
                cache_spec: None,
            },
            event_endpoint: endpoint("", rank)?,
            snapshot_endpoint: endpoint("snapshot_", rank)?,
            replay_endpoint: endpoint("replay_", rank)?,
            topic: string("topic")?,
        });
    }
    Ok(streams)
}

struct Task(tokio::task::JoinHandle<()>);
impl Drop for Task {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Reloads the complete desired Worker list from one JSON file. A bad update
/// retains the last valid configuration. Removing a stream stops its renewals;
/// Indexer lease expiration makes it non-queryable even if the Bridge is killed.
pub async fn run_file(path: &str) -> Result<()> {
    let initial: ReplicaBridgeConfig = serde_json::from_slice(&tokio::fs::read(path).await?)?;
    initial.validate()?;
    let channel = Endpoint::from_shared(initial.indexer_endpoint.clone())?
        .connect_timeout(IO_TIMEOUT)
        .timeout(IO_TIMEOUT)
        .connect_lazy();
    let client = reqwest::Client::builder().timeout(IO_TIMEOUT).build()?;
    let permits = Arc::new(Semaphore::new(initial.snapshot_concurrency));
    let mut tasks: HashMap<String, (DiscoveredStream, Task)> = HashMap::new();
    let mut known: HashMap<String, Vec<DiscoveredStream>> = HashMap::new();
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    loop {
        interval.tick().await;
        let config = async {
            let value: ReplicaBridgeConfig = serde_json::from_slice(&tokio::fs::read(path).await?)?;
            value.validate()?;
            ensure!(
                value.indexer_endpoint == initial.indexer_endpoint
                    && value.queue_capacity == initial.queue_capacity
                    && value.snapshot_concurrency == initial.snapshot_concurrency,
                "pair endpoint/buffer limits require Bridge restart"
            );
            Ok::<_, anyhow::Error>(value)
        }
        .await;
        let config = match config {
            Ok(c) => c,
            Err(e) => {
                warn!(%e, "Bridge config reload rejected");
                continue;
            }
        };
        let urls: HashSet<_> = config.worker_urls.iter().cloned().collect();
        known.retain(|url, _| urls.contains(url));
        let mut discovery = tokio::task::JoinSet::new();
        for url in urls {
            let client = client.clone();
            discovery.spawn(async move {
                let result = discover(&client, &url).await;
                (url, result)
            });
        }
        while let Some(result) = discovery.join_next().await {
            let (url, result) = result?;
            match result {
                Ok(streams) => {
                    known.insert(url, streams);
                }
                Err(e) => {
                    warn!(%url, %e, "Worker discovery failed; retaining prior stream configuration")
                }
            }
        }
        let mut desired = HashMap::new();
        let mut duplicate = false;
        for stream in known.values().flatten() {
            let key = stream.descriptor.key.as_ref().expect("discovered key");
            let id = serde_json::to_string(&(&key.namespace, &key.worker_id, key.dp_rank))?;
            if desired.insert(id, stream.clone()).is_some() {
                duplicate = true;
            }
        }
        if duplicate {
            warn!("duplicate Worker stream identity; refusing discovery update");
            continue;
        }
        tasks.retain(|id, (old, task)| desired.get(id) == Some(old) && !task.0.is_finished());
        for (id, stream) in desired {
            if tasks.contains_key(&id) {
                continue;
            }
            let descriptor = stream.clone();
            let channel = channel.clone();
            let permits = permits.clone();
            let capacity = config.queue_capacity;
            let task = Task(tokio::spawn(async move {
                supervise(stream, channel, permits, capacity).await;
            }));
            tasks.insert(id, (descriptor, task));
        }
    }
}

#[derive(Deserialize)]
struct Header {
    version: u32,
    namespace: String,
    model: String,
    worker_id: String,
    dp_rank: u32,
    worker_epoch: String,
    hash_schema_version: u32,
    page_size: u32,
    is_bigram: bool,
    barrier_seq: u64,
    resume_seq: u64,
    barrier_id: String,
    record_count: u64,
    cache_spec: Option<WorkerCacheSpec>,
}

#[derive(Deserialize)]
struct WireBlock {
    namespace: String,
    block_hash: i64,
    parent_block_hash: Option<i64>,
    block_size: u32,
    tier: i32,
    component_mask: u32,
}

struct Live {
    epoch: String,
    sequence: u64,
    barrier: Option<String>,
    payload: Vec<u8>,
    _bytes: OwnedSemaphorePermit,
}

fn decode_live(message: ZmqMessage, topic: &str, bytes: Arc<Semaphore>) -> Result<Live> {
    let frames = message.into_vec();
    ensure!(
        frames.len() == 3 && frames[1].len() == 8,
        "invalid live frames"
    );
    ensure!(frames[2].len() <= MAX_FRAME, "live frame too large");
    let prefix = format!("{topic}\0sgl-kv-epoch=");
    let text = std::str::from_utf8(&frames[0])?;
    let suffix = text
        .strip_prefix(&prefix)
        .context("missing Worker epoch in topic")?;
    let (epoch, barrier) = match suffix.split_once("\0sgl-kv-snapshot=") {
        Some((e, b)) => (e, Some(b.to_owned())),
        None => (suffix, None),
    };
    ensure!(!epoch.is_empty() && epoch.len() <= 256, "invalid epoch");
    let permit = bytes.try_acquire_many_owned(frames[2].len().max(1) as u32)?;
    Ok(Live {
        epoch: epoch.into(),
        sequence: u64::from_be_bytes(frames[1][..].try_into()?),
        barrier,
        payload: frames[2].to_vec(),
        _bytes: permit,
    })
}

async fn dealer(endpoint: &str, fields: Vec<Bytes>) -> Result<DealerSocket> {
    let mut socket = DealerSocket::new();
    socket.connect(endpoint).await?;
    let mut message = ZmqMessage::from(Bytes::new());
    for field in fields {
        message.push_back(field);
    }
    socket.send(message).await?;
    Ok(socket)
}

async fn reply(socket: &mut DealerSocket) -> Result<(Bytes, Bytes)> {
    let frames = tokio::time::timeout(IO_TIMEOUT, socket.recv())
        .await??
        .into_vec();
    ensure!(
        frames.len() == 3 && frames[0].is_empty(),
        "invalid recovery frames"
    );
    ensure!(frames[2].len() <= MAX_FRAME, "recovery frame too large");
    ensure!(
        frames[1].as_ref() != b"error",
        "Worker recovery error: {}",
        String::from_utf8_lossy(&frames[2])
    );
    Ok((frames[1].clone(), frames[2].clone()))
}

fn check_buffer(overflow: &AtomicBool) -> Result<()> {
    ensure!(
        !overflow.load(Ordering::Acquire),
        "live buffer overflow or subscriber failure; snapshot required"
    );
    Ok(())
}

async fn supervise(
    stream: DiscoveredStream,
    channel: Channel,
    permits: Arc<Semaphore>,
    capacity: usize,
) {
    let mut owner = StreamSession {
        key: stream.descriptor.key.clone(),
        session: String::new(),
    };
    let mut client = KvReplicaClient::new(channel)
        .max_encoding_message_size(MAX_FRAME)
        .max_decoding_message_size(MAX_FRAME);
    loop {
        // Independent jitter plus a per-pair semaphore bounds cold-start storms.
        tokio::time::sleep(Duration::from_millis(100 + rand::random::<u64>() % 500)).await;
        let result = run_stream(&stream, &mut client, &mut owner, permits.clone(), capacity).await;
        if let Err(error) = result {
            warn!(worker = %stream.descriptor.worker_address, %error, "recoverable stream restarting");
            if !owner.session.is_empty() {
                if let Err(e) = client.invalidate_stream(owner.clone()).await {
                    if e.code() == tonic::Code::FailedPrecondition {
                        owner.session.clear();
                    }
                }
            }
        }
    }
}

async fn run_stream(
    stream: &DiscoveredStream,
    client: &mut KvReplicaClient<Channel>,
    owner: &mut StreamSession,
    permits: Arc<Semaphore>,
    capacity: usize,
) -> Result<()> {
    let mut socket = SubSocket::new();
    socket.subscribe(&stream.topic).await?;
    socket.connect(&stream.event_endpoint).await?;
    let (tx, mut rx) = mpsc::channel(capacity);
    let overflow = Arc::new(AtomicBool::new(false));
    let failed = overflow.clone();
    let topic = stream.topic.clone();
    let _reader = Task(tokio::spawn(async move {
        let bytes = Arc::new(Semaphore::new(BUFFER_BYTES));
        loop {
            let event = match socket.recv().await {
                Ok(message) => decode_live(message, &topic, bytes.clone()),
                Err(error) => Err(error.into()),
            };
            if event.is_ok_and(|event| tx.try_send(event).is_ok()) {
                continue;
            }
            failed.store(true, Ordering::Release);
            break;
        }
    }));
    let _permit = permits.acquire_owned().await?;
    let mut snapshot = tokio::time::timeout(
        IO_TIMEOUT,
        dealer(
            &stream.snapshot_endpoint,
            vec![Bytes::from_static(b"snapshot-v2")],
        ),
    )
    .await??;
    let (kind, payload) = reply(&mut snapshot).await?;
    ensure!(kind.as_ref() == b"header", "snapshot header missing");
    let header: Header = rmp_serde::from_slice(&payload)?;
    let expected = stream.descriptor.key.as_ref().context("missing key")?;
    ensure!(
        header.namespace == expected.namespace
            && header.worker_id == expected.worker_id
            && header.dp_rank == expected.dp_rank
            && header.model == stream.descriptor.model
            && header.page_size == stream.descriptor.page_size
            && header.is_bigram == stream.descriptor.is_bigram
            && header.hash_schema_version == stream.descriptor.hash_schema_version,
        "snapshot identity/hash schema disagrees with Worker discovery"
    );
    let mut descriptor = stream.descriptor.clone();
    descriptor.cache_spec = header.cache_spec;
    let cut = SnapshotCut {
        version: header.version,
        stream: Some(descriptor),
        worker_epoch: header.worker_epoch.clone(),
        barrier_seq: header.barrier_seq,
        resume_seq: header.resume_seq,
        barrier_id: header.barrier_id.clone(),
        record_count: header.record_count,
    };
    let started = client
        .begin_snapshot(BeginSnapshotRequest {
            cut: Some(cut),
            session: owner.session.clone(),
        })
        .await?
        .into_inner();
    owner.session = started.session;
    let mut received = 0;
    loop {
        check_buffer(&overflow)?;
        let (kind, payload) = reply(&mut snapshot).await?;
        if kind.as_ref() == b"end" {
            ensure!(
                payload.is_empty() && received == header.record_count,
                "snapshot end/count mismatch"
            );
            break;
        }
        ensure!(kind.as_ref() == b"chunk", "unexpected snapshot reply");
        let records: Vec<WireBlock> = rmp_serde::from_slice(&payload)?;
        ensure!(records.len() <= 4096, "snapshot chunk record limit");
        let mut blocks = Vec::new();
        for record in records {
            ensure!(
                record.namespace == expected.namespace,
                "snapshot block namespace mismatch"
            );
            blocks.push(PlacementBlock {
                block_hash: record.block_hash,
                parent_block_hash: record.parent_block_hash,
                block_size: record.block_size,
                tier: record.tier,
                component_mask: record.component_mask,
            });
        }
        let count = blocks.len() as u64;
        let progress = client
            .snapshot_chunk(SnapshotChunkRequest {
                owner: Some(owner.clone()),
                offset: received,
                blocks,
            })
            .await?
            .into_inner();
        ensure!(!progress.needs_snapshot, "snapshot session expired");
        received += count;
    }
    // A matching live barrier proves SUB establishment before the snapshot cut.
    // A replayed empty batch cannot substitute for this proof.
    tokio::time::timeout(IO_TIMEOUT, async {
        loop {
            check_buffer(&overflow)?;
            let event = rx.recv().await.context("subscriber stopped")?;
            ensure!(
                event.epoch == header.worker_epoch,
                "epoch changed during snapshot"
            );
            if event.sequence == header.barrier_seq
                && event.barrier.as_deref() == Some(&header.barrier_id)
            {
                break;
            }
            ensure!(event.sequence <= header.barrier_seq, "live barrier missed");
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    let mut next = header.resume_seq;
    replay(
        stream,
        client,
        owner,
        &header.worker_epoch,
        &mut next,
        &header.barrier_id,
        header.barrier_seq,
        &overflow,
    )
    .await?;
    drop(_permit);
    info!(worker = %stream.descriptor.worker_address, rank = expected.dp_rank, epoch = %header.worker_epoch, next, "stream READY after snapshot and replay");
    let mut probe = tokio::time::interval(Duration::from_millis(500));
    loop {
        check_buffer(&overflow)?;
        tokio::select! {
            event = rx.recv() => {
                let event = event.context("subscriber stopped")?;
                ensure!(event.epoch == header.worker_epoch, "Worker epoch changed");
                if event.sequence < next { continue; }
                if event.sequence > next {
                    // Let Indexer revoke READY and return its authoritative gap.
                    let p = client.apply_live(LiveBatchRequest { owner: Some(owner.clone()), worker_epoch: event.epoch.clone(), sequence: event.sequence, actions: vec![] }).await?.into_inner();
                    ensure!(!p.needs_snapshot, "stream needs full snapshot");
                    next = p.next_sequence;
                    replay(stream, client, owner, &header.worker_epoch, &mut next, "", 0, &overflow).await?;
                }
                if event.sequence == next {
                    apply(stream, client, owner, &event.epoch, event.sequence, &event.payload, &mut next).await?;
                }
            }
            _ = probe.tick() => {
                replay(stream, client, owner, &header.worker_epoch, &mut next, "", 0, &overflow).await?;
            }
        }
    }
}

async fn apply(
    stream: &DiscoveredStream,
    client: &mut KvReplicaClient<Channel>,
    owner: &StreamSession,
    epoch: &str,
    sequence: u64,
    payload: &[u8],
    next: &mut u64,
) -> Result<()> {
    let actions =
        decode_recoverable_batch(payload, stream.descriptor.key.as_ref().unwrap().dp_rank)?;
    let result = client
        .apply_live(LiveBatchRequest {
            owner: Some(owner.clone()),
            worker_epoch: epoch.into(),
            sequence,
            actions,
        })
        .await?
        .into_inner();
    ensure!(
        !result.needs_snapshot && result.next_sequence == sequence + 1,
        "Indexer rejected sequence; full recovery required"
    );
    *next = result.next_sequence;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn replay(
    stream: &DiscoveredStream,
    client: &mut KvReplicaClient<Channel>,
    owner: &StreamSession,
    epoch: &str,
    next: &mut u64,
    barrier: &str,
    barrier_seq: u64,
    overflow: &AtomicBool,
) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut socket = dealer(
            &stream.replay_endpoint,
            vec![
                Bytes::from_static(b"replay-v2"),
                Bytes::copy_from_slice(epoch.as_bytes()),
                Bytes::copy_from_slice(&next.to_be_bytes()),
            ],
        )
        .await?;
        let (kind, payload) = reply(&mut socket).await?;
        ensure!(kind.as_ref() == b"header", "missing replay header");
        #[derive(Deserialize)]
        struct ReplayHeader {
            worker_epoch: String,
            resume_seq: u64,
        }
        let header: ReplayHeader = rmp_serde::from_slice(&payload)?;
        ensure!(
            header.worker_epoch == epoch && header.resume_seq >= *next,
            "replay generation/watermark mismatch"
        );
        loop {
            check_buffer(overflow)?;
            let (kind, payload) = reply(&mut socket).await?;
            if kind.as_ref() == b"end" {
                ensure!(
                    payload.is_empty() && *next == header.resume_seq,
                    "replay incomplete; snapshot required"
                );
                break;
            }
            ensure!(kind.len() == 8, "invalid replay sequence");
            let sequence = u64::from_be_bytes(kind[..].try_into()?);
            ensure!(
                sequence == *next && sequence < header.resume_seq,
                "replay window gap; snapshot required"
            );
            apply(stream, client, owner, epoch, sequence, &payload, next).await?;
        }
        check_buffer(overflow)?;
        let result = client
            .confirm_stream(ConfirmStreamRequest {
                owner: Some(owner.clone()),
                worker_epoch: epoch.into(),
                resume_seq: *next,
                barrier_id: barrier.into(),
                barrier_seq,
            })
            .await?
            .into_inner();
        ensure!(
            result.ready && !result.needs_snapshot,
            "Indexer has not confirmed coverage"
        );
        Ok::<_, anyhow::Error>(())
    })
    .await?
}
