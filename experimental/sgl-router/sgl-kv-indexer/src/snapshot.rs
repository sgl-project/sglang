// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chunked client for SGLang's `snapshot-v1` ZMQ protocol.

use std::time::Duration;

use bytes::Bytes;
use serde::Deserialize;
use thiserror::Error;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

const REQUEST_V1: &[u8] = b"snapshot-v1";
const REQUEST_V2: &[u8] = b"snapshot-v2";
const HEADER: &[u8] = b"header";
const CHUNK: &[u8] = b"chunk";
const END: &[u8] = b"end";
const ERROR: &[u8] = b"error";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;
const MAX_RECORDS: usize = 10_000_000;
const MAX_HASHES_PER_RECORD: usize = 65_536;
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotHeader {
    pub version: u32,
    pub epoch: String,
    pub replica_rank: u32,
    pub resume_seq: i64,
    pub barrier_seq: i64,
    pub barrier_id: String,
    pub record_count: usize,
    pub metadata: Option<SnapshotMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotMetadata {
    pub namespace: String,
    pub model: String,
    pub worker_id: String,
    pub worker_generation: String,
    pub hash_schema_version: u32,
    pub page_size: u32,
    pub is_bigram: bool,
    pub cache_spec: SnapshotCacheSpec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct SnapshotCacheSpec {
    pub version: u32,
    pub components: u32,
    pub swa_window_tokens: u32,
    pub full_tier_mask: u32,
    pub swa_tier_mask: u32,
    pub mamba_tier_mask: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SnapshotBlock {
    pub parent_block_hash: Option<i64>,
    pub block_hashes: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SnapshotPlacement {
    pub parent_block_hash: Option<i64>,
    pub block_hash: i64,
    pub tier: i32,
    pub component_mask: u32,
    pub block_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementSnapshot {
    pub header: SnapshotHeader,
    pub blocks: Vec<SnapshotBlock>,
    pub placements: Vec<SnapshotPlacement>,
}

#[derive(Debug, Deserialize)]
struct SnapshotHeaderV1 {
    version: u32,
    epoch: String,
    replica_rank: u32,
    resume_seq: i64,
    barrier_seq: i64,
    barrier_id: String,
    record_count: usize,
}

#[derive(Debug, Deserialize)]
struct SnapshotHeaderV2 {
    version: u32,
    namespace: String,
    model: String,
    worker_id: String,
    replica_rank: u32,
    worker_generation: String,
    epoch: String,
    hash_schema_version: u32,
    page_size: u32,
    is_bigram: bool,
    resume_seq: i64,
    barrier_seq: i64,
    barrier_id: String,
    record_count: usize,
    cache_spec: SnapshotCacheSpec,
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("snapshot request to {endpoint} timed out")]
    Timeout { endpoint: String },
    #[error("snapshot ZMQ error at {endpoint}: {source}")]
    Zmq {
        endpoint: String,
        source: zeromq::ZmqError,
    },
    #[error("invalid snapshot frame: {0}")]
    Invalid(String),
    #[error("snapshot decode failed: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("snapshot provider returned error: {0}")]
    Provider(String),
}

pub async fn fetch_snapshot(
    endpoint: &str,
    expected_rank: u32,
) -> Result<PlacementSnapshot, SnapshotError> {
    tokio::time::timeout(
        DEFAULT_TIMEOUT,
        fetch_snapshot_inner(endpoint, expected_rank, VERSION_V1),
    )
    .await
    .map_err(|_| SnapshotError::Timeout {
        endpoint: endpoint.to_owned(),
    })?
}

pub async fn fetch_snapshot_v2(
    endpoint: &str,
    expected_rank: u32,
) -> Result<PlacementSnapshot, SnapshotError> {
    tokio::time::timeout(
        DEFAULT_TIMEOUT,
        fetch_snapshot_inner(endpoint, expected_rank, VERSION_V2),
    )
    .await
    .map_err(|_| SnapshotError::Timeout {
        endpoint: endpoint.to_owned(),
    })?
}

async fn fetch_snapshot_inner(
    endpoint: &str,
    expected_rank: u32,
    requested_version: u32,
) -> Result<PlacementSnapshot, SnapshotError> {
    let mut socket = DealerSocket::new();
    socket
        .connect(endpoint)
        .await
        .map_err(|source| SnapshotError::Zmq {
            endpoint: endpoint.to_owned(),
            source,
        })?;
    let mut request = ZmqMessage::from(Bytes::new());
    request.push_back(Bytes::from_static(if requested_version == VERSION_V2 {
        REQUEST_V2
    } else {
        REQUEST_V1
    }));
    socket
        .send(request)
        .await
        .map_err(|source| SnapshotError::Zmq {
            endpoint: endpoint.to_owned(),
            source,
        })?;

    let first = recv(&mut socket, endpoint).await?;
    if first.kind == ERROR {
        return Err(SnapshotError::Provider(
            String::from_utf8_lossy(&first.payload).into_owned(),
        ));
    }
    if first.kind != HEADER {
        return Err(SnapshotError::Invalid("header expected".into()));
    }
    let header = if requested_version == VERSION_V2 {
        let raw: SnapshotHeaderV2 = rmp_serde::from_slice(&first.payload)?;
        SnapshotHeader {
            version: raw.version,
            epoch: raw.epoch,
            replica_rank: raw.replica_rank,
            resume_seq: raw.resume_seq,
            barrier_seq: raw.barrier_seq,
            barrier_id: raw.barrier_id,
            record_count: raw.record_count,
            metadata: Some(SnapshotMetadata {
                namespace: raw.namespace,
                model: raw.model,
                worker_id: raw.worker_id,
                worker_generation: raw.worker_generation,
                hash_schema_version: raw.hash_schema_version,
                page_size: raw.page_size,
                is_bigram: raw.is_bigram,
                cache_spec: raw.cache_spec,
            }),
        }
    } else {
        let raw: SnapshotHeaderV1 = rmp_serde::from_slice(&first.payload)?;
        SnapshotHeader {
            version: raw.version,
            epoch: raw.epoch,
            replica_rank: raw.replica_rank,
            resume_seq: raw.resume_seq,
            barrier_seq: raw.barrier_seq,
            barrier_id: raw.barrier_id,
            record_count: raw.record_count,
            metadata: None,
        }
    };
    validate_header(&header, expected_rank)?;

    let mut blocks = Vec::with_capacity(header.record_count.min(MAX_RECORDS));
    let mut placements = Vec::with_capacity(header.record_count.min(MAX_RECORDS));
    loop {
        let reply = recv(&mut socket, endpoint).await?;
        match reply.kind.as_slice() {
            CHUNK => {
                if requested_version == VERSION_V2 {
                    let chunk: Vec<SnapshotPlacement> = rmp_serde::from_slice(&reply.payload)?;
                    if placements.len().saturating_add(chunk.len()) > MAX_RECORDS {
                        return Err(SnapshotError::Invalid("too many snapshot records".into()));
                    }
                    if chunk.iter().any(|record| {
                        !(1..=3).contains(&record.tier)
                            || record.component_mask == 0
                            || record.block_size == 0
                    }) {
                        return Err(SnapshotError::Invalid(
                            "invalid snapshot v2 placement".into(),
                        ));
                    }
                    placements.extend(chunk);
                    continue;
                }
                let chunk: Vec<SnapshotBlock> = rmp_serde::from_slice(&reply.payload)?;
                if blocks.len().saturating_add(chunk.len()) > MAX_RECORDS {
                    return Err(SnapshotError::Invalid("too many snapshot records".into()));
                }
                if chunk.iter().any(|record| {
                    record.block_hashes.is_empty()
                        || record.block_hashes.len() > MAX_HASHES_PER_RECORD
                }) {
                    return Err(SnapshotError::Invalid(
                        "invalid snapshot block hash count".into(),
                    ));
                }
                blocks.extend(chunk);
            }
            END => {
                if !reply.payload.is_empty() {
                    return Err(SnapshotError::Invalid(
                        "snapshot end payload must be empty".into(),
                    ));
                }
                break;
            }
            ERROR => {
                return Err(SnapshotError::Provider(
                    String::from_utf8_lossy(&reply.payload).into_owned(),
                ));
            }
            _ => return Err(SnapshotError::Invalid("unexpected snapshot frame".into())),
        }
    }
    let received = if requested_version == VERSION_V2 {
        placements.len()
    } else {
        blocks.len()
    };
    if received != header.record_count {
        return Err(SnapshotError::Invalid(format!(
            "snapshot declared {} records but sent {}",
            header.record_count, received
        )));
    }
    Ok(PlacementSnapshot {
        header,
        blocks,
        placements,
    })
}

fn validate_header(header: &SnapshotHeader, expected_rank: u32) -> Result<(), SnapshotError> {
    if !matches!(header.version, VERSION_V1 | VERSION_V2) {
        return Err(SnapshotError::Invalid(format!(
            "unsupported snapshot version {}",
            header.version
        )));
    }
    if header.replica_rank != expected_rank {
        return Err(SnapshotError::Invalid(format!(
            "snapshot rank {} does not match {expected_rank}",
            header.replica_rank
        )));
    }
    if header.epoch.is_empty() || header.barrier_id.is_empty() || header.barrier_seq < 0 {
        return Err(SnapshotError::Invalid("invalid snapshot identity".into()));
    }
    if header.resume_seq != header.barrier_seq.saturating_add(1) {
        return Err(SnapshotError::Invalid(
            "snapshot resume sequence does not follow barrier".into(),
        ));
    }
    if header.record_count > MAX_RECORDS {
        return Err(SnapshotError::Invalid(
            "snapshot record cap exceeded".into(),
        ));
    }
    if header.version == VERSION_V2 {
        let metadata = header
            .metadata
            .as_ref()
            .ok_or_else(|| SnapshotError::Invalid("snapshot v2 metadata missing".into()))?;
        if metadata.namespace.is_empty()
            || metadata.model.is_empty()
            || metadata.worker_id.is_empty()
            || metadata.worker_generation.is_empty()
            || metadata.hash_schema_version == 0
            || metadata.page_size == 0
            || metadata.cache_spec.components == 0
        {
            return Err(SnapshotError::Invalid(
                "snapshot v2 metadata is incomplete".into(),
            ));
        }
    }
    Ok(())
}

struct Reply {
    kind: Vec<u8>,
    payload: Bytes,
}

async fn recv(socket: &mut DealerSocket, endpoint: &str) -> Result<Reply, SnapshotError> {
    let reply = socket.recv().await.map_err(|source| SnapshotError::Zmq {
        endpoint: endpoint.to_owned(),
        source,
    })?;
    if reply.len() != 3 {
        return Err(SnapshotError::Invalid(format!(
            "expected 3 reply frames, got {}",
            reply.len()
        )));
    }
    let mut frames = reply.into_vecdeque();
    if !frames.pop_front().expect("three frames").is_empty() {
        return Err(SnapshotError::Invalid("delimiter must be empty".into()));
    }
    Ok(Reply {
        kind: frames.pop_front().expect("three frames").to_vec(),
        payload: frames.pop_front().expect("three frames"),
    })
}
