// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chunked client for SGLang's `snapshot-v1` ZMQ protocol.

use std::time::Duration;

use bytes::Bytes;
use serde::Deserialize;
use thiserror::Error;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

const REQUEST: &[u8] = b"snapshot-v1";
const HEADER: &[u8] = b"header";
const CHUNK: &[u8] = b"chunk";
const END: &[u8] = b"end";
const ERROR: &[u8] = b"error";
const VERSION: u32 = 1;
const MAX_RECORDS: usize = 10_000_000;
const MAX_HASHES_PER_RECORD: usize = 65_536;
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SnapshotHeader {
    pub version: u32,
    pub epoch: String,
    pub replica_rank: u32,
    pub resume_seq: i64,
    pub barrier_seq: i64,
    pub barrier_id: String,
    pub record_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SnapshotBlock {
    pub parent_block_hash: Option<i64>,
    pub block_hashes: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementSnapshot {
    pub header: SnapshotHeader,
    pub blocks: Vec<SnapshotBlock>,
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
        fetch_snapshot_inner(endpoint, expected_rank),
    )
    .await
    .map_err(|_| SnapshotError::Timeout {
        endpoint: endpoint.to_owned(),
    })?
}

async fn fetch_snapshot_inner(
    endpoint: &str,
    expected_rank: u32,
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
    request.push_back(Bytes::from_static(REQUEST));
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
    let header: SnapshotHeader = rmp_serde::from_slice(&first.payload)?;
    validate_header(&header, expected_rank)?;

    let mut blocks = Vec::with_capacity(header.record_count.min(MAX_RECORDS));
    loop {
        let reply = recv(&mut socket, endpoint).await?;
        match reply.kind.as_slice() {
            CHUNK => {
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
    if blocks.len() != header.record_count {
        return Err(SnapshotError::Invalid(format!(
            "snapshot declared {} records but sent {}",
            header.record_count,
            blocks.len()
        )));
    }
    Ok(PlacementSnapshot { header, blocks })
}

fn validate_header(header: &SnapshotHeader, expected_rank: u32) -> Result<(), SnapshotError> {
    if header.version != VERSION {
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
