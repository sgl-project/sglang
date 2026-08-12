// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chunked placement-snapshot client for SGLang's ZMQ snapshot provider.
//!
//! The DEALER sends `[empty_delimiter, "snapshot-v1"]`. The worker's ROUTER
//! replies with a header, zero or more record chunks, and an end marker:
//!
//! ```text
//! [empty, "header", msgpack(header)]
//! [empty, "chunk",  msgpack(records)] ...
//! [empty, "end",    empty]
//! ```

use std::fmt;
use std::time::Duration;

use bytes::Bytes;
use serde::de::{self, SeqAccess, Visitor};
use serde::Deserialize;
use thiserror::Error;
use tokio::time::timeout;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

const SNAPSHOT_REQUEST: &[u8] = b"snapshot-v1";
const SNAPSHOT_HEADER: &[u8] = b"header";
const SNAPSHOT_CHUNK: &[u8] = b"chunk";
const SNAPSHOT_END: &[u8] = b"end";
const SNAPSHOT_ERROR: &[u8] = b"error";
const SNAPSHOT_PROTOCOL_VERSION: u32 = 1;

// Full placement views can contain millions of block records. Keep the
// end-to-end transfer bounded without treating an ordinary large snapshot as
// a failed provider.
pub(crate) const DEFAULT_SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_SNAPSHOT_RECORDS: usize = 10_000_000;
const MAX_RECORDS_PER_CHUNK: usize = 8_192;
const MAX_HASHES_PER_RECORD: usize = 65_536;
const MAX_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SnapshotHeader {
    pub version: u32,
    /// Lifecycle identity of one DP replica publisher, not the enclosing
    /// server instance. A local rank restart must produce a new value.
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
    #[error("snapshot request to {endpoint} timed out after {timeout:?}")]
    Timeout { endpoint: String, timeout: Duration },
    #[error("snapshot ZMQ error at {endpoint}: {source}")]
    Zmq {
        endpoint: String,
        source: zeromq::ZmqError,
    },
    #[error("snapshot reply has {got} frames, expected 3")]
    InvalidFrameCount { got: usize },
    #[error("snapshot reply delimiter frame is not empty")]
    InvalidDelimiter,
    #[error("snapshot reply kind {0:?} is not valid UTF-8")]
    InvalidKind(Vec<u8>),
    #[error("snapshot reply started with {0:?}, expected header")]
    HeaderExpected(String),
    #[error("snapshot header decode failed: {0}")]
    HeaderDecode(rmp_serde::decode::Error),
    #[error("snapshot chunk decode failed: {0}")]
    ChunkDecode(rmp_serde::decode::Error),
    #[error("snapshot provider returned error: {0}")]
    Provider(String),
    #[error("unsupported snapshot protocol version {got}; expected {expected}")]
    UnsupportedVersion { got: u32, expected: u32 },
    #[error("snapshot replica rank {got} does not match requested rank {expected}")]
    RankMismatch { got: u32, expected: u32 },
    #[error("snapshot resume_seq {resume_seq} must equal barrier_seq + 1 ({expected})")]
    InvalidResumeSeq { resume_seq: i64, expected: i64 },
    #[error("snapshot barrier_seq {0} must be non-negative")]
    InvalidBarrierSeq(i64),
    #[error("snapshot {0} must not be empty")]
    EmptyField(&'static str),
    #[error("snapshot block record must contain at least one block hash")]
    EmptyBlockHashes,
    #[error("snapshot {field} length {len} exceeds cap {cap}")]
    FieldTooLarge {
        field: &'static str,
        len: usize,
        cap: usize,
    },
    #[error("snapshot ended with {got} records, header declared {expected}")]
    RecordCountMismatch { got: usize, expected: usize },
    #[error("snapshot end marker carried {len} payload bytes, expected 0")]
    InvalidEndPayload { len: usize },
}

pub async fn fetch_snapshot(
    endpoint: &str,
    expected_rank: u32,
) -> Result<PlacementSnapshot, SnapshotError> {
    fetch_snapshot_with_timeout(endpoint, expected_rank, DEFAULT_SNAPSHOT_TIMEOUT).await
}

pub async fn fetch_snapshot_with_timeout(
    endpoint: &str,
    expected_rank: u32,
    timeout_duration: Duration,
) -> Result<PlacementSnapshot, SnapshotError> {
    let endpoint_owned = endpoint.to_owned();
    timeout(
        timeout_duration,
        fetch_snapshot_inner(endpoint, expected_rank),
    )
    .await
    .map_err(|_| SnapshotError::Timeout {
        endpoint: endpoint_owned,
        timeout: timeout_duration,
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
    request.push_back(Bytes::from_static(SNAPSHOT_REQUEST));
    socket
        .send(request)
        .await
        .map_err(|source| SnapshotError::Zmq {
            endpoint: endpoint.to_owned(),
            source,
        })?;

    let first = recv_reply(&mut socket, endpoint).await?;
    if first.kind == SNAPSHOT_ERROR {
        return Err(SnapshotError::Provider(provider_message(&first.payload)));
    }
    if first.kind != SNAPSHOT_HEADER {
        return Err(SnapshotError::HeaderExpected(kind_string(&first.kind)?));
    }
    let header: SnapshotHeader =
        rmp_serde::from_slice(&first.payload).map_err(SnapshotError::HeaderDecode)?;
    validate_header(&header, expected_rank)?;

    let mut blocks = Vec::with_capacity(header.record_count.min(MAX_SNAPSHOT_RECORDS));
    loop {
        let reply = recv_reply(&mut socket, endpoint).await?;
        match reply.kind.as_slice() {
            SNAPSHOT_CHUNK => {
                let chunk: BoundedSnapshotChunk =
                    rmp_serde::from_slice(&reply.payload).map_err(SnapshotError::ChunkDecode)?;
                if blocks.len().saturating_add(chunk.0.len()) > MAX_SNAPSHOT_RECORDS {
                    return Err(SnapshotError::FieldTooLarge {
                        field: "records",
                        len: blocks.len().saturating_add(chunk.0.len()),
                        cap: MAX_SNAPSHOT_RECORDS,
                    });
                }
                for block in &chunk.0 {
                    if block.block_hashes.is_empty() {
                        return Err(SnapshotError::EmptyBlockHashes);
                    }
                    if block.block_hashes.len() > MAX_HASHES_PER_RECORD {
                        return Err(SnapshotError::FieldTooLarge {
                            field: "block_hashes",
                            len: block.block_hashes.len(),
                            cap: MAX_HASHES_PER_RECORD,
                        });
                    }
                }
                blocks.extend(chunk.0);
            }
            SNAPSHOT_END => {
                if !reply.payload.is_empty() {
                    return Err(SnapshotError::InvalidEndPayload {
                        len: reply.payload.len(),
                    });
                }
                break;
            }
            SNAPSHOT_ERROR => {
                return Err(SnapshotError::Provider(provider_message(&reply.payload)));
            }
            _ => return Err(SnapshotError::HeaderExpected(kind_string(&reply.kind)?)),
        }
    }

    if blocks.len() != header.record_count {
        return Err(SnapshotError::RecordCountMismatch {
            got: blocks.len(),
            expected: header.record_count,
        });
    }
    Ok(PlacementSnapshot { header, blocks })
}

fn validate_header(header: &SnapshotHeader, expected_rank: u32) -> Result<(), SnapshotError> {
    if header.version != SNAPSHOT_PROTOCOL_VERSION {
        return Err(SnapshotError::UnsupportedVersion {
            got: header.version,
            expected: SNAPSHOT_PROTOCOL_VERSION,
        });
    }
    if header.replica_rank != expected_rank {
        return Err(SnapshotError::RankMismatch {
            got: header.replica_rank,
            expected: expected_rank,
        });
    }
    if header.barrier_seq < 0 {
        return Err(SnapshotError::InvalidBarrierSeq(header.barrier_seq));
    }
    let expected = header.barrier_seq.saturating_add(1);
    if header.resume_seq != expected {
        return Err(SnapshotError::InvalidResumeSeq {
            resume_seq: header.resume_seq,
            expected,
        });
    }
    for (field, value) in [
        ("epoch", header.epoch.as_str()),
        ("barrier_id", header.barrier_id.as_str()),
    ] {
        if value.is_empty() {
            return Err(SnapshotError::EmptyField(field));
        }
        if value.len() > MAX_ID_BYTES {
            return Err(SnapshotError::FieldTooLarge {
                field,
                len: value.len(),
                cap: MAX_ID_BYTES,
            });
        }
    }
    if header.record_count > MAX_SNAPSHOT_RECORDS {
        return Err(SnapshotError::FieldTooLarge {
            field: "records",
            len: header.record_count,
            cap: MAX_SNAPSHOT_RECORDS,
        });
    }
    Ok(())
}

struct SnapshotReply {
    kind: Vec<u8>,
    payload: Bytes,
}

async fn recv_reply(
    socket: &mut DealerSocket,
    endpoint: &str,
) -> Result<SnapshotReply, SnapshotError> {
    let reply = socket.recv().await.map_err(|source| SnapshotError::Zmq {
        endpoint: endpoint.to_owned(),
        source,
    })?;
    if reply.len() != 3 {
        return Err(SnapshotError::InvalidFrameCount { got: reply.len() });
    }
    let mut frames = reply.into_vecdeque();
    let delimiter = frames.pop_front().expect("frame count checked");
    if !delimiter.is_empty() {
        return Err(SnapshotError::InvalidDelimiter);
    }
    let kind = frames.pop_front().expect("frame count checked").to_vec();
    let payload = frames.pop_front().expect("frame count checked");
    Ok(SnapshotReply { kind, payload })
}

fn kind_string(kind: &[u8]) -> Result<String, SnapshotError> {
    String::from_utf8(kind.to_vec()).map_err(|_| SnapshotError::InvalidKind(kind.to_vec()))
}

fn provider_message(payload: &[u8]) -> String {
    String::from_utf8_lossy(&payload[..payload.len().min(1024)]).into_owned()
}

struct BoundedSnapshotChunk(Vec<SnapshotBlock>);

impl<'de> Deserialize<'de> for BoundedSnapshotChunk {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ChunkVisitor;
        impl<'de> Visitor<'de> for ChunkVisitor {
            type Value = Vec<SnapshotBlock>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a bounded array of snapshot block records")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                if let Some(len) = seq.size_hint() {
                    if len > MAX_RECORDS_PER_CHUNK {
                        return Err(de::Error::custom(format!(
                            "snapshot chunk has {len} records, cap is {MAX_RECORDS_PER_CHUNK}"
                        )));
                    }
                }
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(record) = seq.next_element::<SnapshotBlock>()? {
                    if out.len() >= MAX_RECORDS_PER_CHUNK {
                        return Err(de::Error::custom(format!(
                            "snapshot chunk exceeds record cap {MAX_RECORDS_PER_CHUNK}"
                        )));
                    }
                    out.push(record);
                }
                Ok(out)
            }
        }
        deserializer
            .deserialize_seq(ChunkVisitor)
            .map(BoundedSnapshotChunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio::task::JoinHandle;
    use zeromq::{Endpoint, RouterSocket};

    async fn spawn_provider(replies: Vec<(Vec<u8>, Vec<u8>)>) -> (String, JoinHandle<()>) {
        let mut router = RouterSocket::new();
        let endpoint = router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("bind snapshot ROUTER");
        let port = match endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };
        let task = tokio::spawn(async move {
            let request = router.recv().await.expect("snapshot request");
            assert_eq!(request.len(), 3);
            let identity = request.get(0).expect("identity").clone();
            assert!(request.get(1).expect("delimiter").is_empty());
            assert_eq!(request.get(2).expect("command").as_ref(), SNAPSHOT_REQUEST);

            for (kind, payload) in replies {
                send_reply(&mut router, identity.clone(), &kind, payload).await;
            }
        });
        (format!("tcp://127.0.0.1:{port}"), task)
    }

    fn header_payload(
        replica_rank: u32,
        resume_seq: i64,
        barrier_seq: i64,
        record_count: usize,
    ) -> Vec<u8> {
        rmp_serde::to_vec(&(
            1_u32,
            "epoch-a",
            replica_rank,
            resume_seq,
            barrier_seq,
            "barrier-a",
            record_count,
        ))
        .unwrap()
    }

    async fn send_reply(router: &mut RouterSocket, identity: Bytes, kind: &[u8], payload: Vec<u8>) {
        let mut reply = ZmqMessage::from(identity);
        reply.push_back(Bytes::new());
        reply.push_back(Bytes::copy_from_slice(kind));
        reply.push_back(Bytes::from(payload));
        router.send(reply).await.expect("send snapshot reply");
    }

    #[tokio::test]
    async fn reads_chunked_snapshot() {
        let (endpoint, provider) = spawn_provider(vec![
            (SNAPSHOT_HEADER.to_vec(), header_payload(2, 8, 7, 3)),
            (
                SNAPSHOT_CHUNK.to_vec(),
                rmp_serde::to_vec(&vec![(None::<i64>, vec![11_i64])]).unwrap(),
            ),
            (
                SNAPSHOT_CHUNK.to_vec(),
                rmp_serde::to_vec(&vec![
                    (Some(11_i64), vec![12_i64]),
                    (Some(12_i64), vec![13_i64]),
                ])
                .unwrap(),
            ),
            (SNAPSHOT_END.to_vec(), Vec::new()),
        ])
        .await;
        let snapshot = fetch_snapshot_with_timeout(&endpoint, 2, Duration::from_secs(1))
            .await
            .expect("snapshot succeeds");
        provider.await.expect("provider task");
        assert_eq!(snapshot.header.epoch, "epoch-a");
        assert_eq!(snapshot.header.resume_seq, 8);
        assert_eq!(snapshot.blocks.len(), 3);
        assert_eq!(snapshot.blocks[1].parent_block_hash, Some(11));
        assert_eq!(snapshot.blocks[1].block_hashes, vec![12]);
        assert_eq!(snapshot.blocks[2].block_hashes, vec![13]);
    }

    #[tokio::test]
    async fn rejects_provider_error() {
        let (endpoint, provider) = spawn_provider(vec![(
            SNAPSHOT_ERROR.to_vec(),
            b"snapshot provider busy".to_vec(),
        )])
        .await;
        let error = fetch_snapshot_with_timeout(&endpoint, 0, Duration::from_secs(1))
            .await
            .expect_err("provider error must fail the snapshot");
        provider.await.expect("provider task");
        assert!(matches!(
            error,
            SnapshotError::Provider(message) if message == "snapshot provider busy"
        ));
    }

    #[tokio::test]
    async fn rejects_header_for_another_replica() {
        let (endpoint, provider) =
            spawn_provider(vec![(SNAPSHOT_HEADER.to_vec(), header_payload(1, 8, 7, 0))]).await;
        let error = fetch_snapshot_with_timeout(&endpoint, 0, Duration::from_secs(1))
            .await
            .expect_err("rank mismatch must fail the snapshot");
        provider.await.expect("provider task");
        assert!(matches!(
            error,
            SnapshotError::RankMismatch {
                got: 1,
                expected: 0
            }
        ));
    }

    #[tokio::test]
    async fn rejects_non_contiguous_resume_sequence() {
        let (endpoint, provider) =
            spawn_provider(vec![(SNAPSHOT_HEADER.to_vec(), header_payload(0, 9, 7, 0))]).await;
        let error = fetch_snapshot_with_timeout(&endpoint, 0, Duration::from_secs(1))
            .await
            .expect_err("resume_seq must be barrier_seq + 1");
        provider.await.expect("provider task");
        assert!(matches!(
            error,
            SnapshotError::InvalidResumeSeq {
                resume_seq: 9,
                expected: 8
            }
        ));
    }

    #[tokio::test]
    async fn rejects_record_count_mismatch() {
        let (endpoint, provider) = spawn_provider(vec![
            (SNAPSHOT_HEADER.to_vec(), header_payload(0, 8, 7, 2)),
            (
                SNAPSHOT_CHUNK.to_vec(),
                rmp_serde::to_vec(&vec![(None::<i64>, vec![11_i64])]).unwrap(),
            ),
            (SNAPSHOT_END.to_vec(), Vec::new()),
        ])
        .await;
        let error = fetch_snapshot_with_timeout(&endpoint, 0, Duration::from_secs(1))
            .await
            .expect_err("declared record count must match streamed records");
        provider.await.expect("provider task");
        assert!(matches!(
            error,
            SnapshotError::RecordCountMismatch {
                got: 1,
                expected: 2
            }
        ));
    }
}
