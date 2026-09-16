// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Valkey Streams as the KV event log.
//!
//! The bridge appends every decoded SGLang event batch to one stream,
//! `<prefix>events`, as the prost-encoded `ApplyExternalKvBatchRequest` it
//! would otherwise have sent over gRPC. Indexer servers consume the stream
//! through a consumer group and apply each entry through the same validated
//! path as the gRPC handler. The keyspace is the snapshot, the stream is the
//! replay window: a restarted or added indexer needs nothing from anyone, and
//! a bridge never needs an indexer to be up.
//!
//! # Ordering
//!
//! Applies are idempotent but not commutative: a worker's REPORT and a later
//! REVOKE of the same block must land in that order. Two consumers of one group
//! would interleave them, so consumers of a shared group run under a lease
//! (`SET NX PX`, renewed at a third of its TTL): one applies, the others stand
//! by and take over inside one TTL. The new holder first reclaims entries the
//! old one left pending (`XAUTOCLAIM`), then reads live. Entries are never lost
//! while they sit in the stream, only delayed.
//!
//! An in-memory indexer uses a private group created at `0`, so it rebuilds
//! its index from the retained window on every start and needs no lease.
//!
//! # Delivery
//!
//! At-least-once. An entry is acknowledged after a successful apply, or when
//! the apply is rejected as malformed (a poison entry is logged and dropped,
//! matching the bridge's own handling of undecodable batches). A transient
//! failure leaves the entry pending; the consumer re-reads its own pending
//! entries in order before touching new ones.

use std::future::Future;
use std::time::Duration;

use prost::Message;
use redis::Value;
use tonic::{Code, Status};
use tracing::{debug, info, warn};

use crate::pb::ApplyExternalKvBatchRequest;
use crate::service::{validate_actions, validate_worker_id};
use crate::valkey_backend::{connect_conn, Conn, ValkeyConfig};
use crate::KvIndexerBackend;

/// Approximate stream length kept by `XADD MAXLEN ~`. One entry is one event
/// batch (a few hundred bytes to a few KB), so this is a bounded replay window,
/// not a full history; the keyspace already holds the current state.
pub const DEFAULT_STREAM_MAXLEN: u64 = 1_000_000;
/// Consumer group shared by Valkey-backed indexers.
pub const DEFAULT_CONSUMER_GROUP: &str = "indexers";
/// Lease TTL for the active consumer of a shared group. Bounds failover time.
pub const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(10);

const READ_BATCH: usize = 256;
const READ_BLOCK: Duration = Duration::from_secs(1);
const RETRY_MIN: Duration = Duration::from_millis(100);
const RETRY_MAX: Duration = Duration::from_secs(5);

pub fn stream_key(prefix: &str) -> String {
    format!("{prefix}events")
}

pub fn lease_key(prefix: &str) -> String {
    format!("{prefix}lease:events")
}

/// Appends apply batches to the event stream.
#[derive(Clone)]
pub struct StreamSink {
    conn: Conn,
    key: String,
    maxlen: u64,
}

impl StreamSink {
    pub async fn connect(config: &ValkeyConfig, maxlen: u64) -> Result<Self, Status> {
        Ok(Self {
            conn: connect_conn(config).await?,
            key: stream_key(&config.key_prefix),
            maxlen,
        })
    }

    /// Returns the stream entry id.
    pub async fn publish(&self, request: &ApplyExternalKvBatchRequest) -> Result<String, Status> {
        let mut pipe = redis::pipe();
        pipe.cmd("XADD")
            .arg(&self.key)
            .arg("MAXLEN")
            .arg("~")
            .arg(self.maxlen)
            .arg("*")
            .arg("w")
            .arg(&request.worker_id)
            .arg("s")
            .arg(request.seq)
            .arg("b")
            .arg(request.encode_to_vec());
        let ids: Vec<String> = self.conn.clone().run(&pipe).await?;
        Ok(ids.into_iter().next().unwrap_or_default())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamStart {
    /// New entries only; the keyspace already holds everything before.
    Tail,
    /// Everything retained; for a consumer that rebuilds an empty index.
    Beginning,
}

#[derive(Clone, Debug)]
pub struct StreamConsumerConfig {
    pub group: String,
    pub consumer: String,
    pub start: StreamStart,
    /// `Some(ttl)`: only the lease holder applies (shared group). `None`: this
    /// consumer always applies (private group).
    pub lease_ttl: Option<Duration>,
    /// Drop the group on clean exit; for private groups so they do not pile up.
    pub destroy_group_on_exit: bool,
}

impl StreamConsumerConfig {
    /// A Valkey-backed indexer: shared group, tail, leased.
    pub fn shared(consumer: impl Into<String>) -> Self {
        Self {
            group: DEFAULT_CONSUMER_GROUP.to_string(),
            consumer: consumer.into(),
            start: StreamStart::Tail,
            lease_ttl: Some(DEFAULT_LEASE_TTL),
            destroy_group_on_exit: false,
        }
    }

    /// An in-memory indexer: private group from the beginning, no lease.
    pub fn private(consumer: impl Into<String>) -> Self {
        let consumer = consumer.into();
        Self {
            group: format!("rebuild-{consumer}"),
            consumer,
            start: StreamStart::Beginning,
            lease_ttl: None,
            destroy_group_on_exit: true,
        }
    }
}

struct Entry {
    id: String,
    worker: String,
    seq: u64,
    body: Vec<u8>,
}

/// Reads the event stream and applies it to a backend.
pub struct StreamConsumer<B> {
    conn: Conn,
    key: String,
    lease: String,
    config: StreamConsumerConfig,
    backend: B,
    holds_lease: bool,
}

impl<B: KvIndexerBackend> StreamConsumer<B> {
    pub async fn connect(
        valkey: &ValkeyConfig,
        config: StreamConsumerConfig,
        backend: B,
    ) -> Result<Self, Status> {
        // Blocking reads must finish inside the response timeout.
        let valkey = valkey
            .clone()
            .with_request_timeout(READ_BLOCK * 3 + valkey.request_timeout);
        let mut consumer = Self {
            conn: connect_conn(&valkey).await?,
            key: stream_key(&valkey.key_prefix),
            lease: lease_key(&valkey.key_prefix),
            config,
            backend,
            holds_lease: false,
        };
        // The group's read position is fixed here, so anything published after
        // `connect` returns is delivered even if `run` starts later.
        consumer.ensure_group().await?;
        Ok(consumer)
    }

    /// Runs until `shutdown` resolves. Transient failures back off and retry.
    pub async fn run(mut self, shutdown: impl Future<Output = ()>) -> Result<(), Status> {
        tokio::pin!(shutdown);
        info!(
            stream = %self.key,
            group = %self.config.group,
            consumer = %self.config.consumer,
            leased = self.config.lease_ttl.is_some(),
            "consuming KV event stream"
        );
        let mut delay = RETRY_MIN;
        // Drain pending entries (ours, or a previous holder's) before live ones.
        let mut drain_pending = true;
        loop {
            let outcome = tokio::select! {
                outcome = self.tick(&mut drain_pending) => outcome,
                () = &mut shutdown => break,
            };
            let pause = match outcome {
                Ok(Tick::Applied) => {
                    delay = RETRY_MIN;
                    continue;
                }
                Ok(Tick::Idle) => {
                    delay = RETRY_MIN;
                    continue;
                }
                Ok(Tick::Standby) => self.config.lease_ttl.unwrap_or(READ_BLOCK) / 3,
                Err(status) => {
                    warn!(%status, retry_in = ?delay, "event stream tick failed");
                    drain_pending = true;
                    let pause = delay;
                    delay = (delay * 2).min(RETRY_MAX);
                    pause
                }
            };
            tokio::select! {
                () = tokio::time::sleep(pause) => {}
                () = &mut shutdown => break,
            }
        }
        if self.holds_lease {
            let _ = self.release_lease().await;
        }
        if self.config.destroy_group_on_exit {
            let _ = self.destroy_group().await;
        }
        Ok(())
    }

    async fn tick(&mut self, drain_pending: &mut bool) -> Result<Tick, Status> {
        if let Some(ttl) = self.config.lease_ttl {
            let had = self.holds_lease;
            self.holds_lease = self.try_lease(ttl).await?;
            if !self.holds_lease {
                if had {
                    info!(consumer = %self.config.consumer, "lost the event lease; standing by");
                }
                return Ok(Tick::Standby);
            }
            if !had {
                info!(consumer = %self.config.consumer, "holding the event lease");
                *drain_pending = true;
            }
        }
        let entries = if *drain_pending {
            let mut entries = match self.config.lease_ttl {
                Some(ttl) => self.autoclaim(ttl).await?,
                None => Vec::new(),
            };
            if entries.is_empty() {
                entries = self.read("0", false).await?;
            }
            if entries.is_empty() {
                *drain_pending = false;
                return Ok(Tick::Idle);
            }
            entries
        } else {
            let mut entries = self.read(">", true).await?;
            // An idle tick is the moment to pick up what a dead consumer left
            // pending; entries become claimable once idle for a lease.
            if entries.is_empty() {
                if let Some(ttl) = self.config.lease_ttl {
                    entries = self.autoclaim(ttl).await?;
                }
            }
            entries
        };
        if entries.is_empty() {
            return Ok(Tick::Idle);
        }
        for entry in entries {
            if let Err(status) = self.apply(entry).await {
                *drain_pending = true;
                return Err(status);
            }
        }
        Ok(Tick::Applied)
    }

    async fn ensure_group(&mut self) -> Result<(), Status> {
        let start = match self.config.start {
            StreamStart::Tail => "$",
            StreamStart::Beginning => "0",
        };
        let mut pipe = redis::pipe();
        pipe.cmd("XGROUP")
            .arg("CREATE")
            .arg(&self.key)
            .arg(&self.config.group)
            .arg(start)
            .arg("MKSTREAM");
        match self.conn.run::<Vec<String>>(&pipe).await {
            Ok(_) => Ok(()),
            Err(status) if status.message().contains("BUSYGROUP") => Ok(()),
            Err(status) => Err(status),
        }
    }

    async fn destroy_group(&mut self) -> Result<(), Status> {
        let mut pipe = redis::pipe();
        pipe.cmd("XGROUP")
            .arg("DESTROY")
            .arg(&self.key)
            .arg(&self.config.group);
        self.conn.run::<Vec<i64>>(&pipe).await.map(|_| ())
    }

    /// Acquire or renew the lease. Renewal is conditional on still owning it.
    async fn try_lease(&mut self, ttl: Duration) -> Result<bool, Status> {
        let ttl_ms = ttl.as_millis().max(1) as u64;
        let mut pipe = redis::pipe();
        pipe.cmd("SET")
            .arg(&self.lease)
            .arg(&self.config.consumer)
            .arg("NX")
            .arg("PX")
            .arg(ttl_ms);
        let acquired: Vec<Option<String>> = self.conn.run(&pipe).await?;
        if acquired.first().is_some_and(Option::is_some) {
            return Ok(true);
        }
        let mut pipe = redis::pipe();
        pipe.cmd("GET").arg(&self.lease);
        let holder: Vec<Option<String>> = self.conn.run(&pipe).await?;
        if holder.first().and_then(|h| h.as_deref()) != Some(self.config.consumer.as_str()) {
            return Ok(false);
        }
        let mut pipe = redis::pipe();
        pipe.cmd("SET")
            .arg(&self.lease)
            .arg(&self.config.consumer)
            .arg("XX")
            .arg("PX")
            .arg(ttl_ms);
        let renewed: Vec<Option<String>> = self.conn.run(&pipe).await?;
        Ok(renewed.first().is_some_and(Option::is_some))
    }

    async fn release_lease(&mut self) -> Result<(), Status> {
        let mut pipe = redis::pipe();
        pipe.cmd("EVAL")
            .arg("if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0")
            .arg(1)
            .arg(&self.lease)
            .arg(&self.config.consumer);
        self.conn.run::<Vec<i64>>(&pipe).await.map(|_| ())
    }

    /// Entries a previous consumer left pending for longer than the lease.
    async fn autoclaim(&mut self, idle: Duration) -> Result<Vec<Entry>, Status> {
        let mut pipe = redis::pipe();
        pipe.cmd("XAUTOCLAIM")
            .arg(&self.key)
            .arg(&self.config.group)
            .arg(&self.config.consumer)
            .arg(idle.as_millis() as u64)
            .arg("0-0")
            .arg("COUNT")
            .arg(READ_BATCH);
        let replies: Vec<Value> = self.conn.run(&pipe).await?;
        let Some(reply) = replies.into_iter().next() else {
            return Ok(Vec::new());
        };
        // [next-start-id, [entries], [deleted-ids]]
        let Value::Array(parts) = reply else {
            return Err(Status::internal(
                "valkey backend: unexpected XAUTOCLAIM reply",
            ));
        };
        match parts.into_iter().nth(1) {
            Some(entries) => parse_entries(entries),
            None => Ok(Vec::new()),
        }
    }

    /// `id` is `>` for new entries or `0` for this consumer's pending ones.
    async fn read(&mut self, id: &str, block: bool) -> Result<Vec<Entry>, Status> {
        let mut pipe = redis::pipe();
        let cmd = pipe.cmd("XREADGROUP");
        cmd.arg("GROUP")
            .arg(&self.config.group)
            .arg(&self.config.consumer)
            .arg("COUNT")
            .arg(READ_BATCH);
        if block {
            cmd.arg("BLOCK").arg(READ_BLOCK.as_millis() as u64);
        }
        cmd.arg("STREAMS").arg(&self.key).arg(id);
        let replies: Vec<Value> = self.conn.run(&pipe).await?;
        let Some(reply) = replies.into_iter().next() else {
            return Ok(Vec::new());
        };
        parse_read_reply(reply)
    }

    async fn ack(&mut self, id: &str) -> Result<(), Status> {
        let mut pipe = redis::pipe();
        pipe.cmd("XACK")
            .arg(&self.key)
            .arg(&self.config.group)
            .arg(id);
        self.conn.run::<Vec<i64>>(&pipe).await.map(|_| ())
    }

    async fn apply(&mut self, entry: Entry) -> Result<(), Status> {
        let request = match ApplyExternalKvBatchRequest::decode(entry.body.as_slice()) {
            Ok(request) => request,
            Err(error) => {
                warn!(id = %entry.id, worker = %entry.worker, %error, "dropping undecodable stream entry");
                return self.ack(&entry.id).await;
            }
        };
        let outcome =
            validate_worker_id(&request.worker_id).and(validate_actions(&request.actions));
        let outcome = match outcome {
            Ok(()) => self
                .backend
                .apply_external_kv_batch(request)
                .await
                .map(|_| ()),
            Err(status) => Err(status),
        };
        match outcome {
            Ok(()) => {
                debug!(id = %entry.id, worker = %entry.worker, seq = entry.seq, "applied stream entry");
                self.ack(&entry.id).await
            }
            Err(status) if is_poison(&status) => {
                warn!(id = %entry.id, worker = %entry.worker, seq = entry.seq, %status, "dropping rejected stream entry");
                self.ack(&entry.id).await
            }
            Err(status) => Err(status),
        }
    }
}

enum Tick {
    Applied,
    Idle,
    Standby,
}

/// A rejection the backend will repeat on every retry; retrying it would block
/// the stream forever.
fn is_poison(status: &Status) -> bool {
    matches!(
        status.code(),
        Code::InvalidArgument
            | Code::FailedPrecondition
            | Code::OutOfRange
            | Code::ResourceExhausted
            | Code::Unimplemented
    )
}

fn bytes_of(value: Value, what: &str) -> Result<Vec<u8>, Status> {
    match value {
        Value::BulkString(bytes) => Ok(bytes),
        Value::SimpleString(text) | Value::VerbatimString { text, .. } => Ok(text.into_bytes()),
        Value::Int(number) => Ok(number.to_string().into_bytes()),
        other => Err(Status::internal(format!(
            "valkey backend: unexpected {what} in stream reply: {other:?}"
        ))),
    }
}

fn parse_entries(value: Value) -> Result<Vec<Entry>, Status> {
    let Value::Array(items) = value else {
        return Err(Status::internal(
            "valkey backend: stream entries are not an array",
        ));
    };
    let mut entries = Vec::with_capacity(items.len());
    for item in items {
        // Trimmed entries show up as nil inside XAUTOCLAIM results.
        let Value::Array(parts) = item else {
            continue;
        };
        let mut parts = parts.into_iter();
        let id = String::from_utf8(bytes_of(
            parts
                .next()
                .ok_or_else(|| Status::internal("stream entry without id"))?,
            "entry id",
        )?)
        .map_err(|_| Status::internal("stream entry id is not UTF-8"))?;
        let fields = match parts.next() {
            Some(Value::Array(fields)) => fields,
            Some(Value::Map(pairs)) => pairs.into_iter().flat_map(|(k, v)| [k, v]).collect(),
            _ => Vec::new(),
        };
        let mut worker = String::new();
        let mut seq = 0u64;
        let mut body = Vec::new();
        let mut fields = fields.into_iter();
        while let (Some(name), Some(value)) = (fields.next(), fields.next()) {
            match bytes_of(name, "field name")?.as_slice() {
                b"w" => {
                    worker = String::from_utf8(bytes_of(value, "worker field")?)
                        .map_err(|_| Status::internal("stream worker field is not UTF-8"))?
                }
                b"s" => {
                    seq = String::from_utf8(bytes_of(value, "seq field")?)
                        .ok()
                        .and_then(|text| text.parse().ok())
                        .unwrap_or(0)
                }
                b"b" => body = bytes_of(value, "body field")?,
                _ => {}
            }
        }
        entries.push(Entry {
            id,
            worker,
            seq,
            body,
        });
    }
    Ok(entries)
}

/// `XREADGROUP` returns nil on timeout, otherwise one `[key, entries]` pair per
/// stream (an array in RESP2, a map in RESP3).
fn parse_read_reply(value: Value) -> Result<Vec<Entry>, Status> {
    match value {
        Value::Nil => Ok(Vec::new()),
        Value::Array(streams) => {
            let mut out = Vec::new();
            for stream in streams {
                if let Value::Array(pair) = stream {
                    if let Some(entries) = pair.into_iter().nth(1) {
                        out.extend(parse_entries(entries)?);
                    }
                }
            }
            Ok(out)
        }
        Value::Map(streams) => {
            let mut out = Vec::new();
            for (_, entries) in streams {
                out.extend(parse_entries(entries)?);
            }
            Ok(out)
        }
        other => Err(Status::internal(format!(
            "valkey backend: unexpected XREADGROUP reply: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bulk(text: &str) -> Value {
        Value::BulkString(text.as_bytes().to_vec())
    }

    #[test]
    fn parses_resp2_read_reply() {
        let reply = Value::Array(vec![Value::Array(vec![
            bulk("{p}:events"),
            Value::Array(vec![Value::Array(vec![
                bulk("1-0"),
                Value::Array(vec![
                    bulk("w"),
                    bulk("worker-0"),
                    bulk("s"),
                    bulk("42"),
                    bulk("b"),
                    Value::BulkString(vec![1, 2, 3]),
                ]),
            ])]),
        ])]);
        let entries = parse_read_reply(reply).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "1-0");
        assert_eq!(entries[0].worker, "worker-0");
        assert_eq!(entries[0].seq, 42);
        assert_eq!(entries[0].body, vec![1, 2, 3]);
        assert!(parse_read_reply(Value::Nil).unwrap().is_empty());
    }

    #[test]
    fn skips_trimmed_autoclaim_entries() {
        let entries = parse_entries(Value::Array(vec![
            Value::Nil,
            Value::Array(vec![bulk("2-0"), Value::Array(vec![bulk("w"), bulk("x")])]),
        ]))
        .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "2-0");
    }

    #[test]
    fn poison_codes_are_acked_not_retried() {
        assert!(is_poison(&Status::invalid_argument("bad")));
        assert!(!is_poison(&Status::unavailable("down")));
        assert!(!is_poison(&Status::internal("bug")));
    }
}
