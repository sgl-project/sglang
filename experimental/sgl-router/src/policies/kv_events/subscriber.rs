//! Per-worker, per-DP-rank ZMQ subscriber for SGLang's `ZmqEventPublisher`.
//!
//! This module owns the I/O plumbing between SGLang workers (which publish on
//! a PUB socket via `python/sglang/srt/utils/event_publisher.py` —
//! KV-cache events from `disaggregation/kv_events.py`, load gauges from
//! `managers/scheduler_components/load_publisher.py`) and the in-memory state
//! consumed by [`super::index::KvEventIndex`]. Each `(worker_url, dp_rank)`
//! pair gets its own SUB socket on its own tokio task, decodes msgpack frames
//! by [`SubKind`] (KV batches via [`super::wire`], load via
//! [`crate::policies::engine_load`]), and forwards [`WorkerEvent`]s to a
//! shared mpsc channel.
//!
//! # Wire format (3-frame multipart)
//!
//! Frames published by SGLang:
//! 1. `topic_bytes` — empty by default, present even when empty.
//! 2. `seq_bytes` — 8-byte big-endian signed `i64`. The publisher emits a
//!    `-1` sentinel (`ZmqEventPublisher.END_SEQ`) on its replay DEALER
//!    socket; we defensively recognise the same value on the PUB stream
//!    and surface it as a [`WorkerEvent::PublisherReset`] so the
//!    downstream pump can clear its cursor before a reconnecting publisher
//!    restarts from seq=1.
//! 3. `payload` — msgpack-encoded [`KvEventBatch`].
//!
//! # Endpoint construction
//!
//! Each call to [`KvEventSubscriberRegistry::add_worker`] takes an
//! [`EventConfig`] describing where the worker publishes:
//! `tcp://{cfg.host}:{cfg.port_base + dp_rank}` per rank in
//! `0..cfg.dp_size`. The host comes from the worker's `/server_info`
//! introspection in production (so wildcard bind hosts resolve to the
//! gateway-routable address) or from the worker URL as a fallback.
//!
//! # Reconnect
//!
//! `zeromq::SubSocket::connect` already spawns a background reconnection
//! task that re-sends our subscriptions on every reconnect, so we do not
//! need an outer reconnect loop. The initial `connect` + `subscribe` is
//! wrapped in a bounded exponential-backoff retry so a worker that just
//! booted (publisher not yet bound) doesn't permanently disable its
//! subscriber. Errors surfaced from `recv()` are logged and the task
//! continues; after [`RECV_ERROR_CEILING`] consecutive errors the task
//! exits with an `error!` log so the silent-stall failure mode is
//! detectable.
//!
//! # Ordering
//!
//! Events for one `(worker, dp_rank)` flow through one task and use one
//! mpsc sender — order is preserved per-worker. Order across DP ranks (or
//! across workers) is **not** preserved; downstream consumers must not
//! depend on it.
//!
//! # Backpressure
//!
//! The per-worker task `await`s `tx.send()` and will not consume new ZMQ
//! messages while the channel is full. ZMQ's HWM (configured by the
//! publisher) takes effect upstream — events are dropped at the publisher,
//! not buffered in the subscriber. Tune the `tx` channel buffer to absorb
//! expected event-batch bursts. Backpressure is per-worker: a slow consumer
//! for one worker stalls only that worker's events, not others.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};
use zeromq::{Socket, SocketRecv, SubSocket, ZmqMessage};

use super::discovery::EventConfig;
use super::tree::KvWorkerId;
use super::wire::{decode_event_batch, KvEventBatch};
use crate::policies::engine_load::{decode_load_stat, LoadStat};

/// Maximum number of consecutive `recv()` errors before the subscriber
/// gives up and exits its task. ZMQ's internal reconnect handles transient
/// network errors, so a stream of consecutive failures means the socket is
/// dead from our perspective; spinning forever masks the failure.
const RECV_ERROR_CEILING: u32 = 64;

/// Bounded retry configuration for the initial connect + subscribe handshake.
/// A worker that just booted may need a few hundred ms before its PUB socket
/// accepts connections; this absorbs the race.
const CONNECT_MAX_ATTEMPTS: u32 = 5;
const CONNECT_BACKOFF_BASE: Duration = Duration::from_millis(50);
const CONNECT_BACKOFF_CAP: Duration = Duration::from_secs(2);

/// Sentinel sequence number meaning "publisher is shutting down". Mirrors
/// `ZmqEventPublisher.END_SEQ = (-1).to_bytes(8, 'big', signed=True)`.
/// SGLang's authoritative emission is on the replay DEALER socket
/// (`_service_replay`); we accept the same sentinel on the PUB stream as
/// defense in depth so a future publisher that does broadcast a shutdown
/// signal is handled correctly.
const END_SEQ_SENTINEL: i64 = -1;

/// Which topic a subscriber task listens on, and therefore what kind of
/// [`WorkerEvent`] it produces. KV-cache events feed the hash tree (with
/// sequence-ordered dedup); load snapshots feed the engine-load table (a
/// gauge — no sequence semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubKind {
    /// Cache-delta topic (`BlockStored` / `BlockRemoved` / `AllBlocksCleared`).
    Kv,
    /// Load-snapshot topic (`LoadStat`).
    Load,
}

/// Message forwarded from a per-worker subscriber task to the pump.
///
/// The variants partition by subscriber [`SubKind`]: `Batch` and
/// `PublisherReset` come only from a `SubKind::Kv` subscriber and carry the
/// cache stream's sequence/replay semantics; `Load` comes only from a
/// `SubKind::Load` subscriber and is a seqless gauge. A given subscriber
/// never emits both families.
#[derive(Debug)]
pub enum WorkerEvent {
    /// A normal decoded event batch.
    Batch {
        /// Identity of the SGLang worker (DP rank) that produced this batch.
        worker: KvWorkerId,
        /// 8-byte big-endian sequence number from the publisher's monotonic
        /// counter. Useful for replay / gap detection downstream.
        seq: i64,
        /// Decoded batch payload.
        batch: KvEventBatch,
    },
    /// A runtime load snapshot from the load topic. Carries no sequence
    /// number: load is a gauge, applied last-value-wins with no dedup.
    Load {
        /// Identity of the SGLang worker (DP rank) that produced this load.
        worker: KvWorkerId,
        /// Latest load snapshot for this `(worker, dp_rank)`.
        load: LoadStat,
    },
    /// The publisher emitted its `END_SEQ` (-1) sentinel, signalling
    /// shutdown. A re-connecting publisher will restart its sequence
    /// counter from 1; the pump uses this to reset the cursor so those
    /// fresh events are not filtered as out-of-order.
    PublisherReset { worker: KvWorkerId },
}

impl WorkerEvent {
    /// The worker that produced this event, regardless of variant.
    pub fn worker(&self) -> &KvWorkerId {
        match self {
            Self::Batch { worker, .. } => worker,
            Self::Load { worker, .. } => worker,
            Self::PublisherReset { worker } => worker,
        }
    }
}

/// Internal handle for one running per-(worker, dp_rank) subscriber task.
struct SubscriberHandle {
    cancel: CancellationToken,
    join: JoinHandle<()>,
}

/// Shared inner state for [`KvEventSubscriberRegistry`].
struct Inner {
    tx: mpsc::Sender<WorkerEvent>,
    /// Keyed by `(worker_url, dp_rank)`. Behind a `tokio::sync::Mutex`
    /// because [`KvEventSubscriberRegistry::remove_worker`] and
    /// [`KvEventSubscriberRegistry::shutdown`] await join handles while
    /// holding the lock conceptually — we drop the lock before awaiting,
    /// but using a tokio mutex avoids accidental blocking-mutex misuse if
    /// the implementation evolves.
    handles: Mutex<HashMap<KvWorkerId, SubscriberHandle>>,
}

/// Owns one ZMQ SUB connection per `(worker_url, dp_rank)`. Forwards
/// decoded batches to a tokio mpsc channel supplied at construction time.
///
/// A registry is single-kind: a [`SubKind::Kv`] registry subscribes to the
/// cache topic and emits [`WorkerEvent::Batch`]; a [`SubKind::Load`] registry
/// subscribes to the load topic and emits [`WorkerEvent::Load`]. The index
/// runs one of each, both feeding the same pump channel, so KV and load
/// subscribers for the same worker never collide in the handle map.
pub struct KvEventSubscriberRegistry {
    inner: Arc<Inner>,
    kind: SubKind,
}

impl KvEventSubscriberRegistry {
    /// Build an empty KV-cache registry. `tx` is where decoded events flow
    /// out; the channel buffer capacity is the caller's choice.
    pub fn new(tx: mpsc::Sender<WorkerEvent>) -> Self {
        Self::with_kind(tx, SubKind::Kv)
    }

    /// Build an empty registry of the given kind.
    pub fn with_kind(tx: mpsc::Sender<WorkerEvent>, kind: SubKind) -> Self {
        Self {
            inner: Arc::new(Inner {
                tx,
                handles: Mutex::new(HashMap::new()),
            }),
            kind,
        }
    }

    /// Open one SUB connection per `dp_rank` in `0..cfg.dp_size`,
    /// connecting to `tcp://{cfg.host}:{cfg.port_base + dp_rank}`. Spawns
    /// background tasks. Idempotent: a second `add_worker` for the same
    /// `(worker_url, dp_rank)` pair is a no-op.
    ///
    /// `worker_url` is the HTTP URL the gateway uses for routing (e.g.,
    /// `"http://10.0.0.1:30000"`). It serves as the keying identity in the
    /// registry but the actual ZMQ endpoint comes from `cfg` — the policy
    /// layer is expected to have learned `cfg` from the worker's
    /// `/server_info` introspection (or filled it from a global fallback).
    ///
    /// # Errors
    ///
    /// If `cfg.port_base + dp_rank` overflows `u16`, that rank is skipped
    /// with a `warn!` log and the remaining ranks proceed.
    pub async fn add_worker(&self, worker_url: &str, cfg: &EventConfig) {
        // (port_base, topic) depend on this registry's kind. KV uses the cache
        // socket + configured topic; Load uses the dedicated load socket (own
        // port range) and subscribe-all (that socket carries only load). A Load
        // registry whose worker advertises no load port (older engine) opens no
        // sockets at all.
        let (port_base, topic) = match self.kind {
            SubKind::Kv => (cfg.port_base, cfg.topic.clone()),
            SubKind::Load => match cfg.load_port_base {
                Some(p) => (p, String::new()),
                None => {
                    debug!(
                        worker_url = %worker_url,
                        "kv-events: worker advertises no load port; skipping load subscribers"
                    );
                    return;
                }
            },
        };
        let mut handles = self.inner.handles.lock().await;
        for dp_rank in 0..cfg.dp_size {
            let id = KvWorkerId {
                url: worker_url.to_string(),
                dp_rank,
            };
            if handles.contains_key(&id) {
                debug!(
                    worker_url = %worker_url,
                    dp_rank,
                    "subscriber already registered; skipping"
                );
                continue;
            }
            let port = match u16::try_from(port_base as u32 + dp_rank) {
                Ok(p) => p,
                Err(_) => {
                    warn!(
                        worker_url = %worker_url,
                        dp_rank,
                        port_base,
                        kind = ?self.kind,
                        "ZMQ event port overflows u16; skipping this rank"
                    );
                    continue;
                }
            };
            let endpoint = format!("tcp://{}:{}", cfg.host, port);
            let cancel = CancellationToken::new();
            let join = spawn_subscriber_task(
                id.clone(),
                endpoint,
                topic.clone(),
                self.kind,
                self.inner.tx.clone(),
                cancel.clone(),
            );
            handles.insert(id, SubscriberHandle { cancel, join });
        }
    }

    /// Cancel all subscribers for `worker_url` and await their shutdown.
    pub async fn remove_worker(&self, worker_url: &str) {
        let drained: Vec<SubscriberHandle> = {
            let mut handles = self.inner.handles.lock().await;
            // Pull out every entry whose URL matches; leave the others.
            let to_drop: Vec<KvWorkerId> = handles
                .keys()
                .filter(|k| k.url == worker_url)
                .cloned()
                .collect();
            to_drop
                .into_iter()
                .filter_map(|k| handles.remove(&k))
                .collect()
        };
        for h in drained {
            h.cancel.cancel();
            // A panicked task surfaces here; we log and continue so one
            // poisoned subscriber cannot stall the registry.
            if let Err(e) = h.join.await {
                warn!(
                    worker_url = %worker_url,
                    error = %e,
                    "subscriber task did not join cleanly"
                );
            }
        }
    }

    /// Sync cancellation: triggers every per-worker token without awaiting
    /// the join handles. Use this when you cannot `.await` (e.g., from
    /// `Drop`). After calling this, the subscriber tasks will exit on their
    /// next yield point. Subscriptions and ZMQ sockets are released by
    /// tokio task cleanup.
    ///
    /// If `try_lock` fails, the registry is mid-mutation elsewhere
    /// (`shutdown`, `add_worker`, `remove_worker`); the cancel is
    /// redundant in that case so we drop the call.
    pub fn cancel_all(&self) {
        if let Ok(handles) = self.inner.handles.try_lock() {
            for h in handles.values() {
                h.cancel.cancel();
            }
        }
    }

    /// Cancel everything and await shutdown. Caller is responsible for
    /// draining any remaining events on the receiver side.
    pub async fn shutdown(&self) {
        let drained: Vec<(KvWorkerId, SubscriberHandle)> = {
            let mut handles = self.inner.handles.lock().await;
            handles.drain().collect()
        };
        for (id, h) in drained {
            h.cancel.cancel();
            if let Err(e) = h.join.await {
                warn!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    error = %e,
                    "subscriber task did not join cleanly during shutdown"
                );
            }
        }
    }
}

/// Spawn the background task that owns one SUB socket and forwards
/// decoded batches.
fn spawn_subscriber_task(
    id: KvWorkerId,
    endpoint: String,
    topic: String,
    kind: SubKind,
    tx: mpsc::Sender<WorkerEvent>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        run_subscriber(id, endpoint, topic, kind, tx, cancel).await;
    })
}

/// Inner subscriber loop. Returns when:
///   * the cancellation token fires, OR
///   * the downstream mpsc receiver is dropped, OR
///   * the initial connect/subscribe fails after [`CONNECT_MAX_ATTEMPTS`]
///     attempts with exponential backoff, OR
///   * `recv()` returns errors [`RECV_ERROR_CEILING`] times in a row
///     (escalated to `error!` so the silent stall is detectable).
async fn run_subscriber(
    id: KvWorkerId,
    endpoint: String,
    topic: String,
    kind: SubKind,
    tx: mpsc::Sender<WorkerEvent>,
    cancel: CancellationToken,
) {
    debug!(
        worker_url = %id.url,
        dp_rank = id.dp_rank,
        endpoint = %endpoint,
        topic = %topic,
        kind = ?kind,
        "starting kv-event subscriber"
    );

    let mut sub = match connect_with_backoff(&id, &endpoint, &topic, &cancel).await {
        Some(s) => s,
        None => return,
    };

    let mut errors_in_a_row = 0u32;
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                debug!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    "subscriber cancelled"
                );
                return;
            }
            res = sub.recv() => {
                match res {
                    Ok(msg) => {
                        errors_in_a_row = 0;
                        if let Some(event) = decode_message(&id, msg, kind) {
                            if tx.send(event).await.is_err() {
                                // The pump (or the entire index) is gone.
                                // This is unexpected mid-stream; warn so
                                // operators see it.
                                warn!(
                                    worker_url = %id.url,
                                    dp_rank = id.dp_rank,
                                    "downstream mpsc receiver dropped; exiting"
                                );
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        errors_in_a_row += 1;
                        if errors_in_a_row >= RECV_ERROR_CEILING {
                            error!(
                                worker_url = %id.url,
                                dp_rank = id.dp_rank,
                                endpoint = %endpoint,
                                error = %e,
                                consecutive_errors = errors_in_a_row,
                                "SUB socket has produced {RECV_ERROR_CEILING} consecutive recv errors; giving up on this subscriber"
                            );
                            return;
                        }
                        // SubSocket auto-reconnects internally; transient
                        // errors should resume once a new peer attaches.
                        warn!(
                            worker_url = %id.url,
                            dp_rank = id.dp_rank,
                            error = %e,
                            consecutive_errors = errors_in_a_row,
                            "recv error from SUB socket; continuing"
                        );
                        tokio::task::yield_now().await;
                    }
                }
            }
        }
    }
}

/// Open a `SubSocket`, connect to `endpoint`, and subscribe to the
/// supplied `topic` prefix (empty string = receive every message,
/// matching the prior all-topics behavior).
///
/// Retries with exponential backoff up to [`CONNECT_MAX_ATTEMPTS`] times
/// so a worker that just booted (publisher not yet bound) doesn't
/// permanently disable its KV-event subscriber.
///
/// Returns `None` if cancelled or if every attempt fails. All operations
/// are guarded by the cancellation token so shutdown is not delayed by
/// the backoff.
async fn connect_with_backoff(
    id: &KvWorkerId,
    endpoint: &str,
    topic: &str,
    cancel: &CancellationToken,
) -> Option<SubSocket> {
    let mut delay = CONNECT_BACKOFF_BASE;
    for attempt in 1..=CONNECT_MAX_ATTEMPTS {
        let mut sub = SubSocket::new();
        let connect_res = tokio::select! {
            _ = cancel.cancelled() => {
                debug!(worker_url = %id.url, dp_rank = id.dp_rank, "cancelled before connect");
                return None;
            }
            res = sub.connect(endpoint) => res,
        };
        if let Err(e) = connect_res {
            warn!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                endpoint = %endpoint,
                attempt,
                error = %e,
                "kv-events: connect SUB socket failed; retrying"
            );
        } else {
            let subscribe_res = tokio::select! {
                _ = cancel.cancelled() => {
                    debug!(worker_url = %id.url, dp_rank = id.dp_rank, "cancelled before subscribe");
                    return None;
                }
                res = sub.subscribe(topic) => res,
            };
            match subscribe_res {
                Ok(()) => return Some(sub),
                Err(e) => warn!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    endpoint = %endpoint,
                    attempt,
                    topic = %topic,
                    error = %e,
                    "kv-events: SUB subscribe failed; retrying"
                ),
            }
        }
        if attempt == CONNECT_MAX_ATTEMPTS {
            break;
        }
        tokio::select! {
            _ = cancel.cancelled() => {
                debug!(worker_url = %id.url, dp_rank = id.dp_rank, "cancelled during connect backoff");
                return None;
            }
            _ = tokio::time::sleep(delay) => {}
        }
        delay = (delay * 2).min(CONNECT_BACKOFF_CAP);
    }
    error!(
        worker_url = %id.url,
        dp_rank = id.dp_rank,
        endpoint = %endpoint,
        attempts = CONNECT_MAX_ATTEMPTS,
        "kv-events: gave up establishing SUB socket after {CONNECT_MAX_ATTEMPTS} attempts; this worker's cache-aware routing is disabled until next add_worker"
    );
    None
}

/// Validate, parse, and decode a single 3-frame multipart ZMQ message.
/// Returns `None` (with logging) for any non-event input (bad frame
/// count, sentinel sequence, or msgpack decode error). `kind` selects
/// whether to emit a KV [`WorkerEvent::Batch`] or a [`WorkerEvent::Load`].
fn decode_message(id: &KvWorkerId, msg: ZmqMessage, kind: SubKind) -> Option<WorkerEvent> {
    if msg.len() != 3 {
        warn!(
            worker_url = %id.url,
            dp_rank = id.dp_rank,
            frames = msg.len(),
            "dropping ZMQ message with unexpected frame count (expected 3)"
        );
        return None;
    }

    // Frame 0 is the topic; we don't use it. Frame 1 is the BE i64 seq;
    // frame 2 is the msgpack payload. The `len() == 3` guard above means
    // these indices are always valid, but `?` cleanly bails out if a
    // future change drops the guard.
    let seq_frame = msg.get(1)?;
    let payload = msg.get(2)?;

    // Decode the 8-byte BE seq. Frames smaller or larger than 8 bytes
    // mean a malformed publisher; log and drop.
    let seq_bytes: [u8; 8] = match seq_frame.as_ref().try_into() {
        Ok(b) => b,
        Err(_) => {
            warn!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                seq_len = seq_frame.len(),
                "dropping message with non-8-byte sequence frame"
            );
            return None;
        }
    };
    let seq = i64::from_be_bytes(seq_bytes);

    if seq == END_SEQ_SENTINEL {
        match kind {
            SubKind::Kv => {
                info!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    "publisher signalled shutdown (END_SEQ); forwarding cursor reset"
                );
                return Some(WorkerEvent::PublisherReset { worker: id.clone() });
            }
            // Load has no cursor / replay state to reset — just drop.
            SubKind::Load => return None,
        }
    }

    // Decode by kind: the cache topic carries `KvEventBatch`es, the load
    // topic carries bare `LoadStat` snapshots — two independent wire formats
    // on two independent sockets.
    match kind {
        SubKind::Kv => {
            let batch = match decode_event_batch(payload.as_ref()) {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        worker_url = %id.url,
                        dp_rank = id.dp_rank,
                        seq,
                        error = %e,
                        "failed to decode KV event batch payload; dropping"
                    );
                    return None;
                }
            };
            trace!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                seq,
                n_events = batch.events.len(),
                "decoded KV event batch"
            );
            Some(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch,
            })
        }
        SubKind::Load => {
            let load = match decode_load_stat(payload.as_ref()) {
                Ok(l) => l,
                Err(e) => {
                    warn!(
                        worker_url = %id.url,
                        dp_rank = id.dp_rank,
                        seq,
                        error = %e,
                        "failed to decode load snapshot payload; dropping"
                    );
                    return None;
                }
            };
            trace!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                seq,
                "decoded load snapshot"
            );
            Some(WorkerEvent::Load {
                worker: id.clone(),
                load,
            })
        }
    }
}

/// Pull the host out of a routing URL like `http://10.0.0.1:30000` or
/// `https://[::1]:30000`. Falls back to `None` for inputs the `url` crate
/// cannot parse.
///
/// Test-only helper for fabricating [`EventConfig`]s from a worker URL.
#[cfg(test)]
fn extract_host(worker_url: &str) -> Option<String> {
    let parsed = url::Url::parse(worker_url).ok()?;
    parsed.host_str().map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// Tests — bind real PUB sockets to ephemeral ports and confirm the
// subscriber wires data through correctly. All tests are localhost-only and
// use OS-assigned ports so they can run in parallel without conflict.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use bytes::Bytes;
    use tokio::time::timeout;
    use zeromq::{Endpoint, PubSocket, Socket, SocketSend, ZmqMessage};

    use crate::policies::kv_events::wire::KvCacheEvent;

    mod helpers {
        use super::*;
        use rmp::encode as mp;

        /// Bind a PUB socket to an OS-assigned localhost port and return
        /// `(socket, port)`.
        pub async fn make_pub_bound() -> (PubSocket, u16) {
            let mut sock = PubSocket::new();
            let endpoint = sock
                .bind("tcp://127.0.0.1:0")
                .await
                .expect("bind PUB socket");
            let port = match endpoint {
                Endpoint::Tcp(_, p) => p,
                other => panic!("unexpected endpoint: {other:?}"),
            };
            (sock, port)
        }

        /// Build a minimal [`EventConfig`] for test fixtures: take the host
        /// from `worker_url` (matches the pre-discovery behavior) and fill
        /// the rest with reasonable defaults.
        pub fn cfg_for(worker_url: &str, port_base: u16, dp_size: u32) -> EventConfig {
            EventConfig {
                host: extract_host(worker_url).unwrap_or_else(|| "127.0.0.1".to_string()),
                port_base,
                topic: String::new(),
                load_port_base: None,
                block_size: 64,
                dp_size,
                is_bigram: false,
            }
        }

        /// Encode a minimal AllBlocksCleared batch with the given ts and
        /// optional dp_rank, in the same array layout msgspec emits.
        pub fn encode_all_blocks_cleared_batch(ts: f64, attn_dp_rank: Option<u32>) -> Vec<u8> {
            let mut buf = Vec::new();
            // Outer batch array: [ts, [event], dp_rank?]
            mp::write_array_len(&mut buf, 3).unwrap();
            mp::write_f64(&mut buf, ts).unwrap();
            // events array length 1
            mp::write_array_len(&mut buf, 1).unwrap();
            // event = ["AllBlocksCleared"]
            mp::write_array_len(&mut buf, 1).unwrap();
            mp::write_str(&mut buf, "AllBlocksCleared").unwrap();
            match attn_dp_rank {
                Some(v) => {
                    mp::write_uint(&mut buf, v as u64).unwrap();
                }
                None => mp::write_nil(&mut buf).unwrap(),
            }
            buf
        }

        /// Encode a LoadStat batch `[ts, [["LoadStat", running, waiting,
        /// num_tokens, max_total]], dp_rank?]` in msgspec's array layout.
        /// Encode a bare LoadStat msgpack array `["LoadStat", running, waiting,
        /// num_tokens, max_total, attn_dp_rank]` — the payload on the load
        /// socket (no EventBatch envelope).
        pub fn encode_load_stat(
            running: u64,
            waiting: u64,
            num_tokens: u64,
            max_total: u64,
            attn_dp_rank: u32,
        ) -> Vec<u8> {
            let mut buf = Vec::new();
            mp::write_array_len(&mut buf, 6).unwrap();
            mp::write_str(&mut buf, "LoadStat").unwrap();
            mp::write_uint(&mut buf, running).unwrap();
            mp::write_uint(&mut buf, waiting).unwrap();
            mp::write_uint(&mut buf, num_tokens).unwrap();
            mp::write_uint(&mut buf, max_total).unwrap();
            mp::write_uint(&mut buf, attn_dp_rank as u64).unwrap();
            buf
        }

        /// Build a 3-frame multipart with topic="", the given seq (BE i64),
        /// and the given payload bytes.
        pub fn build_multipart(seq: i64, payload: Vec<u8>) -> ZmqMessage {
            build_multipart_with_topic(b"", seq, payload)
        }

        /// Build a 3-frame multipart with an explicit topic frame.
        pub fn build_multipart_with_topic(topic: &[u8], seq: i64, payload: Vec<u8>) -> ZmqMessage {
            let mut msg = ZmqMessage::from(Bytes::copy_from_slice(topic));
            msg.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
            msg.push_back(Bytes::from(payload));
            msg
        }

        /// Wait briefly for the SubSocket to finish its handshake/subscribe.
        /// 50ms is empirically enough on localhost without making tests
        /// flaky.
        pub async fn settle() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        /// Destructure a `WorkerEvent::Batch`, panicking on any other
        /// variant. Keeps test assertions terse.
        pub fn expect_batch(ev: WorkerEvent) -> (KvWorkerId, i64, KvEventBatch) {
            match ev {
                WorkerEvent::Batch { worker, seq, batch } => (worker, seq, batch),
                WorkerEvent::Load { worker, .. } => {
                    panic!("expected Batch, got Load for {worker:?}")
                }
                WorkerEvent::PublisherReset { worker } => {
                    panic!("expected Batch, got PublisherReset for {worker:?}")
                }
            }
        }
    }

    /// Single subscriber: publish one batch, see one batch.
    #[tokio::test]
    async fn single_subscriber_receives_one_event() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 1),
            )
            .await;
        helpers::settle().await;

        let payload = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        let msg = helpers::build_multipart(7, payload);
        pub_sock.send(msg).await.expect("send");

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("recv timed out")
            .expect("channel closed");
        let (worker, seq, batch) = helpers::expect_batch(event);

        assert_eq!(seq, 7);
        assert_eq!(worker.dp_rank, 0);
        assert_eq!(worker.url, "http://127.0.0.1:30000");
        assert_eq!(batch.events.len(), 1);
        assert!(matches!(batch.events[0], KvCacheEvent::AllBlocksCleared));

        let shutdown_done = timeout(Duration::from_millis(500), registry.shutdown()).await;
        assert!(shutdown_done.is_ok(), "shutdown should return promptly");
    }

    /// When the worker advertises a non-empty topic in
    /// `EventConfig.topic`, the SUB socket must filter on that prefix:
    /// only messages whose first frame *starts with* the topic bytes
    /// reach our pump. ZMQ-level filtering is the only way the
    /// configured topic affects routing — `decode_message` discards
    /// frame 0 regardless — so a SUB socket that ignores `cfg.topic`
    /// and subscribes to `""` lets every message on the endpoint
    /// through, including events from unrelated publishers that
    /// happen to share the host:port (e.g. a colocated worker
    /// running a different model on the same machine).
    ///
    /// Scenario: subscribe to topic "match". Publish two messages on
    /// the same PUB socket: topic=`match` first, then topic=`other`.
    /// The matched message must be delivered AND the unmatched one
    /// must not. We publish matched-first so the negative assertion
    /// is the load-bearing check: a broken SUB filter that subscribes
    /// to `""` (the pre-fix behavior) delivers BOTH messages in send
    /// order, so the seq=22 assertion would still pass but the
    /// stray-recv assertion would catch it. This removes a dependency
    /// on PUB→SUB delivery ordering as the discriminator.
    #[tokio::test]
    async fn subscriber_filters_by_configured_topic() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        let worker_url = "http://127.0.0.1:30100";
        let mut cfg = helpers::cfg_for(worker_url, port, 1);
        cfg.topic = "match".into();
        registry.add_worker(worker_url, &cfg).await;
        helpers::settle().await;

        // Publish matched first, then `other`. A leaky `""` subscription
        // delivers both in order; the topic filter must drop the second.
        let payload_matched = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        let payload_other = helpers::encode_all_blocks_cleared_batch(2.0, Some(0));
        pub_sock
            .send(helpers::build_multipart_with_topic(
                b"match",
                22,
                payload_matched,
            ))
            .await
            .unwrap();
        pub_sock
            .send(helpers::build_multipart_with_topic(
                b"other",
                11,
                payload_other,
            ))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out waiting for matched event")
            .expect("channel closed");
        let (_, seq, _) = helpers::expect_batch(event);
        assert_eq!(seq, 22, "matched message must arrive; got seq={seq}");

        // The load-bearing assertion: no second message in 200ms. A
        // SUB subscribed to `""` would have delivered the `other`
        // message by now; the topic filter must drop it.
        let stray = timeout(Duration::from_millis(200), rx.recv()).await;
        assert!(
            stray.is_err(),
            "second message with topic=`other` must NOT pass the filter \
             (got {stray:?}); cfg.topic is being ignored at subscribe()",
        );

        registry.shutdown().await;
    }

    /// DP rank fan-out: 3 PUB sockets, 3 distinct events, all delivered.
    #[tokio::test]
    async fn dp_rank_fan_out() {
        let (mut pub0, p0) = helpers::make_pub_bound().await;
        let (mut pub1, p1) = helpers::make_pub_bound().await;
        let (mut pub2, p2) = helpers::make_pub_bound().await;
        // We need contiguous ports for `base_port + dp_rank` to land on
        // each PUB socket. OS-assigned ports won't be contiguous, so we
        // bind one PUB socket per dp_rank with the same `worker_url` but
        // call `add_worker` three times with `dp_size=1` and the right
        // base_port for each. The registry does not require contiguous
        // ports per call — but `add_worker` itself does, since it
        // constructs `base_port + rank`. Workaround: use distinct
        // `worker_url`s so each call's dp_rank=0 maps to its own port,
        // and assert via the URL field.
        let url0 = "http://127.0.0.1:30000";
        let url1 = "http://127.0.0.1:30001";
        let url2 = "http://127.0.0.1:30002";

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(16);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry
            .add_worker(url0, &helpers::cfg_for(url0, p0, 1))
            .await;
        registry
            .add_worker(url1, &helpers::cfg_for(url1, p1, 1))
            .await;
        registry
            .add_worker(url2, &helpers::cfg_for(url2, p2, 1))
            .await;
        helpers::settle().await;

        let payload0 = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        let payload1 = helpers::encode_all_blocks_cleared_batch(2.0, Some(1));
        let payload2 = helpers::encode_all_blocks_cleared_batch(3.0, Some(2));

        pub0.send(helpers::build_multipart(10, payload0))
            .await
            .unwrap();
        pub1.send(helpers::build_multipart(20, payload1))
            .await
            .unwrap();
        pub2.send(helpers::build_multipart(30, payload2))
            .await
            .unwrap();

        let mut seq_by_url: HashMap<String, i64> = HashMap::new();
        for _ in 0..3 {
            let event = timeout(Duration::from_millis(500), rx.recv())
                .await
                .expect("timed out")
                .expect("channel closed");
            let (worker, seq, _batch) = helpers::expect_batch(event);
            seq_by_url.insert(worker.url, seq);
        }

        assert_eq!(seq_by_url.len(), 3);
        assert_eq!(seq_by_url[url0], 10);
        assert_eq!(seq_by_url[url1], 20);
        assert_eq!(seq_by_url[url2], 30);

        registry.shutdown().await;
    }

    /// True per-DP fan-out behind a single worker URL: bind 3 PUB
    /// sockets on contiguous ports and subscribe with `dp_size=3`.
    #[tokio::test]
    async fn dp_size_three_per_worker() {
        // Pick a single base port and keep retrying until the next two
        // ports are also free, so `base_port + 1` and `base_port + 2`
        // really resolve to our PUB sockets.
        let mut attempt = 0;
        let (pub0, pub1, pub2, base_port) = loop {
            attempt += 1;
            assert!(attempt < 32, "could not find 3 contiguous free ports");

            // Bind PUB at OS-assigned port to learn what's free, then try
            // to bind the next two ports explicitly.
            let mut p0 = PubSocket::new();
            let ep0 = p0.bind("tcp://127.0.0.1:0").await.unwrap();
            let base = match ep0 {
                Endpoint::Tcp(_, p) => p,
                _ => unreachable!(),
            };

            let mut p1 = PubSocket::new();
            let ep1 = p1.bind(&format!("tcp://127.0.0.1:{}", base + 1)).await;
            if ep1.is_err() {
                continue;
            }

            let mut p2 = PubSocket::new();
            let ep2 = p2.bind(&format!("tcp://127.0.0.1:{}", base + 2)).await;
            if ep2.is_err() {
                continue;
            }
            break (p0, p1, p2, base);
        };

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(16);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", base_port, 3),
            )
            .await;
        helpers::settle().await;

        let mut pub0 = pub0;
        let mut pub1 = pub1;
        let mut pub2 = pub2;
        pub0.send(helpers::build_multipart(
            100,
            helpers::encode_all_blocks_cleared_batch(1.0, Some(0)),
        ))
        .await
        .unwrap();
        pub1.send(helpers::build_multipart(
            200,
            helpers::encode_all_blocks_cleared_batch(2.0, Some(1)),
        ))
        .await
        .unwrap();
        pub2.send(helpers::build_multipart(
            300,
            helpers::encode_all_blocks_cleared_batch(3.0, Some(2)),
        ))
        .await
        .unwrap();

        let mut by_rank: HashMap<u32, i64> = HashMap::new();
        for _ in 0..3 {
            let event = timeout(Duration::from_millis(500), rx.recv())
                .await
                .expect("timed out")
                .expect("channel closed");
            let (worker, seq, _batch) = helpers::expect_batch(event);
            assert_eq!(worker.url, "http://127.0.0.1:30000");
            by_rank.insert(worker.dp_rank, seq);
        }
        assert_eq!(by_rank.get(&0), Some(&100));
        assert_eq!(by_rank.get(&1), Some(&200));
        assert_eq!(by_rank.get(&2), Some(&300));

        registry.shutdown().await;
    }

    /// 8-rank multi-publisher fan-out: a worker that publishes to 8
    /// contiguous ZMQ ports (one per DP rank) must produce 8 distinct
    /// SUB connections and forward every rank's event. The 3-rank
    /// test above pins basic fan-out; this one exercises the wider
    /// fan-out shape that real multi-DP workers exhibit.
    #[tokio::test]
    async fn dp_size_eight_per_worker() {
        const N: usize = 8;
        let mut attempt = 0;
        let mut publishers: Vec<PubSocket> = Vec::new();
        let base_port: u16 = loop {
            attempt += 1;
            assert!(attempt < 64, "could not find 8 contiguous free ports");
            publishers.clear();

            let mut p0 = PubSocket::new();
            let ep0 = p0.bind("tcp://127.0.0.1:0").await.unwrap();
            let base = match ep0 {
                Endpoint::Tcp(_, p) => p,
                _ => unreachable!(),
            };
            // Ensure `base + N - 1` fits in u16 *and* we can bind every
            // contiguous port. Retry on the rare overflow case at the
            // high end of the ephemeral range.
            if u32::from(base) + (N as u32) > u32::from(u16::MAX) {
                continue;
            }
            publishers.push(p0);
            let mut ok = true;
            for offset in 1..N as u16 {
                let mut p = PubSocket::new();
                let res = p.bind(&format!("tcp://127.0.0.1:{}", base + offset)).await;
                if res.is_err() {
                    ok = false;
                    break;
                }
                publishers.push(p);
            }
            if ok {
                break base;
            }
        };

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(64);
        let registry = KvEventSubscriberRegistry::new(tx);
        let worker_url = "http://127.0.0.1:30000";
        registry
            .add_worker(
                worker_url,
                &helpers::cfg_for(worker_url, base_port, N as u32),
            )
            .await;
        helpers::settle().await;

        for (rank, pubsock) in publishers.iter_mut().enumerate() {
            pubsock
                .send(helpers::build_multipart(
                    1000 + rank as i64,
                    helpers::encode_all_blocks_cleared_batch(rank as f64, Some(rank as u32)),
                ))
                .await
                .unwrap();
        }

        let mut by_rank: HashMap<u32, i64> = HashMap::new();
        for _ in 0..N {
            let event = timeout(Duration::from_millis(500), rx.recv())
                .await
                .expect("timed out")
                .expect("channel closed");
            let (worker, seq, _batch) = helpers::expect_batch(event);
            assert_eq!(worker.url, worker_url);
            by_rank.insert(worker.dp_rank, seq);
        }
        assert_eq!(by_rank.len(), N, "every rank must produce an event");
        for rank in 0..N as u32 {
            assert_eq!(
                by_rank.get(&rank),
                Some(&(1000 + rank as i64)),
                "rank {rank} missing or wrong seq",
            );
        }

        registry.shutdown().await;
    }

    /// Bad msgpack payload is logged and dropped; subsequent valid event
    /// still arrives.
    #[tokio::test]
    async fn decoding_error_tolerated() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1",
                &helpers::cfg_for("http://127.0.0.1", port, 1),
            )
            .await;
        helpers::settle().await;

        // Garbage payload (not msgpack).
        pub_sock
            .send(helpers::build_multipart(1, vec![0xff, 0xfe, 0xfd]))
            .await
            .unwrap();
        // Then a valid one.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        pub_sock
            .send(helpers::build_multipart(2, payload))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        let (_worker, seq, batch) = helpers::expect_batch(event);
        assert_eq!(seq, 2);
        // We must NOT have received the bad message.
        assert!(matches!(batch.events[0], KvCacheEvent::AllBlocksCleared));

        registry.shutdown().await;
    }

    /// 2-frame and 4-frame messages are dropped; valid 3-frame still works.
    #[tokio::test]
    async fn wrong_frame_count_tolerated() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1",
                &helpers::cfg_for("http://127.0.0.1", port, 1),
            )
            .await;
        helpers::settle().await;

        // 2-frame: just topic + payload.
        let mut bad2 = ZmqMessage::from(Bytes::new());
        bad2.push_back(Bytes::from_static(b"junk"));
        pub_sock.send(bad2).await.unwrap();

        // 4-frame: topic + seq + payload + extra.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        let mut bad4 = helpers::build_multipart(99, payload.clone());
        bad4.push_back(Bytes::from_static(b"extra"));
        pub_sock.send(bad4).await.unwrap();

        // Valid 3-frame.
        pub_sock
            .send(helpers::build_multipart(42, payload))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        let (_worker, seq, _batch) = helpers::expect_batch(event);
        assert_eq!(seq, 42);

        registry.shutdown().await;
    }

    /// END_SEQ sentinel (-1) is forwarded as a `PublisherReset` so the
    /// downstream pump can clear its cursor; a subsequent valid event
    /// still arrives as a normal `Batch`.
    #[tokio::test]
    async fn sequence_number_sentinel_propagates_as_reset() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1",
                &helpers::cfg_for("http://127.0.0.1", port, 1),
            )
            .await;
        helpers::settle().await;

        pub_sock
            .send(helpers::build_multipart(-1, b"ignored".to_vec()))
            .await
            .unwrap();
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        pub_sock
            .send(helpers::build_multipart(5, payload))
            .await
            .unwrap();

        let first = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert!(
            matches!(first, WorkerEvent::PublisherReset { .. }),
            "END_SEQ must surface as PublisherReset, got {first:?}",
        );

        let second = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        let (_worker, seq, _batch) = helpers::expect_batch(second);
        assert_eq!(seq, 5);

        registry.shutdown().await;
    }

    /// `remove_worker` cancels the task; further publishes are not
    /// received.
    #[tokio::test]
    async fn remove_worker_cancels() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 1),
            )
            .await;
        helpers::settle().await;

        // First event arrives.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        pub_sock
            .send(helpers::build_multipart(1, payload.clone()))
            .await
            .unwrap();
        let _ = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("first event timed out");

        // Remove and verify the handle map empties.
        registry.remove_worker("http://127.0.0.1:30000").await;
        {
            let handles = registry.inner.handles.lock().await;
            assert!(
                handles.is_empty(),
                "handles map should be empty after remove"
            );
        }

        // Publish more — receiver should see nothing.
        pub_sock
            .send(helpers::build_multipart(2, payload))
            .await
            .unwrap();
        let res = timeout(Duration::from_millis(150), rx.recv()).await;
        assert!(
            res.is_err(),
            "no event should arrive after remove_worker (got {:?})",
            res.unwrap()
        );
    }

    /// Calling `add_worker` twice for the same `(url, dp_rank)` pair
    /// must not double-spawn.
    #[tokio::test]
    async fn add_worker_idempotent() {
        let (_pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, _rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 1),
            )
            .await;
        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 1),
            )
            .await;

        {
            let handles = registry.inner.handles.lock().await;
            assert_eq!(handles.len(), 1, "expected 1 entry, got {}", handles.len());
        }

        registry.shutdown().await;
    }

    /// `cancel_all` signals every per-worker token without awaiting; a
    /// subsequent `shutdown` must still complete cleanly. This pins the
    /// contract for any future `Drop` impl that needs a sync cancel path
    /// (e.g. when the registry is dropped without an explicit `shutdown`).
    #[tokio::test]
    async fn cancel_all_then_shutdown_is_clean() {
        let (_pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, _rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry
            .add_worker(
                "http://127.0.0.1:30000",
                &helpers::cfg_for("http://127.0.0.1:30000", port, 2),
            )
            .await;
        helpers::settle().await;

        // Sync cancel — must not block, must not panic.
        registry.cancel_all();

        // shutdown should still join cleanly even though the per-worker
        // tokens were already fired by cancel_all.
        let done = timeout(Duration::from_millis(500), registry.shutdown()).await;
        assert!(done.is_ok(), "shutdown after cancel_all must not hang");
    }

    /// Direct unit test of [`extract_host`] — no socket required.
    #[test]
    fn extract_host_handles_common_urls() {
        assert_eq!(
            extract_host("http://10.0.0.1:30000").as_deref(),
            Some("10.0.0.1")
        );
        assert_eq!(
            extract_host("https://my.host.example:443").as_deref(),
            Some("my.host.example")
        );
        // url crate strips brackets from IPv6 literals in host_str().
        assert_eq!(extract_host("http://[::1]:30000").as_deref(), Some("[::1]"));
        assert!(extract_host("not a url").is_none());
    }

    /// Direct unit test of [`decode_message`] — exercises sentinel and
    /// bad-frame paths without involving sockets.
    #[test]
    fn decode_message_unit() {
        let id = KvWorkerId {
            url: "http://x".to_string(),
            dp_rank: 0,
        };

        // Wrong frame count.
        let one_frame = ZmqMessage::from(Bytes::from_static(b"only"));
        assert!(decode_message(&id, one_frame, SubKind::Kv).is_none());

        // Sentinel seq = -1 now surfaces as PublisherReset (not None) so
        // the downstream pump can clear its cursor before a reconnecting
        // publisher restarts from seq=1.
        let sentinel = helpers::build_multipart(-1, b"ignored".to_vec());
        let reset = decode_message(&id, sentinel, SubKind::Kv).expect("END_SEQ forwards");
        assert!(matches!(reset, WorkerEvent::PublisherReset { .. }));

        // Bad seq frame length.
        let mut bad_seq = ZmqMessage::from(Bytes::new());
        bad_seq.push_back(Bytes::from_static(b"abc")); // 3 bytes, not 8
        bad_seq.push_back(Bytes::from_static(b""));
        assert!(decode_message(&id, bad_seq, SubKind::Kv).is_none());

        // Bad payload.
        let bad_payload = helpers::build_multipart(1, vec![0xff, 0xfe]);
        assert!(decode_message(&id, bad_payload, SubKind::Kv).is_none());

        // Happy path.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        let good = helpers::build_multipart(7, payload);
        let event = decode_message(&id, good, SubKind::Kv).expect("should decode");
        let (worker, seq, _batch) = helpers::expect_batch(event);
        assert_eq!(seq, 7);
        assert_eq!(worker, id);
    }

    /// A `SubKind::Load` subscriber decodes a bare LoadStat frame into
    /// `WorkerEvent::Load`, and drops the END_SEQ sentinel (no cursor state).
    #[test]
    fn decode_message_load_kind() {
        let id = KvWorkerId {
            url: "http://x".to_string(),
            dp_rank: 1,
        };

        // END_SEQ is dropped for the load topic.
        let sentinel = helpers::build_multipart(-1, b"ignored".to_vec());
        assert!(decode_message(&id, sentinel, SubKind::Load).is_none());

        // A bare LoadStat frame becomes WorkerEvent::Load.
        let payload = helpers::encode_load_stat(5, 2, 100, 1000, 1);
        let msg = helpers::build_multipart(3, payload);
        let event = decode_message(&id, msg, SubKind::Load).expect("should decode load");
        match event {
            WorkerEvent::Load { worker, load } => {
                assert_eq!(worker, id);
                assert_eq!(load.num_running_reqs, 5);
                assert_eq!(load.num_waiting_reqs, 2);
                assert_eq!(load.num_tokens, 100);
                assert_eq!(load.max_total_num_tokens, 1000);
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    /// Restart-resume contract: after a worker is removed and then re-added
    /// to the same endpoint, the new subscriber must connect and forward
    /// fresh events.  Confirms that `remove_worker` releases the SUB socket
    /// cleanly enough that a same-endpoint reconnect succeeds within the
    /// settle window, without leaking the previous task's state.
    ///
    /// Events published while the worker is detached are lost (ZMQ PUB/SUB
    /// is fire-and-forget; no replay). Downstream cursor recovery happens
    /// at the [`super::index::KvEventIndex`] layer, which clears the cursor
    /// on `remove_worker` so the re-added worker's seq=1 is not filtered.
    #[tokio::test]
    async fn restart_after_remove_picks_up_new_events() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let worker_url = "http://127.0.0.1:30000";
        let cfg = helpers::cfg_for(worker_url, port, 1);

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        // First incarnation: publish + drain.
        registry.add_worker(worker_url, &cfg).await;
        helpers::settle().await;
        let payload_a = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        pub_sock
            .send(helpers::build_multipart(1, payload_a))
            .await
            .unwrap();
        let event_a = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("recv before remove timed out")
            .expect("channel closed");
        let (_, seq_a, _) = helpers::expect_batch(event_a);
        assert_eq!(seq_a, 1);

        // Detach the subscriber while the publisher keeps going.
        registry.remove_worker(worker_url).await;

        // This batch is sent while no subscriber is attached; it must be
        // dropped (ZMQ PUB without a connected SUB is fire-and-forget) and
        // must not poison the next subscriber's view.
        let payload_b = helpers::encode_all_blocks_cleared_batch(2.0, Some(0));
        pub_sock
            .send(helpers::build_multipart(2, payload_b))
            .await
            .unwrap();
        // Verify rx really has nothing buffered.
        assert!(
            timeout(Duration::from_millis(100), rx.recv())
                .await
                .is_err(),
            "no event must arrive while the worker is detached",
        );

        // Re-attach the SAME worker at the SAME endpoint.
        registry.add_worker(worker_url, &cfg).await;
        helpers::settle().await;

        // Fresh event from the publisher → must surface on the new subscriber.
        let payload_c = helpers::encode_all_blocks_cleared_batch(3.0, Some(0));
        pub_sock
            .send(helpers::build_multipart(3, payload_c))
            .await
            .unwrap();
        let event_c = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("recv after re-add timed out")
            .expect("channel closed");
        let (worker_c, seq_c, _) = helpers::expect_batch(event_c);
        assert_eq!(seq_c, 3);
        assert_eq!(worker_c.url, worker_url);

        registry.shutdown().await;
    }
}
