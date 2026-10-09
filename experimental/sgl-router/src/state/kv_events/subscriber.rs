//! Per-worker, per-DP-rank ZMQ subscriber for SGLang's `ZmqEventPublisher`.
//!
//! This module owns the I/O plumbing between SGLang workers (which publish on
//! a PUB socket via `python/sglang/srt/utils/event_publisher.py` —
//! KV-cache events from `disaggregation/kv_events.py`, load gauges from
//! `managers/scheduler_components/load_publisher.py`) and the in-memory state
//! consumed by [`super::index::KvEventIndex`]. Each `(worker_url, dp_rank)`
//! pair gets its own SUB socket on its own tokio task, decodes msgpack frames
//! by [`SubKind`] (KV batches via [`super::wire`], load via
//! [`crate::state::load_monitor::engine_reported_load`]), and forwards [`WorkerEvent`]s to a
//! shared mpsc channel.
//!
//! # Wire format (3-frame multipart)
//!
//! Frames published by SGLang:
//! 1. `topic_bytes` — empty by default, present even when empty.
//! 2. `seq_bytes` — 8-byte big-endian signed `i64`, dense per publisher. The
//!    `-1` sentinel (`ZmqEventPublisher.END_SEQ`) ends a replay; seen on the
//!    PUB stream it becomes a [`WorkerEvent::PublisherReset`].
//! 3. `payload` — msgpack-encoded [`KvEventBatch`].
//!
//! # Sequence repair
//!
//! A seq that regresses on one socket is a publisher restart and is
//! forwarded as a reset, except batch 0, which the pump already resolves from
//! the stream's origin. A forward gap is re-fetched from the publisher's
//! replay ROUTER when `/server_info` advertises one; live frames wait in the
//! SUB socket meanwhile, so the pump always sees this rank in order.
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
//! need an outer reconnect loop. The initial `connect` + `subscribe` retries
//! with capped backoff until it succeeds or is cancelled, and
//! [`RECV_ERROR_CEILING`] consecutive `recv()` errors rebuild the socket.
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

use anyhow::{anyhow, Context};
use bytes::Bytes;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

use super::discovery::EventConfig;
use super::index::STREAM_ORIGIN_SEQ;
use super::tally::{EventTally, ReplayOutcome};
use super::tree::KvWorkerId;
use super::wire::{decode_event_batch, KvEventBatch};
use crate::state::load_monitor::engine_reported_load::{decode_load_stat, LoadStat};

/// Consecutive `recv()` errors after which the socket is treated as dead
/// and rebuilt; ZMQ's own reconnect covers anything shorter.
const RECV_ERROR_CEILING: u32 = 64;

/// Connect attempts before a still-unreachable publisher is logged as an
/// error; retrying continues at [`CONNECT_BACKOFF_CAP`].
const CONNECT_MAX_ATTEMPTS: u32 = 5;
const CONNECT_BACKOFF_BASE: Duration = Duration::from_millis(50);
const CONNECT_BACKOFF_CAP: Duration = Duration::from_secs(2);

/// `ZmqEventPublisher.END_SEQ`: terminates a replay on the ROUTER socket.
/// Also accepted on the PUB stream, where it means the publisher reset.
const END_SEQ_SENTINEL: i64 = -1;

/// How long live batches may wait behind one gap replay; arbitrary.
const REPLAY_TIMEOUT: Duration = Duration::from_secs(2);

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
    /// counter from 0; the pump uses this to reset the cursor so those
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
    tally: Arc<EventTally>,
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
            tally: Arc::new(EventTally::new()),
        }
    }

    /// Book gap replays on `tally` instead of a private one.
    pub fn with_tally(mut self, tally: Arc<EventTally>) -> Self {
        self.tally = tally;
        self
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
        // socket + configured topic; Load uses its own advertised socket +
        // topic. Refuse an incomplete load descriptor rather than subscribe-all:
        // the load wire is a distinct contract and a future mixed-use socket
        // must not feed unrelated payloads into the load decoder.
        let (port_base, topic) = match self.kind {
            SubKind::Kv => (cfg.port_base, cfg.topic.clone()),
            SubKind::Load => match (&cfg.load_port_base, &cfg.load_topic) {
                (Some(port), Some(topic)) => (*port, topic.clone()),
                _ => {
                    debug!(
                        worker_url = %worker_url,
                        "kv-events: worker lacks a complete load descriptor; skipping load subscribers"
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
            let replay = cfg
                .replay_port_base
                .filter(|_| self.kind == SubKind::Kv)
                .and_then(|base| u16::try_from(base as u32 + dp_rank).ok())
                .map(|port| format!("tcp://{}:{}", cfg.host, port));
            let cancel = CancellationToken::new();
            let join = spawn_subscriber_task(
                id.clone(),
                Endpoints {
                    live: endpoint,
                    replay,
                },
                topic.clone(),
                self.kind,
                self.inner.tx.clone(),
                Arc::clone(&self.tally),
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

/// The live SUB endpoint and, if advertised, the replay ROUTER behind it.
struct Endpoints {
    live: String,
    replay: Option<String>,
}

/// Spawn the background task that owns one SUB socket and forwards
/// decoded batches.
fn spawn_subscriber_task(
    id: KvWorkerId,
    endpoints: Endpoints,
    topic: String,
    kind: SubKind,
    tx: mpsc::Sender<WorkerEvent>,
    tally: Arc<EventTally>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        run_subscriber(id, endpoints, topic, kind, tx, tally, cancel).await;
    })
}

/// Inner subscriber loop. Returns only when the cancellation token fires or
/// the downstream mpsc receiver is dropped.
async fn run_subscriber(
    id: KvWorkerId,
    endpoints: Endpoints,
    topic: String,
    kind: SubKind,
    tx: mpsc::Sender<WorkerEvent>,
    tally: Arc<EventTally>,
    cancel: CancellationToken,
) {
    let Endpoints {
        live: endpoint,
        replay,
    } = endpoints;
    debug!(
        worker_url = %id.url,
        dp_rank = id.dp_rank,
        endpoint = %endpoint,
        topic = %topic,
        kind = ?kind,
        "starting kv-event subscriber"
    );

    let Some(mut sub) = connect_with_backoff(&id, &endpoint, &topic, &cancel).await else {
        return;
    };

    let mut errors_in_a_row = 0u32;
    // Last seq forwarded from this socket's own stream, never a graft cursor.
    let mut last_seq: Option<i64> = None;
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
                        let events = match decode_message(&id, msg, kind) {
                            // A gap replay can take REPLAY_TIMEOUT; don't let
                            // it hold up remove_worker / shutdown.
                            Some(event) => tokio::select! {
                                biased;
                                _ = cancel.cancelled() => return,
                                events = sequence(
                                    &id,
                                    event,
                                    &mut last_seq,
                                    replay.as_deref(),
                                    &tally,
                                ) => events,
                            },
                            None => Vec::new(),
                        };
                        for event in events {
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
                                "SUB socket has produced {RECV_ERROR_CEILING} consecutive recv errors; reconnecting"
                            );
                            let Some(fresh) =
                                connect_with_backoff(&id, &endpoint, &topic, &cancel).await
                            else {
                                return;
                            };
                            sub = fresh;
                            errors_in_a_row = 0;
                            continue;
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
/// Retries with capped exponential backoff until it succeeds, so a publisher
/// that binds late never disables its worker's cache-aware routing. Returns
/// `None` only when cancelled.
async fn connect_with_backoff(
    id: &KvWorkerId,
    endpoint: &str,
    topic: &str,
    cancel: &CancellationToken,
) -> Option<SubSocket> {
    let mut delay = CONNECT_BACKOFF_BASE;
    let mut attempt = 0u32;
    loop {
        attempt += 1;
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
            error!(
                worker_url = %id.url,
                dp_rank = id.dp_rank,
                endpoint = %endpoint,
                "kv-events: SUB socket still unreachable after {CONNECT_MAX_ATTEMPTS} attempts; \
                 cache-aware routing for this worker waits until it connects"
            );
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
}

/// What to forward for `event`. One PUB stream never goes backwards, so a
/// regressed seq is a publisher restart and becomes a reset (batch 0 is left
/// to the pump); a forward gap
/// is filled from the replay socket when one is advertised.
async fn sequence(
    id: &KvWorkerId,
    event: WorkerEvent,
    last_seq: &mut Option<i64>,
    replay: Option<&str>,
    tally: &EventTally,
) -> Vec<WorkerEvent> {
    let mut out = Vec::new();
    match &event {
        WorkerEvent::Batch { seq, .. } => {
            match *last_seq {
                // A restart from batch 0 is left to the pump, which resolves
                // it from the stream's origin (keeping a bootstrapped rank
                // warm) instead of failing the rank as a reset would.
                Some(last) if *seq <= last && *seq != STREAM_ORIGIN_SEQ => {
                    warn!(
                        worker = ?id,
                        last,
                        seq = *seq,
                        "kv-events: sequence regressed; publisher restarted, resetting this rank",
                    );
                    out.push(WorkerEvent::PublisherReset { worker: id.clone() });
                }
                Some(last) if *seq > last + 1 => {
                    if let Some(endpoint) = replay {
                        out = fill_gap(id, endpoint, last + 1, *seq, tally).await;
                    }
                }
                _ => {}
            }
            *last_seq = Some(*seq);
        }
        WorkerEvent::PublisherReset { .. } => *last_seq = None,
        WorkerEvent::Load { .. } => {}
    }
    out.push(event);
    out
}

/// Batches `from..to` re-fetched from the replay socket. Whatever comes back
/// is forwarded; the pump counts anything still missing as lost.
async fn fill_gap(
    id: &KvWorkerId,
    endpoint: &str,
    from: i64,
    to: i64,
    tally: &EventTally,
) -> Vec<WorkerEvent> {
    // Keep received batches outside the timed future so cancellation or a
    // later socket/decode error cannot discard an already recovered removal.
    let mut batches = Vec::new();
    let completed = match tokio::time::timeout(
        REPLAY_TIMEOUT,
        fetch_replay(endpoint, from, to, &mut batches),
    )
    .await
    {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            warn!(worker = ?id, from, to, error = %e, "kv-events: gap replay failed");
            false
        }
        Err(_) => {
            warn!(worker = ?id, from, to, "kv-events: gap replay timed out");
            false
        }
    };
    let outcome = if batches.len() as i64 == to - from {
        ReplayOutcome::Repaired
    } else if !completed && batches.is_empty() {
        ReplayOutcome::Failed
    } else {
        warn!(
            worker = ?id,
            from,
            to,
            recovered = batches.len(),
            "kv-events: replay did not cover the gap",
        );
        ReplayOutcome::Incomplete
    };
    tally.record_replay(outcome);
    batches
        .into_iter()
        .map(|(seq, batch)| WorkerEvent::Batch {
            worker: id.clone(),
            seq,
            batch,
        })
        .collect()
}

/// Ask the publisher's ROUTER for buffered batches from `from` on, retaining
/// only `from..to`. The publisher replies in sequence order, so stop once the
/// gap's end is reached without waiting for newer batches or their END_SEQ.
/// Wire contract: `ZmqEventPublisher._service_replay` in SGLang's
/// `disaggregation/kv_events.py`; replies carry `[b"", topic, seq, payload]`
/// up to `END_SEQ`. Legacy publishers omit the topic.
async fn fetch_replay(
    endpoint: &str,
    from: i64,
    to: i64,
    batches: &mut Vec<(i64, KvEventBatch)>,
) -> anyhow::Result<()> {
    let mut dealer = DealerSocket::new();
    dealer.connect(endpoint).await?;
    let mut request = ZmqMessage::from(Bytes::new());
    request.push_back(Bytes::copy_from_slice(&from.to_be_bytes()));
    dealer.send(request).await?;
    loop {
        let reply = dealer.recv().await?;
        let topic_offset = usize::from(reply.len() == 4);
        let (3 | 4, Some(delim), Some(seq), Some(payload)) = (
            reply.len(),
            reply.get(0),
            reply.get(1 + topic_offset),
            reply.get(2 + topic_offset),
        ) else {
            return Err(anyhow!("replay reply has {} frames", reply.len()));
        };
        if !delim.is_empty() {
            return Err(anyhow!("replay reply lacks the empty delimiter frame"));
        }
        let seq = i64::from_be_bytes(seq.as_ref().try_into().context("replay seq frame")?);
        if seq == END_SEQ_SENTINEL || seq >= to {
            return Ok(());
        }
        // Kept strictly increasing, so a full count in fill_gap means no hole.
        if seq >= from && batches.last().is_none_or(|&(last, _)| seq > last) {
            batches.push((seq, decode_event_batch(payload.as_ref())?));
        }
        if seq == to - 1 {
            return Ok(());
        }
    }
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

    use crate::state::kv_events::wire::KvCacheEvent;

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
                load_topic: None,
                replay_port_base: None,
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
            // Events use msgspec's tagged-map encoding: {"type": "AllBlocksCleared"}.
            mp::write_map_len(&mut buf, 1).unwrap();
            mp::write_str(&mut buf, "type").unwrap();
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

        /// Allows the local PUB/SUB handshake to complete in concurrent tests.
        pub async fn settle() {
            tokio::time::sleep(Duration::from_millis(250)).await;
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

    /// The #34608 load stream has its own advertised topic. It must use that
    /// filter too: accepting every frame on the socket would make a future
    /// colocated publisher influence routing through a coincidentally
    /// decodable payload.
    #[tokio::test]
    async fn load_subscriber_filters_by_advertised_topic() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::with_kind(tx, SubKind::Load);

        let worker_url = "http://127.0.0.1:30101";
        let mut cfg = helpers::cfg_for(worker_url, port, 1);
        cfg.load_port_base = Some(port);
        cfg.load_topic = Some("load".into());
        registry.add_worker(worker_url, &cfg).await;
        helpers::settle().await;

        let payload = helpers::encode_load_stat(5, 2, 100, 1000, 0);
        pub_sock
            .send(helpers::build_multipart_with_topic(b"load", 3, payload))
            .await
            .unwrap();
        let other_payload = helpers::encode_load_stat(99, 0, 0, 0, 0);
        pub_sock
            .send(helpers::build_multipart_with_topic(
                b"other",
                4,
                other_payload,
            ))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out waiting for load event")
            .expect("channel closed");
        match event {
            WorkerEvent::Load { load, .. } => assert_eq!(load.num_running_reqs, 5),
            other => panic!("expected Load, got {other:?}"),
        }
        assert!(
            timeout(Duration::from_millis(200), rx.recv())
                .await
                .is_err(),
            "unmatched load topic must not reach the subscriber"
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
            assert!(attempt < 256, "could not find 3 contiguous free ports");

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

        let mut by_rank: HashMap<u32, i64> = HashMap::new();
        for _ in 0..20 {
            for (rank, pubsock) in publishers.iter_mut().enumerate() {
                pubsock
                    .send(helpers::build_multipart(
                        1000 + rank as i64,
                        helpers::encode_all_blocks_cleared_batch(rank as f64, Some(rank as u32)),
                    ))
                    .await
                    .unwrap();
            }

            for _ in 0..N {
                let Ok(Some(event)) = timeout(Duration::from_millis(50), rx.recv()).await else {
                    break;
                };
                let (worker, seq, _batch) = helpers::expect_batch(event);
                assert_eq!(worker.url, worker_url);
                by_rank.insert(worker.dp_rank, seq);
            }
            if by_rank.len() == N {
                break;
            }
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

    /// A gap on the live stream is filled from the replay ROUTER, in order.
    #[tokio::test]
    async fn gap_is_filled_from_replay_socket() {
        use zeromq::RouterSocket;

        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let mut router = RouterSocket::new();
        let Endpoint::Tcp(_, replay_port) = router.bind("tcp://127.0.0.1:0").await.unwrap() else {
            panic!("tcp endpoint");
        };
        let server = tokio::spawn(async move {
            let request = router.recv().await.unwrap();
            let peer = request.get(0).unwrap().clone();
            assert_eq!(request.get(2).unwrap().as_ref(), 2i64.to_be_bytes());
            for (seq, payload) in [
                (2, helpers::encode_all_blocks_cleared_batch(0.0, None)),
                (3, helpers::encode_all_blocks_cleared_batch(0.0, None)),
                (-1, Vec::new()),
            ] {
                let mut reply = ZmqMessage::from(peer.clone());
                reply.push_back(Bytes::new());
                reply.push_back(Bytes::copy_from_slice(&i64::to_be_bytes(seq)));
                reply.push_back(Bytes::from(payload));
                router.send(reply).await.unwrap();
            }
        });
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let tally = Arc::new(EventTally::new());
        let registry = KvEventSubscriberRegistry::new(tx).with_tally(Arc::clone(&tally));
        let cfg = EventConfig {
            replay_port_base: Some(replay_port),
            ..helpers::cfg_for("http://127.0.0.1", port, 1)
        };
        registry.add_worker("http://127.0.0.1", &cfg).await;
        helpers::settle().await;

        for seq in [1, 4] {
            let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
            pub_sock
                .send(helpers::build_multipart(seq, payload))
                .await
                .unwrap();
        }
        let mut seqs = Vec::new();
        for _ in 0..4 {
            let ev = timeout(Duration::from_secs(2), rx.recv())
                .await
                .unwrap()
                .unwrap();
            seqs.push(helpers::expect_batch(ev).1);
        }
        assert_eq!(seqs, [1, 2, 3, 4]);
        assert_eq!(tally.replays(ReplayOutcome::Repaired), 1);
        server.await.unwrap();
        registry.shutdown().await;
    }

    #[tokio::test]
    async fn replay_accepts_legacy_and_topic_frames() {
        use zeromq::RouterSocket;

        for include_topic in [false, true] {
            let mut router = RouterSocket::new();
            let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap().to_string();
            let server = tokio::spawn(async move {
                let request = router.recv().await.unwrap();
                let peer = request.get(0).unwrap().clone();
                for seq in [1_i64, END_SEQ_SENTINEL] {
                    let mut reply = ZmqMessage::from(peer.clone());
                    reply.push_back(Bytes::new());
                    if include_topic {
                        reply.push_back(if seq == END_SEQ_SENTINEL {
                            Bytes::new()
                        } else {
                            Bytes::from_static(b"kv@prefill@model")
                        });
                    }
                    reply.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
                    reply.push_back(if seq == END_SEQ_SENTINEL {
                        Bytes::new()
                    } else {
                        Bytes::from(helpers::encode_all_blocks_cleared_batch(0.0, None))
                    });
                    router.send(reply).await.unwrap();
                }
            });
            let mut batches = Vec::new();
            timeout(
                Duration::from_secs(2),
                fetch_replay(&endpoint, 1, 3, &mut batches),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(batches.len(), 1, "include_topic={include_topic}");
            assert_eq!(batches[0].0, 1);
            server.await.unwrap();
        }
    }

    /// Replay can time out or fail after useful batches have already arrived.
    #[tokio::test]
    async fn replay_preserves_batches_before_timeout_or_decode_error() {
        use zeromq::RouterSocket;

        for (seqs, malformed_tail, expected) in [
            (vec![2], false, ReplayOutcome::Incomplete),
            (vec![2], true, ReplayOutcome::Incomplete),
            (vec![], true, ReplayOutcome::Failed),
            (vec![2, 3], false, ReplayOutcome::Repaired),
        ] {
            let mut router = RouterSocket::new();
            let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap().to_string();
            let sent = seqs.clone();
            let server = tokio::spawn(async move {
                let request = router.recv().await.unwrap();
                let peer = request.get(0).unwrap().clone();
                for seq in sent {
                    let mut reply = ZmqMessage::from(peer.clone());
                    reply.push_back(Bytes::new());
                    reply.push_back(Bytes::copy_from_slice(&i64::to_be_bytes(seq)));
                    reply.push_back(Bytes::from(helpers::encode_all_blocks_cleared_batch(
                        0.0, None,
                    )));
                    router.send(reply).await.unwrap();
                }
                if malformed_tail {
                    let mut reply = ZmqMessage::from(peer);
                    reply.push_back(Bytes::new());
                    reply.push_back(Bytes::copy_from_slice(&3i64.to_be_bytes()));
                    reply.push_back(Bytes::from_static(&[0xc1])); // Invalid msgpack.
                    router.send(reply).await.unwrap();
                }
                // Keep the socket open without END_SEQ. A complete gap must
                // finish immediately; a partial one must survive the timeout.
                std::future::pending::<()>().await;
            });
            let id = KvWorkerId::new("http://worker".into(), 0);
            let tally = EventTally::new();
            let deadline = if expected == ReplayOutcome::Repaired {
                REPLAY_TIMEOUT / 2
            } else {
                REPLAY_TIMEOUT * 2
            };
            let result = timeout(deadline, fill_gap(&id, &endpoint, 2, 4, &tally)).await;
            server.abort();
            let _ = server.await;
            let recovered: Vec<_> = result
                .expect("replay must finish without waiting for an unrelated tail")
                .into_iter()
                .map(|event| helpers::expect_batch(event).1)
                .collect();
            assert_eq!(recovered, seqs, "malformed_tail={malformed_tail}");
            for outcome in ReplayOutcome::ALL {
                assert_eq!(tally.replays(outcome), u64::from(outcome == expected));
            }
        }
    }

    #[tokio::test]
    async fn sequence_resets_on_regression_and_passes_gaps_without_replay() {
        let id = KvWorkerId::new("http://w".into(), 0);
        let tally = EventTally::new();
        let batch = |seq| WorkerEvent::Batch {
            worker: id.clone(),
            seq,
            batch: decode_event_batch(&helpers::encode_all_blocks_cleared_batch(0.0, None))
                .unwrap(),
        };
        let mut last = None;
        let mut kinds = Vec::new();
        // A regression to batch 0 passes through without a reset: the pump
        // resolves that one from the stream's origin.
        for seq in [5, 9, 2, 0] {
            for ev in sequence(&id, batch(seq), &mut last, None, &tally).await {
                kinds.push(match ev {
                    WorkerEvent::Batch { seq, .. } => seq,
                    _ => -1,
                });
            }
        }
        assert_eq!(kinds, [5, 9, -1, 2, 0]);
        assert_eq!(tally.replays(ReplayOutcome::Failed), 0);
    }
}
