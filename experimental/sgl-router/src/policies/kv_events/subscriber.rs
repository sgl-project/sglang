//! Per-worker, per-DP-rank ZMQ subscriber for SGLang's `ZmqEventPublisher`.
//!
//! This module owns the I/O plumbing between SGLang workers (which publish
//! KV-cache events on a PUB socket — see
//! `python/sglang/srt/disaggregation/kv_events.py`) and the in-memory hash
//! tree (T5). Each `(worker_url, dp_rank)` pair gets its own SUB socket on
//! its own tokio task, decodes msgpack batches via [`super::wire`], and
//! forwards [`WorkerEvent`]s to a shared mpsc channel.
//!
//! # Wire format (3-frame multipart)
//!
//! Frames published by SGLang:
//! 1. `topic_bytes` — empty by default, present even when empty.
//! 2. `seq_bytes` — 8-byte big-endian signed `i64`. Sentinel `-1` indicates
//!    publisher shutdown (`ZmqEventPublisher.END_SEQ`).
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
//! need an outer reconnect loop. Errors surfaced from `recv()` are logged
//! and the task continues — the underlying connection will recover on its
//! own. We exit only on cancellation, on a fatal `recv()` error that
//! cannot be retried, or when the downstream mpsc receiver is dropped.
//!
//! # Ordering
//!
//! Events for one `(worker, dp_rank)` flow through one task and use one
//! mpsc sender — order is preserved per-worker. Order across DP ranks (or
//! across workers) is **not** preserved; T5 must not depend on it.
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

use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, trace, warn};
use zeromq::{Socket, SocketRecv, SubSocket, ZmqMessage};

use super::discovery::EventConfig;
use super::tree::KvWorkerId;
use super::wire::{decode_event_batch, KvEventBatch};

/// Sentinel sequence number meaning "publisher is shutting down". Mirrors
/// `ZmqEventPublisher.END_SEQ = (-1).to_bytes(8, 'big', signed=True)`.
const END_SEQ_SENTINEL: i64 = -1;

/// One decoded event batch tagged with the worker that emitted it.
#[derive(Debug)]
pub struct WorkerEvent {
    /// Identity of the SGLang worker (DP rank) that produced this batch.
    pub worker: KvWorkerId,
    /// 8-byte big-endian sequence number from the publisher's monotonic
    /// counter. Useful for replay / gap detection downstream.
    pub seq: i64,
    /// Decoded batch payload.
    pub batch: KvEventBatch,
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
pub struct KvEventSubscriberRegistry {
    inner: Arc<Inner>,
}

impl KvEventSubscriberRegistry {
    /// Build an empty registry. `tx` is where decoded events flow out;
    /// the channel buffer capacity is the caller's choice.
    pub fn new(tx: mpsc::Sender<WorkerEvent>) -> Self {
        Self {
            inner: Arc::new(Inner {
                tx,
                handles: Mutex::new(HashMap::new()),
            }),
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
            let port = match u16::try_from(cfg.port_base as u32 + dp_rank) {
                Ok(p) => p,
                Err(_) => {
                    warn!(
                        worker_url = %worker_url,
                        dp_rank,
                        port_base = cfg.port_base,
                        "ZMQ event port overflows u16; skipping this rank"
                    );
                    continue;
                }
            };
            let endpoint = format!("tcp://{}:{}", cfg.host, port);
            let cancel = CancellationToken::new();
            let join =
                spawn_subscriber_task(id.clone(), endpoint, self.inner.tx.clone(), cancel.clone());
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
    tx: mpsc::Sender<WorkerEvent>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        run_subscriber(id, endpoint, tx, cancel).await;
    })
}

/// Inner subscriber loop. Returns when:
///   * the cancellation token fires, OR
///   * the downstream mpsc receiver is dropped, OR
///   * we cannot establish a SUB connection (error logged).
async fn run_subscriber(
    id: KvWorkerId,
    endpoint: String,
    tx: mpsc::Sender<WorkerEvent>,
    cancel: CancellationToken,
) {
    debug!(
        worker_url = %id.url,
        dp_rank = id.dp_rank,
        endpoint = %endpoint,
        "starting kv-event subscriber"
    );

    let mut sub = SubSocket::new();
    // `connect` for SubSocket installs its own background reconnect task,
    // so a single call suffices. Subscribe to all topics ("" matches the
    // SGLang default of an empty topic prefix).
    tokio::select! {
        _ = cancel.cancelled() => {
            debug!(worker_url = %id.url, dp_rank = id.dp_rank, "cancelled before connect");
            return;
        }
        res = sub.connect(&endpoint) => {
            if let Err(e) = res {
                warn!(
                    worker_url = %id.url,
                    dp_rank = id.dp_rank,
                    endpoint = %endpoint,
                    error = %e,
                    "failed to connect SUB socket"
                );
                return;
            }
        }
    }
    if let Err(e) = sub.subscribe("").await {
        warn!(
            worker_url = %id.url,
            dp_rank = id.dp_rank,
            error = %e,
            "failed to subscribe to topic ''"
        );
        return;
    }

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
                        if let Some(event) = decode_message(&id, msg) {
                            if tx.send(event).await.is_err() {
                                debug!(
                                    worker_url = %id.url,
                                    dp_rank = id.dp_rank,
                                    "downstream mpsc receiver dropped; exiting"
                                );
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        // SubSocket auto-reconnects internally; recv()
                        // should resume once a new peer attaches. We log
                        // and continue, but for safety we yield so we
                        // don't spin if the error is persistent.
                        warn!(
                            worker_url = %id.url,
                            dp_rank = id.dp_rank,
                            error = %e,
                            "recv error from SUB socket; continuing"
                        );
                        tokio::task::yield_now().await;
                    }
                }
            }
        }
    }
}

/// Validate, parse, and decode a single 3-frame multipart ZMQ message.
/// Returns `None` (with logging) for any non-event input (bad frame
/// count, sentinel sequence, or msgpack decode error).
fn decode_message(id: &KvWorkerId, msg: ZmqMessage) -> Option<WorkerEvent> {
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
        info!(
            worker_url = %id.url,
            dp_rank = id.dp_rank,
            "publisher signalled shutdown (END_SEQ); awaiting reconnect"
        );
        return None;
    }

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

    Some(WorkerEvent {
        worker: id.clone(),
        seq,
        batch,
    })
}

/// Pull the host out of a routing URL like `http://10.0.0.1:30000` or
/// `https://[::1]:30000`. Falls back to `None` for inputs the `url` crate
/// cannot parse — the caller logs and skips the worker.
///
/// Currently only used by the in-crate test helpers; kept at module scope
/// because future production callers (a non-discovery fallback path) will
/// need it.
#[cfg_attr(not(test), allow(dead_code))]
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
                block_size: 64,
                dp_size,
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

        /// Build a 3-frame multipart with topic="", the given seq (BE i64),
        /// and the given payload bytes.
        pub fn build_multipart(seq: i64, payload: Vec<u8>) -> ZmqMessage {
            let mut msg = ZmqMessage::from(Bytes::new());
            // ZmqMessage::from(Bytes::new()) creates a 1-frame message
            // with an empty frame, which is exactly the topic frame. Now
            // append seq and payload.
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
    }

    /// Single subscriber: publish one batch, see one batch.
    #[tokio::test]
    async fn single_subscriber_receives_one_event() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;

        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);

        registry.add_worker("http://127.0.0.1:30000", &helpers::cfg_for("http://127.0.0.1:30000", port, 1)).await;
        helpers::settle().await;

        let payload = helpers::encode_all_blocks_cleared_batch(1.0, Some(0));
        let msg = helpers::build_multipart(7, payload);
        pub_sock.send(msg).await.expect("send");

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("recv timed out")
            .expect("channel closed");

        assert_eq!(event.seq, 7);
        assert_eq!(event.worker.dp_rank, 0);
        assert_eq!(event.worker.url, "http://127.0.0.1:30000");
        assert_eq!(event.batch.events.len(), 1);
        assert!(matches!(
            event.batch.events[0],
            KvCacheEvent::AllBlocksCleared
        ));

        let shutdown_done = timeout(Duration::from_millis(500), registry.shutdown()).await;
        assert!(shutdown_done.is_ok(), "shutdown should return promptly");
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

        registry.add_worker(url0, &helpers::cfg_for(url0, p0, 1)).await;
        registry.add_worker(url1, &helpers::cfg_for(url1, p1, 1)).await;
        registry.add_worker(url2, &helpers::cfg_for(url2, p2, 1)).await;
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

        let mut by_url: HashMap<String, WorkerEvent> = HashMap::new();
        for _ in 0..3 {
            let event = timeout(Duration::from_millis(500), rx.recv())
                .await
                .expect("timed out")
                .expect("channel closed");
            by_url.insert(event.worker.url.clone(), event);
        }

        assert_eq!(by_url.len(), 3);
        assert_eq!(by_url[url0].seq, 10);
        assert_eq!(by_url[url1].seq, 20);
        assert_eq!(by_url[url2].seq, 30);

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
            .add_worker("http://127.0.0.1:30000", &helpers::cfg_for("http://127.0.0.1:30000", base_port, 3))
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
            assert_eq!(event.worker.url, "http://127.0.0.1:30000");
            by_rank.insert(event.worker.dp_rank, event.seq);
        }
        assert_eq!(by_rank.get(&0), Some(&100));
        assert_eq!(by_rank.get(&1), Some(&200));
        assert_eq!(by_rank.get(&2), Some(&300));

        registry.shutdown().await;
    }

    /// Bad msgpack payload is logged and dropped; subsequent valid event
    /// still arrives.
    #[tokio::test]
    async fn decoding_error_tolerated() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry.add_worker("http://127.0.0.1", &helpers::cfg_for("http://127.0.0.1", port, 1)).await;
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
        assert_eq!(event.seq, 2);
        // We must NOT have received the bad message.
        assert!(matches!(
            event.batch.events[0],
            KvCacheEvent::AllBlocksCleared
        ));

        registry.shutdown().await;
    }

    /// 2-frame and 4-frame messages are dropped; valid 3-frame still works.
    #[tokio::test]
    async fn wrong_frame_count_tolerated() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry.add_worker("http://127.0.0.1", &helpers::cfg_for("http://127.0.0.1", port, 1)).await;
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
        assert_eq!(event.seq, 42);

        registry.shutdown().await;
    }

    /// END_SEQ sentinel (-1) is logged and not forwarded; subsequent
    /// valid events still arrive.
    #[tokio::test]
    async fn sequence_number_sentinel_handled() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry.add_worker("http://127.0.0.1", &helpers::cfg_for("http://127.0.0.1", port, 1)).await;
        helpers::settle().await;

        // Sentinel: payload doesn't matter (we drop on seq=-1).
        pub_sock
            .send(helpers::build_multipart(-1, b"ignored".to_vec()))
            .await
            .unwrap();
        // Then a valid one with seq=5.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        pub_sock
            .send(helpers::build_multipart(5, payload))
            .await
            .unwrap();

        let event = timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert_eq!(event.seq, 5);

        registry.shutdown().await;
    }

    /// `remove_worker` cancels the task; further publishes are not
    /// received.
    #[tokio::test]
    async fn remove_worker_cancels() {
        let (mut pub_sock, port) = helpers::make_pub_bound().await;
        let (tx, mut rx) = mpsc::channel::<WorkerEvent>(8);
        let registry = KvEventSubscriberRegistry::new(tx);
        registry.add_worker("http://127.0.0.1:30000", &helpers::cfg_for("http://127.0.0.1:30000", port, 1)).await;
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

        registry.add_worker("http://127.0.0.1:30000", &helpers::cfg_for("http://127.0.0.1:30000", port, 1)).await;
        registry.add_worker("http://127.0.0.1:30000", &helpers::cfg_for("http://127.0.0.1:30000", port, 1)).await;

        {
            let handles = registry.inner.handles.lock().await;
            assert_eq!(handles.len(), 1, "expected 1 entry, got {}", handles.len());
        }

        registry.shutdown().await;
    }

    /// `cancel_all` signals every per-worker token without awaiting; a
    /// subsequent `shutdown` must still complete cleanly. This is the
    /// path `CacheAwareZmqPolicy::Drop` relies on so subscriber tasks
    /// don't leak when the policy is dropped without an explicit
    /// `shutdown()`.
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
        assert!(decode_message(&id, one_frame).is_none());

        // Sentinel seq = -1.
        let sentinel = helpers::build_multipart(-1, b"ignored".to_vec());
        assert!(decode_message(&id, sentinel).is_none());

        // Bad seq frame length.
        let mut bad_seq = ZmqMessage::from(Bytes::new());
        bad_seq.push_back(Bytes::from_static(b"abc")); // 3 bytes, not 8
        bad_seq.push_back(Bytes::from_static(b""));
        assert!(decode_message(&id, bad_seq).is_none());

        // Bad payload.
        let bad_payload = helpers::build_multipart(1, vec![0xff, 0xfe]);
        assert!(decode_message(&id, bad_payload).is_none());

        // Happy path.
        let payload = helpers::encode_all_blocks_cleared_batch(0.0, None);
        let good = helpers::build_multipart(7, payload);
        let event = decode_message(&id, good).expect("should decode");
        assert_eq!(event.seq, 7);
        assert_eq!(event.worker, id);
    }

    /// Restart-resume contract: after a worker is removed and then re-added
    /// to the same endpoint, the new subscriber must connect and forward
    /// fresh events.  Confirms that `remove_worker` releases the SUB socket
    /// cleanly enough that a same-endpoint reconnect succeeds within the
    /// settle window, without leaking the previous task's state.
    ///
    /// Events published while the worker is detached are lost (ZMQ PUB/SUB
    /// is fire-and-forget; no replay).  Downstream cursor recovery is the
    /// manager's job — wired up in M3 Task 6 / Task 7.
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
        assert_eq!(event_a.seq, 1);

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
            timeout(Duration::from_millis(100), rx.recv()).await.is_err(),
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
        assert_eq!(event_c.seq, 3);
        assert_eq!(event_c.worker.url, worker_url);

        registry.shutdown().await;
    }
}
