//! api-server-side TCP client: the counterpart to [`serve_headless`]. The
//! standalone api-server process holds one [`NetClient`] that owns, per DP rank,
//! a pool of `POOL_CONNS` **ingress** connections (requests out, round-robined)
//! and `POOL_CONNS` **egress** connections (frames in, sharded by `rid`).
//!
//! Lock-free hot path:
//!   * each ingress connection is driven by its own writer task fed by an mpsc —
//!     `submit` just enqueues, no shared write lock;
//!   * each egress connection's reader task **owns** its `rid → handler` map
//!     (single owner, no `Mutex`); registrations arrive over an unbounded mpsc
//!     and are drained right before routing each frame.
//!
//! Because egress is sharded by `rid % POOL_CONNS` (matching the rank side), a
//! request always registers on, and is answered by, the same shard — so one
//! reader's map is sufficient and one request's ordered frames ride one socket.
//!
//! [`serve_headless`]: super::serve_headless

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::OnceCell;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

use crate::ids::RequestId;
use crate::message::{EgressItem, RequestKind};
use crate::transport::{Frame, ROLE_EGRESS, ROLE_INGRESS, handshake, read_frame, write_frame};

/// Bound on an ingress connection's pending-write queue (backpressure: a slow
/// socket makes `submit` await rather than buffer unboundedly).
const INGRESS_QUEUE: usize = 256;

/// `submit` → egress-shard reader: register the handler channel for a `rid`.
struct Register {
    rid: u64,
    tx: mpsc::Sender<EgressItem>,
}

/// One DP rank's connection pool.
//
// TODO(headless: connection lifecycle / reconnect). The pool is established once
// in `connect()` and never repaired. We need to:
//   1. Detect a dropped connection. The driver tasks already observe it (a
//      writer's `write_frame` errors, a reader's `read_frame` returns Err) and
//      exit — but they do so silently; nothing reclaims the slot. Surface the
//      drop (e.g. the task signals the owning `RankConn` with its index/shard).
//   2. Remove the stale slot so `submit` stops routing to it. Today a dead
//      `ingress[k]` makes ~1/ingress_pool of writes fail (the writer's rx is
//      gone → `send` errors → `submit` returns Err), and a dead egress shard
//      drops that shard's `rid -> handler` map, orphaning its in-flight requests
//      (their handler `rx` never completes → the HTTP handler hangs until its
//      own timeout). Both need active eviction, and in-flight requests on a lost
//      egress shard should be failed fast, not left to hang.
//   3. Re-establish + reinsert: reconnect (with backoff), redo the handshake for
//      the same role/index, respawn the driver, and swap it back into the slot —
//      so the pool self-heals. The slots would become `Vec<Mutex<...>>` or an
//      arc-swap, or a small supervisor task per rank owning the pool. (The rank
//      side must cooperate for egress: it currently `take()`s a shard's rx once,
//      so a reconnecting egress connection is rejected — see the matching
//      TODO(headless) in `server.rs`.)
struct RankConn {
    /// Ingress writer queues (one per pooled connection); `submit` round-robins.
    ingress: Vec<mpsc::Sender<(Frame, Vec<u8>)>>,
    /// Per-egress-shard registration channels (unbounded: registering must never
    /// block, and entries are tiny + bounded by outstanding requests).
    reg: Vec<mpsc::UnboundedSender<Register>>,
    /// Round-robin cursor across this rank's ingress connections.
    ingress_rr: AtomicUsize,
}

pub struct NetClient {
    ranks: Vec<RankConn>,
    /// Round-robin cursor across ranks (Level-0 routing; load-aware later).
    next: AtomicUsize,
}

impl NetClient {
    /// Connect to every rank endpoint, opening the pool per rank (egress readers
    /// first, so they're ready before any request) and spawning each connection's
    /// driver task. Pools are asymmetric: `ingress_pool` write connections and
    /// `egress_pool` read/shard connections. `egress_pool` must match the rank's
    /// `ServerArgs::egress_pool_size`; `ingress_pool` is the api-server's choice.
    /// Fails if any endpoint is unreachable.
    pub async fn connect(
        endpoints: &[SocketAddr],
        ingress_pool: usize,
        egress_pool: usize,
    ) -> std::io::Result<Arc<Self>> {
        let mut ranks = Vec::with_capacity(endpoints.len());
        for &ep in endpoints {
            // Egress shards first: reader tasks must be live before requests flow.
            let mut reg = Vec::with_capacity(egress_pool);
            for shard in 0..egress_pool {
                let (reg_tx, reg_rx) = mpsc::unbounded_channel::<Register>();
                let mut ec = connect_with_retry(ep).await?;
                ec.set_nodelay(true)?;
                ec.write_all(&handshake(ROLE_EGRESS, shard)).await?;
                let (er, _w) = ec.into_split();
                tokio::spawn(egress_reader(er, reg_rx));
                reg.push(reg_tx);
            }
            // Ingress connections: each gets a writer task draining its queue.
            let mut ingress = Vec::with_capacity(ingress_pool);
            for idx in 0..ingress_pool {
                let (fr_tx, fr_rx) = mpsc::channel::<(Frame, Vec<u8>)>(INGRESS_QUEUE);
                let mut ic = connect_with_retry(ep).await?;
                ic.set_nodelay(true)?;
                ic.write_all(&handshake(ROLE_INGRESS, idx)).await?;
                let (_r, iw) = ic.into_split();
                tokio::spawn(ingress_writer(iw, fr_rx));
                ingress.push(fr_tx);
            }
            ranks.push(RankConn {
                ingress,
                reg,
                ingress_rr: AtomicUsize::new(0),
            });
        }
        Ok(Arc::new(Self {
            ranks,
            next: AtomicUsize::new(0),
        }))
    }

    /// Submit a request and return the channel its egress frames will arrive on.
    /// `id` is assigned by the caller and carried end-to-end.
    pub async fn submit(
        &self,
        id: RequestId,
        kind: RequestKind,
        egress_buf: usize,
    ) -> Result<mpsc::Receiver<EgressItem>, ()> {
        let (tx, rx) = mpsc::channel::<EgressItem>(egress_buf);

        // Level-0 routing: round-robin across ranks.
        let rank = &self.ranks[self.next.fetch_add(1, Ordering::Relaxed) % self.ranks.len()];

        // Register on this rank's egress shard for the rid *before* sending the
        // request, so the answer is never raced (unbounded → never blocks).
        let shard = (id.0 as usize) % rank.reg.len();
        if rank.reg[shard].send(Register { rid: id.0, tx }).is_err() {
            return Err(());
        }

        // Round-robin the write across the rank's ingress connections.
        let (frame, tail) = encode_request(id.0, kind);
        let conn = rank.ingress_rr.fetch_add(1, Ordering::Relaxed) % rank.ingress.len();
        if rank.ingress[conn].send((frame, tail)).await.is_err() {
            return Err(());
        }
        Ok(rx)
    }

    /// Submit a control request to **every** rank (not load-balanced) and return
    /// the channel the first response arrives on. Control/info queries (e.g.
    /// `/server_info`) are answered per-rank, so round-robin can land on a rank
    /// whose answer never comes back, hanging the caller. Broadcasting with one
    /// shared `rid` makes the first rank to respond win; the rest route the same
    /// `rid` (already removed on arrival) and are dropped.
    pub async fn submit_broadcast(
        &self,
        id: RequestId,
        kind: RequestKind,
        egress_buf: usize,
    ) -> Result<mpsc::Receiver<EgressItem>, ()> {
        let (tx, rx) = mpsc::channel::<EgressItem>(egress_buf);
        let (frame, tail) = encode_request(id.0, kind);
        let mut delivered = false;
        for rank in &self.ranks {
            let shard = (id.0 as usize) % rank.reg.len();
            // Register the rid on this rank's shard before sending so the answer
            // isn't raced.
            if rank.reg[shard]
                .send(Register {
                    rid: id.0,
                    tx: tx.clone(),
                })
                .is_err()
            {
                continue; // this rank's egress reader is gone; try the others
            }
            let conn = rank.ingress_rr.fetch_add(1, Ordering::Relaxed) % rank.ingress.len();
            if rank.ingress[conn]
                .send((frame.clone(), tail.clone()))
                .await
                .is_ok()
            {
                delivered = true;
            }
        }
        // Ok as long as at least one rank received the request.
        if delivered { Ok(rx) } else { Err(()) }
    }
}

/// Readiness wrapper for the standalone api-server (Mode A registration).
///
/// The HTTP server binds and serves `/internal/register` immediately, but the
/// TCP pool to the DP ranks is **deferred** until every rank has reported its
/// headless endpoint (`POST /internal/register`). The api-server doesn't know
/// the rank addresses up front — ranks self-report (so multi-node node IPs need
/// no out-of-band discovery, and the api-server is relocatable as one unit).
/// Until the pool is up, [`client`](Self::client) returns `None` and request
/// handlers answer 503.
pub struct NetReady {
    dp_size: usize,
    ingress_pool: usize,
    egress_pool: usize,
    /// `dp_rank → reported headless endpoint`, filled by `/internal/register`.
    /// Std mutex: tiny, non-async critical sections (insert / snapshot).
    pending: StdMutex<HashMap<usize, SocketAddr>>,
    /// Guards the one-shot connect: set by the registration that completes the
    /// set, reset on connect failure so a later (re)registration can retry.
    connecting: AtomicBool,
    /// The pool, established once all `dp_size` ranks register. Lock-free reads
    /// on the request hot path once set.
    client: OnceCell<Arc<NetClient>>,
}

impl NetReady {
    /// New, empty readiness state. Pool sizes are the api-server's `ServerArgs`
    /// values (`egress_pool` must match the ranks' `egress_pool_size`).
    pub fn new(dp_size: usize, ingress_pool: usize, egress_pool: usize) -> Arc<Self> {
        Arc::new(Self {
            dp_size,
            ingress_pool,
            egress_pool,
            pending: StdMutex::new(HashMap::new()),
            connecting: AtomicBool::new(false),
            client: OnceCell::new(),
        })
    }

    /// The connected pool, or `None` until every DP rank has registered.
    pub fn client(&self) -> Option<Arc<NetClient>> {
        self.client.get().cloned()
    }

    /// Record one DP rank's headless endpoint. When the final rank registers,
    /// spawn the (one-shot) pool connect in the background so the registering
    /// request returns promptly. A duplicate `dp_rank` overwrites (a rank that
    /// restarted/re-registered before the pool came up).
    pub fn register(self: &Arc<Self>, dp_rank: usize, addr: SocketAddr) {
        let complete = {
            let mut p = self.pending.lock().unwrap();
            p.insert(dp_rank, addr);
            p.len() == self.dp_size
        };
        // Only the registration that completes the set, and only once, connects.
        if complete && !self.connecting.swap(true, Ordering::SeqCst) {
            let this = Arc::clone(self);
            tokio::spawn(this.connect_all());
        }
    }

    /// Establish the pool to all ranks (ordered by `dp_rank`) and publish it.
    async fn connect_all(self: Arc<Self>) {
        let endpoints: Vec<SocketAddr> = {
            let p = self.pending.lock().unwrap();
            (0..self.dp_size)
                .filter_map(|i| p.get(&i).copied())
                .collect()
        };
        // Every dp_rank in [0, dp_size) must be present (set just reached size).
        if endpoints.len() != self.dp_size {
            tracing::error!(
                got = endpoints.len(),
                want = self.dp_size,
                "register: dp_rank gap; deferring connect"
            );
            self.connecting.store(false, Ordering::SeqCst);
            return;
        }
        match NetClient::connect(&endpoints, self.ingress_pool, self.egress_pool).await {
            Ok(c) => {
                let _ = self.client.set(c);
                tracing::info!(?endpoints, "standalone api-server: rank pool connected");
            }
            Err(e) => {
                tracing::error!(error = %e, "standalone api-server: pool connect failed; retrying on next registration");
                self.connecting.store(false, Ordering::SeqCst);
            }
        }
    }
}

/// Connect to a rank endpoint, retrying briefly. A rank reports its endpoint
/// right after constructing its `Server`, whose TCP listener may bind a hair
/// later, and remote ranks can lag at startup — so a first refusal is expected.
/// Bounded (~5s) so a genuinely dead endpoint still surfaces as an error.
async fn connect_with_retry(ep: SocketAddr) -> std::io::Result<TcpStream> {
    const ATTEMPTS: usize = 50;
    const DELAY: Duration = Duration::from_millis(100);
    let mut last: Option<std::io::Error> = None;
    for _ in 0..ATTEMPTS {
        match TcpStream::connect(ep).await {
            Ok(s) => return Ok(s),
            Err(e) => {
                last = Some(e);
                tokio::time::sleep(DELAY).await;
            }
        }
    }
    Err(last.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::TimedOut, "connect retries exhausted")
    }))
}

/// Drive one ingress connection: drain its queue and write each frame. Exits on
/// write error (connection gone).
async fn ingress_writer(mut write: OwnedWriteHalf, mut rx: mpsc::Receiver<(Frame, Vec<u8>)>) {
    while let Some((frame, tail)) = rx.recv().await {
        if let Err(e) = write_frame(&mut write, &frame, &tail).await {
            tracing::warn!(error = %e, "net: ingress write error; closing");
            // TODO(headless: reconnect) — exiting here leaves a dead `ingress`
            // slot in the pool; signal the `RankConn` to evict + reconnect it.
            break;
        }
    }
}

/// Drive one egress shard connection: owns the `rid → handler` map for its shard,
/// routes each frame, and removes the entry on terminal. Registrations are
/// drained right before routing — a request always registers before its ingress
/// frame is written, so by the time its egress returns (pipeline + round trips)
/// the registration is queued here. Exits when the connection closes.
async fn egress_reader(mut read: OwnedReadHalf, mut reg_rx: mpsc::UnboundedReceiver<Register>) {
    let mut map: HashMap<u64, mpsc::Sender<EgressItem>> = HashMap::new();
    loop {
        let (frame, _tail) = match read_frame(&mut read).await {
            Ok(f) => f,
            // TODO(headless: reconnect) — on disconnect we drop this shard's
            // `map`, orphaning its in-flight requests (handlers hang to timeout).
            // Should fail those `rid`s fast, then evict + reconnect this shard.
            Err(_) => break, // connection closed → rank gone
        };
        while let Ok(reg) = reg_rx.try_recv() {
            map.insert(reg.rid, reg.tx);
        }
        match frame {
            Frame::Egress { rid, item } => {
                let item: EgressItem = item.into();
                // Frame is an intermediate streamed step; everything else ends it.
                let terminal = !matches!(item, EgressItem::Frame(_));
                if let Some(tx) = map.get(&rid) {
                    match tx.try_send(item) {
                        Ok(()) if terminal => {
                            map.remove(&rid);
                        }
                        Ok(()) => {}
                        // Handler gone (client disconnected): stop tracking.
                        Err(TrySendError::Closed(_)) => {
                            map.remove(&rid);
                        }
                        // Handler backpressured: drop this frame, keep the entry.
                        // TODO(headless): propagate backpressure / abort upstream.
                        Err(TrySendError::Full(_)) => {}
                    }
                }
            }
            // Load telemetry feeds the router (Level 1) — ignored until then.
            Frame::Load(_) => {}
            _ => {}
        }
    }
}

/// Serialize a [`RequestKind`] into a wire [`Frame`] plus its raw tail (the
/// request's `input_ids` as i32-LE for a pre-tokenized generate, else empty).
fn encode_request(rid: u64, kind: RequestKind) -> (Frame, Vec<u8>) {
    match kind {
        RequestKind::Generate(g) => {
            // NOTE: `payload.extra` (passthrough body fields) is not forwarded
            // yet; the OpenAI handlers don't populate it. TODO(headless): carry it.
            let p = g.payload;
            let tail = p
                .input_ids
                .map(|ids| {
                    ids.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>()
                })
                .unwrap_or_default();
            let frame = Frame::Generate {
                rid,
                text: p.text,
                sampling_params: p.sampling_params,
                stream: p.stream,
            };
            (frame, tail)
        }
        RequestKind::Control(c) => (
            Frame::Control {
                rid,
                tag: c.tag.to_string(),
            },
            Vec::new(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ControlRequest, GeneratePayload, GenerateRequest};
    use crate::runtime::channels::{Senders, TmEvent};
    use crate::transport::server;

    // Spawn a 2-egress-shard headless rank with a stub pipeline; `answer_control`
    // decides whether it responds to control requests. Returns its address.
    async fn spawn_rank(answer_control: bool) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tm_tx, tm_rx) = flume::bounded::<TmEvent>(16);
        let (tok_tx, _tok_rx) = flume::bounded(1);
        let senders = Senders {
            tm: tm_tx,
            tok: tok_tx,
            detok: vec![],
        };
        tokio::spawn(server::serve_on(listener, senders, 16, 2));
        tokio::spawn(async move {
            while let Ok(TmEvent::Ingress(req)) = tm_rx.recv_async().await {
                match &req.kind {
                    RequestKind::Generate(_) => {
                        let out = crate::message::GenerationOutput {
                            rid: req.id.0.to_string(),
                            finish_reason: Some("stop".into()),
                            ..Default::default()
                        };
                        let _ = req.sink.try_send(EgressItem::Done(out));
                    }
                    // A rank that "doesn't answer" silently drops control reqs —
                    // round-robin onto it would hang the caller.
                    RequestKind::Control(_) if answer_control => {
                        let _ = req
                            .sink
                            .try_send(EgressItem::Control(bytes::Bytes::from_static(b"ok")));
                    }
                    RequestKind::Control(_) => {}
                }
            }
        });
        addr
    }

    // The /server_info bug: a control request round-robined to a rank that never
    // answers hangs. Broadcast must still return the one rank that does answer.
    #[tokio::test]
    async fn control_broadcast_first_response_wins() {
        let a0 = spawn_rank(false).await; // never answers control
        let a1 = spawn_rank(true).await; // answers control
        let client = NetClient::connect(&[a0, a1], 4, 2).await.unwrap();
        let kind = RequestKind::Control(ControlRequest {
            tag: "GetInternalStateReq",
        });
        let mut rx = client
            .submit_broadcast(RequestId(42), kind, 16)
            .await
            .unwrap();
        let item = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("control broadcast timed out (would have hung on the silent rank)")
            .expect("channel closed");
        match item {
            EgressItem::Control(b) => assert_eq!(&b[..], b"ok"),
            other => panic!("expected Control, got {other:?}"),
        }
    }

    // End-to-end over real TCP: NetClient ↔ serve_on (full pool), with a stub
    // "pipeline" that drains the tm inbox and answers each request via its `Net`
    // sink. Exercises the pool handshake, egress sharding, lock-free registration,
    // and the request→response round trip. Multiple rids cover several shards.
    #[tokio::test]
    async fn pool_headless_roundtrip() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (tm_tx, tm_rx) = flume::bounded::<TmEvent>(64);
        let (tok_tx, _tok_rx) = flume::bounded(1);
        let senders = Senders {
            tm: tm_tx,
            tok: tok_tx,
            detok: vec![],
        };
        let (ingress_pool, egress_pool) = (6, 4);
        tokio::spawn(server::serve_on(listener, senders, 64, egress_pool));

        // Stub pipeline: echo a terminal Done for every request.
        tokio::spawn(async move {
            while let Ok(TmEvent::Ingress(req)) = tm_rx.recv_async().await {
                let out = crate::message::GenerationOutput {
                    rid: req.id.0.to_string(),
                    text: format!("ok-{}", req.id.0),
                    finish_reason: Some("stop".into()),
                    ..Default::default()
                };
                let _ = req.sink.try_send(EgressItem::Done(out));
            }
        });

        let client = NetClient::connect(&[addr], ingress_pool, egress_pool)
            .await
            .unwrap();

        // Submit several rids so they land on different egress shards.
        for rid in [1u64, 2, 3, 7, 16] {
            let kind = RequestKind::Generate(GenerateRequest {
                payload: GeneratePayload {
                    text: Some("hi".into()),
                    ..Default::default()
                },
                input_ids: None,
                stream: false,
            });
            let mut rx = client.submit(RequestId(rid), kind, 64).await.unwrap();
            let item = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
                .await
                .expect("egress timed out")
                .expect("channel closed");
            match item {
                EgressItem::Done(o) => {
                    assert_eq!(o.rid, rid.to_string());
                    assert_eq!(o.text, format!("ok-{rid}"));
                }
                other => panic!("expected Done for rid {rid}, got {other:?}"),
            }
        }
    }
}
