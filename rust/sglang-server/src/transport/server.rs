//! Headless TCP transport server (per DP rank). Replaces the embedded HTTP
//! api-server when `dp_size > 1`: a standalone api-server process connects over
//! a pool of TCP connections — `POOL_CONNS` **ingress** (requests in) and
//! `POOL_CONNS` **egress** (frames out) — and this drives the in-process
//! pipeline (tm-ingress → tok → scheduler → detok → tm-egress) via `senders`
//! and `EgressSink::Net`.
//!
//! Each connection announces `[role, index]` on connect. Ingress connections are
//! interchangeable (every frame carries its `rid`). Egress is **sharded**: a
//! request's frames are funneled by `rid % POOL_CONNS` to one shard channel,
//! drained by that shard's egress connection — so one request's ordered frames
//! always ride a single socket, and a slow/large frame on one shard can't
//! head-of-line-block the others. A request's `rid` is assigned by the
//! api-server and carried end-to-end; this side never mints ids.

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::mpsc;

use crate::fsm::RequestState;
use crate::ids::RequestId;
use crate::message::{
    ControlRequest, EgressItem, EgressSink, GeneratePayload, GenerateRequest, Request, RequestKind,
};
use crate::runtime::channels::{Senders, TmEvent};
use crate::transport::{
    Frame, ROLE_EGRESS, ROLE_INGRESS, parse_handshake, read_frame, write_frame,
};

type ShardTx = mpsc::Sender<(RequestId, EgressItem)>;

/// Run the headless TCP server: bind `addr`, then accept the api-server's
/// pool connections and bridge them to the local pipeline. Returns only on bind
/// failure (otherwise loops for the process lifetime).
pub async fn serve_headless(
    bind: SocketAddr,
    senders: Senders,
    egress_buf: usize,
    egress_pool: usize,
) {
    let listener = match TcpListener::bind(bind).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, %bind, "headless: bind failed");
            return;
        }
    };
    tracing::info!(%bind, egress_pool, "sglang-server headless (TCP) listening");
    serve_on(listener, senders, egress_buf, egress_pool).await;
}

/// Accept loop over an already-bound listener (split out so tests can supply an
/// ephemeral-port listener and learn its address). `egress_pool` = egress shard
/// count and must match the api-server's `ServerArgs::egress_pool_size`; the
/// number of ingress connections is up to the api-server (we accept any).
pub(crate) async fn serve_on(
    listener: TcpListener,
    senders: Senders,
    egress_buf: usize,
    egress_pool: usize,
) {
    // One egress shard channel per slot: a request's `Net` sink funnels
    // `(rid, item)` into shard `rid % egress_pool`, drained by that shard's
    // egress connection writer. The first EGRESS connection for a shard takes its
    // rx. `egress_rx_slots` is touched only by this (single, sequential) accept
    // loop, so it needs neither `Arc` nor a lock; `egress_txs` is `Arc` because
    // it *is* shared (cloned into every ingress loop).
    let mut egress_txs: Vec<ShardTx> = Vec::with_capacity(egress_pool);
    let mut egress_rx_slots: Vec<Option<mpsc::Receiver<(RequestId, EgressItem)>>> =
        Vec::with_capacity(egress_pool);
    for _ in 0..egress_pool {
        let (tx, rx) = mpsc::channel::<(RequestId, EgressItem)>(egress_buf);
        egress_txs.push(tx);
        egress_rx_slots.push(Some(rx));
    }
    let egress_txs = Arc::new(egress_txs);

    loop {
        let (mut stream, peer) = match listener.accept().await {
            Ok(x) => x,
            Err(e) => {
                tracing::warn!(error = %e, "headless: accept failed");
                continue;
            }
        };
        // Disable Nagle: egress frames are latency-sensitive token deltas.
        let _ = stream.set_nodelay(true);
        // Handshake: [role: u8][index: u16-BE].
        let mut hdr = [0u8; 3];
        if let Err(e) = stream.read_exact(&mut hdr).await {
            tracing::warn!(error = %e, %peer, "headless: handshake read failed");
            continue;
        }
        let (role, index) = parse_handshake(hdr);
        match role {
            ROLE_INGRESS => {
                let (read, _write) = stream.into_split();
                tokio::spawn(ingress_loop(read, senders.clone(), egress_txs.clone()));
            }
            ROLE_EGRESS if index < egress_pool => {
                let (_read, write) = stream.into_split();
                match egress_rx_slots[index].take() {
                    Some(rx) => {
                        tokio::spawn(egress_loop(write, rx));
                    }
                    // v1 supports one api-server (one connection per shard): once
                    // a shard's rx is taken, a second/reconnecting egress
                    // connection for it is rejected.
                    // TODO(headless: reconnect) — to let a dropped egress
                    // connection re-bind (the client-side counterpart in
                    // `client.rs`), `egress_loop` must return its `rx` to this
                    // slot on exit (e.g. hand the slot an `mpsc` the loop sends
                    // the rx back on, or wrap slots so a reconnect can re-take).
                    None => {
                        tracing::warn!(%peer, shard = index, "headless: extra egress connection")
                    }
                }
            }
            _ => tracing::warn!(role, index, %peer, "headless: bad connection handshake"),
        }
    }
}

/// Read framed requests off one ingress connection and push them into the
/// pipeline, attaching each request a `Net` sink on its egress shard. Exits when
/// the connection closes.
async fn ingress_loop(mut read: OwnedReadHalf, senders: Senders, egress_txs: Arc<Vec<ShardTx>>) {
    let shard_of = |rid: u64| egress_txs[(rid as usize) % egress_txs.len()].clone();
    loop {
        let (frame, tail) = match read_frame(&mut read).await {
            Ok(f) => f,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    tracing::warn!(error = %e, "headless: ingress read error");
                }
                break;
            }
        };
        let req = match frame {
            Frame::Generate {
                rid,
                text,
                sampling_params,
                stream,
            } => build_generate(rid, text, sampling_params, stream, tail, shard_of(rid)),
            Frame::Control { rid, tag } => match build_control(rid, &tag, shard_of(rid)) {
                Some(r) => r,
                None => {
                    tracing::warn!(%tag, "headless: unknown control tag; dropping");
                    continue;
                }
            },
            _ => {
                tracing::warn!("headless: unexpected frame on ingress connection");
                continue;
            }
        };
        if senders.tm.send_async(TmEvent::Ingress(req)).await.is_err() {
            tracing::error!("headless: tm inbox closed");
            break;
        }
    }
}

/// Drain one egress shard channel and write each `(rid, item)` to its egress
/// connection. Exits on write error (connection gone) — which drops the
/// receiver, so subsequent requests' `Net` sinks for this shard fail fast.
//
// TODO(headless: reconnect) — dropping the rx on exit also means the shard can't
// be re-served by a reconnecting egress connection (the slot stays `None`).
// Return the rx to its `egress_rx_slots` entry here so the accept loop can hand
// it to the next connection for this shard.
async fn egress_loop(mut write: OwnedWriteHalf, mut rx: mpsc::Receiver<(RequestId, EgressItem)>) {
    while let Some((rid, item)) = rx.recv().await {
        let frame = Frame::Egress {
            rid: rid.0,
            item: item.into(),
        };
        if let Err(e) = write_frame(&mut write, &frame, &[]).await {
            tracing::warn!(error = %e, "headless: egress write error; closing");
            break;
        }
    }
}

fn build_generate(
    rid: u64,
    text: Option<String>,
    sampling_params: Option<rmpv::Value>,
    stream: bool,
    tail: Bytes,
    egress_tx: ShardTx,
) -> Request {
    // Pre-tokenized prompt? token ids ride in the frame tail as raw i32-LE.
    let input_ids: Option<Vec<i32>> = (!tail.is_empty()).then(|| {
        tail.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    });
    let id = RequestId(rid);
    Request {
        id,
        state: RequestState::Received,
        sink: EgressSink::Net {
            rid: id,
            tx: egress_tx,
        },
        kind: RequestKind::Generate(GenerateRequest {
            payload: GeneratePayload {
                text,
                input_ids,
                stream,
                sampling_params,
                extra: Default::default(),
            },
            // Filled by the tokenizer stage (or copied from the payload by the
            // ingress `classify` when already tokenized) — never on the wire.
            input_ids: None,
            stream,
        }),
    }
}

fn build_control(rid: u64, tag: &str, egress_tx: ShardTx) -> Option<Request> {
    let id = RequestId(rid);
    Some(Request {
        id,
        state: RequestState::Received,
        sink: EgressSink::Net {
            rid: id,
            tx: egress_tx,
        },
        kind: RequestKind::Control(ControlRequest {
            tag: static_control_tag(tag)?,
        }),
    })
}

/// Map a wire control tag back to the `&'static str` the pipeline expects. The
/// set of control tags is small and compile-time fixed, so we match rather than
/// leak a runtime string; unknown tags are rejected.
fn static_control_tag(tag: &str) -> Option<&'static str> {
    match tag {
        "GetInternalStateReq" => Some("GetInternalStateReq"),
        _ => None,
    }
}
