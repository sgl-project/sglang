//! TokenizerManager egress thread — drains the egress ring (scheduler output
//! pushed from Python) and routes each message to the detok shard that owns its
//! `RequestId`. Routing is a pure function of the rid, so it matches the shard
//! the request registered with on ingress — no shared map, no lock.
//!
//! The ring carries a 1-byte frame tag: `BATCH` (a whole decode batch, fanned
//! out here into per-request chunks), `RESULT` (a single control-request JSON
//! payload, e.g. `/server_info`), or `ERROR` (a terminal per-request failure the
//! scheduler ingress couldn't decode, routed back as a 400).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

use crate::ids::RequestId;
use crate::message::{EGRESS_TAG_BATCH, EGRESS_TAG_ERROR, EGRESS_TAG_RESULT, for_each_chunk};
use crate::runtime::Runnable;
use crate::runtime::channels::{DetokMsg, Senders, recv};
use crate::runtime::ring::EgressConsumer;

/// A monotonic counter bumped once per egress-ring frame the dispatcher drains.
/// It's the rust-native equivalent of the Python `TokenizerManager`'s
/// `last_receive_tstamp`: `/health_generate` watches it advance to confirm the
/// scheduler → detok path is alive (the value itself is meaningless).
pub type ActivityCounter = Arc<AtomicU64>;

/// Egress dispatcher stage. Owns the egress-ring consumer + the detok-shard
/// senders, so the runtime spawns it as a [`Runnable`].
pub struct Egress {
    egress: EgressConsumer,
    senders: Senders,
    activity: ActivityCounter,
    shutdown: flume::Receiver<()>,
}

impl Egress {
    pub fn new(
        egress: EgressConsumer,
        senders: Senders,
        activity: ActivityCounter,
        shutdown: flume::Receiver<()>,
    ) -> Self {
        Self {
            egress,
            senders,
            activity,
            shutdown,
        }
    }
}

impl Runnable for Egress {
    fn run(self) {
        while let Some(bytes) = recv(self.egress.receiver(), &self.shutdown) {
            let Some((&tag, body)) = bytes.split_first() else {
                continue;
            };
            match tag {
                // A whole decode batch: decode each request and route it by rid in
                // one pass (no intermediate `Vec<ChunkEvent>`; routing overlaps
                // decode, peak memory is one request). The seq/load-balance the
                // Python side used to track is now implicit in rid-based routing.
                EGRESS_TAG_BATCH => {
                    let ok = for_each_chunk(body, |ev| {
                        // rid is a raw u64 already (numeric column) — no parse.
                        self.route(RequestId(ev.rid), DetokMsg::Chunk(ev));
                    });
                    if !ok {
                        tracing::warn!("egress: bad batch frame");
                    }
                    // Any frame off the ring = the scheduler produced output → the
                    // pipeline is alive.
                    self.activity.fetch_add(1, Ordering::Relaxed);
                }
                EGRESS_TAG_RESULT => {
                    if let Some((rid, msg)) = decode_result(body) {
                        self.route(rid, msg);
                    }
                }
                EGRESS_TAG_ERROR => {
                    if let Some((rid, msg)) = decode_error(body) {
                        self.route(rid, msg);
                    }
                }
                other => tracing::warn!(tag = other, "egress: unknown frame tag"),
            }
        }
    }
}

impl Egress {
    #[inline]
    fn route(&self, rid: RequestId, msg: DetokMsg) {
        if self.senders.detok_for(rid).send(msg).is_err() {
            tracing::error!("egress: detok shard closed");
        }
    }
}

/// Control result: `[rid, payload]` → single non-streamed delivery to the sink.
fn decode_result(body: &[u8]) -> Option<(RequestId, DetokMsg)> {
    let val = rmpv::decode::read_value(&mut &body[..]).ok()?;
    let arr = val.as_array()?;
    let rid = parse_rid(arr.first()?.as_str()?)?;
    let payload = match arr.get(1)? {
        rmpv::Value::Binary(b) => Bytes::copy_from_slice(b),
        rmpv::Value::String(s) => Bytes::copy_from_slice(s.as_bytes()),
        _ => return None,
    };
    Some((rid, DetokMsg::Result { id: rid, payload }))
}

/// Per-request failure: `[rid, message]` → terminal `Error` to the sink (→ 400).
fn decode_error(body: &[u8]) -> Option<(RequestId, DetokMsg)> {
    let val = rmpv::decode::read_value(&mut &body[..]).ok()?;
    let arr = val.as_array()?;
    let rid = parse_rid(arr.first()?.as_str()?)?;
    let message = arr.get(1)?.as_str()?.to_string();
    Some((rid, DetokMsg::Fail { id: rid, message }))
}

fn parse_rid(rid: &str) -> Option<RequestId> {
    match rid.parse::<u64>() {
        Ok(v) => Some(RequestId(v)),
        Err(_) => {
            tracing::warn!(rid = %rid, "egress: unparsable rid");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::frame_egress_error;
    use crate::runtime::channels::DetokMsg;

    /// A framed error round-trips: `frame_egress_error` → tag stripped →
    /// `decode_error` yields the rid + a `Fail` carrying the message.
    #[test]
    fn error_frame_roundtrips_to_fail() {
        let framed = frame_egress_error("42", "invalid request: bad field");
        assert_eq!(framed[0], EGRESS_TAG_ERROR);
        let (rid, msg) = decode_error(&framed[1..]).expect("decodes");
        assert_eq!(rid, RequestId(42));
        match msg {
            DetokMsg::Fail { id, message } => {
                assert_eq!(id, RequestId(42));
                assert_eq!(message, "invalid request: bad field");
            }
            _ => panic!("expected Fail"),
        }
    }
}
