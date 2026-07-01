//! TokenizerManager egress thread — drains the egress ring (scheduler output
//! pushed from Python) and routes each message to the detok shard that owns its
//! `RequestId`. Routing is a pure function of the rid, so it matches the shard
//! the request registered with on ingress — no shared map, no lock.
//!
//! The ring carries a 1-byte frame tag: `CHUNK` (a generation `ChunkEvent`) or
//! `RESULT` (a single control-request JSON payload, e.g. `/server_info`).

use bytes::Bytes;

use crate::ids::RequestId;
use crate::message::{ChunkEvent, EGRESS_TAG_CHUNK, EGRESS_TAG_RESULT};
use crate::runtime::Runnable;
use crate::runtime::channels::{DetokMsg, Senders};
use crate::runtime::ring::EgressConsumer;

/// Egress dispatcher stage. Owns the egress-ring consumer + the detok-shard
/// senders, so the runtime spawns it as a [`Runnable`].
pub struct Egress {
    egress: EgressConsumer,
    senders: Senders,
}

impl Egress {
    pub fn new(egress: EgressConsumer, senders: Senders) -> Self {
        Self { egress, senders }
    }
}

impl Runnable for Egress {
    fn run(self) {
        while let Some(bytes) = self.egress.recv() {
            let Some((&tag, body)) = bytes.split_first() else {
                continue;
            };
            let routed = match tag {
                EGRESS_TAG_CHUNK => decode_chunk(body),
                EGRESS_TAG_RESULT => decode_result(body),
                other => {
                    tracing::warn!(tag = other, "egress: unknown frame tag");
                    continue;
                }
            };
            let Some((rid, msg)) = routed else {
                continue;
            };
            if self.senders.detok_for(rid).send(msg).is_err() {
                tracing::error!("egress: detok shard closed");
            }
        }
    }
}

/// Generation chunk: `[rid, seq, token_ids, finish]` → detok shard.
fn decode_chunk(body: &[u8]) -> Option<(RequestId, DetokMsg)> {
    let ev: ChunkEvent = match rmp_serde::from_slice(body) {
        Ok(ev) => ev,
        Err(e) => {
            tracing::warn!(error = %e, "egress: bad chunk msgpack");
            return None;
        }
    };
    let rid = parse_rid(&ev.rid)?;
    Some((rid, DetokMsg::Chunk(ev)))
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

fn parse_rid(rid: &str) -> Option<RequestId> {
    match rid.parse::<u64>() {
        Ok(v) => Some(RequestId(v)),
        Err(_) => {
            tracing::warn!(rid = %rid, "egress: unparsable rid");
            None
        }
    }
}
