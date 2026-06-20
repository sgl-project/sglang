//! TokenizerManager egress thread — drains the egress ring (scheduler output
//! pushed from Python via `push_chunk`) and routes each `ChunkEvent` to the
//! detok shard that owns its `RequestId`. Routing is a pure function of the rid,
//! so it matches the shard the request registered with on ingress — no shared
//! map, no lock.

use crate::ids::RequestId;
use crate::message::ChunkEvent;
use crate::runtime::channels::{DetokMsg, Senders};
use crate::runtime::ring::EgressConsumer;

pub fn run_egress(egress: EgressConsumer, senders: Senders) {
    while let Some(bytes) = egress.recv() {
        let ev: ChunkEvent = match rmp_serde::from_slice(&bytes) {
            Ok(ev) => ev,
            Err(e) => {
                tracing::warn!(error = %e, "egress: bad chunk msgpack");
                continue;
            }
        };

        let Ok(rid) = ev.rid.parse::<u64>() else {
            tracing::warn!(rid = %ev.rid, "egress: unparsable rid");
            continue;
        };

        let shard = senders.detok_for(RequestId(rid));
        if shard.send(DetokMsg::Chunk(ev)).is_err() {
            tracing::error!("egress: detok shard closed");
        }
    }
}
