//! Messages moved between stages via `flume` (zero-copy moves); variable-length
//! buffers are `bytes::Bytes`, so egress fan-out to detok shards is a refcount bump.
//! Grouped by flow direction: [`request`] (the `/generate` body fan-out, the
//! in-flight request bodies + scheduler ingress wire), [`egress`]
//! (the response back-channel + egress-ring frames and decoded chunk events),
//! [`sampling`] (sampling-params normalization, the Python `SamplingParams` port).
#![allow(dead_code)] // TODO: remove when the consumer PR lands

mod egress;
mod request;
mod sampling;

pub use egress::{
    ChunkEvent, EGRESS_TAG_BATCH, EGRESS_TAG_ERROR, EGRESS_TAG_RESULT, EgressItem, EgressSink,
    SinkError, for_each_chunk,
};
pub use request::{RequestKind, abort_req_msgpack, control_req_msgpack};
pub use sampling::normalize_sampling_params;

use bytes::Bytes;

use crate::fsm::RequestState;
use crate::ids::RidHash;

/// The owned request as it travels ingress stages (single owner, so `state` is
/// mutated lock-free). Common fields here; variant data in [`RequestKind`].
#[derive(Debug)]
pub struct Request {
    /// Routing key: `RidHash::from_rid(&rid)`.
    pub rid_hash: RidHash,
    /// Client-visible request id (uuid hex) — what the scheduler wire and
    /// `meta_info.id` carry.
    pub rid: String,
    pub state: RequestState,
    /// Back-channel to the client connection for egress frames.
    pub sink: EgressSink,
    /// Discriminant + variant body (generate vs control).
    pub kind: RequestKind,
}

/// One ingress-ring entry, split columnar: the scalar `header` (msgpack, `input_ids`
/// omitted) + the raw int64 `ids` cell, so the big tensor never goes through msgpack.
#[derive(Debug)]
pub struct IngressMsg {
    pub header: Bytes,
    pub ids: Bytes,
}

/// Messages to a Detokenizer shard. `Register` carries the per-request sink for
/// the shard's local `rid_hash -> sink` map; everything routes by `RidHash::shard`.
pub enum DetokMsg {
    Register {
        rid_hash: RidHash,
        /// Client-visible rid string — kept in `DetokState` so the shard can
        /// emit `TmEvent::Abort(rid)` (the wire needs the string, not the hash).
        rid: String,
        sink: EgressSink,
        /// Decode logprob token ids to text here (CPU-bound) not on the api threads.
        decode_logprob_text: bool,
        /// `SamplingParams.no_stop_trim`: keep the matched stop; default trims it.
        no_stop_trim: bool,
    },
    /// One decode step's chunks for *this shard*. Batched because `tm-egress` blocks
    /// per send, so one message per request cost ~1.3 µs × batch (5.1x at 4096).
    Chunks(Vec<ChunkEvent>),
    /// Control result: one already-serialized payload delivered to the sink verbatim.
    Result {
        rid_hash: RidHash,
        payload: bytes::Bytes,
    },
    /// Terminal per-request failure → an `Error` to the sink (a 400, not a crash).
    Fail { rid_hash: RidHash, message: String },
    /// Drop the `rid_hash -> sink` entry for a request rejected before the scheduler
    /// (the rejecting stage already answered the client); else `Register` leaks one
    /// entry.
    Deregister { rid_hash: RidHash },
}
