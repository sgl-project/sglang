//! Messages moved between stages via `flume` (zero-copy moves); variable-length
//! buffers are `bytes::Bytes`, so egress fan-out to detok shards is a refcount bump.
//! Grouped by flow direction: [`request`] (the `/generate` body fan-out, the
//! in-flight request bodies + scheduler ingress wire), [`egress`]
//! (the response back-channel + egress-ring frames and decoded chunk events),
//! [`finish_reason`] (the terminal reason a request ended, Python's
//! `FinishReasonDict`), [`sampling`] (sampling-params normalization, the Python
//! `SamplingParams` port), [`io_struct`] (the scheduler wire structs), [`types`]
//! (the shared wire-shape adapters both directions use).

mod egress;
mod finish_reason;
mod io_struct;
mod request;
mod sampling;
mod types;

pub use egress::{
    ChunkEvent, ChunkExtras, EGRESS_TAG_BATCH, EGRESS_TAG_ERROR, EGRESS_TAG_RESULT, EgressItem,
    EgressSink, SinkError, for_each_chunk, frame_egress_batch_cols, frame_egress_error,
    frame_egress_result,
};
pub use finish_reason::Matched;
pub(crate) use io_struct::{AbortReq, ControlRequest, GetInternalStateReq};
pub use request::{GenerateBody, GenerateRequest, RequestKind};
pub(crate) use sampling::{SamplingParams, SamplingParamsInput};
pub(crate) use types::{OneOrMany, OneOrManyItem, TokenIds};

use bytes::Bytes;

use crate::fsm::RequestState;
use crate::ids::Rid;

/// The owned request as it travels ingress stages (single owner, so `state` is
/// mutated lock-free). Common fields here; variant data in [`RequestKind`].
#[derive(Debug)]
pub struct Request {
    /// Client-visible request id (uuid hex) — what the scheduler wire and
    /// `meta_info.id` carry.
    pub rid: Rid,
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
/// the shard's local `rid -> sink` map. The rid STRING is the identity: `Rid::hash`
/// picks the shard (collisions there merely co-locate, which is harmless), but two
/// distinct rids that hash alike must not be the same map entry — that evicted one
/// client's sink and delivered their tokens to the other's connection. Equal rids
/// cannot reach here from different requests: `Rid::from_client` uniquifies every
/// client-supplied one.
pub enum DetokMsg {
    Register {
        /// Client-visible rid string — kept in `DetokState` so the shard can
        /// emit `TmEvent::Abort(rid)` (the wire needs the string, not the hash).
        rid: Rid,
        sink: EgressSink,
        /// Decode logprob token ids to text here (CPU-bound) not on the api threads.
        decode_logprob_text: bool,
        /// `SamplingParams.no_stop_trim`: keep the matched stop; default trims it.
        no_stop_trim: bool,
    },
    /// One decode step's chunks for *this shard*. Batched because `tm-egress` blocks
    /// per send, so one message per request cost ~1.3 µs × batch (5.1x at 4096).
    Chunks(Vec<ChunkEvent>),
    /// Decode a complete token-id sequence — the backend of
    /// [`RequestKind::Detokenize`], the one request kind the detok stage itself
    /// answers (it never reaches the scheduler ring). Sent by tm-ingress right
    /// after the same rid's `Register` on the same channel (FIFO), so the shard
    /// delivers the text through the registered sink like a control `Result`
    /// and drops the entry.
    Decode { rid: Rid, token_ids: Vec<u32> },
    /// Control result: one already-serialized payload delivered to the sink verbatim.
    Result { rid: Rid, payload: bytes::Bytes },
    /// Terminal per-request failure → an `Error` to the sink (a 400, not a crash).
    Fail { rid: Rid, message: String },
    /// Drop the `rid -> sink` entry for a request rejected before the scheduler
    /// (the rejecting stage already answered the client); else `Register` leaks one
    /// entry.
    Deregister { rid: Rid },
}
