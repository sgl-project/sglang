//! Messages moved between stages via `flume` (zero-copy moves); variable-length
//! buffers are `bytes::Bytes`, so egress fan-out to detok shards is a refcount bump.
//! Grouped by flow direction: [`request`] (the `/generate` body fan-out, the
//! in-flight request bodies + scheduler ingress wire), [`egress`]
//! (the response back-channel + egress-ring frames and decoded chunk events),
//! [`sampling`] (sampling-params normalization, the Python `SamplingParams` port).

mod egress;
mod request;
mod sampling;

pub use egress::{
    ChunkEvent, ChunkExtras, EGRESS_TAG_BATCH, EGRESS_TAG_ERROR, EGRESS_TAG_RESULT, EgressItem,
    EgressSink, SinkError, for_each_chunk, frame_egress_batch_cols, frame_egress_error,
    frame_egress_result,
};
pub use request::{
    ControlRequest, GenerateBody, GenerateRequest, IngressMsg, RequestKind, abort_req_msgpack,
    control_req_msgpack,
};
pub use sampling::normalize_sampling_params;

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
