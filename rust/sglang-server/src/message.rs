//! Messages moved between stages via `flume` (zero-copy moves); variable-length
//! buffers are `bytes::Bytes`, so egress fan-out to detok shards is a refcount bump.
//! Grouped by wire: [`body`] (the `/generate` HTTP body fan-out), [`request`]
//! (the in-flight request + scheduler-wire encodes), [`chunk`] (the egress-ring
//! frames and decoded chunk events), [`sampling`] (sampling-params
//! normalization, the Python `SamplingParams` port).

mod body;
mod chunk;
mod request;
pub mod sampling;

pub use body::GenerateBody;
pub use chunk::{
    ChunkEvent, ChunkExtras, EGRESS_TAG_BATCH, EGRESS_TAG_ERROR, EGRESS_TAG_RESULT, for_each_chunk,
    frame_egress_batch_cols, frame_egress_error, frame_egress_result,
};
pub use request::{
    ControlRequest, EgressItem, EgressSink, GenerateRequest, IngressMsg, Request, RequestKind,
    SinkError, abort_req_msgpack, control_req_msgpack,
};
