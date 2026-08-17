//! Messages moved between stages via `flume` (zero-copy moves); variable-length
//! buffers are `bytes::Bytes`, so egress fan-out to detok shards is a refcount bump.
//! Grouped by flow direction: [`request`] (the `/generate` body fan-out, the
//! in-flight request bodies + scheduler ingress wire), [`detok`] (the
//! detokenizer-shard channel), [`egress`]
//! (the response back-channel + egress-ring frames and decoded chunk events),
//! [`finish_reason`] (the terminal reason a request ended, Python's
//! `FinishReasonDict`), [`sampling`] (sampling-params normalization, the Python
//! `SamplingParams` port), [`io_struct`] (the scheduler wire structs), [`types`]
//! (the shared wire-shape adapters both directions use).

pub mod config;
pub mod detok;
pub mod egress;
pub mod finish_reason;
pub mod ids;
pub mod io_struct;
pub mod request;
pub mod sampling;
pub mod types;
