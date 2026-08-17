//! Multimodal worker pool.
//!
//! Rust threads drain requests parked in `Encoding` and run the `sglang-mm`
//! pipeline registered by `Server.start_mm_workers` (decode → preprocess →
//! placeholder expansion → M-RoPE, GIL-free). Each worker parks the result
//! buffers in the rid-keyed [`Sidecar`] and returns only the expanded ids;
//! Python attaches the buffers at drain time (`Server.take_mm`). Inputs the
//! pipeline cannot serve are rejected to the client — no Python fallback.

pub mod payload;
pub mod sidecar;
pub mod worker;

// `ShmSegment` never leaves the module — callers see segments only through
// `FeatureStore`.
mod shm;
