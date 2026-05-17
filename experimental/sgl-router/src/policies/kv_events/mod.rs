//! ZMQ-based KV-cache event indexer for cache-aware routing.
//!
//! Decodes the msgpack wire format emitted by SGLang's `ZmqEventPublisher`
//! (see `python/sglang/srt/disaggregation/kv_events.py`) and maintains the
//! router-side index used for cache-aware request routing.
//!
//! # Submodules
//!
//! - [`wire`]: msgpack types and [`decode_event_batch`] — the contract
//!   between the SGLang publisher (`msgspec.msgpack` with
//!   `array_like=True, omit_defaults=True, gc=False, tag=True`) and this
//!   crate. Pure decoding; no I/O.
//!
//! Subsequent M3 tasks add: `tree`, `subscriber`, `discovery`.

pub mod hash;
pub mod wire;

pub use hash::{compute_block_hashes, sha256_to_i64};
pub use wire::{
    decode_event_batch, BlockRemoved, BlockStored, DecodeError, KvCacheEvent, KvEventBatch,
};
