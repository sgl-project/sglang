//! ZMQ-based KV-cache event indexer for cache-aware routing.
//!
//! Decodes the msgpack wire format emitted by SGLang's `ZmqEventPublisher`
//! (`python/sglang/srt/utils/event_publisher.py`; KV event types in
//! `python/sglang/srt/disaggregation/kv_events.py`) and maintains the
//! router-side index used for cache-aware request routing.
//!
//! # Submodules
//!
//! - [`wire`] — msgpack types and [`decode_event_batch`]; the contract
//!   with the SGLang publisher. Pure decoding; no I/O.
//! - [`hash`] — block-hash compute mirroring SGLang `RadixKey.hash_page`.
//! - [`tree`] — hash-keyed radix tree consumed by the routing path.
//! - [`subscriber`] — per-worker ZMQ SUB tasks.
//! - [`discovery`] — `/server_info` parse → publisher endpoint.
//! - [`index`] — public façade bundling the tree + subscribers + pump.

pub mod block_size_oracle;
pub mod discovery;
pub mod hash;
pub mod index;
pub mod subscriber;
pub mod tree;
pub mod wire;

pub use block_size_oracle::BlockSizeOracle;
pub(crate) use discovery::classify_bigram;
pub use discovery::{fetch_event_config, EventConfig};
pub use hash::{compute_block_hashes, compute_block_hashes_bigram, sha256_to_i64};
pub use index::KvEventIndex;
pub use subscriber::{KvEventSubscriberRegistry, SubKind, WorkerEvent};
pub use tree::{HashTree, KvWorkerId, MatchResult};
pub use wire::{
    decode_event_batch, BlockRemoved, BlockStored, DecodeError, KvCacheEvent, KvEventBatch,
};
