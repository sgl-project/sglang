//! Messages moved between stages via `flume` (zero-copy moves); variable-length
//! buffers are `bytes::Bytes`, so fanning one out to several detok shards is a
//! refcount bump, not a copy.

pub mod config;
pub mod detok;
pub mod finish_reason;
pub mod ids;
pub mod io_struct;
pub mod request;
pub mod response;
pub mod sampling;
pub mod types;
