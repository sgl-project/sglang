// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV Indexer: a gRPC service that tracks externally-managed KV cache
//! block placements (as reported by inference engines such as SGLang HiCache)
//! and answers placement-match queries for KV-aware routing.

pub mod bridge;
pub mod client;

pub mod pb {
    tonic::include_proto!("kv_indexer.v1");
}

mod service;
mod shutdown;

#[cfg(feature = "redis-backend")]
pub mod redis_backend;

pub use client::{
    GrpcPrefixIndex, NoSignalReason, PrefixIndex, PrefixIndexConfig, PrefixMatch, PrefixOutcome,
};
pub use service::{KvIndexerBackend, KvIndexerService};
pub use shutdown::shutdown_signal;

#[cfg(feature = "redis-backend")]
pub use redis_backend::RedisKvIndexerBackend;
