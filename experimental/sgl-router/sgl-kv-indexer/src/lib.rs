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

mod admission;
mod memory_backend;
mod service;
mod shutdown;

pub use admission::stamp_arrival;
pub use client::{
    GrpcPrefixIndex, InvalidEndpoint, PrefixIndex, PrefixIndexConfig, PrefixIndexError,
    PrefixMatch, PrefixOutcome, DEFAULT_QUERY_MAX_INFLIGHT,
};
pub use memory_backend::InMemoryKvIndexerBackend;
pub use service::{
    component_bit, server_builder, BlockComponents, KvIndexerBackend, KvIndexerService,
    WorkerPrefixInput, COMPONENT_FULL, COMPONENT_MAMBA, COMPONENT_SWA,
    DEFAULT_PREFIX_QUERY_MAX_INFLIGHT, MAX_CONCURRENT_STREAMS, MAX_GRPC_DECODING_MESSAGE_SIZE,
};
pub use shutdown::shutdown_signal;
/// Re-exported because [`PrefixIndexError::Rejected`] carries it, so callers can
/// match on a rejection without depending on tonic.
pub use tonic::Code as RpcCode;
