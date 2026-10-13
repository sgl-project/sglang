// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV Indexer: a gRPC service that tracks externally-managed KV cache
//! block placements (as reported by inference engines such as SGLang HiCache)
//! and answers placement-match queries for KV-aware routing.

pub mod bridge;
pub mod client;
pub mod lora;

pub mod pb {
    tonic::include_proto!("kv_indexer.v1");
}

/// Generated types for the vendored Tair KVCache Manager `MetaService`
/// protocol (`proto/kvcm_meta_service.proto`), used by the KVCM backend.
pub mod kvcm_pb {
    tonic::include_proto!("kv_cache_manager.proto.meta");
}

mod admission;
mod kvcm_backend;
mod memory_backend;
mod service;
mod shutdown;

pub use admission::stamp_arrival;
pub use client::{
    GrpcPrefixIndex, InvalidEndpoint, PrefixIndex, PrefixIndexConfig, PrefixIndexError,
    PrefixMatch, PrefixOutcome, DEFAULT_QUERY_MAX_INFLIGHT,
};
pub use kvcm_backend::{
    KvcmAutoRegister, KvcmBackendConfig, KvcmConfigError, KvcmIndexerBackend,
    DEFAULT_APPLY_TIMEOUT_MS, DEFAULT_HEARTBEAT_INTERVAL_MS, DEFAULT_QUERY_TIMEOUT_MS,
};
pub use memory_backend::InMemoryKvIndexerBackend;
pub use service::{
    component_bit, server_builder, server_builder_with_max_concurrent_streams, BlockComponents,
    KvIndexerBackend, KvIndexerService, WorkerPrefixInput, COMPONENT_FULL, COMPONENT_MAMBA,
    COMPONENT_SWA, DEFAULT_PREFIX_QUERY_MAX_INFLIGHT, MAX_CONCURRENT_STREAMS,
    MAX_GRPC_DECODING_MESSAGE_SIZE,
};
pub use shutdown::shutdown_signal;
/// Re-exported because [`PrefixIndexError::Rejected`] carries it, so callers can
/// match on a rejection without depending on tonic.
pub use tonic::Code as RpcCode;
