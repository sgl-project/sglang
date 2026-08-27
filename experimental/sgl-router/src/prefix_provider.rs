// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-owned abstraction for external prefix-placement providers.
//!
//! The current gRPC KV Indexer remains the default implementation. Keeping the
//! contract in the Router allows another placement backend to be introduced
//! without coupling chat routing and policy selection to its transport.

use sgl_kv_indexer::{PrefixIndex, PrefixIndexError, PrefixOutcome};

/// Default external prefix provider used by Router startup.
pub type DefaultPrefixMatchProvider = sgl_kv_indexer::GrpcPrefixIndex;

#[async_trait::async_trait]
pub trait PrefixMatchProvider: Send + Sync {
    async fn match_prefix_for_workers(
        &self,
        hashes: Vec<i64>,
        eligible_worker_addresses: Vec<String>,
    ) -> Result<PrefixOutcome, PrefixIndexError>;

    fn provider_name(&self) -> &'static str;
}

#[async_trait::async_trait]
impl PrefixMatchProvider for DefaultPrefixMatchProvider {
    async fn match_prefix_for_workers(
        &self,
        hashes: Vec<i64>,
        eligible_worker_addresses: Vec<String>,
    ) -> Result<PrefixOutcome, PrefixIndexError> {
        PrefixIndex::match_prefix_for_workers(self, hashes, eligible_worker_addresses).await
    }

    fn provider_name(&self) -> &'static str {
        "kv-indexer"
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use sgl_kv_indexer::{PrefixIndexConfig, DEFAULT_PREFIX_QUERY_MAX_INFLIGHT};

    #[test]
    fn grpc_kv_indexer_is_the_default_provider() {
        let provider = DefaultPrefixMatchProvider::new(PrefixIndexConfig {
            endpoint: "http://127.0.0.1:50051".into(),
            query_deadline: Duration::from_millis(100),
            max_inflight: DEFAULT_PREFIX_QUERY_MAX_INFLIGHT,
        })
        .unwrap();
        assert_eq!(PrefixMatchProvider::provider_name(&provider), "kv-indexer");
    }
}
