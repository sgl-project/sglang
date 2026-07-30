// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-facing client for the prefix-match query.
//!
//! The index is *advisory*: the worker is the source of truth for what it holds,
//! and a stale or unreachable index must never turn into a failed inference
//! request. That safety property is encoded in the types here rather than left to
//! caller discipline:
//!
//!   * [`PrefixOutcome`] has **no error variant**. Every failure — empty result,
//!     unreachable endpoint, timeout, rejection — collapses into
//!     [`PrefixOutcome::NoSignal`], so the router's only two paths are "use this
//!     signal" or "fall back to your existing behaviour".
//!   * The connection is established lazily, so a router does not depend on the
//!     indexer being up at startup.
//!
//! The surface is deliberately tiny (one trait, one method, one outcome type):
//! the router intersects [`PrefixMatch::address`] with its own registered worker
//! URLs and picks by its own load metric, so anything beyond address and prefix
//! length would just be unused weight.

use std::time::Duration;

use tonic::transport::{Channel, Endpoint};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::MatchExternalKvPrefixRequest;

/// Default per-query deadline. Kept far below the bridge's apply RPC timeout so a
/// slow index degrades routing latency negligibly instead of stalling requests.
pub const DEFAULT_QUERY_DEADLINE: Duration = Duration::from_millis(10);

/// One worker's contiguous prefix hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Router-facing routing identity; intersect byte-for-byte with registered
    /// worker URLs. Never empty (the indexer drops unroutable workers).
    pub address: String,
    /// Length of the contiguous request prefix this worker holds.
    pub matched_prefix_blocks: u32,
    /// Opaque worker id, for the caller's logs only.
    pub worker_id: String,
}

/// Why a query produced no usable routing signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoSignalReason {
    /// No worker holds a prefix (or the request had no hashes).
    Empty,
    /// The endpoint could not be reached.
    Unreachable,
    /// The query exceeded its deadline.
    Timeout,
    /// The server rejected the request.
    Rejected,
}

/// Result of a prefix query. Deliberately has no error variant: the index is
/// advisory, so all failures are [`NoSignal`](Self::NoSignal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixOutcome {
    Matched {
        /// Sorted by `matched_prefix_blocks`, descending.
        matches: Vec<PrefixMatch>,
        /// Longest contiguous prefix held by any single worker.
        best_prefix_blocks: u32,
    },
    NoSignal(NoSignalReason),
}

/// Client configuration.
#[derive(Debug, Clone)]
pub struct PrefixIndexConfig {
    /// gRPC endpoint of the indexer, e.g. `http://10.0.0.1:50051`.
    pub endpoint: String,
    /// Per-query deadline.
    pub query_deadline: Duration,
}

impl PrefixIndexConfig {
    /// Config with the default query deadline ([`DEFAULT_QUERY_DEADLINE`]).
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            query_deadline: DEFAULT_QUERY_DEADLINE,
        }
    }
}

/// The prefix-match query the router links against.
#[tonic::async_trait]
pub trait PrefixIndex: Send + Sync {
    /// Queries the longest contiguous prefix each worker holds for `hashes`
    /// (prompt order, `hashes[0]` first). Never fails; see [`PrefixOutcome`].
    async fn match_prefix(&self, hashes: Vec<i64>) -> PrefixOutcome;
}

/// tonic-backed [`PrefixIndex`] with a lazily-established connection.
pub struct GrpcPrefixIndex {
    /// `None` when the endpoint URI could not be parsed; queries then report
    /// [`NoSignalReason::Unreachable`] rather than failing construction.
    channel: Option<Channel>,
    deadline: Duration,
}

impl GrpcPrefixIndex {
    pub fn new(config: PrefixIndexConfig) -> Self {
        let channel = Endpoint::from_shared(config.endpoint)
            .ok()
            .map(|endpoint| endpoint.connect_lazy());
        Self {
            channel,
            deadline: config.query_deadline,
        }
    }
}

#[tonic::async_trait]
impl PrefixIndex for GrpcPrefixIndex {
    async fn match_prefix(&self, hashes: Vec<i64>) -> PrefixOutcome {
        let Some(channel) = self.channel.clone() else {
            return PrefixOutcome::NoSignal(NoSignalReason::Unreachable);
        };
        if hashes.is_empty() {
            return PrefixOutcome::NoSignal(NoSignalReason::Empty);
        }

        let mut client = KvIndexerClient::new(channel);
        let request = MatchExternalKvPrefixRequest {
            // The bridge encodes block hashes as decimal strings; mirror it.
            hashes: hashes.iter().map(|hash| hash.to_string()).collect(),
            max_blocks: 0,
        };

        match tokio::time::timeout(self.deadline, client.match_external_kv_prefix(request)).await {
            Err(_) => PrefixOutcome::NoSignal(NoSignalReason::Timeout),
            Ok(Err(status)) => PrefixOutcome::NoSignal(classify(status.code())),
            Ok(Ok(response)) => {
                let response = response.into_inner();
                if response.matches.is_empty() {
                    return PrefixOutcome::NoSignal(NoSignalReason::Empty);
                }
                let matches = response
                    .matches
                    .into_iter()
                    .map(|m| PrefixMatch {
                        address: m.worker_address,
                        matched_prefix_blocks: m.matched_prefix_blocks,
                        worker_id: m.worker_id,
                    })
                    .collect();
                PrefixOutcome::Matched {
                    matches,
                    best_prefix_blocks: response.best_prefix_blocks,
                }
            }
        }
    }
}

fn classify(code: tonic::Code) -> NoSignalReason {
    match code {
        tonic::Code::Unavailable => NoSignalReason::Unreachable,
        _ => NoSignalReason::Rejected,
    }
}
