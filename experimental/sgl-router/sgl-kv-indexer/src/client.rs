// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-facing client for the prefix-match query.
//!
//! A successful query distinguishes a real match from an empty result. Transport
//! failures, deadlines, and server rejections stay distinct errors so the caller
//! chooses between degrading and failing the request, instead of silently using
//! a different signal.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{Semaphore, SemaphorePermit};
use tonic::transport::{Channel, Endpoint};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::MatchExternalKvPrefixRequest;
use crate::service::MAX_GRPC_DECODING_MESSAGE_SIZE;
use crate::status::{IndexerStatusRegistry, DEFAULT_STATUS_FRESHNESS};

/// Default per-query deadline. Indexer failures are request failures, so this
/// absorbs normal cross-host jitter without stalling a request indefinitely.
pub const DEFAULT_QUERY_DEADLINE: Duration = Duration::from_millis(100);
/// Default process-local bound on prefix-query RPCs issued by one client.
pub const DEFAULT_QUERY_MAX_INFLIGHT: usize = 32;
// Leave room for the packed field tag, length prefix, and future scalar fields.
const PREFIX_QUERY_ENCODING_HEADROOM: usize = 16;
const MAX_PREFIX_HASHES_PER_QUERY: usize =
    (MAX_GRPC_DECODING_MESSAGE_SIZE - PREFIX_QUERY_ENCODING_HEADROOM) / 8;

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

/// A failed prefix query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixIndexError {
    /// The endpoint could not be reached.
    Unreachable,
    /// The query exceeded its deadline.
    Timeout,
    /// The client or Indexer shed the query because its in-flight limit was hit.
    Overloaded,
    /// The query exceeded the Indexer's gRPC message-size limit, so no worker's
    /// prefix was scanned. Bounded by prompt length, not by load: retrying the
    /// same prompt cannot succeed.
    QueryTooLarge,
    /// The server rejected the request.
    Rejected(tonic::Code),
}

impl std::fmt::Display for PrefixIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable => f.write_str("KV Indexer is unreachable"),
            Self::Timeout => f.write_str("KV Indexer query timed out"),
            Self::Overloaded => f.write_str("KV Indexer is overloaded"),
            Self::QueryTooLarge => {
                f.write_str("KV Indexer query exceeded the gRPC message-size limit")
            }
            Self::Rejected(code) => write!(f, "KV Indexer rejected the query with {code}"),
        }
    }
}

impl std::error::Error for PrefixIndexError {}

/// A configured endpoint that is not a usable gRPC target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidEndpoint {
    endpoint: String,
    reason: &'static str,
}

impl std::fmt::Display for InvalidEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid KV Indexer endpoint `{}`: {}",
            self.endpoint, self.reason
        )
    }
}

impl std::error::Error for InvalidEndpoint {}

/// Result of a successful prefix query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixOutcome {
    Matched {
        /// Sorted by `matched_prefix_blocks`, descending.
        matches: Vec<PrefixMatch>,
        /// Longest contiguous prefix held by any single worker.
        best_prefix_blocks: u32,
    },
    /// No worker holds a prefix (or the request had no hashes).
    Empty,
}

/// Client configuration.
#[derive(Debug, Clone)]
pub struct PrefixIndexConfig {
    /// gRPC endpoint of the indexer, e.g. `http://10.0.0.1:50051`.
    pub endpoint: String,
    /// Per-query deadline.
    pub query_deadline: Duration,
    /// Maximum prefix-query RPCs issued concurrently by this client.
    pub max_inflight: usize,
}

impl PrefixIndexConfig {
    /// Config with the default query deadline ([`DEFAULT_QUERY_DEADLINE`]).
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            query_deadline: DEFAULT_QUERY_DEADLINE,
            max_inflight: DEFAULT_QUERY_MAX_INFLIGHT,
        }
    }
}

/// The prefix-match query the router links against.
#[tonic::async_trait]
pub trait PrefixIndex: Send + Sync {
    /// Queries the longest contiguous prefix each worker holds for `hashes`
    /// (prompt order, `hashes[0]` first).
    async fn match_prefix(&self, hashes: Vec<i64>) -> Result<PrefixOutcome, PrefixIndexError>;
}

/// tonic-backed [`PrefixIndex`] with a lazily-established connection.
pub struct GrpcPrefixIndex {
    registry: Arc<IndexerStatusRegistry>,
    channels: Mutex<HashMap<String, Channel>>,
    deadline: Duration,
    prefix_query_semaphore: Semaphore,
}

impl GrpcPrefixIndex {
    /// Rejects an unusable endpoint instead of building a client that can only
    /// fail, so a misconfigured address stops startup rather than silently
    /// costing every request its cache affinity.
    pub fn new(config: PrefixIndexConfig) -> Result<Self, InvalidEndpoint> {
        assert!(
            config.max_inflight > 0,
            "prefix query max inflight must be greater than zero"
        );
        let endpoints: Vec<String> = config
            .endpoint
            .split(',')
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .map(str::to_owned)
            .collect();
        if endpoints.is_empty() {
            return Err(InvalidEndpoint {
                endpoint: config.endpoint,
                reason: "at least one endpoint is required",
            });
        }
        for endpoint in &endpoints {
            parse_endpoint(endpoint)?;
        }
        Ok(Self {
            registry: Arc::new(IndexerStatusRegistry::new(
                endpoints,
                DEFAULT_STATUS_FRESHNESS,
            )),
            channels: Mutex::new(HashMap::new()),
            deadline: config.query_deadline,
            prefix_query_semaphore: Semaphore::new(config.max_inflight),
        })
    }

    pub fn status_registry(&self) -> Arc<IndexerStatusRegistry> {
        Arc::clone(&self.registry)
    }

    fn try_acquire_prefix_query(&self) -> Result<SemaphorePermit<'_>, PrefixIndexError> {
        self.prefix_query_semaphore
            .try_acquire()
            .map_err(|_| PrefixIndexError::Overloaded)
    }

    fn channel_for(&self, endpoint: &str) -> Result<Channel, PrefixIndexError> {
        let mut channels = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(channel) = channels.get(endpoint) {
            return Ok(channel.clone());
        }
        let channel = parse_endpoint(endpoint)
            .map_err(|_| PrefixIndexError::Unreachable)?
            .connect_lazy();
        channels.insert(endpoint.to_owned(), channel.clone());
        Ok(channel)
    }

    async fn query_one(
        &self,
        endpoint: &str,
        hashes: &[i64],
    ) -> Result<PrefixOutcome, PrefixIndexError> {
        let mut client = KvIndexerClient::new(self.channel_for(endpoint)?);
        let mut request = tonic::Request::new(MatchExternalKvPrefixRequest {
            hashes: hashes.to_vec(),
            max_blocks: 0,
        });
        request.set_timeout(self.deadline);
        match tokio::time::timeout(self.deadline, client.match_external_kv_prefix(request)).await {
            Err(_) => Err(PrefixIndexError::Timeout),
            Ok(Err(status)) => Err(classify(status.code())),
            Ok(Ok(response)) => {
                let response = response.into_inner();
                if response.matches.is_empty() {
                    return Ok(PrefixOutcome::Empty);
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
                Ok(PrefixOutcome::Matched {
                    matches,
                    best_prefix_blocks: response.best_prefix_blocks,
                })
            }
        }
    }
}

fn truncate_prefix_query(hashes: &mut Vec<i64>) -> Option<usize> {
    let total = hashes.len();
    hashes.truncate(MAX_PREFIX_HASHES_PER_QUERY);
    (hashes.len() != total).then_some(total)
}

#[tonic::async_trait]
impl PrefixIndex for GrpcPrefixIndex {
    async fn match_prefix(&self, mut hashes: Vec<i64>) -> Result<PrefixOutcome, PrefixIndexError> {
        if hashes.is_empty() {
            return Ok(PrefixOutcome::Empty);
        }

        if let Some(total_hashes) = truncate_prefix_query(&mut hashes) {
            tracing::warn!(
                total_hashes,
                queried_hashes = hashes.len(),
                "KV Indexer query truncated to the gRPC message-size limit"
            );
        }

        let _permit = self.try_acquire_prefix_query()?;

        let candidates = self.registry.candidates();
        if candidates.is_empty() {
            return Err(PrefixIndexError::Unreachable);
        }
        let mut last_error = PrefixIndexError::Unreachable;
        for candidate in candidates {
            match self.query_one(&candidate.endpoint, &hashes).await {
                Ok(outcome) => return Ok(outcome),
                Err(
                    error @ (PrefixIndexError::Unreachable
                    | PrefixIndexError::Timeout
                    | PrefixIndexError::Overloaded),
                ) => {
                    tracing::warn!(
                        indexer_id = %candidate.indexer_id,
                        endpoint = %candidate.endpoint,
                        %error,
                        "KV Indexer query failed; trying next READY replica"
                    );
                    last_error = error;
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error)
    }
}

/// Validates the endpoint the operator configured. tonic itself only checks URI
/// syntax, which accepts a host:port with no scheme and then fails on every
/// connect, so the scheme and host are checked here.
fn parse_endpoint(endpoint: &str) -> Result<Endpoint, InvalidEndpoint> {
    let reject = |reason: &'static str| InvalidEndpoint {
        endpoint: endpoint.to_string(),
        reason,
    };
    let parsed =
        Endpoint::from_shared(endpoint.to_string()).map_err(|_| reject("not a valid URI"))?;
    // A `unix:` endpoint is fully specified by its socket path.
    if endpoint.starts_with("unix:") {
        return Ok(parsed);
    }
    match parsed.uri().scheme_str() {
        None => Err(reject("missing scheme, expected http:// or https://")),
        Some("http" | "https") => {
            if parsed.uri().host().unwrap_or_default().is_empty() {
                Err(reject("missing host"))
            } else {
                Ok(parsed)
            }
        }
        Some(_) => Err(reject("unsupported scheme, expected http:// or https://")),
    }
}

fn classify(code: tonic::Code) -> PrefixIndexError {
    match code {
        tonic::Code::Unavailable => PrefixIndexError::Unreachable,
        // The indexer sheds an expired query as DEADLINE_EXCEEDED, while tonic
        // reports its own enforcement of the same `grpc-timeout` as CANCELLED.
        // This client cancels a query for no other reason.
        tonic::Code::DeadlineExceeded | tonic::Code::Cancelled => PrefixIndexError::Timeout,
        tonic::Code::ResourceExhausted => PrefixIndexError::Overloaded,
        // The indexer's decoder refuses a message past its size limit with
        // OUT_OF_RANGE. A prompt too long to carry is not a disagreement about
        // the request contract, so it stays separable from `Rejected` and the
        // caller can degrade instead of failing the request.
        tonic::Code::OutOfRange => PrefixIndexError::QueryTooLarge,
        _ => PrefixIndexError::Rejected(code),
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;

    #[test]
    fn classifies_resource_exhausted_as_overload() {
        assert_eq!(
            classify(tonic::Code::ResourceExhausted),
            PrefixIndexError::Overloaded
        );
    }

    /// An over-limit query must stay distinguishable from a contract rejection:
    /// the caller degrades on the former and fails the request on the latter.
    #[test]
    fn classifies_over_limit_message_as_too_large() {
        assert_eq!(
            classify(tonic::Code::OutOfRange),
            PrefixIndexError::QueryTooLarge
        );
        assert_eq!(
            classify(tonic::Code::InvalidArgument),
            PrefixIndexError::Rejected(tonic::Code::InvalidArgument)
        );
    }

    #[test]
    fn oversized_query_keeps_a_prefix_within_the_transport_limit() {
        let total = MAX_PREFIX_HASHES_PER_QUERY + 1;
        let mut hashes = vec![-1; total];

        assert_eq!(truncate_prefix_query(&mut hashes), Some(total));
        assert_eq!(hashes.len(), MAX_PREFIX_HASHES_PER_QUERY);
        assert!(
            MatchExternalKvPrefixRequest {
                hashes,
                max_blocks: 0,
            }
            .encoded_len()
                <= MAX_GRPC_DECODING_MESSAGE_SIZE
        );
    }

    #[test]
    fn classifies_both_deadline_signals_as_timeout() {
        for code in [tonic::Code::DeadlineExceeded, tonic::Code::Cancelled] {
            assert_eq!(classify(code), PrefixIndexError::Timeout);
        }
    }

    #[test]
    fn accepts_endpoints_the_client_can_actually_dial() {
        for endpoint in [
            "http://10.0.0.1:50051",
            "https://indexer.svc:443",
            "unix:/tmp/i",
        ] {
            assert!(
                parse_endpoint(endpoint).is_ok(),
                "{endpoint} should be accepted"
            );
        }
    }

    /// A host:port with no scheme parses as a URI but can never connect, which
    /// is the misconfiguration that otherwise only shows up under traffic.
    #[test]
    fn rejects_endpoints_that_could_only_fail_at_query_time() {
        for endpoint in [
            "10.0.0.1:50051",
            "indexer.svc",
            "grpc://10.0.0.1:50051",
            "http://",
        ] {
            let error = parse_endpoint(endpoint)
                .expect_err(&format!("{endpoint} should be rejected"))
                .to_string();
            assert!(
                error.contains(endpoint),
                "error should name the endpoint: {error}"
            );
        }
    }

    #[test]
    fn construction_fails_on_an_invalid_endpoint() {
        assert!(GrpcPrefixIndex::new(PrefixIndexConfig::new("10.0.0.1:50051")).is_err());
    }

    #[tokio::test]
    async fn local_admission_rejects_without_queueing() {
        let index = GrpcPrefixIndex::new(PrefixIndexConfig {
            endpoint: "http://127.0.0.1:1".to_string(),
            query_deadline: DEFAULT_QUERY_DEADLINE,
            max_inflight: 1,
        })
        .expect("valid endpoint");

        let permit = index.try_acquire_prefix_query().unwrap();
        assert_eq!(
            index.try_acquire_prefix_query().unwrap_err(),
            PrefixIndexError::Overloaded
        );

        drop(permit);
        assert!(index.try_acquire_prefix_query().is_ok());
    }

    #[tokio::test]
    async fn fleet_query_fails_over_to_next_ready_indexer() {
        use crate::pb::{
            ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType, TierType,
        };
        use crate::{
            server_builder, InMemoryKvIndexerBackend, IndexerStatusReport, KvIndexerBackend,
            KvIndexerService,
        };

        let backend = Arc::new(InMemoryKvIndexerBackend::new());
        backend
            .apply_external_kv_batch(ApplyExternalKvBatchRequest {
                worker_id: "w".into(),
                seq: 1,
                actions: vec![ExternalKvAction {
                    r#type: ExternalKvActionType::ActionReport as i32,
                    tier: TierType::TierHbm as i32,
                    hashes: vec![7],
                    component_masks: Vec::new(),
                    block_sizes: Vec::new(),
                }],
                worker_address: "http://worker".into(),
                cache_spec: None,
                worker_epoch: String::new(),
                enforce_sequence: false,
            })
            .await
            .unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let service_backend: Arc<dyn KvIndexerBackend> = backend;
        let server = tokio::spawn(async move {
            server_builder()
                .add_service(KvIndexerService::new(service_backend).into_server())
                .serve(addr)
                .await
                .unwrap();
        });
        tokio::time::sleep(Duration::from_millis(30)).await;

        let index = GrpcPrefixIndex::new(PrefixIndexConfig {
            endpoint: "http://127.0.0.1:1".into(),
            query_deadline: Duration::from_millis(50),
            max_inflight: 4,
        })
        .unwrap();
        let registry = index.status_registry();
        registry
            .record(IndexerStatusReport {
                indexer_id: "dead".into(),
                endpoint: "http://127.0.0.1:1".into(),
                ready: true,
                normalized_load: 0.0,
                ready_workers: 1,
                total_workers: 1,
            })
            .unwrap();
        registry
            .record(IndexerStatusReport {
                indexer_id: "healthy".into(),
                endpoint: format!("http://{addr}"),
                ready: true,
                normalized_load: 0.5,
                ready_workers: 1,
                total_workers: 1,
            })
            .unwrap();

        let outcome = index.match_prefix(vec![7]).await.unwrap();
        assert!(matches!(
            outcome,
            PrefixOutcome::Matched {
                best_prefix_blocks: 1,
                ..
            }
        ));
        server.abort();
    }
}
