// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-facing client for the prefix-match query.
//!
//! A successful query distinguishes a real match from an empty result. Transport
//! failures, deadlines, and server rejections stay distinct errors so the caller
//! chooses between degrading and failing the request, instead of silently using
//! a different signal.
//!
//! # Several endpoints
//!
//! Indexer state can be shared (see the Valkey backend), so a deployment can run
//! more than one interchangeable server. Given several endpoints this client
//! keeps querying one of them and moves to the next only when the current one
//! cannot answer: unreachable, shedding load, or out of time. The endpoint that
//! answered becomes the preferred one, so an outage costs one failover rather
//! than a probe per query. A rejection is not a failover: every server would
//! reject the same request, and retrying it elsewhere only spends the caller's
//! deadline.
//!
//! [`PrefixIndexConfig::query_deadline`] is the budget for the whole query
//! including failovers, so adding endpoints cannot make a slow query slower.
//! Within it, every endpoint still to be tried gets an equal share: a preferred
//! endpoint that hangs must not be able to spend the whole deadline and leave
//! nothing for a healthy one, which is the failure a black-holed host produces.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{Semaphore, SemaphorePermit};
use tonic::transport::{Channel, Endpoint};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::MatchExternalKvPrefixRequest;
use crate::service::MAX_GRPC_DECODING_MESSAGE_SIZE;

/// Default per-query deadline. Indexer failures are request failures, so this
/// absorbs normal cross-host jitter without stalling a request indefinitely.
pub const DEFAULT_QUERY_DEADLINE: Duration = Duration::from_millis(100);
/// Default process-local bound on prefix-query RPCs issued by one client.
pub const DEFAULT_QUERY_MAX_INFLIGHT: usize = 32;
// Leave room for the packed field tag, length prefix, and future scalar fields.
const PREFIX_QUERY_ENCODING_HEADROOM: usize = 16;
const MAX_PREFIX_HASHES_PER_QUERY: usize =
    (MAX_GRPC_DECODING_MESSAGE_SIZE - PREFIX_QUERY_ENCODING_HEADROOM) / 8;
/// Below this, an attempt cannot finish a connect plus a round trip, so the
/// remaining budget is better reported as a timeout than spent on a doomed try.
const MIN_ATTEMPT_BUDGET: Duration = Duration::from_millis(5);

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
    /// Interchangeable indexer endpoints (`http://10.0.0.1:50051`) in preference
    /// order. They must share state, or the answer changes per query.
    pub endpoints: Vec<String>,
    /// Budget for one query, failovers included.
    pub query_deadline: Duration,
    /// Maximum prefix-query RPCs issued concurrently by this client.
    pub max_inflight: usize,
}

impl PrefixIndexConfig {
    /// One endpoint, with the default query deadline ([`DEFAULT_QUERY_DEADLINE`]).
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self::with_endpoints(vec![endpoint.into()])
    }

    /// Several interchangeable endpoints, tried in order from the preferred one.
    pub fn with_endpoints(endpoints: Vec<String>) -> Self {
        Self {
            endpoints,
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

struct IndexerEndpoint {
    /// As configured, for logs.
    url: String,
    channel: Channel,
}

/// tonic-backed [`PrefixIndex`] over one or more interchangeable endpoints,
/// each with a lazily-established connection.
pub struct GrpcPrefixIndex {
    endpoints: Vec<IndexerEndpoint>,
    /// Index into `endpoints` to try first; the last one that answered.
    preferred: AtomicUsize,
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
        if config.endpoints.is_empty() {
            return Err(InvalidEndpoint {
                endpoint: String::new(),
                reason: "no endpoint configured",
            });
        }
        let endpoints = config
            .endpoints
            .iter()
            .map(|url| {
                Ok(IndexerEndpoint {
                    url: url.clone(),
                    channel: parse_endpoint(url)?.connect_lazy(),
                })
            })
            .collect::<Result<Vec<_>, InvalidEndpoint>>()?;
        Ok(Self {
            endpoints,
            preferred: AtomicUsize::new(0),
            deadline: config.query_deadline,
            prefix_query_semaphore: Semaphore::new(config.max_inflight),
        })
    }

    fn try_acquire_prefix_query(&self) -> Result<SemaphorePermit<'_>, PrefixIndexError> {
        self.prefix_query_semaphore
            .try_acquire()
            .map_err(|_| PrefixIndexError::Overloaded)
    }

    /// One attempt against one endpoint, bounded by `budget`.
    async fn query_endpoint(
        &self,
        index: usize,
        hashes: Vec<i64>,
        budget: Duration,
    ) -> Result<PrefixOutcome, PrefixIndexError> {
        let mut client = KvIndexerClient::new(self.endpoints[index].channel.clone());
        let mut request = tonic::Request::new(MatchExternalKvPrefixRequest {
            hashes,
            // The policy keeps the full query length as its denominator, so a
            // transport-limited prefix cannot become a perfect hit.
            max_blocks: 0,
        });
        // On the wire so the indexer can drop a query this caller stopped waiting
        // for; the local timeout below stays the hard stop.
        request.set_timeout(budget);

        match tokio::time::timeout(budget, client.match_external_kv_prefix(request)).await {
            Err(_) => Err(PrefixIndexError::Timeout),
            Ok(Err(status)) => Err(classify(&status)),
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

/// Whether another endpoint could answer what this one could not. A rejection or
/// an oversized query is a property of the request, so every server repeats it.
fn worth_failing_over(error: &PrefixIndexError) -> bool {
    match error {
        PrefixIndexError::Unreachable
        | PrefixIndexError::Timeout
        // Inside the query loop this is the server shedding load; the client's
        // own admission limit is checked before any endpoint is tried.
        | PrefixIndexError::Overloaded => true,
        PrefixIndexError::QueryTooLarge | PrefixIndexError::Rejected(_) => false,
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

        let started = Instant::now();
        let first = self.preferred.load(Ordering::Relaxed) % self.endpoints.len();
        let mut last_error = None;
        for offset in 0..self.endpoints.len() {
            let index = (first + offset) % self.endpoints.len();
            let Some(remaining) = self.deadline.checked_sub(started.elapsed()) else {
                break;
            };
            // Fair share of what is left, so one hung endpoint cannot consume the
            // deadline; the last endpoint to try inherits the whole remainder.
            let still_to_try = (self.endpoints.len() - offset) as u32;
            let budget = if still_to_try > 1 {
                remaining / still_to_try
            } else {
                remaining
            };
            if budget < MIN_ATTEMPT_BUDGET {
                break;
            }
            // Each attempt needs its own copy: the request takes the hashes, and
            // only a failover pays for the clone.
            match self.query_endpoint(index, hashes.clone(), budget).await {
                Ok(outcome) => {
                    if index != first {
                        // Sticky, so an outage costs one failover rather than a
                        // probe of the dead endpoint on every later query.
                        self.preferred.store(index, Ordering::Relaxed);
                        tracing::warn!(
                            from = %self.endpoints[first].url,
                            to = %self.endpoints[index].url,
                            "KV Indexer failover: queries now prefer another endpoint"
                        );
                    }
                    return Ok(outcome);
                }
                Err(error) => {
                    if !worth_failing_over(&error) {
                        return Err(error);
                    }
                    tracing::debug!(
                        endpoint = %self.endpoints[index].url,
                        %error,
                        "KV Indexer endpoint could not answer; trying the next one"
                    );
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or(PrefixIndexError::Timeout))
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

fn classify(status: &tonic::Status) -> PrefixIndexError {
    let code = status.code();
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
        // A connection lost mid-request arrives as UNKNOWN, INTERNAL or ABORTED,
        // which the peer could equally have sent, so the transport error decides.
        _ if from_transport(status) => PrefixIndexError::Unreachable,
        _ => PrefixIndexError::Rejected(code),
    }
}

/// Whether this process built the status from a transport error rather than
/// reading it off a peer's trailers, which never carry one.
fn from_transport(status: &tonic::Status) -> bool {
    std::error::Error::source(status).is_some()
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;

    #[test]
    fn classifies_resource_exhausted_as_overload() {
        assert_eq!(
            classify(&tonic::Status::new(tonic::Code::ResourceExhausted, "")),
            PrefixIndexError::Overloaded
        );
    }

    /// An over-limit query must stay distinguishable from a contract rejection:
    /// the caller degrades on the former and fails the request on the latter.
    #[test]
    fn classifies_over_limit_message_as_too_large() {
        assert_eq!(
            classify(&tonic::Status::new(tonic::Code::OutOfRange, "")),
            PrefixIndexError::QueryTooLarge
        );
        assert_eq!(
            classify(&tonic::Status::new(tonic::Code::InvalidArgument, "")),
            PrefixIndexError::Rejected(tonic::Code::InvalidArgument)
        );
    }

    /// A server killed mid-request is reported as UNKNOWN or INTERNAL, the same
    /// codes a server can send itself, so the endpoint list must still fail over.
    #[test]
    fn a_broken_connection_is_transient_but_a_served_error_is_not() {
        let broken = tonic::Status::from_error(Box::new(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "connection reset by peer",
        )));
        let error = classify(&broken);
        assert_eq!(error, PrefixIndexError::Unreachable, "{broken:?}");
        assert!(worth_failing_over(&error));

        let served = tonic::Status::internal("the backend is down");
        assert_eq!(
            classify(&served),
            PrefixIndexError::Rejected(tonic::Code::Internal),
            "an error the peer chose to send is not another endpoint's problem"
        );
        assert!(!worth_failing_over(&classify(&served)));
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
            assert_eq!(
                classify(&tonic::Status::new(code, "")),
                PrefixIndexError::Timeout
            );
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

    /// Every configured endpoint is validated, not just the first, and an empty
    /// list is rejected. Async because endpoint parsing needs a reactor.
    #[tokio::test]
    async fn construction_validates_the_whole_endpoint_list() {
        let error = GrpcPrefixIndex::new(PrefixIndexConfig::with_endpoints(vec![
            "http://a:50051".to_string(),
            "b:50051".to_string(),
        ]))
        .map(|_| ())
        .expect_err("a bad endpoint anywhere in the list must be rejected")
        .to_string();
        assert!(error.contains("b:50051"), "error should name it: {error}");
        assert!(GrpcPrefixIndex::new(PrefixIndexConfig::with_endpoints(Vec::new())).is_err());
        assert!(GrpcPrefixIndex::new(PrefixIndexConfig::with_endpoints(vec![
            "http://a:50051".to_string(),
            "http://b:50051".to_string(),
        ]))
        .is_ok());
    }

    /// A rejection is the same from every server, so it must not spend the
    /// caller's deadline on the rest of the list.
    #[test]
    fn only_transient_failures_are_worth_another_endpoint() {
        for error in [
            PrefixIndexError::Unreachable,
            PrefixIndexError::Timeout,
            PrefixIndexError::Overloaded,
        ] {
            assert!(worth_failing_over(&error), "{error} should fail over");
        }
        for error in [
            PrefixIndexError::QueryTooLarge,
            PrefixIndexError::Rejected(tonic::Code::InvalidArgument),
        ] {
            assert!(!worth_failing_over(&error), "{error} should not fail over");
        }
    }

    /// Two dead endpoints must not cost two deadlines: the budget covers the
    /// whole query, failovers included.
    #[tokio::test]
    async fn the_deadline_bounds_the_whole_query_not_each_attempt() {
        let index = GrpcPrefixIndex::new(PrefixIndexConfig {
            // Both refuse instantly, so the loop is bounded by the budget rather
            // than by connect latency.
            endpoints: vec![
                "http://127.0.0.1:1".to_string(),
                "http://127.0.0.1:2".to_string(),
            ],
            query_deadline: Duration::from_millis(300),
            max_inflight: 4,
        })
        .unwrap_or_else(|error| panic!("valid endpoints: {error}"));

        let started = Instant::now();
        let error = index.match_prefix(vec![1, 2, 3]).await.unwrap_err();
        let elapsed = started.elapsed();
        assert!(worth_failing_over(&error), "unexpected error: {error}");
        assert!(
            elapsed < Duration::from_millis(600),
            "the query took {elapsed:?}, which is more than one deadline"
        );
    }

    #[tokio::test]
    async fn local_admission_rejects_without_queueing() {
        let index = GrpcPrefixIndex::new(PrefixIndexConfig {
            endpoints: vec!["http://127.0.0.1:1".to_string()],
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
}
