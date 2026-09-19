// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Given several interchangeable endpoints, the router's client must keep
//! answering while one of them is down, prefer the one that answered so an
//! outage costs a single failover, and never spread a request the server
//! rejected across the rest of the list.

#[allow(dead_code)]
#[path = "common/net.rs"]
mod test_net;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvAction,
    ExternalKvActionType, ExternalKvPrefixMatch, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, TierType,
};
use sgl_kv_indexer::{
    server_builder, GrpcPrefixIndex, InMemoryKvIndexerBackend, KvIndexerBackend, KvIndexerService,
    PrefixIndex, PrefixIndexConfig, PrefixIndexError, PrefixOutcome,
};
use test_net::bound_incoming;
use tokio::sync::oneshot;
use tonic::Status;

/// How a server behaves when asked for a prefix.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    Answer,
    /// Rejects the request the way a server that disagrees about the contract does.
    Reject,
    /// Accepts the connection and never replies: a black-holed or wedged host,
    /// which is the case a connect-refused endpoint does not exercise.
    Hang,
}

/// Counts prefix queries so a test can tell which endpoint served one.
#[derive(Clone)]
struct CountingBackend {
    inner: Arc<InMemoryKvIndexerBackend>,
    queries: Arc<AtomicUsize>,
    behaviour: Behaviour,
}

impl CountingBackend {
    fn new(behaviour: Behaviour) -> Self {
        Self {
            inner: Arc::new(InMemoryKvIndexerBackend::new()),
            queries: Arc::new(AtomicUsize::new(0)),
            behaviour,
        }
    }

    fn queries(&self) -> usize {
        self.queries.load(Ordering::SeqCst)
    }
}

#[tonic::async_trait]
impl KvIndexerBackend for CountingBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        self.inner.apply_external_kv_batch(request).await
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        self.inner.match_external_kv(request).await
    }

    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        self.queries.fetch_add(1, Ordering::SeqCst);
        match self.behaviour {
            Behaviour::Reject => Err(Status::invalid_argument("this server rejects the contract")),
            Behaviour::Hang => {
                // Long enough to outlast any deadline these tests configure.
                tokio::time::sleep(Duration::from_secs(30)).await;
                Err(Status::internal("unreachable: the caller gave up first"))
            }
            Behaviour::Answer => self.inner.match_external_kv_prefix(request).await,
        }
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        self.inner.get_external_kv_hit_counts(request).await
    }
}

struct Server {
    url: String,
    backend: CountingBackend,
    shutdown: Option<oneshot::Sender<()>>,
}

impl Server {
    /// A server holding one worker's placement for `hashes`.
    async fn start(hashes: &[i64], behaviour: Behaviour) -> Self {
        let backend = CountingBackend::new(behaviour);
        backend
            .apply_external_kv_batch(ApplyExternalKvBatchRequest {
                worker_id: "worker-0".to_string(),
                seq: 1,
                actions: vec![ExternalKvAction {
                    r#type: ExternalKvActionType::ActionReport as i32,
                    tier: TierType::TierHbm as i32,
                    hashes: hashes.to_vec(),
                    component_masks: Vec::new(),
                    block_sizes: Vec::new(),
                    parent_block_hash: None,
                }],
                worker_address: "http://worker-0:30000".to_string(),
                cache_spec: None,
            })
            .await
            .expect("seed");

        let (addr, incoming) = bound_incoming().await;
        let svc = KvIndexerService::new(backend.clone()).into_server();
        let (shutdown, rx) = oneshot::channel();
        tokio::spawn(async move {
            server_builder()
                .add_service(svc)
                .serve_with_incoming_shutdown(incoming, async move {
                    let _ = rx.await;
                })
                .await
                .expect("serve");
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while tokio::net::TcpStream::connect(addr).await.is_err() {
            assert!(Instant::now() < deadline, "server never listened on {addr}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Self {
            url: format!("http://{addr}"),
            backend,
            shutdown: Some(shutdown),
        }
    }

    /// Stops serving and waits until the port refuses connections.
    async fn stop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let addr = self.url.trim_start_matches("http://").to_string();
        let deadline = Instant::now() + Duration::from_secs(5);
        while tokio::net::TcpStream::connect(&addr).await.is_ok() {
            assert!(Instant::now() < deadline, "server kept listening on {addr}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

fn client(urls: &[&str], deadline: Duration) -> GrpcPrefixIndex {
    GrpcPrefixIndex::new(PrefixIndexConfig {
        endpoints: urls.iter().map(|url| url.to_string()).collect(),
        query_deadline: deadline,
        max_inflight: 8,
    })
    .unwrap_or_else(|error| panic!("valid endpoints: {error}"))
}

fn matched(outcome: &PrefixOutcome) -> Vec<ExternalKvPrefixMatch> {
    match outcome {
        PrefixOutcome::Matched { matches, .. } => matches
            .iter()
            .map(|m| ExternalKvPrefixMatch {
                worker_address: m.address.clone(),
                matched_prefix_blocks: m.matched_prefix_blocks,
                worker_id: m.worker_id.clone(),
            })
            .collect(),
        PrefixOutcome::Empty => Vec::new(),
    }
}

const HASHES: [i64; 3] = [1, 2, 3];

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dead_endpoint_costs_one_failover_not_a_probe_per_query() {
    let mut first = Server::start(&HASHES, Behaviour::Answer).await;
    let second = Server::start(&HASHES, Behaviour::Answer).await;
    let index = client(&[&first.url, &second.url], Duration::from_secs(2));

    // The preferred endpoint is the first configured one.
    let outcome = index.match_prefix(HASHES.to_vec()).await.expect("query");
    assert_eq!(matched(&outcome).len(), 1);
    assert_eq!((first.backend.queries(), second.backend.queries()), (1, 0));

    first.stop().await;
    for _ in 0..5 {
        let outcome = index.match_prefix(HASHES.to_vec()).await.expect("query");
        assert_eq!(
            matched(&outcome)[0].matched_prefix_blocks,
            HASHES.len() as u32,
            "the surviving endpoint must answer in full"
        );
    }
    // Five queries, one failover: the dead endpoint is not retried per query.
    assert_eq!(second.backend.queries(), 5);
    assert_eq!(first.backend.queries(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rejected_query_is_not_spread_across_the_fleet() {
    let rejecting = Server::start(&HASHES, Behaviour::Reject).await;
    let healthy = Server::start(&HASHES, Behaviour::Answer).await;
    let index = client(&[&rejecting.url, &healthy.url], Duration::from_secs(2));

    let error = index
        .match_prefix(HASHES.to_vec())
        .await
        .expect_err("a rejection must reach the caller");
    assert_eq!(
        error,
        PrefixIndexError::Rejected(tonic::Code::InvalidArgument)
    );
    assert_eq!(
        (rejecting.backend.queries(), healthy.backend.queries()),
        (1, 0),
        "the healthy endpoint must not have been asked the same rejected question"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_endpoint_down_reports_a_transient_failure_within_the_deadline() {
    let mut only = Server::start(&HASHES, Behaviour::Answer).await;
    let index = client(
        &[&only.url, "http://127.0.0.1:1"],
        Duration::from_millis(500),
    );
    index.match_prefix(HASHES.to_vec()).await.expect("query");
    only.stop().await;

    let started = Instant::now();
    let error = index
        .match_prefix(HASHES.to_vec())
        .await
        .expect_err("no endpoint can answer");
    assert!(
        matches!(
            error,
            PrefixIndexError::Unreachable | PrefixIndexError::Timeout
        ),
        "unexpected error: {error}"
    );
    assert!(
        started.elapsed() < Duration::from_millis(1500),
        "the query took {:?}, more than the whole-query budget allows",
        started.elapsed()
    );
}

/// Preference must keep moving on one client: rotating only once would pin the
/// fleet to a single survivor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preference_keeps_rotating_on_the_same_client() {
    let mut first = Server::start(&HASHES, Behaviour::Answer).await;
    let mut second = Server::start(&HASHES, Behaviour::Answer).await;
    let third = Server::start(&HASHES, Behaviour::Answer).await;
    let index = client(
        &[&first.url, &second.url, &third.url],
        Duration::from_secs(2),
    );

    index.match_prefix(HASHES.to_vec()).await.expect("first");
    assert_eq!(
        (
            first.backend.queries(),
            second.backend.queries(),
            third.backend.queries()
        ),
        (1, 0, 0)
    );

    first.stop().await;
    index.match_prefix(HASHES.to_vec()).await.expect("failover");
    // The dead endpoint was probed once more to discover it was gone.
    assert_eq!((second.backend.queries(), third.backend.queries()), (1, 0));

    // The same client must rotate again, without being rebuilt.
    second.stop().await;
    index
        .match_prefix(HASHES.to_vec())
        .await
        .expect("second failover");
    assert_eq!(third.backend.queries(), 1);

    // And stay on the survivor rather than walking the dead ones again. A stopped
    // server cannot count a query, so its counter standing still is the evidence.
    index.match_prefix(HASHES.to_vec()).await.expect("sticky");
    assert_eq!(third.backend.queries(), 2);
    assert_eq!(
        (first.backend.queries(), second.backend.queries()),
        (1, 1),
        "neither dead endpoint served anything after its failover"
    );
}

/// A preferred endpoint that accepts the connection and never answers must not
/// spend the whole deadline, or the endpoint list protects against nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hung_preferred_endpoint_still_fails_over_inside_the_deadline() {
    let hung = Server::start(&HASHES, Behaviour::Hang).await;
    let healthy = Server::start(&HASHES, Behaviour::Answer).await;
    let deadline = Duration::from_millis(600);
    let index = client(&[&hung.url, &healthy.url], deadline);

    let started = Instant::now();
    let outcome = index
        .match_prefix(HASHES.to_vec())
        .await
        .expect("the healthy endpoint must answer while the preferred one hangs");
    let elapsed = started.elapsed();

    assert_eq!(
        matched(&outcome)[0].matched_prefix_blocks,
        HASHES.len() as u32
    );
    assert_eq!(
        (hung.backend.queries(), healthy.backend.queries()),
        (1, 1),
        "both endpoints should have been asked exactly once"
    );
    assert!(
        elapsed < deadline * 2,
        "the query took {elapsed:?}, past the whole-query budget"
    );

    // And the preference moved, so the next query does not start on the hung one.
    let started = Instant::now();
    index
        .match_prefix(HASHES.to_vec())
        .await
        .expect("second query");
    assert!(
        started.elapsed() < deadline,
        "a second query should go straight to the healthy endpoint"
    );
    assert_eq!(hung.backend.queries(), 1);
}
