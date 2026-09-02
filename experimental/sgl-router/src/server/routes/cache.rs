// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-management admin endpoints.

use crate::policies::kv_events::bootstrap::PRODUCER_CACHE_TTL;
use crate::server::app_context::AppContext;
use crate::workers::worker::Worker;
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Cap on concurrent in-flight `/flush_cache` requests. Bounds how many
/// flushes are issued at once when a large fleet is flushed; the rest queue
/// and run as slots free up.
const MAX_CONCURRENT_FLUSH: usize = 32;

/// One worker that failed to flush, with a human-readable reason.
#[derive(Serialize)]
pub struct FailedWorker {
    pub worker: String,
    pub error: String,
}

/// Per-worker breakdown of a `/flush_cache` fan-out. `total_workers` is the
/// registry size snapshotted at call time; every registered worker is
/// attempted, so `successful.len() + failed.len() == total_workers`.
/// `message` is a human/log summary — the HTTP status is authoritative.
#[derive(Serialize)]
pub struct FlushCacheResult {
    pub successful: Vec<String>,
    pub failed: Vec<FailedWorker>,
    pub total_workers: usize,
    pub message: String,
}

impl FlushCacheResult {
    /// Build a result from a completed fan-out, deriving `message` from the
    /// outcome counts so the count/message coherence lives in one place
    /// rather than at each call site.
    fn from_outcomes(
        total_workers: usize,
        successful: Vec<String>,
        failed: Vec<FailedWorker>,
    ) -> Self {
        let message = if total_workers == 0 {
            "No workers registered; nothing to flush".to_string()
        } else if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} workers",
                total_workers
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };
        Self {
            successful,
            failed,
            total_workers,
            message,
        }
    }
}

/// Query string of `GET /internal/kv_snapshot`.
///
/// Every field optional, so a peer that sends nothing — an older router image,
/// or a caller with no freshness requirement — is served rather than rejected.
#[derive(Debug, Default, Deserialize)]
pub struct SnapshotParams {
    /// Oldest export the caller can use, in milliseconds. Named to match
    /// [`crate::policies::kv_events::bootstrap::MAX_AGE_PARAM`].
    max_age_ms: Option<u64>,
    /// When true, answer with the cursor table alone and omit the tree.
    ///
    /// A splice probe reads one sequence number per rank and has no use for
    /// nodes, so serving it a full export is the dominant cost of the bootstrap
    /// path on a large fleet. Named to match
    /// [`crate::policies::kv_events::bootstrap::CURSORS_ONLY_PARAM`].
    ///
    /// An older router image does not send this and is unaffected; an older
    /// PRODUCER ignores it and answers with a full snapshot, so a mixed-version
    /// fleet pays the old transfer cost instead of failing. Note its cursor
    /// table is narrower than this path's — the full export filters it to
    /// ranks still carrying tree nodes (see
    /// `KvEventIndex::peer_cursors_body`) — so an old producer loses exactly
    /// the witnesses the new path would uniquely know, never the ones an old
    /// fleet could report.
    cursors_only: Option<bool>,
}

/// `GET /internal/kv_snapshot` — serve this replica's cache-aware tree so a
/// newly started sibling can bootstrap from it instead of routing cache-blind.
///
/// `404 NOT_FOUND` when cache-aware-zmq routing is disabled (no tree to
/// share). The consumer treats that identically to an unreachable peer, which
/// is also what an older router image returns for an unknown path — so a mixed
/// -version fleet degrades to cold boots rather than errors.
///
/// The body always reports `producer_ready`, so a peer that is itself still
/// bootstrapping is skipped by the consumer rather than propagating a cold
/// tree. Snapshot construction is single-flighted and briefly cached; see
/// [`crate::policies::kv_events::index::KvEventIndex::peer_snapshot_body`].
///
/// `?max_age_ms=N` states how stale an export the caller can use, which is a
/// correctness input for a bootstrapping consumer rather than a preference —
/// see [`PRODUCER_CACHE_TTL`]. Omitting it accepts whatever is cached within
/// that default, which is what an older router image does. A splice probe sends
/// neither: `?cursors_only=true` is answered from the live cursor map and never
/// goes near the export cache.
///
/// # Exposure
///
/// Unauthenticated, on the main listener, like `/flush_cache` and the
/// pprof endpoint — the router has no auth middleware, so reachability is
/// already the trust boundary for its admin surface. The body is block hashes
/// and worker URLs: no prompt text and no token ids.
pub async fn kv_snapshot(
    State(ctx): State<Arc<AppContext>>,
    Query(params): Query<SnapshotParams>,
) -> Response {
    match ctx.kv_index() {
        Some(index) => {
            if params.cursors_only.unwrap_or(false) {
                // No `max_age_ms` negotiation on this path: the cursors are
                // read live, so the answer beats any freshness a caller could
                // state. Both parameters sent → the live read wins, silently.
                // (`?cursors_only` with no value never reaches here: serde
                // rejects a valueless bool as a 400 — the probe always sends
                // `=true`.)
                return (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    axum::body::Body::from(index.peer_cursors_body()),
                )
                    .into_response();
            }
            let max_age = params
                .max_age_ms
                .map_or(PRODUCER_CACHE_TTL, Duration::from_millis);
            // Pre-encoded and cached by the producer, so a boot herd does not
            // re-serialise one identical multi-megabyte tree per request. Handing
            // `Bytes` to the body is a refcount bump, not a copy.
            let body = index.peer_snapshot_body(max_age).await;
            if body.is_empty() {
                // The encode failed (see `snapshot_entry`). Must NOT be a 200: the
                // consumer's `fetch_snapshot` would fail to decode it, and a
                // decode failure is indistinguishable from a peer whose transport
                // is broken. A non-success status reads as "no snapshot here",
                // which is what this is. Both paths now earn the sweep's peer
                // cooldown, so neither turns a booting sibling into a 250ms retry
                // loop against a multi-megabyte body.
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({
                        "error": "snapshot could not be encoded",
                    })),
                )
                    .into_response();
            }
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                axum::body::Body::from(body),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "cache-aware KV indexing is not enabled on this router",
            })),
        )
            .into_response(),
    }
}

/// `POST /flush_cache` — fan SGLang's `/flush_cache` admin call out to every
/// registered worker and report a per-worker breakdown.
///
/// Targets the whole fleet (plain, prefill, and decode workers all hold KV
/// cache), not just one model's pool. Deliberately **bypasses the circuit
/// breaker**: an operator flushing caches wants every worker hit — including
/// ones whose breaker is open — and recording breaker success/failure for an
/// out-of-band admin call would skew the state the request router uses to
/// pick workers.
///
/// Status: `200 OK` when every worker flushed successfully (or the fleet is
/// empty); `502 BAD_GATEWAY` when at least one worker failed. The JSON body
/// always carries the full breakdown so a partial failure is actionable.
pub async fn flush_cache(State(ctx): State<Arc<AppContext>>) -> Response {
    let workers = ctx.registry.all();
    let total_workers = workers.len();

    if workers.is_empty() {
        // A flush against an empty fleet is a no-op, but it usually means a
        // discovery/config problem (the router knows of no workers), so warn
        // rather than stay silent.
        tracing::warn!("flush_cache called but no workers are registered");
        return (
            StatusCode::OK,
            Json(FlushCacheResult::from_outcomes(0, Vec::new(), Vec::new())),
        )
            .into_response();
    }

    let (successful, failed) = fan_out_flush(
        &workers,
        ctx.proxy.admin_client(),
        ctx.proxy.request_timeout,
    )
    .await;

    // Partial failure is an operational event an operator needs to see at the
    // common production log level — match the rest of the router, which warns
    // on upstream failures.
    if failed.is_empty() {
        tracing::info!(total_workers, "flush_cache: all workers flushed");
    } else {
        tracing::warn!(
            total_workers,
            succeeded = successful.len(),
            failed = failed.len(),
            "flush_cache: some workers failed to flush",
        );
    }

    let status = if failed.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::BAD_GATEWAY
    };

    (
        status,
        Json(FlushCacheResult::from_outcomes(
            total_workers,
            successful,
            failed,
        )),
    )
        .into_response()
}

/// POST `/flush_cache` to each worker concurrently (bounded by
/// [`MAX_CONCURRENT_FLUSH`]) and partition the outcomes into
/// (successful URLs, failed workers). A non-2xx status or a transport
/// error both count as failures.
async fn fan_out_flush(
    workers: &[Arc<Worker>],
    client: &Client,
    timeout: Duration,
) -> (Vec<String>, Vec<FailedWorker>) {
    // Snapshot the URLs into owned Strings up front so the per-worker stream
    // does not borrow the `workers` slice across the await points.
    let urls: Vec<String> = workers.iter().map(|w| w.url.clone()).collect();

    let outcomes = stream::iter(urls)
        .map(|url| {
            let client = client.clone();
            async move {
                let flush_url = format!("{}/flush_cache", url.trim_end_matches('/'));
                let result = client.post(&flush_url).timeout(timeout).send().await;
                (url, result)
            }
        })
        .buffer_unordered(MAX_CONCURRENT_FLUSH)
        .collect::<Vec<_>>()
        .await;

    let mut successful = Vec::new();
    let mut failed = Vec::new();
    for (url, result) in outcomes {
        match result {
            Ok(resp) if resp.status().is_success() => successful.push(url),
            Ok(resp) => failed.push(FailedWorker {
                worker: url,
                error: format!("HTTP {}", resp.status()),
            }),
            // Render the full source chain (`{:#}`), not just reqwest's outer
            // message, so a connect-refused / DNS / TLS / timeout cause is
            // visible in the per-worker error rather than collapsed away.
            Err(e) => failed.push(FailedWorker {
                worker: url,
                error: format!("{:#}", anyhow::Error::new(e)),
            }),
        }
    }
    (successful, failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::server::app_context::AppContext;
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::post;
    use axum::Router;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tower::ServiceExt;

    /// Exercise the REAL route, not a stand-in server: the body is served
    /// pre-encoded from the producer cache rather than through `Json`, so the
    /// content type and parseability are now this handler's responsibility.
    #[tokio::test]
    async fn kv_snapshot_route_serves_parseable_json() {
        use crate::policies::kv_events::bootstrap::{PeerSnapshot, SNAPSHOT_PATH};
        let ctx = Arc::new(AppContext::stub());
        ctx.attach_kv_index(crate::policies::kv_events::KvEventIndex::new());
        let app = crate::server::app::build_router(Arc::clone(&ctx));

        let res = app
            .oneshot(
                Request::builder()
                    .uri(SNAPSHOT_PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            res.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json"),
        );
        let body = res.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice::<PeerSnapshot>(&body)
            .expect("route body must deserialise as a PeerSnapshot");
    }

    /// Compression is the whole point of asking for gzip on the fetch side, and it
    /// lives on the route rather than the app — so it has to be asserted HERE,
    /// against the router `build_router` actually produces. A component test that
    /// mounts its own `CompressionLayer` proves tower-http works, not that this
    /// route is wired to it, and would stay green if the layer were dropped.
    ///
    /// Both directions in one test, because they are one contract: a consumer that
    /// asks gets gzip, and one that does not — an image predating the layer — still
    /// gets parseable JSON.
    #[tokio::test]
    async fn kv_snapshot_route_compresses_only_when_the_caller_accepts_gzip() {
        use crate::policies::kv_events::bootstrap::{PeerSnapshot, SNAPSHOT_PATH};
        use std::io::Read;

        let ctx = Arc::new(AppContext::stub());
        ctx.attach_kv_index(crate::policies::kv_events::KvEventIndex::new());

        let gzipped = crate::server::app::build_router(Arc::clone(&ctx))
            .oneshot(
                Request::builder()
                    .uri(SNAPSHOT_PATH)
                    .header(header::ACCEPT_ENCODING, "gzip")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(gzipped.status(), StatusCode::OK);
        assert_eq!(
            gzipped
                .headers()
                .get(header::CONTENT_ENCODING)
                .and_then(|v| v.to_str().ok()),
            Some("gzip"),
            "the snapshot route must compress for a caller that accepts gzip",
        );
        let compressed = gzipped.into_body().collect().await.unwrap().to_bytes();
        let mut inflated = Vec::new();
        flate2::read::GzDecoder::new(compressed.as_ref())
            .read_to_end(&mut inflated)
            .expect("body must be valid gzip");
        serde_json::from_slice::<PeerSnapshot>(&inflated)
            .expect("inflated body must deserialise as a PeerSnapshot");

        let identity = crate::server::app::build_router(Arc::clone(&ctx))
            .oneshot(
                Request::builder()
                    .uri(SNAPSHOT_PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            identity.headers().get(header::CONTENT_ENCODING).is_none(),
            "a caller that never asked for gzip must get identity",
        );

        // The layer must stay SCOPED to this route. Moving it to the app-level
        // `Router::layer` would compress every JSON response on the proxy hot
        // path, and nothing else would notice — tower-http's default predicate
        // exempts `text/event-stream`, so even the SSE stream would look fine.
        let other = crate::server::app::build_router(Arc::clone(&ctx))
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header(header::ACCEPT_ENCODING, "gzip")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            other.headers().get(header::CONTENT_ENCODING).is_none(),
            "compression must not leak off the snapshot route",
        );
        // Guard against proving that vacuously: tower-http skips bodies at or
        // below 32 bytes, so a tiny response would satisfy the assertion above
        // even under a global layer.
        let other_len = other.into_body().collect().await.unwrap().to_bytes().len();
        assert!(
            other_len > 32,
            "the comparison endpoint must be large enough for the layer to have \
             compressed it; got {other_len} bytes",
        );
    }

    /// The freshness parameter must be OPTIONAL at the extractor. A router image
    /// that predates it sends the bare path, and rejecting that would turn every
    /// mixed-version bootstrap into an unreachable peer.
    #[tokio::test]
    async fn kv_snapshot_route_accepts_a_max_age_and_survives_its_absence() {
        use crate::policies::kv_events::bootstrap::{PeerSnapshot, MAX_AGE_PARAM, SNAPSHOT_PATH};
        let ctx = Arc::new(AppContext::stub());
        ctx.attach_kv_index(crate::policies::kv_events::KvEventIndex::new());

        for uri in [
            SNAPSHOT_PATH.to_string(),
            format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=0"),
            format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=30000"),
        ] {
            let app = crate::server::app::build_router(Arc::clone(&ctx));
            let res = app
                .oneshot(Request::builder().uri(&uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{uri} must be served");
            let body = res.into_body().collect().await.unwrap().to_bytes();
            serde_json::from_slice::<PeerSnapshot>(&body)
                .unwrap_or_else(|e| panic!("{uri} must yield a PeerSnapshot: {e}"));
        }
    }

    /// The route must reach the cheap producer path, not serve a full export
    /// under a different name.
    #[tokio::test]
    async fn kv_snapshot_route_serves_cursors_only_when_asked() {
        use crate::policies::kv_events::bootstrap::{PeerSnapshot, SNAPSHOT_PATH};
        let ctx = Arc::new(AppContext::stub());
        ctx.attach_kv_index(crate::policies::kv_events::KvEventIndex::new());
        let app = crate::server::app::build_router(Arc::clone(&ctx));

        let res = app
            .oneshot(
                Request::builder()
                    // A hand-written literal on purpose, unlike the builder in
                    // the seeded test: this pins the wire contract itself, so a
                    // thoughtless parameter rename fails here.
                    .uri(format!("{SNAPSHOT_PATH}?cursors_only=true"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let snap: PeerSnapshot = serde_json::from_slice(&body).expect("valid snapshot JSON");
        assert!(snap.nodes.is_empty(), "cursors-only must omit the tree");
    }

    /// On an EMPTY index the two bodies are indistinguishable — no nodes and no
    /// cursors either way — so the test above cannot fail if the parameter is
    /// unwired. Seed one block so the full export genuinely differs, then hold
    /// both answers side by side: same witness question answered by both, tree
    /// carried by no more than one.
    #[tokio::test]
    async fn cursors_only_route_serves_the_cursor_alone_on_a_seeded_index() {
        use crate::policies::kv_events::bootstrap::{cursors_url, PeerSnapshot, SNAPSHOT_PATH};
        use crate::policies::kv_events::{BlockSizeOracle, KvWorkerId};

        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.report_worker("http://w1:30000", false);
        let index = crate::policies::kv_events::KvEventIndex::new_with_http_and_oracle(
            Client::new(),
            Arc::clone(&oracle),
        );
        index.seed_stored_block_for_test(
            &KvWorkerId::new("http://w1:30000".to_string(), 0),
            42,
            111,
        );
        let ctx = Arc::new(AppContext::stub());
        ctx.attach_kv_index(index);
        let app = crate::server::app::build_router(Arc::clone(&ctx));

        // Drive the URI through the same builder the probe uses — asserting on
        // a hand-written query string would leave the builder free to drift.
        let thin_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(cursors_url(""))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(thin_res.status(), StatusCode::OK);
        let thin_body = thin_res.into_body().collect().await.unwrap().to_bytes();
        let thin: PeerSnapshot = serde_json::from_slice(&thin_body).expect("valid snapshot JSON");

        let full_res = app
            .oneshot(
                Request::builder()
                    .uri(SNAPSHOT_PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(full_res.status(), StatusCode::OK);
        let full_body = full_res.into_body().collect().await.unwrap().to_bytes();
        let full: PeerSnapshot = serde_json::from_slice(&full_body).expect("valid snapshot JSON");

        // Both bodies answer the witness question...
        assert_eq!(thin.wire_cursor_for("http://w1:30000", 0), Some(42));
        assert_eq!(full.wire_cursor_for("http://w1:30000", 0), Some(42));
        // ...and only now, with a seeded tree, is the difference between them
        // something an assertion can actually see.
        assert!(thin.nodes.is_empty(), "cursors-only must omit the tree");
        assert!(
            !full.nodes.is_empty(),
            "the full export must carry the seeded block",
        );
    }

    /// Guards the one way this parameter could become dangerous: a cursors-only
    /// body reaching the graft path. `from_wire` must refuse it — and the
    /// refusal must come from the empty node list specifically, so the index is
    /// seeded until `producer_ready` and the block size would PASS, leaving
    /// `nodes.is_empty()` as the only reason left to fire.
    #[tokio::test]
    async fn a_cursors_only_body_can_never_be_grafted() {
        use crate::policies::kv_events::bootstrap::{
            PeerSnapshot, VetError, VettedSnapshot, SNAPSHOT_PATH,
        };
        use crate::policies::kv_events::{BlockSizeOracle, KvWorkerId};

        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.report_worker("http://w1:30000", false);
        let index = crate::policies::kv_events::KvEventIndex::new_with_http_and_oracle(
            Client::new(),
            Arc::clone(&oracle),
        );
        index.seed_stored_block_for_test(
            &KvWorkerId::new("http://w1:30000".to_string(), 0),
            42,
            111,
        );
        let ctx = Arc::new(AppContext::stub());
        ctx.attach_kv_index(index);
        let app = crate::server::app::build_router(Arc::clone(&ctx));

        let res = app
            .oneshot(
                Request::builder()
                    .uri(format!("{SNAPSHOT_PATH}?cursors_only=true"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let snap: PeerSnapshot = serde_json::from_slice(&body).unwrap();
        // Sanity: everything BUT the node list now vouches for this body, so
        // the error below can only be the empty-nodes refusal.
        assert!(snap.producer_ready);
        assert_eq!(snap.block_size, 256);
        assert!(snap.nodes.is_empty());

        let live = std::collections::HashSet::new();
        assert_eq!(
            VettedSnapshot::from_wire(snap, &live, Some(256)).unwrap_err(),
            VetError::ProducerCold,
            "an empty node list must be refused as a graft source",
        );
    }

    /// Omitting the parameter must keep the full-export behaviour intact, which
    /// is what an unpatched consumer relies on.
    #[tokio::test]
    async fn kv_snapshot_route_still_serves_a_full_snapshot_by_default() {
        use crate::policies::kv_events::bootstrap::{PeerSnapshot, SNAPSHOT_PATH};
        let ctx = Arc::new(AppContext::stub());
        ctx.attach_kv_index(crate::policies::kv_events::KvEventIndex::new());
        let app = crate::server::app::build_router(Arc::clone(&ctx));

        let res = app
            .oneshot(
                Request::builder()
                    .uri(SNAPSHOT_PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice::<PeerSnapshot>(&body).expect("valid snapshot JSON");
    }

    /// Spawn a fake worker that answers `POST /flush_cache` with `status`.
    /// Returns its base URL and a shutdown handle (drop or send to stop).
    async fn spawn_fake_flush_worker(status: StatusCode) -> (String, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route("/flush_cache", post(move || async move { status }));
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), tx)
    }

    /// Reserve a port then drop the listener so a connect attempt fails fast
    /// with ConnectionRefused (no waiting on the connect timeout).
    fn unused_port() -> u16 {
        use std::net::TcpListener;
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    }

    fn ctx_with_workers(urls: &[&str]) -> Arc<AppContext> {
        let ctx = AppContext::stub();
        for (i, url) in urls.iter().enumerate() {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(format!("w-{i}")),
                    url: (*url).to_string(),
                    mode: WorkerMode::Plain,
                    model_ids: vec![ModelId("stub-model".into())],
                    bootstrap_port: None,
                    transfer_group: None,
                })
                .expect("worker accepted");
        }
        Arc::new(ctx)
    }

    async fn post_flush(ctx: Arc<AppContext>) -> (StatusCode, Value) {
        let app = crate::server::app::build_router(ctx);
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/flush_cache")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn all_workers_succeed_returns_200() {
        let (u1, _s1) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (u2, _s2) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (status, body) = post_flush(ctx_with_workers(&[&u1, &u2])).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 2);
        assert_eq!(body["successful"].as_array().unwrap().len(), 2);
        assert!(body["failed"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn partial_failure_returns_502_with_breakdown() {
        let (ok_url, _s1) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (err_url, _s2) = spawn_fake_flush_worker(StatusCode::INTERNAL_SERVER_ERROR).await;
        let (status, body) = post_flush(ctx_with_workers(&[&ok_url, &err_url])).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["total_workers"], 2);
        assert_eq!(
            body["successful"].as_array().unwrap(),
            &vec![Value::String(ok_url.clone())]
        );
        let failed = body["failed"].as_array().unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0]["worker"], err_url);
        assert!(failed[0]["error"].as_str().unwrap().contains("500"));
    }

    #[tokio::test]
    async fn empty_registry_returns_200_with_zero_workers() {
        let (status, body) = post_flush(Arc::new(AppContext::stub())).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 0);
        assert!(body["successful"].as_array().unwrap().is_empty());
        assert!(body["failed"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unreachable_worker_is_reported_failed() {
        let url = format!("http://127.0.0.1:{}", unused_port());
        let (status, body) = post_flush(ctx_with_workers(&[&url])).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        let failed = body["failed"].as_array().unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0]["worker"], url);
    }

    /// A non-5xx, non-2xx status (e.g. 404) is still a failure and still
    /// drives the top-level 502, with the status echoed in the error.
    #[tokio::test]
    async fn non_5xx_error_status_is_reported_failed() {
        let (ok_url, _s1) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (nf_url, _s2) = spawn_fake_flush_worker(StatusCode::NOT_FOUND).await;
        let (status, body) = post_flush(ctx_with_workers(&[&ok_url, &nf_url])).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        let failed = body["failed"].as_array().unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0]["worker"], nf_url);
        assert!(failed[0]["error"].as_str().unwrap().contains("404"));
    }

    /// A worker URL with a trailing slash must still resolve to
    /// `<url>/flush_cache` (not `<url>//flush_cache`). Guards the
    /// `trim_end_matches('/')` in `fan_out_flush` against a regression that
    /// would 404 every slash-suffixed worker.
    #[tokio::test]
    async fn worker_url_with_trailing_slash_is_flushed() {
        let (base, _s) = spawn_fake_flush_worker(StatusCode::OK).await;
        let url = format!("{base}/");
        let (status, body) = post_flush(ctx_with_workers(&[&url])).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["successful"].as_array().unwrap(),
            &vec![Value::String(url.clone())]
        );
        assert!(body["failed"].as_array().unwrap().is_empty());
    }

    /// The fan-out targets the whole fleet, not one model's pool: prefill
    /// and decode workers (which also hold KV cache) must both be flushed.
    /// Asserted through the handler — `registry::all()` returning mixed modes
    /// is necessary but not sufficient if a mode filter ever slips into the
    /// handler path.
    #[tokio::test]
    async fn flushes_prefill_and_decode_workers() {
        let (p_url, _s1) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (d_url, _s2) = spawn_fake_flush_worker(StatusCode::OK).await;
        let ctx = AppContext::stub();
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("p".into()),
                url: p_url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: Some(8998),
                transfer_group: None,
            })
            .expect("prefill accepted");
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("d".into()),
                url: d_url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: None,
                transfer_group: None,
            })
            .expect("decode accepted");

        let (status, body) = post_flush(Arc::new(ctx)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 2);
        let mut succeeded: Vec<&str> = body["successful"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        succeeded.sort_unstable();
        let mut expected = [p_url.as_str(), d_url.as_str()];
        expected.sort_unstable();
        assert_eq!(succeeded, expected);
    }
}
