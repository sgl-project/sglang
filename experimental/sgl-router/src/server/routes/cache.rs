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
    /// nodes, so serving it a full export is the dominant cost of the
    /// bootstrap path on a large fleet. Named to match
    /// [`crate::policies::kv_events::bootstrap::CURSORS_ONLY_PARAM`].
    ///
    /// An older router image does not send this and is unaffected; an older
    /// PRODUCER ignores it and answers with a full snapshot, so a
    /// mixed-version fleet pays the old transfer cost instead of failing. Note
    /// that an old producer's cursor table is NARROWER than this path's — the
    /// full export filters it to ranks still carrying tree nodes (see
    /// `KvEventIndex::peer_cursors_body`) — so it loses exactly the witnesses
    /// the new path would uniquely know, never ones an old fleet could report.
    cursors_only: Option<bool>,
}

/// `GET /internal/kv_snapshot` — serve this replica's cache-aware tree so a
/// newly started sibling can bootstrap from it instead of routing cache-blind.
///
/// `404 NOT_FOUND` when this router maintains no local tree (cache-aware
/// KV indexing disabled, or an external Indexer is the routing signal). The
/// consumer treats that identically to an unreachable peer, which is also what
/// an older router image returns for an unknown path — so a mixed-version
/// fleet degrades to cold boots rather than errors.
///
/// The body always reports `producer_ready`, so a peer that has nothing worth
/// copying is skipped by the consumer rather than propagating a cold tree.
/// Snapshot construction is single-flighted and briefly cached; see
/// [`crate::policies::kv_events::KvEventIndex::peer_snapshot_body`].
///
/// `?max_age_ms=N` states how stale an export the caller can use, which is a
/// correctness input for a bootstrapping consumer rather than a preference —
/// see [`PRODUCER_CACHE_TTL`]. Omitting it accepts whatever is cached within
/// that default, which is what an older router image does. A splice probe
/// sends neither: `?cursors_only=true` is answered from the live cursor map
/// and never goes near the export cache.
///
/// # Exposure
///
/// Unauthenticated, on the main listener, like `/flush_cache` — the router has
/// no auth middleware, so reachability is already the trust boundary for its
/// admin surface. The body is block hashes and worker URLs: no prompt text and
/// no token ids.
pub async fn kv_snapshot(
    State(ctx): State<Arc<AppContext>>,
    Query(params): Query<SnapshotParams>,
) -> Response {
    let Some(index) = ctx.kv_index.as_ref() else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "cache-aware KV indexing is not enabled on this router",
            })),
        )
            .into_response();
    };
    if params.cursors_only.unwrap_or(false) {
        // No `max_age_ms` negotiation on this path: the cursors are read live,
        // so the answer beats any freshness a caller could state. Both
        // parameters sent → the live read wins, silently. (`?cursors_only`
        // with no value never reaches here: serde rejects a valueless bool as
        // a 400 — the probe always sends `=true`.)
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
        // The encode failed (see `peer_snapshot_body`). Must NOT be a 200: the
        // consumer would fail to decode it, and a decode failure is
        // indistinguishable from a peer whose transport is broken. A
        // non-success status reads as "no snapshot here", which is what this
        // is, and earns the consumer's per-peer cooldown rather than turning a
        // booting sibling into a retry loop against a multi-megabyte body.
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
            })
            .expect("prefill accepted");
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("d".into()),
                url: d_url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: None,
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

    // -----------------------------------------------------------------------
    // GET /internal/kv_snapshot
    // -----------------------------------------------------------------------

    use crate::policies::kv_events::bootstrap::{
        PeerSnapshot, CURSORS_ONLY_PARAM, MAX_AGE_PARAM, SNAPSHOT_PATH,
    };
    use crate::policies::kv_events::{KvEventIndex, KvWorkerId};
    use axum::http::header;

    /// An `AppContext` whose snapshot route is live, holding one block for one
    /// rank and a cursor for a second rank it no longer carries — the shape
    /// that makes the cursors-only table a strict superset of the export's.
    fn ctx_with_seeded_index() -> Arc<AppContext> {
        let index = KvEventIndex::new();
        index.block_size_oracle().try_set(64).unwrap();
        index.block_size_oracle().set_bigram(false);
        index.seed_stored_block_for_test(&KvWorkerId::new("http://carrier:30000".into(), 0), 41, 7);
        index.seed_cursor_only_for_test(&KvWorkerId::new("http://witness:30000".into(), 0), 12);
        let mut ctx = AppContext::stub();
        ctx.kv_index = index.snapshot_source();
        Arc::new(ctx)
    }

    async fn get_snapshot(ctx: Arc<AppContext>, uri: &str) -> Response {
        crate::server::app::build_router(ctx)
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    async fn parse_snapshot(resp: Response) -> PeerSnapshot {
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body).expect("snapshot body must be parseable")
    }

    #[tokio::test]
    async fn kv_snapshot_route_serves_parseable_json() {
        let resp = get_snapshot(ctx_with_seeded_index(), SNAPSHOT_PATH).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let snap = parse_snapshot(resp).await;
        assert_eq!(snap.format, 1);
        assert_eq!(snap.block_size, 64);
        assert!(snap.producer_ready, "a seeded tree is worth copying");
        assert_eq!(snap.nodes.len(), 1);
        // Only the carrier appears: the full export's cursor table is filtered
        // to ranks a graft recipient could actually get blocks from.
        assert_eq!(snap.workers.len(), 1);
        assert_eq!(snap.workers[0].url, "http://carrier:30000");
        assert_eq!(snap.cursors, vec![(0, 41)]);
    }

    /// A router that maintains no local tree must answer 404, not an empty
    /// snapshot: a consumer reads 404 the same way it reads an unreachable
    /// peer, where an empty 200 would have to be distinguished from a peer
    /// whose tree is genuinely empty.
    #[tokio::test]
    async fn kv_snapshot_route_is_absent_without_a_local_tree() {
        let resp = get_snapshot(Arc::new(AppContext::stub()), SNAPSHOT_PATH).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// Compression is the whole point of asking for gzip on the fetch side,
    /// and it is route-scoped, so it has to be asserted on THIS route: a test
    /// that mounts its own `CompressionLayer` proves tower-http works, not
    /// that this endpoint is wired to it. Both directions matter — a consumer
    /// that asks gets gzip, and one that does not (an image predating the
    /// layer) still gets a body it can parse.
    #[tokio::test]
    async fn kv_snapshot_route_compresses_only_when_the_caller_accepts_gzip() {
        use std::io::Read;

        let ctx = ctx_with_seeded_index();
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
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut inflated)
            .expect("body must be valid gzip");
        let _: PeerSnapshot = serde_json::from_slice(&inflated).unwrap();

        let plain = get_snapshot(ctx, SNAPSHOT_PATH).await;
        assert_eq!(plain.status(), StatusCode::OK);
        assert!(
            plain.headers().get(header::CONTENT_ENCODING).is_none(),
            "a caller that never asked for gzip must get identity",
        );
        let _ = parse_snapshot(plain).await;
    }

    /// `max_age_ms` is a correctness input for a bootstrapping consumer, and
    /// an older image omits it entirely. Every shape must be served.
    #[tokio::test]
    async fn kv_snapshot_route_accepts_a_max_age_and_survives_its_absence() {
        for uri in [
            SNAPSHOT_PATH.to_string(),
            format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=0"),
            format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=30000"),
        ] {
            let resp = get_snapshot(ctx_with_seeded_index(), &uri).await;
            assert_eq!(resp.status(), StatusCode::OK, "{uri}");
            let snap = parse_snapshot(resp).await;
            assert_eq!(snap.nodes.len(), 1, "{uri}");
        }
    }

    /// The probe path: cursors for every observed rank, and no tree. The
    /// witness rank is present here and absent from the full export, which is
    /// the asymmetry the two tables exist for.
    #[tokio::test]
    async fn kv_snapshot_route_serves_cursors_only_when_asked() {
        let resp = get_snapshot(
            ctx_with_seeded_index(),
            &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let snap = parse_snapshot(resp).await;

        assert!(
            snap.nodes.is_empty(),
            "a cursors-only body must be ungraftable by construction",
        );
        let mut seen: Vec<(String, i64)> = snap
            .cursors
            .iter()
            .map(|&(i, seq)| (snap.workers[i as usize].url.clone(), seq))
            .collect();
        seen.sort();
        assert_eq!(
            seen,
            vec![
                ("http://carrier:30000".to_string(), 41),
                ("http://witness:30000".to_string(), 12),
            ],
            "every observed rank is a witness, carrier or not",
        );
        assert!(snap.producer_ready);
    }

    /// `?cursors_only=true` reads live, so it must not populate or consume the
    /// export cache — otherwise a probe would either serve a stale answer or
    /// poison the next bootstrap fetch with a node-less body.
    #[tokio::test]
    async fn cursors_only_neither_fills_nor_reads_the_export_cache() {
        let ctx = ctx_with_seeded_index();
        let cursors = get_snapshot(
            Arc::clone(&ctx),
            &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        )
        .await;
        assert!(parse_snapshot(cursors).await.nodes.is_empty());

        // A generous max_age would happily reuse a cached entry, so a full
        // export right after the probe proves the probe left none behind.
        let full = get_snapshot(ctx, &format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=600000")).await;
        assert_eq!(parse_snapshot(full).await.nodes.len(), 1);
    }

    /// An index that has not established both halves of its hashing config
    /// must not advertise itself as a source: the recipient would graft blocks
    /// hashed under an identity the body misreports, and never match one.
    #[tokio::test]
    async fn a_half_published_hash_config_is_not_a_bootstrap_source() {
        let index = KvEventIndex::new();
        index.block_size_oracle().try_set(64).unwrap(); // no bigram report yet
        index.seed_stored_block_for_test(&KvWorkerId::new("http://carrier:30000".into(), 0), 1, 7);
        let mut ctx = AppContext::stub();
        ctx.kv_index = index.snapshot_source();

        let snap = parse_snapshot(get_snapshot(Arc::new(ctx), SNAPSHOT_PATH).await).await;
        assert!(
            !snap.producer_ready,
            "a half-published hash config must not be advertised as copyable",
        );
        assert!(!snap.nodes.is_empty(), "the tree itself is still reported");
    }

    /// An empty tree is not a source either, which is what keeps two replicas
    /// in a rolling update from bootstrapping off each other and both
    /// inheriting nothing.
    #[tokio::test]
    async fn an_empty_tree_is_not_a_bootstrap_source() {
        let index = KvEventIndex::new();
        index.block_size_oracle().try_set(64).unwrap();
        index.block_size_oracle().set_bigram(false);
        let mut ctx = AppContext::stub();
        ctx.kv_index = index.snapshot_source();

        let snap = parse_snapshot(get_snapshot(Arc::new(ctx), SNAPSHOT_PATH).await).await;
        assert!(!snap.producer_ready);
        assert!(snap.nodes.is_empty());
    }
}
