// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-management admin endpoints.

use crate::server::app_context::AppContext;
use crate::state::kv_events::bootstrap::PRODUCER_CACHE_TTL;
use crate::workers::worker::Worker;
use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
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
/// Every field optional, so a caller that sends none is served.
#[derive(Debug, Default, Deserialize)]
pub struct SnapshotParams {
    /// Oldest export the caller can use, in milliseconds. Named to match
    /// [`crate::state::kv_events::bootstrap::MAX_AGE_PARAM`].
    max_age_ms: Option<u64>,
    /// When true, answer with the live cursor table and no tree. Named to
    /// match [`crate::state::kv_events::bootstrap::CURSORS_ONLY_PARAM`].
    #[serde(default)]
    cursors_only: bool,
}

/// `GET /internal/kv_snapshot` — serve this replica's KV tree so a newly
/// started sibling can bootstrap from it instead of routing cache-blind.
///
/// `404 NOT_FOUND` when [`AppContext::kv_index`] is `None`. Construction is
/// single-flighted and cached; see
/// [`crate::state::kv_events::KvEventIndex::peer_snapshot_body`]. The full
/// export is served gzip-encoded when `Accept-Encoding` lists `gzip`, and as
/// identity JSON otherwise.
///
/// `?max_age_ms=N` bounds how stale a cached export may be; omitted,
/// [`PRODUCER_CACHE_TTL`] applies. `?cursors_only=true` is answered from the
/// live cursor map and bypasses the export cache.
///
/// # Exposure
///
/// Unauthenticated on the main listener, like `/flush_cache`. The body is
/// block hashes and worker URLs: no prompt text and no token ids.
pub async fn kv_snapshot(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Query(params): Query<SnapshotParams>,
) -> Response {
    let Some(index) = ctx.kv_index.as_ref() else {
        return json_error(
            StatusCode::NOT_FOUND,
            "cache-aware KV indexing is not enabled on this router",
        );
    };
    if params.cursors_only {
        // Read live, so there is nothing for `max_age_ms` to bound; it is
        // ignored when both are sent.
        let body = index.peer_cursors_body();
        if body.is_empty() {
            // Same shape as the full export's failure answer below: a 200
            // carrying an empty body would hand the caller JSON it cannot
            // decode. Unreachable in practice (the body is a plain struct),
            // kept so the two paths cannot drift apart.
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "cursor table could not be encoded",
            );
        }
        return json_ok(body);
    }
    let max_age = params
        .max_age_ms
        .map_or(PRODUCER_CACHE_TTL, Duration::from_millis);
    // Pre-encoded (both encodings) and cached by the producer; handing `Bytes`
    // to the body is a refcount bump, not a copy.
    let body = index.peer_snapshot_body(max_age).await;
    if body.is_empty() {
        // The encode failed. A 200 would carry a body the caller cannot
        // decode; a non-success status reads as "no snapshot here".
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "snapshot could not be encoded",
        );
    }
    let mut resp = if accepts_gzip(&headers) {
        let mut resp = json_ok(body.gzip);
        resp.headers_mut()
            .insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        resp
    } else {
        json_ok(body.identity)
    };
    resp.headers_mut()
        .insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    resp
}

/// Whether `Accept-Encoding` lists the `gzip` coding without refusing it.
/// Tokens match case-insensitively; a `q=0` weight is a refusal, and other
/// weights are not ranked.
fn accepts_gzip(headers: &HeaderMap) -> bool {
    headers
        .get_all(header::ACCEPT_ENCODING)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .any(|coding| {
            let mut parts = coding.split(';');
            let is_gzip = parts
                .next()
                .is_some_and(|name| name.trim().eq_ignore_ascii_case("gzip"));
            let refused = parts.any(|param| {
                param
                    .trim()
                    .strip_prefix("q=")
                    .and_then(|q| q.trim().parse::<f32>().ok())
                    .is_some_and(|q| q == 0.0)
            });
            is_gzip && !refused
        })
}

/// A `200` carrying an already-encoded JSON body.
fn json_ok(body: Bytes) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        axum::body::Body::from(body),
    )
        .into_response()
}

/// A `{"error": message}` JSON response.
fn json_error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
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
/// The caller's `Authorization` header is forwarded, so engines started with
/// `--api-key` flush only for a caller holding that key.
///
/// Status: `200 OK` when every worker flushed successfully (or the fleet is
/// empty); `502 BAD_GATEWAY` when at least one worker failed. The JSON body
/// always carries the full breakdown so a partial failure is actionable.
pub async fn flush_cache(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Response {
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
        headers.get(header::AUTHORIZATION),
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
    auth: Option<&HeaderValue>,
) -> (Vec<String>, Vec<FailedWorker>) {
    // Snapshot the URLs into owned Strings up front so the per-worker stream
    // does not borrow the `workers` slice across the await points.
    let urls: Vec<String> = workers.iter().map(|w| w.url.clone()).collect();

    let outcomes = stream::iter(urls)
        .map(|url| {
            let mut request = client
                .post(format!("{}/flush_cache", url.trim_end_matches('/')))
                .timeout(timeout);
            if let Some(auth) = auth {
                request = request.header(header::AUTHORIZATION, auth);
            }
            async move { (url, request.send().await) }
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
                    ..Default::default()
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

    /// The caller's `Authorization` reaches workers started with `--api-key`.
    #[tokio::test]
    async fn forwards_caller_authorization() {
        let app = Router::new().route(
            "/flush_cache",
            post(|headers: HeaderMap| async move {
                match headers.get(header::AUTHORIZATION) {
                    Some(v) if v == "Bearer k" => StatusCode::OK,
                    _ => StatusCode::UNAUTHORIZED,
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await });

        let keyed = Request::post("/flush_cache")
            .header(header::AUTHORIZATION, "Bearer k")
            .body(Body::empty())
            .unwrap();
        let res = crate::server::app::build_router(ctx_with_workers(&[&url]))
            .oneshot(keyed)
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let (status, _) = post_flush(ctx_with_workers(&[&url])).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
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
                ..Default::default()
            })
            .expect("prefill accepted");
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("d".into()),
                url: d_url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("stub-model".into())],
                ..Default::default()
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

    use crate::state::kv_events::bootstrap::{
        PeerSnapshot, CURSORS_ONLY_PARAM, MAX_AGE_PARAM, SNAPSHOT_PATH,
    };
    use crate::state::kv_events::{KvEventIndex, KvWorkerId};
    use axum::http::header;

    /// An `AppContext` serving `index`, with a 64-token unigram hash config
    /// established.
    fn ctx_for(index: &Arc<KvEventIndex>) -> Arc<AppContext> {
        index.block_size_oracle().try_set(64).unwrap();
        index.block_size_oracle().set_bigram(false);
        let mut ctx = AppContext::stub();
        ctx.kv_index = index.snapshot_source();
        Arc::new(ctx)
    }

    /// One block on a carrier rank, plus a cursor for a witness rank that
    /// carries none, so the cursors-only and full cursor tables differ.
    fn ctx_with_seeded_index() -> Arc<AppContext> {
        let index = KvEventIndex::new();
        index.seed_stored_block_for_test(&KvWorkerId::new("http://carrier:30000".into(), 0), 41, 7);
        index.seed_cursor_only_for_test(&KvWorkerId::new("http://witness:30000".into(), 0), 12);
        ctx_for(&index)
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
        // The full export lists only carriers.
        assert_eq!(snap.workers.len(), 1);
        assert_eq!(snap.workers[0].url, "http://carrier:30000");
        assert_eq!(snap.cursors, vec![(0, 41)]);
    }

    /// No local index means 404, not an empty snapshot.
    #[tokio::test]
    async fn kv_snapshot_route_is_absent_without_a_local_tree() {
        let resp = get_snapshot(Arc::new(AppContext::stub()), SNAPSHOT_PATH).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// Asserted through `build_router`, so it covers this route's wiring: gzip
    /// for a caller that accepts it, identity otherwise, and the gzip inflates
    /// to the identity body byte for byte.
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
        assert_eq!(
            gzipped
                .headers()
                .get(header::VARY)
                .and_then(|v| v.to_str().ok()),
            Some("accept-encoding"),
        );
        let compressed = gzipped.into_body().collect().await.unwrap().to_bytes();
        let mut inflated = Vec::new();
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut inflated)
            .expect("body must be valid gzip");

        let plain = get_snapshot(ctx, SNAPSHOT_PATH).await;
        assert_eq!(plain.status(), StatusCode::OK);
        assert!(
            plain.headers().get(header::CONTENT_ENCODING).is_none(),
            "a caller that never asked for gzip must get identity",
        );
        let identity = plain.into_body().collect().await.unwrap().to_bytes();
        let _: PeerSnapshot = serde_json::from_slice(&identity).unwrap();
        assert_eq!(inflated, identity, "gzip must inflate to the identity body");
    }

    #[test]
    fn accepts_gzip_matches_the_coding_token() {
        let accepts = |value: &str| {
            let mut headers = HeaderMap::new();
            headers.insert(header::ACCEPT_ENCODING, value.parse().unwrap());
            accepts_gzip(&headers)
        };
        assert!(accepts("gzip"));
        assert!(accepts("GZip"));
        assert!(accepts("br, gzip;q=0.8, deflate"));
        assert!(!accepts("identity"));
        assert!(!accepts("br, deflate"));
        assert!(!accepts("x-gzip-ish"));
        assert!(!accepts("gzip;q=0"));
        assert!(!accepts("br, gzip; q=0.0"));
        assert!(!accepts_gzip(&HeaderMap::new()));
    }

    /// `max_age_ms` is optional; every shape is served.
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

    /// A valueless `?cursors_only` is not a bool, so it is rejected rather
    /// than read as either answer.
    #[tokio::test]
    async fn a_valueless_cursors_only_is_rejected() {
        let resp = get_snapshot(
            ctx_with_seeded_index(),
            &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// `?cursors_only=true`: cursors for every observed rank, and no tree.
    #[tokio::test]
    async fn kv_snapshot_route_serves_cursors_only_when_asked() {
        let resp = get_snapshot(
            ctx_with_seeded_index(),
            &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let snap = parse_snapshot(resp).await;

        assert!(snap.nodes.is_empty(), "a cursors-only body carries no tree");
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
            "every observed rank is reported, carrier or not",
        );
        assert!(snap.producer_ready);
    }

    /// `?cursors_only=true` reads live: it neither fills nor reads the export
    /// cache.
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
        let full = get_snapshot(
            Arc::clone(&ctx),
            &format!("{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=600000"),
        )
        .await;
        assert_eq!(parse_snapshot(full).await.nodes.len(), 1);

        // The export cache is now warm. A rank first seen after it was filled
        // must still appear in the next probe, which it cannot if the probe
        // answered from the cached export.
        ctx.kv_index
            .as_ref()
            .unwrap()
            .seed_cursor_only_for_test(&KvWorkerId::new("http://late:30000".into(), 0), 5);
        let probe = parse_snapshot(
            get_snapshot(ctx, &format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true")).await,
        )
        .await;
        assert!(
            probe
                .cursors
                .iter()
                .any(|&(i, seq)| probe.workers[i as usize].url == "http://late:30000" && seq == 5),
            "a cursors-only probe must read the live map, not the cached export",
        );
    }

    /// A half-published hash config yields `producer_ready: false`.
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

    /// An empty tree yields `producer_ready: false`.
    #[tokio::test]
    async fn an_empty_tree_is_not_a_bootstrap_source() {
        let ctx = ctx_for(&KvEventIndex::new());
        let snap = parse_snapshot(get_snapshot(ctx, SNAPSHOT_PATH).await).await;
        assert!(!snap.producer_ready);
        assert!(snap.nodes.is_empty());
    }
}
