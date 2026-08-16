//! Per-worker KV-event publisher discovery.
//!
//! Calls the worker's `/server_info` endpoint (extended on the SGLang
//! Python side) to learn where to connect its ZMQ KV-event publisher.
//! Returns an [`EventConfig`] on success or `Ok(None)` when the worker
//! is reachable but explicitly does not run an event publisher (older
//! SGLang, `kv-events-config` unset, `null` publisher, etc.).
//!
//! # Failure semantics
//!
//! - Network errors and 5xx responses are **transient** and retried
//!   inside [`fetch_event_config`] up to [`FETCH_MAX_ATTEMPTS`] with
//!   exponential backoff. If every attempt fails, the call returns
//!   `Err(_)` so the caller can distinguish "definitely not publishing"
//!   (`Ok(None)`) from "we couldn't tell" (`Err`).
//! - 4xx responses are non-retriable (the worker answered
//!   authoritatively) and surface as `Err`.
//! - Caller behaviour: [`super::index::KvEventIndex::add_worker`] logs
//!   the error and skips subscription, but the worker remains in the
//!   broader router registry. Future re-discovery may retry.

use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use tracing::{debug, info, warn};
use url::Url;

/// Per-worker KV-event publisher configuration, resolved to something the
/// gateway can directly use to open ZMQ SUB sockets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventConfig {
    /// The host the gateway should connect to. If the worker reports a
    /// wildcard bind host (`*`, `0.0.0.0`, `::`) this is replaced by the
    /// host parsed out of the worker URL; otherwise the explicit
    /// `endpoint_host` is kept verbatim.
    pub host: String,
    /// Base port for rank 0. Per-rank port = `port_base + dp_rank`.
    pub port_base: u16,
    /// ZMQ topic prefix the gateway should SUBSCRIBE to.
    pub topic: String,
    /// Base port of the worker's dedicated load-snapshot socket range
    /// (per-rank load port = `load_port_base + dp_rank`). `None` when the
    /// worker predates load publishing — the load subscriber is then skipped
    /// and selection falls back to the router-side in-flight counter.
    pub load_port_base: Option<u16>,
    /// The size, in tokens, at which this worker actually publishes KV
    /// blocks: the advertised `page_size` widened by `dcp_size` (see
    /// `kv_event_block_size`). Callers MUST hash at this size; a
    /// mismatch produces silent miscompute since
    /// [`super::hash::compute_block_hashes`] is keyed on the caller's
    /// value, not on this one.
    pub block_size: u32,
    /// Number of attention-DP ranks publishing. The gateway opens this
    /// many SUB connections (one per rank), skipping any rank whose
    /// `port_base + dp_rank` overflows `u16`.
    pub dp_size: u32,
    /// Whether the worker uses EAGLE-family speculative decoding (EAGLE /
    /// EAGLE3 / FROZEN_KV_MTP), reported via `/server_info`'s top-level
    /// `speculative_algorithm`. When true the worker hashes KV blocks over
    /// overlapping token *bigrams* (`is_bigram = is_eagle`), so the router must
    /// use [`super::hash::compute_block_hashes_bigram`] for its query hashes to
    /// match the worker's stored hashes — otherwise cache-aware routing
    /// silently never matches and degrades to min-load.
    pub is_bigram: bool,
}

/// Default timeout for the `/server_info` introspection request. The
/// worker is on the same network as the gateway in production; 2 seconds
/// is generous and still bounds gateway-startup latency.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(2);

/// Bounded retry for transient `/server_info` failures. A worker that just
/// booted may need a few hundred ms before its HTTP server accepts
/// requests; retry absorbs the race without permanently disabling
/// cache-aware routing for that worker.
const FETCH_MAX_ATTEMPTS: u32 = 3;
const FETCH_BACKOFF_BASE: Duration = Duration::from_millis(100);

/// Fetch the worker's KV-event publisher config via `/server_info`.
///
/// Returns:
/// - `Ok(Some(cfg))` when the worker exposed a usable `kv_events` block.
/// - `Ok(None)` when the worker is **reachable** but explicitly does not
///   expose one (older SGLang, `kv-events-config` unset, `null`
///   publisher, etc.). Cache-aware routing is disabled for that worker.
/// - `Err(_)` when `worker_url` cannot be parsed, OR when every transient
///   attempt failed (network error or 5xx). Caller decides whether to
///   retry; the worker is still added to the registry but cache-aware
///   routing is disabled until a future re-discovery.
pub async fn fetch_event_config(
    worker_url: &str,
    client: &reqwest::Client,
) -> Result<Option<EventConfig>> {
    let parsed =
        Url::parse(worker_url).map_err(|e| anyhow!("invalid worker_url {worker_url}: {e}"))?;
    let worker_host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("worker_url {worker_url} has no host"))?
        .to_owned();

    let server_info_url = format!("{}/server_info", worker_url.trim_end_matches('/'));

    let body = fetch_with_retry(&server_info_url, worker_url, client).await?;

    // EAGLE-family speculative decoding ⇒ the worker hashes KV blocks over
    // token bigrams; the router must mirror that on the selection side.
    let is_bigram = classify_bigram(body.speculative_algorithm.as_deref(), worker_url);

    let block = match body.kv_events {
        Some(b) => b,
        None => {
            debug!(
                worker_url = worker_url,
                "kv-events discovery: /server_info has no kv_events block; worker is not publishing"
            );
            return Ok(None);
        }
    };

    // The worker advertises its unwidened `page_size`; under DCP it publishes
    // at `page_size * dcp_size`. Resolve the size the router must hash at.
    // `None` means no sound value exists, so fail rather than subscribe at a
    // block size that would silently match nothing.
    let block_size = kv_event_block_size(block.block_size, body.dcp_size, worker_url)
        .ok_or_else(|| anyhow!("cannot resolve a KV-event block size for {worker_url}"))?;

    // Wildcard bind hosts mean "any interface" on the worker side — the
    // gateway has to connect to a routable address, which it learns from
    // the worker URL.
    let host = if matches!(
        block.endpoint_host.as_str(),
        "*" | "0.0.0.0" | "::" | "[::]"
    ) {
        worker_host
    } else {
        block.endpoint_host
    };

    Ok(Some(EventConfig {
        host,
        port_base: block.endpoint_port_base,
        topic: block.topic,
        load_port_base: block.load_endpoint_port_base,
        block_size,
        dp_size: block.dp_size,
        is_bigram,
    }))
}

/// Issue the `/server_info` request with bounded retry on transient errors
/// (network failures, 5xx). 4xx responses and JSON-parse errors are
/// non-retriable: the worker answered, just not with what we expect.
async fn fetch_with_retry(
    server_info_url: &str,
    worker_url: &str,
    client: &reqwest::Client,
) -> Result<ServerInfoResponse> {
    let mut last_err: Option<String> = None;
    let mut delay = FETCH_BACKOFF_BASE;
    for attempt in 1..=FETCH_MAX_ATTEMPTS {
        match client
            .get(server_info_url)
            .timeout(DEFAULT_TIMEOUT)
            .send()
            .await
        {
            Err(e) => {
                last_err = Some(format!("network error: {e}"));
                warn!(
                    worker_url = worker_url,
                    attempt,
                    error = %e,
                    "kv-events discovery: /server_info request failed; will retry"
                );
            }
            Ok(resp) if resp.status().is_server_error() => {
                last_err = Some(format!("server error: {}", resp.status()));
                warn!(
                    worker_url = worker_url,
                    attempt,
                    status = resp.status().as_u16(),
                    "kv-events discovery: /server_info returned 5xx; will retry"
                );
            }
            Ok(resp) if !resp.status().is_success() => {
                // 4xx — worker answered authoritatively, retrying won't help.
                return Err(anyhow!(
                    "/server_info returned {} (non-retriable)",
                    resp.status()
                ));
            }
            Ok(resp) => {
                return resp
                    .json::<ServerInfoResponse>()
                    .await
                    .map_err(|e| anyhow!("/server_info JSON parse failed: {e}"));
            }
        }
        if attempt < FETCH_MAX_ATTEMPTS {
            tokio::time::sleep(delay).await;
            delay *= 2;
        }
    }
    Err(anyhow!(
        "/server_info failed after {} attempts: {}",
        FETCH_MAX_ATTEMPTS,
        last_err.unwrap_or_else(|| "unknown".into()),
    ))
}

#[derive(Deserialize)]
struct ServerInfoResponse {
    #[serde(default)]
    kv_events: Option<KvEventsBlock>,
    /// Top-level `/server_info` field. EAGLE-family values
    /// (EAGLE / EAGLE3 / FROZEN_KV_MTP) mean the worker hashes KV blocks over
    /// token bigrams — see [`EventConfig::is_bigram`].
    #[serde(default)]
    speculative_algorithm: Option<String>,
    /// Top-level `/server_info` field mirroring `ServerArgs.dcp_size`: the
    /// decode context-parallel size. Absent on engines that predate the
    /// field — see [`kv_event_block_size`].
    ///
    /// `/server_info` is `dataclasses.asdict(server_args)`, so this tracks the
    /// dataclass *field* name, not the `--decode-context-parallel-size` CLI
    /// alias. A Python-side rename would land here as `None` with no signal —
    /// one more reason the wire-level cross-check in [`kv_event_block_size`]
    /// is the durable check.
    #[serde(default)]
    dcp_size: Option<u32>,
}

/// The size, in tokens, at which a worker actually publishes KV blocks, or
/// `None` when no sound value exists and the subscription must be skipped.
///
/// Event granularity is set by the worker's **radix tree** page size. Under
/// decode context parallelism (`--dcp-size N`) the tree adopts the KV
/// allocator's page verbatim, and SGLang's generic paged allocator pages at
/// `page_size * N`; the tree then emits one block hash per widened page (the
/// publisher coalesces runs of them into a single `BlockStored`). Meanwhile
/// `/server_info` keeps advertising the *unwidened* `page_size` in its
/// `kv_events` block, with `dcp_size` alongside it at the top level.
///
/// Hashing queries at the advertised `page_size` therefore produces hashes a
/// DCP worker never stores: every cache-aware lookup misses and routing
/// degrades to min-load.
///
/// **Nothing downstream catches that.** [`super::block_size_oracle`] only
/// reconciles workers against *each other*, so a uniformly-DCP fleet agrees
/// perfectly while every lookup misses; correctness rests entirely on this
/// function matching the engine. The untapped check is the wire itself — every
/// [`super::wire::BlockStored`] carries the publisher's own `block_size`, which
/// the pump decodes and discards. Comparing it against the latched oracle value
/// (`>` is proof of a mismatch; a trailing partial page is legitimately `<`)
/// would make this whole class of failure loud without trusting anyone's
/// reading of the engine.
///
/// `worker_url` is for logging only, mirroring [`classify_bigram`].
pub(crate) fn kv_event_block_size(
    advertised_page_size: u32,
    dcp_size: Option<u32>,
    worker_url: &str,
) -> Option<u32> {
    let dcp_size = match dcp_size {
        // Absent on engines predating the field, which by construction ran no
        // DCP. Silent: this is the ordinary non-DCP reading.
        None => 1,
        // Defensive only: `ServerArgs` defaults `dcp_size` to 1 and rejects
        // anything below it, so no healthy SGLang emits this. Reaching it means
        // the payload is not the contract we think it is — say so rather than
        // quietly repairing it and proceeding on a reading just disproven.
        Some(0) => {
            warn!(
                worker_url = worker_url,
                "kv-events discovery: worker reported dcp_size=0, which no healthy \
                 SGLang emits; assuming no DCP. If this worker is running DCP, its \
                 cache-aware routing will silently never match",
            );
            1
        }
        Some(n) => n,
    };

    let Some(block_size) = advertised_page_size.checked_mul(dcp_size) else {
        // Saturating would hand back a value that passes every downstream guard
        // and then destroys prefix matching: `u32::MAX` latches the oracle
        // fleet-wide, and `div_ceil(u32::MAX)` collapses any prompt to a single
        // hash. There is no "closest representable" block size, only a corrupt
        // one, so refuse the worker instead of fabricating it.
        warn!(
            worker_url = worker_url,
            page_size = advertised_page_size,
            dcp_size,
            "kv-events discovery: page_size * dcp_size overflows u32; skipping this \
             worker's KV subscription rather than routing on a corrupt block size",
        );
        return None;
    };

    if dcp_size > 1 {
        // Logged here rather than at the call sites so both `EventConfig`
        // builders inherit it, and at info! because the router's default level
        // is info and a wrong block size is invisible in every metric.
        info!(
            worker_url = worker_url,
            page_size = advertised_page_size,
            dcp_size,
            block_size,
            "kv-events discovery: DCP worker publishes at the widened page size; \
             hashing at page_size * dcp_size",
        );
    }
    Some(block_size)
}

/// Whether a worker's `/server_info` `speculative_algorithm` means it hashes KV
/// blocks over token bigrams. Recognizes the engine's `is_eagle()` set
/// (`EAGLE`, `EAGLE3`, `FROZEN_KV_MTP`, case-insensitive).
///
/// An *unrecognized* value that looks EAGLE-family (contains `EAGLE` or `MTP`)
/// is logged loudly and treated as non-bigram — it most likely means a new
/// EAGLE variant the router doesn't know yet, which would otherwise silently
/// zero out cache-aware routing (the exact failure this whole path fixes).
/// Recognized non-EAGLE algorithms (and the absent field) map to `false`
/// silently.
pub(crate) fn classify_bigram(speculative_algorithm: Option<&str>, worker_url: &str) -> bool {
    let Some(algo) = speculative_algorithm else {
        return false;
    };
    let upper = algo.to_ascii_uppercase();
    match upper.as_str() {
        "EAGLE" | "EAGLE3" | "FROZEN_KV_MTP" => true,
        _ => {
            if upper.contains("EAGLE") || upper.contains("MTP") {
                tracing::warn!(
                    worker_url = %worker_url,
                    speculative_algorithm = %algo,
                    "kv-events: unrecognized EAGLE-like speculative_algorithm; treating as \
                     non-bigram (unigram) hashing. If this is an EAGLE-family algorithm, \
                     cache-aware routing will silently never match — add it to classify_bigram",
                );
            }
            false
        }
    }
}

#[derive(Deserialize)]
struct KvEventsBlock {
    // `publisher` is captured for forward-compatibility but unused: the
    // only publisher implementation supported on the gateway side is
    // ZMQ. Keeping the field optional means a future SGLang that adds a
    // non-ZMQ publisher string won't fail this deserialize; the
    // resulting subscriber will still try to open a ZMQ connection on
    // `endpoint_host:endpoint_port_base` and fail visibly there.
    #[allow(dead_code)]
    #[serde(default)]
    publisher: Option<String>,
    endpoint_host: String,
    endpoint_port_base: u16,
    #[serde(default)]
    topic: String,
    /// Base port of the dedicated load-snapshot socket range. Absent on
    /// workers that predate load publishing (`None` ⇒ no load subscriber).
    #[serde(default)]
    load_endpoint_port_base: Option<u16>,
    block_size: u32,
    dp_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Json, Router};
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    /// Spin up a tiny axum server that returns `body` on GET /server_info.
    /// Returns the base URL (`http://127.0.0.1:<port>`) and a shutdown handle.
    async fn spawn_fake_worker(body: Arc<Value>) -> (String, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let body_clone = body.clone();
        let app = Router::new().route(
            "/server_info",
            get(move || {
                let body = body_clone.clone();
                async move { Json((*body).clone()) }
            }),
        );
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

    fn client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap()
    }

    /// Happy path: worker advertises a ZMQ publisher; gateway substitutes
    /// `*` with the worker host.
    #[tokio::test]
    async fn fetch_returns_event_config_when_block_present() {
        let body = Arc::new(json!({
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "load_endpoint_port_base": 5559,
                "block_size": 64,
                "dp_size": 2,
            }
        }));
        let (url, _shutdown) = spawn_fake_worker(body).await;
        let got = fetch_event_config(&url, &client()).await.unwrap();
        assert_eq!(
            got,
            Some(EventConfig {
                host: "127.0.0.1".to_string(),
                port_base: 5557,
                topic: "kv".to_string(),
                load_port_base: Some(5559),
                block_size: 64,
                dp_size: 2,
                is_bigram: false,
            })
        );
    }

    /// EAGLE-family `speculative_algorithm` (and only those) must set
    /// `is_bigram`, so the router selects the bigram hasher and its query
    /// hashes match the worker's bigram-stored block hashes.
    #[tokio::test]
    async fn fetch_sets_is_bigram_for_eagle_family_only() {
        for (algo, expected) in [
            (Some("EAGLE"), true),
            (Some("EAGLE3"), true),
            (Some("FROZEN_KV_MTP"), true),
            (Some("eagle"), true), // case-insensitive
            (Some("NONE"), false),
            (Some("NEXTN"), false), // non-eagle speculative algorithm
            (None, false),          // no speculative decoding
        ] {
            let mut obj = json!({
                "kv_events": {
                    "publisher": "zmq",
                    "endpoint_host": "*",
                    "endpoint_port_base": 5557,
                    "topic": "",
                    "block_size": 64,
                    "dp_size": 1,
                }
            });
            if let Some(a) = algo {
                obj["speculative_algorithm"] = json!(a);
            }
            let (url, _shutdown) = spawn_fake_worker(Arc::new(obj)).await;
            let got = fetch_event_config(&url, &client()).await.unwrap().unwrap();
            assert_eq!(
                got.is_bigram, expected,
                "speculative_algorithm={algo:?} should map to is_bigram={expected}"
            );
        }
    }

    /// Under DCP the worker publishes at `page_size * dcp_size`, so discovery
    /// must report the widened size. Hashing at the advertised `page_size`
    /// would miss every block the worker actually stores.
    #[tokio::test]
    async fn fetch_widens_block_size_by_dcp_size() {
        let body = Arc::new(json!({
            "dcp_size": 8,
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 1,
            }
        }));
        let (url, _shutdown) = spawn_fake_worker(body).await;
        let got = fetch_event_config(&url, &client()).await.unwrap().unwrap();
        assert_eq!(got.block_size, 512);
    }

    /// A worker that reports DCP disabled keeps the advertised page size
    /// verbatim. (`dcp_size` absent entirely is covered by
    /// `fetch_returns_event_config_when_block_present`.)
    #[tokio::test]
    async fn fetch_keeps_page_size_when_dcp_disabled() {
        let body = Arc::new(json!({
            "dcp_size": 1,
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 1,
            }
        }));
        let (url, _shutdown) = spawn_fake_worker(body).await;
        let got = fetch_event_config(&url, &client()).await.unwrap().unwrap();
        assert_eq!(got.block_size, 64);
    }

    #[test]
    fn kv_event_block_size_widens_only_under_dcp() {
        const URL: &str = "http://w1:30000";
        assert_eq!(kv_event_block_size(64, Some(8), URL), Some(512));
        assert_eq!(kv_event_block_size(64, Some(1), URL), Some(64));
        assert_eq!(
            kv_event_block_size(64, None, URL),
            Some(64),
            "an engine predating the dcp_size field is not running DCP"
        );
        assert_eq!(
            kv_event_block_size(64, Some(0), URL),
            Some(64),
            "a reported 0 must not reach compute_block_hashes, which asserts \
             block_size > 0 and would panic the selection hot path"
        );
        assert_eq!(
            kv_event_block_size(u32::MAX, Some(2), URL),
            None,
            "overflow must refuse the worker, not saturate: u32::MAX passes the \
             oracle and every downstream guard, then collapses any prompt to a \
             single hash via div_ceil"
        );
    }

    /// Worker reports a specific bind host (not wildcard): gateway must
    /// honour it instead of overwriting from the URL.
    #[tokio::test]
    async fn fetch_keeps_explicit_bind_host() {
        let body = Arc::new(json!({
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "10.1.2.3",
                "endpoint_port_base": 6000,
                "topic": "",
                "block_size": 128,
                "dp_size": 1,
            }
        }));
        let (url, _shutdown) = spawn_fake_worker(body).await;
        let got = fetch_event_config(&url, &client()).await.unwrap();
        assert_eq!(got.unwrap().host, "10.1.2.3");
    }

    /// Worker reachable but the `kv_events` field is null / missing:
    /// caller should fall back to its static config.
    #[tokio::test]
    async fn fetch_returns_none_when_block_null() {
        let body = Arc::new(json!({ "kv_events": null }));
        let (url, _shutdown) = spawn_fake_worker(body).await;
        let got = fetch_event_config(&url, &client()).await.unwrap();
        assert!(got.is_none());
    }

    /// Worker is reachable but its `/server_info` response doesn't even
    /// have a `kv_events` field (older SGLang).
    #[tokio::test]
    async fn fetch_returns_none_when_field_absent() {
        let body = Arc::new(json!({ "other_stuff": 1 }));
        let (url, _shutdown) = spawn_fake_worker(body).await;
        let got = fetch_event_config(&url, &client()).await.unwrap();
        assert!(got.is_none());
    }

    /// Connection-refused: no server at the URL. The retry loop exhausts
    /// every attempt and propagates `Err`. The caller (KvEventIndex) logs
    /// + skips the subscriber so a single flaky worker doesn't poison
    /// startup, but the failure remains distinguishable from "worker
    /// reachable but not publishing" (`Ok(None)`) so future re-discovery
    /// can retry.
    #[tokio::test]
    async fn fetch_returns_err_on_connection_failure() {
        let url = "http://127.0.0.1:1"; // port 1 is reserved / refused
        let got = fetch_event_config(url, &client_fast_retry()).await;
        assert!(got.is_err(), "expected Err on permanent connect refused");
    }

    /// HTTP client with a short timeout so the connection-failure tests don't
    /// pay the full 2s × FETCH_MAX_ATTEMPTS budget.
    fn client_fast_retry() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_millis(100))
            .build()
            .unwrap()
    }

    /// Invalid worker URL is the one case we propagate as Err — there's
    /// nothing to fall back to and the operator config is broken.
    #[tokio::test]
    async fn fetch_returns_err_on_invalid_url() {
        let got = fetch_event_config("not a url", &client()).await;
        assert!(got.is_err());
    }

    /// Multi-DP publisher contract: a worker reporting `dp_size = 8`
    /// produces an `EventConfig` with `dp_size = 8` and the base port
    /// preserved.  The subscriber is responsible for opening 8 SUB
    /// sockets at `port_base + 0..8`; discovery just carries the values.
    #[tokio::test]
    async fn fetch_handles_multi_dp_publisher_dp_size_eight() {
        let body = Arc::new(json!({
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 8,
            }
        }));
        let (url, _shutdown) = spawn_fake_worker(body).await;
        let got = fetch_event_config(&url, &client()).await.unwrap().unwrap();
        assert_eq!(got.dp_size, 8);
        assert_eq!(got.port_base, 5557);
        // Verify the implicit port range fits in u16.
        let max_port = u32::from(got.port_base) + got.dp_size - 1;
        assert!(
            max_port <= u32::from(u16::MAX),
            "max per-rank port {max_port} must fit in u16",
        );
    }

    /// Documents the discovery-layer contract for ports near the u16 ceiling:
    /// discovery does NOT validate `port_base + dp_size` overflow. The
    /// subscriber MUST defend against `port_base + dp_rank > u16::MAX`
    /// when opening sockets.  Pinning this so that a future addition of
    /// validation at the discovery layer is a deliberate design change,
    /// not an accident.
    #[tokio::test]
    async fn fetch_accepts_high_port_base_near_u16_max() {
        let body = Arc::new(json!({
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                // u16::MAX = 65535. With dp_size = 4, ranks 2 and 3 would
                // overflow.  Discovery still returns the EventConfig as-is.
                "endpoint_port_base": 65533,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 4,
            }
        }));
        let (url, _shutdown) = spawn_fake_worker(body).await;
        let got = fetch_event_config(&url, &client()).await.unwrap().unwrap();
        assert_eq!(got.port_base, 65533);
        assert_eq!(got.dp_size, 4);
        let last_rank = u32::from(got.port_base) + got.dp_size - 1;
        assert!(
            last_rank > u32::from(u16::MAX),
            "test fixture must put the last rank's port past u16::MAX so subscriber-level overflow handling is exercised by its own tests",
        );
    }
}
