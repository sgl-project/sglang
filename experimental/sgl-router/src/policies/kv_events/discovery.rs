//! Per-worker KV-event publisher discovery.
//!
//! Calls the worker's `/server_info` endpoint (extended on the SGLang
//! Python side) to learn where to connect its ZMQ KV-event publisher.
//! Returns an [`EventConfig`] on success or `Ok(None)` when the worker
//! isn't running an event publisher (older SGLang, `kv-events-config`
//! unset, `null` publisher, etc.). The caller is expected to fall back to
//! a globally-configured `event_port` in that case.
//!
//! Failure semantics are deliberately permissive: any HTTP-level problem
//! (timeout, non-200, malformed payload) maps to `Ok(None)` rather than
//! `Err`, so that one flaky worker introspection never prevents gateway
//! startup or worker registration.

use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use tracing::{debug, warn};
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
    /// Worker-reported `page_size`. The gateway cross-checks against its
    /// own configured `block_size`; a mismatch is silent miscompute and
    /// must produce a warn at the policy layer.
    pub block_size: u32,
    /// Number of attention-DP ranks publishing. The gateway opens this
    /// many SUB connections (one per rank).
    pub dp_size: u32,
}

/// Default timeout for the `/server_info` introspection request. The
/// worker is on the same network as the gateway in production; 2 seconds
/// is generous and still bounds gateway-startup latency.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(2);

/// Fetch the worker's KV-event publisher config via `/server_info`.
///
/// Returns:
/// - `Ok(Some(cfg))` when the worker exposed a usable `kv_events` block.
/// - `Ok(None)` when the worker is reachable but doesn't expose one
///   (older SGLang, `kv-events-config` unset, `null` publisher, etc.) —
///   the caller should fall back to its globally-configured event port.
/// - `Err(_)` only when `worker_url` itself cannot be parsed; network or
///   parsing failures degrade to `Ok(None)` with a `warn!` log.
pub async fn fetch_event_config(
    worker_url: &str,
    client: &reqwest::Client,
) -> Result<Option<EventConfig>> {
    let parsed = Url::parse(worker_url)
        .map_err(|e| anyhow!("invalid worker_url {worker_url}: {e}"))?;
    let worker_host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("worker_url {worker_url} has no host"))?
        .to_owned();

    let server_info_url = format!("{}/server_info", worker_url.trim_end_matches('/'));

    let resp = match client
        .get(&server_info_url)
        .timeout(DEFAULT_TIMEOUT)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!(
                worker_url = worker_url,
                error = %e,
                "kv-events discovery: /server_info request failed; falling back to static event_port"
            );
            return Ok(None);
        }
    };
    if !resp.status().is_success() {
        debug!(
            worker_url = worker_url,
            status = resp.status().as_u16(),
            "kv-events discovery: /server_info returned non-success; falling back"
        );
        return Ok(None);
    }
    let body: ServerInfoResponse = match resp.json().await {
        Ok(b) => b,
        Err(e) => {
            warn!(
                worker_url = worker_url,
                error = %e,
                "kv-events discovery: /server_info JSON parse failed; falling back"
            );
            return Ok(None);
        }
    };

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

    // Wildcard bind hosts mean "any interface" on the worker side — the
    // gateway has to connect to a routable address, which it learns from
    // the worker URL.
    let host = if matches!(block.endpoint_host.as_str(), "*" | "0.0.0.0" | "::" | "[::]")
    {
        worker_host
    } else {
        block.endpoint_host
    };

    Ok(Some(EventConfig {
        host,
        port_base: block.endpoint_port_base,
        topic: block.topic,
        block_size: block.block_size,
        dp_size: block.dp_size,
    }))
}

#[derive(Deserialize)]
struct ServerInfoResponse {
    #[serde(default)]
    kv_events: Option<KvEventsBlock>,
}

#[derive(Deserialize)]
struct KvEventsBlock {
    // `publisher` is in the wire shape but unused on the gateway side:
    // the only publisher implementation we support is ZMQ, and the
    // Python side already filters non-ZMQ from the block. Keeping the
    // field optional means we won't refuse to deserialize if future
    // SGLang versions add a different publisher value.
    #[allow(dead_code)]
    #[serde(default)]
    publisher: Option<String>,
    endpoint_host: String,
    endpoint_port_base: u16,
    #[serde(default)]
    topic: String,
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
                block_size: 64,
                dp_size: 2,
            })
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

    /// Connection-refused: no server at the URL. Must degrade to
    /// `Ok(None)` so a single flaky worker doesn't poison startup.
    #[tokio::test]
    async fn fetch_returns_none_on_connection_failure() {
        let url = "http://127.0.0.1:1"; // port 1 is reserved / refused
        let got = fetch_event_config(url, &client()).await.unwrap();
        assert!(got.is_none());
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
