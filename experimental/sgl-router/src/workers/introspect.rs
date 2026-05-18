// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Single-shot `/server_info` introspection for newly-discovered workers.
//!
//! Combines what used to be two separate round-trips (the worker
//! manager's `served_model_name` fetch and `KvEventIndex::add_worker`'s
//! `fetch_event_config`) into one HTTP request. The result is dispatched
//! by the manager: registry consumes `served_model_name`, the optional
//! `KvEventIndex` consumes the resolved `EventConfig`.
//!
//! # Failure semantics
//!
//! `fetch` is **infallible** — any error (network, non-2xx, JSON parse,
//! invalid worker URL) is logged at `warn!` and returns an empty
//! `ServerInfo` so the caller can register the worker with empty
//! `model_ids` and no kv-events attachment. Workers that need accuracy
//! around publisher availability use `kv_events::discovery::fetch_event_config`
//! directly (it returns `Result<Option<EventConfig>>`); the manager
//! intentionally doesn't.

use std::time::Duration;

use serde::Deserialize;
use tracing::warn;
use url::Url;

use crate::policies::kv_events::EventConfig;

/// Default timeout for `/server_info`. Conservative for a small JSON
/// payload served by SGLang's HTTP server.
const SERVER_INFO_TIMEOUT: Duration = Duration::from_secs(2);

/// Resolved per-worker bootstrap state, both halves optional.
///
/// `served_model_name` populates the registry; `event_config` is handed
/// to `KvEventIndex::add_worker` (skipping its own fetch).
#[derive(Debug, Clone, Default)]
pub struct ServerInfo {
    pub served_model_name: Option<String>,
    pub event_config: Option<EventConfig>,
}

/// Performs the single `/server_info` round-trip and projects the
/// response into both halves of `ServerInfo`. Cheap to clone — wraps a
/// `reqwest::Client` (which is internally `Arc`-backed).
#[derive(Clone)]
pub struct WorkerIntrospector {
    client: reqwest::Client,
}

impl WorkerIntrospector {
    /// Build with a private `reqwest::Client` carrying the supplied
    /// request timeout.  Production callers pass `SERVER_INFO_TIMEOUT`
    /// via `default()`; tests may pass shorter timeouts.
    pub fn new(timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("introspector http client builds");
        Self { client }
    }

    /// Reuse a caller-owned `reqwest::Client`. Useful in tests that want
    /// to assert request shape via a fake HTTP transport, or to share a
    /// connection pool across components.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Fetch `/server_info` for the worker.  Never returns an error:
    /// any failure is logged at `warn!` and yields a default
    /// `ServerInfo` with both halves `None`. Callers register the
    /// worker with empty model IDs and no event subscription on the
    /// failure path; future re-discovery will retry.
    pub async fn fetch(&self, worker_url: &str) -> ServerInfo {
        let server_info_url = format!("{}/server_info", worker_url.trim_end_matches('/'));
        let resp = match self.client.get(&server_info_url).send().await {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    worker_url = %worker_url,
                    error = %e,
                    "introspect: /server_info request failed; registering worker with empty model_ids"
                );
                return ServerInfo::default();
            }
        };
        if !resp.status().is_success() {
            warn!(
                worker_url = %worker_url,
                status = %resp.status(),
                "introspect: /server_info returned non-2xx; registering worker with empty model_ids"
            );
            return ServerInfo::default();
        }
        let parsed: ServerInfoBody = match resp.json().await {
            Ok(p) => p,
            Err(e) => {
                warn!(
                    worker_url = %worker_url,
                    error = %e,
                    "introspect: /server_info JSON parse failed; registering worker with empty model_ids"
                );
                return ServerInfo::default();
            }
        };

        let served_model_name = match parsed.served_model_name {
            Some(name) if !name.is_empty() => Some(name),
            Some(_) => {
                warn!(
                    worker_url = %worker_url,
                    "introspect: /server_info has empty `served_model_name`; registering worker with empty model_ids"
                );
                None
            }
            None => None,
        };

        let event_config = parsed
            .kv_events
            .map(|block| resolve_event_config(block, worker_url));

        ServerInfo {
            served_model_name,
            event_config,
        }
    }
}

impl Default for WorkerIntrospector {
    fn default() -> Self {
        Self::new(SERVER_INFO_TIMEOUT)
    }
}

/// Substitute a wildcard bind host (`*`, `0.0.0.0`, `::`, `[::]`) with
/// the host parsed from the worker URL — the gateway has to connect to
/// a routable address.  An unparsable worker URL leaves the host
/// unchanged: the subsequent ZMQ connect will fail visibly with the
/// wildcard literal, which is the same observable failure mode that
/// would occur today if the bind/connect were skipped.
pub(crate) fn resolve_event_config(block: KvEventsBlock, worker_url: &str) -> EventConfig {
    let host = if matches!(
        block.endpoint_host.as_str(),
        "*" | "0.0.0.0" | "::" | "[::]"
    ) {
        match Url::parse(worker_url)
            .ok()
            .and_then(|u| u.host_str().map(|s| s.to_owned()))
        {
            Some(h) => h,
            None => {
                warn!(
                    worker_url = %worker_url,
                    "introspect: cannot parse worker_url for wildcard substitution; keeping advertised host"
                );
                block.endpoint_host
            }
        }
    } else {
        block.endpoint_host
    };
    EventConfig {
        host,
        port_base: block.endpoint_port_base,
        topic: block.topic,
        block_size: block.block_size,
        dp_size: block.dp_size,
    }
}

/// Projection of `/server_info` used by the introspector.  Both halves
/// are `#[serde(default)]` so a worker that exposes only one of them
/// still deserialises; downstream callers handle `None` as "absent".
#[derive(Debug, Default, Deserialize)]
struct ServerInfoBody {
    #[serde(default)]
    served_model_name: Option<String>,
    #[serde(default)]
    kv_events: Option<KvEventsBlock>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct KvEventsBlock {
    // Forward-compatibility: the only publisher implementation
    // supported on the gateway side is ZMQ. Keeping the field optional
    // means a future SGLang that adds a non-ZMQ publisher string won't
    // fail deserialize; the resulting subscriber will still try to open
    // a ZMQ connection and fail visibly.
    #[allow(dead_code)]
    #[serde(default)]
    publisher: Option<String>,
    pub endpoint_host: String,
    pub endpoint_port_base: u16,
    #[serde(default)]
    pub topic: String,
    pub block_size: u32,
    pub dp_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Json, Router};
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    async fn spawn_fake_worker(body: Value) -> (String, oneshot::Sender<()>) {
        let body = Arc::new(body);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route(
            "/server_info",
            get(move || {
                let body = body.clone();
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

    fn fast_introspector() -> WorkerIntrospector {
        WorkerIntrospector::new(Duration::from_millis(500))
    }

    #[tokio::test]
    async fn fetch_returns_both_served_model_name_and_event_config() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "Qwen3-0.6B",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "10.1.2.3",
                "endpoint_port_base": 6000,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 2,
            }
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.served_model_name.as_deref(), Some("Qwen3-0.6B"));
        let cfg = got.event_config.expect("kv_events present");
        assert_eq!(cfg.host, "10.1.2.3");
        assert_eq!(cfg.port_base, 6000);
        assert_eq!(cfg.topic, "kv");
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.dp_size, 2);
    }

    #[tokio::test]
    async fn fetch_substitutes_wildcard_host() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "kv",
                "block_size": 64,
                "dp_size": 1,
            }
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        let cfg = got.event_config.expect("kv_events present");
        assert_eq!(cfg.host, "127.0.0.1");
    }

    #[tokio::test]
    async fn fetch_returns_empty_on_connection_refused() {
        // Port 1 is reserved; bind a temp listener to reserve a free
        // port then drop it so the connect fails fast.
        let temp = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = temp.local_addr().unwrap().port();
        drop(temp);
        let url = format!("http://127.0.0.1:{port}");
        let got = fast_introspector().fetch(&url).await;
        assert!(
            got.served_model_name.is_none(),
            "served_model_name must be None on connection refused"
        );
        assert!(
            got.event_config.is_none(),
            "event_config must be None on connection refused"
        );
    }

    #[tokio::test]
    async fn fetch_only_served_model_name_when_kv_events_absent() {
        let (url, _shutdown) = spawn_fake_worker(json!({"served_model_name": "m"})).await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.served_model_name.as_deref(), Some("m"));
        assert!(got.event_config.is_none());
    }

    #[tokio::test]
    async fn fetch_only_event_config_when_served_model_name_absent() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "127.0.0.1",
                "endpoint_port_base": 5557,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            }
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(got.served_model_name.is_none());
        let cfg = got.event_config.expect("kv_events present");
        assert_eq!(cfg.port_base, 5557);
    }
}
