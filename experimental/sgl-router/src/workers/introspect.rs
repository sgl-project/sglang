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

/// Retry budget for transient `/server_info` failures (connect/timeout/5xx).
/// 4xx + JSON-parse errors short-circuit — they're authoritative.
/// EndpointSlice can flip ready=true before the worker's HTTP server is
/// actually serving; without retry, that race lands a worker in the
/// registry with empty model_ids and chat dispatch fails with 502.
const FETCH_MAX_ATTEMPTS: u32 = 3;
const FETCH_BACKOFF_BASE: Duration = Duration::from_millis(100);

/// Resolved per-worker bootstrap state.
///
/// `served_model_name` populates the registry; `event_config` is handed
/// to `KvEventIndex::add_worker` (skipping its own fetch);
/// `disaggregation_role` lets the worker manager override the discovery
/// backend's PD classification (and fill in `WorkerSpec.bootstrap_port`
/// for prefill workers) — see `manager::register_one`.
#[derive(Debug, Clone, Default)]
pub struct ServerInfo {
    pub served_model_name: Option<String>,
    pub event_config: Option<EventConfig>,
    pub disaggregation_role: Option<DisaggregationRole>,
}

/// PD classification derived from a worker's `/server_info` response.
///
/// `Some(_)` means the worker self-disclosed its role and we should trust
/// it over the discovery backend's classification. `None` (the
/// `ServerInfo::disaggregation_role` value, not a variant here) means the
/// worker didn't tell us — older SGLang, missing field, or a partial
/// response — and the backend's classification wins. See the resolution
/// table in `resolve_disaggregation_role`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisaggregationRole {
    Plain,
    Prefill { bootstrap_port: u16 },
    Decode,
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

    /// Like [`Self::default`], but attach `Authorization: Bearer <key>`
    /// as a default header on every introspection request, for fleets
    /// whose workers enforce `--api-key`.
    pub fn with_worker_api_key(key: &str) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        let mut v = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .expect("worker api key must be a valid HTTP header value (validated at startup)");
        v.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, v);
        let client = reqwest::Client::builder()
            .timeout(SERVER_INFO_TIMEOUT)
            .default_headers(headers)
            .build()
            .expect("introspector http client builds");
        Self { client }
    }

    /// Fetch `/server_info` for the worker.  Never returns an error:
    /// any failure is logged at `warn!` and yields a default
    /// `ServerInfo` with both halves `None`. Callers register the
    /// worker with empty model IDs and no event subscription on the
    /// failure path; future re-discovery will retry.
    ///
    /// Transient failures (network errors, 5xx) are retried up to
    /// `FETCH_MAX_ATTEMPTS` times with exponential backoff. 4xx
    /// responses and JSON-parse errors short-circuit immediately —
    /// the worker answered authoritatively, retrying won't help.
    pub async fn fetch(&self, worker_url: &str) -> ServerInfo {
        let server_info_url = format!("{}/server_info", worker_url.trim_end_matches('/'));
        let parsed = match Self::fetch_with_retry(&self.client, &server_info_url, worker_url).await
        {
            Some(p) => p,
            None => return ServerInfo::default(),
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

        // EAGLE-family speculative decoding ⇒ the worker hashes KV blocks over
        // token bigrams; the router must mirror that on the selection side.
        let is_bigram = crate::policies::kv_events::classify_bigram(
            parsed.speculative_algorithm.as_deref(),
            worker_url,
        );
        let event_config = parsed
            .kv_events
            .map(|block| resolve_event_config(block, worker_url, is_bigram));

        let disaggregation_role = resolve_disaggregation_role(
            parsed.disaggregation_mode.as_deref(),
            parsed.disaggregation_bootstrap_port,
            worker_url,
        );

        ServerInfo {
            served_model_name,
            event_config,
            disaggregation_role,
        }
    }

    /// Issue the `/server_info` GET with bounded retry on transient
    /// errors. Returns `Some(body)` on success, `None` after exhausting
    /// retries (the caller falls back to default `ServerInfo`).
    async fn fetch_with_retry(
        client: &reqwest::Client,
        server_info_url: &str,
        worker_url: &str,
    ) -> Option<ServerInfoBody> {
        let mut delay = FETCH_BACKOFF_BASE;
        for attempt in 1..=FETCH_MAX_ATTEMPTS {
            match client.get(server_info_url).send().await {
                Err(e) => {
                    warn!(
                        worker_url = %worker_url,
                        attempt,
                        error = %e,
                        "introspect: /server_info request failed; will retry"
                    );
                }
                Ok(resp) if resp.status().is_server_error() => {
                    warn!(
                        worker_url = %worker_url,
                        attempt,
                        status = %resp.status(),
                        "introspect: /server_info returned 5xx; will retry"
                    );
                }
                Ok(resp) if !resp.status().is_success() => {
                    warn!(
                        worker_url = %worker_url,
                        status = %resp.status(),
                        "introspect: /server_info returned non-2xx; registering worker with empty model_ids"
                    );
                    return None;
                }
                Ok(resp) => match resp.json::<ServerInfoBody>().await {
                    Ok(body) => return Some(body),
                    Err(e) => {
                        warn!(
                            worker_url = %worker_url,
                            error = %e,
                            "introspect: /server_info JSON parse failed; registering worker with empty model_ids"
                        );
                        return None;
                    }
                },
            }
            if attempt < FETCH_MAX_ATTEMPTS {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
        warn!(
            worker_url = %worker_url,
            attempts = FETCH_MAX_ATTEMPTS,
            "introspect: /server_info failed after retries; registering worker with empty model_ids"
        );
        None
    }
}

/// Map the two `disaggregation_*` fields from `/server_info` into a
/// `DisaggregationRole`. Returns `None` when the worker hasn't told us
/// enough to be useful — the caller treats that as "defer to the
/// discovery backend's classification" instead of forcing Plain, which
/// preserves backwards compatibility with SGLang versions that predate
/// the field.
///
/// Resolution table:
///
/// | `disaggregation_mode`        | `disaggregation_bootstrap_port` | Result                              |
/// |------------------------------|----------------------------------|-------------------------------------|
/// | `None` (older SGLang)        | _any_                            | `None` — defer to backend           |
/// | `Some("null")`               | _any_                            | `Some(Plain)`                       |
/// | `Some("prefill")`            | `Some(p)`                        | `Some(Prefill { bootstrap_port: p })` |
/// | `Some("prefill")`            | `None`                           | warn + `None` — defer to backend    |
/// | `Some("decode")`             | _any_                            | `Some(Decode)`                      |
/// | `Some(other)`                | _any_                            | warn + `None`                       |
fn resolve_disaggregation_role(
    mode: Option<&str>,
    bootstrap_port: Option<u16>,
    worker_url: &str,
) -> Option<DisaggregationRole> {
    match mode {
        None => None,
        Some("null") => Some(DisaggregationRole::Plain),
        Some("prefill") => match bootstrap_port {
            Some(p) => Some(DisaggregationRole::Prefill { bootstrap_port: p }),
            None => {
                warn!(
                    worker_url = %worker_url,
                    "introspect: /server_info reports disaggregation_mode=\"prefill\" but \
                     disaggregation_bootstrap_port is missing; deferring to the discovery \
                     backend's classification"
                );
                None
            }
        },
        Some("decode") => Some(DisaggregationRole::Decode),
        Some(other) => {
            warn!(
                worker_url = %worker_url,
                disaggregation_mode = %other,
                "introspect: /server_info has unknown disaggregation_mode value; \
                 deferring to the discovery backend's classification"
            );
            None
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
pub(crate) fn resolve_event_config(
    block: KvEventsBlock,
    worker_url: &str,
    is_bigram: bool,
) -> EventConfig {
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
        is_bigram,
    }
}

/// Projection of `/server_info` used by the introspector. Every field is
/// `#[serde(default)]` so a worker that exposes only some of them still
/// deserialises; downstream callers handle `None` as "absent".
#[derive(Debug, Default, Deserialize)]
struct ServerInfoBody {
    #[serde(default)]
    served_model_name: Option<String>,
    #[serde(default)]
    kv_events: Option<KvEventsBlock>,
    /// Top-level `speculative_algorithm`. EAGLE-family values
    /// (EAGLE / EAGLE3 / FROZEN_KV_MTP) ⇒ the worker hashes KV blocks over
    /// token bigrams. Absent on workers without speculative decoding.
    #[serde(default)]
    speculative_algorithm: Option<String>,
    /// Carries the value of `ServerArgs.disaggregation_mode`
    /// (`"null"` | `"prefill"` | `"decode"`). Absent on older SGLang
    /// versions that predate the field.
    #[serde(default)]
    disaggregation_mode: Option<String>,
    /// `ServerArgs.disaggregation_bootstrap_port`. Meaningful only when
    /// `disaggregation_mode == "prefill"`; the prefill server's
    /// bootstrap server binds to exactly this port (no internal offset).
    #[serde(default)]
    disaggregation_bootstrap_port: Option<u16>,
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

    /// The PRIMARY `/server_info` path (the introspector, not the discovery.rs
    /// fallback) must flag `is_bigram` for an EAGLE worker so the policy picks
    /// the bigram hasher. Regression guard for the duplicated parse + the
    /// `resolve_event_config(.., is_bigram)` threading.
    #[tokio::test]
    async fn fetch_sets_is_bigram_for_eagle_worker() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "speculative_algorithm": "EAGLE",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            }
        }))
        .await;
        let cfg = fast_introspector()
            .fetch(&url)
            .await
            .event_config
            .expect("kv_events present");
        assert!(
            cfg.is_bigram,
            "EAGLE worker via the introspector must set is_bigram"
        );
    }

    /// A non-speculative worker (no `speculative_algorithm`) must NOT be bigram.
    #[tokio::test]
    async fn fetch_no_bigram_without_speculative_algorithm() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            }
        }))
        .await;
        let cfg = fast_introspector()
            .fetch(&url)
            .await
            .event_config
            .expect("kv_events present");
        assert!(!cfg.is_bigram, "non-speculative worker must not be bigram");
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

    /// `disaggregation_mode=prefill` + a bootstrap port → manager should
    /// see the worker as a prefill peer with the supplied port. This is
    /// the happy path that lets PD-on-K8s skip pod annotations entirely.
    #[tokio::test]
    async fn fetch_resolves_prefill_role_with_bootstrap_port() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 8998,
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(
            got.disaggregation_role,
            Some(DisaggregationRole::Prefill {
                bootstrap_port: 8998
            }),
        );
    }

    /// `disaggregation_mode=decode` → role is Decode regardless of any
    /// bootstrap-port field value (decode workers don't bind one).
    #[tokio::test]
    async fn fetch_resolves_decode_role() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "decode",
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.disaggregation_role, Some(DisaggregationRole::Decode));
    }

    /// `disaggregation_mode="null"` is SGLang's explicit "not
    /// disaggregated" value — we trust it and force the worker to Plain
    /// even if the discovery backend mistakenly classified it as
    /// prefill/decode.
    #[tokio::test]
    async fn fetch_resolves_plain_role_when_mode_is_null() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "null",
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert_eq!(got.disaggregation_role, Some(DisaggregationRole::Plain));
    }

    /// Partial data (`prefill` mode with no bootstrap port) returns
    /// `None` so the manager keeps the discovery backend's
    /// classification. The alternative — forcing Plain — would silently
    /// demote a misconfigured prefill worker to plain dispatch.
    #[tokio::test]
    async fn fetch_defers_to_backend_when_prefill_mode_lacks_bootstrap_port() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(
            got.disaggregation_role.is_none(),
            "prefill with no bootstrap port must defer to backend, got {:?}",
            got.disaggregation_role,
        );
    }

    /// Older SGLang doesn't expose `disaggregation_mode`. The
    /// introspector must not invent a classification — the discovery
    /// backend's seed (K8s labels, static-urls Plain default) still
    /// drives mode for these workers.
    #[tokio::test]
    async fn fetch_defers_to_backend_when_mode_field_is_absent() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(got.disaggregation_role.is_none());
    }

    /// Unknown `disaggregation_mode` value (future SGLang adds a new
    /// disaggregation flavor, network garbled the field, etc.) → defer
    /// to backend rather than guessing.
    #[tokio::test]
    async fn fetch_defers_to_backend_when_mode_is_unrecognized() {
        let (url, _shutdown) = spawn_fake_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "encode_only",
            "disaggregation_bootstrap_port": 8998,
        }))
        .await;
        let got = fast_introspector().fetch(&url).await;
        assert!(got.disaggregation_role.is_none());
    }
}
