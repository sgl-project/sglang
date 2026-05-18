// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;
use crate::discovery::{DiscoveryEvent, ModelId, WorkerSpec};
use crate::health::circuit_breaker::CircuitBreakerConfig;
use crate::policies::kv_events::KvEventIndex;
use crate::workers::WorkerRegistry;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// The slice of `/server_info` consumed by the worker manager.  We don't
/// share any other module's `ServerInfo` struct because each consumer
/// projects a different subset of the response, and coupling them would
/// force every new field used by one consumer to be plumbed through all.
#[derive(Debug, Deserialize)]
struct ServerInfoResponse {
    #[serde(default)]
    served_model_name: Option<String>,
}

/// Default timeout for the `/server_info` fetch — conservative for a
/// small JSON read served by SGLang's HTTP server.
const SERVER_INFO_TIMEOUT: Duration = Duration::from_secs(2);

/// Build the default HTTP client used by `run` and `run_with_config`.
fn default_http_client() -> Arc<reqwest::Client> {
    Arc::new(
        reqwest::Client::builder()
            .timeout(SERVER_INFO_TIMEOUT)
            .build()
            .expect("default http client builds"),
    )
}

/// Fetch the worker's `served_model_name` via `GET <worker_url>/server_info`.
///
/// Returns `Some(name)` only when the response is a 2xx with a non-empty
/// `served_model_name`.  All other outcomes — network error, non-2xx,
/// malformed JSON, missing or empty field — return `None` and emit a
/// `warn!` carrying enough context (url + error) for an operator to find
/// the misbehaving worker in the logs.
///
/// Failure semantics are intentional: workers are still registered with
/// `model_ids = []` so the rest of the proxy plane treats them as
/// reachable; the resolved-zero-model state is observable via the
/// registry's `model_ids` field.
async fn fetch_served_model_name(worker_url: &str, http: &reqwest::Client) -> Option<String> {
    let url = format!("{}/server_info", worker_url.trim_end_matches('/'));
    let resp = match http.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                worker_url = %worker_url,
                error = %e,
                "manager: /server_info request failed; registering worker with empty model_ids"
            );
            return None;
        }
    };
    if !resp.status().is_success() {
        tracing::warn!(
            worker_url = %worker_url,
            status = %resp.status(),
            "manager: /server_info returned non-2xx; registering worker with empty model_ids"
        );
        return None;
    }
    let parsed: ServerInfoResponse = match resp.json().await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(
                worker_url = %worker_url,
                error = %e,
                "manager: /server_info JSON parse failed; registering worker with empty model_ids"
            );
            return None;
        }
    };
    let name = parsed.served_model_name.unwrap_or_default();
    if name.is_empty() {
        tracing::warn!(
            worker_url = %worker_url,
            "manager: /server_info has no `served_model_name`; registering worker with empty model_ids"
        );
        return None;
    }
    Some(name)
}

/// Resolve the circuit-breaker config for all model IDs carried by a spec.
///
/// Workers may serve multiple models; we use the config of the **first** model
/// that has an explicit CB config, falling back to `None` (default config).
fn cb_config_for_spec(spec: &WorkerSpec, cfg: &Config) -> Option<CircuitBreakerConfig> {
    for model_id in &spec.model_ids {
        if let Some(mc) = cfg.models.iter().find(|m| m.id == model_id.0) {
            if let Some(cbc) = &mc.circuit_breaker {
                return Some(CircuitBreakerConfig {
                    threshold: cbc.threshold,
                    cool_down: Duration::from_secs(cbc.cool_down_secs),
                });
            }
        }
    }
    None
}

pub async fn run(rx: mpsc::Receiver<DiscoveryEvent>, registry: Arc<WorkerRegistry>) {
    run_with_config(rx, registry, None, None).await;
}

/// Run the worker manager, optionally honoring per-model circuit-breaker
/// configuration from `cfg` and an optional KV-event index that is
/// notified on every worker add / remove.  When `kv_index` is `None` the
/// cache-aware-zmq path is disabled (selection falls through to the
/// non-cache-aware policies); when `cfg` is `None` the default CB config
/// is used for every worker (threshold = 3).
///
/// Uses the default HTTP client (2-second timeout) for `/server_info`
/// introspection.  Tests that want a tighter timeout call
/// [`run_with_http`] directly.
pub async fn run_with_config(
    rx: mpsc::Receiver<DiscoveryEvent>,
    registry: Arc<WorkerRegistry>,
    cfg: Option<Arc<Config>>,
    kv_index: Option<Arc<KvEventIndex>>,
) {
    run_with_http(rx, registry, cfg, kv_index, default_http_client()).await
}

/// Internal entry point used by tests so they can supply a custom HTTP
/// client (e.g. a shorter timeout, or a fake transport).  Production
/// callers use [`run_with_config`].
pub async fn run_with_http(
    mut rx: mpsc::Receiver<DiscoveryEvent>,
    registry: Arc<WorkerRegistry>,
    cfg: Option<Arc<Config>>,
    kv_index: Option<Arc<KvEventIndex>>,
    http: Arc<reqwest::Client>,
) {
    while let Some(event) = rx.recv().await {
        match event {
            DiscoveryEvent::Added(mut spec) => {
                tracing::info!("discovery: +worker {} ({:?})", spec.id, spec.mode);
                // Resolve the served model name via HTTP introspection.
                // Failure registers the worker with empty `model_ids`
                // (the failure path is logged inside `fetch_…`).
                if let Some(name) = fetch_served_model_name(&spec.url, &http).await {
                    spec.model_ids = vec![ModelId(name)];
                }
                let cb = cfg.as_ref().and_then(|c| cb_config_for_spec(&spec, c));
                let worker_url = spec.url.clone();
                registry.add_with_cb(spec, cb);
                if let Some(idx) = &kv_index {
                    idx.add_worker(&worker_url).await;
                }
            }
            DiscoveryEvent::Removed { id } => {
                tracing::info!("discovery: -worker {id}");
                // Look up the URL before dropping the entry so the
                // KV-event index can clear its per-(url, dp_rank) state.
                let worker_url = registry.get(&id).map(|w| w.url.clone());
                registry.remove(&id);
                match (&kv_index, worker_url) {
                    (Some(idx), Some(url)) => {
                        idx.remove_worker(&url).await;
                    }
                    (Some(_), None) => {
                        // Registry didn't know this worker but kv-events
                        // is enabled — duplicate Removed or out-of-order
                        // event. KvEventIndex state for this id (if any)
                        // leaks until process shutdown; log so it's
                        // detectable.
                        tracing::warn!(
                            id = %id,
                            "discovery: Removed without a known URL; kv-events state (if any) not cleared",
                        );
                    }
                    (None, _) => {}
                }
            }
            DiscoveryEvent::ModeChanged { id, mode } => {
                // Mutate mode in place — preserves active_requests counter
                // (in-flight LoadGuards stay valid) and CircuitBreaker state
                // (open/half-open survives PD role flips).
                //
                // workers_for_mode filters at query time via w.mode(), so no
                // secondary index needs updating.
                match registry.get(&id) {
                    Some(w) => {
                        tracing::info!("discovery: ~worker {id} mode→{mode:?}");
                        w.set_mode(mode);
                    }
                    None => {
                        tracing::warn!(
                            id = %id,
                            mode = ?mode,
                            "discovery: ModeChanged for unknown worker — out-of-order event from backend",
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CircuitBreakerConfig as RawCbConfig, DiscoveryBackend, DiscoveryConfig, ModelConfig,
        PolicyKind, ServerConfig, StaticFileDiscoveryConfig,
    };
    use crate::discovery::{WorkerId, WorkerMode};
    use axum::{routing::get, Json, Router};
    use serde_json::{json, Value};
    use std::num::NonZeroU32;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    fn cfg_with_model_cb(id: &str, threshold: u32, cool_down_secs: u64) -> Config {
        Config {
            server: ServerConfig {
                host: "0".into(),
                port: 0,
            },
            observability: Default::default(),
            models: vec![ModelConfig {
                id: id.into(),
                tokenizer_path: "/tmp/x".into(),
                policy: PolicyKind::RoundRobin,
                circuit_breaker: Some(RawCbConfig {
                    threshold: NonZeroU32::new(threshold).unwrap(),
                    cool_down_secs,
                }),
            }],
            discovery: DiscoveryConfig {
                backend: DiscoveryBackend::StaticFile(StaticFileDiscoveryConfig {
                    path: "/tmp/w".into(),
                    poll_interval_ms: 200,
                }),
            },
        }
    }

    #[test]
    fn cb_config_for_spec_carries_threshold_and_cool_down() {
        let cfg = cfg_with_model_cb("m", 5, 60);
        let spec = WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://x".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
        };
        let cb = cb_config_for_spec(&spec, &cfg).expect("model has cb config");
        assert_eq!(cb.threshold.get(), 5);
        assert_eq!(cb.cool_down, Duration::from_secs(60));
    }

    /// Helper: spawn a tiny fake worker that returns the supplied JSON body
    /// on `GET /server_info`. Returns the worker URL + a shutdown channel.
    async fn spawn_fake_server_info_worker(body: Value) -> (String, oneshot::Sender<()>) {
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

    /// Reserve a TCP port and immediately drop the listener so subsequent
    /// connection attempts during the test fail fast with
    /// ConnectionRefused.
    fn unused_port() -> u16 {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    }

    fn fast_http_client() -> Arc<reqwest::Client> {
        Arc::new(
            reqwest::Client::builder()
                .timeout(Duration::from_millis(500))
                .build()
                .unwrap(),
        )
    }

    /// `/server_info` returns `served_model_name` => the registry entry
    /// carries that as a single `ModelId`.
    #[tokio::test]
    async fn manager_resolves_model_id_from_server_info() {
        let (worker_url, _shutdown) =
            spawn_fake_server_info_worker(json!({"served_model_name": "Qwen3-0.6B"})).await;

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_http(
            rx,
            registry.clone(),
            None,
            None,
            fast_http_client(),
        ));

        let spec = WorkerSpec {
            id: WorkerId("w-1".into()),
            url: worker_url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
        };
        tx.send(DiscoveryEvent::Added(spec.clone())).await.unwrap();

        let registered = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(w) = registry.get(&spec.id) {
                    if w.model_ids.iter().any(|m| m.0 == "Qwen3-0.6B") {
                        return true;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(registered.is_ok(), "manager did not resolve model id");

        drop(tx);
        let _ = manager_handle.await;
    }

    /// Worker unreachable (connection refused) => registry still has the
    /// worker, with `model_ids` empty. No panic; manager continues running.
    #[tokio::test]
    async fn manager_registers_with_empty_model_ids_when_server_info_unreachable() {
        let port = unused_port();
        let worker_url = format!("http://127.0.0.1:{port}");

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_http(
            rx,
            registry.clone(),
            None,
            None,
            fast_http_client(),
        ));

        let spec = WorkerSpec {
            id: WorkerId("w-2".into()),
            url: worker_url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
        };
        tx.send(DiscoveryEvent::Added(spec.clone())).await.unwrap();

        let registered = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(w) = registry.get(&spec.id) {
                    return w.model_ids.is_empty();
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            matches!(registered, Ok(true)),
            "manager must register worker with empty model_ids when /server_info fails: {registered:?}"
        );

        drop(tx);
        let _ = manager_handle.await;
    }

    /// `/server_info` returns a JSON object without `served_model_name`
    /// (or with the empty string): manager logs a warn and registers the
    /// worker with empty `model_ids`.
    #[tokio::test]
    async fn manager_registers_with_empty_model_ids_when_served_model_name_missing() {
        let (no_field_url, _no_field_shutdown) =
            spawn_fake_server_info_worker(json!({"other_field": "value"})).await;
        let (empty_url, _empty_shutdown) =
            spawn_fake_server_info_worker(json!({"served_model_name": ""})).await;

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_http(
            rx,
            registry.clone(),
            None,
            None,
            fast_http_client(),
        ));

        for (id, url) in [("w-no-field", no_field_url), ("w-empty", empty_url)] {
            let spec = WorkerSpec {
                id: WorkerId(id.into()),
                url,
                mode: WorkerMode::Plain,
                model_ids: Vec::new(),
            };
            tx.send(DiscoveryEvent::Added(spec.clone())).await.unwrap();
            let registered = tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    if let Some(w) = registry.get(&spec.id) {
                        return w.model_ids.is_empty();
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await;
            assert!(
                matches!(registered, Ok(true)),
                "manager must register worker {id} with empty model_ids when served_model_name is missing/empty: {registered:?}"
            );
        }

        drop(tx);
        let _ = manager_handle.await;
    }

    /// End-to-end wiring smoke test: spin up a fake worker, run the
    /// manager with a real `KvEventIndex` against that worker URL, and
    /// verify both `Added` and `Removed` propagate through to the
    /// index's internal worker map.
    ///
    /// The fake worker advertises a `kv_events` block in `/server_info`,
    /// so the manager → KvEventIndex → discovery → registry path is
    /// exercised end-to-end. The ZMQ connect itself targets an unused
    /// port and fails (port is closed), but the *index-level* state still
    /// records the worker — which is exactly the invariant under test:
    /// `add_worker` registers the worker URL in `KvEventIndex.workers`
    /// even when the per-rank SUB connect fails.
    #[tokio::test]
    async fn manager_drives_kv_index_lifecycle() {
        use tokio::time::timeout;

        // The fake worker advertises both `kv_events` for KvEventIndex AND
        // `served_model_name` so the worker-manager HTTP introspection
        // also resolves a model id.
        let body = json!({
            "served_model_name": "m",
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "127.0.0.1",
                "endpoint_port_base": 60000,
                "topic": "",
                "block_size": 64,
                "dp_size": 1,
            }
        });
        let (worker_url, _shutdown) = spawn_fake_server_info_worker(body).await;

        let registry = Arc::new(WorkerRegistry::default());
        let kv_index = KvEventIndex::new();
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_config(
            rx,
            registry.clone(),
            None,
            Some(kv_index.clone()),
        ));

        let spec = WorkerSpec {
            id: WorkerId("w-1".into()),
            url: worker_url.clone(),
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
        };
        tx.send(DiscoveryEvent::Added(spec.clone())).await.unwrap();
        // Wait until the manager has both registered the worker AND
        // resolved /server_info — bound the wait so a hang surfaces.
        let added = timeout(Duration::from_secs(2), async {
            loop {
                if registry.get(&spec.id).is_some() && kv_index.known_worker_count() == 1 {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(added.is_ok(), "manager failed to propagate Added");

        tx.send(DiscoveryEvent::Removed {
            id: spec.id.clone(),
        })
        .await
        .unwrap();
        let removed = timeout(Duration::from_secs(2), async {
            loop {
                if registry.get(&spec.id).is_none() && kv_index.known_worker_count() == 0 {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(removed.is_ok(), "manager failed to propagate Removed");

        drop(tx);
        let _ = manager_handle.await;
        kv_index.shutdown().await;
    }

    /// `Removed` for an unknown id with kv-events enabled must not panic.
    /// The kv_index has no entry for that id either, so it must remain
    /// empty after the no-op.
    #[tokio::test]
    async fn manager_removed_unknown_id_is_noop() {
        use tokio::time::sleep;

        let registry = Arc::new(WorkerRegistry::default());
        let kv_index = KvEventIndex::new();
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_config(
            rx,
            registry.clone(),
            None,
            Some(kv_index.clone()),
        ));

        tx.send(DiscoveryEvent::Removed {
            id: WorkerId("never-added".into()),
        })
        .await
        .unwrap();
        // Let the manager process the event.
        sleep(Duration::from_millis(50)).await;

        assert_eq!(kv_index.known_worker_count(), 0);
        drop(tx);
        let _ = manager_handle.await;
        kv_index.shutdown().await;
    }
}
