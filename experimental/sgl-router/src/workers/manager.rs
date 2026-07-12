// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;
use crate::discovery::{DiscoveryEvent, ModelId, WorkerId, WorkerMode, WorkerSpec};
use crate::health::circuit_breaker::CircuitBreakerConfig;
use crate::policies::active_load::ActiveLoadRegistry;
use crate::policies::kv_events::KvEventIndex;
use crate::workers::introspect::{DisaggregationRole, WorkerIntrospector};
use crate::workers::WorkerRegistry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Resolve the circuit-breaker config for all model IDs carried by a spec.
///
/// The router serves a single configured model; apply its circuit-breaker
/// config when this worker advertises that model id. Falls back to `None`
/// (default config) otherwise.
fn cb_config_for_spec(spec: &WorkerSpec, cfg: &Config) -> Option<CircuitBreakerConfig> {
    let model = &cfg.model;
    let cbc = model.circuit_breaker.as_ref()?;
    if spec.model_ids.iter().any(|id| id.0 == model.id) {
        return Some(CircuitBreakerConfig {
            threshold: cbc.threshold,
            cool_down: Duration::from_secs(cbc.cool_down_secs),
        });
    }
    None
}

pub async fn run(rx: mpsc::Receiver<DiscoveryEvent>, registry: Arc<WorkerRegistry>) {
    run_with_config(rx, registry, None, None, None).await;
}

/// Run the worker manager, optionally honoring per-model circuit-breaker
/// configuration from `cfg`, an optional KV-event index that is notified
/// on every worker add / remove, and an optional active-load registry
/// that is asked to forget per-worker counters on `Removed`.
///
/// When `kv_index` is `None` the cache-aware-zmq path is disabled
/// (selection falls through to the non-cache-aware policies); when
/// `active_load` is `None` the active-load bookkeeping is not pruned
/// on worker removal (leaks one `WorkerCounters` slot per departed
/// worker — fine for tests, but production passes `Some(...)`); when
/// `cfg` is `None` the default CB config is used for every worker
/// (threshold = 3).
///
/// Uses the default HTTP client (2-second timeout) for `/server_info`
/// introspection.  Tests that want a tighter timeout call
/// [`run_with_introspector`] directly.
pub async fn run_with_config(
    rx: mpsc::Receiver<DiscoveryEvent>,
    registry: Arc<WorkerRegistry>,
    cfg: Option<Arc<Config>>,
    kv_index: Option<Arc<KvEventIndex>>,
    active_load: Option<Arc<ActiveLoadRegistry>>,
) {
    run_with_introspector(
        rx,
        registry,
        cfg,
        kv_index,
        active_load,
        Arc::new(WorkerIntrospector::default()),
    )
    .await
}

/// Internal entry point used by tests so they can supply a custom
/// [`WorkerIntrospector`] (e.g. shorter timeout, fake transport).
/// Production callers use [`run_with_config`].
///
/// # Concurrency model
///
/// - **Added(spec):** spawned onto a `tokio::task` so multiple workers
///   can fetch `/server_info` and register concurrently.  Without this,
///   a burst of N workers would serialize N × `SERVER_INFO_TIMEOUT`
///   worth of registration latency on the event loop.
/// - **Removed / ModeChanged:** processed sequentially on the event
///   loop, but first **await** any in-flight `Added` task for the same
///   id so the mutation observes the post-Added registry state.
///   Without this await, a `Removed` queued while `Added` is still
///   fetching would no-op (registry empty), then the deferred Added
///   write would leak the worker indefinitely.
pub async fn run_with_introspector(
    mut rx: mpsc::Receiver<DiscoveryEvent>,
    registry: Arc<WorkerRegistry>,
    cfg: Option<Arc<Config>>,
    kv_index: Option<Arc<KvEventIndex>>,
    active_load: Option<Arc<ActiveLoadRegistry>>,
    introspector: Arc<WorkerIntrospector>,
) {
    // In-flight `Added` registrations, keyed by worker id. Subsequent
    // `Removed` / `ModeChanged` events for the same id `await` the
    // handle so they observe the registry write the spawned task is
    // about to perform.  Entries are removed on completion (Added's
    // own task drops the slot before returning).
    let mut pending: HashMap<WorkerId, JoinHandle<()>> = HashMap::new();

    while let Some(event) = rx.recv().await {
        // Opportunistically reap handles whose tasks have already
        // completed so the map doesn't grow without bound under steady-
        // state churn.  This is O(map.len()) per event but the map only
        // holds in-flight Added events (typically << total workers).
        pending.retain(|_, h| !h.is_finished());

        match event {
            DiscoveryEvent::Added(spec) => {
                tracing::info!("discovery: +worker {} ({:?})", spec.id, spec.mode);
                let id = spec.id.clone();
                // If a previous Added for the same id is still in-flight,
                // drain it first so the upsert observes a consistent
                // pre-state (and so the new spawn doesn't race with the
                // old).
                if let Some(prev) = pending.remove(&id) {
                    let _ = prev.await;
                }
                let registry_t = registry.clone();
                let cfg_t = cfg.clone();
                let kv_index_t = kv_index.clone();
                let introspector_t = introspector.clone();
                let handle = tokio::spawn(async move {
                    register_one(spec, registry_t, cfg_t, kv_index_t, introspector_t).await;
                });
                pending.insert(id, handle);
            }
            DiscoveryEvent::Removed { id } => {
                tracing::info!("discovery: -worker {id}");
                if let Some(prev) = pending.remove(&id) {
                    // Wait for the matching Added to finish its registry
                    // write so the Removed observes (and clears) it.
                    let _ = prev.await;
                }
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
                // Drop the active-load per-worker counters slot.
                // Idempotent on the registry side, so we call it
                // unconditionally — a Removed for an unknown worker
                // (duplicate event) is a no-op. In-flight guards
                // pointing at this id are NOT invalidated; their drop
                // still removes the per-request entry cleanly, but the
                // per-worker counters slot will not be re-created
                // (selectors no longer see the worker, so no new
                // requests can register against it).
                if let Some(al) = &active_load {
                    al.forget_worker(&id);
                }
            }
            DiscoveryEvent::ModeChanged { id, mode } => {
                if let Some(prev) = pending.remove(&id) {
                    // Same rationale as Removed: wait for the registry
                    // write so the mode flip lands on the new entry.
                    let _ = prev.await;
                }
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

    // Drain any still-running registration tasks so callers `await`ing
    // the manager handle (tests, shutdown paths) see all registry
    // mutations land before the future resolves.
    for (_, h) in pending.drain() {
        let _ = h.await;
    }
}

/// Onboard a single worker: introspect once, then dispatch the result
/// to the registry and (if enabled) the KV-event index. Failure of any
/// step is logged inside the call chain; we still register the worker
/// with empty `model_ids` so the rest of the proxy plane treats it as
/// reachable.
async fn register_one(
    mut spec: WorkerSpec,
    registry: Arc<WorkerRegistry>,
    cfg: Option<Arc<Config>>,
    kv_index: Option<Arc<KvEventIndex>>,
    introspector: Arc<WorkerIntrospector>,
) {
    let worker_url = spec.url.clone();
    let info = introspector.fetch(&worker_url).await;
    if let Some(name) = info.served_model_name {
        spec.model_ids = vec![ModelId(name)];
    }
    // Trust `/server_info` over the discovery backend when the worker
    // self-disclosed its PD role: the server's own ServerArgs is the
    // authoritative source for `disaggregation_mode` and
    // `disaggregation_bootstrap_port`. The backend's mode (from K8s
    // labels, static-urls seed, etc.) was a best-guess seed; if the
    // server says it's actually a prefill peer on port 8998, that wins.
    // `None` here means the worker didn't tell us — keep the backend's
    // classification (older SGLang without the field, partial response,
    // unknown mode value, etc.).
    if let Some(role) = info.disaggregation_role {
        let (new_mode, new_port) = match role {
            DisaggregationRole::Plain => (WorkerMode::Plain, None),
            DisaggregationRole::Prefill { bootstrap_port } => {
                (WorkerMode::Prefill, Some(bootstrap_port))
            }
            DisaggregationRole::Decode => (WorkerMode::Decode, None),
        };
        if (new_mode, new_port) != (spec.mode, spec.bootstrap_port) {
            tracing::info!(
                worker_url = %worker_url,
                backend_mode = ?spec.mode,
                resolved_mode = ?new_mode,
                backend_bootstrap_port = ?spec.bootstrap_port,
                resolved_bootstrap_port = ?new_port,
                "/server_info overrode discovery-backend classification",
            );
            spec.mode = new_mode;
            spec.bootstrap_port = new_port;
        }
    }
    let cb = cfg.as_ref().and_then(|c| cb_config_for_spec(&spec, c));
    if let Err(e) = registry.add_with_cb(spec, cb) {
        // Mixed PD + plain on the same model is rejected at registration
        // time. Log loudly so the operator notices the conflicting
        // worker — the alternative (silently dropping into either pool)
        // makes the resolver surface the wrong 5xx code under partial
        // outages. Skip the kv_index hook too: a worker we didn't add
        // shouldn't drive cache-aware tree state.
        tracing::error!(
            worker_url = %worker_url,
            error = %e,
            "worker manager: refused to register worker due to mixed PD/plain configuration",
        );
        return;
    }
    if let Some(idx) = kv_index {
        // Pass the pre-resolved EventConfig so the KvEventIndex does
        // not issue a second `/server_info` round-trip.
        idx.add_worker(&worker_url, info.event_config).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ActiveLoadConfig, CircuitBreakerConfig as RawCbConfig, DiscoveryBackend, ModelConfig,
        PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
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
                pd_flip_router_admin_api_key: None,
            },
            observability: Default::default(),
            model: ModelConfig {
                id: id.into(),
                tokenizer_path: "/tmp/x".into(),
                policy: PolicyKind::RoundRobin,
                circuit_breaker: Some(RawCbConfig {
                    threshold: NonZeroU32::new(threshold).unwrap(),
                    cool_down_secs,
                }),
                cache_aware: None,
                sticky: None,
            },
            discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec!["http://test:30000".into()],
            }),
            proxy: ProxyConfig::default(),
            active_load: ActiveLoadConfig::default(),
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
            bootstrap_port: None,
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

    fn fast_introspector() -> Arc<WorkerIntrospector> {
        Arc::new(WorkerIntrospector::new(Duration::from_millis(500)))
    }

    /// `/server_info` returns `served_model_name` => the registry entry
    /// carries that as a single `ModelId`.
    #[tokio::test]
    async fn manager_resolves_model_id_from_server_info() {
        let (worker_url, _shutdown) =
            spawn_fake_server_info_worker(json!({"served_model_name": "Qwen3-0.6B"})).await;

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_introspector(
            rx,
            registry.clone(),
            None,
            None,
            None,
            fast_introspector(),
        ));

        let spec = WorkerSpec {
            id: WorkerId("w-1".into()),
            url: worker_url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
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
        let manager_handle = tokio::spawn(run_with_introspector(
            rx,
            registry.clone(),
            None,
            None,
            None,
            fast_introspector(),
        ));

        let spec = WorkerSpec {
            id: WorkerId("w-2".into()),
            url: worker_url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
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
        let manager_handle = tokio::spawn(run_with_introspector(
            rx,
            registry.clone(),
            None,
            None,
            None,
            fast_introspector(),
        ));

        for (id, url) in [("w-no-field", no_field_url), ("w-empty", empty_url)] {
            let spec = WorkerSpec {
                id: WorkerId(id.into()),
                url,
                mode: WorkerMode::Plain,
                model_ids: Vec::new(),
                bootstrap_port: None,
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
            None,
        ));

        let spec = WorkerSpec {
            id: WorkerId("w-1".into()),
            url: worker_url.clone(),
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
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
            None,
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

    /// Task B: `DiscoveryEvent::Removed` calls
    /// `ActiveLoadRegistry::forget_worker` so the per-worker counters
    /// slot is reaped. Without this, a long-lived cluster with worker
    /// churn would leak one `WorkerCounters` entry per departed worker.
    #[tokio::test]
    async fn manager_calls_active_load_forget_on_removed() {
        use tokio::time::timeout;

        // Fake worker is needed so the introspection step succeeds and
        // the Removed path observes a known URL — same shape as the
        // existing `manager_drives_kv_index_lifecycle` test.
        let (worker_url, _shutdown) =
            spawn_fake_server_info_worker(json!({"served_model_name": "m"})).await;

        let registry = Arc::new(WorkerRegistry::default());
        let active_load = ActiveLoadRegistry::with_defaults();
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_introspector(
            rx,
            registry.clone(),
            None,
            None,
            Some(Arc::clone(&active_load)),
            fast_introspector(),
        ));

        let id = WorkerId("w-1".into());
        let spec = WorkerSpec {
            id: id.clone(),
            url: worker_url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
        };
        tx.send(DiscoveryEvent::Added(spec.clone())).await.unwrap();
        // Wait for the manager to land the registry write so the
        // subsequent register/forget round trip exercises a live slot.
        let added = timeout(Duration::from_secs(2), async {
            loop {
                if registry.get(&id).is_some() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(added.is_ok(), "manager failed to register worker");

        // Mint a guard to force the active-load registry to create a
        // per-worker counters slot for this id.
        let _g = active_load.register(id.clone(), "test://", 10, 1);
        assert!(active_load.is_known(&id));

        // Now drive the Removed event and assert the counters slot is
        // gone.  We tear down the guard last so the request entry is
        // exercised on the post-forget path.
        tx.send(DiscoveryEvent::Removed { id: id.clone() })
            .await
            .unwrap();
        let removed = timeout(Duration::from_secs(2), async {
            loop {
                if !active_load.is_known(&id) && registry.get(&id).is_none() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            removed.is_ok(),
            "manager must call active_load.forget_worker on Removed",
        );

        drop(tx);
        let _ = manager_handle.await;
    }

    /// Discovery backend emits a `Plain` worker with no bootstrap port,
    /// but `/server_info` says `disaggregation_mode="prefill"` with
    /// `disaggregation_bootstrap_port=8998`. The manager must trust
    /// `/server_info` and register the worker as Prefill with the
    /// disclosed port — this is the load-bearing assertion for
    /// PD-on-K8s, where the K8s backend always emits Plain + None for
    /// `bootstrap_port` and the manager has to recover the role from
    /// the worker's self-disclosure.
    #[tokio::test]
    async fn manager_overrides_backend_classification_from_server_info() {
        let (worker_url, _shutdown) = spawn_fake_server_info_worker(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 8998,
        }))
        .await;

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_introspector(
            rx,
            registry.clone(),
            None,
            None,
            None,
            fast_introspector(),
        ));

        // Backend says Plain + None — the shape the K8s backend always
        // emits today.
        let spec = WorkerSpec {
            id: WorkerId("w-prefill".into()),
            url: worker_url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
        };
        tx.send(DiscoveryEvent::Added(spec.clone())).await.unwrap();

        let resolved = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(w) = registry.get(&spec.id) {
                    if w.mode() == WorkerMode::Prefill && w.bootstrap_port() == Some(8998) {
                        return true;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            resolved.is_ok(),
            "manager must apply /server_info disaggregation_role override; \
             expected mode=Prefill bootstrap_port=Some(8998), got {:?}",
            registry
                .get(&spec.id)
                .map(|w| (w.mode(), w.bootstrap_port())),
        );

        drop(tx);
        let _ = manager_handle.await;
    }
}
