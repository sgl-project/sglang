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

/// Production reconcile cadence. Workers that register without resolving
/// their model IDs (a `/server_info` introspection that failed at `Added`
/// time — e.g. the EndpointSlice flipped `ready=true` before the engine's
/// scheduler-backed `/server_info` could answer) are re-introspected on
/// this interval until they join their model pool. The worst-case
/// "registered but invisible" window is about one interval plus the
/// introspection round-trip; steady state costs one cheap registry scan
/// per interval. See `reconcile_unresolved_workers` for the (benign)
/// case of a worker that answers but never advertises a model name.
const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);

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
/// Production callers use [`run_with_config`]. Reconciles unresolved
/// workers on [`RECONCILE_INTERVAL`]; tests that need a tighter cadence
/// call [`run_with_introspector_and_reconcile`] directly.
pub async fn run_with_introspector(
    rx: mpsc::Receiver<DiscoveryEvent>,
    registry: Arc<WorkerRegistry>,
    cfg: Option<Arc<Config>>,
    kv_index: Option<Arc<KvEventIndex>>,
    active_load: Option<Arc<ActiveLoadRegistry>>,
    introspector: Arc<WorkerIntrospector>,
) {
    run_with_introspector_and_reconcile(
        rx,
        registry,
        cfg,
        kv_index,
        active_load,
        introspector,
        RECONCILE_INTERVAL,
    )
    .await
}

/// As [`run_with_introspector`], but with a caller-chosen reconcile
/// cadence (tests use a short interval to converge quickly).
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
/// - **Reconcile tick:** every `reconcile_interval`, re-introspects any
///   registered worker whose `model_ids` are still empty (see
///   [`reconcile_unresolved_workers`]). Runs on the same loop and shares
///   `pending` with the discovery events so re-registrations stay
///   serialized per id against concurrent `Added` / `Removed`.
pub async fn run_with_introspector_and_reconcile(
    mut rx: mpsc::Receiver<DiscoveryEvent>,
    registry: Arc<WorkerRegistry>,
    cfg: Option<Arc<Config>>,
    kv_index: Option<Arc<KvEventIndex>>,
    active_load: Option<Arc<ActiveLoadRegistry>>,
    introspector: Arc<WorkerIntrospector>,
    reconcile_interval: Duration,
) {
    // In-flight registrations, keyed by worker id. Subsequent
    // `Removed` / `ModeChanged` events (and reconcile passes) for the
    // same id `await` the handle so they observe the registry write the
    // spawned task is about to perform. The spawned task does NOT remove
    // its own slot; slots are reaped by the `retain(!is_finished())`
    // sweeps below, or drained by a `Removed` / `Added` / reconcile for
    // the same id.
    let mut pending: HashMap<WorkerId, JoinHandle<()>> = HashMap::new();

    // Periodic reconcile. `interval_at` delays the first tick by one
    // interval (no point scanning an empty registry at t=0); `Skip` means
    // a reconcile pass that runs long never builds a backlog of catch-up
    // ticks.
    let mut reconcile = tokio::time::interval_at(
        tokio::time::Instant::now() + reconcile_interval,
        reconcile_interval,
    );
    reconcile.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            maybe_event = rx.recv() => {
                let Some(event) = maybe_event else {
                    break;
                };
                // Opportunistically reap handles whose tasks have already
                // completed so the map doesn't grow without bound under
                // steady-state churn. O(map.len()) per event, but the map
                // only holds in-flight registrations (typically << total
                // workers).
                pending.retain(|_, h| !h.is_finished());
                handle_discovery_event(
                    event,
                    &registry,
                    &cfg,
                    &kv_index,
                    &active_load,
                    &introspector,
                    &mut pending,
                )
                .await;
            }
            _ = reconcile.tick() => {
                pending.retain(|_, h| !h.is_finished());
                reconcile_unresolved_workers(
                    &registry,
                    &cfg,
                    &kv_index,
                    &introspector,
                    &mut pending,
                );
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

/// Apply a single discovery event to the registry (and the optional
/// KV-event index / active-load registry). Extracted from the manager
/// loop so the loop can also service reconcile ticks via
/// `tokio::select!`. See the concurrency-model doc on
/// [`run_with_introspector_and_reconcile`] for the per-id ordering
/// contract `pending` enforces.
async fn handle_discovery_event(
    event: DiscoveryEvent,
    registry: &Arc<WorkerRegistry>,
    cfg: &Option<Arc<Config>>,
    kv_index: &Option<Arc<KvEventIndex>>,
    active_load: &Option<Arc<ActiveLoadRegistry>>,
    introspector: &Arc<WorkerIntrospector>,
    pending: &mut HashMap<WorkerId, JoinHandle<()>>,
) {
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
                // Wait for the matching Added (or in-flight reconcile) to
                // finish its registry write so the Removed observes (and
                // clears) it — this is what prevents a mid-reconcile
                // worker from being resurrected after it genuinely left.
                let _ = prev.await;
            }
            // Look up the URL before dropping the entry so the
            // KV-event index can clear its per-(url, dp_rank) state.
            let worker_url = registry.get(&id).map(|w| w.url.clone());
            registry.remove(&id);
            match (kv_index, worker_url) {
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
            if let Some(al) = active_load {
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

/// Re-introspect workers that registered without resolving their model
/// IDs.
///
/// A worker lands in the registry with empty `model_ids` when its
/// `/server_info` introspection failed at `Added` time (e.g. the
/// EndpointSlice flips `ready=true` before the engine can answer the
/// scheduler-backed `/server_info` round-trip, and the introspector's
/// bounded retry budget is exhausted). Such a worker is present in
/// `by_id` but absent from every `by_model` pool, so it gets zero traffic
/// — and it never recovers on its own: discovery re-lists carry the same
/// empty `model_ids` the backend always emits, so they produce no new
/// `Added` event.
///
/// This pass re-runs `register_one` (an idempotent registry upsert +
/// idempotent kv-events subscribe) for each such worker until the
/// introspection succeeds and the worker joins its model pool. A worker
/// already being (re-)registered is skipped via `pending`, so a slow
/// `/server_info` never stacks duplicate tasks for one id; and because
/// `pending` is shared with the event loop, a `Removed` that arrives
/// mid-reconcile awaits the in-flight handle before clearing the
/// registry, so a worker that genuinely left is not resurrected.
///
/// A worker that answers `/server_info` but never advertises a
/// `served_model_name` also stays in this set and is re-introspected
/// every interval. That is a benign, bounded poll (one cheap round-trip
/// per worker per interval), not a leak — but it is also never escalated,
/// so the per-attempt failure logging stays at `debug!`; the introspector
/// itself emits the `warn!` that surfaces a persistently failing worker.
fn reconcile_unresolved_workers(
    registry: &Arc<WorkerRegistry>,
    cfg: &Option<Arc<Config>>,
    kv_index: &Option<Arc<KvEventIndex>>,
    introspector: &Arc<WorkerIntrospector>,
    pending: &mut HashMap<WorkerId, JoinHandle<()>>,
) {
    for worker in registry.all() {
        if !worker.model_ids.is_empty() {
            continue;
        }
        let id = worker.id.clone();
        if pending.contains_key(&id) {
            // A registration for this id is already in flight; let it
            // finish rather than racing a second introspection.
            continue;
        }
        // Rebuild a discovery-shaped spec: empty `model_ids` so
        // `register_one` re-resolves them from `/server_info`; current
        // mode + bootstrap_port as the seed (`register_one` re-applies
        // any `/server_info` override).
        let spec = WorkerSpec {
            id: id.clone(),
            url: worker.url.clone(),
            mode: worker.mode(),
            model_ids: Vec::new(),
            bootstrap_port: worker.bootstrap_port(),
        };
        // `debug!` not `info!`: this fires every interval for each
        // still-unresolved worker, so info-level would spam for a worker
        // that is permanently model-less. The introspector logs the
        // underlying `/server_info` failure at `warn!` on each attempt,
        // which is the operator-facing signal.
        tracing::debug!(
            worker_id = %id,
            worker_url = %worker.url,
            "reconcile: re-introspecting worker that registered without model_ids",
        );
        let registry_t = registry.clone();
        let cfg_t = cfg.clone();
        let kv_index_t = kv_index.clone();
        let introspector_t = introspector.clone();
        let handle = tokio::spawn(async move {
            register_one(spec, registry_t, cfg_t, kv_index_t, introspector_t).await;
        });
        pending.insert(id, handle);
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

    /// End-to-end wiring check: spin up a fake worker, run the
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

    /// Fake `/server_info` worker whose readiness is switchable at
    /// runtime. While `ready` is false it answers `503` (mimicking an
    /// engine whose EndpointSlice flipped `ready=true` before its
    /// scheduler-backed `/server_info` could answer); flip `ready` to
    /// true and it serves `body`.
    async fn spawn_switchable_server_info_worker(
        body: Value,
        ready: Arc<std::sync::atomic::AtomicBool>,
    ) -> (String, oneshot::Sender<()>) {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        let body = Arc::new(body);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route(
            "/server_info",
            get(move || {
                let body = body.clone();
                let ready = ready.clone();
                async move {
                    if ready.load(std::sync::atomic::Ordering::SeqCst) {
                        Json((*body).clone()).into_response()
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    }
                }
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

    /// A worker that registers with empty `model_ids` because
    /// `/server_info` was failing at `Added` time must be re-introspected
    /// by the periodic reconcile loop and join its model pool once
    /// `/server_info` recovers — with NO new discovery event. This is the
    /// regression guard for the "EndpointSlice flips ready before the
    /// engine can answer /server_info → worker invisible forever" bug.
    #[tokio::test]
    async fn reconcile_re_introspects_worker_that_failed_initial_server_info() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use tokio::time::timeout;

        let ready = Arc::new(AtomicBool::new(false));
        let (worker_url, _shutdown) =
            spawn_switchable_server_info_worker(json!({"served_model_name": "m"}), ready.clone())
                .await;

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        // Short reconcile cadence so the test converges quickly.
        let manager_handle = tokio::spawn(run_with_introspector_and_reconcile(
            rx,
            registry.clone(),
            None,
            None,
            None,
            fast_introspector(),
            Duration::from_millis(150),
        ));

        let id = WorkerId("w-slow".into());
        let model = ModelId("m".into());
        let spec = WorkerSpec {
            id: id.clone(),
            url: worker_url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
        };
        tx.send(DiscoveryEvent::Added(spec)).await.unwrap();

        // Phase 1: the worker registers (present in `by_id`) but stays out
        // of the model pool while `/server_info` keeps failing.
        let stuck = timeout(Duration::from_secs(2), async {
            loop {
                if let Some(w) = registry.get(&id) {
                    if w.model_ids.is_empty() && registry.workers_for(&model).is_empty() {
                        return true;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            stuck.is_ok(),
            "worker should register with empty model_ids (invisible to routing) while /server_info fails",
        );

        // The engine finishes coming up: `/server_info` now answers.
        ready.store(true, Ordering::SeqCst);

        // Phase 2: the reconcile loop must re-introspect and move the
        // worker into the model pool with no new discovery event.
        let recovered = timeout(Duration::from_secs(3), async {
            loop {
                if !registry.workers_for(&model).is_empty() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            recovered.is_ok(),
            "reconcile loop must re-introspect the worker and add it to the model pool once /server_info recovers",
        );
        assert_eq!(
            registry.get(&id).unwrap().model_ids,
            vec![ModelId("m".into())],
            "recovered worker must carry the resolved model id",
        );

        drop(tx);
        let _ = manager_handle.await;
    }

    /// Resurrection safety: a `Removed` that arrives while a reconcile
    /// re-introspection for the same id is in-flight must NOT resurrect
    /// the worker. The `Removed` handler awaits the in-flight handle (which
    /// re-adds the worker), then clears it — so the worker ends up gone and
    /// stays gone. Guards the per-id ordering contract that `pending`
    /// enforces for the reconcile path specifically (distinct from the
    /// Added path's `removed_awaits_in_flight_added`).
    #[tokio::test]
    async fn reconcile_does_not_resurrect_worker_removed_mid_reintrospection() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use tokio::sync::Notify;
        use tokio::time::timeout;

        // While unarmed, /server_info returns 200 with no model name (the
        // worker registers unresolved). Once armed, the next call signals
        // `entered` and blocks on `release` — parking the reconcile
        // re-introspection's register_one task in `pending` — then returns
        // a valid body so the late re-add is real.
        let arm = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let arm_h = arm.clone();
        let entered_h = entered.clone();
        let release_h = release.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route(
            "/server_info",
            get(move || {
                let arm = arm_h.clone();
                let entered = entered_h.clone();
                let release = release_h.clone();
                async move {
                    if arm.load(Ordering::SeqCst) {
                        entered.notify_one();
                        release.notified().await;
                        Json(json!({"served_model_name": "m"}))
                    } else {
                        Json(json!({}))
                    }
                }
            }),
        );
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        let url = format!("http://127.0.0.1:{port}");

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        // Timeout well above the release latency so the parked fetch
        // completes on release, not by timing out.
        let introspector = Arc::new(WorkerIntrospector::new(Duration::from_secs(5)));
        let manager_handle = tokio::spawn(run_with_introspector_and_reconcile(
            rx,
            registry.clone(),
            None,
            None,
            None,
            introspector,
            Duration::from_millis(80),
        ));

        let id = WorkerId("w-race".into());
        let model = ModelId("m".into());
        tx.send(DiscoveryEvent::Added(WorkerSpec {
            id: id.clone(),
            url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
        }))
        .await
        .unwrap();

        // Phase 1: registered but unresolved.
        let stuck = timeout(Duration::from_secs(2), async {
            loop {
                if let Some(w) = registry.get(&id) {
                    if w.model_ids.is_empty() && registry.workers_for(&model).is_empty() {
                        return true;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(stuck.is_ok(), "worker should register unresolved");

        // Arm so the next reconcile re-introspection parks in-flight, and
        // wait until it provably reaches the handler.
        arm.store(true, Ordering::SeqCst);
        timeout(Duration::from_secs(2), entered.notified())
            .await
            .expect("a reconcile re-introspection should reach the handler within a few ticks");

        // The worker genuinely leaves while its re-introspection is parked.
        tx.send(DiscoveryEvent::Removed { id: id.clone() })
            .await
            .unwrap();
        // Let the manager dequeue the Removed and reach the
        // `pending.remove(&id).await` join point before we release.
        tokio::time::sleep(Duration::from_millis(50)).await;
        release.notify_one();

        // The late re-add must lose to the Removed: worker absent and stays
        // absent (reconcile never re-sees it — it's gone from the registry).
        let gone = timeout(Duration::from_secs(2), async {
            loop {
                if registry.get(&id).is_none() && registry.workers_for(&model).is_empty() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            gone.is_ok(),
            "Removed must win over the in-flight reconcile re-add (no resurrection)",
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            registry.get(&id).is_none(),
            "worker must not reappear after removal",
        );
        assert!(
            registry.workers_for(&model).is_empty(),
            "removed worker must not re-enter the model pool",
        );

        drop(tx);
        let _ = timeout(Duration::from_secs(2), manager_handle).await;
        let _ = shutdown_tx.send(());
    }

    /// Single-flight: while one re-introspection for a worker is in-flight,
    /// subsequent reconcile ticks must NOT spawn a second `/server_info`
    /// fetch for the same id (the `pending` guard). A slow, never-resolving
    /// worker is hammered by ticks faster than the fetch completes; the max
    /// observed concurrency must stay at 1.
    #[tokio::test]
    async fn reconcile_does_not_stack_concurrent_introspections_per_worker() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        let in_flight_h = in_flight.clone();
        let max_h = max_in_flight.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route(
            "/server_info",
            get(move || {
                let in_flight = in_flight_h.clone();
                let max = max_h.clone();
                async move {
                    let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    max.fetch_max(cur, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    // No served_model_name => worker stays unresolved, so
                    // reconcile keeps trying every interval.
                    Json(json!({}))
                }
            }),
        );
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        let url = format!("http://127.0.0.1:{port}");

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        // Timeout above the 200ms handler so each fetch completes.
        let introspector = Arc::new(WorkerIntrospector::new(Duration::from_secs(2)));
        // Reconcile faster than the handler so ticks pile up against one
        // in-flight fetch.
        let manager_handle = tokio::spawn(run_with_introspector_and_reconcile(
            rx,
            registry.clone(),
            None,
            None,
            None,
            introspector,
            Duration::from_millis(60),
        ));

        tx.send(DiscoveryEvent::Added(WorkerSpec {
            id: WorkerId("w-stuck".into()),
            url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
        }))
        .await
        .unwrap();

        // Several reconcile intervals elapse while the handler is slow.
        tokio::time::sleep(Duration::from_millis(900)).await;

        assert_eq!(
            max_in_flight.load(Ordering::SeqCst),
            1,
            "reconcile must keep re-introspection single-flight per worker; \
             the `pending` guard should prevent stacking concurrent /server_info calls",
        );

        drop(tx);
        let _ = tokio::time::timeout(Duration::from_secs(2), manager_handle).await;
        let _ = shutdown_tx.send(());
    }

    /// A worker that resolved on its initial `Added` must NOT be
    /// re-introspected by later reconcile ticks — the `model_ids.is_empty()`
    /// skip is the steady-state cost guarantee. Asserts the `/server_info`
    /// hit count stays at the single onboarding fetch across many ticks.
    #[tokio::test]
    async fn reconcile_skips_workers_that_already_resolved() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::time::timeout;

        let hits = Arc::new(AtomicUsize::new(0));
        let hits_h = hits.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route(
            "/server_info",
            get(move || {
                let hits = hits_h.clone();
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    Json(json!({"served_model_name": "m"}))
                }
            }),
        );
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        let url = format!("http://127.0.0.1:{port}");

        let registry = Arc::new(WorkerRegistry::default());
        let (tx, rx) = mpsc::channel::<DiscoveryEvent>(8);
        let manager_handle = tokio::spawn(run_with_introspector_and_reconcile(
            rx,
            registry.clone(),
            None,
            None,
            None,
            fast_introspector(),
            Duration::from_millis(80),
        ));

        let id = WorkerId("w-resolved".into());
        tx.send(DiscoveryEvent::Added(WorkerSpec {
            id: id.clone(),
            url,
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
        }))
        .await
        .unwrap();

        // Wait until it resolves into the model pool.
        let resolved = timeout(Duration::from_secs(2), async {
            loop {
                if !registry.workers_for(&ModelId("m".into())).is_empty() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            resolved.is_ok(),
            "worker should resolve on the initial Added"
        );

        // Let many reconcile intervals (80ms) pass.
        tokio::time::sleep(Duration::from_millis(500)).await;

        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "a resolved worker must not be re-introspected by reconcile; only the \
             initial Added fetch should hit /server_info",
        );

        drop(tx);
        let _ = manager_handle.await;
        let _ = shutdown_tx.send(());
    }
}
