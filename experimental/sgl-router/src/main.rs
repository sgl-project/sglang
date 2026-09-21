// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::Parser;
use sgl_kv_indexer::{GrpcPrefixIndex, PrefixIndex, PrefixIndexConfig};
use sgl_router::{
    config::{CachePrefixProvider, Cli, Config, KvIndexerEndpointConfig, LogFormat, PolicyKind},
    discovery::spawn_discovery,
    policies::{
        factory::build_registry as build_policy_registry, prefix_provider::RadixTreePrefixProvider,
        PolicyRegistry,
    },
    proxy::Proxy,
    server::{app::build_router, app_context::AppContext, shutdown::drain_for_termination},
    state::{
        kv_events::{BlockSizeOracle, KvEventIndex},
        load_monitor::router_inflight_load::{
            spawn_janitor, JanitorHandle, RouterInflightLoadRegistry, SystemTimeClock,
        },
    },
    tokenizer::TokenizerRegistry,
    workers::{manager, WorkerRegistry},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    net::TcpListener,
    signal::unix::{signal, Signal, SignalKind},
    sync::{oneshot, watch},
    task::JoinHandle,
};

const DRAIN_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
/// Heartbeat escalates INFO -> WARN here: earlier is a routine rollout draining
/// a long response; later the pod risks SIGKILL with work still open.
const DRAIN_WARN_AFTER: Duration = Duration::from_secs(30);

// Main components started by this binary:
// - Engine monitor (`KvEventIndex`): receives load statistics over ZMQ and, without a
//   remote KV indexer, KV events to maintain a local radix tree. Remote prefix lookups
//   use `GrpcPrefixIndex` over gRPC.
// - Engine discovery (`spawn_discovery`): watches Kubernetes pods or loads static URLs,
//   sending `DiscoveryEvent`s to `manager::run_with_config` to update `WorkerRegistry`.
// - HTTP server (`axum::serve`): serves OpenAI APIs, health/readiness, and metrics
//   through routes built by `build_router`, sharing state via `AppContext`.
#[tokio::main]
async fn main() -> Result<()> {
    // Resolve CLI configuration and set up startup logging.
    let cli = Cli::parse();
    install_bootstrap_subscriber();
    let config = cli
        .into_config()
        .context("resolve configuration from CLI flags")?;
    init_tracing(
        &config.observability.log_level,
        config.observability.log_format,
    )?;

    // Buffer termination signals before tokenizer loading or discovery can block startup.
    let (sigterm, sigint) = install_signal_handlers()?;
    log_startup(&config);

    // Load tokenizers used to prepare requests for routing.
    let tokenizers =
        Arc::new(TokenizerRegistry::load_from_config(&config).context("load tokenizers")?);

    // (Optional) Create a gRPC client only when routing uses an external KV indexer.
    let external_kv_indexer_client = create_external_kv_indexer_client(&config)?;

    // Monitor engine-reported KV-cache events and load statistics for routing.
    let engine_state = start_engine_state_monitor(external_kv_indexer_client.is_some());

    // Build the policies that choose which workers receive each request.
    let routing_policies = Arc::new(
        build_policy_registry(
            &config,
            engine_state.tree(),
            engine_state.block_size_oracle(),
        )
        .context("build policy registry")?,
    );

    // Track this router's local view of in-flight requests.
    let (local_inflight_requests, inflight_cleanup) = start_local_inflight_tracker(&config);

    // Discovery feeds worker changes to the manager, which maintains this routing catalog.
    let worker_registry = Arc::new(WorkerRegistry::default());
    let (discovery_handle, worker_manager_handle) = start_worker_discovery_and_manager(
        &config,
        &worker_registry,
        &engine_state,
        &local_inflight_requests,
    )
    .await?;

    // Share routing dependencies with HTTP handlers and mark startup complete.
    let app_context = build_app_context(
        &config,
        tokenizers,
        worker_registry,
        routing_policies,
        local_inflight_requests,
        &engine_state,
        external_kv_indexer_client,
    )?;
    app_context.mark_ready();

    // Serve HTTP requests until shutdown, allowing in-flight requests to finish.
    let listen_addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("bind {listen_addr}"))?;
    tracing::info!("listening on {listen_addr}");
    let outcome = serve(listener, app_context, sigterm, sigint).await;

    // Stop background tasks once the HTTP server has finished draining.
    discovery_handle.abort();
    worker_manager_handle.abort();
    inflight_cleanup.shutdown().await;
    log_shutdown(&outcome.result, outcome.inflight_drain_secs);
    outcome.result
}

// Respect RUST_LOG and tolerate an already-installed subscriber.
fn init_tracing(default_level: &str, format: LogFormat) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    let install_result = match format {
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .json()
            .try_init(),
        LogFormat::Text => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .try_init(),
    };
    if let Err(e) = install_result {
        tracing::debug!(
            default_level = %default_level,
            ?format,
            error = %e,
            "tracing subscriber already installed; continuing"
        );
    }
    Ok(())
}

// Provide startup logging before configuration resolution; later installs are no-ops.
fn install_bootstrap_subscriber() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

fn install_signal_handlers() -> Result<(Signal, Signal)> {
    let sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    Ok((sigterm, sigint))
}

fn log_startup(config: &Config) {
    if let Some(advisory) = sgl_router::config::shutdown_drain_advisory(
        config.server.shutdown_drain_secs,
        config.server.termination_grace_secs,
    ) {
        tracing::warn!(
            shutdown_drain_secs = advisory.shutdown_drain_secs,
            termination_grace_secs = advisory.termination_grace_secs,
            grace_declared = advisory.grace_declared,
            "shutdown drain leaves no room under terminationGracePeriodSeconds for the \
             in-flight drain that follows it; raise the grace period to at least the drain \
             plus in-flight request time, or lower the drain. If the grace period is already \
             higher, declare it with --termination-grace-secs",
        );
    }

    tracing::info!(
        configured_decode_policy = ?config.model.decode_policy,
        "sgl-router {} starting on {}:{}",
        env!("CARGO_PKG_VERSION"),
        config.server.host,
        config.server.port
    );
}

fn create_external_kv_indexer_client(config: &Config) -> Result<Option<Arc<dyn PrefixIndex>>> {
    let endpoint = config
        .model
        .cache_aware
        .as_ref()
        .filter(|cache| {
            config.model.policy == PolicyKind::CacheAware
                && cache.prefix_provider == CachePrefixProvider::Indexer
        })
        .and_then(|cache| cache.kv_indexer_endpoint.as_ref());
    endpoint
        .map(|endpoint| {
            GrpcPrefixIndex::new(prefix_index_config(endpoint))
                .map(|index| Arc::new(index) as Arc<dyn PrefixIndex>)
                .context("configure KV Indexer client")
        })
        .transpose()
}

fn prefix_index_config(indexer: &KvIndexerEndpointConfig) -> PrefixIndexConfig {
    PrefixIndexConfig {
        endpoint: indexer.url.clone(),
        query_deadline: Duration::from_millis(indexer.query_timeout_ms),
        max_inflight: indexer.query_max_inflight,
    }
}

fn start_engine_state_monitor(use_external_indexer: bool) -> Arc<KvEventIndex> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("default http client builds");
    if use_external_indexer {
        // External indexing still needs worker hash metadata and engine load, but no local KV tree.
        KvEventIndex::new_metadata_only_with_http_and_oracle(http, BlockSizeOracle::new())
    } else {
        KvEventIndex::new_with_http(http)
    }
}

fn start_local_inflight_tracker(
    config: &Config,
) -> (Arc<RouterInflightLoadRegistry>, JanitorHandle) {
    let timeout_secs = config.router_inflight_load.stale_request_timeout_secs;
    let local_inflight_requests = RouterInflightLoadRegistry::new(
        Arc::new(SystemTimeClock),
        Duration::from_secs(timeout_secs),
    );
    // Reap stale requests at one tenth of their timeout, bounded to 1–60 seconds.
    let sweep_interval = Duration::from_secs((timeout_secs / 10).clamp(1, 60));
    let inflight_cleanup = spawn_janitor(Arc::clone(&local_inflight_requests), sweep_interval);
    (local_inflight_requests, inflight_cleanup)
}

async fn start_worker_discovery_and_manager(
    config: &Config,
    worker_registry: &Arc<WorkerRegistry>,
    engine_state: &Arc<KvEventIndex>,
    local_inflight_requests: &Arc<RouterInflightLoadRegistry>,
) -> Result<(JoinHandle<()>, JoinHandle<()>)> {
    let (worker_events, discovery_handle) =
        spawn_discovery(config).await.context("spawn discovery")?;
    // Keep engine subscriptions and local request counters in sync with worker membership.
    let worker_manager_handle = tokio::spawn(manager::run_with_config(
        worker_events,
        Arc::clone(worker_registry),
        Some(Arc::new(config.clone())),
        Some(Arc::clone(engine_state)),
        Some(Arc::clone(local_inflight_requests)),
    ));
    Ok((discovery_handle, worker_manager_handle))
}

fn build_app_context(
    config: &Config,
    tokenizers: Arc<TokenizerRegistry>,
    worker_registry: Arc<WorkerRegistry>,
    routing_policies: Arc<PolicyRegistry>,
    local_inflight_requests: Arc<RouterInflightLoadRegistry>,
    engine_state: &KvEventIndex,
    external_kv_indexer_client: Option<Arc<dyn PrefixIndex>>,
) -> Result<Arc<AppContext>> {
    let block_size_oracle = engine_state.block_size_oracle();
    let proxy = Arc::new(
        Proxy::new(Duration::from_secs(config.proxy.request_timeout_secs))
            .context("build proxy client")?,
    );

    let mut app_context = AppContext::with_router_inflight_load(
        config.clone(),
        tokenizers,
        proxy,
        worker_registry,
        routing_policies,
        local_inflight_requests,
    );
    app_context.prefix_index = external_kv_indexer_client;
    app_context.radix_tree_prefix_provider = (config.model.policy == PolicyKind::CacheAware
        && config
            .model
            .cache_aware
            .as_ref()
            .is_some_and(|cache| cache.prefix_provider == CachePrefixProvider::RadixTree))
    .then(|| RadixTreePrefixProvider::new(engine_state.tree(), Arc::clone(&block_size_oracle)));
    app_context.block_size_oracle = block_size_oracle;
    app_context.engine_reported_load = engine_state.engine_reported_load();
    app_context.kv_metrics = engine_state.metrics_source();
    Ok(Arc::new(app_context))
}

/// How serving ended; `inflight_drain_secs` is `None` when the server stopped
/// without ever reaching the in-flight drain.
struct ServeOutcome {
    result: Result<()>,
    inflight_drain_secs: Option<u64>,
}

async fn serve(
    listener: TcpListener,
    app_context: Arc<AppContext>,
    sigterm: Signal,
    sigint: Signal,
) -> ServeOutcome {
    let app = build_router(Arc::clone(&app_context));
    let drain = app_context.config.server.shutdown_drain();
    let (drain_tx, drain_rx) = watch::channel(None);
    let heartbeat = tokio::spawn(report_drain_progress(
        Arc::clone(&app_context),
        drain_rx.clone(),
    ));
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal(sigterm, sigint, app_context, drain).await;
            let _ = drain_tx.send(Some(Instant::now()));
        })
        .await
        .context("axum serve");
    heartbeat.abort();
    let inflight_drain_secs = drain_rx.borrow().map(|at| at.elapsed().as_secs());
    ServeOutcome {
        result,
        inflight_drain_secs,
    }
}

async fn report_drain_progress(
    app_context: Arc<AppContext>,
    mut drain_rx: watch::Receiver<Option<Instant>>,
) {
    // Start reporting only after the readiness pause, when axum begins draining requests.
    let Ok(started) = drain_rx
        .wait_for(Option::is_some)
        .await
        .map(|at| at.expect("wait_for only resolves once the instant is published"))
    else {
        return;
    };
    let mut ticker = tokio::time::interval(DRAIN_HEARTBEAT_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ticker.tick().await; // the first tick completes immediately

    macro_rules! heartbeat {
        ($level:ident, $elapsed:expr) => {
            tracing::$level!(
                elapsed_secs = $elapsed,
                inflight_http = app_context.inflight_http.count(),
                inflight_proxied = app_context.router_inflight_load.inflight_count(),
                "still draining in-flight requests; this phase is unbounded and ends at \
                 SIGKILL when terminationGracePeriodSeconds expires",
            )
        };
    }
    loop {
        ticker.tick().await;
        let elapsed = started.elapsed();
        if elapsed < DRAIN_WARN_AFTER {
            heartbeat!(info, elapsed.as_secs());
        } else {
            heartbeat!(warn, elapsed.as_secs());
        }
    }
}

fn log_shutdown(result: &Result<()>, inflight_drain_secs: Option<u64>) {
    match (result, inflight_drain_secs) {
        (Ok(()), Some(inflight_drain_secs)) => {
            tracing::info!(inflight_drain_secs, "shutdown complete")
        }
        (Ok(()), None) => {
            tracing::info!("shutdown complete; the server stopped without a termination signal")
        }
        (Err(e), Some(inflight_drain_secs)) => tracing::error!(
            error = %e,
            inflight_drain_secs,
            "shutdown complete, but the server exited with an error",
        ),
        (Err(e), None) => tracing::error!(
            error = %e,
            "the server exited with an error before any termination signal",
        ),
    }
}

// SIGTERM pauses for readiness propagation; SIGINT goes straight to the in-flight drain.
async fn shutdown_signal(
    mut sigterm: Signal,
    mut sigint: Signal,
    app_context: Arc<AppContext>,
    drain: Duration,
) {
    let sigterm_first = tokio::select! {
        _ = sigterm.recv() => {
            tracing::info!("got SIGTERM, shutting down");
            true
        }
        _ = sigint.recv() => {
            tracing::info!("got SIGINT, shutting down without the readiness drain");
            false
        }
    };

    let (expedite_tx, expedite_rx) = oneshot::channel::<()>();
    let mut expedite_tx = sigterm_first.then_some(expedite_tx);
    // Keep consuming signals during the in-flight drain; tokio never restores default handlers.
    tokio::spawn(async move {
        loop {
            let delivered = tokio::select! {
                delivered = sigterm.recv() => delivered,
                delivered = sigint.recv() => delivered,
            };
            if delivered.is_none() {
                return;
            }
            handle_further_signal(&mut expedite_tx, sigterm_first);
        }
    });

    if sigterm_first {
        let expedite = async move {
            let _ = expedite_rx.await;
        };
        drain_for_termination(&app_context, drain, expedite).await;
    }
}

#[derive(Debug, PartialEq, Eq)]
enum FurtherSignal {
    Expedited,
    Ignored,
}

fn handle_further_signal(
    expedite_tx: &mut Option<oneshot::Sender<()>>,
    sigterm_first: bool,
) -> FurtherSignal {
    // A failed `send` means the pause already elapsed; it must fall through to
    // the notice below. Discarding the `Err` once swallowed the first
    // post-pause signal (see the regression test).
    if let Some(tx) = expedite_tx.take() {
        if tx.send(()).is_ok() {
            return FurtherSignal::Expedited;
        }
    }
    if sigterm_first {
        tracing::warn!(
            "further termination signal ignored: the readiness pause is over and the \
             in-flight drain cannot be cut short; send SIGKILL to force an immediate exit",
        );
    } else {
        tracing::warn!(
            "further termination signal ignored: SIGINT skips the readiness pause and the \
             in-flight drain cannot be cut short; send SIGKILL to force an immediate exit",
        );
    }
    FurtherSignal::Ignored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_index_config_preserves_router_limits() {
        let config = prefix_index_config(&KvIndexerEndpointConfig {
            url: "http://127.0.0.1:50051".to_string(),
            query_timeout_ms: 25,
            query_max_inflight: 17,
        });
        assert_eq!(config.endpoint, "http://127.0.0.1:50051");
        assert_eq!(config.query_deadline, Duration::from_millis(25));
        assert_eq!(config.max_inflight, 17);
    }

    #[tokio::test]
    async fn install_signal_handlers_returns_both() {
        assert!(install_signal_handlers().is_ok());
    }

    #[test]
    fn a_further_signal_expedites_a_running_pause() {
        let (tx, mut rx) = oneshot::channel::<()>();
        let mut expedite_tx = Some(tx);
        assert_eq!(
            handle_further_signal(&mut expedite_tx, true),
            FurtherSignal::Expedited,
        );
        assert!(
            expedite_tx.is_none(),
            "the sender is spent once the pause has been expedited",
        );
        assert!(
            rx.try_recv().is_ok(),
            "the running pause must actually be notified",
        );
    }

    #[test]
    fn the_first_signal_after_the_pause_ends_is_reported_not_swallowed() {
        let (tx, rx) = oneshot::channel::<()>();
        drop(rx); // the pause elapsed and dropped its receiver
        let mut expedite_tx = Some(tx);
        assert_eq!(
            handle_further_signal(&mut expedite_tx, true),
            FurtherSignal::Ignored,
            "a send into a dropped receiver expedites nothing and must say so",
        );
        assert_eq!(
            handle_further_signal(&mut expedite_tx, true),
            FurtherSignal::Ignored,
        );
    }

    #[test]
    fn a_further_signal_on_the_sigint_path_is_always_ignored() {
        let mut expedite_tx = None;
        assert_eq!(
            handle_further_signal(&mut expedite_tx, false),
            FurtherSignal::Ignored,
        );
    }

    #[test]
    fn init_tracing_is_idempotent() {
        let _ = init_tracing("info", LogFormat::Text);
        let _ = init_tracing("info", LogFormat::Text);
    }

    #[test]
    fn init_tracing_accepts_json_format() {
        assert!(init_tracing("info", LogFormat::Json).is_ok());
    }
}
