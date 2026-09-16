// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::Parser;
use sgl_kv_indexer::{GrpcPrefixIndex, PrefixIndex, PrefixIndexConfig};
use sgl_router::{
    config::{CachePrefixProvider, Cli, Config, KvIndexerEndpointConfig, LogFormat, PolicyKind},
    discovery::spawn_discovery,
    policies::{
        active_load::{spawn_janitor, ActiveLoadRegistry, JanitorHandle, SystemTimeClock},
        factory::build_registry,
        kv_events::{BlockSizeOracle, KvEventIndex},
        prefix_provider::RadixTreePrefixProvider,
        PolicyRegistry,
    },
    proxy::Proxy,
    server::{app::build_router, app_context::AppContext, shutdown::drain_for_termination},
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
const DRAIN_WARN_AFTER: Duration = Duration::from_secs(30);

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    install_bootstrap_subscriber();
    let cfg = cli
        .into_config()
        .context("resolve configuration from CLI flags")?;
    init_tracing(&cfg.observability.log_level, cfg.observability.log_format)?;
    // Buffer termination signals before tokenizer loading or discovery can block startup.
    let (sigterm, sigint) = install_signal_handlers()?;
    log_startup(&cfg);

    let tokenizers =
        Arc::new(TokenizerRegistry::load_from_config(&cfg).context("load tokenizers")?);
    let registry = Arc::new(WorkerRegistry::default());
    let prefix_index = build_prefix_index(&cfg)?;
    let kv_index = start_kv_events(prefix_index.is_some());
    let policies = Arc::new(
        build_registry(&cfg, kv_index.tree(), kv_index.block_size_oracle())
            .context("build policy registry")?,
    );
    let (active_load, janitor_handle) = start_load_monitor(&cfg);
    let (discovery_handle, manager_handle) =
        start_worker_discovery(&cfg, &registry, &kv_index, &active_load).await?;
    let ctx = build_app_context(
        &cfg,
        tokenizers,
        registry,
        policies,
        active_load,
        &kv_index,
        prefix_index,
    )?;
    ctx.mark_ready();

    let bind = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!("listening on {bind}");
    let (server_result, inflight_drain_secs) = serve(listener, ctx, sigterm, sigint).await;

    discovery_handle.abort();
    manager_handle.abort();
    janitor_handle.shutdown().await;
    log_shutdown(&server_result, inflight_drain_secs);
    server_result
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

fn log_startup(cfg: &Config) {
    if let Some(advisory) = sgl_router::config::shutdown_drain_advisory(
        cfg.server.shutdown_drain_secs,
        cfg.server.termination_grace_secs,
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
        configured_decode_policy = ?cfg.model.decode_policy,
        "sgl-router {} starting on {}:{}",
        env!("CARGO_PKG_VERSION"),
        cfg.server.host,
        cfg.server.port
    );
}

fn build_prefix_index(cfg: &Config) -> Result<Option<Arc<dyn PrefixIndex>>> {
    let endpoint = cfg
        .model
        .cache_aware
        .as_ref()
        .filter(|cache| {
            cfg.model.policy == PolicyKind::CacheAware
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

fn start_kv_events(use_external_indexer: bool) -> Arc<KvEventIndex> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("default http client builds");
    if use_external_indexer {
        // External indexing still needs worker hash metadata, but no local KV-event tree.
        KvEventIndex::new_metadata_only_with_http_and_oracle(http, BlockSizeOracle::new())
    } else {
        KvEventIndex::new_with_http(http)
    }
}

fn start_load_monitor(cfg: &Config) -> (Arc<ActiveLoadRegistry>, JanitorHandle) {
    let timeout_secs = cfg.active_load.stale_request_timeout_secs;
    let active_load =
        ActiveLoadRegistry::new(Arc::new(SystemTimeClock), Duration::from_secs(timeout_secs));
    // Reap stale requests at one tenth of their timeout, bounded to 1–60 seconds.
    let sweep_interval = Duration::from_secs((timeout_secs / 10).clamp(1, 60));
    let handle = spawn_janitor(Arc::clone(&active_load), sweep_interval);
    (active_load, handle)
}

async fn start_worker_discovery(
    cfg: &Config,
    registry: &Arc<WorkerRegistry>,
    kv_index: &Arc<KvEventIndex>,
    active_load: &Arc<ActiveLoadRegistry>,
) -> Result<(JoinHandle<()>, JoinHandle<()>)> {
    let (event_rx, discovery_handle) = spawn_discovery(cfg).await.context("spawn discovery")?;
    let manager_handle = tokio::spawn(manager::run_with_config(
        event_rx,
        Arc::clone(registry),
        Some(Arc::new(cfg.clone())),
        Some(Arc::clone(kv_index)),
        Some(Arc::clone(active_load)),
    ));
    Ok((discovery_handle, manager_handle))
}

fn build_app_context(
    cfg: &Config,
    tokenizers: Arc<TokenizerRegistry>,
    registry: Arc<WorkerRegistry>,
    policies: Arc<PolicyRegistry>,
    active_load: Arc<ActiveLoadRegistry>,
    kv_index: &KvEventIndex,
    prefix_index: Option<Arc<dyn PrefixIndex>>,
) -> Result<Arc<AppContext>> {
    let block_size_oracle = kv_index.block_size_oracle();
    let proxy = Arc::new(
        Proxy::new(Duration::from_secs(cfg.proxy.request_timeout_secs))
            .context("build proxy client")?,
    );

    let mut app_ctx = AppContext::with_active_load(
        cfg.clone(),
        tokenizers,
        proxy,
        registry,
        policies,
        active_load,
    );
    app_ctx.prefix_index = prefix_index;
    app_ctx.radix_tree_prefix_provider = (cfg.model.policy == PolicyKind::CacheAware
        && cfg
            .model
            .cache_aware
            .as_ref()
            .is_some_and(|cache| cache.prefix_provider == CachePrefixProvider::RadixTree))
    .then(|| RadixTreePrefixProvider::new(kv_index.tree(), Arc::clone(&block_size_oracle)));
    app_ctx.block_size_oracle = block_size_oracle;
    app_ctx.engine_load = kv_index.engine_load();
    app_ctx.kv_metrics = kv_index.metrics_source();
    Ok(Arc::new(app_ctx))
}

async fn serve(
    listener: TcpListener,
    ctx: Arc<AppContext>,
    sigterm: Signal,
    sigint: Signal,
) -> (Result<()>, Option<u64>) {
    let app = build_router(Arc::clone(&ctx));
    let drain = ctx.config.server.shutdown_drain();
    let (drain_tx, drain_rx) = watch::channel(None);
    let heartbeat = tokio::spawn(report_drain_progress(Arc::clone(&ctx), drain_rx.clone()));
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal(sigterm, sigint, ctx, drain).await;
            let _ = drain_tx.send(Some(Instant::now()));
        })
        .await
        .context("axum serve");
    heartbeat.abort();
    let inflight_drain_secs = drain_rx.borrow().map(|at| at.elapsed().as_secs());
    (result, inflight_drain_secs)
}

async fn report_drain_progress(
    ctx: Arc<AppContext>,
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
                inflight_http = ctx.inflight_http.count(),
                inflight_proxied = ctx.active_load.inflight_count(),
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
    ctx: Arc<AppContext>,
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
        drain_for_termination(&ctx, drain, expedite).await;
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
    // A closed receiver means the readiness pause already ended.
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
