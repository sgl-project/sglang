// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::Parser;
use sgl_router::config::{CachePrefixProvider, Cli, LogFormat, PolicyKind};
use std::sync::Arc;
use tokio::signal::unix::{signal, Signal, SignalKind};

/// Install the global tracing subscriber.
///
/// Idempotent: a second call returns `Ok` without panicking. When
/// `try_init` errors, some other code has already installed a subscriber,
/// so the `tracing::debug!` below is delivered through THAT subscriber —
/// no recursive init.
///
/// `format` selects the output shape: `Json` emits one JSON record per
/// line (target for production / k8s log aggregators), `Text` is the
/// human-readable default. The `RUST_LOG` environment variable always
/// wins over `default_level`.
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
        // A second install attempt; the existing subscriber is fine.
        // Surface the attempted default level so an operator can see
        // what we tried.
        tracing::debug!(
            default_level = %default_level,
            ?format,
            error = %e,
            "tracing subscriber already installed; continuing"
        );
    }
    Ok(())
}

/// Install a minimal text-format subscriber BEFORE config resolution so a
/// config-resolution error has somewhere to surface. The real subscriber
/// (driven by `Config.observability`) is installed after; the second
/// `try_init` is a no-op because a subscriber is already present.
/// The bootstrap subscriber respects `RUST_LOG` so an operator can
/// debug startup with `RUST_LOG=debug` even when configuration resolution
/// fails.
fn install_bootstrap_subscriber() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

/// How often to report progress while axum drains in-flight requests. That
/// phase is unbounded, so without a heartbeat a pod SIGKILLed at
/// `terminationGracePeriodSeconds` leaves no evidence of what it was waiting on.
const DRAIN_HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// Install SIGTERM and SIGINT handlers up front so a failure here surfaces
/// before `axum::serve` starts. If installation fails (rare: container
/// without signal capability, seccomp policy), we return an error and the
/// process exits cleanly rather than running deaf to k8s termination.
fn install_signal_handlers() -> Result<(Signal, Signal)> {
    let sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    Ok((sigterm, sigint))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    // Bootstrap subscriber so a config-resolution error has structured
    // output. The configured-format subscriber installs after this and
    // becomes a no-op via try_init's idempotency.
    install_bootstrap_subscriber();
    let cfg = cli
        .into_config()
        .context("resolve configuration from CLI flags")?;

    init_tracing(&cfg.observability.log_level, cfg.observability.log_format)?;

    // Emitted here rather than from `Config::validate`: this is startup advice
    // about the deployment, not a validation failure, and keeping it out of
    // `validate` leaves that function free of side effects. It also runs after
    // the configured subscriber is installed, and carries the value as a
    // structured field rather than only inside the message text.
    if let Some(msg) = sgl_router::config::shutdown_drain_advisory(cfg.server.shutdown_drain_secs) {
        tracing::warn!(
            shutdown_drain_secs = cfg.server.shutdown_drain_secs,
            "{msg}"
        );
    }

    tracing::info!(
        configured_decode_policy = ?cfg.model.decode_policy,
        "sgl-router {} starting on {}:{}",
        env!("CARGO_PKG_VERSION"),
        cfg.server.host,
        cfg.server.port
    );

    let tokenizers = Arc::new(
        sgl_router::tokenizer::TokenizerRegistry::load_from_config(&cfg)
            .context("load tokenizers")?,
    );

    let registry = Arc::new(sgl_router::workers::WorkerRegistry::default());
    let cache_aware_uses_indexer = cfg.model.policy == PolicyKind::CacheAware
        && cfg
            .model
            .cache_aware
            .as_ref()
            .is_some_and(|cache| cache.prefix_provider == CachePrefixProvider::Indexer);
    let prefix_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>> = cache_aware_uses_indexer
        .then_some(cfg.model.cache_aware.as_ref())
        .flatten()
        .and_then(|cache| cache.kv_indexer_endpoint.as_ref())
        .map(|indexer| {
            let config = prefix_index_config(indexer);
            sgl_kv_indexer::GrpcPrefixIndex::new(config)
                .map(|index| Arc::new(index) as Arc<dyn sgl_kv_indexer::PrefixIndex>)
                .context("configure KV Indexer client")
        })
        .transpose()?;

    // Build the local prefix index and block metadata used by the Radix Tree
    // provider. An external Indexer only needs hash metadata, so it does not
    // subscribe to the local KV-event stream.
    let block_size_oracle = sgl_router::policies::kv_events::BlockSizeOracle::new();
    let kv_event_http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .expect("default http client builds");
    let kv_index = if prefix_index.is_some() {
        sgl_router::policies::kv_events::KvEventIndex::new_metadata_only_with_http_and_oracle(
            kv_event_http,
            Arc::clone(&block_size_oracle),
        )
    } else {
        sgl_router::policies::kv_events::KvEventIndex::new_with_http_and_oracle(
            kv_event_http,
            Arc::clone(&block_size_oracle),
        )
    };
    let policies = Arc::new(
        sgl_router::policies::factory::build_registry(
            &cfg,
            kv_index.tree(),
            Arc::clone(&block_size_oracle),
        )
        .context("build policy registry")?,
    );

    // Shared ActiveLoadRegistry + janitor task. The janitor reaps
    // request entries whose lifetime exceeded `stale_request_timeout`,
    // so a leaked guard (proxy task panic, etc.) does not inflate a
    // worker's load forever. The registry is built BEFORE the manager
    // is spawned so the manager can call `forget_worker` on
    // `DiscoveryEvent::Removed`.
    let stale_timeout = std::time::Duration::from_secs(cfg.active_load.stale_request_timeout_secs);
    let active_load = sgl_router::policies::active_load::ActiveLoadRegistry::new(
        Arc::new(sgl_router::policies::active_load::SystemTimeClock),
        stale_timeout,
    );
    // Sweep cadence is 1/10 of the configured timeout, clamped to
    // [1 s, 60 s]. A short timeout (test setting) needs frequent
    // sweeps to fire within the test's window; a long timeout
    // (production) doesn't need sub-minute checks.
    let sweep_interval = std::time::Duration::from_secs(
        (cfg.active_load.stale_request_timeout_secs / 10).clamp(1, 60),
    );
    let janitor_handle =
        sgl_router::policies::active_load::spawn_janitor(Arc::clone(&active_load), sweep_interval);

    // Spawn discovery + manager tasks.
    let (event_rx, discovery_handle) = sgl_router::discovery::spawn_discovery(&cfg)
        .await
        .context("spawn discovery")?;
    let kv_index_opt: Option<Arc<sgl_router::policies::kv_events::KvEventIndex>> =
        Some(Arc::clone(&kv_index));
    let manager_handle = tokio::spawn(sgl_router::workers::manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        kv_index_opt,
        Some(Arc::clone(&active_load)),
    ));

    let proxy = Arc::new(
        sgl_router::proxy::Proxy::new(std::time::Duration::from_secs(
            cfg.proxy.request_timeout_secs,
        ))
        .context("build proxy client")?,
    );

    let mut app_ctx = sgl_router::server::app_context::AppContext::with_active_load(
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
    .then(|| {
        sgl_router::policies::prefix_provider::RadixTreePrefixProvider::new(
            kv_index.tree(),
            Arc::clone(&block_size_oracle),
        )
    });
    app_ctx.block_size_oracle = block_size_oracle;
    app_ctx.engine_load = kv_index.engine_load();
    let ctx = Arc::new(app_ctx);
    ctx.mark_ready();

    let app = sgl_router::server::app::build_router(ctx.clone());

    let bind = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!("listening on {bind}");

    let (sigterm, sigint) = install_signal_handlers()?;

    // Published the moment the readiness drain finishes, i.e. when axum starts
    // its in-flight drain. That phase — not the pause, and not the uptime
    // before it — is what the heartbeat below reports on.
    let (inflight_drain_tx, inflight_drain_rx) =
        tokio::sync::watch::channel(None::<std::time::Instant>);
    let shutdown_ctx = ctx.clone();
    let drain = cfg.server.shutdown_drain();
    let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
        shutdown_signal(sigterm, sigint, shutdown_ctx, drain).await;
        let _ = inflight_drain_tx.send(Some(std::time::Instant::now()));
    });

    let heartbeat_ctx = ctx.clone();
    let mut heartbeat_rx = inflight_drain_rx.clone();
    let heartbeat = tokio::spawn(async move {
        // Stay silent until the in-flight drain actually begins; an `Err` here
        // means the sender went away without one, so there is nothing to report.
        let Ok(started) = heartbeat_rx
            .wait_for(Option::is_some)
            .await
            .map(|at| at.expect("wait_for only resolves once the instant is published"))
        else {
            return;
        };
        let mut ticker = tokio::time::interval(DRAIN_HEARTBEAT_INTERVAL);
        // Delay, not the default Burst: the runtime stalling is exactly the
        // condition this heartbeat exists to report, and Burst would answer it
        // with a clump of back-dated ticks instead of one line per interval.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await; // the first tick completes immediately
        loop {
            ticker.tick().await;
            tracing::warn!(
                elapsed_secs = started.elapsed().as_secs(),
                // Proxied requests only: a connection axum is still holding for
                // another reason does not appear here, so 0 means "not waiting
                // on a worker", not "not waiting".
                inflight_proxied = heartbeat_ctx.active_load.inflight_count(),
                "still draining in-flight requests; this phase is unbounded and ends at \
                 SIGKILL when terminationGracePeriodSeconds expires",
            );
        }
    });
    let server_result = serve.await.context("axum serve");
    heartbeat.abort();
    let inflight_drain_secs = inflight_drain_rx.borrow().map(|at| at.elapsed().as_secs());

    // Best-effort: cancel discovery + manager + janitor on shutdown.
    // The janitor handle's drop signals cancellation; we additionally
    // await `shutdown` so the task joins cleanly before the process
    // exits — useful for tracing tail logs. `JanitorHandle::shutdown` caps its
    // own join at 2 s, so it cannot hang the exit — though those 2 s are still
    // charged to terminationGracePeriodSeconds.
    discovery_handle.abort();
    manager_handle.abort();
    janitor_handle.shutdown().await;
    // The ERROR arms exist because otherwise the log says "shutdown complete"
    // at INFO and the error leaves the process through `Termination`, never
    // through `tracing` — so a severity-based alert sees nothing wrong with a
    // crashed router. `None` means the server stopped without ever reaching the
    // drain, which is not the same as draining instantly, so it gets its own
    // message rather than `inflight_drain_secs = 0`.
    match (&server_result, inflight_drain_secs) {
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
    server_result
}

/// Build the external Indexer client with the Router's bounded query settings.
fn prefix_index_config(
    indexer: &sgl_router::config::KvIndexerEndpointConfig,
) -> sgl_kv_indexer::PrefixIndexConfig {
    sgl_kv_indexer::PrefixIndexConfig {
        endpoint: indexer.url.clone(),
        query_deadline: std::time::Duration::from_millis(indexer.query_timeout_ms),
        max_inflight: indexer.query_max_inflight,
    }
}

/// Resolve when a termination signal arrives, then hand control to axum's
/// graceful drain. On SIGTERM (k8s pod termination) first run the readiness
/// drain — flip `/readyz` to 503 and keep serving for `drain` so the endpoint
/// removal reaches kube-proxy before we stop accepting, closing the
/// rolling-update race. SIGINT (local Ctrl-C) skips the readiness drain
/// entirely — no 503 flip, no pause — so dev iteration does not pay it.
///
/// Either way axum's own in-flight drain runs afterwards and is unbounded: a
/// long streaming completion still holds the process until it finishes or
/// `terminationGracePeriodSeconds` expires. A further termination signal cuts
/// the pause short but cannot reach that phase; it is logged instead.
async fn shutdown_signal(
    mut sigterm: Signal,
    mut sigint: Signal,
    ctx: Arc<sgl_router::server::app_context::AppContext>,
    drain: std::time::Duration,
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

    let (expedite_tx, expedite_rx) = tokio::sync::oneshot::channel::<()>();
    // Only the SIGTERM path runs a pause, so only it has something to cut
    // short; on the SIGINT path the first further signal goes straight to the
    // warning below.
    let mut expedite_tx = sigterm_first.then_some(expedite_tx);
    // Hand both streams to a task that outlives this future, on EITHER branch.
    // Dropping them here would make every later signal vanish: tokio never
    // restores the default disposition, so the process would neither expedite
    // nor die, and nothing would be logged.
    tokio::spawn(async move {
        loop {
            let delivered = tokio::select! {
                delivered = sigterm.recv() => delivered,
                delivered = sigint.recv() => delivered,
            };
            if delivered.is_none() {
                // The signal driver is gone (runtime shutting down). Looping
                // would spin without ever receiving again.
                return;
            }
            match expedite_tx.take() {
                // The first further signal cuts the readiness pause short, so
                // an operator watching a stuck rollout is not held for a window
                // that has stopped being useful.
                Some(tx) => {
                    let _ = tx.send(());
                }
                // Past the pause there is nothing left to cut short — say so
                // rather than swallowing the signal silently.
                None => tracing::warn!(
                    "further termination signal ignored: the readiness pause is over \
                     and the in-flight drain cannot be cut short; send SIGKILL to \
                     force an immediate exit",
                ),
            }
        }
    });

    if sigterm_first {
        let expedite = async move {
            let _ = expedite_rx.await;
        };
        sgl_router::server::shutdown::drain_for_termination(&ctx, drain, expedite).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_index_config_preserves_router_limits() {
        let config = prefix_index_config(&sgl_router::config::KvIndexerEndpointConfig {
            url: "http://127.0.0.1:50051".to_string(),
            query_timeout_ms: 25,
            query_max_inflight: 17,
        });
        assert_eq!(config.endpoint, "http://127.0.0.1:50051");
        assert_eq!(config.query_deadline, std::time::Duration::from_millis(25));
        assert_eq!(config.max_inflight, 17);
    }

    #[tokio::test]
    async fn install_signal_handlers_returns_both() {
        // Pins the contract that handler installation works on a standard
        // tokio runtime. If this fails on a sandboxed runner, the real
        // service would also fail to install — which is the point.
        assert!(install_signal_handlers().is_ok());
    }

    #[test]
    fn init_tracing_is_idempotent() {
        let _ = init_tracing("info", LogFormat::Text);
        let _ = init_tracing("info", LogFormat::Text);
    }

    #[test]
    fn init_tracing_accepts_json_format() {
        // Doesn't matter whether we win or lose the race against another
        // subscriber install — the function must return Ok either way.
        assert!(init_tracing("info", LogFormat::Json).is_ok());
    }
}
