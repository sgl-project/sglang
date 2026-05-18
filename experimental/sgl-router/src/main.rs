// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal::unix::{signal, Signal, SignalKind};

#[derive(Parser, Debug)]
#[command(name = "sgl-router", version)]
struct Cli {
    #[arg(long, env = "SGL_ROUTER_CONFIG")]
    config: PathBuf,
}

/// Install the global tracing subscriber.
///
/// Idempotent: a second call returns `Ok` without panicking. When
/// `try_init` errors, some other code has already installed a subscriber,
/// so the `tracing::debug!` below is delivered through THAT subscriber —
/// no recursive init.
fn init_tracing(default_level: &str) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    if let Err(e) = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init()
    {
        // A second install attempt; the existing subscriber is fine.
        // Surface the attempted default level so an operator can see
        // what we tried.
        tracing::debug!(
            default_level = %default_level,
            error = %e,
            "tracing subscriber already installed; continuing"
        );
    }
    Ok(())
}

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
    let cfg = sgl_router::config::Config::from_path(&cli.config)
        .with_context(|| format!("load config from {}", cli.config.display()))?;

    init_tracing(&cfg.observability.log_level)?;

    tracing::info!(
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

    // Build the KV-event index up front so the cache-aware-zmq policy can
    // share its `HashTree` handle. When no model uses `cache_aware_zmq`, the
    // index is still constructed (cheap) but no subscribers are ever added.
    let kv_index = sgl_router::policies::kv_events::KvEventIndex::new();
    let policies = Arc::new(
        sgl_router::policies::factory::build_registry(
            &cfg,
            kv_index.tree(),
            Arc::clone(&tokenizers),
        )
        .context("build policy registry")?,
    );

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
    ));

    let proxy = Arc::new(
        sgl_router::proxy::Proxy::new(std::time::Duration::from_secs(60))
            .context("build proxy client")?,
    );

    let ctx = Arc::new(sgl_router::server::app_context::AppContext::new(
        cfg.clone(),
        tokenizers,
        proxy,
        registry,
        policies,
    ));
    ctx.mark_ready();

    let app = sgl_router::server::app::build_router(ctx.clone());

    let bind = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!("listening on {bind}");

    let (sigterm, sigint) = install_signal_handlers()?;

    let serve = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(sigterm, sigint));
    let server_result = serve.await.context("axum serve");

    // Best-effort: cancel discovery + manager on shutdown.
    discovery_handle.abort();
    manager_handle.abort();
    server_result
}

async fn shutdown_signal(mut sigterm: Signal, mut sigint: Signal) {
    tokio::select! {
        _ = sigterm.recv() => tracing::info!("got SIGTERM, shutting down"),
        _ = sigint.recv()  => tracing::info!("got SIGINT, shutting down"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn install_signal_handlers_returns_both() {
        // Pins the contract that handler installation works on a standard
        // tokio runtime. If this fails on a sandboxed runner, the real
        // service would also fail to install — which is the point.
        assert!(install_signal_handlers().is_ok());
    }

    #[test]
    fn init_tracing_is_idempotent() {
        let _ = init_tracing("info");
        let _ = init_tracing("info");
    }
}
