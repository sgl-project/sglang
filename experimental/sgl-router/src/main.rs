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
/// Uses `try_init` so a second call (e.g. from a test harness that already
/// installed a subscriber, or from `main` being pulled in twice) is a no-op
/// rather than a panic. Returns `Ok(())` on first successful install AND on
/// the "already installed" case — both mean tracing works going forward.
fn init_tracing(default_level: &str) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    // `try_init` returns Err if a global subscriber is already set. Treat
    // that as success: tracing already works, no panic, no double-install.
    if let Err(e) = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init()
    {
        // Can't use `tracing` here — install failed. Use stderr.
        eprintln!("tracing subscriber already initialized: {e}");
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
    // Default 60s request timeout; per-worker override via `request_timeout_ms`
    // in config. Streaming endpoints do NOT apply this timeout (long
    // generations are valid).
    let request_timeout =
        std::time::Duration::from_millis(cfg.workers[0].request_timeout_ms.unwrap_or(60_000));
    let proxy = Arc::new(
        sgl_router::proxy::Proxy::new(cfg.workers[0].url.clone(), request_timeout)
            .context("build proxy client")?,
    );

    match proxy.probe_health(std::time::Duration::from_secs(2)).await {
        Ok(()) => tracing::info!("worker probe ok"),
        Err(e) => tracing::warn!("startup worker probe failed: {e}; continuing anyway"),
    }

    let ctx = Arc::new(sgl_router::server::app_context::AppContext::new(
        cfg.clone(),
        tokenizers,
        proxy,
    ));
    ctx.mark_ready();

    let app = sgl_router::server::app::build_router(ctx.clone());

    let bind = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!("listening on {bind}");

    // Install signal handlers BEFORE serve starts. If this fails we exit
    // cleanly rather than starting an HTTP server that ignores SIGTERM and
    // sits unresponsive for the k8s grace period.
    let (sigterm, sigint) = install_signal_handlers()?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(sigterm, sigint))
        .await
        .context("axum serve")
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
        // Double-init must not panic. `try_init` returns Err on the second
        // call, but `init_tracing` swallows it after logging to stderr.
        let _ = init_tracing("info");
        let _ = init_tracing("info");
    }
}
