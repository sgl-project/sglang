// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "sgl-router", version)]
struct Cli {
    #[arg(long, env = "SGL_ROUTER_CONFIG")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = sgl_router::config::Config::from_path(&cli.config)
        .with_context(|| format!("load config from {}", cli.config.display()))?;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new(&cfg.observability.log_level)
            }),
        )
        .with_target(true)
        .init();

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
    let proxy = Arc::new(sgl_router::proxy::Proxy::new(cfg.workers[0].url.clone()));

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

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve")
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT");
    tokio::select! {
        _ = sigterm.recv() => tracing::info!("got SIGTERM, shutting down"),
        _ = sigint.recv()  => tracing::info!("got SIGINT, shutting down"),
    }
}
