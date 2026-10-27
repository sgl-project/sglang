// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_kv_indexer::bridge::{run_bridge_until, BridgeConfig};
use sgl_kv_indexer::shutdown_signal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = BridgeConfig::from_env()?;
    run_bridge_until(config, shutdown_signal()).await?;
    Ok(())
}
