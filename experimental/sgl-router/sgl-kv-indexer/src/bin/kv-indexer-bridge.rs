// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_kv_indexer::bridge::{run_bridge_until, BridgeConfig};
use sgl_kv_indexer::{run_recoverable_bridge_fleet_until, shutdown_signal, BridgeFleetConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    if let Some(config) = BridgeFleetConfig::from_env()? {
        run_recoverable_bridge_fleet_until(config, shutdown_signal()).await?;
    } else {
        let config = BridgeConfig::from_env()?;
        run_bridge_until(config, shutdown_signal()).await?;
    }
    Ok(())
}
