// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Static-file discovery backend — stub.
//!
//! Real implementation lands in Task 4 (TOML reader + `notify` watcher).

use crate::config::StaticFileDiscoveryConfig;
use anyhow::Result;
use tokio::sync::mpsc;

pub async fn spawn(
    _cfg: StaticFileDiscoveryConfig,
    _tx: mpsc::Sender<super::DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    anyhow::bail!("static_file discovery not implemented yet (Task 4)")
}
