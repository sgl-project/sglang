// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod k8s;
pub mod static_urls;
pub mod types;
pub use types::*;

use crate::config::{Config, DiscoveryBackend};
use anyhow::Result;
use tokio::sync::mpsc;

/// Channel capacity for discovery → registry events.  Bounded to 128 —
/// pod-add/remove is infrequent, but a bound prevents unbounded memory
/// growth under any pathological burst.
pub const DISCOVERY_CHANNEL_CAP: usize = 128;

/// Spawn the configured discovery backend.
///
/// Returns the consumer end of the event channel and a [`tokio::task::JoinHandle`]
/// for the producer task.  The static_urls backend's task exits once the
/// initial fan-out completes; the k8s backend's task runs for the lifetime
/// of the watch.
pub async fn spawn_discovery(
    cfg: &Config,
) -> Result<(mpsc::Receiver<DiscoveryEvent>, tokio::task::JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel(DISCOVERY_CHANNEL_CAP);
    let handle = match &cfg.discovery.backend {
        DiscoveryBackend::StaticUrls(s) => static_urls::spawn(s.clone(), tx).await?,
        DiscoveryBackend::K8s(k) => k8s::spawn(k.clone(), tx).await?,
    };
    Ok((rx, handle))
}
