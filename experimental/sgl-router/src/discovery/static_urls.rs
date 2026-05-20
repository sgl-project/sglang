// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Static-URL discovery backend.
//!
//! Takes a fixed list of worker URLs and fans one [`DiscoveryEvent::Added`]
//! per entry. After the initial fan-out the task exits — there is no
//! hot-reload; topology changes require a restart.
//!
//! Each emitted [`WorkerSpec`] uses the URL itself as the `WorkerId` and
//! seeds `mode = Plain` with empty `model_ids` and `bootstrap_port = None`.
//! The worker manager fills those in from each worker's `/server_info`
//! response (see [`crate::workers::introspect`]) and overrides the seeded
//! mode/bootstrap when the worker self-discloses a PD role — so prefill,
//! decode, and plain workers can all appear in the same `urls` list and
//! end up classified correctly.
//!
//! Requires modern SGLang that exposes `disaggregation_mode` in
//! `/server_info`. Workers on older SGLang versions that predate that
//! field stay seeded as `Plain` because the manager has no signal to
//! override with — operators running PD with such a worker should use
//! the K8s backend (which can still classify via pod labels).

use crate::config::StaticUrlsDiscoveryConfig;
use crate::discovery::{DiscoveryEvent, WorkerId, WorkerMode, WorkerSpec};
use anyhow::Result;
use tokio::sync::mpsc;

/// Spawn the static-URLs producer task and return its `JoinHandle`.
///
/// Returns `Result` for parity with [`crate::discovery::k8s::spawn`] (which
/// can fail to construct a `kube::Client`); this backend itself is
/// infallible.
pub async fn spawn(
    cfg: StaticUrlsDiscoveryConfig,
    tx: mpsc::Sender<DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    let handle = tokio::spawn(async move {
        for url in cfg.urls {
            let spec = WorkerSpec {
                id: WorkerId(url.clone()),
                url,
                mode: WorkerMode::Plain,
                model_ids: Vec::new(),
                bootstrap_port: None,
            };
            if tx.send(DiscoveryEvent::Added(spec)).await.is_err() {
                tracing::info!(
                    "static_urls discovery: event channel closed during fan-out; exiting"
                );
                return;
            }
        }
        tracing::debug!("static_urls discovery: initial fan-out complete; task exiting");
    });
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Task exits cleanly when the consumer drops the receiver mid-fanout.
    /// Without this early exit, the producer would block forever on the
    /// closed channel and shutdown would have to abort it. Kept in-source
    /// (rather than as a component test) because it inspects the
    /// `send().is_err()` branch, which is an implementation detail of
    /// this module — fan-out and event-shape assertions live in
    /// `tests/component/discovery/static_urls.rs`.
    #[tokio::test]
    async fn exits_when_receiver_dropped() {
        let cfg = StaticUrlsDiscoveryConfig {
            urls: (0..10).map(|i| format!("http://w{i}:30000")).collect(),
        };
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let h = spawn(cfg, tx).await.unwrap();
        // No panic, no hang — task exits on the first send error.
        h.await.unwrap();
    }
}
