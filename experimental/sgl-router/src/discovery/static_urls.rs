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
        tracing::debug!(
            "static_urls discovery: initial fan-out complete; parking until channel closes"
        );
        // After fan-out the static backend has no further work — but
        // `server::supervisor::supervise_critical_tasks` treats *any*
        // discovery exit as fatal and flips `/readyz` to 503. Park here
        // until the consumer drops the receiver. `tx.closed()` resolves
        // the moment every `Receiver` has been dropped; the supervisor's
        // normal-shutdown path aborts this task before that. So
        // reaching the `info!` below means either (a) we lost the abort
        // race during a clean shutdown, or (b) the worker manager exited
        // unexpectedly — in case (b) the supervisor will catch the
        // subsequent discovery-task exit and `error!` + mark unready,
        // and this breadcrumb gives operator triage a starting point.
        tx.closed().await;
        tracing::info!(
            "static_urls discovery: event channel closed by receiver \
             (worker manager dropped its end, or shutdown abort raced); exiting"
        );
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

    /// After fan-out the task must STAY ALIVE so the critical-task
    /// supervisor (`server::supervisor::supervise_critical_tasks`)
    /// doesn't treat the exit as a failure and flip `/readyz` to 503.
    /// The static_urls backend has no hot-reload, so the only reasons
    /// it should ever exit are (a) the consumer dropped the receiver,
    /// or (b) the supervisor aborted it on shutdown. A "natural" exit
    /// after fan-out used to be the third path, and was wrongly
    /// interpreted as a panic by the supervisor — pinned here so a
    /// regression to "exit after fan-out" can't sneak back in.
    #[tokio::test]
    async fn stays_alive_after_fanout_until_receiver_dropped() {
        use std::time::Duration;

        let cfg = StaticUrlsDiscoveryConfig {
            urls: vec!["http://w0:30000".into(), "http://w1:30000".into()],
        };
        let (tx, mut rx) = mpsc::channel(8);
        let h = spawn(cfg, tx).await.unwrap();

        // Drain the fan-out so the task is past the for-loop.
        for _ in 0..2 {
            let _ = rx.recv().await.expect("fan-out event");
        }

        // Now give the task a long-by-test-standards moment to exit
        // post-fanout. Pre-fix this would have completed in under a
        // millisecond; post-fix it must time out.
        let mut handle = h;
        let exited = tokio::time::timeout(Duration::from_millis(200), &mut handle).await;
        let still_running = exited.is_err();
        if !still_running {
            panic!(
                "static_urls task exited after fan-out (joined as {exited:?}); \
                 this trips `supervise_critical_tasks` → mark_unready and the pod \
                 becomes /readyz 503. The task must park until the receiver is dropped."
            );
        }
        // Clean shutdown: dropping the receiver closes the channel, which
        // the post-fix task uses as its "time to exit" signal. Pin both
        // halves of the contract — parks while the receiver is alive AND
        // exits cleanly once it's dropped — so a future refactor that
        // parks the task on the wrong signal (e.g., a sleep, a token that
        // never fires) is caught here rather than silently lingering.
        drop(rx);
        let joined = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("task must exit promptly after the receiver is dropped");
        joined.expect("task panicked during clean shutdown");
    }
}
