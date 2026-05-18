// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Static-file discovery backend.
//!
//! Watches a TOML file with [`notify`]. Initial load fans `Added` events;
//! subsequent edits diff against prior state and emit `Added`/`Removed`/
//! `ModeChanged`. URL or model_ids changes → `Removed` + `Added` so the
//! registry rebuilds cleanly.
//!
//! Reload errors (read or parse failures) do NOT kill the watcher — the
//! prior good state is kept and serving continues. The first failure logs
//! at WARN. After [`RELOAD_ESCALATE_AFTER`] consecutive failures (i.e. the
//! file has been in a broken state long enough that an operator probably
//! never noticed the WARN), each subsequent failure escalates to ERROR.
//! Counters reset on the next successful reload.

use crate::config::StaticFileDiscoveryConfig;
use crate::discovery::{DiscoveryEvent, ModelId, WorkerId, WorkerMode, WorkerSpec};
use anyhow::{Context, Result};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

#[derive(Debug, Deserialize)]
struct StaticFile {
    #[serde(default)]
    workers: Vec<StaticWorker>,
}

#[derive(Debug, Deserialize)]
struct StaticWorker {
    id: String,
    url: String,
    mode: WorkerMode,
    #[serde(default)]
    model_ids: Vec<String>,
}

fn parse(content: &str) -> Result<HashMap<WorkerId, WorkerSpec>> {
    let parsed: StaticFile = toml::from_str(content).context("parse workers.toml")?;
    let mut out = HashMap::new();
    for w in parsed.workers {
        let id = WorkerId(w.id.clone());
        out.insert(
            id.clone(),
            WorkerSpec {
                id,
                url: w.url,
                mode: w.mode,
                model_ids: w.model_ids.into_iter().map(ModelId).collect(),
            },
        );
    }
    Ok(out)
}

/// Number of consecutive reload failures before WARN escalates to ERROR.
pub(crate) const RELOAD_ESCALATE_AFTER: u32 = 3;

async fn diff_and_emit(
    prev: &mut HashMap<WorkerId, WorkerSpec>,
    next: HashMap<WorkerId, WorkerSpec>,
    tx: &mpsc::Sender<DiscoveryEvent>,
) -> Result<(), mpsc::error::SendError<DiscoveryEvent>> {
    let prev_ids: HashSet<_> = prev.keys().cloned().collect();
    let next_ids: HashSet<_> = next.keys().cloned().collect();
    for id in next_ids.difference(&prev_ids) {
        if let Some(spec) = next.get(id) {
            tx.send(DiscoveryEvent::Added(spec.clone())).await?;
        }
    }
    for id in prev_ids.difference(&next_ids) {
        tx.send(DiscoveryEvent::Removed { id: id.clone() }).await?;
    }
    for id in prev_ids.intersection(&next_ids) {
        let p = prev.get(id);
        let n = next.get(id);
        if let (Some(p), Some(n)) = (p, n) {
            if p.mode != n.mode {
                tx.send(DiscoveryEvent::ModeChanged {
                    id: id.clone(),
                    mode: n.mode,
                })
                .await?;
            }
            if p.url != n.url || p.model_ids != n.model_ids {
                // URL or model set change: treat as Removed+Added so the
                // registry rebuilds cleanly.
                tx.send(DiscoveryEvent::Removed { id: id.clone() }).await?;
                tx.send(DiscoveryEvent::Added(n.clone())).await?;
            }
        }
    }
    *prev = next;
    Ok(())
}

/// Log a reload error at WARN, escalating to ERROR after
/// [`RELOAD_ESCALATE_AFTER`] consecutive failures.  The `kind` discriminates
/// "read" vs "parse" in the log record so operators can grep for the right
/// fix (file missing/permissions vs TOML typo).
fn log_reload_failure(
    kind: &'static str,
    path: &std::path::Path,
    err: &dyn std::fmt::Display,
    consecutive: u32,
) {
    if consecutive >= RELOAD_ESCALATE_AFTER {
        tracing::error!(
            kind,
            path = %path.display(),
            consecutive,
            error = %err,
            "static_file: reload failed repeatedly; registry is serving stale topology",
        );
    } else {
        tracing::warn!(
            kind,
            path = %path.display(),
            consecutive,
            error = %err,
            "static_file: reload failed; keeping prior state",
        );
    }
}

pub async fn spawn(
    cfg: StaticFileDiscoveryConfig,
    tx: mpsc::Sender<DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    let path = PathBuf::from(&cfg.path);
    // Initial load.
    let initial_content = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("read initial {}", path.display()))?;
    let initial = parse(&initial_content).context("initial parse")?;
    let prev = Arc::new(Mutex::new(HashMap::new()));
    diff_and_emit(&mut *prev.lock().await, initial, &tx)
        .await
        .map_err(|_| anyhow::anyhow!("initial fan-out: discovery channel closed"))?;

    // notify watcher → channel
    let (notify_tx, mut notify_rx) = mpsc::channel(64);
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        if let Ok(e) = res {
            // The notify-rs callback runs on the watcher's internal thread.
            // blocking_send would block that thread when full, silently dropping
            // subsequent fs events. Use try_send instead: warn on Full, no-op on Closed.
            match notify_tx.try_send(e) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    tracing::warn!("static_file discovery: event channel full, dropping event");
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    // Consumer dropped; nothing useful to do on the notify thread.
                }
            }
        }
    })
    .context("create file watcher")?;
    watcher
        .watch(&path, RecursiveMode::NonRecursive)
        .context("watch path")?;

    let prev_clone = prev.clone();
    let path_clone = path.clone();
    let tx_clone = tx.clone();
    let handle = tokio::spawn(async move {
        // Keep `watcher` alive for the lifetime of this task.
        let _watcher = watcher;
        let mut consecutive_read_failures: u32 = 0;
        let mut consecutive_parse_failures: u32 = 0;
        while let Some(event) = notify_rx.recv().await {
            match event.kind {
                EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) => {
                    let content = match tokio::fs::read_to_string(&path_clone).await {
                        Ok(c) => {
                            consecutive_read_failures = 0;
                            c
                        }
                        Err(e) => {
                            consecutive_read_failures = consecutive_read_failures.saturating_add(1);
                            log_reload_failure("read", &path_clone, &e, consecutive_read_failures);
                            continue;
                        }
                    };
                    let next = match parse(&content) {
                        Ok(n) => {
                            consecutive_parse_failures = 0;
                            n
                        }
                        Err(e) => {
                            consecutive_parse_failures =
                                consecutive_parse_failures.saturating_add(1);
                            log_reload_failure(
                                "parse",
                                &path_clone,
                                &e,
                                consecutive_parse_failures,
                            );
                            continue;
                        }
                    };
                    if diff_and_emit(&mut *prev_clone.lock().await, next, &tx_clone)
                        .await
                        .is_err()
                    {
                        tracing::info!(
                            "static_file discovery: event channel closed; exiting watcher"
                        );
                        return;
                    }
                }
                _ => {}
            }
        }
        tracing::warn!("static_file watcher channel ended; discovery task exiting");
    });
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str, url: &str, mode: WorkerMode) -> (WorkerId, WorkerSpec) {
        let id = WorkerId(id.into());
        (
            id.clone(),
            WorkerSpec {
                id,
                url: url.into(),
                mode,
                model_ids: vec![ModelId("m1".into())],
            },
        )
    }

    #[tokio::test]
    async fn diff_and_emit_returns_err_when_consumer_dropped() {
        let mut prev: HashMap<WorkerId, WorkerSpec> = HashMap::new();
        let next: HashMap<WorkerId, WorkerSpec> =
            [worker("w1", "http://x:30000", WorkerMode::Plain)]
                .into_iter()
                .collect();
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let r = diff_and_emit(&mut prev, next, &tx).await;
        assert!(r.is_err(), "send into a closed channel must surface");
    }

    #[tokio::test]
    async fn diff_and_emit_emits_added_on_first_load() {
        let mut prev: HashMap<WorkerId, WorkerSpec> = HashMap::new();
        let next: HashMap<WorkerId, WorkerSpec> =
            [worker("w1", "http://x:30000", WorkerMode::Plain)]
                .into_iter()
                .collect();
        let (tx, mut rx) = mpsc::channel(4);
        diff_and_emit(&mut prev, next, &tx).await.unwrap();
        assert!(matches!(rx.try_recv().unwrap(), DiscoveryEvent::Added(_),));
    }

    #[tokio::test]
    async fn diff_and_emit_emits_removed_when_worker_drops() {
        let mut prev: HashMap<WorkerId, WorkerSpec> =
            [worker("w1", "http://x:30000", WorkerMode::Plain)]
                .into_iter()
                .collect();
        let next: HashMap<WorkerId, WorkerSpec> = HashMap::new();
        let (tx, mut rx) = mpsc::channel(4);
        diff_and_emit(&mut prev, next, &tx).await.unwrap();
        assert!(matches!(
            rx.try_recv().unwrap(),
            DiscoveryEvent::Removed { .. },
        ));
    }

    #[tokio::test]
    async fn diff_and_emit_emits_removed_plus_added_on_url_change() {
        let mut prev: HashMap<WorkerId, WorkerSpec> =
            [worker("w1", "http://x:30000", WorkerMode::Plain)]
                .into_iter()
                .collect();
        let next: HashMap<WorkerId, WorkerSpec> =
            [worker("w1", "http://y:30000", WorkerMode::Plain)]
                .into_iter()
                .collect();
        let (tx, mut rx) = mpsc::channel(4);
        diff_and_emit(&mut prev, next, &tx).await.unwrap();
        let first = rx.try_recv().unwrap();
        let second = rx.try_recv().unwrap();
        assert!(matches!(first, DiscoveryEvent::Removed { .. }), "{first:?}");
        assert!(matches!(second, DiscoveryEvent::Added(_)), "{second:?}");
    }
}
