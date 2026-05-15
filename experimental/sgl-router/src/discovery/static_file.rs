// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Static-file discovery backend.
//!
//! Watches a TOML file with [`notify`]. Initial load fans `Added` events;
//! subsequent edits diff against prior state and emit `Added`/`Removed`/
//! `ModeChanged`. URL or model_ids changes → `Removed` + `Added` so the
//! registry rebuilds cleanly. Invalid TOML logs `WARN` and continues —
//! the watcher does NOT die.

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

async fn diff_and_emit(
    prev: &mut HashMap<WorkerId, WorkerSpec>,
    next: HashMap<WorkerId, WorkerSpec>,
    tx: &mpsc::Sender<DiscoveryEvent>,
) {
    let prev_ids: HashSet<_> = prev.keys().cloned().collect();
    let next_ids: HashSet<_> = next.keys().cloned().collect();
    for id in next_ids.difference(&prev_ids) {
        if let Some(spec) = next.get(id) {
            let _ = tx.send(DiscoveryEvent::Added(spec.clone())).await;
        }
    }
    for id in prev_ids.difference(&next_ids) {
        let _ = tx.send(DiscoveryEvent::Removed { id: id.clone() }).await;
    }
    for id in prev_ids.intersection(&next_ids) {
        let p = prev.get(id);
        let n = next.get(id);
        if let (Some(p), Some(n)) = (p, n) {
            if p.mode != n.mode {
                let _ = tx
                    .send(DiscoveryEvent::ModeChanged {
                        id: id.clone(),
                        mode: n.mode,
                    })
                    .await;
            }
            if p.url != n.url || p.model_ids != n.model_ids {
                // URL or model set change: treat as Removed+Added so the
                // registry rebuilds cleanly.
                let _ = tx.send(DiscoveryEvent::Removed { id: id.clone() }).await;
                let _ = tx.send(DiscoveryEvent::Added(n.clone())).await;
            }
        }
    }
    *prev = next;
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
    diff_and_emit(&mut *prev.lock().await, initial, &tx).await;

    // notify watcher → channel
    let (notify_tx, mut notify_rx) = mpsc::channel(64);
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        if let Ok(e) = res {
            let _ = notify_tx.blocking_send(e);
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
        while let Some(event) = notify_rx.recv().await {
            match event.kind {
                EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) => {
                    let content = match tokio::fs::read_to_string(&path_clone).await {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!(
                                "static_file: failed to read {}: {e}",
                                path_clone.display()
                            );
                            continue;
                        }
                    };
                    let next = match parse(&content) {
                        Ok(n) => n,
                        Err(e) => {
                            tracing::warn!(
                                "static_file: invalid toml in {}: {e}",
                                path_clone.display()
                            );
                            continue;
                        }
                    };
                    diff_and_emit(&mut *prev_clone.lock().await, next, &tx_clone).await;
                }
                _ => {}
            }
        }
    });
    Ok(handle)
}
