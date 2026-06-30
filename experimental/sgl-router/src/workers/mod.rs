// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod introspect;
pub mod manager;
pub mod registry;
pub mod worker;

pub use introspect::{ServerInfo, WorkerIntrospector};
pub use registry::WorkerRegistry;
pub use worker::LoadGuard;
pub use worker::WireProtocol;
pub use worker::Worker;

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Spawn a background task that periodically reclaims in-flight admission slots
/// held longer than `ttl` across every registered worker.
///
/// Defense-in-depth: the SSE idle timeout already bounds the common streaming
/// leak path, but a slot held past `ttl` for any other reason (a future code
/// path that forgets to drop a guard, an unforeseen hang) is force-released
/// here, so the per-worker cap can never stay pinned and false-shed forever. A
/// reclaim is logged at WARN — it means a guard leaked and is worth alerting on.
pub fn spawn_load_janitor(
    registry: Arc<WorkerRegistry>,
    ttl: Duration,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let now = Instant::now();
            let mut total = 0usize;
            for w in registry.all() {
                let n = w.reclaim_stale_load(ttl, now);
                if n > 0 {
                    total += n;
                    tracing::warn!(
                        worker = %w.url,
                        reclaimed = n,
                        ttl_secs = ttl.as_secs(),
                        "reclaimed leaked in-flight admission slot(s) past TTL",
                    );
                }
            }
            if total > 0 {
                tracing::warn!(total, "worker-load janitor swept leaked admission slots");
            }
        }
    })
}
