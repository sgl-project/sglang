//! Thread-group machinery: the frontend's CPU set and the pinned-thread
//! spawners (for [`Runnable`] stages) used by `runtime::start`.
//!
//! Adding a new thread group (encoder, weight loader, KV-cache offloader, …) is
//! two small steps and no spawn boilerplate:
//!   1. a struct implementing [`Runnable`];
//!   2. one [`spawn_pool`] call (N workers, or 1 for a singleton stage).

use std::thread::JoinHandle;
use std::time::Duration;

use rustix::thread::{CpuSet, sched_setaffinity};

use super::runtime::Runnable;
use crate::message::config::RuntimeConfig;

/// The CPU set every frontend thread runs on, from `cores` (`None`/empty →
/// run unpinned). The caller (Python `_partition_cores`) passes this rank's
/// NUMA-local cores minus the scheduler's reserved launch cores, so the set
/// keeps the frontend off the scheduler.
///
/// All thread groups share the set instead of owning one core each. A thread
/// pinned to a single core can't migrate: when anything else runs there (the
/// MM workers' decoder threads, which float over this set, or another
/// process), it waits in that core's run queue while the rest of the set
/// idles, and every stream it serves stalls with it. On a shared set the
/// kernel wakes it on any idle core.
pub(super) fn frontend_cpu_set(cfg: &RuntimeConfig) -> Option<CpuSet> {
    let cores = cfg.rust_server_args.cores.as_deref().unwrap_or_default();
    let mut set = CpuSet::new();
    for &core in cores {
        if core < CpuSet::MAX_CPU {
            set.set(core);
        } else {
            tracing::warn!(core, "frontend core id out of range; ignored");
        }
    }
    (set.count() > 0).then_some(set)
}

/// Pin the calling thread to `cpus` if a set was given (no-op otherwise).
pub(super) fn pin_current(cpus: Option<&CpuSet>) {
    if let Some(set) = cpus
        && let Err(error) = sched_setaffinity(None, set)
    {
        tracing::warn!(%error, "could not pin a frontend thread to the frontend cores");
    }
}

/// Spawn a single [`Runnable`] stage on a named thread, optionally pinned.
/// Used by [`spawn_pool`]; every group goes through the pool spawner now.
fn spawn_stage(
    name: &str,
    cpus: Option<CpuSet>,
    stage: impl Runnable,
    threads: &mut Vec<JoinHandle<()>>,
) {
    let handle = std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            pin_current(cpus.as_ref());
            stage.run();
        })
        .expect("spawn stage");
    threads.push(handle);
}

/// Spawn a pool of `count` [`Runnable`] workers, each pinned to `cpus` (when
/// given) and named `{name}-{i}`. `build(i)` constructs worker `i` — cloning
/// shared handles, or moving a per-worker resource out of a captured iterator.
pub(super) fn spawn_pool<R, F>(
    name: &str,
    cpus: Option<CpuSet>,
    count: usize,
    threads: &mut Vec<JoinHandle<()>>,
    mut build: F,
) where
    R: Runnable,
    F: FnMut(usize) -> R,
{
    for i in 0..count {
        spawn_stage(&format!("{name}-{i}"), cpus, build(i), threads);
    }
}

/// Join every handle, giving up after `timeout`.
pub(super) fn join_all_with_timeout(handles: Vec<JoinHandle<()>>, timeout: Duration) -> bool {
    if handles.is_empty() {
        return true;
    }
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        for h in handles {
            let _ = h.join();
        }
        let _ = done_tx.send(());
    });
    done_rx.recv_timeout(timeout).is_ok()
}
