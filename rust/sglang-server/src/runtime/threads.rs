//! Thread-group machinery: the [`Runnable`] stage trait, CPU-core partitioning,
//! and the pinned-thread spawners used by `runtime::start`.
//!
//! Adding a new thread group (encoder, weight loader, KV-cache offloader, …) is
//! three small steps and no spawn boilerplate:
//!   1. a struct implementing [`Runnable`];
//!   2. a core set for it (a field on [`CorePlan`] + a slice in [`plan_cores`]);
//!   3. one [`spawn_pool`] (N pinned workers) or [`spawn_stage`] (singleton) call.

use std::thread::JoinHandle;

use core_affinity::CoreId;

use super::RuntimeConfig;

/// A pipeline stage that owns its channel handles + config and runs a blocking
/// loop until its inbox closes. Lets the runtime spawn stages uniformly via
/// [`spawn_stage`] / [`spawn_pool`] instead of free `run_*` functions with
/// positional handles. Implemented by every CPU-bound worker and TM router.
pub trait Runnable: Send + 'static {
    fn run(self);
}

/// Cores reserved for the two TokenizerManager router threads (`tm-ingress`,
/// `tm-egress`) — light, latency-sensitive channel routers, so one core each.
///
/// TODO(tm-scaling): both TM threads are single-consumer serialization points,
/// each with its own ceiling. `tm-ingress` runs validate + `normalize_sampling_params`
/// for *every* request before fanning out to the (pooled) tokenizer workers, so a
/// high request-arrival / short-request workload is bounded by that one thread's
/// per-request cost (kept O(fields), see `sampling::normalize_sampling_params`).
/// Sharding ingress by rid — like the tokenizer/detok pools — lifts that ceiling.
///
/// `tm-egress` is a head-of-line ceiling of a different kind — it
/// does a *blocking* send per chunk to the owning detok shard, so one slow shard
/// stalls the dispatcher and thus every shard (see `Egress::route`). Sharding the
/// dispatcher alone doesn't fix it: each egress-ring frame is a whole batch fanned
/// to *all* shards, so any dispatcher still blocks on the slow one. The real fix
/// is a per-shard egress ring (the scheduler pushing each request's output to its
/// shard's ring), each drained by its own dispatcher — at which point this needs
/// one core per ingress/egress shard rather than a fixed 2.
const TM_CORES: usize = 2;

/// Partition the machine's cores into four disjoint sets: the I/O-bound API
/// pool, the CPU-bound tokenizer and detokenizer pools, and the two TM router
/// threads. Falls back to no pinning if affinity isn't available or there aren't
/// enough cores for the (CPU-bound) pools. A new thread group adds a field here
/// and a slice in [`plan_cores`].
pub(super) struct CorePlan {
    pub(super) api: Vec<CoreId>,
    pub(super) tok: Vec<CoreId>,
    pub(super) detok: Vec<CoreId>,
    pub(super) tm: Vec<CoreId>,
}

pub(super) fn plan_cores(cfg: &RuntimeConfig) -> Option<CorePlan> {
    // `cores` carries the pinning decision: `None`/empty → run unpinned. The
    // caller (Python `_partition_cores`) passes this rank's NUMA-local cores
    // minus the scheduler's reserved launch cores.
    let cores: Vec<CoreId> = match &cfg.cores {
        Some(ids) if !ids.is_empty() => ids.iter().map(|&id| CoreId { id }).collect(),
        _ => return None,
    };
    if cores.len() < cfg.api_worker_num + cfg.tokenizer_worker_num + cfg.detokenizer_worker_num {
        tracing::warn!(
            available = cores.len(),
            "not enough cores to pin all pools; running unpinned"
        );
        return None;
    }
    let mut it = cores.into_iter();
    let api: Vec<CoreId> = it.by_ref().take(cfg.api_worker_num).collect();
    let tok = it.by_ref().take(cfg.tokenizer_worker_num).collect();
    let detok = it.by_ref().take(cfg.detokenizer_worker_num).collect();
    // The two TM router threads get up to `TM_CORES` leftover cores; when none
    // are spare they fall back to the API set so they never float onto the
    // CPU-bound tokenizer/detok cores.
    let mut tm: Vec<CoreId> = it.by_ref().take(TM_CORES).collect();
    if tm.is_empty() {
        tm = api.clone();
    }
    Some(CorePlan {
        api,
        tok,
        detok,
        tm,
    })
}

/// Pin the calling thread to `core` if one was assigned (no-op otherwise).
fn pin_current(core: Option<CoreId>) {
    if let Some(c) = core {
        core_affinity::set_for_current(c);
    }
}

/// Pick the pinned core for worker `i` from an optional pool core set.
pub(super) fn pool_core(cores: &Option<Vec<CoreId>>, i: usize) -> Option<CoreId> {
    cores.as_ref().and_then(|c| c.get(i).copied())
}

/// Spawn a single [`Runnable`] stage on a named thread, optionally pinned.
/// Used by [`spawn_pool`]; every group goes through the pool spawner now.
fn spawn_stage(
    name: &str,
    core: Option<CoreId>,
    stage: impl Runnable,
    threads: &mut Vec<JoinHandle<()>>,
) {
    let handle = std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            pin_current(core);
            stage.run();
        })
        .expect("spawn stage");
    threads.push(handle);
}

/// Spawn a pool of `count` [`Runnable`] workers, each pinned to `cores[i]` (when
/// available) and named `{name}-{i}`. `build(i)` constructs worker `i` — cloning
/// shared handles, or moving a per-worker resource out of a captured iterator.
pub(super) fn spawn_pool<R, F>(
    name: &str,
    cores: Option<Vec<CoreId>>,
    count: usize,
    threads: &mut Vec<JoinHandle<()>>,
    mut build: F,
) where
    R: Runnable,
    F: FnMut(usize) -> R,
{
    for i in 0..count {
        let core = pool_core(&cores, i);
        spawn_stage(&format!("{name}-{i}"), core, build(i), threads);
    }
}
