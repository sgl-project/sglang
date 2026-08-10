//! The crate's only parallelism seam.
//!
//! Every fan-out in the crate goes through the functions below, so whether
//! this crate owns worker threads at all is decided in exactly one place: the
//! `parallel` cargo feature.
//!
//! * **feature on** (the PyO3 extension): work is fanned out on the crate's
//!   rayon pool. A Python processor calls in from one or two worker threads
//!   with the GIL released, so intra-call parallelism is the whole point.
//! * **feature off** (the pure-Rust `rlib` that `sglang-server` links): rayon
//!   is not even a dependency, and everything runs inline on the calling
//!   thread. A server supplies concurrency across requests and owns its own
//!   core budget (it pins threads via `core_affinity`), so a library that
//!   silently spawns its own pools would fight it.
//!
//! Note that "sequential" here means *inline on the caller*, not a one-thread
//! pool: `ThreadPool::install` injects work into the pool and blocks the
//! caller, so sizing a pool to 1 would serialize every concurrent request in
//! the process instead of just declining to fan out.
//!
//! Results are identical either way — the fan-outs are order-preserving maps
//! and writes into disjoint slices, never reductions.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Map `items`, short-circuiting on the first error. Output order matches input
/// order. CPU-bound work: decode, resize, patchify, hash.
#[cfg(feature = "parallel")]
pub fn try_map<'a, T, R, E>(
    items: &'a [T],
    f: impl Fn(&'a T) -> Result<R, E> + Send + Sync,
) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    R: Send,
    E: Send,
{
    super::pool().install(|| items.par_iter().map(f).collect())
}

#[cfg(not(feature = "parallel"))]
pub fn try_map<'a, T, R, E>(
    items: &'a [T],
    f: impl Fn(&'a T) -> Result<R, E> + Send + Sync,
) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    R: Send,
    E: Send,
{
    items.iter().map(f).collect()
}

/// Apply `f(chunk_index, chunk)` over disjoint `chunk_size`-element windows of
/// `buf`. The final chunk is short when `chunk_size` does not divide the length.
#[cfg(feature = "parallel")]
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    super::pool().install(|| {
        buf.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(index, chunk)| f(index, chunk));
    });
}

#[cfg(not(feature = "parallel"))]
pub fn for_chunks_mut<T: Send>(
    buf: &mut [T],
    chunk_size: usize,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    for (index, chunk) in buf.chunks_mut(chunk_size).enumerate() {
        f(index, chunk);
    }
}

/// Run `f` with the CPU pool already entered, so nested [`for_chunks_mut`]
/// calls inside it reuse this entry instead of injecting a job each. Use it to
/// wrap a multi-stage leaf (e.g. the two passes of a separable resize) that
/// would otherwise pay per-stage pool entry.
#[cfg(feature = "parallel")]
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    super::pool().install(f)
}

#[cfg(not(feature = "parallel"))]
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    f()
}
