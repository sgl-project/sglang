// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware tree-lookup microbench.
//!
//! Mirrors the shape of `sgl-model-gateway/benches/radix_tree_benchmark.rs`
//! (specifically the `TokenTree` / `PositionalIndexer` paths — which serve
//! the same role as sgl-router's `HashTree`). The bench measures:
//!
//!   * `insert` — populate one worker's prefix.
//!   * `match_prefix` — score an incoming request against the tree.
//!   * `insert_continuation` — insert with `parent_hash = Some(..)`, which
//!     is what the pump emits for every block after a sequence's first.
//!     `HashTree::route_insert` resolves that parent across ALL shards
//!     before writing one, so it costs strictly more than a `None` insert;
//!     the paired `parent_none` case is the same block count without the
//!     scan.
//!   * `contended_match` — reader `match_prefix` throughput WHILE a
//!     background writer hammers `insert` / `remove`. The case sharding
//!     targets: under one process-wide lock every event write blocks every
//!     routing read. Run against both writer shapes so the headline ratio
//!     is not read off the cheapest possible writer.
//!
//! Output is `criterion`'s default (target/criterion/...). To run:
//!
//!   cargo bench --bench tree_lookup
//!   cargo bench --bench tree_lookup -- --sample-size 30   # faster
//!
//! See `BENCHMARKS.md` for the SMG↔sgl-router comparison table.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sgl_router::state::kv_events::tree::{HashTree, KvWorkerId};

fn build_tree(num_workers: usize, blocks_per_worker: usize, seed: u64) -> HashTree {
    let tree = HashTree::new();
    let mut rng = StdRng::seed_from_u64(seed);
    for w in 0..num_workers {
        let worker = KvWorkerId::new(format!("http://w{w}:30000"), 0);
        // Each worker holds a distinct (random) prefix so the trees fan
        // out — this is the realistic case for cache-aware routing.
        let hashes: Vec<i64> = (0..blocks_per_worker).map(|_| rng.gen::<i64>()).collect();
        tree.insert(&worker, None, &hashes);
    }
    tree
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_insert");
    for &n_blocks in &[8usize, 32, 128, 512] {
        group.throughput(Throughput::Elements(n_blocks as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_blocks), &n_blocks, |b, &n| {
            let mut rng = StdRng::seed_from_u64(0xC0FFEE);
            let hashes: Vec<i64> = (0..n).map(|_| rng.gen::<i64>()).collect();
            b.iter_batched(
                HashTree::new,
                |tree| {
                    let worker = KvWorkerId::new("http://w:30000".to_string(), 0);
                    tree.insert(&worker, None, black_box(&hashes));
                    tree
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Cost of an insert that carries a parent, against one that does not.
///
/// `route_insert` early-returns on `parent_hash = None` and writes a single
/// shard; a `Some(p)` takes a read lock on every shard to find which one
/// already holds `p`. The pump passes `Some` for every block after a
/// sequence's first, so that is the steady-state write path, and without
/// this case the suite would time only the one shape the pump rarely
/// sends. The scan is real but small next to the descent it precedes
/// (~36ns against ~800ns measured), and the unsharded tree shows no gap
/// between the two cases at all, having no shards to scan.
///
/// Both cases re-insert blocks the tree already holds, which is idempotent
/// — the tree neither grows nor needs teardown between iterations, so the
/// two are timed on one prebuilt tree at equal block counts.
fn bench_insert_continuation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_insert_continuation");
    let tree = build_tree(64, 64, 0xDEADBEEF);
    // Worker 0 and its chain, re-derived from `build_tree`'s seed.
    let worker = KvWorkerId::new("http://w0:30000".to_string(), 0);
    let chain: Vec<i64> = {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        (0..64).map(|_| rng.gen::<i64>()).collect()
    };
    let (head, tail) = chain.split_at(32);
    let parent_hash = head[head.len() - 1];

    group.throughput(Throughput::Elements(tail.len() as u64));
    group.bench_function("parent_none", |b| {
        b.iter(|| tree.insert(&worker, None, black_box(head)))
    });
    group.bench_function("parent_some", |b| {
        b.iter(|| tree.insert(&worker, Some(black_box(parent_hash)), black_box(tail)))
    });
    group.finish();
}

fn bench_match_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_match_prefix");
    // (workers, blocks_per_worker, query_len) cases that span the
    // realistic operating window: small fleet w/ moderate prefixes,
    // medium fleet w/ long prefixes, and a stress case.
    let cases = [
        (4usize, 32usize, 8usize),
        (16, 64, 32),
        (64, 128, 64),
        (128, 256, 128),
    ];
    for (workers, bpw, query_len) in cases {
        let label = format!("w{workers}_bpw{bpw}_q{query_len}");
        group.throughput(Throughput::Elements(query_len as u64));
        let tree = build_tree(workers, bpw, 0xDEADBEEF);
        // Re-derive worker 0's prefix (the first `bpw` i64s `build_tree`
        // drew from this seed) so the bench times a real descent; fresh
        // randoms would miss at the root and time a single lookup.
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let probe: Vec<i64> = (0..query_len).map(|_| rng.gen::<i64>()).collect();
        group.bench_function(label, |b| {
            b.iter(|| {
                let m = tree.match_prefix(None, black_box(&probe));
                black_box(m.matched_blocks)
            });
        });
    }
    group.finish();
}

/// How the background writer shapes its inserts. A `Rooted` writer touches
/// exactly one shard; a `Continuation` writer first resolves its parent
/// across all of them, which is the shape the pump emits for every block
/// after a sequence's first.
///
/// Both are measured because the cross-shard resolution looks like it
/// should erase the sharding win and does not: `route_insert` scans under
/// READ locks, which readers share, so only its single `write()` blocks
/// anyone. Measured, the continuation shape costs the reader ~7% over the
/// rooted one, against ~450x for removing the global lock. This case
/// exists to keep that true — a future `route_insert` that took write
/// locks to scan, or serialised the scan behind the reader path, would
/// show up here and nowhere else.
#[derive(Clone, Copy)]
enum WriterShape {
    Rooted,
    Continuation,
}

impl WriterShape {
    fn label(self) -> &'static str {
        match self {
            Self::Rooted => "reader_under_write_pressure",
            Self::Continuation => "reader_under_continuation_write_pressure",
        }
    }
}

/// Reader `match_prefix` throughput while a background writer hammers
/// `insert` / `remove` — the read-vs-write contention sharding is built
/// for. A single global lock serialises the two paths and the reader rate
/// collapses; sharded, the writer's churn leaves reads on other roots
/// uncontended, to the extent the writer stays off their shards.
fn bench_contended_match(c: &mut Criterion) {
    for shape in [WriterShape::Rooted, WriterShape::Continuation] {
        bench_contended_match_with(c, shape);
    }
}

fn bench_contended_match_with(c: &mut Criterion, shape: WriterShape) {
    let mut group = c.benchmark_group("hashtree_contended_match");
    let tree = Arc::new(build_tree(64, 64, 0xDEADBEEF));
    // Worker 0's chain, re-derived from the same seed, so the reader gets a
    // non-trivial full match.
    let warm_chain: Vec<i64> = {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        (0..64).map(|_| rng.gen::<i64>()).collect()
    };

    // One background writer, insert + remove of a fresh 4-block chain per
    // round, so it keeps taking write locks with the tree size bounded.
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let tree = tree.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            let scratch = KvWorkerId::new("http://scratch:30000".to_string(), 0);
            let mut round = 0i64;
            while !stop.load(Ordering::Relaxed) {
                // Cycled, so a long run cannot drift the scratch roots into
                // the warm chain's space or overflow the multiply.
                let base = 1_000_000 + (round % 100_000) * 7;
                let chain = [base, base + 1, base + 2, base + 3];
                match shape {
                    WriterShape::Rooted => tree.insert(&scratch, None, &chain),
                    // Root the chain, then extend it the way the pump does:
                    // only a sequence's first block carries no parent.
                    WriterShape::Continuation => {
                        tree.insert(&scratch, None, &chain[..1]);
                        tree.insert(&scratch, Some(base), &chain[1..]);
                    }
                }
                tree.remove(&scratch, &chain);
                round = round.wrapping_add(1);
            }
        })
    };
    // Signals the writer on the way out however we leave — a panic in the
    // bench body unwinds past any explicit store and would otherwise leave
    // the thread spinning for the rest of the process.
    let _stop_writer = StopOnDrop(stop);

    group.throughput(Throughput::Elements(warm_chain.len() as u64));
    group.bench_function(shape.label(), |b| {
        b.iter(|| {
            let m = tree.match_prefix(None, black_box(&warm_chain));
            black_box(m.matched_blocks)
        });
    });

    drop(_stop_writer);
    writer.join().expect("bench writer thread panicked");
    group.finish();
}

/// Sets its flag on drop, so a background thread parked on it is stopped by
/// an unwind as reliably as by the normal path.
struct StopOnDrop(Arc<AtomicBool>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

criterion_group!(
    benches,
    bench_insert,
    bench_insert_continuation,
    bench_match_prefix,
    bench_contended_match
);
criterion_main!(benches);
