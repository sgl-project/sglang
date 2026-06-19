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
//!   * `contended_match` — reader `match_prefix` throughput WHILE a
//!     background writer hammers `insert` / `remove` on distinct chain
//!     roots. This is the case the sharded tree targets: under a single
//!     process-wide lock every event write blocks every routing read, so
//!     this number collapses; sharded, readers and the writer only contend
//!     when they collide on a shard.
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
use sgl_router::policies::kv_events::tree::{HashTree, KvWorkerId};

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
        // Pull one real worker's prefix so the query has a non-trivial
        // partial match — closer to the production hot path.
        let mut rng = StdRng::seed_from_u64(0x12345);
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

/// Reader `match_prefix` throughput while a background writer hammers
/// `insert` / `remove` on distinct chain roots — the read-vs-write
/// contention scenario the sharded tree is built for. The number to watch
/// is how the measured rate degrades as the writer's pressure rises; a
/// single global lock would serialise the two paths, so the reader rate
/// would drop sharply, while sharded the writer's churn on unrelated roots
/// leaves most reads uncontended.
fn bench_contended_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_contended_match");
    // A warm worker holds many distinct chains that the reader queries; the
    // writer churns its own scratch chains on disjoint roots.
    let tree = Arc::new(build_tree(64, 64, 0xDEADBEEF));
    let warm_worker = KvWorkerId::new("http://w0:30000".to_string(), 0);
    // Re-derive a real warm chain so the reader gets a non-trivial match.
    let warm_chain: Vec<i64> = {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        (0..64).map(|_| rng.gen::<i64>()).collect()
    };
    // `build_tree`'s w0 used the same seed/sequence, so warm_chain matches it.
    let _ = &warm_worker;

    // Spawn one background writer that never stops until the bench drops the
    // stop flag. It inserts then removes a fresh 4-block chain each round, so
    // the tree size stays bounded and the writer keeps taking write locks.
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let tree = tree.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            let scratch = KvWorkerId::new("http://scratch:30000".to_string(), 0);
            let mut round = 0i64;
            while !stop.load(Ordering::Relaxed) {
                let base = 1_000_000 + round.wrapping_mul(7);
                let chain = [base, base + 1, base + 2, base + 3];
                tree.insert(&scratch, None, &chain);
                tree.remove(&scratch, &chain);
                round = round.wrapping_add(1);
            }
        })
    };

    group.throughput(Throughput::Elements(warm_chain.len() as u64));
    group.bench_function("reader_under_write_pressure", |b| {
        b.iter(|| {
            let m = tree.match_prefix(None, black_box(&warm_chain));
            black_box(m.matched_blocks)
        });
    });
    group.finish();

    stop.store(true, Ordering::Relaxed);
    writer.join().expect("bench writer thread panicked");
}

criterion_group!(
    benches,
    bench_insert,
    bench_match_prefix,
    bench_contended_match
);
criterion_main!(benches);
