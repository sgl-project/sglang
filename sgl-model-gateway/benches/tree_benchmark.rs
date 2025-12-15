//! Benchmarks for the radix tree implementation used in cache-aware routing.
//!
//! Run with: cargo bench --bench tree_benchmark

use std::{sync::Arc, thread};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{
    distr::{Alphanumeric, SampleString},
    rng as thread_rng,
};
// Import the tree module
use sgl_model_gateway::policies::tree::Tree;

/// Generate random ASCII strings of given length
fn random_ascii_string(len: usize) -> String {
    Alphanumeric.sample_string(&mut thread_rng(), len)
}

/// Generate random strings with common prefixes (simulates real request patterns)
fn random_prefixed_strings(prefix: &str, suffix_len: usize, count: usize) -> Vec<String> {
    (0..count)
        .map(|_| format!("{}{}", prefix, random_ascii_string(suffix_len)))
        .collect()
}

/// Benchmark single-threaded insert throughput
fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");

    for text_len in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("random_text", text_len),
            text_len,
            |b, &len| {
                let tree = Tree::new();
                let strings: Vec<String> = (0..1000).map(|_| random_ascii_string(len)).collect();
                let mut idx = 0;

                b.iter(|| {
                    tree.insert(black_box(&strings[idx % strings.len()]), "tenant1");
                    idx += 1;
                });
            },
        );
    }

    // Benchmark with shared prefixes (common cache scenario)
    group.bench_function("shared_prefix_100", |b| {
        let tree = Tree::new();
        let prefixes = ["system:", "user:", "assistant:", "tool:"];
        let strings: Vec<String> = prefixes
            .iter()
            .flat_map(|p| random_prefixed_strings(p, 50, 250))
            .collect();
        let mut idx = 0;

        b.iter(|| {
            tree.insert(black_box(&strings[idx % strings.len()]), "tenant1");
            idx += 1;
        });
    });

    group.finish();
}

/// Benchmark prefix_match latency
fn bench_prefix_match_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_match_latency");

    // Setup: pre-populate tree with data
    let tree = Tree::new();
    let prefixes = ["system:", "user:", "assistant:", "tool:"];
    let strings: Vec<String> = prefixes
        .iter()
        .flat_map(|p| random_prefixed_strings(p, 50, 1000))
        .collect();

    for s in &strings {
        tree.insert(s, "tenant1");
    }

    // Benchmark cache hit (exact match)
    group.bench_function("cache_hit", |b| {
        let mut idx = 0;
        b.iter(|| {
            let result = tree.prefix_match(black_box(&strings[idx % strings.len()]));
            idx += 1;
            result
        });
    });

    // Benchmark cache miss (no match)
    let miss_strings: Vec<String> = (0..1000).map(|_| random_ascii_string(50)).collect();
    group.bench_function("cache_miss", |b| {
        let mut idx = 0;
        b.iter(|| {
            let result = tree.prefix_match(black_box(&miss_strings[idx % miss_strings.len()]));
            idx += 1;
            result
        });
    });

    // Benchmark partial match
    group.bench_function("partial_match", |b| {
        let partial_strings: Vec<String> = prefixes
            .iter()
            .map(|p| format!("{}partial_query", p))
            .collect();
        let mut idx = 0;
        b.iter(|| {
            let result =
                tree.prefix_match(black_box(&partial_strings[idx % partial_strings.len()]));
            idx += 1;
            result
        });
    });

    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");
    group.sample_size(50); // Reduce sample size for concurrent tests

    // Mixed read/write workload
    for num_threads in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("mixed_workload", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let tree = Arc::new(Tree::new());
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            thread::spawn(move || {
                                let tenant = format!("tenant{}", t);
                                for i in 0..100 {
                                    let text = format!("thread{}_request{}", t, i);
                                    if i % 3 == 0 {
                                        tree.prefix_match(&text);
                                    } else {
                                        tree.insert(&text, &tenant);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark eviction performance
fn bench_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction");
    group.sample_size(20); // Eviction is expensive

    for tree_size in [1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("evict_to_half", tree_size),
            tree_size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        // Setup: create tree with many entries
                        let tree = Tree::new();
                        for i in 0..size {
                            tree.insert(&format!("entry_{:05}", i), "tenant1");
                        }
                        tree
                    },
                    |tree| {
                        // Evict to half size
                        tree.evict_tenant_by_size(size / 2);
                    },
                );
            },
        );
    }

    group.finish();
}

/// Benchmark UTF-8 handling vs ASCII
fn bench_utf8_vs_ascii(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding");

    let tree_ascii = Tree::new();
    let tree_utf8 = Tree::new();

    // Pre-populate
    let ascii_strings: Vec<String> = (0..1000).map(|_| random_ascii_string(50)).collect();
    let utf8_strings: Vec<String> = (0..1000).map(|i| format!("你好世界_{}", i)).collect();

    for s in &ascii_strings {
        tree_ascii.insert(s, "tenant1");
    }
    for s in &utf8_strings {
        tree_utf8.insert(s, "tenant1");
    }

    group.bench_function("ascii_match", |b| {
        let mut idx = 0;
        b.iter(|| {
            let result =
                tree_ascii.prefix_match(black_box(&ascii_strings[idx % ascii_strings.len()]));
            idx += 1;
            result
        });
    });

    group.bench_function("utf8_match", |b| {
        let mut idx = 0;
        b.iter(|| {
            let result = tree_utf8.prefix_match(black_box(&utf8_strings[idx % utf8_strings.len()]));
            idx += 1;
            result
        });
    });

    group.finish();
}

/// Benchmark multi-tenant scenarios
fn bench_multi_tenant(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_tenant");

    let tree = Tree::new();

    // Setup: multiple tenants with overlapping data
    let tenants = ["worker1", "worker2", "worker3", "worker4"];
    let prefixes = ["prompt:", "completion:", "context:"];

    for tenant in &tenants {
        for prefix in &prefixes {
            for i in 0..100 {
                tree.insert(&format!("{}data_{}", prefix, i), tenant);
            }
        }
    }

    group.bench_function("shared_prefix_lookup", |b| {
        let queries: Vec<String> = prefixes
            .iter()
            .flat_map(|p| (0..10).map(move |i| format!("{}data_{}", p, i)))
            .collect();
        let mut idx = 0;

        b.iter(|| {
            let result = tree.prefix_match(black_box(&queries[idx % queries.len()]));
            idx += 1;
            result
        });
    });

    group.bench_function("tenant_specific_match", |b| {
        let queries: Vec<(String, &str)> = tenants
            .iter()
            .flat_map(|&t| (0..10).map(move |i| (format!("prompt:data_{}", i), t)))
            .collect();
        let mut idx = 0;

        b.iter(|| {
            let (query, tenant) = &queries[idx % queries.len()];
            let result = tree.prefix_match_tenant(black_box(query), black_box(tenant));
            idx += 1;
            result
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_throughput,
    bench_prefix_match_latency,
    bench_concurrent_operations,
    bench_eviction,
    bench_utf8_vs_ascii,
    bench_multi_tenant,
);
criterion_main!(benches);
