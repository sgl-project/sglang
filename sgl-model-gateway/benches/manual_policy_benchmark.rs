//! Benchmarks for ManualPolicy - sticky session routing based on routing key header
//!
//! Run with: cargo bench --bench manual_policy_benchmark
//!
//! Key performance aspects tested:
//! - Fast path vs Slow path hit rates
//! - DashMap concurrent read/write performance
//! - Worker lookup efficiency (O(n) linear search)
//! - Scalability across different worker counts
//! - Concurrent access patterns

use std::{sync::Arc, thread};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sgl_model_gateway::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{LoadBalancingPolicy, ManualPolicy, SelectWorkerInfo},
};

fn create_workers(count: usize) -> Vec<Arc<dyn Worker>> {
    (0..count)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{}:8000", i))
                    .worker_type(WorkerType::Regular)
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}

fn headers_with_routing_key(key: &str) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-routing-key", key.parse().unwrap());
    headers
}

/// Benchmark fast path hit performance (routing key already cached)
fn bench_fast_path_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/fast_path");

    for worker_count in [4, 16, 64, 256].iter() {
        let policy = ManualPolicy::new();
        let workers = create_workers(*worker_count);

        // Pre-warm the cache with routing keys
        let routing_keys: Vec<String> = (0..1000).map(|i| format!("user-{}", i)).collect();
        for key in &routing_keys {
            let headers = headers_with_routing_key(key);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            policy.select_worker(&workers, &info);
        }

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, _| {
                let mut idx = 0;
                b.iter(|| {
                    let key = &routing_keys[idx % routing_keys.len()];
                    let headers = headers_with_routing_key(key);
                    let info = SelectWorkerInfo {
                        headers: Some(&headers),
                        ..Default::default()
                    };
                    let result = policy.select_worker(black_box(&workers), black_box(&info));
                    idx += 1;
                    result
                });
            },
        );
    }

    group.finish();
}

/// Benchmark slow path performance (new routing key)
fn bench_slow_path_vacant(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/slow_path_vacant");

    for worker_count in [4, 16, 64, 256].iter() {
        let workers = create_workers(*worker_count);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, _| {
                // Fresh policy each iteration to ensure vacant path
                let policy = ManualPolicy::new();
                let mut idx = 0;
                b.iter(|| {
                    let key = format!("new-user-{}", idx);
                    let headers = headers_with_routing_key(&key);
                    let info = SelectWorkerInfo {
                        headers: Some(&headers),
                        ..Default::default()
                    };
                    let result = policy.select_worker(black_box(&workers), black_box(&info));
                    idx += 1;
                    result
                });
            },
        );
    }

    group.finish();
}

/// Benchmark no routing key (random fallback)
fn bench_no_routing_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/no_routing_key");

    for worker_count in [4, 16, 64, 256].iter() {
        let policy = ManualPolicy::new();
        let workers = create_workers(*worker_count);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, _| {
                b.iter(|| {
                    let info = SelectWorkerInfo::default();
                    policy.select_worker(black_box(&workers), black_box(&info))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark failover scenario (cached worker becomes unhealthy)
fn bench_failover(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/failover");
    group.sample_size(50);

    for worker_count in [4, 16, 64].iter() {
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let policy = ManualPolicy::new();
                        let workers = create_workers(count);

                        // Warm up with routing key
                        let headers = headers_with_routing_key("failover-test");
                        let info = SelectWorkerInfo {
                            headers: Some(&headers),
                            ..Default::default()
                        };
                        let idx = policy.select_worker(&workers, &info).unwrap();

                        // Mark the selected worker as unhealthy
                        workers[idx].set_healthy(false);

                        (policy, workers)
                    },
                    |(policy, workers)| {
                        let headers = headers_with_routing_key("failover-test");
                        let info = SelectWorkerInfo {
                            headers: Some(&headers),
                            ..Default::default()
                        };
                        policy.select_worker(black_box(&workers), black_box(&info))
                    },
                );
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent access patterns
fn bench_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/concurrent");
    group.sample_size(50);

    for num_threads in [2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            num_threads,
            |b, &threads| {
                b.iter(|| {
                    let policy = Arc::new(ManualPolicy::new());
                    let workers: Arc<Vec<Arc<dyn Worker>>> = Arc::new(create_workers(16));

                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let policy = Arc::clone(&policy);
                            let workers = Arc::clone(&workers);
                            thread::spawn(move || {
                                for i in 0..500 {
                                    // Mix of fast path (reusing keys) and slow path (new keys)
                                    let key = if i % 5 == 0 {
                                        format!("thread{}_user{}", t, i)
                                    } else {
                                        format!("shared_user{}", i % 50)
                                    };
                                    let headers = headers_with_routing_key(&key);
                                    let info = SelectWorkerInfo {
                                        headers: Some(&headers),
                                        ..Default::default()
                                    };
                                    let _ = black_box(policy.select_worker(&workers, &info));
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

/// Benchmark read-heavy workload (90% fast path, 10% new keys)
fn bench_read_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/read_heavy");

    let policy = ManualPolicy::new();
    let workers = create_workers(32);

    // Pre-warm with 100 routing keys
    let cached_keys: Vec<String> = (0..100).map(|i| format!("cached-user-{}", i)).collect();
    for key in &cached_keys {
        let headers = headers_with_routing_key(key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        policy.select_worker(&workers, &info);
    }

    group.throughput(Throughput::Elements(1));
    group.bench_function("90_read_10_write", |b| {
        let mut idx = 0;
        b.iter(|| {
            let key = if idx % 10 == 0 {
                format!("new-user-{}", idx)
            } else {
                cached_keys[idx % cached_keys.len()].clone()
            };
            let headers = headers_with_routing_key(&key);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            let result = policy.select_worker(black_box(&workers), black_box(&info));
            idx += 1;
            result
        });
    });

    group.finish();
}

/// Benchmark with varying cache sizes (number of unique routing keys)
fn bench_cache_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/cache_size");

    for cache_size in [100, 1000, 10000, 100000].iter() {
        let policy = ManualPolicy::new();
        let workers = create_workers(16);

        // Pre-warm cache
        let routing_keys: Vec<String> =
            (0..*cache_size).map(|i| format!("user-{}", i)).collect();
        for key in &routing_keys {
            let headers = headers_with_routing_key(key);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            policy.select_worker(&workers, &info);
        }

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("keys", cache_size),
            cache_size,
            |b, _| {
                let mut idx = 0;
                b.iter(|| {
                    let key = &routing_keys[idx % routing_keys.len()];
                    let headers = headers_with_routing_key(key);
                    let info = SelectWorkerInfo {
                        headers: Some(&headers),
                        ..Default::default()
                    };
                    let result = policy.select_worker(black_box(&workers), black_box(&info));
                    idx += 1;
                    result
                });
            },
        );
    }

    group.finish();
}

/// Benchmark header parsing overhead
fn bench_header_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/header_parsing");

    let policy = ManualPolicy::new();
    let workers = create_workers(8);

    // Pre-warm
    let headers = headers_with_routing_key("test-user");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    policy.select_worker(&workers, &info);

    // Different routing key lengths
    for key_len in [8, 32, 128, 512].iter() {
        let key: String = (0..*key_len).map(|_| 'x').collect();

        // Pre-warm this key
        let headers = headers_with_routing_key(&key);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        policy.select_worker(&workers, &info);

        group.bench_with_input(BenchmarkId::new("key_len", key_len), key_len, |b, _| {
            let headers = headers_with_routing_key(&key);
            b.iter(|| {
                let info = SelectWorkerInfo {
                    headers: Some(&headers),
                    ..Default::default()
                };
                policy.select_worker(black_box(&workers), black_box(&info))
            });
        });
    }

    group.finish();
}

/// Compare ManualPolicy against baseline (no policy, direct random selection)
fn bench_comparison_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/vs_baseline");

    let workers = create_workers(16);

    // Baseline: direct random selection
    group.bench_function("baseline_random", |b| {
        b.iter(|| {
            let idx = rand::random::<usize>() % workers.len();
            black_box(idx)
        });
    });

    // ManualPolicy with no routing key (should be similar to baseline)
    let policy = ManualPolicy::new();
    group.bench_function("manual_no_key", |b| {
        b.iter(|| {
            let info = SelectWorkerInfo::default();
            policy.select_worker(black_box(&workers), black_box(&info))
        });
    });

    // ManualPolicy with fast path hit
    let headers = headers_with_routing_key("cached-user");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    policy.select_worker(&workers, &info); // Warm up

    group.bench_function("manual_fast_path", |b| {
        b.iter(|| {
            let headers = headers_with_routing_key("cached-user");
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            policy.select_worker(black_box(&workers), black_box(&info))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fast_path_hit,
    bench_slow_path_vacant,
    bench_no_routing_key,
    bench_failover,
    bench_concurrent,
    bench_read_heavy,
    bench_cache_size_impact,
    bench_header_parsing,
    bench_comparison_baseline,
);
criterion_main!(benches);

