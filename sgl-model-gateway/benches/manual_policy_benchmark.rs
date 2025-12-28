use std::{sync::Arc, thread};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sgl_model_gateway::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{LoadBalancingPolicy, ManualPolicy, SelectWorkerInfo},
};

// ============================================================================
// Test Helpers
// ============================================================================

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

fn select_with_key<'a>(
    policy: &ManualPolicy,
    workers: &[Arc<dyn Worker>],
    key: &str,
    headers_buf: &'a mut http::HeaderMap,
) -> Option<usize> {
    *headers_buf = headers_with_routing_key(key);
    let info = SelectWorkerInfo {
        headers: Some(headers_buf),
        ..Default::default()
    };
    policy.select_worker(workers, &info)
}

fn warmup_keys(policy: &ManualPolicy, workers: &[Arc<dyn Worker>], keys: &[String]) {
    let mut headers = http::HeaderMap::new();
    for key in keys {
        select_with_key(policy, workers, key, &mut headers);
    }
}

fn gen_keys(count: usize, prefix: &str) -> Vec<String> {
    (0..count).map(|i| format!("{}{}", prefix, i)).collect()
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_fast_path_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/fast_path");

    for worker_count in [4, 16, 64, 256] {
        let policy = ManualPolicy::new();
        let workers = create_workers(worker_count);
        let keys = gen_keys(1000, "user-");
        warmup_keys(&policy, &workers, &keys);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            &worker_count,
            |b, _| {
                let mut idx = 0;
                let mut headers = http::HeaderMap::new();
                b.iter(|| {
                    let result =
                        select_with_key(&policy, &workers, &keys[idx % keys.len()], &mut headers);
                    idx += 1;
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_slow_path_vacant(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/slow_path_vacant");

    for worker_count in [4, 16, 64, 256] {
        let workers = create_workers(worker_count);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            &worker_count,
            |b, _| {
                let policy = ManualPolicy::new();
                let mut idx = 0;
                let mut headers = http::HeaderMap::new();
                b.iter(|| {
                    let key = format!("new-user-{}", idx);
                    let result = select_with_key(&policy, &workers, &key, &mut headers);
                    idx += 1;
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_no_routing_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/no_routing_key");

    for worker_count in [4, 16, 64, 256] {
        let policy = ManualPolicy::new();
        let workers = create_workers(worker_count);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            &worker_count,
            |b, _| {
                let info = SelectWorkerInfo::default();
                b.iter(|| black_box(policy.select_worker(&workers, &info)));
            },
        );
    }
    group.finish();
}

fn bench_failover(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/failover");
    group.sample_size(50);

    for worker_count in [4, 16, 64] {
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            &worker_count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let policy = ManualPolicy::new();
                        let workers = create_workers(count);
                        let mut headers = http::HeaderMap::new();
                        let idx =
                            select_with_key(&policy, &workers, "failover-test", &mut headers)
                                .unwrap();
                        workers[idx].set_healthy(false);
                        (policy, workers)
                    },
                    |(policy, workers)| {
                        let mut headers = http::HeaderMap::new();
                        black_box(select_with_key(
                            &policy,
                            &workers,
                            "failover-test",
                            &mut headers,
                        ))
                    },
                );
            },
        );
    }
    group.finish();
}

fn bench_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/concurrent");
    group.sample_size(50);

    for num_threads in [2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let policy = Arc::new(ManualPolicy::new());
                    let workers: Arc<Vec<Arc<dyn Worker>>> = Arc::new(create_workers(16));

                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let policy = Arc::clone(&policy);
                            let workers = Arc::clone(&workers);
                            thread::spawn(move || {
                                let mut headers = http::HeaderMap::new();
                                for i in 0..500 {
                                    let key = if i % 5 == 0 {
                                        format!("thread{}_user{}", t, i)
                                    } else {
                                        format!("shared_user{}", i % 50)
                                    };
                                    headers = headers_with_routing_key(&key);
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

fn bench_cache_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/cache_size");

    for cache_size in [100, 1000, 10000, 100000] {
        let policy = ManualPolicy::new();
        let workers = create_workers(16);
        let keys = gen_keys(cache_size, "user-");
        warmup_keys(&policy, &workers, &keys);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("keys", cache_size), &cache_size, |b, _| {
            let mut idx = 0;
            let mut headers = http::HeaderMap::new();
            b.iter(|| {
                let result =
                    select_with_key(&policy, &workers, &keys[idx % keys.len()], &mut headers);
                idx += 1;
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_comparison_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_policy/vs_baseline");
    let workers = create_workers(16);
    let policy = ManualPolicy::new();

    group.bench_function("baseline_random", |b| {
        b.iter(|| black_box(rand::random::<usize>() % workers.len()));
    });

    group.bench_function("manual_no_key", |b| {
        let info = SelectWorkerInfo::default();
        b.iter(|| black_box(policy.select_worker(&workers, &info)));
    });

    let mut headers = http::HeaderMap::new();
    select_with_key(&policy, &workers, "cached-user", &mut headers);

    group.bench_function("manual_fast_path", |b| {
        let mut headers = http::HeaderMap::new();
        b.iter(|| black_box(select_with_key(&policy, &workers, "cached-user", &mut headers)));
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
    bench_cache_size_impact,
    bench_comparison_baseline,
);
criterion_main!(benches);
