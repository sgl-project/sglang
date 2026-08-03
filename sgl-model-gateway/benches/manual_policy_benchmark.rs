use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smg::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{LoadBalancingPolicy, ManualPolicy, SelectWorkerInfo},
};
use tokio::runtime::Runtime;

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

fn select_with_key(
    rt: &Runtime,
    policy: &ManualPolicy,
    workers: &[Arc<dyn Worker>],
    key: &str,
) -> Option<usize> {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-routing-key", key.parse().unwrap());
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    rt.block_on(policy.select_worker(workers, &info))
}

fn warmup_keys(rt: &Runtime, policy: &ManualPolicy, workers: &[Arc<dyn Worker>], keys: &[String]) {
    for key in keys {
        select_with_key(rt, policy, workers, key);
    }
}

fn gen_keys(count: usize, prefix: &str) -> Vec<String> {
    (0..count).map(|i| format!("{}{}", prefix, i)).collect()
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_fast_path_hit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("manual_policy/fast_path");

    for worker_count in [4, 16, 64, 256] {
        let policy = ManualPolicy::new();
        let workers = create_workers(worker_count);
        let keys = gen_keys(1000, "user-");
        warmup_keys(&rt, &policy, &workers, &keys);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            &worker_count,
            |b, _| {
                let mut idx = 0;
                b.iter(|| {
                    let result = select_with_key(&rt, &policy, &workers, &keys[idx % keys.len()]);
                    idx += 1;
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_slow_path_vacant(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
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
                b.iter(|| {
                    let key = format!("new-user-{}", idx);
                    let result = select_with_key(&rt, &policy, &workers, &key);
                    idx += 1;
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_no_routing_key(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
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
                b.iter(|| black_box(rt.block_on(policy.select_worker(&workers, &info))));
            },
        );
    }
    group.finish();
}

fn bench_failover(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
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
                        let idx = select_with_key(&rt, &policy, &workers, "failover-test").unwrap();
                        workers[idx].set_healthy(false);
                        (policy, workers)
                    },
                    |(policy, workers)| {
                        black_box(select_with_key(&rt, &policy, &workers, "failover-test"))
                    },
                );
            },
        );
    }
    group.finish();
}

fn bench_concurrent(c: &mut Criterion) {
    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .build()
            .unwrap(),
    );
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

                    rt.block_on(async {
                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let policy = Arc::clone(&policy);
                                let workers = Arc::clone(&workers);
                                tokio::spawn(async move {
                                    for i in 0..500 {
                                        let key = if i % 5 == 0 {
                                            format!("thread{}_user{}", t, i)
                                        } else {
                                            format!("shared_user{}", i % 50)
                                        };
                                        let mut headers = http::HeaderMap::new();
                                        headers.insert("x-smg-routing-key", key.parse().unwrap());
                                        let info = SelectWorkerInfo {
                                            headers: Some(&headers),
                                            ..Default::default()
                                        };
                                        let _ =
                                            black_box(policy.select_worker(&workers, &info).await);
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.await.unwrap();
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

fn bench_cache_size_impact(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("manual_policy/cache_size");

    for cache_size in [100, 1000, 10000, 100000] {
        let policy = ManualPolicy::new();
        let workers = create_workers(16);
        let keys = gen_keys(cache_size, "user-");
        warmup_keys(&rt, &policy, &workers, &keys);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("keys", cache_size), &cache_size, |b, _| {
            let mut idx = 0;
            b.iter(|| {
                let result = select_with_key(&rt, &policy, &workers, &keys[idx % keys.len()]);
                idx += 1;
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_comparison_baseline(c: &mut Criterion) {
    use rand::Rng;

    let mut group = c.benchmark_group("manual_policy/vs_baseline");
    let workers = create_workers(16);

    // Baseline: raw random selection without any policy overhead
    group.bench_function("raw_random", |b| {
        let mut rng = rand::rng();
        b.iter(|| black_box(rng.random_range(0..workers.len())));
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
