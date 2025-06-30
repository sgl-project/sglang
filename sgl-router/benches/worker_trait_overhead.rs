//! Performance benchmarks for Worker trait overhead
//!
//! This benchmark suite measures the performance impact of using trait objects
//! for worker management, including:
//! - Trait object dispatch overhead vs direct calls
//! - Memory allocation overhead of Arc<dyn Worker>
//! - Concurrent worker operations performance
//! - Worker creation and cloning overhead

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sglang_router_rs::core::worker::{RegularWorker, Worker, WorkerFactory, WorkerType};

// Direct implementation for comparison (bypassing trait object)
struct DirectWorker {
    url: String,
    load: Arc<AtomicUsize>,
}

impl DirectWorker {
    fn new(url: String) -> Self {
        Self {
            url,
            load: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn url(&self) -> &str {
        &self.url
    }

    fn worker_type(&self) -> WorkerType {
        WorkerType::Regular
    }

    fn load(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.load)
    }

    fn is_healthy(&self) -> bool {
        true // Simplified for benchmarking
    }
}

fn bench_worker_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_creation");

    // Benchmark trait object creation
    group.bench_function("trait_object_creation", |b| {
        b.iter(|| {
            let worker = WorkerFactory::create_regular(black_box("http://test:8080".to_string()));
            black_box(worker);
        });
    });

    // Benchmark direct worker creation
    group.bench_function("direct_worker_creation", |b| {
        b.iter(|| {
            let worker = DirectWorker::new(black_box("http://test:8080".to_string()));
            black_box(worker);
        });
    });

    // Benchmark different worker types creation
    group.bench_function("regular_worker_creation", |b| {
        b.iter(|| {
            let worker = WorkerFactory::create_regular(black_box("http://test:8080".to_string()));
            black_box(worker);
        });
    });

    group.bench_function("decode_worker_creation", |b| {
        b.iter(|| {
            let worker = WorkerFactory::create_decode(black_box("http://test:8080".to_string()));
            black_box(worker);
        });
    });

    group.bench_function("prefill_worker_creation", |b| {
        b.iter(|| {
            let worker = WorkerFactory::create_prefill(
                black_box("http://test:8080".to_string()),
                black_box(Some(9000)),
            );
            black_box(worker);
        });
    });

    group.finish();
}

fn bench_method_dispatch_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_dispatch");

    let trait_worker = WorkerFactory::create_regular("http://test:8080".to_string());
    let direct_worker = DirectWorker::new("http://test:8080".to_string());

    // URL access benchmarks
    group.bench_function("trait_object_url_access", |b| {
        b.iter(|| {
            let url = black_box(&trait_worker).url();
            black_box(url);
        });
    });

    group.bench_function("direct_worker_url_access", |b| {
        b.iter(|| {
            let url = black_box(&direct_worker).url();
            black_box(url);
        });
    });

    // Worker type access benchmarks
    group.bench_function("trait_object_worker_type", |b| {
        b.iter(|| {
            let worker_type = black_box(&trait_worker).worker_type();
            black_box(worker_type);
        });
    });

    group.bench_function("direct_worker_worker_type", |b| {
        b.iter(|| {
            let worker_type = black_box(&direct_worker).worker_type();
            black_box(worker_type);
        });
    });

    // Load counter access benchmarks
    group.bench_function("trait_object_load_access", |b| {
        b.iter(|| {
            let load = black_box(&trait_worker).load();
            black_box(load);
        });
    });

    group.bench_function("direct_worker_load_access", |b| {
        b.iter(|| {
            let load = black_box(&direct_worker).load();
            black_box(load);
        });
    });

    // Health check benchmarks
    group.bench_function("trait_object_health_check", |b| {
        b.iter(|| {
            let is_healthy = black_box(&trait_worker).is_healthy();
            black_box(is_healthy);
        });
    });

    group.bench_function("direct_worker_health_check", |b| {
        b.iter(|| {
            let is_healthy = black_box(&direct_worker).is_healthy();
            black_box(is_healthy);
        });
    });

    group.finish();
}

fn bench_worker_cloning(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_cloning");

    let trait_worker = WorkerFactory::create_regular("http://test:8080".to_string());

    // Trait object cloning (Arc clone)
    group.bench_function("trait_object_arc_clone", |b| {
        b.iter(|| {
            let cloned = black_box(&trait_worker).clone();
            black_box(cloned);
        });
    });

    // Direct worker cloning
    group.bench_function("direct_worker_clone", |b| {
        let direct_worker = Arc::new(DirectWorker::new("http://test:8080".to_string()));
        b.iter(|| {
            let cloned = black_box(&direct_worker).clone();
            black_box(cloned);
        });
    });

    group.finish();
}

fn bench_concurrent_worker_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");

    // Set up workers for concurrent access
    let trait_worker = WorkerFactory::create_regular("http://test:8080".to_string());
    let direct_worker = Arc::new(DirectWorker::new("http://test:8080".to_string()));

    // Benchmark concurrent load counter increments
    group.bench_function("trait_object_concurrent_load_increment", |b| {
        b.iter(|| {
            let worker = black_box(&trait_worker);
            for _ in 0..100 {
                worker.load().fetch_add(1, Ordering::Relaxed);
            }
        });
    });

    group.bench_function("direct_worker_concurrent_load_increment", |b| {
        b.iter(|| {
            let worker = black_box(&direct_worker);
            for _ in 0..100 {
                worker.load().fetch_add(1, Ordering::Relaxed);
            }
        });
    });

    // Benchmark concurrent health checks
    group.bench_function("trait_object_concurrent_health_checks", |b| {
        b.iter(|| {
            let worker = black_box(&trait_worker);
            for _ in 0..100 {
                let _ = worker.is_healthy();
            }
        });
    });

    group.bench_function("direct_worker_concurrent_health_checks", |b| {
        b.iter(|| {
            let worker = black_box(&direct_worker);
            for _ in 0..100 {
                let _ = worker.is_healthy();
            }
        });
    });

    group.finish();
}

fn bench_worker_collections(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_collections");

    // Test performance with different collection sizes
    for size in [10, 100, 1000].iter() {
        // Create collections of workers
        let trait_workers: Vec<Arc<dyn Worker>> = (0..*size)
            .map(|i| WorkerFactory::create_regular(format!("http://test{}:8080", i)))
            .collect();

        let direct_workers: Vec<Arc<DirectWorker>> = (0..*size)
            .map(|i| Arc::new(DirectWorker::new(format!("http://test{}:8080", i))))
            .collect();

        // Benchmark iteration over trait objects
        group.bench_with_input(
            BenchmarkId::new("trait_object_iteration", size),
            size,
            |b, _| {
                b.iter(|| {
                    for worker in black_box(&trait_workers) {
                        let _ = worker.is_healthy();
                        let _ = worker.url();
                    }
                });
            },
        );

        // Benchmark iteration over direct workers
        group.bench_with_input(
            BenchmarkId::new("direct_worker_iteration", size),
            size,
            |b, _| {
                b.iter(|| {
                    for worker in black_box(&direct_workers) {
                        let _ = worker.is_healthy();
                        let _ = worker.url();
                    }
                });
            },
        );

        // Benchmark load counter operations on collections
        group.bench_with_input(
            BenchmarkId::new("trait_object_load_operations", size),
            size,
            |b, _| {
                b.iter(|| {
                    for worker in black_box(&trait_workers) {
                        worker.load().fetch_add(1, Ordering::Relaxed);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("direct_worker_load_operations", size),
            size,
            |b, _| {
                b.iter(|| {
                    for worker in black_box(&direct_workers) {
                        worker.load().fetch_add(1, Ordering::Relaxed);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    group.throughput(Throughput::Elements(1));

    // Benchmark memory allocation patterns for different worker types
    group.bench_function("regular_worker_allocation", |b| {
        b.iter(|| {
            let worker = WorkerFactory::create_regular(black_box("http://test:8080".to_string()));
            // Force allocation and deallocation
            drop(worker);
        });
    });

    group.bench_function("decode_worker_allocation", |b| {
        b.iter(|| {
            let worker = WorkerFactory::create_decode(black_box("http://test:8080".to_string()));
            drop(worker);
        });
    });

    group.bench_function("prefill_worker_allocation", |b| {
        b.iter(|| {
            let worker = WorkerFactory::create_prefill(
                black_box("http://test:8080".to_string()),
                black_box(Some(9000)),
            );
            drop(worker);
        });
    });

    // Benchmark bulk allocation and deallocation
    group.bench_function("bulk_worker_allocation", |b| {
        b.iter(|| {
            let workers: Vec<Arc<dyn Worker>> = (0..100)
                .map(|i| WorkerFactory::create_regular(format!("http://test{}:8080", i)))
                .collect();
            black_box(workers);
        });
    });

    group.finish();
}

fn bench_worker_type_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_type_operations");

    let worker_types = vec![
        WorkerType::Regular,
        WorkerType::Decode,
        WorkerType::Prefill(None),
        WorkerType::Prefill(Some(9000)),
    ];

    // Benchmark worker type comparison
    group.bench_function("worker_type_equality", |b| {
        b.iter(|| {
            for (i, worker_type) in black_box(&worker_types).iter().enumerate() {
                for (j, other_type) in worker_types.iter().enumerate() {
                    let _ = black_box(worker_type == other_type);
                    if i != j {
                        let _ = black_box(worker_type != other_type);
                    }
                }
            }
        });
    });

    // Benchmark worker type display formatting
    group.bench_function("worker_type_display", |b| {
        b.iter(|| {
            for worker_type in black_box(&worker_types) {
                let display_str = format!("{}", worker_type);
                black_box(display_str);
            }
        });
    });

    // Benchmark worker type endpoint configuration
    group.bench_function("worker_type_endpoints", |b| {
        b.iter(|| {
            for worker_type in black_box(&worker_types) {
                let endpoints = worker_type.get_endpoints();
                black_box(endpoints);
            }
        });
    });

    group.finish();
}

criterion_group!(
    worker_trait_benches,
    bench_worker_creation,
    bench_method_dispatch_overhead,
    bench_worker_cloning,
    bench_concurrent_worker_operations,
    bench_worker_collections,
    bench_memory_allocation_patterns,
    bench_worker_type_operations
);

criterion_main!(worker_trait_benches);
