//! Benchmarks for the radix tree implementation used in cache-aware routing.
//!
//! This benchmark simulates realistic cache-aware routing scenarios with:
//! - Multiple tenants representing HTTP/gRPC endpoints (10 endpoints)
//! - High-pressure workloads with concurrent operations
//! - Realistic request text patterns (system prompts, user queries, etc.)
//!
//! Run with: cargo bench --bench tree_benchmark
//!
//! For quick validation (CI): cargo bench --bench tree_benchmark -- benchmark_summary --exact

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{
    distr::{Alphanumeric, SampleString},
    rng as thread_rng, Rng,
};
// Import the tree module
use sgl_model_gateway::policies::tree::Tree;

// Global results storage for summary
lazy_static::lazy_static! {
    static ref BENCHMARK_RESULTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
}

fn add_result(category: &str, result: String) {
    let mut results = BENCHMARK_RESULTS.lock().unwrap();
    let index = results.len();
    let key = format!("{:03}_{}", index, category);
    // Print result immediately so it's captured even if process is killed later
    eprintln!("[BENCH_RESULT] {} | {}", category, result);
    results.insert(key, result);
}

/// Simulated HTTP/gRPC endpoints representing worker nodes
/// These mirror real-world deployment patterns with 10 tenants
const ENDPOINT_TENANTS: [&str; 10] = [
    "http://worker-0.sglang.svc.cluster.local:8000",
    "http://worker-1.sglang.svc.cluster.local:8000",
    "http://worker-2.sglang.svc.cluster.local:8000",
    "http://worker-3.sglang.svc.cluster.local:8000",
    "http://worker-4.sglang.svc.cluster.local:8000",
    "grpc://worker-5.sglang.svc.cluster.local:50051",
    "grpc://worker-6.sglang.svc.cluster.local:50051",
    "grpc://worker-7.sglang.svc.cluster.local:50051",
    "http://10.0.0.100:8000",
    "http://10.0.0.101:8000",
];

/// Common conversation prefixes that create shared tree paths
const CONVERSATION_PREFIXES: [&str; 6] = [
    "<|system|>\nYou are a helpful assistant.\n<|user|>\n",
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
    "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n",
    "Human: ",
    "User: ",
    "### Instruction:\n",
];

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

/// Generate realistic LLM request texts with system prompts and user queries
fn generate_realistic_requests(count: usize) -> Vec<String> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let prefix_idx = rng.random_range(0..CONVERSATION_PREFIXES.len());
            // Realistic LLM request sizes: 1000-3000 chars (~250-750 tokens)
            // This represents typical user queries with context
            let query_len = rng.random_range(1000..3000);
            format!(
                "{}{}",
                CONVERSATION_PREFIXES[prefix_idx],
                random_ascii_string(query_len)
            )
        })
        .collect()
}

/// Benchmark single-threaded insert throughput with endpoint tenants
fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");

    for text_len in [10, 50, 100, 500].iter() {
        let printed = Arc::new(AtomicBool::new(false));
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("random_text", text_len),
            text_len,
            |b, &len| {
                let tree = Tree::new();
                let strings: Vec<String> = (0..1000).map(|_| random_ascii_string(len)).collect();
                let mut idx = 0;
                let printed_clone = printed.clone();

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let tenant = ENDPOINT_TENANTS[idx % ENDPOINT_TENANTS.len()];
                        tree.insert(black_box(&strings[idx % strings.len()]), tenant);
                        idx += 1;
                    }
                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let ops_per_sec = iters as f64 / duration.as_secs_f64();
                        let result = format!(
                            "{:<25} | {:>8} | {:>12.0} | {:>10}",
                            format!("random_text_{}", len),
                            len,
                            ops_per_sec,
                            10
                        );
                        add_result("insert", result);
                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    // Benchmark with shared prefixes (common cache scenario) - distributed across endpoints
    let printed_prefix = Arc::new(AtomicBool::new(false));
    group.bench_function("shared_prefix_100", |b| {
        let tree = Tree::new();
        let prefixes = ["system:", "user:", "assistant:", "tool:"];
        let strings: Vec<String> = prefixes
            .iter()
            .flat_map(|p| random_prefixed_strings(p, 50, 250))
            .collect();
        let mut idx = 0;
        let printed = printed_prefix.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let tenant = ENDPOINT_TENANTS[idx % ENDPOINT_TENANTS.len()];
                tree.insert(black_box(&strings[idx % strings.len()]), tenant);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>10}",
                    "shared_prefix", "~58", ops_per_sec, 10
                );
                add_result("insert", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Benchmark with realistic LLM request patterns
    let printed_llm = Arc::new(AtomicBool::new(false));
    group.bench_function("realistic_llm_requests", |b| {
        let tree = Tree::new();
        let requests = generate_realistic_requests(2000);
        let mut idx = 0;
        let printed = printed_llm.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let tenant = ENDPOINT_TENANTS[idx % ENDPOINT_TENANTS.len()];
                tree.insert(black_box(&requests[idx % requests.len()]), tenant);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>10}",
                    "realistic_llm", "~100", ops_per_sec, 10
                );
                add_result("insert", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

/// Benchmark prefix_match latency with multi-tenant tree
fn bench_prefix_match_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_match_latency");

    // Setup: pre-populate tree with data distributed across all endpoints
    let tree = Arc::new(Tree::new());
    let prefixes = ["system:", "user:", "assistant:", "tool:"];
    let strings: Vec<String> = prefixes
        .iter()
        .flat_map(|p| random_prefixed_strings(p, 50, 1000))
        .collect();

    // Distribute entries across all 10 endpoint tenants
    for (i, s) in strings.iter().enumerate() {
        let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
        tree.insert(s, tenant);
    }

    // Benchmark cache hit (exact match)
    let printed_hit = Arc::new(AtomicBool::new(false));
    let tree_clone = tree.clone();
    let strings_clone = strings.clone();
    group.bench_function("cache_hit", |b| {
        let mut idx = 0;
        let printed = printed_hit.clone();
        let tree = tree_clone.clone();
        let strings = strings_clone.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let result = tree.prefix_match(black_box(&strings[idx % strings.len()]));
                black_box(result);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let latency_ns = duration.as_nanos() as f64 / iters as f64;
                let result = format!(
                    "{:<20} | {:>12.0} | {:>12.1}",
                    "cache_hit", ops_per_sec, latency_ns
                );
                add_result("prefix_match", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Benchmark cache miss (no match)
    let miss_strings: Vec<String> = (0..1000).map(|_| random_ascii_string(50)).collect();
    let printed_miss = Arc::new(AtomicBool::new(false));
    let tree_clone = tree.clone();
    group.bench_function("cache_miss", |b| {
        let mut idx = 0;
        let printed = printed_miss.clone();
        let tree = tree_clone.clone();
        let miss_strings = miss_strings.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let result = tree.prefix_match(black_box(&miss_strings[idx % miss_strings.len()]));
                black_box(result);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let latency_ns = duration.as_nanos() as f64 / iters as f64;
                let result = format!(
                    "{:<20} | {:>12.0} | {:>12.1}",
                    "cache_miss", ops_per_sec, latency_ns
                );
                add_result("prefix_match", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Benchmark partial match
    let printed_partial = Arc::new(AtomicBool::new(false));
    let tree_clone = tree.clone();
    group.bench_function("partial_match", |b| {
        let partial_strings: Vec<String> = prefixes
            .iter()
            .map(|p| format!("{}partial_query", p))
            .collect();
        let mut idx = 0;
        let printed = printed_partial.clone();
        let tree = tree_clone.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let result =
                    tree.prefix_match(black_box(&partial_strings[idx % partial_strings.len()]));
                black_box(result);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let latency_ns = duration.as_nanos() as f64 / iters as f64;
                let result = format!(
                    "{:<20} | {:>12.0} | {:>12.1}",
                    "partial_match", ops_per_sec, latency_ns
                );
                add_result("prefix_match", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

/// Benchmark concurrent operations with high pressure (10 endpoint tenants)
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");
    group.sample_size(50); // Reduce sample size for concurrent tests

    // Mixed read/write workload with endpoint-style tenants
    for num_threads in [2, 4, 8, 16].iter() {
        let printed = Arc::new(AtomicBool::new(false));
        group.bench_with_input(
            BenchmarkId::new("mixed_workload", num_threads),
            num_threads,
            |b, &threads| {
                let printed_clone = printed.clone();

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let tree = Arc::new(Tree::new());
                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let tree = Arc::clone(&tree);
                                thread::spawn(move || {
                                    // Each thread uses a different endpoint tenant
                                    let tenant = ENDPOINT_TENANTS[t % ENDPOINT_TENANTS.len()];
                                    for i in 0..200 {
                                        let text = format!(
                                            "{}thread{}_request{}",
                                            CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()],
                                            t,
                                            i
                                        );
                                        if i % 3 == 0 {
                                            tree.prefix_match(&text);
                                        } else {
                                            tree.insert(&text, tenant);
                                        }
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }
                    }
                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total_ops = iters * threads as u64 * 200;
                        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                        let result = format!(
                            "{:<25} | {:>8} | {:>12.0} | {:>12.0}",
                            format!("mixed_workload_{}_threads", threads),
                            threads,
                            ops_per_sec,
                            ops_per_sec / threads as f64
                        );
                        add_result("concurrent", result);
                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    // High-contention scenario: all threads sharing same prefixes
    let printed_contention = Arc::new(AtomicBool::new(false));
    group.bench_function("high_contention_10_tenants", |b| {
        let printed = printed_contention.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let tree = Arc::new(Tree::new());
                let handles: Vec<_> = (0..10)
                    .map(|t| {
                        let tree = Arc::clone(&tree);
                        thread::spawn(move || {
                            let tenant = ENDPOINT_TENANTS[t];
                            // All threads insert similar prefixes to create contention
                            for i in 0..100 {
                                let text = format!(
                                    "<|system|>\nYou are a helpful assistant.\n<|user|>\nQuery {}",
                                    i
                                );
                                tree.insert(&text, tenant);
                                tree.prefix_match(&text);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let total_ops = iters * 10 * 200; // 10 threads * 200 ops (100 inserts + 100 matches)
                let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.0}",
                    "high_contention",
                    10,
                    ops_per_sec,
                    ops_per_sec / 10.0
                );
                add_result("concurrent", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

/// Benchmark eviction performance with multi-tenant scenarios
fn bench_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction");
    group.sample_size(20); // Eviction is expensive

    for tree_size in [1000, 5000, 10000].iter() {
        let printed = Arc::new(AtomicBool::new(false));
        group.bench_with_input(
            BenchmarkId::new("evict_to_half_single_tenant", tree_size),
            tree_size,
            |b, &size| {
                let printed_clone = printed.clone();

                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        // Setup: create tree with many entries for single tenant
                        let tree = Tree::new();
                        let tenant = ENDPOINT_TENANTS[0];
                        for i in 0..size {
                            tree.insert(&format!("entry_{:05}", i), tenant);
                        }

                        let start = Instant::now();
                        tree.evict_tenant_by_size(size / 2);
                        total_duration += start.elapsed();
                    }

                    if !printed_clone.load(Ordering::Relaxed) {
                        let ops_per_sec = iters as f64 / total_duration.as_secs_f64();
                        let latency_ms = total_duration.as_millis() as f64 / iters as f64;
                        let result = format!(
                            "{:<25} | {:>8} | {:>12.0} | {:>12.2}",
                            format!("single_tenant_{}", size),
                            size,
                            ops_per_sec,
                            latency_ms
                        );
                        add_result("eviction", result);
                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    total_duration
                });
            },
        );
    }

    // Multi-tenant eviction: 10 tenants with overlapping data
    for tree_size in [1000, 5000, 10000].iter() {
        let printed = Arc::new(AtomicBool::new(false));
        group.bench_with_input(
            BenchmarkId::new("evict_multi_tenant_10", tree_size),
            tree_size,
            |b, &size| {
                let printed_clone = printed.clone();

                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        // Setup: create tree with entries distributed across 10 tenants
                        let tree = Tree::new();
                        for i in 0..size {
                            let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
                            // Use shared prefixes to create overlapping tree structure
                            let prefix = CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()];
                            tree.insert(&format!("{}entry_{:05}", prefix, i), tenant);
                        }

                        let start = Instant::now();
                        tree.evict_tenant_by_size(size / 20);
                        total_duration += start.elapsed();
                    }

                    if !printed_clone.load(Ordering::Relaxed) {
                        let ops_per_sec = iters as f64 / total_duration.as_secs_f64();
                        let latency_ms = total_duration.as_millis() as f64 / iters as f64;
                        let result = format!(
                            "{:<25} | {:>8} | {:>12.0} | {:>12.2}",
                            format!("multi_tenant_{}", size),
                            size,
                            ops_per_sec,
                            latency_ms
                        );
                        add_result("eviction", result);
                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark UTF-8 handling vs ASCII with multiple endpoint tenants
fn bench_utf8_vs_ascii(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding");

    let tree_ascii = Arc::new(Tree::new());
    let tree_utf8 = Arc::new(Tree::new());

    // Pre-populate with data distributed across endpoints
    let ascii_strings: Vec<String> = (0..1000).map(|_| random_ascii_string(50)).collect();
    let utf8_strings: Vec<String> = (0..1000).map(|i| format!("你好世界_{}", i)).collect();

    for (i, s) in ascii_strings.iter().enumerate() {
        let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
        tree_ascii.insert(s, tenant);
    }
    for (i, s) in utf8_strings.iter().enumerate() {
        let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
        tree_utf8.insert(s, tenant);
    }

    let printed_ascii = Arc::new(AtomicBool::new(false));
    let tree_ascii_clone = tree_ascii.clone();
    let ascii_strings_clone = ascii_strings.clone();
    group.bench_function("ascii_match", |b| {
        let mut idx = 0;
        let printed = printed_ascii.clone();
        let tree = tree_ascii_clone.clone();
        let strings = ascii_strings_clone.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let result = tree.prefix_match(black_box(&strings[idx % strings.len()]));
                black_box(result);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let result = format!(
                    "{:<20} | {:>12.0} | {:>12}",
                    "ASCII", ops_per_sec, "baseline"
                );
                add_result("encoding", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    let printed_utf8 = Arc::new(AtomicBool::new(false));
    let tree_utf8_clone = tree_utf8.clone();
    let utf8_strings_clone = utf8_strings.clone();
    group.bench_function("utf8_match", |b| {
        let mut idx = 0;
        let printed = printed_utf8.clone();
        let tree = tree_utf8_clone.clone();
        let strings = utf8_strings_clone.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let result = tree.prefix_match(black_box(&strings[idx % strings.len()]));
                black_box(result);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let result = format!("{:<20} | {:>12.0} | {:>12}", "UTF-8", ops_per_sec, "N/A");
                add_result("encoding", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

/// Benchmark multi-tenant scenarios with 10 HTTP/gRPC endpoint tenants
fn bench_multi_tenant(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_tenant");

    let tree = Arc::new(Tree::new());

    // Setup: 10 endpoint tenants with overlapping data patterns
    let prefixes = ["prompt:", "completion:", "context:", "system:", "user:"];

    for tenant in &ENDPOINT_TENANTS {
        for prefix in &prefixes {
            for i in 0..200 {
                tree.insert(&format!("{}data_{}", prefix, i), tenant);
            }
        }
    }

    let printed_shared = Arc::new(AtomicBool::new(false));
    let tree_clone = tree.clone();
    group.bench_function("shared_prefix_lookup_10_tenants", |b| {
        let queries: Vec<String> = prefixes
            .iter()
            .flat_map(|p| (0..50).map(move |i| format!("{}data_{}", p, i)))
            .collect();
        let mut idx = 0;
        let printed = printed_shared.clone();
        let tree = tree_clone.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let result = tree.prefix_match(black_box(&queries[idx % queries.len()]));
                black_box(result);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let result = format!(
                    "{:<30} | {:>10} | {:>12.0}",
                    "shared_prefix_lookup", 10, ops_per_sec
                );
                add_result("multi_tenant", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    let printed_specific = Arc::new(AtomicBool::new(false));
    let tree_clone = tree.clone();
    group.bench_function("tenant_specific_match_10_tenants", |b| {
        let queries: Vec<(String, &str)> = ENDPOINT_TENANTS
            .iter()
            .flat_map(|&t| (0..20).map(move |i| (format!("prompt:data_{}", i), t)))
            .collect();
        let mut idx = 0;
        let printed = printed_specific.clone();
        let tree = tree_clone.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let (query, tenant) = &queries[idx % queries.len()];
                let result = tree.prefix_match_tenant(black_box(query), black_box(tenant));
                black_box(result);
                idx += 1;
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let result = format!(
                    "{:<30} | {:>10} | {:>12.0}",
                    "tenant_specific_match", 10, ops_per_sec
                );
                add_result("multi_tenant", result);
                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Benchmark tenant removal (simulates worker going offline)
    let printed_removal = Arc::new(AtomicBool::new(false));
    group.bench_function("tenant_removal", |b| {
        let printed = printed_removal.clone();

        b.iter_custom(|iters| {
            let mut total_duration = std::time::Duration::ZERO;
            for _ in 0..iters {
                // Setup: create tree with all endpoints
                let tree = Tree::new();
                for tenant in &ENDPOINT_TENANTS {
                    for prefix in &prefixes {
                        for i in 0..100 {
                            tree.insert(&format!("{}data_{}", prefix, i), tenant);
                        }
                    }
                }

                let start = Instant::now();
                tree.remove_tenant(ENDPOINT_TENANTS[0]);
                total_duration += start.elapsed();
            }

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / total_duration.as_secs_f64();
                let latency_ms = total_duration.as_millis() as f64 / iters as f64;
                let result = format!(
                    "{:<30} | {:>10} | {:>12.0} | {:>10.2}ms",
                    "tenant_removal", 10, ops_per_sec, latency_ms
                );
                add_result("multi_tenant", result);
                printed.store(true, Ordering::Relaxed);
            }

            total_duration
        });
    });

    group.finish();
}

/// Generate worker endpoint URLs for scaling tests
fn generate_worker_endpoints(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            if i % 4 == 0 {
                format!("grpc://worker-{}.sglang.svc.cluster.local:50051", i)
            } else {
                format!("http://worker-{}.sglang.svc.cluster.local:8000", i)
            }
        })
        .collect()
}

/// Benchmark summary for CI - runs a subset of representative benchmarks
fn bench_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_summary");

    // Reduce warmup and measurement time for faster CI runs
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));

    // Configuration constants
    const TREE_SIZE: usize = 10_000; // Realistic cache size
    const INSERT_POOL_SIZE: usize = 10_000; // Unique requests for insert tests
    const NUM_THREADS: usize = 64; // Match GPU runner's 64 CPU cores
    const OPS_PER_THREAD: usize = 200;

    // Worker scaling configurations to test
    // Full range to demonstrate scaling behavior on GPU runner
    const WORKER_COUNTS: [usize; 4] = [10, 50, 100, 500];

    // Pre-generate requests for tree population and queries
    let requests = generate_realistic_requests(TREE_SIZE);
    let avg_len: usize = requests.iter().map(|r| r.len()).sum::<usize>() / requests.len();

    // Report test configuration upfront
    add_result("config", "Test Configuration:".to_string());
    add_result(
        "config",
        format!(
            "  Request size: ~{} chars (~{} tokens)",
            avg_len,
            avg_len / 4
        ),
    );
    add_result(
        "config",
        format!("  Tree size: {} entries (for MATCH tests)", TREE_SIZE),
    );
    add_result(
        "config",
        format!("  Insert pool: {} unique requests", INSERT_POOL_SIZE),
    );
    add_result(
        "config",
        format!(
            "  Concurrency: {} threads x {} ops/thread",
            NUM_THREADS, OPS_PER_THREAD
        ),
    );
    add_result(
        "config",
        format!("  Worker counts tested: {:?}", WORKER_COUNTS),
    );

    // Test INSERT performance at different worker scales
    // Use large pool of unique requests to avoid measuring cache-hit behavior
    let insert_requests = generate_realistic_requests(INSERT_POOL_SIZE);

    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("insert_{}w", num_workers);
        let insert_requests = insert_requests.clone();

        group.bench_function(&bench_name, |b| {
            let workers = workers.clone();
            let printed = printed.clone();
            let insert_requests = insert_requests.clone();

            b.iter_custom(|iters| {
                // Fresh tree for each measurement to test pure insert performance
                let tree = Tree::new();

                let start = Instant::now();
                for i in 0..iters {
                    let tenant = &workers[i as usize % workers.len()];
                    // Use pre-generated unique requests from large pool
                    let text = &insert_requests[i as usize % insert_requests.len()];
                    tree.insert(black_box(text), tenant);
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * avg_len as f64) / 1_000_000.0;
                    add_result(
                        "summary",
                        format!(
                            "INSERT {:>3} workers: {:>8.0} ops/sec | {:>5.1} µs/op | {:>6.1} MB/s | ~{} chars",
                            num_workers, ops_per_sec, latency_us, throughput_mb, avg_len
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    // Test MATCH performance at different worker scales
    // Tree is pre-populated with TREE_SIZE entries distributed across workers
    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);
        let tree = Arc::new(Tree::new());

        // Populate tree with requests distributed across workers
        for (i, req) in requests.iter().enumerate() {
            let tenant = &workers[i % workers.len()];
            tree.insert(req, tenant);
        }

        let printed = Arc::new(AtomicBool::new(false));
        let requests_clone = requests.clone();
        let bench_name = format!("match_{}w", num_workers);

        group.bench_function(&bench_name, |b| {
            let tree = tree.clone();
            let requests = requests_clone.clone();
            let mut idx = 0;
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = tree.prefix_match(black_box(&requests[idx % requests.len()]));
                    black_box(result);
                    idx += 1;
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * avg_len as f64) / 1_000_000.0;
                    add_result(
                        "summary",
                        format!(
                            "MATCH  {:>3} workers: {:>8.0} ops/sec | {:>5.1} µs/op | {:>6.1} MB/s | {}k tree entries",
                            num_workers, ops_per_sec, latency_us, throughput_mb, TREE_SIZE / 1000
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    // Concurrent benchmark with scaling workers
    // Reduced sample size and measurement time for CI
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));
    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("concurrent_{}w", num_workers);

        group.bench_function(&bench_name, |b| {
            let printed = printed.clone();
            let workers = workers.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let tree = Arc::new(Tree::new());
                    let workers_ref = &workers;
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let worker = workers_ref[t % workers_ref.len()].clone();
                            thread::spawn(move || {
                                for i in 0..OPS_PER_THREAD {
                                    let text = format!(
                                        "{}thread{}_request{}",
                                        CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()],
                                        t,
                                        i
                                    );
                                    if i % 3 == 0 {
                                        tree.prefix_match(&text);
                                    } else {
                                        tree.insert(&text, &worker);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let total_ops = iters * NUM_THREADS as u64 * OPS_PER_THREAD as u64;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    let per_thread_ops = ops_per_sec / NUM_THREADS as f64;
                    add_result(
                        "summary",
                        format!(
                            "CONCURRENT {:>3} workers: {:>6.0} ops/sec | {} threads | {:.0} ops/thread",
                            num_workers, ops_per_sec, NUM_THREADS, per_thread_ops
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    group.finish();
}

/// Print final summary table
fn print_summary() {
    // Ensure output is flushed immediately
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    eprintln!("\n{}", "=".repeat(100));
    eprintln!("RADIX TREE BENCHMARK SUMMARY (Cache-Aware Routing)");
    eprintln!("{}", "=".repeat(100));

    let results = BENCHMARK_RESULTS.lock().unwrap();
    eprintln!("Total benchmark results collected: {}", results.len());

    let mut current_category = String::new();
    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");

        if category != current_category {
            current_category = category.clone();

            // Print section header based on category
            eprintln!("\n{}", "-".repeat(100));
            match category.as_str() {
                "insert" => {
                    eprintln!("INSERT THROUGHPUT (10 endpoint tenants)");
                    eprintln!(
                        "{:<25} | {:>8} | {:>12} | {:>10}",
                        "Test Case", "Size", "Ops/sec", "Tenants"
                    );
                }
                "prefix_match" => {
                    eprintln!("PREFIX MATCH LATENCY");
                    eprintln!(
                        "{:<20} | {:>12} | {:>12}",
                        "Match Type", "Ops/sec", "Latency(ns)"
                    );
                }
                "concurrent" => {
                    eprintln!("CONCURRENT OPERATIONS (mixed read/write)");
                    eprintln!(
                        "{:<25} | {:>8} | {:>12} | {:>12}",
                        "Configuration", "Threads", "Total Ops/s", "Per-Thread"
                    );
                }
                "eviction" => {
                    eprintln!("EVICTION PERFORMANCE");
                    eprintln!(
                        "{:<25} | {:>8} | {:>12} | {:>12}",
                        "Configuration", "Size", "Ops/sec", "Latency(ms)"
                    );
                }
                "encoding" => {
                    eprintln!("ENCODING (ASCII vs UTF-8)");
                    eprintln!(
                        "{:<20} | {:>12} | {:>12}",
                        "Encoding", "Ops/sec", "Comparison"
                    );
                }
                "multi_tenant" => {
                    eprintln!("MULTI-TENANT SCENARIOS (10 HTTP/gRPC endpoints)");
                    eprintln!(
                        "{:<30} | {:>10} | {:>12}",
                        "Operation", "Tenants", "Ops/sec"
                    );
                }
                "config" => {
                    eprintln!("TEST CONFIGURATION");
                }
                "summary" => {
                    eprintln!("BENCHMARK RESULTS");
                }
                _ => {}
            }
            eprintln!("{}", "-".repeat(100));
        }

        eprintln!("{}", value);
    }

    eprintln!("\n{}", "=".repeat(100));
    eprintln!("Endpoint tenants used:");
    for (i, tenant) in ENDPOINT_TENANTS.iter().enumerate() {
        eprintln!("  [{}] {}", i, tenant);
    }
    eprintln!("{}", "=".repeat(100));
}

fn run_benchmarks(c: &mut Criterion) {
    bench_insert_throughput(c);
    bench_prefix_match_latency(c);
    bench_concurrent_operations(c);
    bench_eviction(c);
    bench_utf8_vs_ascii(c);
    bench_multi_tenant(c);
    bench_summary(c);

    // Print summary at the end
    print_summary();
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);
