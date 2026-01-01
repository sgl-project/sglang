//! Benchmarks for the radix tree implementations used in cache-aware routing.
//!
//! This benchmark compares both implementations:
//! - StringTree: Character-based tree for HTTP router (text input)
//! - TokenTree: Token-based tree for gRPC router (pre-tokenized input)
//!
//! Run with: cargo bench --bench radix_tree_benchmark
//!
//! For quick validation: cargo bench --bench radix_tree_benchmark -- benchmark_summary --exact

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{
    distr::{Alphanumeric, SampleString},
    rng as thread_rng, Rng,
};
// Import both old and new tree implementations
use sgl_model_gateway::policies::tree::Tree as OldTree;
use sgl_model_gateway::radix_tree::{StringTree, TokenTree};

// Global results storage for summary
lazy_static::lazy_static! {
    static ref BENCHMARK_RESULTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
}

fn add_result(category: &str, result: String) {
    let mut results = BENCHMARK_RESULTS.lock().unwrap();
    let index = results.len();
    let key = format!("{:03}_{}", index, category);
    eprintln!("[BENCH_RESULT] {} | {}", category, result);
    results.insert(key, result);
}

/// Common conversation prefixes that create shared tree paths
const CONVERSATION_PREFIXES: [&str; 6] = [
    "<|system|>\nYou are a helpful assistant.\n<|user|>\n",
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
    "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n",
    "Human: ",
    "User: ",
    "### Instruction:\n",
];

/// Token ID type
type TokenId = u32;

/// Generate random ASCII strings of given length
fn random_ascii_string(len: usize) -> String {
    Alphanumeric.sample_string(&mut thread_rng(), len)
}

/// Generate fixed-size strings for benchmarks
fn generate_fixed_char_strings(count: usize, char_len: usize) -> Vec<String> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let prefix_idx = rng.random_range(0..CONVERSATION_PREFIXES.len());
            let prefix = CONVERSATION_PREFIXES[prefix_idx];
            let remaining = char_len.saturating_sub(prefix.len());
            format!("{}{}", prefix, random_ascii_string(remaining))
        })
        .collect()
}

/// Generate fixed-size token sequences for size-specific benchmarks
fn generate_fixed_token_sequences(count: usize, token_len: usize) -> Vec<Vec<TokenId>> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| (0..token_len).map(|_| rng.random_range(0..50000)).collect())
        .collect()
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

/// Main benchmark comparing StringTree vs TokenTree
fn bench_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_summary");

    // Reduce warmup and measurement time for faster runs
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));

    // Configuration constants
    const TREE_SIZE: usize = 10_000;
    const INSERT_POOL_SIZE: usize = 10_000;
    const NUM_THREADS: usize = 64;
    const OPS_PER_THREAD: usize = 200;

    // Worker counts and sizes to test
    const WORKER_COUNTS: [usize; 4] = [10, 50, 100, 500];
    const TOKEN_SIZES: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
    const CHAR_SIZES: [usize; 5] = [4096, 8192, 16384, 32768, 65536]; // ~4x token sizes

    // Report test configuration
    add_result("config", "Test Configuration:".to_string());
    add_result("config", format!("  Token sizes: {:?}", TOKEN_SIZES));
    add_result("config", format!("  Char sizes:  {:?}", CHAR_SIZES));
    add_result("config", format!("  Worker counts: {:?}", WORKER_COUNTS));
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

    // ========================================================================
    // OLD StringTree Benchmark (same format as NEW)
    // ========================================================================
    add_result("old_string", "".to_string());

    // Test OLD tree with the middle size (16k chars)
    let old_char_size = 16384;
    let old_requests = generate_fixed_char_strings(TREE_SIZE, old_char_size);

    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        // Pre-populate OLD tree
        let old_tree = Arc::new(OldTree::new());
        for (i, req) in old_requests.iter().enumerate() {
            let tenant = &workers[i % workers.len()];
            old_tree.insert(req, tenant);
        }

        // OLD tree INSERT
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("old_string_insert_{}w", num_workers);
        let insert_requests = old_requests.clone();
        let workers_clone = workers.clone();

        group.bench_function(&bench_name, |b| {
            let workers = workers_clone.clone();
            let printed = printed.clone();
            let insert_requests = insert_requests.clone();

            b.iter_custom(|iters| {
                let tree = OldTree::new();
                let start = Instant::now();
                for i in 0..iters {
                    let tenant = &workers[i as usize % workers.len()];
                    let text = &insert_requests[i as usize % insert_requests.len()];
                    tree.insert(black_box(text), tenant);
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * old_char_size as f64) / 1_000_000.0;
                    add_result(
                        "old_string",
                        format!(
                            "INSERT {:>3}w {:>5} chars: {:>8.0} ops/sec | {:>6.1} µs/op | {:>6.1} MB/s",
                            num_workers, old_char_size, ops_per_sec, latency_us, throughput_mb
                        ),
                    );
                }

                duration
            });
        });

        // OLD tree MATCH
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("old_string_match_{}w", num_workers);
        let tree_clone = old_tree.clone();
        let requests_clone = old_requests.clone();

        group.bench_function(&bench_name, |b| {
            let tree = tree_clone.clone();
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

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * old_char_size as f64) / 1_000_000.0;
                    add_result(
                        "old_string",
                        format!(
                            "MATCH  {:>3}w {:>5} chars: {:>8.0} ops/sec | {:>6.1} µs/op | {:>6.1} MB/s",
                            num_workers, old_char_size, ops_per_sec, latency_us, throughput_mb
                        ),
                    );
                }

                duration
            });
        });
    }

    // ========================================================================
    // NEW StringTree Benchmark (same format as OLD)
    // ========================================================================
    add_result("new_string", "".to_string());

    // Test NEW tree with the same middle size (16k chars)
    let new_char_size = 16384;
    let new_requests = generate_fixed_char_strings(TREE_SIZE, new_char_size);

    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        // Pre-populate NEW tree
        let new_tree = Arc::new(StringTree::new());
        for (i, req) in new_requests.iter().enumerate() {
            let tenant = &workers[i % workers.len()];
            new_tree.insert_text(req, tenant);
        }

        // NEW tree INSERT
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("new_string_insert_{}w", num_workers);
        let insert_requests = new_requests.clone();
        let workers_clone = workers.clone();

        group.bench_function(&bench_name, |b| {
            let workers = workers_clone.clone();
            let printed = printed.clone();
            let insert_requests = insert_requests.clone();

            b.iter_custom(|iters| {
                let tree = StringTree::new();
                let start = Instant::now();
                for i in 0..iters {
                    let tenant = &workers[i as usize % workers.len()];
                    let text = &insert_requests[i as usize % insert_requests.len()];
                    tree.insert_text(black_box(text), tenant);
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * new_char_size as f64) / 1_000_000.0;
                    add_result(
                        "new_string",
                        format!(
                            "INSERT {:>3}w {:>5} chars: {:>8.0} ops/sec | {:>6.1} µs/op | {:>6.1} MB/s",
                            num_workers, new_char_size, ops_per_sec, latency_us, throughput_mb
                        ),
                    );
                }

                duration
            });
        });

        // NEW tree MATCH
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("new_string_match_{}w", num_workers);
        let tree_clone = new_tree.clone();
        let requests_clone = new_requests.clone();

        group.bench_function(&bench_name, |b| {
            let tree = tree_clone.clone();
            let requests = requests_clone.clone();
            let mut idx = 0;
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = tree.prefix_match_legacy(black_box(&requests[idx % requests.len()]));
                    black_box(result);
                    idx += 1;
                }
                let duration = start.elapsed();

                if !printed.swap(true, Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * new_char_size as f64) / 1_000_000.0;
                    add_result(
                        "new_string",
                        format!(
                            "MATCH  {:>3}w {:>5} chars: {:>8.0} ops/sec | {:>6.1} µs/op | {:>6.1} MB/s",
                            num_workers, new_char_size, ops_per_sec, latency_us, throughput_mb
                        ),
                    );
                }

                duration
            });
        });
    }

    // ========================================================================
    // TokenTree: Per-worker tables with different token sizes
    // ========================================================================
    for &num_workers in &WORKER_COUNTS {
        let category = format!("token_{}w", num_workers);
        add_result(&category, "".to_string());

        let workers = generate_worker_endpoints(num_workers);

        for &token_size in &TOKEN_SIZES {
            let fixed_sequences = generate_fixed_token_sequences(INSERT_POOL_SIZE, token_size);

            // Pre-populate tree for MATCH
            let token_tree = Arc::new(TokenTree::new());
            for (i, seq) in fixed_sequences.iter().take(TREE_SIZE).enumerate() {
                let tenant = &workers[i % workers.len()];
                token_tree.insert_tokens(seq, tenant);
            }

            // TokenTree INSERT
            let printed = Arc::new(AtomicBool::new(false));
            let bench_name = format!("token_insert_{}w_{}tok", num_workers, token_size);
            let workers_clone = workers.clone();
            let seqs_clone = fixed_sequences.clone();

            group.bench_function(&bench_name, |b| {
                let workers = workers_clone.clone();
                let seqs = seqs_clone.clone();
                let printed = printed.clone();

                b.iter_custom(|iters| {
                    let tree = TokenTree::new();
                    let start = Instant::now();
                    for i in 0..iters {
                        let tenant = &workers[i as usize % workers.len()];
                        let tokens = &seqs[i as usize % seqs.len()];
                        tree.insert_tokens(black_box(tokens), tenant);
                    }
                    let duration = start.elapsed();

                    if !printed.swap(true, Ordering::Relaxed) {
                        let ops_per_sec = iters as f64 / duration.as_secs_f64();
                        let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                        let throughput_mtok = (ops_per_sec * token_size as f64) / 1_000_000.0;
                        add_result(
                            &category,
                            format!(
                                "INSERT {:>5} tokens: {:>8.0} ops/sec | {:>6.1} µs/op | {:>6.1} Mtok/s",
                                token_size, ops_per_sec, latency_us, throughput_mtok
                            ),
                        );
                    }

                    duration
                });
            });

            // TokenTree MATCH
            let printed = Arc::new(AtomicBool::new(false));
            let bench_name = format!("token_match_{}w_{}tok", num_workers, token_size);
            let tree_clone = token_tree.clone();
            let seqs_clone = fixed_sequences.clone();

            group.bench_function(&bench_name, |b| {
                let tree = tree_clone.clone();
                let seqs = seqs_clone.clone();
                let mut idx = 0;
                let printed = printed.clone();

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let result = tree.prefix_match_legacy(black_box(&seqs[idx % seqs.len()]));
                        black_box(result);
                        idx += 1;
                    }
                    let duration = start.elapsed();

                    if !printed.swap(true, Ordering::Relaxed) {
                        let ops_per_sec = iters as f64 / duration.as_secs_f64();
                        let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                        let throughput_mtok = (ops_per_sec * token_size as f64) / 1_000_000.0;
                        add_result(
                            &category,
                            format!(
                                "MATCH  {:>5} tokens: {:>8.0} ops/sec | {:>6.1} µs/op | {:>6.1} Mtok/s",
                                token_size, ops_per_sec, latency_us, throughput_mtok
                            ),
                        );
                    }

                    duration
                });
            });
        }
    }

    // ========================================================================
    // StringTree: Per-worker tables with different char sizes
    // ========================================================================
    for &num_workers in &WORKER_COUNTS {
        let category = format!("string_{}w", num_workers);
        add_result(&category, "".to_string());

        let workers = generate_worker_endpoints(num_workers);

        for &char_size in &CHAR_SIZES {
            let fixed_strings = generate_fixed_char_strings(INSERT_POOL_SIZE, char_size);

            // Pre-populate tree for MATCH
            let string_tree = Arc::new(StringTree::new());
            for (i, s) in fixed_strings.iter().take(TREE_SIZE).enumerate() {
                let tenant = &workers[i % workers.len()];
                string_tree.insert_text(s, tenant);
            }

            // StringTree INSERT
            let printed = Arc::new(AtomicBool::new(false));
            let bench_name = format!("string_insert_{}w_{}c", num_workers, char_size);
            let workers_clone = workers.clone();
            let strings_clone = fixed_strings.clone();

            group.bench_function(&bench_name, |b| {
                let workers = workers_clone.clone();
                let strings = strings_clone.clone();
                let printed = printed.clone();

                b.iter_custom(|iters| {
                    let tree = StringTree::new();
                    let start = Instant::now();
                    for i in 0..iters {
                        let tenant = &workers[i as usize % workers.len()];
                        let text = &strings[i as usize % strings.len()];
                        tree.insert_text(black_box(text), tenant);
                    }
                    let duration = start.elapsed();

                    if !printed.swap(true, Ordering::Relaxed) {
                        let ops_per_sec = iters as f64 / duration.as_secs_f64();
                        let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                        let throughput_mb = (ops_per_sec * char_size as f64) / 1_000_000.0;
                        add_result(
                            &category,
                            format!(
                                "INSERT {:>5} chars: {:>8.0} ops/sec | {:>6.1} µs/op | {:>6.1} MB/s",
                                char_size, ops_per_sec, latency_us, throughput_mb
                            ),
                        );
                    }

                    duration
                });
            });

            // StringTree MATCH
            let printed = Arc::new(AtomicBool::new(false));
            let bench_name = format!("string_match_{}w_{}c", num_workers, char_size);
            let tree_clone = string_tree.clone();
            let strings_clone = fixed_strings.clone();

            group.bench_function(&bench_name, |b| {
                let tree = tree_clone.clone();
                let strings = strings_clone.clone();
                let mut idx = 0;
                let printed = printed.clone();

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let result = tree.prefix_match_legacy(black_box(&strings[idx % strings.len()]));
                        black_box(result);
                        idx += 1;
                    }
                    let duration = start.elapsed();

                    if !printed.swap(true, Ordering::Relaxed) {
                        let ops_per_sec = iters as f64 / duration.as_secs_f64();
                        let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                        let throughput_mb = (ops_per_sec * char_size as f64) / 1_000_000.0;
                        add_result(
                            &category,
                            format!(
                                "MATCH  {:>5} chars: {:>8.0} ops/sec | {:>6.1} µs/op | {:>6.1} MB/s",
                                char_size, ops_per_sec, latency_us, throughput_mb
                            ),
                        );
                    }

                    duration
                });
            });
        }
    }

    // ========================================================================
    // CONCURRENT benchmarks
    // ========================================================================
    group.sample_size(10);

    add_result("concurrent", "".to_string());

    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        // TokenTree CONCURRENT
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("token_concurrent_{}w", num_workers);
        let workers_clone = workers.clone();

        group.bench_function(&bench_name, |b| {
            let printed = printed.clone();
            let workers = workers_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let tree = Arc::new(TokenTree::new());
                    let workers_ref = &workers;
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let worker = workers_ref[t % workers_ref.len()].clone();
                            thread::spawn(move || {
                                for i in 0..OPS_PER_THREAD {
                                    // Generate deterministic token sequence
                                    let tokens: Vec<TokenId> = (0..1024)
                                        .map(|j| (t * 10000 + i * 100 + j) as u32)
                                        .collect();
                                    if i % 3 == 0 {
                                        tree.prefix_match_legacy(&tokens);
                                    } else {
                                        tree.insert_tokens(&tokens, &worker);
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

                if !printed.swap(true, Ordering::Relaxed) {
                    let total_ops = iters * NUM_THREADS as u64 * OPS_PER_THREAD as u64;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    let per_thread_ops = ops_per_sec / NUM_THREADS as f64;
                    add_result(
                        "concurrent",
                        format!(
                            "TokenTree  {:>3}w: {:>7.0} ops/sec | {} threads | {:.0} ops/thread",
                            num_workers, ops_per_sec, NUM_THREADS, per_thread_ops
                        ),
                    );
                }

                duration
            });
        });

        // StringTree CONCURRENT
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("string_concurrent_{}w", num_workers);
        let workers_clone = workers.clone();

        group.bench_function(&bench_name, |b| {
            let printed = printed.clone();
            let workers = workers_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let tree = Arc::new(StringTree::new());
                    let workers_ref = &workers;
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let worker = workers_ref[t % workers_ref.len()].clone();
                            thread::spawn(move || {
                                for i in 0..OPS_PER_THREAD {
                                    let text = format!(
                                        "{}thread{}_request{}_padding{}",
                                        CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()],
                                        t,
                                        i,
                                        "x".repeat(1000)
                                    );
                                    if i % 3 == 0 {
                                        tree.prefix_match_legacy(&text);
                                    } else {
                                        tree.insert_text(&text, &worker);
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

                if !printed.swap(true, Ordering::Relaxed) {
                    let total_ops = iters * NUM_THREADS as u64 * OPS_PER_THREAD as u64;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    let per_thread_ops = ops_per_sec / NUM_THREADS as f64;
                    add_result(
                        "concurrent",
                        format!(
                            "StringTree {:>3}w: {:>7.0} ops/sec | {} threads | {:.0} ops/thread",
                            num_workers, ops_per_sec, NUM_THREADS, per_thread_ops
                        ),
                    );
                }

                duration
            });
        });
    }

    group.finish();
}

/// Print final summary table
fn print_summary() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    eprintln!("\n{}", "=".repeat(100));
    eprintln!("RADIX TREE BENCHMARK SUMMARY");
    eprintln!("{}", "=".repeat(100));

    let results = BENCHMARK_RESULTS.lock().unwrap();

    // Collect results by category
    let mut config_results = Vec::new();
    let mut old_string_results = Vec::new();
    let mut new_string_results = Vec::new();
    let mut token_10w_results = Vec::new();
    let mut token_50w_results = Vec::new();
    let mut token_100w_results = Vec::new();
    let mut token_500w_results = Vec::new();
    let mut string_10w_results = Vec::new();
    let mut string_50w_results = Vec::new();
    let mut string_100w_results = Vec::new();
    let mut string_500w_results = Vec::new();
    let mut concurrent_results = Vec::new();

    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");
        match category.as_str() {
            "config" => config_results.push(value.clone()),
            "old_string" => old_string_results.push(value.clone()),
            "new_string" => new_string_results.push(value.clone()),
            "token_10w" => token_10w_results.push(value.clone()),
            "token_50w" => token_50w_results.push(value.clone()),
            "token_100w" => token_100w_results.push(value.clone()),
            "token_500w" => token_500w_results.push(value.clone()),
            "string_10w" => string_10w_results.push(value.clone()),
            "string_50w" => string_50w_results.push(value.clone()),
            "string_100w" => string_100w_results.push(value.clone()),
            "string_500w" => string_500w_results.push(value.clone()),
            "concurrent" => concurrent_results.push(value.clone()),
            _ => {}
        }
    }

    // Print config
    eprintln!("\n{}", "-".repeat(100));
    eprintln!("TEST CONFIGURATION");
    eprintln!("{}", "-".repeat(100));
    for v in &config_results {
        eprintln!("{}", v);
    }

    // Print OLD StringTree
    eprintln!("\n{}", "-".repeat(100));
    eprintln!("OLD STRINGTREE (policies::tree) - 16k chars");
    eprintln!("{}", "-".repeat(100));
    for v in &old_string_results {
        if !v.is_empty() {
            eprintln!("{}", v);
        }
    }

    // Print NEW StringTree
    eprintln!("\n{}", "-".repeat(100));
    eprintln!("NEW STRINGTREE (radix_tree::StringTree) - 16k chars");
    eprintln!("{}", "-".repeat(100));
    for v in &new_string_results {
        if !v.is_empty() {
            eprintln!("{}", v);
        }
    }

    // Print TokenTree tables by worker count
    let token_tables = [
        ("TOKENTREE - 10 WORKERS", &token_10w_results),
        ("TOKENTREE - 50 WORKERS", &token_50w_results),
        ("TOKENTREE - 100 WORKERS", &token_100w_results),
        ("TOKENTREE - 500 WORKERS", &token_500w_results),
    ];

    for (title, results) in token_tables {
        eprintln!("\n{}", "-".repeat(100));
        eprintln!("{}", title);
        eprintln!("{}", "-".repeat(100));
        for v in results {
            if !v.is_empty() {
                eprintln!("{}", v);
            }
        }
    }

    // Print StringTree tables by worker count
    let string_tables = [
        ("STRINGTREE - 10 WORKERS", &string_10w_results),
        ("STRINGTREE - 50 WORKERS", &string_50w_results),
        ("STRINGTREE - 100 WORKERS", &string_100w_results),
        ("STRINGTREE - 500 WORKERS", &string_500w_results),
    ];

    for (title, results) in string_tables {
        eprintln!("\n{}", "-".repeat(100));
        eprintln!("{}", title);
        eprintln!("{}", "-".repeat(100));
        for v in results {
            if !v.is_empty() {
                eprintln!("{}", v);
            }
        }
    }

    // Print concurrent results
    eprintln!("\n{}", "-".repeat(100));
    eprintln!("CONCURRENT MIXED OPERATIONS (64 threads x 200 ops)");
    eprintln!("{}", "-".repeat(100));
    for v in &concurrent_results {
        if !v.is_empty() {
            eprintln!("{}", v);
        }
    }

    eprintln!("\n{}", "=".repeat(100));
}

fn run_benchmarks(c: &mut Criterion) {
    bench_summary(c);
    print_summary();
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);
