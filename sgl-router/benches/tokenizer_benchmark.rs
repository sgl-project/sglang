//! Comprehensive tokenizer benchmark with clean summary output
//! Each test adds a row to the final summary table

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use sglang_router_rs::tokenizer::{
    huggingface::HuggingFaceTokenizer, sequence::Sequence, stop::*, stream::DecodeStream, traits::*,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

// Include the common test utilities
#[path = "../tests/common/mod.rs"]
mod common;
use common::ensure_tokenizer_cached;

// Cache the tokenizer path for the entire benchmark run
static TOKENIZER_PATH: OnceLock<PathBuf> = OnceLock::new();

fn get_tokenizer_path() -> &'static PathBuf {
    TOKENIZER_PATH.get_or_init(ensure_tokenizer_cached)
}

// Production target: 100k tokens per second
const TARGET_TOKENS_PER_SECOND: u64 = 100_000;

// Typical prompt sizes
const SHORT_PROMPT: &str = "What is the capital of France?";
const MEDIUM_PROMPT: &str = "Write a detailed explanation of quantum computing, including its principles, current applications, and future potential. Be sure to cover both the theoretical foundations and practical implementations.";
const LONG_PROMPT: &str = "You are an expert software engineer. Review the following code and provide detailed feedback on performance optimizations, potential bugs, and architectural improvements. Consider scalability, maintainability, and best practices. The code implements a distributed caching system with the following requirements: 1) High availability across multiple regions, 2) Sub-millisecond latency for cache hits, 3) Automatic failover and recovery, 4) Support for both LRU and LFU eviction policies, 5) Real-time monitoring and alerting. Please analyze each component thoroughly and suggest concrete improvements with code examples where appropriate.";

// System prompts can be quite large
fn generate_system_prompt(size: usize) -> String {
    let base = "You are a helpful AI assistant with expertise in ";
    let domains = vec![
        "mathematics",
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "engineering",
        "medicine",
        "law",
        "economics",
        "philosophy",
    ];

    let mut prompt = base.to_string();
    while prompt.len() < size {
        for domain in &domains {
            prompt.push_str(domain);
            prompt.push_str(", ");
            if prompt.len() >= size {
                break;
            }
        }
    }
    prompt
}

// Global results storage
lazy_static::lazy_static! {
    static ref BENCHMARK_RESULTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
}

fn add_result(category: &str, result: String) {
    let mut results = BENCHMARK_RESULTS.lock().unwrap();
    let index = results.len();
    results.insert(format!("{:03}_{}", index, category), result);
}

fn bench_encode_throughput(c: &mut Criterion) {
    let tokenizer_path = get_tokenizer_path();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    // Pre-generate system prompts
    let system_1k = generate_system_prompt(1000);
    let system_4k = generate_system_prompt(4000);
    let system_16k = generate_system_prompt(16000);

    let test_cases = vec![
        ("short_30B", SHORT_PROMPT),
        ("medium_230B", MEDIUM_PROMPT),
        ("long_670B", LONG_PROMPT),
        ("system_1KB", system_1k.as_str()),
        ("system_4KB", system_4k.as_str()),
        ("system_16KB", system_16k.as_str()),
    ];

    let mut group = c.benchmark_group("encode_throughput");

    for (name, prompt) in test_cases {
        let prompt_len = prompt.len();
        let tokenizer_clone = tokenizer.clone();

        // Get token count once
        let encoding = tokenizer.encode(prompt).unwrap();
        let token_count = encoding.token_ids().len();

        // Track if metrics have been printed for this test case
        let printed = Arc::new(AtomicBool::new(false));

        group.throughput(Throughput::Bytes(prompt_len as u64));
        group.bench_function(name, |b| {
            let printed_clone = printed.clone();
            let tokenizer = tokenizer_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(tokenizer.encode(prompt).unwrap());
                }
                let duration = start.elapsed();

                // Store result only once per test case
                if !printed_clone.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let chars_per_sec = (iters as f64 * prompt_len as f64) / duration.as_secs_f64();
                    let tokens_per_sec =
                        (iters as f64 * token_count as f64) / duration.as_secs_f64();

                    let result = format!(
                        "{:<15} | {:>8} | {:>8} | {:>12.0} | {:>12.0} | {:>10.0} | {:>10}",
                        name,
                        prompt_len,
                        token_count,
                        chars_per_sec,
                        tokens_per_sec,
                        ops_per_sec,
                        1
                    );
                    add_result("encode", result);

                    printed_clone.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    group.finish();
}

fn bench_batch_encode(c: &mut Criterion) {
    let tokenizer_path = get_tokenizer_path();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let batch_sizes = vec![1, 8, 16, 32, 64, 128];
    let prompt = MEDIUM_PROMPT;
    let prompt_len = prompt.len();
    let encoding = tokenizer.encode(prompt).unwrap();
    let token_count = encoding.token_ids().len();

    let mut group = c.benchmark_group("batch_encode");

    for batch_size in batch_sizes {
        let prompts: Vec<&str> = vec![prompt; batch_size];
        let printed = Arc::new(AtomicBool::new(false));
        let tokenizer_clone = tokenizer.clone();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let printed_clone = printed.clone();
                let tokenizer = tokenizer_clone.clone();

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        black_box(tokenizer.encode_batch(&prompts).unwrap());
                    }
                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let prompts_per_sec = (iters as f64 * size as f64) / duration.as_secs_f64();
                        let tokens_per_sec = prompts_per_sec * token_count as f64;
                        let chars_per_sec = prompts_per_sec * prompt_len as f64;

                        let result = format!(
                            "{:<15} | {:>8} | {:>8} | {:>12.0} | {:>12.0} | {:>10.0} | {:>10}",
                            format!("batch_{}", size),
                            prompt_len * size,
                            token_count * size,
                            prompts_per_sec,
                            tokens_per_sec,
                            chars_per_sec,
                            1
                        );
                        add_result("batch", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_encode(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let client_counts = vec![1, 4, 8, 16, 32];

    let mut group = c.benchmark_group("concurrent_encode");
    group.measurement_time(Duration::from_secs(2));

    for num_clients in client_counts {
        let printed = Arc::new(AtomicBool::new(false));
        let tokenizer_clone = tokenizer.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_clients),
            &num_clients,
            |b, &clients| {
                let printed_clone = printed.clone();

                b.iter_custom(|_iters| {
                    let tokenizer = tokenizer_clone.clone();
                    let total_operations = Arc::new(AtomicU64::new(0));
                    let total_chars = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..clients)
                        .map(|client_id| {
                            let tokenizer = tokenizer.clone();
                            let total_ops = total_operations.clone();
                            let total_ch = total_chars.clone();

                            thread::spawn(move || {
                                let prompts = [SHORT_PROMPT, MEDIUM_PROMPT, LONG_PROMPT];
                                let prompt = prompts[client_id % prompts.len()];
                                let mut local_ops = 0u64;
                                let mut local_chars = 0u64;

                                while start.elapsed() < Duration::from_millis(500) {
                                    let _ = tokenizer.encode(prompt).unwrap();
                                    local_ops += 1;
                                    local_chars += prompt.len() as u64;
                                }

                                total_ops.fetch_add(local_ops, Ordering::Relaxed);
                                total_ch.fetch_add(local_chars, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total_ops = total_operations.load(Ordering::Relaxed);
                        let total_ch = total_chars.load(Ordering::Relaxed);
                        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                        let chars_per_sec = total_ch as f64 / duration.as_secs_f64();
                        let per_client = ops_per_sec / clients as f64;

                        let result = format!(
                            "{:<15} | {:>10} | {:>12.0} | {:>12.0} | {:>15.0}",
                            format!("{}_clients", clients),
                            total_ops,
                            ops_per_sec,
                            chars_per_sec,
                            per_client
                        );
                        add_result("concurrent", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_decode_performance(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let test_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let encoding = tokenizer.encode(&test_text).unwrap();
    let tokens = encoding.token_ids();
    let num_tokens = tokens.len();

    let mut group = c.benchmark_group("decode_performance");

    // Test direct decode
    let printed_direct = Arc::new(AtomicBool::new(false));
    group.bench_function("direct_decode", |b| {
        let printed = printed_direct.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(tokenizer.decode(tokens, false).unwrap());
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let tokens_per_sec = ops_per_sec * num_tokens as f64;

                let result = format!(
                    "{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "Direct", num_tokens, tokens_per_sec, ops_per_sec, 1
                );
                add_result("decode", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test DecodeStream
    let printed_stream = Arc::new(AtomicBool::new(false));
    group.bench_function("decode_stream", |b| {
        let printed = printed_stream.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
                let mut output = String::new();
                for token in tokens {
                    if let Some(text) = decoder.step(*token).unwrap() {
                        output.push_str(&text);
                    }
                }
                black_box(output);
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let tokens_per_sec = ops_per_sec * num_tokens as f64;

                let result = format!(
                    "{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "DecodeStream", num_tokens, tokens_per_sec, ops_per_sec, 1
                );
                add_result("decode", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test Sequence
    let printed_seq = Arc::new(AtomicBool::new(false));
    group.bench_function("sequence_decode", |b| {
        let printed = printed_seq.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut sequence = Sequence::new(tokenizer.clone());
                let mut output = String::new();
                for token in tokens {
                    let text = sequence.append_token(*token).unwrap();
                    output.push_str(&text);
                }
                black_box(output);
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let tokens_per_sec = ops_per_sec * num_tokens as f64;

                let result = format!(
                    "{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "Sequence", num_tokens, tokens_per_sec, ops_per_sec, 1
                );
                add_result("decode", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_streaming_decode_100k(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let sample_text = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    let encoding = tokenizer.encode(&sample_text).unwrap();
    let all_tokens = encoding.token_ids();

    let mut group = c.benchmark_group("streaming_100k");
    group.measurement_time(Duration::from_secs(1));

    // Test DecodeStream
    let printed_stream = Arc::new(AtomicBool::new(false));
    group.bench_function("decode_stream_100k", |b| {
        let printed = printed_stream.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|_iters| {
            let start = Instant::now();
            let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
            let mut output = String::new();
            let mut tokens_processed = 0u64;

            for token in all_tokens.iter().cycle() {
                if start.elapsed() >= Duration::from_millis(500) {
                    break;
                }

                if let Some(text) = decoder.step(*token).unwrap() {
                    output.push_str(&text);
                }
                tokens_processed += 1;
            }

            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let tokens_per_sec = tokens_processed as f64 / duration.as_secs_f64();
                let status = if tokens_per_sec >= TARGET_TOKENS_PER_SECOND as f64 {
                    "PASS"
                } else {
                    "BELOW"
                };

                let result = format!(
                    "{:<20} | {:>12} | {:>12.0} | {:>12} | {:>10} | {:>12}",
                    "DecodeStream",
                    tokens_processed,
                    tokens_per_sec,
                    TARGET_TOKENS_PER_SECOND,
                    1,
                    status
                );
                add_result("streaming_100k", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test Sequence
    let printed_seq = Arc::new(AtomicBool::new(false));
    group.bench_function("sequence_100k", |b| {
        let printed = printed_seq.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|_iters| {
            let start = Instant::now();
            let mut sequence = Sequence::new(tokenizer.clone());
            let mut output = String::new();
            let mut tokens_processed = 0u64;

            for token in all_tokens.iter().cycle() {
                if start.elapsed() >= Duration::from_millis(500) {
                    break;
                }

                let text = sequence.append_token(*token).unwrap();
                output.push_str(&text);
                tokens_processed += 1;
            }

            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let tokens_per_sec = tokens_processed as f64 / duration.as_secs_f64();
                let status = if tokens_per_sec >= TARGET_TOKENS_PER_SECOND as f64 {
                    "PASS"
                } else {
                    "BELOW"
                };

                let result = format!(
                    "{:<20} | {:>12} | {:>12.0} | {:>12} | {:>10} | {:>12}",
                    "Sequence",
                    tokens_processed,
                    tokens_per_sec,
                    TARGET_TOKENS_PER_SECOND,
                    1,
                    status
                );
                add_result("streaming_100k", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_latency_distribution(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    // Test latency for individual token processing
    let sample_tokens = vec![1, 450, 6635, 3290, 491, 278, 3474, 29892];

    let mut group = c.benchmark_group("latency");

    // Encode latency
    let system_4k = generate_system_prompt(4000);
    let test_cases = vec![
        ("encode_short", SHORT_PROMPT),
        ("encode_medium", MEDIUM_PROMPT),
        ("encode_long", LONG_PROMPT),
        ("encode_4KB", system_4k.as_str()),
    ];

    for (name, prompt) in test_cases {
        let printed = Arc::new(AtomicBool::new(false));
        group.bench_function(name, |b| {
            let printed_clone = printed.clone();
            let tokenizer = tokenizer.clone();

            b.iter_custom(|iters| {
                // Only collect detailed latency on first iteration
                let total_duration = if !printed_clone.load(Ordering::Relaxed) {
                    let mut latencies = Vec::new();

                    // Warm up
                    for _ in 0..100 {
                        let _ = tokenizer.encode(prompt).unwrap();
                    }

                    // Measure for statistics
                    for _ in 0..1000 {
                        let start = Instant::now();
                        let _ = tokenizer.encode(prompt).unwrap();
                        let latency = start.elapsed();
                        latencies.push(latency);
                    }

                    latencies.sort();
                    let p50 = latencies[latencies.len() / 2];
                    let p95 = latencies[latencies.len() * 95 / 100];
                    let p99 = latencies[latencies.len() * 99 / 100];
                    let max = latencies.last().unwrap();

                    let result = format!(
                        "{:<20} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10}",
                        name,
                        p50.as_micros() as f64,
                        p95.as_micros() as f64,
                        p99.as_micros() as f64,
                        max.as_micros() as f64,
                        1000
                    );
                    add_result("latency", result);

                    printed_clone.store(true, Ordering::Relaxed);

                    // Return median for consistency
                    p50 * iters as u32
                } else {
                    // Regular benchmark iterations
                    let start = Instant::now();
                    for _ in 0..iters {
                        black_box(tokenizer.encode(prompt).unwrap());
                    }
                    start.elapsed()
                };

                total_duration
            });
        });
    }

    // Decode token latency
    let printed_decode = Arc::new(AtomicBool::new(false));
    group.bench_function("decode_token", |b| {
        let printed_clone = printed_decode.clone();
        let tokenizer = tokenizer.clone();
        let tokens = sample_tokens.clone();

        b.iter_custom(|iters| {
            let total_duration = if !printed_clone.load(Ordering::Relaxed) {
                let mut latencies = Vec::new();
                let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);

                for token in tokens.iter().cycle().take(1000) {
                    let start = Instant::now();
                    let _ = decoder.step(*token).unwrap();
                    let latency = start.elapsed();
                    latencies.push(latency);
                }

                latencies.sort();
                let p50 = latencies[latencies.len() / 2];
                let p95 = latencies[latencies.len() * 95 / 100];
                let p99 = latencies[latencies.len() * 99 / 100];
                let max = latencies.last().unwrap();

                let result = format!(
                    "{:<20} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10}",
                    "decode_token",
                    p50.as_micros() as f64,
                    p95.as_micros() as f64,
                    p99.as_micros() as f64,
                    max.as_micros() as f64,
                    1000
                );
                add_result("latency", result);

                // Check target latency
                let target_latency = Duration::from_micros(10);
                if p50 > target_latency {
                    let warning = format!(
                        "WARNING: P50 latency exceeds target of {:?} for 100k tokens/sec",
                        target_latency
                    );
                    add_result("latency_warning", warning);
                }

                printed_clone.store(true, Ordering::Relaxed);

                // Return approximate time for consistency
                p50 * iters as u32
            } else {
                // Regular benchmark iterations
                let start = Instant::now();
                for _ in 0..iters {
                    let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
                    for token in tokens.iter().take(10) {
                        black_box(decoder.step(*token).unwrap());
                    }
                }
                start.elapsed()
            };

            total_duration
        });
    });

    group.finish();
}

fn bench_concurrent_streaming(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let num_sequences = 16;
    let tokens_per_sequence = 10_000;

    let sample_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let encoding = tokenizer.encode(&sample_text).unwrap();
    let token_batch: Vec<u32> = encoding.token_ids().to_vec();

    let mut group = c.benchmark_group("concurrent_streaming");
    group.measurement_time(Duration::from_secs(2));

    let printed = Arc::new(AtomicBool::new(false));
    group.bench_function("concurrent_16_sequences", |b| {
        let printed_clone = printed.clone();
        let tokenizer = tokenizer.clone();
        let tokens = token_batch.clone();

        b.iter_custom(|_iters| {
            let total_tokens = Arc::new(AtomicU64::new(0));
            let start = Instant::now();

            let handles: Vec<_> = (0..num_sequences)
                .map(|_seq_id| {
                    let tokenizer = tokenizer.clone();
                    let tokens = tokens.clone();
                    let total_tokens = total_tokens.clone();

                    thread::spawn(move || {
                        let mut decoder = DecodeStream::new(tokenizer, &[], false);
                        let mut output = String::new();
                        let mut local_count = 0u64;

                        for token in tokens.iter().cycle().take(tokens_per_sequence) {
                            if let Some(text) = decoder.step(*token).unwrap() {
                                output.push_str(&text);
                            }
                            local_count += 1;
                        }

                        total_tokens.fetch_add(local_count, Ordering::Relaxed);
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let total = total_tokens.load(Ordering::Relaxed);
                let throughput = total as f64 / duration.as_secs_f64();
                let per_seq = throughput / num_sequences as f64;

                let result = format!(
                    "{:<20} | {:>10} | {:>12.0} | {:>15.0} | {:>15}",
                    format!("{}_sequences", num_sequences),
                    total,
                    throughput,
                    per_seq,
                    num_sequences
                );
                add_result("concurrent_streaming", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_stop_sequences(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let config = StopSequenceConfig::default()
        .with_stop_sequence("</s>")
        .with_stop_sequence("\n\n")
        .with_stop_sequence("###")
        .with_stop_token(2);

    let sample_text = "Hello world! This is a test. ### Stop here. Continue after.".repeat(100);
    let encoding = tokenizer.encode(&sample_text).unwrap();
    let tokens = encoding.token_ids();

    let mut group = c.benchmark_group("stop_sequences");

    // No stops
    let printed_no_stop = Arc::new(AtomicBool::new(false));
    group.bench_function("no_stops", |b| {
        let printed_clone = printed_no_stop.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            let mut total_tokens = 0u64;

            for _ in 0..iters {
                let mut decoder = StopSequenceDecoder::new(
                    tokenizer.clone(),
                    StopSequenceConfig::default(),
                    false,
                );
                for token in tokens {
                    let _ = decoder.process_token(*token).unwrap();
                    total_tokens += 1;
                }
            }

            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let tokens_per_sec = total_tokens as f64 / duration.as_secs_f64();
                let seq_per_sec = iters as f64 / duration.as_secs_f64();

                let result = format!(
                    "{:<20} | {:>10} | {:>12} | {:>12.0} | {:>10.0}",
                    "No stops", iters, total_tokens, tokens_per_sec, seq_per_sec
                );
                add_result("stop_sequences", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // With stops
    let printed_with_stops = Arc::new(AtomicBool::new(false));
    group.bench_function("with_stops", |b| {
        let printed_clone = printed_with_stops.clone();
        let tokenizer = tokenizer.clone();
        let config = config.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            let mut total_tokens = 0u64;
            let mut total_sequences = 0u64;

            for _ in 0..iters {
                let mut decoder =
                    StopSequenceDecoder::new(tokenizer.clone(), config.clone(), false);
                let mut sequence_tokens = 0u64;

                for token in tokens {
                    let result = decoder.process_token(*token).unwrap();
                    sequence_tokens += 1;

                    if matches!(
                        result,
                        SequenceDecoderOutput::Stopped | SequenceDecoderOutput::StoppedWithText(_)
                    ) {
                        break;
                    }
                }

                total_tokens += sequence_tokens;
                total_sequences += 1;
            }

            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let tokens_per_sec = total_tokens as f64 / duration.as_secs_f64();
                let seq_per_sec = total_sequences as f64 / duration.as_secs_f64();

                let result = format!(
                    "{:<20} | {:>10} | {:>12} | {:>12.0} | {:>10.0}",
                    "With stops", total_sequences, total_tokens, tokens_per_sec, seq_per_sec
                );
                add_result("stop_sequences", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_multithreaded_encode(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let thread_counts = vec![1, 2, 4, 8, 16];
    let operations_per_thread = 1000;

    // Test with medium-sized prompt for balanced workload
    let test_prompt = MEDIUM_PROMPT;

    let mut group = c.benchmark_group("multithreaded_encode");
    group.measurement_time(Duration::from_secs(2));

    let mut baseline_throughput = 0.0;

    for num_threads in thread_counts {
        let printed = Arc::new(AtomicBool::new(false));
        let tokenizer_clone = tokenizer.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &threads| {
                let printed_clone = printed.clone();
                let tokenizer = tokenizer_clone.clone();

                b.iter_custom(|_iters| {
                    let total_operations = Arc::new(AtomicU64::new(0));
                    let total_tokens = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tokenizer = tokenizer.clone();
                            let total_ops = total_operations.clone();
                            let total_tok = total_tokens.clone();

                            thread::spawn(move || {
                                for _ in 0..operations_per_thread {
                                    let encoding = tokenizer.encode(test_prompt).unwrap();
                                    total_tok.fetch_add(
                                        encoding.token_ids().len() as u64,
                                        Ordering::Relaxed,
                                    );
                                }
                                total_ops
                                    .fetch_add(operations_per_thread as u64, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total_ops = total_operations.load(Ordering::Relaxed);
                        let total_tok = total_tokens.load(Ordering::Relaxed);
                        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                        let tokens_per_sec = total_tok as f64 / duration.as_secs_f64();

                        if threads == 1 {
                            baseline_throughput = tokens_per_sec;
                        }

                        let efficiency = if threads == 1 {
                            100.0
                        } else {
                            (tokens_per_sec / (baseline_throughput * threads as f64)) * 100.0
                        };

                        let result = format!(
                            "{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10} | {:>11.1}%",
                            format!("encode_{}_threads", threads),
                            total_ops,
                            ops_per_sec,
                            tokens_per_sec,
                            threads,
                            efficiency
                        );
                        add_result("mt_encode", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_multithreaded_decode(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let thread_counts = vec![1, 2, 4, 8, 16];
    let tokens_per_thread = 5000;

    // Generate tokens for decoding
    let test_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let encoding = tokenizer.encode(&test_text).unwrap();
    let test_tokens: Vec<u32> = encoding.token_ids().to_vec();

    let mut group = c.benchmark_group("multithreaded_decode");
    group.measurement_time(Duration::from_secs(2));

    let mut baseline_throughput = 0.0;

    for num_threads in thread_counts {
        let printed = Arc::new(AtomicBool::new(false));
        let tokenizer_clone = tokenizer.clone();
        let tokens = test_tokens.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &threads| {
                let printed_clone = printed.clone();
                let tokenizer = tokenizer_clone.clone();
                let tokens = tokens.clone();

                b.iter_custom(|_iters| {
                    let total_tokens = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tokenizer = tokenizer.clone();
                            let tokens = tokens.clone();
                            let total_tok = total_tokens.clone();

                            thread::spawn(move || {
                                let mut decoder = DecodeStream::new(tokenizer, &[], false);
                                let mut output = String::new();
                                let mut local_tokens = 0u64;

                                for token in tokens.iter().cycle().take(tokens_per_thread) {
                                    if let Some(text) = decoder.step(*token).unwrap() {
                                        output.push_str(&text);
                                    }
                                    local_tokens += 1;
                                }

                                total_tok.fetch_add(local_tokens, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total = total_tokens.load(Ordering::Relaxed);
                        let tokens_per_sec = total as f64 / duration.as_secs_f64();

                        if threads == 1 {
                            baseline_throughput = tokens_per_sec;
                        }

                        let efficiency = if threads == 1 {
                            100.0
                        } else {
                            (tokens_per_sec / (baseline_throughput * threads as f64)) * 100.0
                        };

                        let result = format!(
                            "{:<20} | {:>12} | {:>12.0} | {:>10} | {:>11.1}%",
                            format!("decode_{}_threads", threads),
                            total,
                            tokens_per_sec,
                            threads,
                            efficiency
                        );
                        add_result("mt_decode", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let large_text = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    let encoding = tokenizer.encode(&large_text).unwrap();

    let mut group = c.benchmark_group("memory");

    // Track owned baseline time
    let mut owned_time_ns = 0.0;

    // Owned
    let printed_owned = Arc::new(AtomicBool::new(false));
    group.bench_function("token_ids_owned", |b| {
        let printed_clone = printed_owned.clone();
        let encoding = encoding.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(encoding.token_ids());
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_call = duration.as_nanos() as f64 / iters as f64;
                owned_time_ns = time_per_call;

                let result = format!(
                    "{:<20} | {:>12.0} | {:>11.0}ns | {:>12}",
                    "token_ids(owned)", ops_per_sec, time_per_call, "baseline"
                );
                add_result("memory", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Reference
    let printed_ref = Arc::new(AtomicBool::new(false));

    group.bench_function("token_ids_ref", |b| {
        let printed_clone = printed_ref.clone();
        let encoding = encoding.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(encoding.token_ids());
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_call = duration.as_nanos() as f64 / iters as f64;

                // Calculate improvement
                let improvement = if owned_time_ns > 0.0 {
                    format!("{:.1}x faster", owned_time_ns / time_per_call)
                } else {
                    "N/A".to_string()
                };

                let result = format!(
                    "{:<20} | {:>12.0} | {:>11.0}ns | {:>12}",
                    "token_ids_ref(ref)", ops_per_sec, time_per_call, improvement
                );
                add_result("memory", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_scaling_characteristics(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let thread_counts = vec![1, 2, 4, 8, 16];
    let tokens_per_thread = 10000;

    let mut group = c.benchmark_group("scaling");
    group.measurement_time(Duration::from_secs(2));

    let mut baseline_throughput = 0.0;

    for num_threads in thread_counts {
        let printed = Arc::new(AtomicBool::new(false));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &threads| {
                let printed_clone = printed.clone();
                let tokenizer = tokenizer.clone();

                b.iter_custom(|_iters| {
                    let total_tokens = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tokenizer = tokenizer.clone();
                            let total_tokens = total_tokens.clone();

                            thread::spawn(move || {
                                let mut decoder = DecodeStream::new(tokenizer, &[], false);
                                let sample_tokens = [1, 450, 6635, 3290, 491];

                                for token in sample_tokens.iter().cycle().take(tokens_per_thread) {
                                    let _ = decoder.step(*token).unwrap();
                                }

                                total_tokens.fetch_add(tokens_per_thread as u64, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total = total_tokens.load(Ordering::Relaxed);
                        let throughput = total as f64 / duration.as_secs_f64();

                        if threads == 1 {
                            baseline_throughput = throughput;
                        }

                        let efficiency = if threads == 1 {
                            100.0
                        } else {
                            (throughput / (baseline_throughput * threads as f64)) * 100.0
                        };

                        let result = format!(
                            "{:<15} | {:>12} | {:>12.0} | {:>11.1}%",
                            format!("{}_threads", threads),
                            total,
                            throughput,
                            efficiency
                        );
                        add_result("scaling", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

// Print final summary table
fn print_summary() {
    println!("\n{}", "=".repeat(120));
    println!("TOKENIZER BENCHMARK SUMMARY");
    println!("{}", "=".repeat(120));

    let results = BENCHMARK_RESULTS.lock().unwrap();

    let mut current_category = String::new();
    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");

        if category != current_category {
            current_category = category.clone();

            // Print section header based on category
            println!("\n{}", "-".repeat(120));
            match category.as_str() {
                "encode" => {
                    println!("ENCODING THROUGHPUT");
                    println!(
                        "{:<15} | {:>8} | {:>8} | {:>12} | {:>12} | {:>10} | {:>10}",
                        "Test Case",
                        "Size(B)",
                        "Tokens",
                        "Chars/sec",
                        "Tokens/sec",
                        "Ops/sec",
                        "Thread"
                    );
                }
                "batch" => {
                    println!("BATCH ENCODING");
                    println!(
                        "{:<15} | {:>8} | {:>8} | {:>12} | {:>12} | {:>10} | {:>10}",
                        "Batch Size",
                        "Size(B)",
                        "Tokens",
                        "Prompts/sec",
                        "Tokens/sec",
                        "Chars/sec",
                        "Thread"
                    );
                }
                "concurrent" => {
                    println!("CONCURRENT ENCODING");
                    println!(
                        "{:<15} | {:>10} | {:>12} | {:>12} | {:>15}",
                        "Clients", "Total Ops", "Ops/sec", "Chars/sec", "Per-Client/sec"
                    );
                }
                "mt_encode" => {
                    println!("MULTI-THREADED ENCODING");
                    println!(
                        "{:<20} | {:>10} | {:>12} | {:>12} | {:>10} | {:>12}",
                        "Configuration",
                        "Total Ops",
                        "Ops/sec",
                        "Tokens/sec",
                        "Threads",
                        "Efficiency"
                    );
                }
                "decode" => {
                    println!("DECODE PERFORMANCE");
                    println!(
                        "{:<20} | {:>10} | {:>12} | {:>12} | {:>10}",
                        "Method", "Tokens", "Tokens/sec", "Ops/sec", "Thread"
                    );
                }
                "mt_decode" => {
                    println!("MULTI-THREADED DECODING");
                    println!(
                        "{:<20} | {:>12} | {:>12} | {:>10} | {:>12}",
                        "Configuration", "Total Tokens", "Tokens/sec", "Threads", "Efficiency"
                    );
                }
                "streaming_100k" => {
                    println!("STREAMING DECODE (100K Target)");
                    println!(
                        "{:<20} | {:>12} | {:>12} | {:>12} | {:>10} | {:>12}",
                        "Method", "Tokens", "Tokens/sec", "Target", "Thread", "Status"
                    );
                }
                "concurrent_streaming" => {
                    println!("CONCURRENT STREAMING");
                    println!(
                        "{:<20} | {:>10} | {:>12} | {:>15} | {:>15}",
                        "Sequences", "Total", "Aggregate/sec", "Per-Seq/sec", "Threads"
                    );
                }
                "stop_sequences" => {
                    println!("STOP SEQUENCE PERFORMANCE");
                    println!(
                        "{:<20} | {:>10} | {:>12} | {:>12} | {:>10}",
                        "Config", "Sequences", "Tokens", "Tokens/sec", "Seq/sec"
                    );
                }
                "latency" => {
                    println!("LATENCY DISTRIBUTION");
                    println!(
                        "{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
                        "Operation", "P50(µs)", "P95(µs)", "P99(µs)", "Max(µs)", "Samples"
                    );
                }
                "scaling" => {
                    println!("SCALING CHARACTERISTICS");
                    println!(
                        "{:<15} | {:>12} | {:>12} | {:>12}",
                        "Threads", "Total Tokens", "Tokens/sec", "Efficiency"
                    );
                }
                "memory" => {
                    println!("MEMORY EFFICIENCY");
                    println!(
                        "{:<20} | {:>12} | {:>12} | {:>12}",
                        "Operation", "Calls/sec", "Time/call", "Improvement"
                    );
                }
                _ => {}
            }
            println!("{}", "-".repeat(120));
        }

        println!("{}", value);
    }

    println!("\n{}", "=".repeat(120));
}

fn run_benchmarks(c: &mut Criterion) {
    bench_encode_throughput(c);
    bench_batch_encode(c);
    bench_concurrent_encode(c);
    bench_multithreaded_encode(c);
    bench_decode_performance(c);
    bench_multithreaded_decode(c);
    bench_streaming_decode_100k(c);
    bench_concurrent_streaming(c);
    bench_stop_sequences(c);
    bench_latency_distribution(c);
    bench_scaling_characteristics(c);
    bench_memory_efficiency(c);

    // Print summary at the end
    print_summary();
}

criterion_group!(benches, run_benchmarks);
criterion::criterion_main!(benches);
