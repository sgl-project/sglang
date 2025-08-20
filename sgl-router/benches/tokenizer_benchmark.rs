//! Comprehensive tokenizer benchmark with clear metrics output
//! Covers all test cases from the test files with proper metric reporting

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sglang_router_rs::tokenizer::{
    huggingface::HuggingFaceTokenizer, sequence::Sequence, stop::*, stream::DecodeStream,
    traits::*,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const TEST_TOKENIZER: &str = "tests/data/TinyLlama_v1.1/tokenizer.json";

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
        "mathematics", "physics", "chemistry", "biology", "computer science",
        "engineering", "medicine", "law", "economics", "philosophy"
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

// Global flag to track if header has been printed
static HEADER_PRINTED: AtomicBool = AtomicBool::new(false);

fn print_benchmark_header(category: &str) {
    if !HEADER_PRINTED.load(Ordering::Relaxed) {
        println!("\n{}", "=".repeat(120));
        println!("TOKENIZER BENCHMARK METRICS");
        println!("{}", "=".repeat(120));
        HEADER_PRINTED.store(true, Ordering::Relaxed);
    }
    println!("\n{}", "-".repeat(120));
    println!("{}", category);
    println!("{}", "-".repeat(120));
}

fn bench_encode_throughput(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("ENCODING THROUGHPUT (Characters/Second)");
    
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
    
    // Print header
    println!("{:<15} | {:>8} | {:>8} | {:>12} | {:>12} | {:>10} | {:>10}",
        "Test Case", "Size(B)", "Tokens", "Chars/sec", "Tokens/sec", "Ops/sec", "Thread");
    println!("{}", "-".repeat(95));
    
    let mut group = c.benchmark_group("encode_throughput");
    
    for (name, prompt) in test_cases {
        let prompt_len = prompt.len();
        let tokenizer_clone = tokenizer.clone();
        
        // Get token count once
        let token_count = tokenizer.encode(prompt).unwrap().token_ids().len();
        
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
                
                // Print metrics only once per test case
                if !printed_clone.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let chars_per_sec = (iters as f64 * prompt_len as f64) / duration.as_secs_f64();
                    let tokens_per_sec = (iters as f64 * token_count as f64) / duration.as_secs_f64();
                    
                    println!("{:<15} | {:>8} | {:>8} | {:>12.0} | {:>12.0} | {:>10.0} | {:>10}",
                        name, prompt_len, token_count, chars_per_sec, tokens_per_sec, ops_per_sec, 1);
                    
                    printed_clone.store(true, Ordering::Relaxed);
                }
                
                duration
            });
        });
    }
    
    group.finish();
}

fn bench_batch_encode(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("BATCH ENCODING THROUGHPUT");
    
    let batch_sizes = vec![1, 8, 16, 32, 64, 128];
    let prompt = MEDIUM_PROMPT;
    let prompt_len = prompt.len();
    let token_count = tokenizer.encode(prompt).unwrap().token_ids().len();
    
    // Print header
    println!("{:<15} | {:>8} | {:>8} | {:>12} | {:>12} | {:>10} | {:>10}",
        "Batch Size", "Size(B)", "Tokens", "Prompts/sec", "Tokens/sec", "Chars/sec", "Thread");
    println!("{}", "-".repeat(95));
    
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
                        
                        println!("{:<15} | {:>8} | {:>8} | {:>12.0} | {:>12.0} | {:>10.0} | {:>10}",
                            format!("batch_{}", size), 
                            prompt_len * size, 
                            token_count * size,
                            prompts_per_sec, 
                            tokens_per_sec,
                            chars_per_sec,
                            1);
                        
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
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("CONCURRENT ENCODING (Multiple Clients)");
    
    let client_counts = vec![1, 4, 8, 16, 32];
    
    // Print header
    println!("{:<15} | {:>10} | {:>12} | {:>12} | {:>15} | {:>12}",
        "Clients", "Total Ops", "Ops/sec", "Chars/sec", "Per-Client/sec", "Efficiency");
    println!("{}", "-".repeat(90));
    
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
                                let prompts = vec![SHORT_PROMPT, MEDIUM_PROMPT, LONG_PROMPT];
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
                        
                        // Simple efficiency calculation
                        let efficiency = if clients == 1 { 100.0 } else { (per_client / (ops_per_sec / clients as f64)) * 100.0 };
                        
                        println!("{:<15} | {:>10} | {:>12.0} | {:>12.0} | {:>15.0} | {:>11.1}%",
                            format!("{}_clients", clients),
                            total_ops,
                            ops_per_sec,
                            chars_per_sec,
                            per_client,
                            efficiency);
                        
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
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("DECODE PERFORMANCE");
    
    let test_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let tokens = tokenizer.encode(&test_text).unwrap().token_ids();
    let num_tokens = tokens.len();
    
    // Print header
    println!("{:<20} | {:>10} | {:>12} | {:>12} | {:>10}",
        "Method", "Tokens", "Tokens/sec", "Ops/sec", "Thread");
    println!("{}", "-".repeat(70));
    
    let mut group = c.benchmark_group("decode_performance");
    
    // Test direct decode
    let printed_direct = Arc::new(AtomicBool::new(false));
    group.bench_function("direct_decode", |b| {
        let printed = printed_direct.clone();
        let tokenizer = tokenizer.clone();
        let tokens = tokens.clone();
        
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(tokenizer.decode(&tokens, false).unwrap());
            }
            let duration = start.elapsed();
            
            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let tokens_per_sec = ops_per_sec * num_tokens as f64;
                
                println!("{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "Direct", num_tokens, tokens_per_sec, ops_per_sec, 1);
                
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
        let tokens = tokens.clone();
        
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
                let mut output = String::new();
                for token in &tokens {
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
                
                println!("{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "DecodeStream", num_tokens, tokens_per_sec, ops_per_sec, 1);
                
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
        let tokens = tokens.clone();
        
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut sequence = Sequence::new(tokenizer.clone());
                let mut output = String::new();
                for token in &tokens {
                    let text = sequence.append_token(*token).unwrap();
                    output.push_str(&text);
                }
                black_box(output);
            }
            let duration = start.elapsed();
            
            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let tokens_per_sec = ops_per_sec * num_tokens as f64;
                
                println!("{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "Sequence", num_tokens, tokens_per_sec, ops_per_sec, 1);
                
                printed.store(true, Ordering::Relaxed);
            }
            
            duration
        });
    });
    
    group.finish();
}

fn bench_streaming_decode_100k(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("STREAMING DECODE AT 100K TOKENS/SEC TARGET");
    
    let sample_text = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    let all_tokens = tokenizer.encode(&sample_text).unwrap().token_ids();
    
    // Print header
    println!("{:<20} | {:>12} | {:>12} | {:>12} | {:>10} | {:>12}",
        "Method", "Tokens", "Tokens/sec", "Target", "Thread", "Status");
    println!("{}", "-".repeat(85));
    
    let mut group = c.benchmark_group("streaming_100k");
    group.measurement_time(Duration::from_secs(1));
    
    // Test DecodeStream
    let printed_stream = Arc::new(AtomicBool::new(false));
    group.bench_function("decode_stream_100k", |b| {
        let printed = printed_stream.clone();
        let tokenizer = tokenizer.clone();
        let tokens = all_tokens.clone();
        
        b.iter_custom(|_iters| {
            let start = Instant::now();
            let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
            let mut output = String::new();
            let mut tokens_processed = 0u64;
            
            for token in tokens.iter().cycle() {
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
                
                println!("{:<20} | {:>12} | {:>12.0} | {:>12} | {:>10} | {:>12}",
                    "DecodeStream",
                    tokens_processed,
                    tokens_per_sec,
                    TARGET_TOKENS_PER_SECOND,
                    1,
                    status);
                
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
        let tokens = all_tokens.clone();
        
        b.iter_custom(|_iters| {
            let start = Instant::now();
            let mut sequence = Sequence::new(tokenizer.clone());
            let mut output = String::new();
            let mut tokens_processed = 0u64;
            
            for token in tokens.iter().cycle() {
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
                
                println!("{:<20} | {:>12} | {:>12.0} | {:>12} | {:>10} | {:>12}",
                    "Sequence",
                    tokens_processed,
                    tokens_per_sec,
                    TARGET_TOKENS_PER_SECOND,
                    1,
                    status);
                
                printed.store(true, Ordering::Relaxed);
            }
            
            duration
        });
    });
    
    group.finish();
}

fn bench_concurrent_streaming(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("CONCURRENT STREAMING (Multiple Sequences)");
    
    let num_sequences = 16;
    let tokens_per_sequence = 10_000;
    
    let sample_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let token_batch = tokenizer.encode(&sample_text).unwrap().token_ids();
    
    // Print header
    println!("{:<20} | {:>10} | {:>12} | {:>15} | {:>15}",
        "Sequences", "Total", "Aggregate/sec", "Per-Seq/sec", "Threads");
    println!("{}", "-".repeat(80));
    
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
                
                println!("{:<20} | {:>10} | {:>12.0} | {:>15.0} | {:>15}",
                    format!("{}_sequences", num_sequences),
                    total,
                    throughput,
                    per_seq,
                    num_sequences);
                
                printed_clone.store(true, Ordering::Relaxed);
            }
            
            duration
        });
    });
    
    group.finish();
}

fn bench_latency_distribution(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("LATENCY DISTRIBUTION");
    
    // Test latency for individual token processing
    let sample_tokens = vec![1, 450, 6635, 3290, 491, 278, 3474, 29892];
    
    // Print header
    println!("{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Operation", "P50(µs)", "P95(µs)", "P99(µs)", "Max(µs)", "Samples");
    println!("{}", "-".repeat(75));
    
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
            
            b.iter(|| {
                if !printed_clone.load(Ordering::Relaxed) {
                    let mut latencies = Vec::new();
                    
                    // Warm up
                    for _ in 0..100 {
                        let _ = tokenizer.encode(prompt).unwrap();
                    }
                    
                    // Measure
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
                    
                    println!("{:<20} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10}",
                        name,
                        p50.as_micros() as f64,
                        p95.as_micros() as f64,
                        p99.as_micros() as f64,
                        max.as_micros() as f64,
                        1000);
                    
                    printed_clone.store(true, Ordering::Relaxed);
                }
                
                // Dummy work for iterations after first
                black_box(tokenizer.encode(prompt).unwrap());
            });
        });
    }
    
    // Decode token latency
    let printed_decode = Arc::new(AtomicBool::new(false));
    group.bench_function("decode_token", |b| {
        let printed_clone = printed_decode.clone();
        let tokenizer = tokenizer.clone();
        let tokens = sample_tokens.clone();
        
        b.iter(|| {
            if !printed_clone.load(Ordering::Relaxed) {
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
                
                println!("{:<20} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10}",
                    "decode_token",
                    p50.as_micros() as f64,
                    p95.as_micros() as f64,
                    p99.as_micros() as f64,
                    max.as_micros() as f64,
                    1000);
                
                // Check target latency
                let target_latency = Duration::from_micros(10);
                if p50 > target_latency {
                    println!("    WARNING: P50 latency exceeds target of {:?} for 100k tokens/sec", target_latency);
                }
                
                printed_clone.store(true, Ordering::Relaxed);
            }
            
            // Dummy work for iterations after first
            let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
            for token in tokens.iter().take(10) {
                black_box(decoder.step(*token).unwrap());
            }
        });
    });
    
    group.finish();
}

fn bench_scaling_characteristics(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("SCALING CHARACTERISTICS");
    
    let thread_counts = vec![1, 2, 4, 8, 16];
    let tokens_per_thread = 10000;
    
    // Print header  
    println!("{:<15} | {:>12} | {:>12} | {:>12}",
        "Threads", "Total Tokens", "Tokens/sec", "Efficiency");
    println!("{}", "-".repeat(55));
    
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
                                let sample_tokens = vec![1, 450, 6635, 3290, 491];
                                
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
                        
                        println!("{:<15} | {:>12} | {:>12.0} | {:>11.1}%",
                            threads,
                            total,
                            throughput,
                            efficiency);
                        
                        printed_clone.store(true, Ordering::Relaxed);
                    }
                    
                    duration
                });
            },
        );
    }
    
    group.finish();
}

fn bench_stop_sequences(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("STOP SEQUENCE PERFORMANCE");
    
    let config = StopSequenceConfig::default()
        .with_stop_sequence("</s>")
        .with_stop_sequence("\n\n")
        .with_stop_sequence("###")
        .with_stop_token(2);
    
    let sample_text = "Hello world! This is a test. ### Stop here. Continue after.".repeat(100);
    let tokens = tokenizer.encode(&sample_text).unwrap().token_ids();
    
    // Print header
    println!("{:<20} | {:>10} | {:>12} | {:>12} | {:>10}",
        "Config", "Sequences", "Tokens", "Tokens/sec", "Seq/sec");
    println!("{}", "-".repeat(70));
    
    let mut group = c.benchmark_group("stop_sequences");
    
    // No stops
    let printed_no_stop = Arc::new(AtomicBool::new(false));
    group.bench_function("no_stops", |b| {
        let printed_clone = printed_no_stop.clone();
        let tokenizer = tokenizer.clone();
        let tokens = tokens.clone();
        
        b.iter_custom(|iters| {
            let start = Instant::now();
            let mut total_tokens = 0u64;
            
            for _ in 0..iters {
                let mut decoder = StopSequenceDecoder::new(
                    tokenizer.clone(),
                    StopSequenceConfig::default(),
                    false,
                );
                for token in &tokens {
                    let _ = decoder.process_token(*token).unwrap();
                    total_tokens += 1;
                }
            }
            
            let duration = start.elapsed();
            
            if !printed_clone.load(Ordering::Relaxed) {
                let tokens_per_sec = total_tokens as f64 / duration.as_secs_f64();
                let seq_per_sec = iters as f64 / duration.as_secs_f64();
                
                println!("{:<20} | {:>10} | {:>12} | {:>12.0} | {:>10.0}",
                    "No stops",
                    iters,
                    total_tokens,
                    tokens_per_sec,
                    seq_per_sec);
                
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
        let tokens = tokens.clone();
        let config = config.clone();
        
        b.iter_custom(|iters| {
            let start = Instant::now();
            let mut total_tokens = 0u64;
            let mut total_sequences = 0u64;
            
            for _ in 0..iters {
                let mut decoder = StopSequenceDecoder::new(tokenizer.clone(), config.clone(), false);
                let mut sequence_tokens = 0u64;
                
                for token in &tokens {
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
                
                println!("{:<20} | {:>10} | {:>12} | {:>12.0} | {:>10.0}",
                    "With stops",
                    total_sequences,
                    total_tokens,
                    tokens_per_sec,
                    seq_per_sec);
                
                printed_clone.store(true, Ordering::Relaxed);
            }
            
            duration
        });
    });
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(TEST_TOKENIZER).expect("Failed to load tokenizer"),
    );
    
    print_benchmark_header("MEMORY EFFICIENCY");
    
    let large_text = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    let encoding = tokenizer.encode(&large_text).unwrap();
    
    // Print header
    println!("{:<20} | {:>12} | {:>12} | {:>12}",
        "Operation", "Calls/sec", "Time/call", "Improvement");
    println!("{}", "-".repeat(60));
    
    let mut group = c.benchmark_group("memory");
    
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
                
                println!("{:<20} | {:>12.0} | {:>11.0}ns | {:>12}",
                    "token_ids(owned)",
                    ops_per_sec,
                    time_per_call,
                    "baseline");
                
                printed_clone.store(true, Ordering::Relaxed);
            }
            
            duration
        });
    });
    
    // Reference
    let printed_ref = Arc::new(AtomicBool::new(false));
    let mut owned_time = Duration::from_secs(0);
    
    group.bench_function("token_ids_ref", |b| {
        let printed_clone = printed_ref.clone();
        let encoding = encoding.clone();
        
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(encoding.token_ids_ref());
            }
            let duration = start.elapsed();
            
            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_call = duration.as_nanos() as f64 / iters as f64;
                
                // Calculate improvement
                let improvement = if owned_time.as_nanos() > 0 {
                    format!("{:.1}x faster", owned_time.as_nanos() as f64 / duration.as_nanos() as f64)
                } else {
                    "N/A".to_string()
                };
                
                println!("{:<20} | {:>12.0} | {:>11.0}ns | {:>12}",
                    "token_ids_ref(ref)",
                    ops_per_sec,
                    time_per_call,
                    improvement);
                
                printed_clone.store(true, Ordering::Relaxed);
            } else {
                owned_time = duration; // Store for comparison
            }
            
            duration
        });
    });
    
    group.finish();
    
    println!("\n{}", "=".repeat(120));
}

criterion_group!(
    benches,
    bench_encode_throughput,
    bench_batch_encode,
    bench_concurrent_encode,
    bench_decode_performance,
    bench_streaming_decode_100k,
    bench_concurrent_streaming,
    bench_latency_distribution,
    bench_scaling_characteristics,
    bench_stop_sequences,
    bench_memory_efficiency,
);

criterion_main!(benches);