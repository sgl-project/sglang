// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Local benchmark for the `dynamo-tokenizers` version bump. Not part of the
//! router's own request path — this exists to measure the encode() cost this
//! session's flamegraph found dominating production CPU (87% of samples in
//! `tokenizers::utils::cache::Cache::get`) without needing a live cluster.
//! Comparing this against a live production flamegraph confirmed a real
//! ~4x drop in that same frame's share after the bump (see Cargo.toml's
//! comment on the `dynamo-tokenizers` dependency for the numbers) — not from
//! dynamo's own L1 prefix cache (`Tokenizer::from_file`, what we call, never
//! constructs that wrapper), but from something in a transitive dependency
//! the bump pulled forward.
//!
//! Still uses a long shared prefix followed by a short unique per-call
//! suffix, encoded repeatedly across threads, since that's the shape a real
//! multi-turn chat session has (a stable system prompt/context, a varying
//! tail) — a single one-off encode of wholly unique text wouldn't be
//! representative of production traffic.
//!
//! Run with: `cargo run --release --features profiling --example
//! tokenize_bench -- <tokenizer.json path> [threads] [iterations_per_thread]`
//! Writes `tokenize_bench_flamegraph.svg` to the current directory.

use dynamo_tokenizers::Tokenizer;
use std::sync::Arc;
use std::time::Instant;

fn build_prefix(target_tokens: usize, tokenizer: &Tokenizer) -> String {
    let sentence = "The quick brown fox jumps over the lazy dog near the riverbank while the sun sets slowly behind the distant mountains. ";
    let mut text = String::new();
    loop {
        text.push_str(sentence);
        // Cheap enough to just re-check length by re-encoding periodically
        // rather than every append.
        if text.len() % (sentence.len() * 50) < sentence.len() {
            let n = tokenizer
                .encode(&text)
                .expect("prefix encode")
                .token_ids()
                .len();
            if n >= target_tokens {
                break;
            }
        }
    }
    text
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tokenizer_path = args
        .get(1)
        .cloned()
        .expect("usage: tokenize_bench <tokenizer.json> [threads] [iterations]");
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path).expect("load tokenizer"));

    eprintln!("building ~70k-token shared prefix...");
    let prefix = Arc::new(build_prefix(70_000, &tokenizer));
    let prefix_tokens = tokenizer.encode(&prefix).unwrap().token_ids().len();
    eprintln!(
        "prefix built: {} chars, {} tokens. Running {} threads x {} iterations...",
        prefix.len(),
        prefix_tokens,
        threads,
        iterations
    );

    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(200)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .expect("start profiler");

    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let tokenizer = Arc::clone(&tokenizer);
            let prefix = Arc::clone(&prefix);
            std::thread::spawn(move || {
                let mut total_tokens = 0usize;
                for i in 0..iterations {
                    let suffix = format!(
                        " Unique turn marker thread={t} iter={i} value={}",
                        i * 7 + t
                    );
                    let text = format!("{prefix}{suffix}");
                    let encoding = tokenizer.encode(&text).expect("encode");
                    total_tokens += encoding.token_ids().len();
                }
                total_tokens
            })
        })
        .collect();

    let total_tokens: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let elapsed = start.elapsed();
    let total_calls = threads * iterations;

    println!(
        "encoded {total_calls} requests, {total_tokens} total tokens, in {:.3}s ({:.1} req/s, {:.0} tok/s)",
        elapsed.as_secs_f64(),
        total_calls as f64 / elapsed.as_secs_f64(),
        total_tokens as f64 / elapsed.as_secs_f64(),
    );

    let report = guard.report().build().expect("build report");
    let mut svg = Vec::new();
    report.flamegraph(&mut svg).expect("render flamegraph");
    std::fs::write("tokenize_bench_flamegraph.svg", &svg).expect("write flamegraph");
    println!("wrote tokenize_bench_flamegraph.svg ({} bytes)", svg.len());
}
