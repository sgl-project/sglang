//! Standalone tokenize microbench — the flamegraph vehicle for the TTFT
//! breakdown's dominant TM-side stage (kan/rust-tm-ttft-breakdown).
//!
//! perf cannot attach to (or wrap) the CUDA scheduler process in restricted
//! containers, but it CAN profile a small own-child process. This bin replays
//! exactly what `TokenizerWorker` does per request — a dynamo-tokenizers
//! encode of the prompt text — so `perf record -g -- tokenize_bench ...`
//! (or `cargo flamegraph --bin tokenize_bench -- ...`) yields a full-rate
//! flamegraph of the encode path.
//!
//! Usage: tokenize_bench <tokenizer.json> <text-file> [iters=20]

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: tokenize_bench <tokenizer.json> <text-file> [iters]");
        std::process::exit(2);
    }
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file_with_options(
        &args[1],
        dynamo_tokenizers::TokenizerOptions {
            add_special_tokens: true,
        },
    )
    .expect("tokenizer load failed");
    let text = std::fs::read_to_string(&args[2]).expect("text file read failed");
    let iters: usize = args.get(3).map_or(20, |s| s.parse().expect("bad iters"));

    let t0 = Instant::now();
    let mut total_tokens = 0usize;
    for _ in 0..iters {
        let encoding = tokenizer.encode(&text).expect("encode failed");
        total_tokens += encoding.token_ids().len();
    }
    let elapsed = t0.elapsed();
    println!(
        "{iters} iters, {total_tokens} tokens, {elapsed:.2?} total, {:.2?}/iter, {:.2} us/token",
        elapsed / iters as u32,
        elapsed.as_micros() as f64 / total_tokens as f64,
    );
}
