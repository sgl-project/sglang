// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Local benchmark for `dynamo-tokenizers` encode() cost and BPE merge-cache
//! lock contention.
//!
//! Two workload modes:
//!
//! * `shared-prefix` (default): a long shared prefix + a short unique
//!   per-call suffix, encoded repeatedly across threads — the shape of a
//!   real multi-turn chat session (stable system prompt/context, varying
//!   tail). This was used for the `dynamo-tokenizers` version-bump
//!   measurement (see Cargo.toml's comment on that dependency: a live
//!   `Cache::get` share of 87% dropped to a measured 20% after that bump —
//!   the only numbers here checkable against a committed artifact) but a
//!   later investigation into the SAME frame (`tokenizers::utils::cache::
//!   Cache::get` / `RwLock::try_read` contention, observed at 20-27% of
//!   total CPU across separate live captures taken after the version bump
//!   had already landed — the range reflects natural variance between
//!   samples, not a precise bound) found this mode keeps that frame under
//!   1% locally — repeated shared-prefix content warms the per-word merge
//!   cache almost immediately, so most calls only ever hit cheap cache
//!   reads on words they've already seen.
//! * `diverse`: procedurally generates a structurally different word
//!   sequence per call (random word lengths/characters, no shared prefix
//!   across threads or iterations), to stress the merge cache the way
//!   diverse real traffic — many different requests, not one repeated
//!   session — would: lots of first-sight words forcing merge computation
//!   AND lots of concurrent readers/writers on the same underlying
//!   `RwLock<AHashMap<..>>`.
//!
//! `--shards N` loads N independent `Tokenizer` instances (matching
//! `sgl_router::tokenizer::TokenizerRegistry`'s sharding) and round-robins
//! across them instead of sharing one; use this to compare contention
//! before (`--shards 1`) and after (`--shards 8`, say) in the same binary.
//!
//! Run with: `cargo run --release --features profiling --example
//! tokenize_bench -- <tokenizer.json> [threads] [iterations_per_thread]
//! [mode] [shards]`
//! Writes `tokenize_bench_flamegraph.svg` to the current directory.

use dynamo_tokenizers::Tokenizer;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn build_shared_prefix(target_tokens: usize, tokenizer: &Tokenizer) -> String {
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

/// Deterministic, dependency-free xorshift64 PRNG — good enough for
/// generating varied benchmark text, no need for the `rand` crate here.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Build one procedurally-generated, non-repeated "sentence": a sequence of
/// random-length lowercase words (plus occasional punctuation/digits) so the
/// BPE merge cache sees many genuinely novel byte-pair sequences per call —
/// unlike `build_shared_prefix`, nothing here is shared across calls.
fn build_diverse_text(rng: &mut Rng, target_words: usize) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    let mut text = String::new();
    for w in 0..target_words {
        if w > 0 {
            text.push(' ');
        }
        let len = 3 + rng.next_range(9); // 3..12 chars
        for _ in 0..len {
            text.push(ALPHABET[rng.next_range(ALPHABET.len())] as char);
        }
        // Occasional digit-suffixed or capitalized "word" and punctuation,
        // so the vocab mix isn't purely lowercase alpha (closer to real
        // diverse text — identifiers, numbers, sentence breaks).
        match rng.next_range(20) {
            0 => text.push_str(&format!("{}", rng.next_range(100_000))),
            1 => text.push(','),
            2 => text.push('.'),
            _ => {}
        }
    }
    text
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    SharedPrefix,
    Diverse,
}

impl Mode {
    fn parse(s: &str) -> Self {
        match s {
            "shared-prefix" => Mode::SharedPrefix,
            "diverse" => Mode::Diverse,
            other => panic!("unknown mode {other:?}; expected shared-prefix or diverse"),
        }
    }
}

/// N independent tokenizer instances, selected round-robin — mirrors
/// `sgl_router::tokenizer::TokenizerShards` so this benchmark measures
/// exactly the production sharding strategy, not just "N tokenizers".
struct Shards {
    instances: Vec<Arc<Tokenizer>>,
    next: AtomicUsize,
}

impl Shards {
    fn load(path: &str, n: usize) -> Self {
        // Clamp like the production sgl_router::tokenizer::TokenizerShards
        // does — `--shards 0` is an easy typo from the CLI, and panicking on
        // `% 0` in `pick()` is a worse failure mode than silently using 1.
        let n = n.max(1);
        let instances = (0..n)
            .map(|_| Arc::new(Tokenizer::from_file(path).expect("load tokenizer")))
            .collect();
        Self {
            instances,
            next: AtomicUsize::new(0),
        }
    }

    fn pick(&self) -> &Arc<Tokenizer> {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.instances.len();
        &self.instances[i]
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tokenizer_path = args.get(1).cloned().expect(
        "usage: tokenize_bench <tokenizer.json> [threads] [iterations] [mode] [shards]\n\
         mode: shared-prefix (default) | diverse\n\
         shards: number of independent Tokenizer instances, round-robined (default 1)",
    );
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    let mode = args
        .get(4)
        .map(|s| Mode::parse(s))
        .unwrap_or(Mode::SharedPrefix);
    let shard_count: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    eprintln!(
        "loading {shard_count} tokenizer shard(s) from {tokenizer_path}, mode={}, threads={threads}, iterations={iterations}...",
        match mode {
            Mode::SharedPrefix => "shared-prefix",
            Mode::Diverse => "diverse",
        }
    );
    let shards = Shards::load(&tokenizer_path, shard_count);

    // Built once, up front (outside the timed/profiled region) since a
    // shared prefix is meant to model stable context, not per-call cost.
    let shared_prefix = if mode == Mode::SharedPrefix {
        let prefix = build_shared_prefix(70_000, &shards.instances[0]);
        let n = shards.instances[0]
            .encode(&prefix)
            .unwrap()
            .token_ids()
            .len();
        eprintln!("shared prefix built: {} chars, {n} tokens.", prefix.len());
        Some(Arc::new(prefix))
    } else {
        None
    };

    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(200)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .expect("start profiler");

    let start = Instant::now();
    let shards = Arc::new(shards);
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let shards = Arc::clone(&shards);
            let shared_prefix = shared_prefix.clone();
            std::thread::spawn(move || {
                // Per-thread RNG seed so diverse text differs across threads
                // as well as across iterations within a thread.
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1));
                let mut total_tokens = 0usize;
                // Per-call encode() latency, in nanoseconds. Aggregate
                // throughput alone can hide contention on a CPU-bound,
                // oversubscribed box (everyone's slower, but roughly
                // proportionally) — the per-call distribution's tail is the
                // more sensitive signal: lock contention shows up as a
                // heavier tail (calls that lose the CAS race and retry)
                // even when the median barely moves.
                let mut latencies_ns = Vec::with_capacity(iterations);
                for i in 0..iterations {
                    let text = match &shared_prefix {
                        Some(prefix) => {
                            let suffix = format!(
                                " Unique turn marker thread={t} iter={i} value={}",
                                i * 7 + t
                            );
                            format!("{prefix}{suffix}")
                        }
                        None => build_diverse_text(&mut rng, 200),
                    };
                    let tokenizer = shards.pick();
                    let call_start = Instant::now();
                    let encoding = tokenizer.encode(&text).expect("encode");
                    latencies_ns.push(call_start.elapsed().as_nanos() as u64);
                    total_tokens += encoding.token_ids().len();
                }
                (total_tokens, latencies_ns)
            })
        })
        .collect();

    let mut total_tokens = 0usize;
    let mut all_latencies_ns: Vec<u64> = Vec::with_capacity(threads * iterations);
    for h in handles {
        let (tokens, mut lat) = h.join().unwrap();
        total_tokens += tokens;
        all_latencies_ns.append(&mut lat);
    }
    let elapsed = start.elapsed();
    let total_calls = threads * iterations;

    all_latencies_ns.sort_unstable();
    let pct = |p: f64| -> f64 {
        let idx = ((all_latencies_ns.len() as f64 - 1.0) * p).round() as usize;
        all_latencies_ns[idx] as f64 / 1000.0 // us
    };
    let sum_ns: u64 = all_latencies_ns.iter().sum();

    println!(
        "encoded {total_calls} requests, {total_tokens} total tokens, in {:.3}s wall \
         ({:.1} req/s, {:.0} tok/s); per-call encode() latency: \
         p50={:.1}us p90={:.1}us p99={:.1}us p99.9={:.1}us max={:.1}us; \
         sum of per-call latencies={:.3}s (CPU-time proxy across all threads)",
        elapsed.as_secs_f64(),
        total_calls as f64 / elapsed.as_secs_f64(),
        total_tokens as f64 / elapsed.as_secs_f64(),
        pct(0.50),
        pct(0.90),
        pct(0.99),
        pct(0.999),
        all_latencies_ns.last().copied().unwrap_or(0) as f64 / 1000.0,
        sum_ns as f64 / 1e9,
    );

    let report = guard.report().build().expect("build report");
    let mut svg = Vec::new();
    report.flamegraph(&mut svg).expect("render flamegraph");
    std::fs::write("tokenize_bench_flamegraph.svg", &svg).expect("write flamegraph");
    println!("wrote tokenize_bench_flamegraph.svg ({} bytes)", svg.len());
}
