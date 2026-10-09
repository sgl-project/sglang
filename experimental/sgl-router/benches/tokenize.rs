// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Tokenization hot-path microbench.
//!
//! The cache-aware-zmq policy tokenizes the incoming prompt on the router,
//! synchronously, *before* the request is forwarded — so this cost lands
//! directly on time-to-first-token. `tree_lookup.rs` benches the prefix tree
//! the resulting tokens feed into; this file benches the step before it: turning
//! request text into token ids.
//!
//! Two groups:
//!
//!   * `encode` — pure `adapter::encode(text)` vs input length. The dominant
//!     variable for router-side TTFT overhead.
//!   * `request_path` — what the policy actually does per cache-aware request:
//!     parse the JSON body, pull the chat `content`, then encode. Isolates how
//!     much of the per-request cost is JSON parsing vs tokenization.
//!
//! To run:
//!
//!   cargo bench --bench tokenize
//!   cargo bench --bench tokenize -- --sample-size 30   # faster
//!
//! Fixture caveat: `tests/fixtures/tiny_tokenizer.json` is a 257-entry
//! byte-level tokenizer with no merge table. It exercises the full encode
//! pipeline (normalization + ByteLevel pre-tokenization + model lookup +
//! id-vec build), whose cost scales with input length — but a production BPE
//! with a large merge table does strictly more work per token, so these numbers
//! are a *lower bound* on real tokenization cost, not a prediction of it.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sgl_router::tokenizer::adapter;
use std::sync::Arc;

const TINY_TOKENIZER: &str = "tests/fixtures/tiny_tokenizer.json";

/// Input sizes in bytes spanning a short chat turn → a long context that stays
/// under the router's 1 MiB body cap. Token count tracks byte count closely for
/// the byte-level fixture, so this doubles as a token-count sweep.
const SIZES: &[usize] = &[64, 1_024, 16_384, 131_072];

/// Build a text body of roughly `target` bytes from English-like words, so the
/// ByteLevel pre-tokenizer does realistic word splitting rather than collapsing
/// a single repeated character.
fn text_of_len(target: usize) -> String {
    const SAMPLE: &str = "The quick brown fox jumps over the lazy dog while the engine warms up. ";
    let mut s = String::with_capacity(target + SAMPLE.len());
    while s.len() < target {
        s.push_str(SAMPLE);
    }
    s.truncate(target);
    s
}

/// Mirror what the cache-aware-zmq policy pulls out of a chat body: the
/// concatenated `content` of every message. Kept deliberately minimal — the
/// point is to measure parse + extract, not to re-implement the full extractor.
fn extract_chat_content(v: &serde_json::Value) -> String {
    v.get("messages")
        .and_then(|m| m.as_array())
        .map(|msgs| {
            msgs.iter()
                .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn bench_encode(c: &mut Criterion) {
    let tok: Arc<_> = adapter::load(TINY_TOKENIZER)
        .expect("load tiny tokenizer fixture (run from the crate root)");
    let mut group = c.benchmark_group("tokenize/encode");
    for &size in SIZES {
        let text = text_of_len(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text, |b, text| {
            b.iter(|| {
                let ids = adapter::encode(tok.as_ref(), black_box(text)).expect("encode");
                black_box(ids);
            });
        });
    }
    group.finish();
}

fn bench_request_path(c: &mut Criterion) {
    let tok: Arc<_> = adapter::load(TINY_TOKENIZER)
        .expect("load tiny tokenizer fixture (run from the crate root)");
    let mut group = c.benchmark_group("tokenize/request_path");
    for &size in SIZES {
        // Serialize once outside the timed loop — we measure the router's
        // per-request work (parse + extract + encode), not body construction.
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": text_of_len(size)}],
        }))
        .unwrap();
        group.throughput(Throughput::Bytes(body.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &body, |b, body| {
            b.iter(|| {
                let v: serde_json::Value = serde_json::from_slice(black_box(body)).unwrap();
                let text = extract_chat_content(&v);
                let ids = adapter::encode(tok.as_ref(), &text).expect("encode");
                black_box(ids);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_request_path);
criterion_main!(benches);
