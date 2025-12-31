use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sgl_model_gateway::tokenizer::{cache::l1::L1Cache, mock::MockTokenizer};

/// Generates a realistic ChatML prompt with N turns to test boundary scaling
fn generate_prompt(turns: usize) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\nYou are a helpful AI thought partner.<|im_end|>");
    for i in 0..turns {
        prompt.push_str(&format!("<|im_start|>user\nIteration {} prompt text that is somewhat long to test hashing performance.<|im_end|>", i));
        prompt.push_str("<|im_start|>assistant\nCertainly! Here is a detailed response to your query.<|im_end|>");
    }
    prompt
}

fn bench_l1_cache(c: &mut Criterion) {
    let special_tokens = vec!["<|im_start|>", "<|im_end|>"];
    let tokenizer = MockTokenizer::new();

    // Test with varying turn counts (number of boundaries)
    // Current code is O(B*L), optimization makes it O(L)
    for turns in [1, 5, 20, 50].iter() {
        let input = generate_prompt(*turns);
        let mut group = c.benchmark_group(format!("L1Cache-Turns-{}", turns));

        // --- 1. Insertion Benchmark ---
        // This is where "Incremental Hashing & Tokenization" helps most
        group.bench_function("insert_at_boundaries", |b| {
            b.iter(|| {
                let cache = L1Cache::new(100 * 1024 * 1024); // 100MB
                let _ = cache.insert_at_boundaries(
                    black_box(&input),
                    black_box(&tokenizer),
                    black_box(&special_tokens),
                    black_box(false),
                );
            })
        });

        // --- 2. Lookup Benchmark ---
        // Measures search efficiency across many boundaries
        group.bench_function("longest_prefix_match", |b| {
            let cache = L1Cache::new(100 * 1024 * 1024);
            let _ = cache.insert_at_boundaries(&input, &tokenizer, &special_tokens, false);

            b.iter(|| {
                let _ = cache.longest_prefix_match(black_box(&input), black_box(&special_tokens));
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_l1_cache);
criterion_main!(benches);
