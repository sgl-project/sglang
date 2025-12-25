use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sgl_model_gateway::tokenizer::cache::L1Cache;
use sgl_model_gateway::tokenizer::mock::MockTokenizer;

fn bench_l1_insertion_scaling(c: &mut Criterion) {
    let cache = L1Cache::new(100 * 1024 * 1024); // 100MB
    let tokenizer = MockTokenizer::new();
    let special_tokens = vec!["<|im_start|>", "<|im_end|>"];

    // A long 50-turn conversation
    // This creates 50 special token boundaries.
    let mut stress_prompt = String::new();
    for i in 0..50 {
        stress_prompt.push_str(&format!(
            "<|im_start|>user\nThis is turn number {} of a very long conversation that requires many cache points.<|im_end|>\n",
            i
        ));
        stress_prompt.push_str(&format!(
            "<|im_start|>assistant\nI am processing turn {} and keeping the history in context.<|im_end|>\n",
            i
        ));
    }

    let mut group = c.benchmark_group("L1_Cache_Insertion");

    group.bench_function("insert_50_boundaries", |b| {
        b.iter(|| {
            // We clear to ensure we are measuring the insertion logic, not just lookups
            cache.clear();
            let _ = cache.insert_at_boundaries(
                black_box(&stress_prompt),
                black_box(&tokenizer),
                black_box(&special_tokens),
            );
        })
    });

    group.finish();
}

criterion_group!(benches, bench_l1_insertion_scaling);
criterion_main!(benches);
