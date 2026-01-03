use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use sgl_model_gateway::tokenizer::{cache::L1Cache, mock::MockTokenizer};

fn generate_prompt(turns: usize) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>");
    for i in 0..turns {
        prompt.push_str(&format!(
            "<|im_start|>user\nIteration {} prompt text to test hashing and tokenization performance.<|im_end|>",
            i
        ));
        prompt.push_str("<|im_start|>assistant\nI am processing your request and generating a valid response.<|im_end|>");
    }
    prompt
}

fn bench_l1_cache(c: &mut Criterion) {
    let special_tokens = vec!["<|im_start|>", "<|im_end|>"];
    let tokenizer = MockTokenizer::new();

    // We test with exponentially increasing turns to see the O(N^2) impact vs O(N)
    for turns in [2, 10, 50].iter() {
        let input = generate_prompt(*turns);
        let mut group = c.benchmark_group(format!("L1-Cache-Turns-{}", turns));

        // Measure throughput in terms of characters processed per second
        group.throughput(Throughput::Elements(input.len() as u64));

        // Insertion Benchmark
        // Current code re-hashes and re-tokenizes the prefix at every boundary.
        // Optimization targets this method specifically.
        group.bench_function("insert_at_boundaries", |b| {
            b.iter(|| {
                // We create a new cache per iteration to ensure we are benchmarking
                // the full insertion logic and not a "no-op" on existing entries.
                let cache = L1Cache::new(100 * 1024 * 1024);
                let _ = cache.insert_at_boundaries(
                    black_box(&input),
                    black_box(&tokenizer),
                    black_box(&special_tokens),
                    black_box(false),
                );
            })
        });

        // Lookup Benchmark
        // This measures the efficiency of the backward search.
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
