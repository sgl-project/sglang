use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

/// Naive implementation as described in the issue
fn find_special_token_boundaries_naive(
    text: &str,
    special_tokens: &[String],
) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::new();
    for token in special_tokens {
        let mut start = 0;
        while let Some(pos) = text[start..].find(token) {
            let actual_pos = start + pos;
            boundaries.push((actual_pos, actual_pos + token.len()));
            start = actual_pos + token.len();
        }
    }
    // Reconstruct the order of tokens as they appear in the text
    boundaries.sort_by_key(|b| b.0);
    boundaries
}

fn bench_naive_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_token_search_naive");

    // Simulate a typical LLM prompt context (approx 50KB)
    let text = "User: Hello! Assistant: How can I help you today? ".repeat(1000);

    // Test varying complexity of stop sequences
    for token_count in [5, 20, 50] {
        let special_tokens: Vec<String> = (0..token_count)
            .map(|i| format!("<|stop_sequence_{}|>", i))
            .collect();

        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function(format!("naive_tokens_{}", token_count), |b| {
            b.iter(|| {
                find_special_token_boundaries_naive(black_box(&text), black_box(&special_tokens))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_naive_search);
criterion_main!(benches);
