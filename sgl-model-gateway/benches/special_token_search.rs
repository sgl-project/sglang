use aho_corasick::AhoCorasick;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

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
    boundaries.sort_by_key(|b| b.0);
    boundaries
}

fn find_special_token_boundaries_aho(text: &str, ac: &AhoCorasick) -> Vec<(usize, usize)> {
    ac.find_iter(text)
        .map(|mat| (mat.start(), mat.end()))
        .collect()
}

fn bench_token_search_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_boundary_search");
    let text = "User: Hello! Assistant: How can I help you today? ".repeat(1000);

    for token_count in [5, 50] {
        let special_tokens: Vec<String> = (0..token_count)
            .map(|i| format!("<|stop_sequence_{}|>", i))
            .collect();

        let ac = AhoCorasick::new(&special_tokens).unwrap();
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function(format!("naive_tokens_{}", token_count), |b| {
            b.iter(|| {
                find_special_token_boundaries_naive(black_box(&text), black_box(&special_tokens))
            })
        });

        group.bench_function(format!("aho_tokens_{}", token_count), |b| {
            b.iter(|| find_special_token_boundaries_aho(black_box(&text), black_box(&ac)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_token_search_comparison);
criterion_main!(benches);
