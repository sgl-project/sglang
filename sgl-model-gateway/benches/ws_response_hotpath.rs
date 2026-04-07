use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smg::bench_support;

fn bench_frame_emission(c: &mut Criterion) {
    let mut group = c.benchmark_group("ws_frame_emission");

    for (delta_events, delta_bytes) in [(16usize, 64usize), (64, 128), (128, 256)] {
        let label = format!("{delta_events}x{delta_bytes}");
        group.throughput(Throughput::Bytes((delta_events * delta_bytes) as u64));

        group.bench_with_input(
            BenchmarkId::new("ws", &label),
            &(delta_events, delta_bytes),
            |b, &(events, bytes)| {
                b.iter(|| black_box(bench_support::bench_emit_ws_text_stream(events, bytes)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sse", &label),
            &(delta_events, delta_bytes),
            |b, &(events, bytes)| {
                b.iter(|| black_box(bench_support::bench_emit_sse_text_stream(events, bytes)));
            },
        );
    }

    group.finish();
}

fn bench_cached_continuation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_continuation_shape");

    for history_turns in [4usize, 16, 64] {
        group.bench_with_input(
            BenchmarkId::new("cached_to_items", history_turns),
            &history_turns,
            |b, &turns| {
                b.iter(|| black_box(bench_support::bench_cached_response_to_items(turns, 256)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cache_hit_merge", history_turns),
            &history_turns,
            |b, &turns| {
                b.iter(|| {
                    black_box(bench_support::bench_cached_continuation_shape(
                        turns, 256, 4, 256,
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_incremental_input_shaping(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_input_shaping");

    for item_count in [8usize, 32, 128, 512] {
        group.bench_with_input(
            BenchmarkId::new("normalize_items", item_count),
            &item_count,
            |b, &count| {
                b.iter(|| {
                    black_box(bench_support::bench_normalize_incremental_request(
                        count, 256,
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    ws_response_hotpath,
    bench_frame_emission,
    bench_cached_continuation,
    bench_incremental_input_shaping
);
criterion_main!(ws_response_hotpath);
