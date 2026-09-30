//! Microbench for `BreakerTrackedStream` per-poll overhead.
//!
//! The cancel feature wraps every upstream streaming response body in
//! `BreakerTrackedStream`, which sits in the per-chunk hot path. This bench
//! drains a synthetic in-memory chunk stream both bare and wrapped so the
//! delta isolates the wrapper's cost (state machine + Pin dispatch +
//! terminal-state bookkeeping) with no network noise.
//!
//! Run via `cargo bench --bench streaming_utils_bench`. For A/B against
//! main, use Criterion's baseline machinery:
//! `cargo bench --bench streaming_utils_bench -- --save-baseline main`
//! on main, then `--baseline main` on the feature branch.

use std::{fmt, sync::Arc};

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures_util::StreamExt;
use smg::{
    core::{BasicWorkerBuilder, Worker},
    routers::streaming_utils::BreakerTrackedStream,
};
use tokio::runtime::Runtime;

/// Minimal `Display`-able error type — keeps the wrapper generic so we
/// don't have to fabricate `reqwest::Error` instances.
#[derive(Debug)]
struct BenchErr;

impl fmt::Display for BenchErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bench")
    }
}

/// Build `n` `Ok` chunks of `chunk_size` zero bytes. `Bytes::clone` is
/// cheap (refcount bump), so each call reuses the same allocation.
fn make_chunks(n: usize, chunk_size: usize) -> Vec<Result<Bytes, BenchErr>> {
    let chunk = Bytes::from(vec![0u8; chunk_size]);
    (0..n).map(|_| Ok::<_, BenchErr>(chunk.clone())).collect()
}

fn make_worker() -> Arc<dyn Worker> {
    Arc::new(BasicWorkerBuilder::new("http://bench-worker").build())
}

/// Drain a bare `stream::iter` — the reference cost without the
/// `BreakerTrackedStream` wrapper. Whatever delta the `wrapped` bench
/// shows on top of this is the per-chunk overhead of the wrapper.
fn bench_iter_baseline(c: &mut Criterion) {
    let rt = Runtime::new().expect("runtime");
    let mut group = c.benchmark_group("streaming_utils_baseline_iter");
    const CHUNK_SIZE: usize = 1024;

    for &n in &[16usize, 64, 256] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let chunks = make_chunks(n, CHUNK_SIZE);
                    let mut stream = futures_util::stream::iter(chunks);
                    while let Some(item) = stream.next().await {
                        black_box(item.expect("bench chunk"));
                    }
                });
            });
        });
    }
    group.finish();
}

/// Drain a `BreakerTrackedStream` over the same synthetic chunks. This is
/// the cancel-feature hot path: every upstream byte chunk on a streaming
/// request is polled through this wrapper.
fn bench_tracked_clean(c: &mut Criterion) {
    let rt = Runtime::new().expect("runtime");
    let mut group = c.benchmark_group("streaming_utils_tracked_clean");
    const CHUNK_SIZE: usize = 1024;

    for &n in &[16usize, 64, 256] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let chunks = make_chunks(n, CHUNK_SIZE);
                    let worker = make_worker();
                    let mut stream = BreakerTrackedStream::new(
                        futures_util::stream::iter(chunks),
                        worker,
                        "http://bench".to_string(),
                    );
                    while let Some(item) = stream.next().await {
                        black_box(item.expect("bench chunk"));
                    }
                });
            });
        });
    }
    group.finish();
}

/// `mark_completed` is the PD-streaming `[DONE]`-sentinel fast-path: the
/// caller pre-marks the wrapper completed and stops polling. Times the
/// allocation + mark + drop sequence so changes to the `Terminal`
/// state-machine or `Drop` impl surface here too.
fn bench_tracked_mark_completed_drop(c: &mut Criterion) {
    c.bench_function("streaming_utils_tracked_mark_completed_drop", |b| {
        b.iter(|| {
            let worker = make_worker();
            let inner = futures_util::stream::pending::<Result<Bytes, BenchErr>>();
            let mut tracked = BreakerTrackedStream::new(inner, worker, "http://bench".to_string());
            tracked.mark_completed();
            black_box(&tracked);
            drop(tracked);
        });
    });
}

criterion_group!(
    benches,
    bench_iter_baseline,
    bench_tracked_clean,
    bench_tracked_mark_completed_drop,
);
criterion_main!(benches);
