// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Policy-selection throughput microbench.
//!
//! Mirrors `sgl-model-gateway/benches/manual_policy_benchmark.rs` —
//! measures how fast the routing layer returns a worker for a given
//! request context, across round-robin, random, and power-of-two choices.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::power_of_two::PowerOfTwoChoicesPolicy;
use sgl_router::policies::random::RandomPolicy;
use sgl_router::policies::round_robin::RoundRobinPolicy;
use sgl_router::policies::scoring::{prefix_cache::PrefixCachePolicy, ScoringPolicy};
use sgl_router::policies::{Policy, SelectionContext};
use sgl_router::state::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId};
use sgl_router::workers::{Worker, WorkerRegistry};
use std::sync::Arc;

fn workers(n: usize, model: &str) -> Vec<Arc<Worker>> {
    let registry = WorkerRegistry::default();
    for i in 0..n {
        registry
            .add(WorkerSpec {
                id: WorkerId(format!("w{i}")),
                url: format!("http://w{i}:30000"),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId(model.into())],
                ..Default::default()
            })
            .expect("test workers are unmixed");
    }
    registry.workers_for(&ModelId(model.into()))
}

fn bench_policy(c: &mut Criterion, name: &str, policy: Arc<dyn Policy>) {
    let mut group = c.benchmark_group(format!("policy_select::{name}"));
    for &n in &[4usize, 16, 64, 256] {
        let workers = workers(n, "tiny");
        let model = ModelId("tiny".into());
        // Same body across iterations — measures the policy's per-call
        // cost rather than body-parsing overhead.
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello world"}],
        }))
        .unwrap();
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let ctx = SelectionContext::new(&model, Some(&body));
                let chosen = policy.select(black_box(&workers), &ctx);
                black_box(chosen);
            });
        });
    }
    group.finish();
}

fn bench_round_robin(c: &mut Criterion) {
    bench_policy(c, "round_robin", Arc::new(RoundRobinPolicy::new()));
}

fn bench_random(c: &mut Criterion) {
    bench_policy(c, "random", Arc::new(RandomPolicy::new()));
}

fn bench_power_of_two(c: &mut Criterion) {
    bench_policy(c, "power_of_two", Arc::new(PowerOfTwoChoicesPolicy::new()));
}

fn bench_prefix_cache(c: &mut Criterion) {
    const BLOCK_SIZE: usize = 16;
    const BLOCKS: usize = 16;

    let mut group = c.benchmark_group("policy_select::prefix_cache");
    for &n in &[16usize, 64, 256, 1024] {
        let workers = workers(n, "tiny");
        let tokens: Vec<u32> = (0..(BLOCK_SIZE * BLOCKS) as u32).collect();
        let hashes = compute_block_hashes(&tokens, BLOCK_SIZE);
        let tree = Arc::new(HashTree::new());
        for (rank, worker) in workers.iter().enumerate() {
            tree.insert(
                &KvWorkerId::new(worker.url.clone(), rank as u32),
                None,
                &hashes,
            );
        }
        let oracle = BlockSizeOracle::new();
        oracle.try_set(BLOCK_SIZE as u32).unwrap();
        let policy = PrefixCachePolicy::new(tree, oracle, 1.0);
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&tokens));

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(policy.scores(black_box(&workers), black_box(&ctx))));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_round_robin,
    bench_random,
    bench_power_of_two,
    bench_prefix_cache
);
criterion_main!(benches);
