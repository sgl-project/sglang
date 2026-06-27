// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Policy-selection throughput microbench.
//!
//! Mirrors `sgl-model-gateway/benches/manual_policy_benchmark.rs` —
//! measures how fast the routing layer returns a worker for a given
//! request context, across the policies sgl-router actually ships
//! (round-robin, random, power-of-two-choices). The cache-aware-zmq
//! policy lives in `tree_lookup.rs`; this file targets the non-tree
//! policies' steady-state hot path.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::power_of_two::PowerOfTwoChoicesPolicy;
use sgl_router::policies::random::RandomPolicy;
use sgl_router::policies::round_robin::RoundRobinPolicy;
use sgl_router::policies::{Policy, SelectionContext};
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
                bootstrap_port: None,
                min_priority: None,
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

criterion_group!(benches, bench_round_robin, bench_random, bench_power_of_two);
criterion_main!(benches);
