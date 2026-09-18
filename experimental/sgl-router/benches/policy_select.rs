// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Policy-selection throughput microbench.
//!
//! Mirrors `sgl-model-gateway/benches/manual_policy_benchmark.rs` —
//! measures how fast the routing layer returns a worker for a given
//! request context, across round-robin, random, and power-of-two choices.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies::random::RandomPolicy;
use sgl_router::policies::round_robin::RoundRobinPolicy;
use sgl_router::policies::state::engine_load::{EngineLoadTable, LoadView};
use sgl_router::policies::{Admission, AffinityScope, PickMode, PickRequest, Policy, RoutingStage};
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
        let table = EngineLoadTable::new();
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let load = LoadView::new(&table);
                let request = PickRequest {
                    model: &model,
                    stage: RoutingStage::Plain,
                    bucket_id: "global",
                    scope: AffinityScope::Bucket,
                    mode: PickMode::Normal,
                    input_tokens: 16,
                    expected_peak_sequence_tokens: None,
                    session_id: None,
                    routing_key: None,
                    tokens: None,
                    prefix: None,
                    affinity_enabled: true,
                    load: &load,
                };
                let chosen =
                    futures::executor::block_on(policy.pick(black_box(&workers), &request));
                black_box(chosen.is_ok());
            });
        });
    }
    group.finish();
}

fn bench_round_robin(c: &mut Criterion) {
    bench_policy(
        c,
        "round_robin",
        Arc::new(RoundRobinPolicy::new(Admission::allow_all())),
    );
}

fn bench_random(c: &mut Criterion) {
    bench_policy(
        c,
        "random",
        Arc::new(RandomPolicy::new(Admission::allow_all())),
    );
}

fn bench_power_of_two(c: &mut Criterion) {
    bench_policy(
        c,
        "power_of_two",
        Arc::new(PowerOfTwoPolicy::new(Admission::allow_all())),
    );
}

criterion_group!(benches, bench_round_robin, bench_random, bench_power_of_two);
criterion_main!(benches);
