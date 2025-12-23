use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use axum::{
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dashmap::DashMap;
use sgl_model_gateway::routers::{router_manager::RouterId, RouterTrait};

#[derive(Debug)]
struct MockRouter {
    is_pd: bool,
}

#[async_trait]
impl RouterTrait for MockRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn router_type(&self) -> &'static str {
        if self.is_pd {
            "pd"
        } else {
            "regular"
        }
    }
    fn is_pd_mode(&self) -> bool {
        self.is_pd
    }
    async fn route_chat(
        &self,
        _: Option<&HeaderMap>,
        _: &sgl_model_gateway::protocols::chat::ChatCompletionRequest,
        _: Option<&str>,
    ) -> axum::response::Response {
        StatusCode::OK.into_response()
    }
}

// BEFORE OPTIMIZATION
fn current_logic(
    routers: &DashMap<RouterId, Arc<dyn RouterTrait>>,
) -> Option<Arc<dyn RouterTrait>> {
    // This represents the per-request heap allocation in the current code
    let candidate_routers: Vec<Arc<dyn RouterTrait>> = routers
        .iter()
        .map(|entry| Arc::clone(entry.value()))
        .collect::<Vec<_>>();

    let mut best_router = None;
    let mut best_score = 0.0;
    for router in candidate_routers {
        let score = if router.is_pd_mode() { 2.0 } else { 1.0 };
        if score > best_score {
            best_score = score;
            best_router = Some(router);
        }
    }
    best_router
}

//  AFTER OPTIMIZATION (Snapshot)
fn snapshot_logic(snapshot: &ArcSwap<Vec<Arc<dyn RouterTrait>>>) -> Option<Arc<dyn RouterTrait>> {
    // Atomic load: Zero allocation, lock-free
    let routers = snapshot.load();

    let mut best_router = None;
    let mut best_score = 0.0;
    for router in routers.iter() {
        let score = if router.is_pd_mode() { 2.0 } else { 1.0 };
        if score > best_score {
            best_score = score;
            best_router = Some(Arc::clone(router));
        }
    }
    best_router
}

//  BENCHMARK RUNNER
fn bench_router_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Router Selection Improvement");

    for size in [2, 10, 100] {
        // Setup mock data
        let routers_map = DashMap::new();
        let mut routers_vec = Vec::new();
        for i in 0..size {
            let id = RouterId::new(format!("router-{}", i));
            let router = Arc::new(MockRouter { is_pd: i % 2 == 0 }) as Arc<dyn RouterTrait>;
            routers_map.insert(id, router.clone());
            routers_vec.push(router);
        }
        let snapshot = ArcSwap::from_pointee(routers_vec);

        // Run "Before"
        group.bench_with_input(
            BenchmarkId::new("Before (Allocating)", size),
            &size,
            |b, _| {
                b.iter(|| black_box(current_logic(&routers_map)));
            },
        );

        // Run "After"
        group.bench_with_input(BenchmarkId::new("After (Snapshot)", size), &size, |b, _| {
            b.iter(|| black_box(snapshot_logic(&snapshot)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_router_optimization);
criterion_main!(benches);
