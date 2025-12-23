use std::sync::Arc;

use async_trait::async_trait;
use ax_http::HeaderMap; // Ensure this matches your project's http/header crate
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dashmap::DashMap;
use sgl_model_gateway::{
    core::{ConnectionMode, RuntimeType, Worker, WorkerRegistry, WorkerType},
    routers::{RouterId, RouterTrait},
};

// --- MOCK OBJECTS ---

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
    // Mandatory trait method
    async fn route_chat(
        &self,
        _: Option<&HeaderMap>,
        _: &sgl_model_gateway::protocols::chat::ChatCompletionRequest,
        _: Option<&str>,
    ) -> axum::response::Response {
        todo!()
    }
}

// --- LOGIC PATHS ---

/// Current implementation with Vec allocation
fn current_logic(
    routers: &DashMap<RouterId, Arc<dyn RouterTrait>>,
    num_regular: usize,
    num_pd: usize,
) -> Option<Arc<dyn RouterTrait>> {
    // This simulates the collect() call in the current codebase
    let candidate_routers = routers
        .iter()
        .map(|entry| entry.value().clone())
        .collect::<Vec<_>>();

    let mut best_router = None;
    let mut best_score = 0.0;

    for router in candidate_routers {
        let mut score = 1.0;
        let is_pd = router.is_pd_mode();

        // Simplified scoring
        if is_pd {
            score += 1.0;
        }

        let valid_router = (is_pd && num_pd > 0) || (!is_pd && num_regular > 0);
        if score > best_score && valid_router {
            best_score = score;
            best_router = Some(router);
        }
    }
    best_router
}

/// Fixed implementation using Single-Pass Iterator
fn fixed_logic(
    routers: &DashMap<RouterId, Arc<dyn RouterTrait>>,
    num_regular: usize,
    num_pd: usize,
) -> Option<Arc<dyn RouterTrait>> {
    let mut best_router = None;
    let mut best_score = 0.0;

    // Zero-allocation: iterate directly
    for entry in routers.iter() {
        let router = entry.value();
        let mut score = 1.0;
        let is_pd = router.is_pd_mode();

        if is_pd {
            score += 1.0;
        }

        let valid_router = (is_pd && num_pd > 0) || (!is_pd && num_regular > 0);
        if score > best_score && valid_router {
            best_score = score;
            best_router = Some(Arc::clone(router));
        }
    }
    best_router
}

// --- BENCHMARK RUNNER ---

fn bench_routers(c: &mut Criterion) {
    let mut group = c.benchmark_group("Router Selection Scaling");

    for size in [2, 10, 100, 500].iter() {
        // Setup: Create a map with 'size' routers
        let routers = DashMap::new();
        for i in 0..*size {
            let id = RouterId::new(format!("router-{}", i));
            routers.insert(
                id,
                Arc::new(MockRouter { is_pd: i % 2 == 0 }) as Arc<dyn RouterTrait>,
            );
        }
        let routers = Arc::new(routers);

        group.bench_with_input(
            BenchmarkId::new("Current (Allocating)", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(current_logic(&routers, 5, 5));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Fixed (Zero-Alloc)", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(fixed_logic(&routers, 5, 5));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_routers);
criterion_main!(benches);
