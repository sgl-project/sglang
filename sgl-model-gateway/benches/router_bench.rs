use std::sync::Arc;

use async_trait::async_trait;
// FIX: Correct imports for StatusCode and IntoResponse trait
use axum::{
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dashmap::DashMap;
use sgl_model_gateway::routers::{router_manager::RouterId, RouterTrait};

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

    // Implement the required trait method with the correct signature
    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &sgl_model_gateway::protocols::chat::ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> axum::response::Response {
        // FIX: StatusCode::OK now works because the trait IntoResponse is in scope
        StatusCode::OK.into_response()
    }
}

// --- LOGIC PATHS ---

/// Current implementation with Vec allocation
fn current_logic(
    routers: &DashMap<RouterId, Arc<dyn RouterTrait>>,
    num_regular: usize,
    num_pd: usize,
) -> Option<Arc<dyn RouterTrait>> {
    // Explicit type to help compiler inference
    let candidate_routers: Vec<Arc<dyn RouterTrait>> = routers
        .iter()
        .map(|entry| Arc::clone(entry.value()))
        .collect::<Vec<_>>();

    let mut best_router = None;
    let mut best_score = 0.0;

    for router in candidate_routers {
        let mut score = 1.0;
        let is_pd = router.is_pd_mode();

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

/// Fixed implementation using Single-Pass Iterator (Zero Allocation)
fn fixed_logic(
    routers: &DashMap<RouterId, Arc<dyn RouterTrait>>,
    num_regular: usize,
    num_pd: usize,
) -> Option<Arc<dyn RouterTrait>> {
    let mut best_router: Option<Arc<dyn RouterTrait>> = None;
    let mut best_score = 0.0;

    // Zero-allocation: iterate directly over the DashMap
    for entry in routers.iter() {
        let router: &Arc<dyn RouterTrait> = entry.value();
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

    // Test scaling from 2 to 100 routers
    for size in [2, 10, 100].iter() {
        let routers = DashMap::new();
        for i in 0..*size {
            let id = RouterId::new(format!("router-{}", i));
            routers.insert(
                id,
                Arc::new(MockRouter { is_pd: i % 2 == 0 }) as Arc<dyn RouterTrait>,
            );
        }
        let routers_ref = &routers;

        group.bench_with_input(
            BenchmarkId::new("Current (Allocating)", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(current_logic(routers_ref, 5, 5));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Fixed (Zero-Alloc)", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(fixed_logic(routers_ref, 5, 5));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_routers);
criterion_main!(benches);
