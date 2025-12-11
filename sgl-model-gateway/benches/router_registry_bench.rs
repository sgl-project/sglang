use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use std::collections::HashMap;
use sgl_model_gateway::core::{WorkerRegistry, BasicWorkerBuilder, WorkerType, CircuitBreakerConfig};
// Fix 1: Correct import paths for RouterManager and RouterId
use sgl_model_gateway::routers::router_manager::{RouterManager, RouterId};
use sgl_model_gateway::routers::RouterTrait;
// Fix 2: Import GenerateRequest from its submodule
use sgl_model_gateway::protocols::generate::GenerateRequest;
use axum::http::HeaderMap;
use axum::extract::Request;
use axum::response::Response;
use async_trait::async_trait;

// --- Mocks for Benchmark ---

// Fix 3: Add #[derive(Debug)] as required by RouterTrait
#[derive(Debug)]
struct MockRouter;

#[async_trait]
impl RouterTrait for MockRouter {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn router_type(&self) -> &'static str { "mock" }

    // Implement minimal stubs for required methods
    async fn health_generate(&self, _req: Request<axum::body::Body>) -> Response { unimplemented!() }
    async fn get_server_info(&self, _req: Request<axum::body::Body>) -> Response { unimplemented!() }
    async fn get_models(&self, _req: Request<axum::body::Body>) -> Response { unimplemented!() }
    async fn get_model_info(&self, _req: Request<axum::body::Body>) -> Response { unimplemented!() }
    async fn route_generate(&self, _h: Option<&HeaderMap>, _b: &GenerateRequest, _m: Option<&str>) -> Response { unimplemented!() }
    async fn route_chat(&self, _h: Option<&HeaderMap>, _b: &sgl_model_gateway::protocols::chat::ChatCompletionRequest, _m: Option<&str>) -> Response { unimplemented!() }
    async fn route_completion(&self, _h: Option<&HeaderMap>, _b: &sgl_model_gateway::protocols::completion::CompletionRequest, _m: Option<&str>) -> Response { unimplemented!() }
    async fn route_responses(&self, _h: Option<&HeaderMap>, _b: &sgl_model_gateway::protocols::responses::ResponsesRequest, _m: Option<&str>) -> Response { unimplemented!() }
    async fn get_response(&self, _h: Option<&HeaderMap>, _id: &str, _p: &sgl_model_gateway::protocols::responses::ResponsesGetParams) -> Response { unimplemented!() }
    async fn cancel_response(&self, _h: Option<&HeaderMap>, _id: &str) -> Response { unimplemented!() }
    async fn delete_response(&self, _h: Option<&HeaderMap>, _id: &str) -> Response { unimplemented!() }
    async fn list_response_input_items(&self, _h: Option<&HeaderMap>, _id: &str) -> Response { unimplemented!() }
    async fn route_embeddings(&self, _h: Option<&HeaderMap>, _b: &sgl_model_gateway::protocols::embedding::EmbeddingRequest, _m: Option<&str>) -> Response { unimplemented!() }
    async fn route_classify(&self, _h: Option<&HeaderMap>, _b: &sgl_model_gateway::protocols::classify::ClassifyRequest, _m: Option<&str>) -> Response { unimplemented!() }
    async fn route_rerank(&self, _h: Option<&HeaderMap>, _b: &sgl_model_gateway::protocols::rerank::RerankRequest, _m: Option<&str>) -> Response { unimplemented!() }
    // Add is_pd_mode implementation to the trait block if required, otherwise it might be a separate impl
}

// Separate impl block for methods not in the trait (if any specific ones are called)
impl MockRouter {
    fn is_pd_mode(&self) -> bool { false }
}

// Helper to populate registry
fn setup_registry(count: usize) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());

    for i in 0..count {
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "benchmark-model".to_string());

        let worker_type = if i % 2 == 0 { WorkerType::Regular } else { WorkerType::Decode };

        let worker = BasicWorkerBuilder::new(&format!("http://worker-{}:8000", i))
            .worker_type(worker_type)
            .labels(labels)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build();

        registry.register(Arc::from(worker));
    }
    registry
}

fn bench_worker_registry(c: &mut Criterion) {
    let mut group = c.benchmark_group("WorkerRegistry Allocations");

    // Benchmark 1: Direct cost of get_all()
    for size in [100, 1000, 5000].iter() {
        let registry = setup_registry(*size);

        group.bench_with_input(BenchmarkId::new("get_all", size), size, |b, &_s| {
            b.iter(|| {
                black_box(registry.get_all());
            });
        });
    }
    group.finish();
}

fn bench_router_manager(c: &mut Criterion) {
    let mut group = c.benchmark_group("RouterManager Hot Path");

    // Benchmark 2: Full path impact on request routing
    for size in [100, 1000, 5000].iter() {
        let registry = setup_registry(*size);
        let manager = RouterManager::new(registry.clone());

        manager.register_router(
            RouterId::new("http-regular".to_string()),
            Arc::new(MockRouter)
        );

        group.bench_with_input(BenchmarkId::new("select_router", size), size, |b, &_s| {
            b.iter(|| {
                let _ = black_box(manager.select_router_for_request(None, None));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_worker_registry, bench_router_manager);
criterion_main!(benches);
