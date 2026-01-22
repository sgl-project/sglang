use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smg::core::{
    model_type::ModelType,
    worker::{WorkerMetadata, WorkerRoutingKeyLoad},
    worker_registry::HashRing,
    CircuitBreaker, ConnectionMode, HealthConfig, RuntimeType, Worker, WorkerResult, WorkerType,
};

// A minimal mock worker for benchmarking to avoid complex builder overhead
#[derive(Debug)]
struct MockWorker {
    url: String,
    metadata: WorkerMetadata,
}

#[async_trait::async_trait]
impl Worker for MockWorker {
    fn url(&self) -> &str {
        &self.url
    }
    fn api_key(&self) -> &Option<String> {
        &None
    }
    fn worker_type(&self) -> &WorkerType {
        &self.metadata.worker_type
    }
    fn connection_mode(&self) -> &ConnectionMode {
        &self.metadata.connection_mode
    }
    fn is_healthy(&self) -> bool {
        true
    }
    fn set_healthy(&self, _: bool) {}
    async fn check_health_async(&self) -> WorkerResult<()> {
        Ok(())
    }
    fn load(&self) -> usize {
        0
    }
    fn increment_load(&self) {}
    fn decrement_load(&self) {}
    fn worker_routing_key_load(&self) -> &WorkerRoutingKeyLoad {
        unimplemented!()
    }
    fn processed_requests(&self) -> usize {
        0
    }
    fn increment_processed(&self) {}
    fn metadata(&self) -> &WorkerMetadata {
        &self.metadata
    }
    fn circuit_breaker(&self) -> &CircuitBreaker {
        unimplemented!()
    }
    async fn get_grpc_client(
        &self,
    ) -> WorkerResult<Option<Arc<smg::routers::grpc::client::GrpcClient>>> {
        Ok(None)
    }
    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        Ok(true)
    }
    async fn http_health_check(&self) -> WorkerResult<bool> {
        Ok(true)
    }
}

fn setup_workers(count: usize) -> Vec<Arc<dyn Worker>> {
    (0..count)
        .map(|i| {
            let url = format!("http://worker-{}.internal:8080", i);
            let metadata = WorkerMetadata {
                url: url.clone(),
                worker_type: WorkerType::Regular,
                connection_mode: ConnectionMode::Http,
                runtime_type: RuntimeType::Sglang,
                labels: std::collections::HashMap::new(),
                health_config: HealthConfig::default(),
                api_key: None,
                bootstrap_host: "localhost".to_string(),
                bootstrap_port: None,
                models: vec![],
                default_provider: None,
                default_model_type: ModelType::LLM,
            };
            Arc::new(MockWorker { url, metadata }) as Arc<dyn Worker>
        })
        .collect()
}

fn bench_hashring_construction(c: &mut Criterion) {
    let workers = setup_workers(1000);

    c.bench_function("hashring_new_1000_workers", |b| {
        b.iter(|| {
            let ring = HashRing::new(black_box(&workers));
            black_box(ring);
        })
    });
}

criterion_group!(benches, bench_hashring_construction);
criterion_main!(benches);
