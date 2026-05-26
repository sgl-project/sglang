//! Worker metadata discovery integration tests.

use smg::{config::RouterConfig, core::Job};

use crate::common::{
    create_test_context,
    mock_worker::{HealthStatus, MockWorkerConfig, OpenAiOnlyMockWorker, WorkerType},
    AppTestContext,
};

#[cfg(test)]
mod worker_discovery_tests {
    use super::*;

    /// Normal path: model name is discovered from /server_info.
    #[tokio::test]
    async fn test_model_name_discovered_via_server_info() {
        let ctx = AppTestContext::new(vec![MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let discovered_models = ctx.app_context.worker_registry.get_models();
        assert!(
            discovered_models.contains(&"mock-model-path".to_string()),
            "Expected 'mock-model-path' discovered via /server_info, got: {:?}",
            discovered_models
        );

        ctx.shutdown().await;
    }

    /// Fallback path: when /server_info is unavailable, model name is discovered via /v1/models.
    #[tokio::test]
    async fn test_model_name_discovered_via_v1_models_fallback() {
        let mut worker = OpenAiOnlyMockWorker::new("my-model");
        let url = worker.start().await.unwrap();

        let config = RouterConfig::builder()
            .regular_mode(vec![url.clone()])
            .random_policy()
            .host("127.0.0.1")
            .port(0)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let app_context = create_test_context(config.clone()).await;

        let job_queue = app_context
            .worker_job_queue
            .get()
            .expect("JobQueue should be initialized");
        job_queue
            .submit(Job::InitializeWorkersFromConfig {
                router_config: Box::new(config),
            })
            .await
            .expect("Failed to submit worker initialization job");

        let start = tokio::time::Instant::now();
        loop {
            if app_context
                .worker_registry
                .get_all()
                .iter()
                .any(|w| w.is_healthy())
            {
                break;
            }
            if start.elapsed().as_secs() > 10 {
                panic!("Timeout waiting for worker to become healthy");
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        let discovered_models = app_context.worker_registry.get_models();
        assert!(
            discovered_models.contains(&"my-model".to_string()),
            "Expected 'my-model' discovered via /v1/models fallback, got: {:?}",
            discovered_models
        );

        worker.stop().await;
    }
}
