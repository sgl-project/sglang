//! Worker metadata discovery integration tests.

use smg::{config::RouterConfig, core::Job};

use crate::common::{
    create_test_context,
    mock_worker::{HealthStatus, MockWorkerConfig, OpenAiOnlyMockWorker, WorkerType},
    AppTestContext,
};

#[cfg(test)]
mod worker_discovery_tests {
    use axum::{
        http::{HeaderMap, StatusCode},
        routing::get,
        Router,
    };
    use smg::{
        core::{
            steps::{create_local_worker_workflow_data, DetectConnectionModeStep},
            ConnectionMode,
        },
        protocols::worker_spec::WorkerConfigRequest,
    };
    use tokio::net::TcpListener;
    use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowInstanceId};

    use super::*;

    async fn detect_worker_connection(
        required_key: Option<&str>,
        api_key: Option<&str>,
    ) -> (Result<StepResult, WorkflowError>, Option<ConnectionMode>) {
        let expected_auth = required_key.map(|key| format!("Bearer {key}"));
        let app = Router::new().route(
            "/health",
            get(move |headers: HeaderMap| {
                let expected_auth = expected_auth.clone();
                async move {
                    let auth = headers.get("authorization").and_then(|v| v.to_str().ok());
                    if auth == expected_auth.as_deref() {
                        StatusCode::OK
                    } else {
                        StatusCode::UNAUTHORIZED
                    }
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let router_config = RouterConfig::builder()
            .regular_mode(vec![])
            .api_key("router-key")
            .build_unchecked();
        let app_context = create_test_context(router_config).await;
        let config: WorkerConfigRequest = serde_json::from_value(serde_json::json!({
            "url": url,
            "api_key": api_key,
            "health_check_timeout_secs": 1,
        }))
        .unwrap();
        let data = create_local_worker_workflow_data(config, app_context);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        let result = DetectConnectionModeStep.execute(&mut context).await;

        server.abort();
        let _ = server.await;
        (result, context.data.connection_mode)
    }

    /// A worker protecting /health must be detected as HTTP with its own key;
    /// missing or incorrect credentials must leave connection detection failed.
    #[tokio::test]
    async fn test_connection_detection_with_worker_api_key() {
        for key in [Some("worker-key"), None, Some("wrong-key")] {
            let (result, mode) = detect_worker_connection(Some("worker-key"), key).await;
            if key == Some("worker-key") {
                assert_eq!(result.unwrap(), StepResult::Success);
                assert_eq!(mode, Some(ConnectionMode::Http));
            } else {
                let WorkflowError::StepFailed { message, .. } = result.unwrap_err() else {
                    panic!("Expected a connection detection failure");
                };
                assert!(message.contains("401"), "{message}");
                assert_eq!(mode, None);
            }
        }
    }

    /// A worker without a key must remain reachable without inheriting the router key.
    #[tokio::test]
    async fn test_connection_detection_without_worker_api_key() {
        let (result, mode) = detect_worker_connection(None, None).await;
        assert_eq!(result.unwrap(), StepResult::Success);
        assert_eq!(mode, Some(ConnectionMode::Http));
    }

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
