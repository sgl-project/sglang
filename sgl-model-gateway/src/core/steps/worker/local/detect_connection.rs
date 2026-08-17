//! Connection mode detection step.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, Url};
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::strip_protocol;
use crate::{
    core::{steps::workflow_data::LocalWorkerWorkflowData, ConnectionMode},
    routers::grpc::client::GrpcClient,
};

/// Try HTTP health check.
async fn try_http_health_check(
    url: &str,
    timeout_secs: u64,
    endpoint: &str,
    client: &Client,
) -> Result<(), String> {
    let base_url = if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("http://{}", strip_protocol(url))
    };
    let health_url = Url::parse(&base_url)
        .and_then(|base| base.join(endpoint))
        .map_err(|e| format!("Invalid health check URL: {}", e))?;

    client
        .get(health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {}", e))?;

    Ok(())
}

/// Perform gRPC health check with runtime type.
async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<(), String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {}", e))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {}", e))?;

    Ok(())
}

/// Try gRPC health check (tries SGLang first, then vLLM if not specified).
async fn try_grpc_health_check(
    url: &str,
    timeout_secs: u64,
    runtime_type: Option<&str>,
) -> Result<(), String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    match runtime_type {
        Some(runtime) => do_grpc_health_check(&grpc_url, timeout_secs, runtime).await,
        None => {
            // Try SGLang first, then vLLM as fallback
            if let Ok(()) = do_grpc_health_check(&grpc_url, timeout_secs, "sglang").await {
                return Ok(());
            }
            do_grpc_health_check(&grpc_url, timeout_secs, "vllm")
                .await
                .map_err(|e| format!("gRPC failed (tried SGLang and vLLM): {}", e))
        }
    }
}

/// Step 1: Detect connection mode by probing HTTP and gRPC.
pub struct DetectConnectionModeStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for DetectConnectionModeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        debug!(
            "Detecting connection mode for {} (timeout: {}s, max_attempts: {})",
            config.url, config.health_check_timeout_secs, config.max_connection_attempts
        );

        // Try both protocols in parallel
        let url = config.url.clone();
        let timeout = config.health_check_timeout_secs;
        let client = &app_context.client;
        let endpoint = &app_context.router_config.health_check.endpoint;
        let runtime_type = config.runtime.as_deref();

        let (http_result, grpc_result) = tokio::join!(
            try_http_health_check(&url, timeout, endpoint, client),
            try_grpc_health_check(&url, timeout, runtime_type)
        );

        let connection_mode = match (http_result, grpc_result) {
            (Ok(_), _) => {
                debug!("{} detected as HTTP", config.url);
                ConnectionMode::Http
            }
            (_, Ok(_)) => {
                debug!("{} detected as gRPC", config.url);
                ConnectionMode::Grpc { port: None }
            }
            (Err(http_err), Err(grpc_err)) => {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("detect_connection_mode"),
                    message: format!(
                        "Both HTTP and gRPC health checks failed for {}: HTTP: {}, gRPC: {}",
                        config.url, http_err, grpc_err
                    ),
                });
            }
        };

        context.data.connection_mode = Some(connection_mode);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use axum::{routing::get, Router};
    use tokio::{net::TcpListener, sync::oneshot};
    use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

    use super::DetectConnectionModeStep;
    use crate::{
        app_context::AppContext,
        config::{HealthCheckConfig, RouterConfig},
        core::{
            steps::{create_local_worker_workflow_data, workflow_data::WorkerConfigRequest},
            ConnectionMode,
        },
    };

    #[tokio::test]
    async fn test_detect_connection_uses_configured_health_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route("/custom-health", get(|| async { "ok" }));

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });

        let client = reqwest::Client::new();
        let health_url = format!("http://{}/custom-health", addr);
        for _ in 0..10 {
            if let Ok(response) = client.get(&health_url).send().await {
                if response.status().is_success() {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let response = client.get(&health_url).send().await.unwrap();
        assert!(response.status().is_success(), "server did not start in time");

        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .health_check_config(HealthCheckConfig {
                endpoint: "/custom-health".to_string(),
                ..Default::default()
            })
            .build_unchecked();
        let app_context = Arc::new(AppContext::from_config(config, 30).await.unwrap());

        let worker_config = WorkerConfigRequest {
            url: format!("http://{}", addr),
            api_key: None,
            worker_type: Some("regular".to_string()),
            labels: HashMap::new(),
            model_id: None,
            priority: None,
            cost: None,
            runtime: None,
            tokenizer_path: None,
            reasoning_parser: None,
            tool_parser: None,
            chat_template: None,
            bootstrap_port: None,
            health_check_timeout_secs: 1,
            health_check_interval_secs: 1,
            health_success_threshold: 1,
            health_failure_threshold: 1,
            disable_health_check: false,
            max_connection_attempts: 1,
            dp_aware: false,
        };

        let workflow_data = create_local_worker_workflow_data(worker_config, app_context);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), workflow_data);

        let result = DetectConnectionModeStep.execute(&mut context).await;

        let _ = shutdown_tx.send(());
        let _ = server.await;

        assert!(result.is_ok());
        assert_eq!(context.data.connection_mode, Some(ConnectionMode::Http));
    }
}
