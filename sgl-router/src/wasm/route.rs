//! WASM HTTP API Routes
//!
//! Provides REST API endpoints for managing WASM modules:
//! - POST /wasm - Add modules
//! - DELETE /wasm/:uuid - Remove a module
//! - GET /wasm - List all modules with metrics

use std::{sync::Arc, time::Duration};

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use uuid::Uuid;

use crate::{
    core::{job_queue::Job, workflow::steps::WasmModuleConfigRequest},
    server::AppState,
    wasm::module::{
        WasmMetrics, WasmModuleAddRequest, WasmModuleAddResponse, WasmModuleAddResult,
        WasmModuleListResponse,
    },
};

/// Wait for job completion by polling job status
/// Returns the job result message if successful
async fn wait_for_job_completion(
    job_queue: &crate::core::job_queue::JobQueue,
    status_key: &str,
    timeout_duration: Duration,
) -> Result<String, String> {
    let start = std::time::Instant::now();
    let mut poll_interval = Duration::from_millis(100);
    let max_poll_interval = Duration::from_millis(2000);
    let poll_backoff = Duration::from_millis(200);

    loop {
        if start.elapsed() > timeout_duration {
            return Err(format!("Job timeout after {}s", timeout_duration.as_secs()));
        }

        if let Some(job_status) = job_queue.get_status(status_key) {
            match job_status.status.as_str() {
                "pending" | "processing" => {
                    tokio::time::sleep(poll_interval).await;
                    poll_interval = (poll_interval + poll_backoff).min(max_poll_interval);
                    continue;
                }
                "failed" => {
                    let error_msg = job_status
                        .message
                        .unwrap_or_else(|| "Unknown error".to_string());
                    job_queue.remove_status(status_key);
                    return Err(error_msg);
                }
                _ => {
                    // Should not happen, but handle gracefully
                    job_queue.remove_status(status_key);
                    return Err("Unexpected job status".to_string());
                }
            }
        } else {
            // Job completed successfully (status was removed by record_job_completion)
            // We need to get the result from the job execution
            // Since job queue removes status on success, we can't get the result here
            // We'll need to query the wasm manager to find the module by name
            // For now, return a success message and let caller extract UUID from manager
            return Ok("Job completed successfully".to_string());
        }
    }
}

pub async fn add_wasm_module(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WasmModuleAddRequest>,
) -> Response {
    let Some(_) = state.context.wasm_manager.as_ref() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };

    let Some(job_queue) = state.context.worker_job_queue.get() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };

    let mut status = StatusCode::OK;
    let mut modules = config.modules.clone();

    for module in modules.iter_mut() {
        let wasm_config = WasmModuleConfigRequest {
            descriptor: module.clone(),
        };

        let job = Job::AddWasmModule {
            config: Box::new(wasm_config),
        };

        let worker_url = job.worker_url().to_string();

        // Submit job to queue
        match job_queue.submit(job).await {
            Ok(_) => {
                // Wait for job completion (timeout: 5 minutes)
                let timeout = Duration::from_secs(300);
                match wait_for_job_completion(job_queue, &worker_url, timeout).await {
                    Ok(_) => {
                        // Job completed successfully, but we need to get the UUID
                        // Since job queue removes status on success, we need to query
                        // the workflow engine or wasm manager to get the UUID
                        // For now, let's try to get it from the wasm manager by name
                        if let Some(wasm_manager) = state.context.wasm_manager.as_ref() {
                            if let Ok(all_modules) = wasm_manager.get_modules() {
                                if let Some(registered_module) = all_modules
                                    .iter()
                                    .find(|m| m.module_meta.name == module.name)
                                {
                                    module.add_result = Some(WasmModuleAddResult::Success(
                                        registered_module.module_uuid,
                                    ));
                                } else {
                                    module.add_result = Some(WasmModuleAddResult::Error(
                                        "Module registered but UUID not found".to_string(),
                                    ));
                                    status = StatusCode::BAD_REQUEST;
                                }
                            } else {
                                module.add_result = Some(WasmModuleAddResult::Error(
                                    "Failed to query registered modules".to_string(),
                                ));
                                status = StatusCode::BAD_REQUEST;
                            }
                        } else {
                            module.add_result = Some(WasmModuleAddResult::Error(
                                "WASM manager not available".to_string(),
                            ));
                            status = StatusCode::BAD_REQUEST;
                        }
                    }
                    Err(e) => {
                        module.add_result = Some(WasmModuleAddResult::Error(e));
                        status = StatusCode::BAD_REQUEST;
                    }
                }
            }
            Err(e) => {
                module.add_result = Some(WasmModuleAddResult::Error(format!(
                    "Failed to submit job: {}",
                    e
                )));
                status = StatusCode::BAD_REQUEST;
            }
        }
    }

    let response = WasmModuleAddResponse { modules };
    (status, Json(response)).into_response()
}

pub async fn remove_wasm_module(
    State(state): State<Arc<AppState>>,
    Path(module_uuid_str): Path<String>,
) -> Response {
    let Ok(module_uuid) = Uuid::parse_str(&module_uuid_str) else {
        return StatusCode::BAD_REQUEST.into_response();
    };

    let Some(_) = state.context.wasm_manager.as_ref() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };

    let Some(job_queue) = state.context.worker_job_queue.get() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };

    use crate::core::workflow::steps::WasmModuleRemovalRequest;

    let removal_request = WasmModuleRemovalRequest::new(module_uuid);

    let job = Job::RemoveWasmModule {
        request: Box::new(removal_request),
    };

    let worker_url = job.worker_url().to_string();

    // Submit job to queue
    match job_queue.submit(job).await {
        Ok(_) => {
            // Wait for job completion (timeout: 1 minute)
            let timeout = Duration::from_secs(60);
            match wait_for_job_completion(job_queue, &worker_url, timeout).await {
                Ok(_) => (StatusCode::OK, "Module removed successfully").into_response(),
                Err(e) => (StatusCode::BAD_REQUEST, e).into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to submit job: {}", e),
        )
            .into_response(),
    }
}

pub async fn list_wasm_modules(State(state): State<Arc<AppState>>) -> Response {
    let Some(wasm_manager) = state.context.wasm_manager.as_ref() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let modules = wasm_manager.get_modules();
    if let Ok(modules) = modules {
        let (total, success, failed, total_time_ms, max_time_ms) = wasm_manager.get_metrics();
        let average_execution_time_ms = if total > 0 {
            Some(total_time_ms as f64 / total as f64)
        } else {
            None
        };
        let metrics = WasmMetrics {
            total_executions: total,
            successful_executions: success,
            failed_executions: failed,
            total_execution_time_ms: total_time_ms,
            max_execution_time_ms: max_time_ms,
            average_execution_time_ms,
        };
        let response = WasmModuleListResponse { modules, metrics };
        (StatusCode::OK, Json(response)).into_response()
    } else {
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    }
}
