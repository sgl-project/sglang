//! WASM HTTP API Routes
//!
//! Provides REST API endpoints for managing WASM modules:
//! - POST /wasm - Add modules
//! - DELETE /wasm/:uuid - Remove a module
//! - GET /wasm - List all modules with metrics

use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use uuid::Uuid;

use crate::{
    server::AppState,
    wasm::module::{
        WasmMetrics, WasmModuleAddRequest, WasmModuleAddResponse, WasmModuleAddResult,
        WasmModuleListResponse,
    },
};

pub async fn add_wasm_module(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WasmModuleAddRequest>,
) -> Response {
    let Some(wasm_manager) = state.context.wasm_manager.as_ref() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let mut status = StatusCode::OK;
    let mut modules = config.modules.clone();
    for module in modules.iter_mut() {
        let result = wasm_manager.add_module(module.clone());
        if let Ok(module_uuid) = result {
            module.add_result = Some(WasmModuleAddResult::Success(module_uuid));
        } else {
            // We know result is Err here, so unwrap_err() is safe
            module.add_result = Some(WasmModuleAddResult::Error(result.unwrap_err().to_string()));
            status = StatusCode::BAD_REQUEST;
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
    let Some(wasm_manager) = state.context.wasm_manager.as_ref() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    if let Err(e) = wasm_manager.remove_module(module_uuid) {
        return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
    }
    (StatusCode::OK, "Module removed successfully").into_response()
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
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
}
