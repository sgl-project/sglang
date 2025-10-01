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
        WasmModuleAddRequest, WasmModuleAddResponse, WasmModuleAddResult, WasmModuleListResponse,
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
            module.add_result = Some(WasmModuleAddResult::Error(
                result.err().unwrap().to_string(),
            ));
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
        let response = WasmModuleListResponse { modules };
        (StatusCode::OK, Json(response)).into_response()
    } else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
}
