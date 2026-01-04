//! Mesh management endpoints
//!
//! Provides REST API for mesh cluster management:
//! - Configuration CRUD operations
//! - Health checks
//! - Cluster status

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn};

/// Mesh cluster status response
#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterStatusResponse {
    pub node_name: String,
    pub node_count: usize,
    pub nodes: Vec<NodeInfo>,
    pub stores: StoreStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeInfo {
    pub name: String,
    pub address: String,
    pub status: String,
    pub version: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoreStatus {
    pub membership_count: usize,
    pub worker_count: usize,
    pub policy_count: usize,
    pub app_count: usize,
}

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct MeshHealthResponse {
    pub status: String,
    pub node_name: String,
    pub cluster_size: usize,
    pub stores_healthy: bool,
}

/// Get mesh cluster status
pub async fn get_cluster_status(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };
    let state = handler.state.read();
    let nodes: Vec<NodeInfo> = state
        .values()
        .map(|node| NodeInfo {
            name: node.name.clone(),
            address: node.address.clone(),
            status: format!("{:?}", node.status),
            version: node.version,
        })
        .collect();

    // Get store counts (if stores are available)
    let stores = StoreStatus {
        membership_count: state.len(),
        worker_count: 0, // TODO: Get from stores if available
        policy_count: 0,
        app_count: 0,
    };

    let response = ClusterStatusResponse {
        node_name: handler.self_name.clone(),
        node_count: nodes.len(),
        nodes,
        stores,
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// Get mesh health status
pub async fn get_mesh_health(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };
    let state = handler.state.read();
    let cluster_size = state.len();

    let response = MeshHealthResponse {
        status: "healthy".to_string(),
        node_name: handler.self_name.clone(),
        cluster_size,
        stores_healthy: true, // TODO: Check actual store health
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// Get worker states from mesh store
pub async fn get_worker_states(State(app_state): State<Arc<AppState>>) -> Response {
    match &app_state.mesh_sync_manager {
        Some(manager) => {
            let workers = manager.get_all_worker_states();
            (StatusCode::OK, Json(workers)).into_response()
        }
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "mesh sync manager not available"})),
        )
            .into_response(),
    }
}

/// Get policy states from mesh store
pub async fn get_policy_states(State(app_state): State<Arc<AppState>>) -> Response {
    match &app_state.mesh_sync_manager {
        Some(manager) => {
            let policies = manager.get_all_policy_states();
            (StatusCode::OK, Json(policies)).into_response()
        }
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "mesh sync manager not available"})),
        )
            .into_response(),
    }
}

/// Get a specific worker state
pub async fn get_worker_state(
    Path(worker_id): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    match &app_state.mesh_sync_manager {
        Some(manager) => match manager.get_worker_state(&worker_id) {
            Some(state) => (StatusCode::OK, Json(state)).into_response(),
            None => (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Worker not found"})),
            )
                .into_response(),
        },
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "mesh sync manager not available"})),
        )
            .into_response(),
    }
}

/// Get a specific policy state
pub async fn get_policy_state(
    Path(model_id): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    match &app_state.mesh_sync_manager {
        Some(manager) => match manager.get_policy_state(&model_id) {
            Some(state) => (StatusCode::OK, Json(state)).into_response(),
            None => (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Policy not found"})),
            )
                .into_response(),
        },
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "mesh sync manager not available"})),
        )
            .into_response(),
    }
}

/// Update app configuration
#[derive(Debug, Deserialize)]
pub struct UpdateAppConfigRequest {
    pub key: String,
    pub value: String, // Hex encoded string
}

pub async fn update_app_config(
    State(app_state): State<Arc<AppState>>,
    Json(request): Json<UpdateAppConfigRequest>,
) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };

    // Decode hex string to bytes
    // Simple hex decoding without external dependency
    let value = if request.value.len() % 2 == 0 {
        match (0..request.value.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&request.value[i..i + 2], 16))
            .collect::<Result<Vec<u8>, _>>()
        {
            Ok(v) => v,
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid hex encoding"})),
                )
                    .into_response();
            }
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Hex string must have even length"})),
        )
            .into_response();
    };
    handler.write_data(request.key.clone(), value);
    info!("Updated app config: {}", request.key);
    (
        StatusCode::OK,
        Json(json!({"status": "updated", "key": request.key})),
    )
        .into_response()
}

/// Get app configuration
pub async fn get_app_config(
    Path(key): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };
    match handler.read_data(key.clone()) {
        Some(value) => {
            // Return value as hex encoded string for JSON compatibility
            let hex_value: String = value.iter().map(|b| format!("{:02x}", b)).collect();
            (
                StatusCode::OK,
                Json(json!({"key": key, "value": hex_value, "format": "hex"})),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Config not found"})),
        )
            .into_response(),
    }
}

/// Set global rate limit configuration
#[derive(Debug, Deserialize)]
pub struct SetRateLimitRequest {
    pub limit_per_second: u64,
}

pub async fn set_global_rate_limit(
    State(app_state): State<Arc<AppState>>,
    Json(request): Json<SetRateLimitRequest>,
) -> Response {
    // Store configuration in AppStore
    let config = RateLimitConfig {
        limit_per_second: request.limit_per_second,
    };

    if let Ok(config_bytes) = serde_json::to_vec(&config) {
        let handler = match &app_state.mesh_handler {
            Some(h) => h,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"error": "mesh not enabled"})),
                )
                    .into_response();
            }
        };

        handler.write_data(GLOBAL_RATE_LIMIT_KEY.to_string(), config_bytes);
        info!("Set global rate limit: {} req/s", request.limit_per_second);

        (
            StatusCode::OK,
            Json(json!({
                "status": "updated",
                "limit_per_second": request.limit_per_second
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to serialize rate limit config"})),
        )
            .into_response()
    }
}

/// Get global rate limit configuration
pub async fn get_global_rate_limit(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };

    match handler.read_data(GLOBAL_RATE_LIMIT_KEY.to_string()) {
        Some(value) => match serde_json::from_slice::<RateLimitConfig>(&value) {
            Ok(config) => (
                StatusCode::OK,
                Json(json!({
                    "limit_per_second": config.limit_per_second
                })),
            )
                .into_response(),
            Err(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Failed to deserialize rate limit config"})),
            )
                .into_response(),
        },
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Global rate limit not configured"})),
        )
            .into_response(),
    }
}

/// Get global rate limit statistics
pub async fn get_global_rate_limit_stats(State(app_state): State<Arc<AppState>>) -> Response {
    let sync_manager = match &app_state.mesh_sync_manager {
        Some(m) => m,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh sync manager not available"})),
            )
                .into_response();
        }
    };

    // Get configuration
    let handler = match &app_state.mesh_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };

    let config = handler
        .read_data(GLOBAL_RATE_LIMIT_KEY.to_string())
        .and_then(|v| serde_json::from_slice::<RateLimitConfig>(&v).ok())
        .unwrap_or_default();

    // Get current counter value
    let current_count = sync_manager
        .get_rate_limit_value(crate::mesh::stores::GLOBAL_RATE_LIMIT_COUNTER_KEY)
        .unwrap_or(0);

    (
        StatusCode::OK,
        Json(json!({
            "limit_per_second": config.limit_per_second,
            "current_count": current_count,
            "remaining": if config.limit_per_second > 0 {
                (config.limit_per_second as i64).saturating_sub(current_count).max(0)
            } else {
                -1 // Unlimited
            }
        })),
    )
        .into_response()
}

/// Trigger graceful shutdown
pub async fn trigger_graceful_shutdown(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.mesh_handler {
        Some(h) => h.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "mesh not enabled"})),
            )
                .into_response();
        }
    };
    info!("Graceful shutdown triggered via API");
    tokio::spawn(async move {
        if let Err(e) = handler.graceful_shutdown().await {
            warn!("Error during graceful shutdown: {}", e);
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(json!({"status": "shutdown initiated"})),
    )
        .into_response()
}

use std::sync::Arc;

use crate::{
    mesh::stores::{RateLimitConfig, GLOBAL_RATE_LIMIT_KEY},
    server::AppState,
};
