//! HA management endpoints
//!
//! Provides REST API for HA cluster management:
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
use tracing::{debug, info, warn};

use super::{
    service::HAServerHandler,
    stores::{MembershipState, PolicyState, StoreType, WorkerState},
    sync::HASyncManager,
};

/// HA cluster status response
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
pub struct HAHealthResponse {
    pub status: String,
    pub node_name: String,
    pub cluster_size: usize,
    pub stores_healthy: bool,
}

/// Get HA cluster status
pub async fn get_cluster_status(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.ha_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "HA not enabled"})),
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

/// Get HA health status
pub async fn get_ha_health(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.ha_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "HA not enabled"})),
            )
                .into_response();
        }
    };
    let state = handler.state.read();
    let cluster_size = state.len();

    let response = HAHealthResponse {
        status: "healthy".to_string(),
        node_name: handler.self_name.clone(),
        cluster_size,
        stores_healthy: true, // TODO: Check actual store health
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// Get worker states from HA store
pub async fn get_worker_states(State(app_state): State<Arc<AppState>>) -> Response {
    match &app_state.ha_sync_manager {
        Some(manager) => {
            let workers = manager.get_all_worker_states();
            (StatusCode::OK, Json(workers)).into_response()
        }
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "HA sync manager not available"})),
        )
            .into_response(),
    }
}

/// Get policy states from HA store
pub async fn get_policy_states(State(app_state): State<Arc<AppState>>) -> Response {
    match &app_state.ha_sync_manager {
        Some(manager) => {
            let policies = manager.get_all_policy_states();
            (StatusCode::OK, Json(policies)).into_response()
        }
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "HA sync manager not available"})),
        )
            .into_response(),
    }
}

/// Get a specific worker state
pub async fn get_worker_state(
    Path(worker_id): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    match &app_state.ha_sync_manager {
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
            Json(json!({"error": "HA sync manager not available"})),
        )
            .into_response(),
    }
}

/// Get a specific policy state
pub async fn get_policy_state(
    Path(model_id): Path<String>,
    State(app_state): State<Arc<AppState>>,
) -> Response {
    match &app_state.ha_sync_manager {
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
            Json(json!({"error": "HA sync manager not available"})),
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
    let handler = match &app_state.ha_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "HA not enabled"})),
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
    let handler = match &app_state.ha_handler {
        Some(h) => h,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "HA not enabled"})),
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

/// Trigger graceful shutdown
pub async fn trigger_graceful_shutdown(State(app_state): State<Arc<AppState>>) -> Response {
    let handler = match &app_state.ha_handler {
        Some(h) => h.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "HA not enabled"})),
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

use crate::server::AppState;
