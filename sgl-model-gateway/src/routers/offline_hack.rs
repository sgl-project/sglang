use axum::{
    body::Body,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use reqwest::RequestBuilder;
use serde::Deserialize;
use tracing::{error, info};

use crate::core::WorkerLoadGuard;

pub struct OfflineStore {
    results: DashMap<String, OfflineResult>,
}

struct OfflineResult {
    body: Bytes,
    status: StatusCode,
}

pub static STORE: Lazy<OfflineStore> = Lazy::new(OfflineStore::new);

impl OfflineStore {
    fn new() -> Self {
        Self {
            results: DashMap::new(),
        }
    }

    pub fn store(&self, receipt_id: String, body: Bytes, status: StatusCode) {
        info!(
            receipt_id = %receipt_id,
            status = %status,
            body_len = body.len(),
            "Offline store: storing result"
        );
        self.results
            .insert(receipt_id, OfflineResult { body, status });
    }

    pub fn retrieve(&self, receipt_id: &str) -> Option<(Bytes, StatusCode)> {
        self.results.remove(receipt_id).map(|(id, r)| {
            info!(
                receipt_id = %id,
                status = %r.status,
                body_len = r.body.len(),
                "Offline store: retrieved and removed result"
            );
            (r.body, r.status)
        })
    }
}

pub fn is_offline_mode(headers: Option<&HeaderMap>) -> bool {
    headers
        .and_then(|h| h.get("X-SMG-Offline"))
        .and_then(|v| v.to_str().ok())
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn generate_receipt_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn receipt_response(receipt_id: &str) -> Response {
    Json(serde_json::json!({"receipt_id": receipt_id})).into_response()
}

pub fn spawn_offline_request(
    request_builder: RequestBuilder,
    load_guard: Option<WorkerLoadGuard>,
) -> Response {
    let receipt_id = generate_receipt_id();
    let receipt_id_clone = receipt_id.clone();
    info!(receipt_id = %receipt_id, "Offline mode: spawning background request");

    tokio::spawn(async move {
        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!(receipt_id = %receipt_id_clone, error = %e, "Offline mode: request failed");
                let error_body = serde_json::json!({
                    "error": {
                        "message": format!("Request failed: {}", e),
                        "type": "offline_request_error"
                    }
                });
                STORE.store(
                    receipt_id_clone,
                    Bytes::from(error_body.to_string()),
                    StatusCode::INTERNAL_SERVER_ERROR,
                );
                drop(load_guard);
                return;
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        match res.bytes().await {
            Ok(body) => {
                STORE.store(receipt_id_clone, body, status);
            }
            Err(e) => {
                error!(receipt_id = %receipt_id_clone, error = %e, "Offline mode: failed to read body");
                let error_body = serde_json::json!({
                    "error": {
                        "message": format!("Failed to get response body: {}", e),
                        "type": "offline_fetch_error"
                    }
                });
                STORE.store(
                    receipt_id_clone,
                    Bytes::from(error_body.to_string()),
                    StatusCode::INTERNAL_SERVER_ERROR,
                );
            }
        }
        drop(load_guard);
    });

    receipt_response(&receipt_id)
}

#[derive(Deserialize)]
pub struct RetrieveRequest {
    pub receipt_id: String,
}

pub async fn query_maybe_retrieve(Json(req): Json<RetrieveRequest>) -> Response {
    match STORE.retrieve(&req.receipt_id) {
        Some((body, status)) => {
            let mut resp = Response::new(Body::from(body));
            *resp.status_mut() = status;
            resp
        }
        None => Json(serde_json::json!({"status": "pending"})).into_response(),
    }
}
