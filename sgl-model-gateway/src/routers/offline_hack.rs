use axum::{
    body::Body,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use reqwest::Response as ReqwestResponse;
use serde::Deserialize;
use tracing::info;

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

pub fn generate_receipt_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub fn receipt_response(receipt_id: &str) -> Response {
    Json(serde_json::json!({"receipt_id": receipt_id})).into_response()
}

pub fn handle_offline_response(
    res: ReqwestResponse,
    status: StatusCode,
    load_guard: Option<WorkerLoadGuard>,
) -> Response {
    let receipt_id = generate_receipt_id();
    let receipt_id_clone = receipt_id.clone();
    tokio::spawn(async move {
        let result = match res.bytes().await {
            Ok(body) => {
                STORE.store(receipt_id_clone, body, status);
                Ok(())
            }
            Err(e) => {
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
                Err(e)
            }
        };
        drop(load_guard);
        result
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
