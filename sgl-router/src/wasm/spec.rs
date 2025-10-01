//! WIT Bindings and Type Conversions
//!
//! Contains wasmtime component bindings generated from WIT definitions,
//! and helper functions to convert between Axum HTTP types and WIT types.

use axum::{body::Body, extract::Request};
use serde::{Deserialize, Serialize};

wasmtime::component::bindgen!({
    path: "src/wasm/wit",
    world: "sgl-router",
    imports: { default: async | trappable },
    exports: { default: async },
});

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitMiddlewareHeader {
    pub name: String,
    pub value: String,
}

// onRequest
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitMiddlewareHeaderOnRequest {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: Vec<WitMiddlewareHeader>,
    pub body: Vec<u8>,
    #[serde(rename = "request-id")]
    pub request_id: String,
    #[serde(rename = "now-epoch-ms")]
    pub now_epoch_ms: u64,
}

impl WitMiddlewareHeaderOnRequest {
    pub fn to_wit_request(&self) -> sgl::router::middleware_types::Request {
        sgl::router::middleware_types::Request {
            method: self.method.clone(),
            path: self.path.clone(),
            query: self.query.clone(),
            headers: self
                .headers
                .iter()
                .map(|h| sgl::router::middleware_types::Header {
                    name: h.name.clone(),
                    value: h.value.clone(),
                })
                .collect(),
            body: self.body.clone(),
            request_id: self.request_id.clone(),
            now_epoch_ms: self.now_epoch_ms,
        }
    }
}

// onResponse
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitMiddlewareHeaderOnResponse {
    pub status: u16,
    pub headers: Vec<WitMiddlewareHeader>,
    pub body: Vec<u8>,
}

// Return actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WitMiddlewareAction {
    Continue,
    Reject(u16),
    Modify(WitMiddlewareModifyAction),
}

impl WitMiddlewareAction {
    pub fn from_wit(action: sgl::router::middleware_types::Action) -> Self {
        use sgl::router::middleware_types::Action;
        match action {
            Action::Continue => Self::Continue,
            Action::Reject(status) => Self::Reject(status),
            Action::Modify(modify) => Self::Modify(WitMiddlewareModifyAction {
                status: modify.status,
                headers_set: modify
                    .headers_set
                    .into_iter()
                    .map(|h| WitMiddlewareHeader {
                        name: h.name,
                        value: h.value,
                    })
                    .collect(),
                headers_add: modify
                    .headers_add
                    .into_iter()
                    .map(|h| WitMiddlewareHeader {
                        name: h.name,
                        value: h.value,
                    })
                    .collect(),
                headers_remove: modify.headers_remove,
                body_replace: modify.body_replace,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitMiddlewareModifyAction {
    pub status: Option<u16>,
    #[serde(rename = "headers-set")]
    pub headers_set: Vec<WitMiddlewareHeader>,
    #[serde(rename = "headers-add")]
    pub headers_add: Vec<WitMiddlewareHeader>,
    #[serde(rename = "headers-remove")]
    pub headers_remove: Vec<String>,
    #[serde(rename = "body-replace")]
    pub body_replace: Option<Vec<u8>>,
}

/// Convert axum Request to WIT Request
///
/// This helper function extracts all necessary information from an axum Request
/// and converts it to the WIT Request type.
pub async fn build_wit_request_from_axum(
    request: Request<Body>,
    request_id: String,
) -> Result<sgl::router::middleware_types::Request, String> {
    // Extract metadata before consuming the request
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let query = request.uri().query().unwrap_or("").to_string();

    // Extract headers
    let mut headers = Vec::new();
    for (name, value) in request.headers() {
        if let Ok(value_str) = value.to_str() {
            headers.push(sgl::router::middleware_types::Header {
                name: name.as_str().to_string(),
                value: value_str.to_string(),
            });
        }
    }

    // TODO: Extract body (this consumes the request)
    let body = axum::body::to_bytes(request.into_body(), usize::MAX)
        .await
        .map_err(|e| format!("Failed to read request body: {}", e))?
        .to_vec();

    // Build WIT Request
    Ok(sgl::router::middleware_types::Request {
        method,
        path,
        query,
        headers,
        body,
        request_id,
        // SystemTime::duration_since only fails if the system time is before UNIX_EPOCH,
        // which should never happen in normal operation. If it does, use 0 as fallback.
        now_epoch_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| {
                // Fallback to 0 if system time is invalid
                // This should never occur in practice, but provides a safe fallback
                std::time::Duration::from_millis(0)
            })
            .as_millis() as u64,
    })
}

/// Convert axum Response to WIT Response
///
/// This helper function extracts all necessary information from an axum Response
/// and converts it to the WIT Response type.
pub async fn build_wit_response_from_axum(
    response: axum::response::Response<Body>,
) -> Result<sgl::router::middleware_types::Response, String> {
    // Extract status before consuming the response
    let status = response.status().as_u16();

    // Extract headers
    let mut headers = Vec::new();
    for (name, value) in response.headers() {
        if let Ok(value_str) = value.to_str() {
            headers.push(sgl::router::middleware_types::Header {
                name: name.as_str().to_string(),
                value: value_str.to_string(),
            });
        }
    }

    // TODO: Extract body (this consumes the response)
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .map_err(|e| format!("Failed to read response body: {}", e))?
        .to_vec();

    // Build WIT Response
    Ok(sgl::router::middleware_types::Response {
        status,
        headers,
        body,
    })
}
