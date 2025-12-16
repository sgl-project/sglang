//! HTTP client for encode worker in EPD (Encode-Prefill-Decode) disaggregation mode
//!
//! The encode worker processes multimodal inputs (images, videos, audio) and sends
//! embeddings directly to the prefill scheduler via ZMQ. This module handles the
//! HTTP REST API communication with the encode worker.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

/// Default timeout for encode HTTP requests (seconds)
const DEFAULT_ENCODE_TIMEOUT_SECS: u64 = 60;

/// Request to the encode worker's `/encode` endpoint
#[derive(Debug, Clone, Serialize)]
pub struct EncodeRequest {
    /// List of multimodal items (image URLs, video URLs, audio URLs, or base64 data)
    pub mm_items: Vec<String>,

    /// Request ID for tracking
    pub req_id: String,

    /// Total number of encode worker parts (for distributed encoding)
    pub num_parts: i32,

    /// Index of this part (0-indexed)
    pub part_idx: i32,

    /// Hostname of the prefill worker (for ZMQ communication)
    pub prefill_host: String,

    /// Optional list of embedding ports for each prefill rank
    /// If None, the runtime will use dynamic port allocation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_port: Option<Vec<i32>>,
}

/// Response from the encode worker's `/encode` endpoint
///
/// For `zmq_to_scheduler` backend, the response is typically null/empty
/// since embeddings are sent directly via ZMQ.
/// For `mooncake` backend, returns embedding metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct EncodeResponse {
    /// Embedding size in bytes (mooncake backend)
    #[serde(default)]
    pub embedding_size: Option<u64>,

    /// Embedding length (mooncake backend)
    #[serde(default)]
    pub embedding_len: Option<i64>,

    /// Embedding dimension (mooncake backend)
    #[serde(default)]
    pub embedding_dim: Option<i64>,
}

/// Error types for encode client operations
#[derive(Debug)]
pub enum EncodeError {
    /// HTTP request failed
    HttpError(String),
    /// Encode worker returned an error
    EncodeWorkerError(String),
    /// Request timeout
    Timeout,
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::HttpError(msg) => write!(f, "HTTP error: {}", msg),
            EncodeError::EncodeWorkerError(msg) => write!(f, "Encode worker error: {}", msg),
            EncodeError::Timeout => write!(f, "Encode request timed out"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// HTTP client for communicating with encode workers
#[derive(Clone)]
pub struct EncodeHttpClient {
    client: Client,
    timeout: Duration,
}

impl Default for EncodeHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeHttpClient {
    /// Create a new encode HTTP client with default timeout
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(DEFAULT_ENCODE_TIMEOUT_SECS))
    }

    /// Create a new encode HTTP client with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self { client, timeout }
    }

    /// Send an encode request to the encode worker
    ///
    /// # Arguments
    /// * `encode_url` - Base URL of the encode worker (e.g., "http://localhost:30300")
    /// * `request` - The encode request containing multimodal items
    ///
    /// # Returns
    /// * `Ok(EncodeResponse)` - Encode completed successfully
    /// * `Err(EncodeError)` - Encode failed
    pub async fn encode(
        &self,
        encode_url: &str,
        request: EncodeRequest,
    ) -> Result<EncodeResponse, EncodeError> {
        let url = format!("{}/encode", encode_url.trim_end_matches('/'));

        debug!(
            req_id = %request.req_id,
            mm_items_count = request.mm_items.len(),
            num_parts = request.num_parts,
            part_idx = request.part_idx,
            prefill_host = %request.prefill_host,
            "Sending encode request to {}",
            url
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    warn!(
                        req_id = %request.req_id,
                        "Encode request timed out after {:?}",
                        self.timeout
                    );
                    EncodeError::Timeout
                } else {
                    error!(
                        req_id = %request.req_id,
                        error = %e,
                        "HTTP error sending encode request"
                    );
                    EncodeError::HttpError(e.to_string())
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!(
                req_id = %request.req_id,
                status = %status,
                error = %error_text,
                "Encode worker returned error"
            );
            return Err(EncodeError::EncodeWorkerError(format!(
                "Status {}: {}",
                status, error_text
            )));
        }

        // For zmq_to_scheduler backend, response body may be null
        // Try to parse as JSON, but fall back to empty response
        let encode_response = response
            .json::<Option<EncodeResponse>>()
            .await
            .map_err(|e| {
                // This is usually not an error - zmq_to_scheduler returns null
                debug!(
                    req_id = %request.req_id,
                    "Could not parse encode response as JSON (this is normal for zmq_to_scheduler): {}",
                    e
                );
                // Return empty response instead of error
                EncodeError::HttpError(e.to_string())
            })
            .unwrap_or(Some(EncodeResponse {
                embedding_size: None,
                embedding_len: None,
                embedding_dim: None,
            }))
            .unwrap_or(EncodeResponse {
                embedding_size: None,
                embedding_len: None,
                embedding_dim: None,
            });

        debug!(
            req_id = %request.req_id,
            "Encode request completed successfully"
        );

        Ok(encode_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_request_serialization() {
        let request = EncodeRequest {
            mm_items: vec!["https://example.com/image.jpg".to_string()],
            req_id: "test-123".to_string(),
            num_parts: 1,
            part_idx: 0,
            prefill_host: "localhost".to_string(),
            embedding_port: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("mm_items"));
        assert!(json.contains("req_id"));
        assert!(json.contains("num_parts"));
        assert!(json.contains("part_idx"));
        assert!(json.contains("prefill_host"));
        // embedding_port should be skipped when None
        assert!(!json.contains("embedding_port"));
    }

    #[test]
    fn test_encode_request_with_embedding_port() {
        let request = EncodeRequest {
            mm_items: vec!["https://example.com/image.jpg".to_string()],
            req_id: "test-123".to_string(),
            num_parts: 1,
            part_idx: 0,
            prefill_host: "localhost".to_string(),
            embedding_port: Some(vec![8998, 8999]),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("embedding_port"));
        assert!(json.contains("[8998,8999]"));
    }

    #[test]
    fn test_encode_error_display() {
        let err = EncodeError::HttpError("connection refused".to_string());
        assert_eq!(format!("{}", err), "HTTP error: connection refused");

        let err = EncodeError::Timeout;
        assert_eq!(format!("{}", err), "Encode request timed out");
    }
}
