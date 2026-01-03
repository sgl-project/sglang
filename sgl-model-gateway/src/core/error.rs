//! Error types for the SGLang router core
//!
//! This module defines error types used throughout the router for worker operations.

/// Worker-related errors
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Health check failed for worker {url}: {reason}")]
    HealthCheckFailed { url: String, reason: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Invalid worker configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Network error for worker {url}: {error}")]
    NetworkError { url: String, error: String },

    #[error("Worker at capacity: {url}")]
    WorkerAtCapacity { url: String },

    #[error("Invalid URL format: {url}")]
    InvalidUrl { url: String },

    #[error("Connection failed for worker {url}: {reason}")]
    ConnectionFailed { url: String, reason: String },
}

/// Result type for worker operations
pub type WorkerResult<T> = Result<T, WorkerError>;

/// Convert from reqwest errors to worker errors
impl From<reqwest::Error> for WorkerError {
    fn from(err: reqwest::Error) -> Self {
        WorkerError::NetworkError {
            url: err.url().map(|u| u.to_string()).unwrap_or_default(),
            error: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn test_health_check_failed_display() {
        let error = WorkerError::HealthCheckFailed {
            url: "http://worker1:8080".to_string(),
            reason: "Connection refused".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Health check failed for worker http://worker1:8080: Connection refused"
        );
    }

    #[test]
    fn test_worker_not_found_display() {
        let error = WorkerError::WorkerNotFound {
            url: "http://worker2:8080".to_string(),
        };
        assert_eq!(error.to_string(), "Worker not found: http://worker2:8080");
    }

    #[test]
    fn test_invalid_configuration_display() {
        let error = WorkerError::InvalidConfiguration {
            message: "Missing port number".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid worker configuration: Missing port number"
        );
    }

    #[test]
    fn test_network_error_display() {
        let error = WorkerError::NetworkError {
            url: "http://worker3:8080".to_string(),
            error: "Timeout after 30s".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Network error for worker http://worker3:8080: Timeout after 30s"
        );
    }

    #[test]
    fn test_worker_at_capacity_display() {
        let error = WorkerError::WorkerAtCapacity {
            url: "http://worker4:8080".to_string(),
        };
        assert_eq!(error.to_string(), "Worker at capacity: http://worker4:8080");
    }

    #[test]
    fn test_worker_error_implements_std_error() {
        let error = WorkerError::WorkerNotFound {
            url: "http://test".to_string(),
        };
        let _: &dyn Error = &error;
        assert!(error.source().is_none());
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WorkerError>();
    }

    #[test]
    fn test_worker_result_type_alias() {
        let result: WorkerResult<i32> = Ok(42);
        assert!(matches!(result, Ok(42)));

        let error = WorkerError::WorkerNotFound {
            url: "test".to_string(),
        };
        let result: WorkerResult<i32> = Err(error);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_url_handling() {
        let error1 = WorkerError::HealthCheckFailed {
            url: "".to_string(),
            reason: "No connection".to_string(),
        };
        assert_eq!(
            error1.to_string(),
            "Health check failed for worker : No connection"
        );

        let error2 = WorkerError::NetworkError {
            url: "".to_string(),
            error: "DNS failure".to_string(),
        };
        assert_eq!(error2.to_string(), "Network error for worker : DNS failure");

        let error3 = WorkerError::WorkerNotFound {
            url: "".to_string(),
        };
        assert_eq!(error3.to_string(), "Worker not found: ");
    }

    #[test]
    fn test_special_characters_in_messages() {
        let error = WorkerError::InvalidConfiguration {
            message: "Invalid JSON: {\"error\": \"test\"}".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid worker configuration: Invalid JSON: {\"error\": \"test\"}"
        );

        let error2 = WorkerError::HealthCheckFailed {
            url: "http://测试:8080".to_string(),
            reason: "连接被拒绝".to_string(),
        };
        assert_eq!(
            error2.to_string(),
            "Health check failed for worker http://测试:8080: 连接被拒绝"
        );
    }

    #[test]
    fn test_very_long_error_messages() {
        let long_message = "A".repeat(10000);
        let error = WorkerError::InvalidConfiguration {
            message: long_message.clone(),
        };
        let display = error.to_string();
        assert!(display.contains(&long_message));
        assert_eq!(
            display.len(),
            "Invalid worker configuration: ".len() + long_message.len()
        );
    }

    #[test]
    fn test_reqwest_error_conversion() {
        let network_error = WorkerError::NetworkError {
            url: "http://example.com".to_string(),
            error: "connection timeout".to_string(),
        };

        match network_error {
            WorkerError::NetworkError { url, error } => {
                assert_eq!(url, "http://example.com");
                assert_eq!(error, "connection timeout");
            }
            _ => panic!("Expected NetworkError variant"),
        }
    }

    #[test]
    fn test_error_equality() {
        let error1 = WorkerError::WorkerNotFound {
            url: "http://test".to_string(),
        };
        let error2 = WorkerError::WorkerNotFound {
            url: "http://test".to_string(),
        };
        assert_eq!(error1.to_string(), error2.to_string());
    }
}
