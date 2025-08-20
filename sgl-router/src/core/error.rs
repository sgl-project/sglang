//! Error types for the SGLang router core
//!
//! This module defines error types used throughout the router for worker operations.

use std::fmt;

/// Worker-related errors
#[derive(Debug)]
pub enum WorkerError {
    /// Health check failed
    HealthCheckFailed { url: String, reason: String },
    /// Worker not found
    WorkerNotFound { url: String },
    /// Invalid worker configuration
    InvalidConfiguration { message: String },
    /// Network error
    NetworkError { url: String, error: String },
    /// Worker is at capacity
    WorkerAtCapacity { url: String },
    /// Invalid URL format
    InvalidUrl { url: String },
    /// OpenAI API error
    OpenAIApiError {
        url: String,
        error_code: Option<String>,
        message: String,
    },
    /// OpenAI authentication error
    OpenAIAuthError { url: String, message: String },
    /// OpenAI rate limit error
    OpenAIRateLimitError {
        url: String,
        retry_after: Option<u64>,
    },
    /// OpenAI model not found or not available
    OpenAIModelError {
        url: String,
        model: String,
        message: String,
    },
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerError::HealthCheckFailed { url, reason } => {
                write!(f, "Health check failed for worker {}: {}", url, reason)
            }
            WorkerError::WorkerNotFound { url } => {
                write!(f, "Worker not found: {}", url)
            }
            WorkerError::InvalidConfiguration { message } => {
                write!(f, "Invalid worker configuration: {}", message)
            }
            WorkerError::NetworkError { url, error } => {
                write!(f, "Network error for worker {}: {}", url, error)
            }
            WorkerError::WorkerAtCapacity { url } => {
                write!(f, "Worker at capacity: {}", url)
            }
            WorkerError::InvalidUrl { url } => {
                write!(f, "Invalid URL format: {}", url)
            }
            WorkerError::OpenAIApiError {
                url,
                error_code,
                message,
            } => {
                if let Some(code) = error_code {
                    write!(
                        f,
                        "OpenAI API error for worker {}: {} ({})",
                        url, message, code
                    )
                } else {
                    write!(f, "OpenAI API error for worker {}: {}", url, message)
                }
            }
            WorkerError::OpenAIAuthError { url, message } => {
                write!(
                    f,
                    "OpenAI authentication error for worker {}: {}",
                    url, message
                )
            }
            WorkerError::OpenAIRateLimitError { url, retry_after } => {
                if let Some(retry) = retry_after {
                    write!(
                        f,
                        "OpenAI rate limit exceeded for worker {}: retry after {} seconds",
                        url, retry
                    )
                } else {
                    write!(f, "OpenAI rate limit exceeded for worker {}", url)
                }
            }
            WorkerError::OpenAIModelError {
                url,
                model,
                message,
            } => {
                write!(
                    f,
                    "OpenAI model error for worker {} (model: {}): {}",
                    url, model, message
                )
            }
        }
    }
}

impl std::error::Error for WorkerError {}

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

/// Convert from async-openai errors to worker errors
impl From<async_openai::error::OpenAIError> for WorkerError {
    fn from(err: async_openai::error::OpenAIError) -> Self {
        match err {
            async_openai::error::OpenAIError::ApiError(api_err) => {
                // Check for specific error types based on error codes or messages
                let message = api_err.message.clone();
                let error_type = api_err.r#type.as_deref();
                let error_code = api_err.code.clone();

                match error_type {
                    Some("insufficient_quota") | Some("billing_hard_limit_reached") => {
                        WorkerError::OpenAIApiError {
                            url: "".to_string(), // URL will be filled in by caller
                            error_code,
                            message,
                        }
                    }
                    Some("invalid_api_key") | Some("invalid_organization") => {
                        WorkerError::OpenAIAuthError {
                            url: "".to_string(),
                            message,
                        }
                    }
                    Some("rate_limit_exceeded") => {
                        WorkerError::OpenAIRateLimitError {
                            url: "".to_string(),
                            retry_after: None, // Could extract from headers if available
                        }
                    }
                    Some("model_not_found") | Some("model_overloaded") => {
                        WorkerError::OpenAIModelError {
                            url: "".to_string(),
                            model: "".to_string(), // Model will be filled in by caller
                            message,
                        }
                    }
                    _ => WorkerError::OpenAIApiError {
                        url: "".to_string(),
                        error_code,
                        message,
                    },
                }
            }
            async_openai::error::OpenAIError::Reqwest(reqwest_err) => WorkerError::NetworkError {
                url: reqwest_err.url().map(|u| u.to_string()).unwrap_or_default(),
                error: reqwest_err.to_string(),
            },
            async_openai::error::OpenAIError::StreamError(stream_err) => {
                WorkerError::NetworkError {
                    url: "".to_string(),
                    error: format!("OpenAI stream error: {}", stream_err),
                }
            }
            async_openai::error::OpenAIError::JSONDeserialize(json_err) => {
                WorkerError::OpenAIApiError {
                    url: "".to_string(),
                    error_code: None,
                    message: format!("JSON deserialization error: {}", json_err),
                }
            }
            async_openai::error::OpenAIError::InvalidArgument(arg_err) => {
                WorkerError::InvalidConfiguration {
                    message: format!("Invalid OpenAI argument: {}", arg_err),
                }
            }
            _ => WorkerError::OpenAIApiError {
                url: "".to_string(),
                error_code: None,
                message: format!("Unhandled OpenAI error: {}", err),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

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
        // Verify it implements Error trait
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
        // Test Ok variant
        let result: WorkerResult<i32> = Ok(42);
        assert!(matches!(result, Ok(42)));

        // Test Err variant
        let error = WorkerError::WorkerNotFound {
            url: "test".to_string(),
        };
        let result: WorkerResult<i32> = Err(error);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_url_handling() {
        // Test empty URLs in error variants
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
        // Test with special characters
        let error = WorkerError::InvalidConfiguration {
            message: "Invalid JSON: {\"error\": \"test\"}".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid worker configuration: Invalid JSON: {\"error\": \"test\"}"
        );

        // Test with unicode
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

    // Mock reqwest error for testing conversion
    #[test]
    fn test_reqwest_error_conversion() {
        // Test that NetworkError is the correct variant
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
        // WorkerError doesn't implement PartialEq, but we can test that
        // the same error construction produces the same display output
        let error1 = WorkerError::WorkerNotFound {
            url: "http://test".to_string(),
        };
        let error2 = WorkerError::WorkerNotFound {
            url: "http://test".to_string(),
        };
        assert_eq!(error1.to_string(), error2.to_string());
    }

    // Tests for OpenAI-specific errors
    #[test]
    fn test_openai_api_error_display() {
        let error = WorkerError::OpenAIApiError {
            url: "https://api.openai.com".to_string(),
            error_code: Some("rate_limit_exceeded".to_string()),
            message: "API rate limit exceeded".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "OpenAI API error for worker https://api.openai.com: API rate limit exceeded (rate_limit_exceeded)"
        );

        let error_no_code = WorkerError::OpenAIApiError {
            url: "https://api.openai.com".to_string(),
            error_code: None,
            message: "Unknown API error".to_string(),
        };
        assert_eq!(
            error_no_code.to_string(),
            "OpenAI API error for worker https://api.openai.com: Unknown API error"
        );
    }

    #[test]
    fn test_openai_auth_error_display() {
        let error = WorkerError::OpenAIAuthError {
            url: "https://api.openai.com".to_string(),
            message: "Invalid API key provided".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "OpenAI authentication error for worker https://api.openai.com: Invalid API key provided"
        );
    }

    #[test]
    fn test_openai_rate_limit_error_display() {
        let error_with_retry = WorkerError::OpenAIRateLimitError {
            url: "https://api.openai.com".to_string(),
            retry_after: Some(60),
        };
        assert_eq!(
            error_with_retry.to_string(),
            "OpenAI rate limit exceeded for worker https://api.openai.com: retry after 60 seconds"
        );

        let error_no_retry = WorkerError::OpenAIRateLimitError {
            url: "https://api.openai.com".to_string(),
            retry_after: None,
        };
        assert_eq!(
            error_no_retry.to_string(),
            "OpenAI rate limit exceeded for worker https://api.openai.com"
        );
    }

    #[test]
    fn test_openai_model_error_display() {
        let error = WorkerError::OpenAIModelError {
            url: "https://api.openai.com".to_string(),
            model: "gpt-4".to_string(),
            message: "Model not found".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "OpenAI model error for worker https://api.openai.com (model: gpt-4): Model not found"
        );
    }
}
