//! Integration tests for connection mode determination with --grpc-mode flag

use sglang_router_rs::core::ConnectionMode;

/// Helper function to simulate determine_connection_mode logic
/// This tests the logic without depending on internal CLI implementation
fn determine_connection_mode(
    worker_urls: &[String],
    grpc_mode_flag: bool,
) -> Result<ConnectionMode, String> {
    // If --grpc-mode flag is set, validate and return gRPC mode
    if grpc_mode_flag {
        // Check for conflicting HTTP/HTTPS URLs
        for url in worker_urls {
            if url.starts_with("http://") || url.starts_with("https://") {
                return Err(format!(
                    "--grpc-mode flag conflicts with HTTP/HTTPS URL: {}. Use grpc:// prefix or remove the flag.",
                    url
                ));
            }
        }
        return Ok(ConnectionMode::Grpc { port: None });
    }

    // If flag not set, detect from URL prefixes
    for url in worker_urls {
        if url.starts_with("grpc://") || url.starts_with("grpcs://") {
            return Ok(ConnectionMode::Grpc { port: None });
        }
    }

    Ok(ConnectionMode::Http)
}

#[test]
fn test_grpc_mode_flag_forces_grpc() {
    let urls = vec!["worker1:8000".to_string(), "worker2:8000".to_string()];
    let result = determine_connection_mode(&urls, true);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ConnectionMode::Grpc { .. }));
}

#[test]
fn test_grpc_mode_with_http_url_fails() {
    let urls = vec!["http://worker1:8000".to_string()];
    let result = determine_connection_mode(&urls, true);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("--grpc-mode flag conflicts"));
}

#[test]
fn test_grpc_mode_with_https_url_fails() {
    let urls = vec!["https://worker1:8000".to_string()];
    let result = determine_connection_mode(&urls, true);
    assert!(result.is_err());
}

#[test]
fn test_no_flag_detects_grpc_from_url() {
    let urls = vec!["grpc://worker1:8000".to_string()];
    let result = determine_connection_mode(&urls, false);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ConnectionMode::Grpc { .. }));
}

#[test]
fn test_no_flag_defaults_to_http() {
    let urls = vec!["worker1:8000".to_string()];
    let result = determine_connection_mode(&urls, false);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ConnectionMode::Http));
}

#[test]
fn test_grpc_mode_with_grpc_url_succeeds() {
    let urls = vec!["grpc://worker1:8000".to_string()];
    let result = determine_connection_mode(&urls, true);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ConnectionMode::Grpc { .. }));
}
