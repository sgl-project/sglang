pub mod mock_worker;

use actix_web::web;
use reqwest::Client;
use sglang_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use sglang_router_rs::server::AppState;

/// Helper function to create test router configuration
pub fn create_test_config(worker_urls: Vec<String>) -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::Regular { worker_urls },
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 3001,
        max_payload_size: 256 * 1024 * 1024, // 256MB
        request_timeout_secs: 600,
        worker_startup_timeout_secs: 300,
        worker_startup_check_interval_secs: 10,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: None,
    }
}

/// Helper function to create test router configuration with no health check
pub fn create_test_config_no_workers() -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![],
        }, // Empty to skip health check
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 3001,
        max_payload_size: 256 * 1024 * 1024, // 256MB
        request_timeout_secs: 600,
        worker_startup_timeout_secs: 0, // No wait
        worker_startup_check_interval_secs: 10,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: None,
    }
}

/// Helper function to create test app state
pub async fn create_test_app_state(config: RouterConfig) -> Result<web::Data<AppState>, String> {
    // Create a non-blocking client
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
        .build()
        .map_err(|e| e.to_string())?;

    let app_state = AppState::new(config, client)?;
    Ok(web::Data::new(app_state))
}
