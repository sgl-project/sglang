pub mod mock_worker;
pub mod test_app;

use sglang_router_rs::config::RouterConfig;
use sglang_router_rs::server::AppContext;
use std::sync::Arc;

/// Helper function to create AppContext for tests
pub fn create_test_context(config: RouterConfig) -> Arc<AppContext> {
    Arc::new(AppContext::new(
        config.clone(),
        reqwest::Client::new(),
        config.max_concurrent_requests,
    ))
}
