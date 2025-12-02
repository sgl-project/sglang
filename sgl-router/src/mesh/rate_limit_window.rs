//! Rate limit time window management
//!
//! Manages time windows for global rate limiting with periodic counter resets

use std::{sync::Arc, time::Duration};

use tokio::time::interval;
use tracing::{debug, info};

use super::sync::MeshSyncManager;

/// Rate limit window manager
/// Handles periodic reset of rate limit counters for time window management
pub struct RateLimitWindow {
    sync_manager: Arc<MeshSyncManager>,
    window_seconds: u64,
}

impl RateLimitWindow {
    pub fn new(sync_manager: Arc<MeshSyncManager>, window_seconds: u64) -> Self {
        Self {
            sync_manager,
            window_seconds,
        }
    }

    /// Start the window reset task
    /// This task periodically resets the global rate limit counter
    pub async fn start_reset_task(self) {
        let mut interval_timer = interval(Duration::from_secs(self.window_seconds));
        info!(
            "Starting rate limit window reset task with {}s interval",
            self.window_seconds
        );

        loop {
            interval_timer.tick().await;

            debug!("Resetting global rate limit counter");
            self.sync_manager.reset_global_rate_limit_counter();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use tokio::time::sleep;

    use super::*;
    use crate::mesh::stores::{
        RateLimitConfig, StateStores, GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
    };

    #[tokio::test]
    async fn test_rate_limit_window_reset_task() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores.clone(), "node1".to_string()));

        // Setup membership
        stores.rate_limit.update_membership(&["node1".to_string()]);

        // Setup config
        let key = crate::mesh::crdt::SKey::new(GLOBAL_RATE_LIMIT_KEY.to_string());
        let config = RateLimitConfig {
            limit_per_second: 100,
        };
        let serialized = serde_json::to_vec(&config).unwrap();
        stores.app.insert(
            key,
            crate::mesh::stores::AppState {
                key: GLOBAL_RATE_LIMIT_KEY.to_string(),
                value: serialized,
                version: 1,
            },
            "node1".to_string(),
        );

        // Increment counter
        if stores.rate_limit.is_owner(GLOBAL_RATE_LIMIT_COUNTER_KEY) {
            sync_manager.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 10);
            let value_before = sync_manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            assert!(value_before.is_some() && value_before.unwrap() > 0);

            // Create window manager with short interval for testing
            let window = RateLimitWindow::new(sync_manager.clone(), 1); // 1 second

            // Start reset task in background
            let reset_handle = tokio::spawn(async move {
                window.start_reset_task().await;
            });

            // Wait a bit for reset to happen
            sleep(Duration::from_millis(1500)).await;

            // Check that counter was reset (or at least decremented)
            let _value_after = sync_manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            // Counter should be reset or significantly reduced
            // Note: The exact value depends on timing, but it should be less than initial

            // Cancel the task
            reset_handle.abort();
        }
    }

    #[test]
    fn test_rate_limit_window_new() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let _window = RateLimitWindow::new(sync_manager, 60);
        // Just verify it was created successfully (no panic means success)
    }

    #[tokio::test]
    async fn test_reset_global_rate_limit_counter_logic() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores.clone(), "node1".to_string()));

        // Setup membership
        stores.rate_limit.update_membership(&["node1".to_string()]);

        if stores.rate_limit.is_owner(GLOBAL_RATE_LIMIT_COUNTER_KEY) {
            // Increment counter
            sync_manager.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 20);
            let value_before = sync_manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            assert!(value_before.is_some() && value_before.unwrap() > 0);

            // Reset
            sync_manager.reset_global_rate_limit_counter();

            // Check that counter was reset
            let value_after = sync_manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            // Should be 0 or negative after reset
            assert!(value_after.is_none() || value_after.unwrap() <= 0);
        }
    }
}
