//! Rate limit time window management
//!
//! Manages time windows for global rate limiting with periodic counter resets

use std::{sync::Arc, time::Duration};

use tokio::time::interval;
use tracing::{debug, info};

use super::sync::HASyncManager;

/// Rate limit window manager
/// Handles periodic reset of rate limit counters for time window management
pub struct RateLimitWindow {
    sync_manager: Arc<HASyncManager>,
    window_seconds: u64,
}

impl RateLimitWindow {
    pub fn new(sync_manager: Arc<HASyncManager>, window_seconds: u64) -> Self {
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
