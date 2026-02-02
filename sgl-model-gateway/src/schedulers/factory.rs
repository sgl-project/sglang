use std::sync::Arc;

use super::proportion::{ProportionScheduler, ProportionSchedulerConfig};
use super::SchedulerPolicy; 
use crate::config::SchedulerConfig;
use crate::app_context::AppContext;

pub struct SchedulerFactory;
use tracing::info;

impl SchedulerFactory {
    /// Create a scheduler policy from configuration.
    pub fn create_from_config(config: &SchedulerConfig, app_context: &Arc<AppContext>) -> Arc<dyn SchedulerPolicy> {
        match config {
            SchedulerConfig::Proportion {
                adjust_interval,
                adjust_window,
                balance_abs_threshold,
                balance_rel_threshold,
                regular_worker_weight,
            } => {
                let scheduler_config = ProportionSchedulerConfig {
                    adjust_interval: std::time::Duration::from_secs(*adjust_interval as u64),
                    adjust_window: std::time::Duration::from_secs(*adjust_window as u64),
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    regular_worker_weight: *regular_worker_weight,
                };
                info!("====== {:#?}", scheduler_config);

                // 调用 ProportionScheduler 的 new 方法
                Arc::new(ProportionScheduler::new(
                    scheduler_config,
                    Arc::clone(&app_context.worker_registry),
                ))
            }
        }
    }
}
