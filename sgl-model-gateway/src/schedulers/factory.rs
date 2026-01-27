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
                // 将从配置文件解析的参数转换为调度器内部使用的配置结构体
                let scheduler_config = ProportionSchedulerConfig {
                    // 注意：这里假设你的 SchedulerConfig 字段是 Duration
                    // 如果它们是 usize (秒数)，你需要这样转换：
                    // adjust_interval: std::time::Duration::from_secs(*adjust_interval as u64),
                    // adjust_window: std::time::Duration::from_secs(*adjust_window as u64),

                    // 假设它们已经是 Duration 了
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
