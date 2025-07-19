//! Factory for creating router instances

use super::{pd_router::PDRouter, router::Router, RouterTrait};
use crate::config::{PolicyConfig, RouterConfig, RoutingMode};
use crate::policies::PolicyFactory;

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from configuration
    pub fn create_router(config: &RouterConfig) -> Result<Box<dyn RouterTrait>, String> {
        match &config.mode {
            RoutingMode::Regular { worker_urls } => {
                Self::create_regular_router(worker_urls, &config.policy, config)
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
            } => Self::create_pd_router(prefill_urls, decode_urls, &config.policy, config),
        }
    }

    /// Create a regular router with injected policy
    fn create_regular_router(
        worker_urls: &[String],
        policy_config: &PolicyConfig,
        router_config: &RouterConfig,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Create policy
        let policy = PolicyFactory::create_from_config(policy_config);

        // Create regular router with injected policy
        let router = Router::new(
            worker_urls.to_vec(),
            policy,
            router_config.worker_startup_timeout_secs,
            router_config.worker_startup_check_interval_secs,
        )?;

        Ok(Box::new(router))
    }

    /// Create a PD router with injected policy
    fn create_pd_router(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        policy_config: &PolicyConfig,
        router_config: &RouterConfig,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Create policy directly from PolicyConfig
        // All policies now support PD mode through the select_worker_pair method
        let policy = PolicyFactory::create_from_config(policy_config);

        // Create PD router with injected policy
        let router = PDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            policy,
            router_config.worker_startup_timeout_secs,
            router_config.worker_startup_check_interval_secs,
        )?;

        Ok(Box::new(router))
    }
}
