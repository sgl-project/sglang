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
                prefill_policy,
                decode_policy,
            } => Self::create_pd_router(
                prefill_urls,
                decode_urls,
                prefill_policy.as_ref(),
                decode_policy.as_ref(),
                &config.policy,
                config,
            ),
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
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        router_config: &RouterConfig,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Create policies - use specific policies if provided, otherwise fall back to main policy
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        // Create PD router with separate policies
        let router = PDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            prefill_policy,
            decode_policy,
            router_config.worker_startup_timeout_secs,
            router_config.worker_startup_check_interval_secs,
        )?;

        Ok(Box::new(router))
    }
}
