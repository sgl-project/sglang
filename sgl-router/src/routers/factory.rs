//! Factory for creating router instances

use super::{
    http::{pd_router::PDRouter, router::Router},
    RouterTrait,
};
use crate::config::{ConnectionMode, PolicyConfig, RoutingMode};
use crate::policies::PolicyFactory;
use crate::server::AppContext;
use std::sync::Arc;

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        // Check if IGW mode is enabled
        if ctx.router_config.enable_igw {
            return Self::create_igw_router(ctx).await;
        }

        // Check connection mode and route to appropriate implementation
        match ctx.router_config.connection_mode {
            ConnectionMode::Grpc => {
                // Route to gRPC implementation based on routing mode
                match &ctx.router_config.mode {
                    RoutingMode::Regular { worker_urls } => {
                        Self::create_grpc_router(worker_urls, &ctx.router_config.policy, ctx).await
                    }
                    RoutingMode::PrefillDecode {
                        prefill_urls,
                        decode_urls,
                        prefill_policy,
                        decode_policy,
                    } => {
                        Self::create_grpc_pd_router(
                            prefill_urls,
                            decode_urls,
                            prefill_policy.as_ref(),
                            decode_policy.as_ref(),
                            &ctx.router_config.policy,
                            ctx,
                        )
                        .await
                    }
                }
            }
            ConnectionMode::Http => {
                // Route to HTTP implementation based on routing mode
                match &ctx.router_config.mode {
                    RoutingMode::Regular { worker_urls } => {
                        Self::create_regular_router(worker_urls, &ctx.router_config.policy, ctx)
                            .await
                    }
                    RoutingMode::PrefillDecode {
                        prefill_urls,
                        decode_urls,
                        prefill_policy,
                        decode_policy,
                    } => {
                        Self::create_pd_router(
                            prefill_urls,
                            decode_urls,
                            prefill_policy.as_ref(),
                            decode_policy.as_ref(),
                            &ctx.router_config.policy,
                            ctx,
                        )
                        .await
                    }
                }
            }
        }
    }

    /// Create a regular router with injected policy
    async fn create_regular_router(
        worker_urls: &[String],
        policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Create policy
        let policy = PolicyFactory::create_from_config(policy_config);

        // Create regular router with injected policy and client
        let router = Router::new(
            worker_urls.to_vec(),
            policy,
            ctx.client.clone(),
            ctx.router_config.worker_startup_timeout_secs,
            ctx.router_config.worker_startup_check_interval_secs,
            ctx.router_config.dp_aware,
            ctx.router_config.api_key.clone(),
            ctx.router_config.retry.clone(),
            ctx.router_config.circuit_breaker.clone(),
            ctx.router_config.health_check.clone(),
        )
        .await?;

        Ok(Box::new(router))
    }

    /// Create a PD router with injected policy
    async fn create_pd_router(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Create policies - use specific policies if provided, otherwise fall back to main policy
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        // Create PD router with separate policies and client
        let router = PDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            prefill_policy,
            decode_policy,
            ctx.client.clone(),
            ctx.router_config.request_timeout_secs,
            ctx.router_config.worker_startup_timeout_secs,
            ctx.router_config.worker_startup_check_interval_secs,
            ctx.router_config.retry.clone(),
            ctx.router_config.circuit_breaker.clone(),
            ctx.router_config.health_check.clone(),
        )
        .await?;

        Ok(Box::new(router))
    }

    /// Create a gRPC router with injected policy
    pub async fn create_grpc_router(
        worker_urls: &[String],
        policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        use super::grpc::router::GrpcRouter;

        // Create policy
        let policy = PolicyFactory::create_from_config(policy_config);

        // Determine which tokenizer path to use
        // Priority: tokenizer_path > model_path
        let tokenizer_path = ctx
            .router_config
            .tokenizer_path
            .clone()
            .or_else(|| ctx.router_config.model_path.clone())
            .ok_or_else(|| {
                "gRPC router requires either --tokenizer-path or --model-path to be specified"
                    .to_string()
            })?;

        // Create gRPC router
        let router = GrpcRouter::new(
            worker_urls.to_vec(),
            policy,
            ctx.router_config.worker_startup_timeout_secs,
            ctx.router_config.worker_startup_check_interval_secs,
            ctx.router_config.dp_aware,
            ctx.router_config.api_key.clone(),
            ctx.router_config.effective_retry_config(),
            ctx.router_config.effective_circuit_breaker_config(),
            ctx.router_config.health_check.clone(),
            tokenizer_path,
        )
        .await?;

        Ok(Box::new(router))
    }

    /// Create a gRPC PD router with tokenizer and worker configuration
    pub async fn create_grpc_pd_router(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        use super::grpc::pd_router::GrpcPDRouter;

        // Create policies - use specific policies if provided, otherwise fall back to main policy
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        // Determine which tokenizer path to use
        // Priority: tokenizer_path > model_path
        let tokenizer_path = ctx
            .router_config
            .tokenizer_path
            .clone()
            .or_else(|| ctx.router_config.model_path.clone())
            .ok_or_else(|| {
                "gRPC PD router requires either --tokenizer-path or --model-path to be specified"
                    .to_string()
            })?;

        // Create gRPC PD router
        let router = GrpcPDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            prefill_policy,
            decode_policy,
            ctx.router_config.worker_startup_timeout_secs,
            ctx.router_config.worker_startup_check_interval_secs,
            ctx.router_config.dp_aware,
            ctx.router_config.api_key.clone(),
            ctx.router_config.effective_retry_config(),
            ctx.router_config.effective_circuit_breaker_config(),
            ctx.router_config.health_check.clone(),
            tokenizer_path,
        )
        .await?;

        Ok(Box::new(router))
    }

    /// Create an IGW router (placeholder for future implementation)
    async fn create_igw_router(_ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        // For now, return an error indicating IGW is not yet implemented
        Err("IGW mode is not yet implemented".to_string())
    }
}
