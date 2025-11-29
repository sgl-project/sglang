//! Factory for creating router instances

use std::sync::Arc;

use super::{
    grpc::{pd_router::GrpcPDRouter, router::GrpcRouter},
    http::{pd_router::PDRouter, router::Router},
    openai::OpenAIRouter,
    RouterTrait,
};
use crate::{
    app_context::AppContext,
    config::{PolicyConfig, RoutingMode},
    core::ConnectionMode,
    policies::PolicyFactory,
};

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        match ctx.router_config.connection_mode {
            ConnectionMode::Grpc { .. } => match &ctx.router_config.mode {
                RoutingMode::Regular { .. } => Self::create_grpc_router(ctx).await,
                RoutingMode::PrefillDecode {
                    prefill_policy,
                    decode_policy,
                    ..
                } => {
                    Self::create_grpc_pd_router(
                        prefill_policy.as_ref(),
                        decode_policy.as_ref(),
                        &ctx.router_config.policy,
                        ctx,
                    )
                    .await
                }
                RoutingMode::OpenAI { .. } => {
                    Err("OpenAI mode requires HTTP connection_mode".to_string())
                }
            },
            ConnectionMode::Http => match &ctx.router_config.mode {
                RoutingMode::Regular { .. } => Self::create_regular_router(ctx).await,
                RoutingMode::PrefillDecode {
                    prefill_policy,
                    decode_policy,
                    ..
                } => {
                    Self::create_pd_router(
                        prefill_policy.as_ref(),
                        decode_policy.as_ref(),
                        &ctx.router_config.policy,
                        ctx,
                    )
                    .await
                }
                RoutingMode::OpenAI { worker_urls } => {
                    Self::create_openai_router(worker_urls.clone(), ctx).await
                }
            },
        }
    }

    /// Create a regular router
    pub async fn create_regular_router(
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let router = Router::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a PD router with injected policy
    pub async fn create_pd_router(
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);

        let router = PDRouter::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a gRPC router with injected policy
    pub async fn create_grpc_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        let router = GrpcRouter::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a gRPC PD router with tokenizer and worker configuration
    pub async fn create_grpc_pd_router(
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);
        let router = GrpcPDRouter::new(ctx).await?;

        Ok(Box::new(router))
    }

    /// Create an OpenAI router
    async fn create_openai_router(
        worker_urls: Vec<String>,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        if worker_urls.is_empty() {
            return Err("OpenAI mode requires at least one worker URL".to_string());
        }

        let router = OpenAIRouter::new(worker_urls, ctx).await?;

        Ok(Box::new(router))
    }
}
