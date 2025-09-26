//! Factory for creating router instances

use super::grpc::pd_router::GrpcPDRouter;
use super::grpc::router::GrpcRouter;
use super::{
    http::{openai_router::OpenAIRouter, pd_router::PDRouter, router::Router},
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
        match ctx.router_config.connection_mode {
            ConnectionMode::Grpc => match &ctx.router_config.mode {
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
                RoutingMode::OpenAI { worker_urls, .. } => {
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
        let base_url = worker_urls
            .first()
            .cloned()
            .ok_or_else(|| "OpenAI mode requires at least one worker URL".to_string())?;

        let router = OpenAIRouter::new(
            base_url,
            Some(ctx.router_config.circuit_breaker.clone()),
            ctx.response_storage.clone(),
        )
        .await?;

        Ok(Box::new(router))
    }
}
