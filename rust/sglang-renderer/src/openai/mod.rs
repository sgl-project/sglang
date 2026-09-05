//! OpenAI HTTP frontend and render-only routes.

use std::sync::Arc;

use axum::Router;
use dynamo_protocols::types::CompletionUsage;

use crate::engine::HttpGenerateClient;

mod chat;
mod completions;
mod error;
mod protocol;
mod render;
mod submission;
mod tokenize;

#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;

use protocol::{ChatCompletionRequest, CompletionRequest};

const DEFAULT_REQUEST_BODY_LIMIT_BYTES: usize = 32 * 1024 * 1024;

pub(crate) struct OpenAIHttpFrontend {
    pub(crate) renderer: Arc<crate::RendererService>,
    pub(crate) generate_client: HttpGenerateClient,
}

impl OpenAIHttpFrontend {
    pub(crate) fn new(
        renderer: Arc<crate::RendererService>,
        generate_client: HttpGenerateClient,
    ) -> Self {
        Self {
            renderer,
            generate_client,
        }
    }
}

pub(crate) fn inference_routes(frontend: OpenAIHttpFrontend) -> Router<()> {
    Router::new()
        .merge(chat::routes())
        .merge(completions::routes())
        .with_state(Arc::new(frontend))
}

fn renderer_routes(renderer: Arc<crate::RendererService>) -> Router<()> {
    render::routes(renderer.clone()).merge(tokenize::routes(renderer))
}

fn with_request_body_limit(routes: Router<()>) -> Router<()> {
    routes.layer(axum::extract::DefaultBodyLimit::max(
        DEFAULT_REQUEST_BODY_LIMIT_BYTES,
    ))
}

pub(super) fn unix_seconds_u32() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u32::try_from(duration.as_secs()).unwrap_or(u32::MAX))
        .unwrap_or(0)
}

pub(super) fn completion_usage(prompt_tokens: u32, completion_tokens: u32) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        ..Default::default()
    }
}

pub(crate) fn standalone_routes(frontend: OpenAIHttpFrontend) -> Router<()> {
    let renderer = frontend.renderer.clone();
    let generate_client = frontend.generate_client.clone();
    let routes = inference_routes(frontend).merge(renderer_routes(renderer));
    let routes = routes.merge(render::engine_health_route(generate_client));
    with_request_body_limit(routes)
}

pub(crate) fn render_only_routes(renderer: Arc<crate::RendererService>) -> Router<()> {
    let routes = renderer_routes(renderer).merge(render::health_route());
    with_request_body_limit(routes)
}

pub(crate) fn hosted_routes(
    frontend: OpenAIHttpFrontend,
    upstream_url: String,
) -> Result<Router<()>, String> {
    let renderer = frontend.renderer.clone();
    let proxy = crate::runtime::RustServerProxy::new(upstream_url)?;
    Ok(inference_routes(frontend)
        .merge(renderer_routes(renderer))
        .merge(render::readiness_route())
        .fallback(move |request| {
            let proxy = proxy.clone();
            async move { proxy.forward(request).await }
        })
        .layer(axum::extract::DefaultBodyLimit::max(
            DEFAULT_REQUEST_BODY_LIMIT_BYTES,
        )))
}
