//! OpenAI HTTP frontend and render-only routes.

use std::sync::Arc;

use axum::Router;

use crate::engine::HttpGenerateClient;
use crate::openai::OpenAIService;

mod chat;
mod completions;
mod error;
mod proxy;
mod render;
mod response;
mod tokenize;

#[cfg(test)]
mod tests;

use crate::openai::protocol::{ChatCompletionRequest, CompletionRequest};

const DEFAULT_REQUEST_BODY_LIMIT_BYTES: usize = 32 * 1024 * 1024;

pub(crate) fn inference_routes(frontend: OpenAIService) -> Router<()> {
    Router::new()
        .merge(chat::routes())
        .merge(completions::routes())
        .with_state(Arc::new(frontend))
}

fn renderer_routes(renderer: Arc<crate::RendererService>) -> Router<()> {
    render::routes(renderer.clone()).merge(tokenize::routes(renderer))
}

fn with_request_body_limit(routes: Router<()>) -> Router<()> {
    // Limit JSON extraction without buffering or limiting raw proxy bodies.
    routes.layer(axum::extract::DefaultBodyLimit::max(
        DEFAULT_REQUEST_BODY_LIMIT_BYTES,
    ))
}

pub(crate) fn standalone_routes(
    frontend: OpenAIService,
    health_client: HttpGenerateClient,
) -> Router<()> {
    let renderer = frontend.renderer.clone();
    let routes = inference_routes(frontend).merge(renderer_routes(renderer));
    let routes = routes.merge(render::engine_health_route(health_client));
    with_request_body_limit(routes)
}

pub(crate) fn render_only_routes(renderer: Arc<crate::RendererService>) -> Router<()> {
    let routes = renderer_routes(renderer).merge(render::health_route());
    with_request_body_limit(routes)
}

pub(crate) fn hosted_routes(
    frontend: OpenAIService,
    upstream_url: String,
) -> Result<Router<()>, String> {
    let renderer = frontend.renderer.clone();
    let proxy = proxy::RustServerProxy::new(upstream_url)?;
    let routes = inference_routes(frontend)
        .merge(renderer_routes(renderer))
        .merge(render::readiness_route())
        .fallback(move |request| {
            let proxy = proxy.clone();
            async move { proxy.forward(request).await }
        });
    Ok(with_request_body_limit(routes))
}
