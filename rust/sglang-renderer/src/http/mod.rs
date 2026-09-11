//! OpenAI HTTP frontend and render-only routes.

use std::sync::Arc;

use axum::Router;

use crate::engine::HttpGenerateClient;

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
    let proxy = proxy::RustServerProxy::new(upstream_url)?;
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
