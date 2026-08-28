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
    let renderer = frontend.renderer.clone();
    Router::new()
        .merge(chat::routes())
        .merge(completions::routes())
        .with_state(Arc::new(frontend))
        .merge(tokenize::routes(renderer))
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
    inference_routes(frontend)
        .merge(render::routes(renderer))
        .merge(render::health_route())
}

pub(crate) fn hosted_routes(
    frontend: OpenAIHttpFrontend,
    fallback_url: String,
) -> Result<Router<()>, String> {
    let renderer = frontend.renderer.clone();
    let proxy = crate::runtime::RustServerProxy::new(fallback_url)?;
    Ok(inference_routes(frontend)
        .merge(render::routes(renderer))
        .fallback(move |request| {
            let proxy = proxy.clone();
            async move { proxy.forward(request).await }
        })
        .layer(axum::extract::DefaultBodyLimit::disable()))
}
