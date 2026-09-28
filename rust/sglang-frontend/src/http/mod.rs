//! HTTP adapters shared by embedded and standalone frontends.

use crate::openai::protocol::{ChatCompletionRequest, CompletionRequest};
use crate::{OpenAIService, RendererService};
use axum::Router;
use std::sync::Arc;

mod chat;
mod completions;
mod error;
mod render;
mod response;
mod tokenize;

pub fn inference_routes(frontend: OpenAIService) -> Router {
    Router::new()
        .merge(chat::routes())
        .merge(completions::routes())
        .with_state(Arc::new(frontend))
}

pub fn renderer_routes(renderer: Arc<RendererService>) -> Router {
    render::routes(renderer.clone()).merge(tokenize::routes(renderer))
}
