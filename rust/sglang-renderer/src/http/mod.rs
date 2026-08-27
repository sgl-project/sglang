//! OpenAI HTTP frontends built on the engine-neutral renderer contracts.

use std::sync::Arc;

use axum::Router;

mod error;
mod generate_client;
mod openai;
mod render;
mod runtime;
mod submission;
mod tokenize;

pub use generate_client::HttpGenerateClient;
pub use runtime::{RendererRuntimeConfig, serve};
#[cfg(test)]
mod test_utils;

pub struct OpenAIHttpFrontend {
    pub(crate) renderer: Arc<crate::RendererService>,
    pub(crate) generate_client: HttpGenerateClient,
}

impl OpenAIHttpFrontend {
    pub fn new(renderer: Arc<crate::RendererService>, generate_client: HttpGenerateClient) -> Self {
        Self {
            renderer,
            generate_client,
        }
    }
}

pub fn inference_routes(frontend: OpenAIHttpFrontend) -> Router<()> {
    let renderer = frontend.renderer.clone();
    Router::new()
        .merge(openai::routes())
        .with_state(Arc::new(frontend))
        .merge(tokenize::routes(renderer))
}

pub fn render_routes(renderer: Arc<crate::RendererService>) -> Router<()> {
    render::routes(renderer.clone()).merge(tokenize::routes(renderer))
}

pub fn standalone_routes(frontend: OpenAIHttpFrontend) -> Router<()> {
    let renderer = frontend.renderer.clone();
    inference_routes(frontend).merge(render::routes(renderer))
}
