//! OpenAI HTTP frontends built on the engine-neutral renderer contracts.

use std::sync::Arc;

use axum::Router;
use dynamo_protocols::types::CreateChatCompletionRequest;
use serde::Deserialize;

mod error;
mod generate_client;
mod openai;
mod render;
mod runtime;
mod submission;
mod tokenize;

#[derive(Deserialize)]
pub(crate) struct ExtendedChatCompletionRequest {
    #[serde(flatten)]
    pub request: CreateChatCompletionRequest,
    #[serde(default)]
    pub chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub continue_final_message: bool,
}

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

#[cfg(test)]
mod tests {
    use super::ExtendedChatCompletionRequest;

    #[test]
    fn extended_chat_request_preserves_template_controls() {
        let request: ExtendedChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": false},
            "continue_final_message": true
        }))
        .unwrap();

        assert_eq!(
            request
                .chat_template_kwargs
                .as_ref()
                .and_then(|args| args.get("enable_thinking")),
            Some(&serde_json::Value::Bool(false))
        );
        assert!(request.continue_final_message);
    }
}
