//! Renderer process state and HTTP listener.

use std::net::SocketAddr;
use std::sync::Arc;

use crate::{DynamoTokenizer, RendererConfig, RendererService, TextTokenizer, load_tokenizer};

use crate::engine::{GenerationService, HttpGenerateClient, TokenDecoder};
use crate::frontend::http::{hosted_routes, render_only_routes, standalone_routes};
use crate::openai::OpenAIService;

#[derive(Clone, Debug)]
pub struct RendererRuntimeConfig {
    pub http_addr: SocketAddr,
    pub http_workers: usize,
    pub tokenizer_workers: usize,
    pub queue_capacity: usize,
    /// Optional SGLang engine origin. When absent, inference routes are not mounted.
    pub engine_url: Option<String>,
    /// Proxy routes not owned by the renderer to `engine_url`.
    pub proxy_unhandled_routes: bool,
    pub renderer: RendererConfig,
}

pub async fn serve(config: RendererRuntimeConfig) -> Result<(), String> {
    let mode = match (&config.engine_url, config.proxy_unhandled_routes) {
        (None, false) => "render-only",
        (Some(_), false) => "serving",
        (Some(_), true) => "hosted",
        (None, true) => return Err("proxy_unhandled_routes requires engine_url".to_string()),
    };
    let tokenizer_without_specials = load_tokenizer(
        (!config.renderer.tokenizer_path.is_empty())
            .then_some(config.renderer.tokenizer_path.as_str()),
        config.renderer.revision.as_deref(),
        false,
    )?;
    let tokenizer_with_specials = load_tokenizer(
        (!config.renderer.tokenizer_path.is_empty())
            .then_some(config.renderer.tokenizer_path.as_str()),
        config.renderer.revision.as_deref(),
        true,
    )?;
    let encode_tokenizer: Arc<dyn TextTokenizer> = Arc::new(DynamoTokenizer::new(
        tokenizer_without_specials.clone(),
        tokenizer_with_specials,
    ));
    let renderer = Arc::new(RendererService::with_tokenizer(
        config.renderer,
        encode_tokenizer,
        config.tokenizer_workers,
        config.queue_capacity,
    ));
    let app = match (config.engine_url, config.proxy_unhandled_routes) {
        (None, false) => render_only_routes(renderer),
        (Some(engine_url), false) => {
            let generate_client = HttpGenerateClient::new(engine_url)?;
            standalone_routes(
                OpenAIService::new(
                    renderer,
                    GenerationService::new(
                        Arc::new(generate_client.clone()),
                        TokenDecoder::new(tokenizer_without_specials),
                    ),
                ),
                generate_client,
            )
        }
        (Some(engine_url), true) => {
            let generate_client = HttpGenerateClient::new(&engine_url)?;
            hosted_routes(
                OpenAIService::new(
                    renderer,
                    GenerationService::new(
                        Arc::new(generate_client),
                        TokenDecoder::new(tokenizer_without_specials),
                    ),
                ),
                engine_url,
            )?
        }
        (None, true) => unreachable!("runtime topology was validated above"),
    };
    let listener = tokio::net::TcpListener::bind(config.http_addr)
        .await
        .map_err(|error| format!("binding renderer on {} failed: {error}", config.http_addr))?;
    tracing::info!(address = %config.http_addr, mode, "renderer listening");
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error, "installing renderer shutdown signal failed");
            }
        })
        .await
        .map_err(|error| format!("renderer HTTP server failed: {error}"))
}
