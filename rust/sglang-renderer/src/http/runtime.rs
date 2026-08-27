//! Standalone renderer process state and HTTP listener.

use std::net::SocketAddr;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    DynamoTokenizer, OpenAIRequestLowerer, PooledTokenizer, RendererConfig, RendererService,
    TextTokenizer, load_tokenizer,
};

use super::{HttpGenerateClient, OpenAIHttpFrontend, standalone_routes};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RendererRuntimeConfig {
    pub http_addr: SocketAddr,
    #[serde(default = "default_http_workers")]
    pub http_workers: usize,
    #[serde(default = "default_tokenizer_workers")]
    pub tokenizer_workers: usize,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
    pub engine_url: String,
    pub renderer: RendererConfig,
}

pub async fn serve(config: RendererRuntimeConfig) -> Result<(), String> {
    if config.renderer.skip_tokenizer_init {
        return Err("standalone rendering requires a tokenizer".into());
    }
    let tokenizer = load_tokenizer(
        (!config.renderer.tokenizer_path.is_empty())
            .then_some(config.renderer.tokenizer_path.as_str()),
        config.renderer.revision.as_deref(),
        false,
    )?
    .ok_or_else(|| "standalone rendering requires a tokenizer".to_owned())?;
    let encode_tokenizer: Arc<dyn TextTokenizer> =
        Arc::new(DynamoTokenizer::new(tokenizer.clone()));
    let tokenizer_backend = Arc::new(PooledTokenizer::new(
        encode_tokenizer,
        config.tokenizer_workers,
        config.queue_capacity,
    ));
    let renderer = Arc::new(RendererService::new(
        OpenAIRequestLowerer::new(config.renderer),
        tokenizer_backend,
    ));
    let generate_client = HttpGenerateClient::new(config.engine_url, tokenizer)?;
    let listener = tokio::net::TcpListener::bind(config.http_addr)
        .await
        .map_err(|error| format!("binding renderer on {} failed: {error}", config.http_addr))?;
    tracing::info!(address = %config.http_addr, "standalone renderer listening");
    axum::serve(
        listener,
        standalone_routes(OpenAIHttpFrontend::new(renderer, generate_client)).into_make_service(),
    )
    .with_graceful_shutdown(async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "installing renderer shutdown signal failed");
        }
    })
    .await
    .map_err(|error| format!("renderer HTTP server failed: {error}"))
}

const fn default_http_workers() -> usize {
    2
}

const fn default_tokenizer_workers() -> usize {
    4
}

const fn default_queue_capacity() -> usize {
    128
}
