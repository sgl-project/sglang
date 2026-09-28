//! Standalone deployment topology, health checks, and proxy fallback.

use std::sync::Arc;

use axum::{Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use sglang_frontend::{
    OpenAIService,
    http::{inference_routes, renderer_routes},
};
use sglang_renderer::RendererService;

use crate::engine::HttpGenerateClient;

pub(crate) const DEFAULT_REQUEST_BODY_LIMIT_BYTES: usize = 32 * 1024 * 1024;

fn with_request_body_limit(routes: Router) -> Router {
    routes.layer(axum::extract::DefaultBodyLimit::max(
        DEFAULT_REQUEST_BODY_LIMIT_BYTES,
    ))
}

pub(crate) fn standalone_routes(
    frontend: OpenAIService,
    health_client: HttpGenerateClient,
) -> Router {
    let renderer = frontend.renderer().clone();
    with_request_body_limit(
        inference_routes(frontend)
            .merge(renderer_routes(renderer))
            .merge(
                Router::new()
                    .route("/health", get(engine_health))
                    .with_state(health_client),
            ),
    )
}

pub(crate) fn render_only_routes(renderer: Arc<RendererService>) -> Router {
    with_request_body_limit(
        renderer_routes(renderer).route("/health", get(|| async { StatusCode::OK })),
    )
}

pub(crate) fn hosted_routes(
    frontend: OpenAIService,
    upstream_url: String,
) -> Result<Router, String> {
    let renderer = frontend.renderer().clone();
    let proxy = crate::proxy::RustServerProxy::new(upstream_url)?;
    Ok(with_request_body_limit(
        inference_routes(frontend)
            .merge(renderer_routes(renderer))
            .route("/_sglang_renderer/ready", get(readiness))
            .fallback(move |request| {
                let proxy = proxy.clone();
                async move { proxy.forward(request).await }
            }),
    ))
}

async fn engine_health(State(client): State<HttpGenerateClient>) -> StatusCode {
    match client.health_status().await {
        Ok(status) => status,
        Err(error) => {
            tracing::warn!(message = %error.message, "engine health check failed");
            StatusCode::SERVICE_UNAVAILABLE
        }
    }
}

async fn readiness() -> impl IntoResponse {
    (StatusCode::NO_CONTENT, [("x-sglang-renderer", "ready")])
}
