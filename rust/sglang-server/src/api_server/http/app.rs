//! Router assembly and the shared handler state: every endpoint module
//! registers its routes here, and [`serve`] runs the assembled app on the
//! pre-bound listener until shutdown.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;

use super::{hicache, info, loads, metrics, native_api, openai};
use crate::HttpLayer;
use crate::api_server::core::{CoreHandle, HiCacheManager};
use crate::api_server::disaggregation::bootstrap as pd_bootstrap;
use crate::api_server::log;
use crate::message::config::ServerArgs;
use crate::message::request::GenerationPolicy;
use crate::utils::environ;

/// HTTP adapter state: the shared core capability, immutable server
/// configuration needed for HTTP request preparation, and the chat formatter.
///
/// axum clones the router state into **every** request, so it is mounted as
/// `Arc<AppState>` — one refcount bump per request instead of cloning the
/// core handle and chat formatter. Deliberately not `Clone`, so it
/// can only be shared through that `Arc`.
pub(super) struct AppState {
    pub(super) core: CoreHandle,
    pub(super) server_args: Arc<ServerArgs>,
    pub(super) generation_policy: GenerationPolicy,
    pub(super) chat_formatter: Option<openai::ChatFormatter>,
}

/// Every listener warms its own rank; health stays 503 until it succeeds,
/// and a failed or timed-out warmup leaves the listener unready.
async fn startup_warmup(core: CoreHandle, server_args: Arc<ServerArgs>) {
    if core.is_ready() {
        return;
    }
    let timeout = match environ::env_i64("SGLANG_WARMUP_TIMEOUT", -1) {
        t if t > 0 => t as u64,
        _ if server_args.is_disaggregation() => 1800,
        _ => 600,
    };
    match tokio::time::timeout(
        Duration::from_secs(timeout),
        core.warm_up(server_args.skip_tokenizer_init),
    )
    .await
    {
        Ok(Ok(())) => {
            core.mark_ready();
            tracing::info!("The server is fired up and ready to roll!");
        }
        Ok(Err(e)) => tracing::error!(error = %e, "startup warmup failed"),
        Err(_) => tracing::error!(timeout_secs = timeout, "startup warmup timed out"),
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn serve(
    listener: std::net::TcpListener,
    core: CoreHandle,
    server_args: Arc<ServerArgs>,
    // Whether this listener serves the node's `/metrics`.
    metrics_listener: bool,
    http_layer: Option<HttpLayer>,
    // The runtime's shutdown signal, shared with every worker stage: it fires
    // (disconnects) when `Runtime::request_shutdown` drops the sender, at
    // which point `serve` stops accepting and its in-flight handlers are
    // aborted with the api runtime.
    shutdown: flume::Receiver<()>,
    hicache_manager: Arc<HiCacheManager>,
) {
    let chat_formatter = openai::load_chat_support(&server_args);
    let generation_policy = GenerationPolicy::from(server_args.as_ref());
    let state = Arc::new(AppState {
        core,
        server_args: server_args.clone(),
        generation_policy,
        chat_formatter,
    });
    // Cancelled with the api runtime on shutdown.
    tokio::spawn(startup_warmup(state.core.clone(), server_args.clone()));

    // Each endpoint module registers its own routes and merges here.
    let mut router = Router::new()
        .merge(info::routes())
        .merge(loads::routes())
        .merge(native_api::routes())
        .merge(openai::routes());
    if server_args.metrics.is_some() && metrics_listener {
        router = router.merge(metrics::routes());
    }

    // TODO(auth): no API-key boundary yet. Python gates every route (except
    // /health*, /metrics*, OPTIONS) via `add_api_key_middleware`; until ported,
    // a configured `api_key` does NOT protect these routes.
    //
    // No body limit, matching the Python server.
    let mut app = router
        .with_state(state)
        .merge(hicache::routes(hicache_manager))
        .layer(axum::extract::DefaultBodyLimit::disable());
    if let Some(http_layer) = http_layer {
        app = http_layer(app);
    }

    // Prefill-only KV bootstrap registry. Merged AFTER `with_state` — its
    // router carries its own Arc<Registry> state, so it cannot merge into the
    // Router<Arc<AppState>> above — and before `log::apply`, so bootstrap traffic
    // shows in the access log.
    if server_args.enable_pd_bootstrap() {
        let (routes, sweeper) = pd_bootstrap::router_and_sweeper();
        tokio::spawn(sweeper); // cancelled with the runtime on shutdown
        app = app.merge(routes);
        tracing::info!("PD KV bootstrap registry mounted on the api listener");
    }

    // Apply logging and access log middleware.
    let app = log::apply(app, &server_args);

    // The listener was already bound synchronously in `runtime::start` (so a port
    // conflict fails startup); adopt it into the tokio reactor here.
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "failed to adopt pre-bound listener");
            return;
        }
    };
    // `with_connect_info` exposes the peer address to the access-log middleware.
    let serve = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    );
    tokio::select! {
        r = serve => {
            if let Err(e) = r {
                tracing::error!(error = %e, "axum serve exited");
            }
        }
        _ = shutdown.recv_async() => {
            tracing::info!("shutdown: stopping accepts, aborting in-flight handlers");
        }
    }
}
