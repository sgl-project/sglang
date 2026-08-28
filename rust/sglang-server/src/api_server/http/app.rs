//! The axum route table and server. Every endpoint delegates to the sibling
//! handler modules; [`serve`] runs the layered router on the pre-bound
//! listener until shutdown. The shared handler state is built once in
//! `utils::runtime` (so the gRPC transport mounts the same instance) and
//! arrives as `Arc<CoreState>`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path, Request};
use axum::routing::{get, post};
use http::StatusCode;

use super::disaggregation as pd_bootstrap;
use super::response::status_response;
use super::{common, native_api, openai};
use crate::api_server::layers::Peer;
use crate::api_server::layers::access_log::AccessLogLayer;
use crate::api_server::layers::auth::ApiKeyAuthLayer;
use crate::utils::environ;

/// The HTTP transport's name for the shared [`CoreState`].
pub(super) use crate::api_server::core::state::CoreState as AppState;

impl crate::api_server::layers::RejectionBody for axum::body::Body {
    fn empty() -> Self {
        axum::body::Body::empty()
    }
    fn from_static(bytes: &'static [u8]) -> Self {
        axum::body::Body::from(bytes)
    }
}

/// The route table over the shared state, bare (tests mount it as-is;
/// [`serve`] wraps it in the auth / access-log layers). Both env knobs are
/// resolved ONCE at construction (server startup) — changing them on a live
/// process needs a restart.
pub(crate) fn router(
    state: Arc<AppState>,
    bootstrap: Option<Arc<pd_bootstrap::Registry>>,
) -> Router {
    let health_timeout = Duration::from_secs(environ::env_u64("SGLANG_HEALTH_CHECK_TIMEOUT", 20));
    // `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` (default true, mirroring
    // Python): whether `/health` runs the deep generate probe or is a plain
    // 200 (routing the request already proves the frontend is up).
    let health_probes_generation =
        environ::env_bool("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", true);

    Router::new()
        .route("/generate", {
            let state = state.clone();
            post(move |req: Request| native_api::generate(state, req))
        })
        .route("/health", {
            let state = state.clone();
            get(move || async move {
                if health_probes_generation {
                    native_api::health_generate(state, health_timeout).await
                } else {
                    status_response(StatusCode::OK)
                }
            })
        })
        .route("/health_generate", {
            let state = state.clone();
            get(move || native_api::health_generate(state, health_timeout))
        })
        .route("/server_info", {
            let state = state.clone();
            get(move || common::server_info(state))
        })
        .route("/get_model_info", {
            let state = state.clone();
            get(move || common::model_info(state))
        })
        .route("/model_info", {
            let state = state.clone();
            get(move || common::model_info(state))
        })
        .route("/v1/models", {
            let state = state.clone();
            get(move || openai::available_models(state))
        })
        .route("/v1/models/{model}", {
            let state = state.clone();
            get(move |Path(model): Path<String>| async move {
                openai::retrieve_model(state, &model).await
            })
        })
        .route("/v1/completions", {
            let state = state.clone();
            post(move |req: Request| openai::completions(state, req))
        })
        .route("/v1/chat/completions", {
            let state = state.clone();
            post(move |req: Request| openai::chat_completions(state, req))
        })
        // PD KV bootstrap routes, resolved after the main table and before the
        // 404. Python gates them with the same auth middleware, so the
        // fallback sits inside the layers `serve` adds.
        .fallback(move |req: Request| {
            let bootstrap = bootstrap.clone();
            async move {
                if let Some(registry) = &bootstrap
                    && let Some(response) = pd_bootstrap::dispatch(registry, req).await
                {
                    return response;
                }
                status_response(StatusCode::NOT_FOUND)
            }
        })
        // No body cap, matching the Python server (`read_json` buffers uncapped).
        .layer(DefaultBodyLimit::disable())
}

pub async fn serve(
    listener: std::net::TcpListener,
    state: Arc<AppState>,
    // The runtime's shutdown signal, shared with every worker stage: it fires
    // (disconnects) when `Runtime::request_shutdown` drops the sender, at
    // which point `serve` stops accepting and its in-flight handlers are
    // aborted with the api runtime.
    shutdown: flume::Receiver<()>,
) {
    use std::future::IntoFuture;

    let server_args = state.server_args.clone();

    // Prefill-only KV bootstrap registry.
    let bootstrap = server_args.enable_pd_bootstrap().then(|| {
        let (registry, sweeper) = pd_bootstrap::registry_and_sweeper();
        tokio::spawn(sweeper); // cancelled with the runtime on shutdown
        tracing::info!("PD KV bootstrap registry mounted on the api listener");
        registry
    });

    let auth = tower::util::option_layer(state.api_key.as_deref().map(ApiKeyAuthLayer::new));
    let access_log = tower::util::option_layer(
        server_args
            .http_access_log_enabled()
            .then_some(AccessLogLayer),
    );
    // Stamp axum's per-connection `ConnectInfo` as the transport-neutral
    // `Peer` before the access log (which reads it back). Shared tower
    // layers: auth inside, access log outermost (a 401 is still logged).
    let stamp_peer = tower::util::MapRequestLayer::new(|mut req: Request| {
        if let Some(&ConnectInfo(peer)) = req.extensions().get::<ConnectInfo<SocketAddr>>() {
            req.extensions_mut().insert(Peer(peer));
        }
        req
    });
    let app = router(state, bootstrap).layer(
        tower::ServiceBuilder::new()
            .layer(stamp_peer)
            .layer(access_log)
            .layer(auth),
    );

    // The listener was already bound synchronously in `runtime::start` (so a
    // port conflict fails startup); adopt it into the tokio reactor here.
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "failed to adopt pre-bound listener");
            return;
        }
    };

    // Selecting away drops the server and its connections, which drops the
    // in-flight handler futures and fires their AbortGuards.
    tokio::select! {
        result = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .into_future() => {
            if let Err(e) = result {
                tracing::error!(error = %e, "http server exited");
            }
        }
        _ = shutdown.recv_async() => {
            tracing::info!("shutdown: stopping accepts, aborting in-flight handlers");
        }
    }
}
