//! Router assembly and the shared handler state: every endpoint module
//! registers its routes here, and [`serve`] runs the assembled app on the
//! pre-bound listener until shutdown.

use std::sync::Arc;

use axum::{Router, http::StatusCode};

use super::disaggregation::bootstrap as pd_bootstrap;
use super::{auth, common, log, native_api, openai};
use crate::api_server::auth::{AuthConfig, AuthLevel};
use crate::message::config::ServerArgs;
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::Senders;

/// Shared handler state: submission handles, immutable server configuration,
/// and the API-owned chat formatter.
///
/// axum clones the router state into **every** request, so it is mounted as
/// `Arc<AppState>` — one refcount bump per request instead of cloning each
/// `flume::Sender` and the chat formatter. Deliberately not `Clone`, so it
/// can only be shared through that `Arc`.
pub(super) struct AppState {
    pub(super) senders: Senders,
    pub(super) response_buf: usize,
    pub(super) server_args: Arc<ServerArgs>,
    pub(super) chat_formatter: Option<openai::ChatFormatter>,
    /// Response heartbeat (bumped per drained ring frame).
    pub(super) response_activity: ActivityCounter,
}

pub async fn serve(
    listener: std::net::TcpListener,
    senders: Senders,
    response_buf: usize,
    server_args: Arc<ServerArgs>,
    auth_config: Arc<AuthConfig>,
    response_activity: ActivityCounter,
    // The runtime's shutdown signal, shared with every worker stage: it fires
    // (disconnects) when `Runtime::request_shutdown` drops the sender, at
    // which point `serve` stops accepting and its in-flight handlers are
    // aborted with the api runtime.
    shutdown: flume::Receiver<()>,
) {
    let chat_formatter = openai::load_chat_support(&server_args);
    let state = Arc::new(AppState {
        senders,
        response_buf,
        server_args: server_args.clone(),
        chat_formatter,
        response_activity,
    });
    // Each endpoint module registers its own routes and merges here.
    let public_router = Router::new()
        .merge(common::routes())
        .merge(native_api::routes())
        .merge(openai::routes())
        // Python's global ASGI middleware authenticates unknown customer paths
        // before they become a 404. Register the public fallback before the
        // AuthZ layer to preserve that observable 401/404 ordering.
        .fallback(public_not_found)
        // No body limit, matching the Python server.
        .layer(axum::extract::DefaultBodyLimit::disable());

    // Protect only customer-facing routes. This must happen before the PD
    // bootstrap router is merged below: Axum layers wrap routes already present
    // at the call site, which is the structural AuthZ boundary for v1.
    let mut app = auth::protect(public_router, auth_config, AuthLevel::Normal).with_state(state);

    // Prefill-only KV bootstrap registry. Merged AFTER `with_state` — its
    // router carries its own Arc<Registry> state, so it cannot merge into the
    // Router<Arc<AppState>> above — and before `log::apply`, so bootstrap traffic
    // shows in the access log. The bootstrap router must keep Axum's default
    // fallback: the public router already owns the custom, protected fallback,
    // and Axum rejects merging two routers that both define one.
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

async fn public_not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}
