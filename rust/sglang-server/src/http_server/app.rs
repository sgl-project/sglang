//! Router assembly and the shared handler state: every endpoint module
//! registers its routes here, and [`serve`] runs the assembled app on the
//! pre-bound listener until shutdown.

use std::sync::Arc;

use axum::Router;

use super::disaggregation::bootstrap as pd_bootstrap;
use super::{common, log, native_api, openai};
use crate::message::config::ServerArgs;
use crate::tokenizer_manager::egress::ActivityCounter;
use crate::tokenizer_manager::wiring::Senders;

/// Shared handler state: submission handles, immutable server configuration,
/// and the API-owned chat formatter.
#[derive(Clone)]
pub(super) struct AppState {
    pub(super) senders: Senders,
    pub(super) egress_buf: usize,
    pub(super) server_args: Arc<ServerArgs>,
    pub(super) chat_formatter: Option<openai::ChatFormatter>,
    /// Egress heartbeat (bumped per drained ring frame).
    pub(super) egress_activity: ActivityCounter,
}

pub async fn serve(
    listener: std::net::TcpListener,
    senders: Senders,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    egress_activity: ActivityCounter,
    // The SAME set ingress releases from — see `Ingress::on_abort`. Constructing a
    // local one here would leave the api server admitting rids that nothing ever
    // releases.
    shutdown: flume::Receiver<()>,
) {
    let chat_formatter = openai::load_chat_support(&server_args);
    let state = AppState {
        senders,
        egress_buf,
        server_args: server_args.clone(),
        chat_formatter,
        egress_activity,
    };
    // Each endpoint module registers its own routes and merges here.
    let router = Router::new()
        .merge(common::routes())
        .merge(native_api::routes())
        .merge(openai::routes());

    // TODO(auth): no API-key boundary yet. Python gates every route (except
    // /health*, /metrics*, OPTIONS) via `add_api_key_middleware`; until ported,
    // a configured `api_key` does NOT protect these routes.
    //
    // No body limit, matching the Python server.
    let mut app = router
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(state);

    // Prefill-only KV bootstrap registry. Merged AFTER `with_state` — its
    // router carries its own Arc<Registry> state, so it cannot merge into the
    // Router<AppState> above — and before `log::apply`, so bootstrap traffic
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
