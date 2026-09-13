//! Router assembly and the shared handler state: every endpoint module
//! registers its routes here, and [`serve`] runs the assembled app on the
//! pre-bound listener until shutdown.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use axum::{
    Router,
    extract::{Request, State},
    middleware::Next,
    response::Response,
};

use super::disaggregation::bootstrap as pd_bootstrap;
use super::{common, log, native_api, openai};
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
    /// Whether the main process's startup warmup has completed. The listener
    /// binds before warmup so `/model_info` is available to construct that
    /// request, but health endpoints must not advertise readiness yet.
    pub(super) startup_readiness: StartupReadiness,
}

pub(super) struct StartupReadiness(AtomicBool);

impl StartupReadiness {
    fn new(skip_server_warmup: bool) -> Self {
        Self(AtomicBool::new(skip_server_warmup))
    }

    pub(super) fn is_ready(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    fn record_warmup_status(&self, status: axum::http::StatusCode) {
        if status.is_success() {
            self.0.store(true, Ordering::Release);
        }
    }
}

impl Default for StartupReadiness {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Private marker attached by the main process to its startup warmup request.
/// The middleware flips readiness only after that request returns successfully.
const STARTUP_WARMUP_HEADER: &str = "x-sglang-startup-warmup";

async fn mark_startup_ready(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let is_startup_warmup = req.headers().contains_key(STARTUP_WARMUP_HEADER)
        && matches!(
            req.uri().path(),
            "/generate" | "/encode" | "/v1/chat/completions"
        );
    let response = next.run(req).await;
    if is_startup_warmup && response.status().is_success() {
        state
            .startup_readiness
            .record_warmup_status(response.status());
    }
    response
}

pub async fn serve(
    listener: std::net::TcpListener,
    senders: Senders,
    response_buf: usize,
    server_args: Arc<ServerArgs>,
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
        startup_readiness: StartupReadiness::new(server_args.skip_server_warmup),
    });
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
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            mark_startup_ready,
        ))
        .with_state(state);

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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn startup_readiness_requires_successful_warmup_unless_skipped() {
        let readiness = StartupReadiness::new(false);
        assert!(!readiness.is_ready());

        readiness.record_warmup_status(StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!readiness.is_ready());

        readiness.record_warmup_status(StatusCode::OK);
        assert!(readiness.is_ready());

        assert!(StartupReadiness::new(true).is_ready());
    }
}
