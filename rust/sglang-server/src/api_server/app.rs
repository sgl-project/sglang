//! Router assembly and the shared handler state: every endpoint module
//! registers its routes here, and [`serve`] runs the assembled app on the
//! pre-bound listener until shutdown.

use std::sync::Arc;

use axum::{
    Router,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};

use super::disaggregation::bootstrap as pd_bootstrap;
use super::{common, log, native_api, openai};
use crate::frontend::FrontendHandle;

/// HTTP's existing state name now refers to the operation state shared with
/// gRPC: frontend capability, immutable launch policy, and one chat formatter.
/// HTTP routing, middleware, and response framing remain in this module tree.
///
/// axum clones the router state into **every** request, so it is mounted as
/// `Arc<AppState>` — one refcount bump per request instead of cloning the
/// frontend handle and chat formatter. Deliberately not `Clone`, so it
/// can only be shared through that `Arc`.
pub(super) use crate::openai::OpenAiState as AppState;

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
    record_startup_warmup_status(&state.frontend, is_startup_warmup, response.status());
    response
}

fn record_startup_warmup_status(
    frontend: &FrontendHandle,
    is_startup_warmup: bool,
    status: StatusCode,
) {
    if is_startup_warmup && status.is_success() {
        frontend.mark_ready();
    }
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
    let server_args = state.server_args.clone();
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

    fn frontend() -> FrontendHandle {
        FrontendHandle::new(
            flume::unbounded().0,
            flume::unbounded().0,
            crate::frontend::FrontendConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: false,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: crate::frontend::FrontendMetadata::default(),
            },
        )
    }

    #[test]
    fn only_a_successful_recognized_warmup_marks_frontend_ready() {
        let frontend = frontend();

        record_startup_warmup_status(&frontend, false, StatusCode::OK);
        assert!(!frontend.is_ready());

        record_startup_warmup_status(&frontend, true, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!frontend.is_ready());

        record_startup_warmup_status(&frontend, true, StatusCode::OK);
        assert!(frontend.is_ready());
    }
}
