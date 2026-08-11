//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime. Only
//! this module knows HTTP, so other protocols can mount the same `AppState`.
//! `/generate` submits a `Request` then awaits one `Done` (unary) or relays SSE
//! frames (`data: {json}` … `[DONE]`), byte-compatible with Python
//! `http_server.generate_request`; `/server_info` reuses it for one control result.
mod common;
mod frame;
mod guard;
mod log;
mod native_api;
mod openai;
mod pd_bootstrap;
mod prefetch;
mod submit;

use std::sync::Arc;

use axum::Router;

use crate::runtime::ServerArgs;
use crate::tokenizer_manager::ActivityCounter;
use crate::tokenizer_manager::Senders;

/// TTFT profiling: CLOCK_MONOTONIC ns at which the request head reached the
/// handler stack (before the body was read). Stored in request extensions by
/// [`stamp_request_start`]; logged by the `/generate` handler once the rid is
/// known.
#[derive(Clone, Copy)]
pub(crate) struct RequestStartNs(pub u64);

/// TTFT profiling middleware: stamp request-head arrival into extensions.
async fn stamp_request_start(
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    req.extensions_mut()
        .insert(RequestStartNs(crate::ttft_stamp::mono_ns()));
    next.run(req).await
}

/// Shared handler state: submission handles, immutable server configuration,
/// and the API-owned chat formatter.
#[derive(Clone)]
struct AppState {
    senders: Senders,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    chat_formatter: Option<openai::ChatFormatter>,
    /// Egress heartbeat (bumped per drained ring frame).
    egress_activity: ActivityCounter,
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
    let mut app = Router::new()
        .merge(common::routes())
        .merge(native_api::routes())
        .merge(openai::routes())
        // TODO(auth): no API-key boundary yet. Python gates every route (except
        // /health*, /metrics*, OPTIONS) via `add_api_key_middleware`; until ported,
        // a configured `api_key` does NOT protect these routes.
        //
        // No body limit, matching the Python server.
        .layer(axum::extract::DefaultBodyLimit::disable())
        // TTFT profiling: stamp request-head arrival before the body is read.
        .layer(axum::middleware::from_fn(stamp_request_start))
        .with_state(state);
    if server_args.enable_pd_bootstrap() {
        // Merged after `with_state` (the registry carries its own state) and
        // before `log::apply`, so bootstrap traffic shows in the access log.
        let (bootstrap_routes, sweeper) = pd_bootstrap::router_and_sweeper();
        tokio::spawn(sweeper); // cancelled with the runtime on shutdown
        app = app.merge(bootstrap_routes);
        tracing::info!("PD KV bootstrap registry mounted on the api listener");
    }
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
