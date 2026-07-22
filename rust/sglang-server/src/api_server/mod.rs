//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime. Only
//! this module knows HTTP, so other protocols can mount the same `AppState`.
//! `/generate` submits a `Request` then awaits one `Done` (unary) or relays SSE
//! frames (`data: {json}` … `[DONE]`), byte-compatible with Python
//! `http_server.generate_request`; `/server_info` reuses it for one control result.
mod common;
mod frame;
mod guard;
mod log;
mod openai;
mod sglang;

use std::sync::Arc;

use axum::Router;

use crate::runtime::ServerArgs;
use crate::runtime::channels::Senders;
use crate::tokenizer_manager::ActivityCounter;

/// Shared handler state: the submit machinery (`senders`, `egress_buf`)
/// + shared tokenizer. `Clone` is cheap refcount bumps (every field is `Arc`).
#[derive(Clone)]
struct AppState {
    senders: Senders,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    /// Egress heartbeat (bumped per drained ring frame). `/health_generate`
    /// watches it advance to confirm the scheduler → detok path is alive.
    egress_activity: ActivityCounter,
}

pub async fn serve(
    listener: std::net::TcpListener,
    senders: Senders,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    egress_activity: ActivityCounter,
    shutdown: flume::Receiver<()>,
) {
    let state = AppState {
        senders,
        egress_buf,
        server_args: server_args.clone(),
        egress_activity,
    };
    // Each endpoint module registers its own routes and merges here.
    let app = Router::new()
        .merge(common::routes())
        .merge(sglang::routes())
        .merge(openai::routes())
        // TODO(auth): no API-key boundary yet. Python gates every route (except
        // /health*, /metrics*, OPTIONS) via `add_api_key_middleware`; until ported,
        // a configured `api_key` does NOT protect these routes.
        //
        // No body limit, matching the Python server.
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(state);
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
    if let Ok(addr) = listener.local_addr() {
        tracing::info!(%addr, "sglang-server api listening");
    }
    // Non-graceful shutdown: on the signal, stop accepting and RETURN without
    // waiting for in-flight handlers (a `/generate` blocked on egress would wedge
    // the join). Returning unwinds `block_on` in `runtime::start` → the api tokio
    // runtime drops → detached handlers cancel → their `AbortGuard`s fire, release
    // `Senders` clones → tok/detok channels close → workers exit. Full drain is
    // deferred (see `request_shutdown`).
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
