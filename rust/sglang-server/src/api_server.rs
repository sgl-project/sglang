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
mod submit;
mod vertex;

use std::sync::Arc;

use axum::Router;

use crate::runtime::ServerArgs;
use crate::tokenizer_manager::ActivityCounter;
use crate::tokenizer_manager::Senders;

/// Shared handler state: the submit machinery (`senders`, `egress_buf`)
/// + shared tokenizer.
#[derive(Clone)]
struct AppState {
    senders: Senders,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    /// Egress heartbeat (bumped per drained ring frame).
    egress_activity: ActivityCounter,
    /// Client-visible rids currently in flight. Detok `Register` is an
    /// insert-overwrite keyed on the rid's hash, so two concurrent requests
    /// sharing a rid would evict each other's sink and cross-wire their output;
    /// this rejects the second one instead, as Python does ("Duplicate request ID
    /// detected"). Entries are removed by [`guard::AbortGuard`] on drop.
    live_rids: LiveRids,
}

/// The in-flight rid set (see [`AppState::live_rids`]). A `std::sync::Mutex` is
/// right here: it is held only across one hash-set operation, never across an
/// await.
use crate::tokenizer_manager::LiveRids;

pub async fn serve(
    listener: std::net::TcpListener,
    senders: Senders,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    egress_activity: ActivityCounter,
    // The SAME set ingress releases from — see `Ingress::on_abort`. Constructing a
    // local one here would leave the api server admitting rids that nothing ever
    // releases.
    live_rids: LiveRids,
    shutdown: flume::Receiver<()>,
) {
    let state = AppState {
        senders,
        egress_buf,
        server_args: server_args.clone(),
        egress_activity,
        live_rids,
    };
    // Each endpoint module registers its own routes and merges here.
    let app = Router::new()
        .merge(common::routes())
        .merge(native_api::routes())
        .merge(openai::routes())
        .merge(vertex::routes())
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
