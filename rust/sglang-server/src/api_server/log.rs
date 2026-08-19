//! HTTP access logging — one INFO line per request, content-matching the
//! Python server's uvicorn access log. Gated exactly like uvicorn's
//! (`--log-level-http warning` turns it off, see
//! `ServerArgs::http_access_log_enabled`); when disabled the middleware is not
//! installed at all — zero cost.

use axum::{Router, response::Response};

use crate::runtime::ServerArgs;

/// Install the access-log middleware when `server_args` enables it; identity
/// otherwise (the layer is never installed, so disabled stays zero-cost).
pub(super) fn apply(app: Router, server_args: &ServerArgs) -> Router {
    if server_args.http_access_log_enabled() {
        app.layer(axum::middleware::from_fn(access_log))
    } else {
        app
    }
}

/// Access log — one INFO line per request, content-matching the Python server's
/// uvicorn access log (`127.0.0.1:54232 - "GET /model_info HTTP/1.1" 200 OK`).
/// Logged when the response head is ready; for SSE that's stream start, same as
/// uvicorn.
async fn access_log(
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let version = req.version();
    let res = next.run(req).await;
    let status = res.status();
    tracing::info!(
        "{peer} - \"{method} {uri} {version:?}\" {} {}",
        status.as_u16(),
        status.canonical_reason().unwrap_or("")
    );
    res
}
