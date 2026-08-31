//! Access logging as a transport-generic tower layer — one INFO line per
//! request, content-matching the Python server's uvicorn access log. Gated
//! exactly like uvicorn's (`--log-level-http warning` turns it off, see
//! `ServerArgs::http_access_log_enabled`); when disabled the layer is not
//! installed at all — zero cost.

use std::task::{Context, Poll};

use tower::{Layer, Service};

#[derive(Clone)]
pub(crate) struct AccessLogLayer;

impl<S> Layer<S> for AccessLogLayer {
    type Service = AccessLog<S>;
    fn layer(&self, inner: S) -> Self::Service {
        AccessLog { inner }
    }
}

#[derive(Clone)]
pub(crate) struct AccessLog<S> {
    inner: S,
}

/// The uvicorn-format line (`127.0.0.1:54232 - "GET /model_info HTTP/1.1"
/// 200 OK`), factored out so the byte-shape is pinned by a unit test.
fn access_line(
    peer: Option<std::net::SocketAddr>,
    method: &http::Method,
    uri: &http::Uri,
    version: http::Version,
    status: http::StatusCode,
) -> String {
    let peer = peer
        .map(|p| p.to_string())
        .unwrap_or_else(|| "-".to_string());
    format!(
        "{peer} - \"{method} {uri} {version:?}\" {} {}",
        status.as_u16(),
        status.canonical_reason().unwrap_or("")
    )
}

/// The peer address, whichever transport recorded it: the HTTP accept loop
/// inserts [`crate::api_server::layers::Peer`], tonic's transport inserts
/// `TcpConnectInfo`.
fn peer_addr<B>(req: &http::Request<B>) -> Option<std::net::SocketAddr> {
    if let Some(crate::api_server::layers::Peer(peer)) =
        req.extensions().get::<crate::api_server::layers::Peer>()
    {
        return Some(*peer);
    }
    req.extensions()
        .get::<tonic::transport::server::TcpConnectInfo>()
        .and_then(|info| info.remote_addr())
}

impl<S, B, RB> Service<http::Request<B>> for AccessLog<S>
where
    S: Service<http::Request<B>, Response = http::Response<RB>>,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        let peer = peer_addr(&req);
        let method = req.method().clone();
        let uri = req.uri().clone();
        let version = req.version();
        let fut = self.inner.call(req);
        Box::pin(async move {
            // Logged when the response head is ready; for a stream that's
            // stream start, same as uvicorn.
            let res = fut.await?;
            tracing::info!(
                "{}",
                access_line(peer, &method, &uri, version, res.status())
            );
            Ok(res)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line is byte-for-byte what api_server/log.rs (and uvicorn) logged:
    /// `{peer} - "{method} {uri} {version:?}" {code} {reason}`.
    #[test]
    fn line_matches_uvicorn_format() {
        let line = access_line(
            Some("127.0.0.1:54232".parse().unwrap()),
            &http::Method::GET,
            &"/model_info".parse().unwrap(),
            http::Version::HTTP_11,
            http::StatusCode::OK,
        );
        assert_eq!(
            line,
            "127.0.0.1:54232 - \"GET /model_info HTTP/1.1\" 200 OK"
        );
    }
}
