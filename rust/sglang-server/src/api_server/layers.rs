//! Cross-cutting tower layers shared by both transports. Each is generic over
//! `http::Request<B>` / `http::Response<RB>`, so one implementation applies to
//! the axum router and tonic's `Server::builder().layer(...)` alike — the
//! transports differ only in their body types (see [`RejectionBody`]).

pub(crate) mod access_log;
pub(crate) mod auth;

/// The peer address, stamped as a request extension from axum's `ConnectInfo`
/// in `http::app::serve` (tonic's transport records its own `TcpConnectInfo`
/// instead).
#[derive(Clone, Copy)]
pub(crate) struct Peer(pub(crate) std::net::SocketAddr);

/// The one place a layer must construct a response body of the transport's
/// own type: an auth rejection. HTTP rejections carry a small JSON body; gRPC
/// rejections are trailers-only (the status rides the headers).
pub(crate) trait RejectionBody {
    fn empty() -> Self;
    fn from_static(bytes: &'static [u8]) -> Self;
}

impl RejectionBody for tonic::body::Body {
    fn empty() -> Self {
        tonic::body::Body::empty()
    }
    fn from_static(bytes: &'static [u8]) -> Self {
        tonic::body::Body::new(http_body_util::Full::new(bytes::Bytes::from_static(bytes)))
    }
}
