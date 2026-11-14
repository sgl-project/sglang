//! The HTTP listener's accept loop. Every connection speaks HTTP/1.1; with
//! `--enable-http2` (`ServerArgs::http2`) a connection that opens with the
//! HTTP/2 preface is served as HTTP/2 over cleartext (h2c, prior knowledge)
//! on the same port, as Python's Granian server serves `--enable-http2`.

use std::future::Future;
use std::net::SocketAddr;

use axum::Router;
use axum::extract::ConnectInfo;
use axum::serve::Listener;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tower::Layer;

use crate::message::config::Http2Config;

// Granian's `HTTP2Settings` defaults, which Python's `--enable-http2` server
// runs with past the two settings its flags set.
const INITIAL_STREAM_WINDOW_SIZE: u32 = 1024 * 1024;
const MAX_FRAME_SIZE: u32 = 16 * 1024;
const MAX_HEADER_LIST_SIZE: u32 = 16 * 1024 * 1024;
const MAX_SEND_BUF_SIZE: usize = 400 * 1024;

/// The connection builder: HTTP/1.1 only, or also HTTP/2 with `http2`'s
/// settings and Granian's defaults for the rest (no adaptive window, no
/// keep-alive pings). Like `axum::serve`'s, it has no timer, so HTTP/1.1 has
/// no header-read timeout and idle keep-alive connections stay open.
fn builder(http2: Option<&Http2Config>) -> auto::Builder<TokioExecutor> {
    let mut builder = auto::Builder::new(TokioExecutor::new());
    let Some(http2) = http2 else {
        return builder.http1_only();
    };
    builder
        .http2()
        .max_concurrent_streams(http2.max_concurrent_streams)
        .initial_connection_window_size(http2.initial_connection_window_size)
        .initial_stream_window_size(INITIAL_STREAM_WINDOW_SIZE)
        .max_frame_size(MAX_FRAME_SIZE)
        .max_header_list_size(MAX_HEADER_LIST_SIZE)
        .max_send_buf_size(MAX_SEND_BUF_SIZE)
        .adaptive_window(false)
        .keep_alive_interval(None);
    builder
}

/// Serve `app` on `listener` until `signal` resolves, then stop accepting,
/// let the open connections finish their requests (idle ones close at once),
/// and return once every connection closed, as `axum::serve`'s graceful
/// shutdown does. Each request carries the peer's `ConnectInfo<SocketAddr>`.
///
/// `listener` handles accept errors as `axum::serve` does: one that failed
/// before it was accepted is skipped, anything else (out of file descriptors,
/// say) is logged and retried after a second. The retry runs inside the
/// accept future, so `signal` cuts it short.
pub(super) async fn serve(
    mut listener: impl Listener<Addr = SocketAddr>,
    app: Router,
    http2: Option<&Http2Config>,
    signal: impl Future<Output = ()>,
) {
    let builder = builder(http2);
    let graceful = GracefulShutdown::new();
    let mut signal = std::pin::pin!(signal);
    loop {
        let (stream, peer) = tokio::select! {
            accepted = listener.accept() => accepted,
            () = &mut signal => break,
        };
        let service =
            TowerToHyperService::new(axum::Extension(ConnectInfo(peer)).layer(app.clone()));
        let connection = graceful.watch(
            builder
                .serve_connection(TokioIo::new(stream), service)
                .into_owned(),
        );
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::trace!(%peer, "connection ended with an error: {e:#}");
            }
        });
    }
    drop(listener);
    graceful.shutdown().await;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::Notify;

    use super::*;

    /// A listener whose accept failed and is waiting out its retry delay.
    struct BackingOff(Arc<Notify>);

    impl Listener for BackingOff {
        type Io = tokio::net::TcpStream;
        type Addr = SocketAddr;

        async fn accept(&mut self) -> (Self::Io, Self::Addr) {
            self.0.notify_one();
            std::future::pending().await
        }

        fn local_addr(&self) -> std::io::Result<Self::Addr> {
            unimplemented!()
        }
    }

    /// A drain that arrives during an accept-error retry delay stops the
    /// listener at once rather than after the delay.
    #[tokio::test]
    async fn drain_cuts_an_accept_error_retry_short() {
        let backing_off = Arc::new(Notify::new());
        let signal = {
            let backing_off = backing_off.clone();
            async move { backing_off.notified().await }
        };
        let serve = serve(BackingOff(backing_off), Router::new(), None, signal);
        tokio::time::timeout(Duration::from_secs(5), serve)
            .await
            .expect("the drain waited out the accept retry");
    }
}
