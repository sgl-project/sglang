//! Socket helpers for the API listener.

use std::net::SocketAddr;
use std::net::TcpListener;

const BACKLOG: i32 = 2048; // matches uvicorn's default (asyncio's own is 100)
const RECV_BUF_SIZE: usize = 16 * 1024 * 1024;

/// Bind and tune the API listener, returning it ready for
/// `tokio::net::TcpListener::from_std`.
///
/// socket2 rather than `TcpListener` so options (SO_RCVBUF, ...) can
/// be set before `listen`.
pub fn bind_tcp_listener(addr: SocketAddr) -> Result<TcpListener, String> {
    let socket = socket2::Socket::new(
        socket2::Domain::for_address(addr),
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )
    .map_err(|e| format!("socket for {addr} failed: {e}"))?;
    socket
        .set_reuse_address(true)
        .map_err(|e| format!("set_reuseaddr failed: {e}"))?;
    socket
        .set_recv_buffer_size(RECV_BUF_SIZE)
        .map_err(|e| format!("set_recv_buffer_size failed: {e}"))?;
    // Matches Python, accepted sockets inherit TCP_NODELAY.
    socket
        .set_tcp_nodelay(true)
        .map_err(|e| format!("set_tcp_nodelay failed: {e}"))?;
    socket
        .bind(&addr.into())
        .map_err(|e| format!("bind {addr} failed: {e}"))?;
    socket
        .listen(BACKLOG)
        .map_err(|e| format!("listen on {addr} failed: {e}"))?;
    let listener: std::net::TcpListener = socket.into();
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("listener set_nonblocking failed: {e}"))?;
    Ok(listener)
}
