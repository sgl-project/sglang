//! Socket helpers for the API listener.

use std::io;
use std::net::SocketAddr;
use std::net::TcpListener;

const BACKLOG: i32 = 2048; // matches uvicorn's default (asyncio's own is 100)
const RECV_BUF_SIZE: usize = 16 * 1024 * 1024;

/// Bind and tune the API listener, returning it ready for
/// `tokio::net::TcpListener::from_std`.
///
/// socket2 rather than `TcpListener` so options (SO_RCVBUF, ...) can
/// be set before `listen`.
pub fn bind_tcp_listener(addr: SocketAddr) -> io::Result<TcpListener> {
    let socket = socket2::Socket::new(
        socket2::Domain::for_address(addr),
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )?;
    socket.set_reuse_address(true)?;
    if let Err(e) = socket.set_recv_buffer_size(RECV_BUF_SIZE) {
        tracing::warn!(
            "set_recv_buffer_size({RECV_BUF_SIZE}) failed: {e}; continuing with the default size"
        );
    }
    // Matches Python, accepted sockets inherit TCP_NODELAY.
    if let Err(e) = socket.set_tcp_nodelay(true) {
        tracing::warn!("set_tcp_nodelay failed: {e}; continuing without TCP_NODELAY");
    }
    socket.bind(&addr.into())?;
    socket.listen(BACKLOG)?;
    let listener: std::net::TcpListener = socket.into();
    listener.set_nonblocking(true)?;
    Ok(listener)
}
