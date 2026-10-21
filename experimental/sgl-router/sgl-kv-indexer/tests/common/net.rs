// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;

use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;

/// Binds an ephemeral loopback port and hands the socket to `serve_with_incoming`;
/// reserving a port and releasing it loses it to another test binary.
pub async fn bound_incoming() -> (SocketAddr, TcpListenerStream) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    (addr, TcpListenerStream::new(listener))
}
