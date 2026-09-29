// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared test harness re-exports.

pub mod cache_aware_fixture;
pub mod mock_worker;
pub mod streaming;

#[allow(dead_code)] // not every test file inspects forwarded rids
pub fn is_engine_shaped_rid(rid: &str) -> bool {
    rid.len() == 32
        && rid
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// A loopback URL that refuses connections. Keep the returned socket alive: it
/// holds the port bound without listening, so no parallel test can reuse it.
#[allow(dead_code)] // not every test file needs an unreachable upstream
pub fn refused_url() -> (String, tokio::net::TcpSocket) {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    (format!("http://{}", socket.local_addr().unwrap()), socket)
}
