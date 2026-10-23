// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;

/// Reserves an ephemeral loopback port. The listener is dropped immediately,
/// so callers that spawn a server should retain their connect-retry loop.
pub fn free_addr() -> SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}
