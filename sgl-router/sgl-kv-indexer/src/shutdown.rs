// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Termination signalling shared by the server and bridge binaries.

use tracing::{info, warn};

/// Resolves once the process has been asked to terminate, on `SIGTERM` (which
/// container runtimes send before escalating to `SIGKILL`) or `SIGINT`.
///
/// A handler that cannot be installed never resolves: firing immediately would
/// shut the process down at startup instead of leaving it running unsupervised.
pub async fn shutdown_signal() {
    let interrupt = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!(%error, "cannot handle SIGINT");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(error) => {
                warn!(%error, "cannot handle SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = interrupt => info!("received SIGINT; shutting down"),
        () = terminate => info!("received SIGTERM; shutting down"),
    }
}
