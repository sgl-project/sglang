//! Process-wide `tracing` setup for the embedded server.

use std::sync::OnceLock;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

/// Keeps the non-blocking log writer's background thread alive for the process
/// lifetime (dropping the guard would stop log delivery).
static LOG_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

/// Install the global `tracing` subscriber once; a no-op if the host process
/// (or an earlier call) already set one.
pub fn init_tracing() {
    let (writer, guard) = tracing_appender::non_blocking(std::io::stdout());
    let _ = LOG_GUARD.set(guard);
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(writer)
        .try_init();
}
