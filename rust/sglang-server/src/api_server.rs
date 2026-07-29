//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime. Only
//! this module knows HTTP, so other protocols can mount the same `AppState`.
//! `/generate` submits a `Request` then awaits one `Done` (unary) or relays SSE
//! frames (`data: {json}` … `[DONE]`), byte-compatible with Python
//! `http_server.generate_request`; `/server_info` reuses it for one control result.

mod frame;
mod guard;
mod log;
mod submit;

use std::sync::Arc;

use crate::runtime::ServerArgs;
use crate::tokenizer_manager::ActivityCounter;
use crate::tokenizer_manager::Senders;

/// Shared handler state: the submit machinery (`senders`, `egress_buf`)
/// + shared tokenizer.
#[derive(Clone)]
struct AppState {
    senders: Senders,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    /// Egress heartbeat (bumped per drained ring frame).
    egress_activity: ActivityCounter,
}
