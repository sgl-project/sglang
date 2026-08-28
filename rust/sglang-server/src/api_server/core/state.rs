//! The shared per-request state every transport mounts.

use std::sync::Arc;

use crate::api_server::core::openai::template::ChatFormatter;
use crate::message::config::ServerArgs;
use crate::tokenizer_manager::from_scheduler::ActivityCounter;
use crate::tokenizer_manager::wiring::Senders;

/// Shared handler state: submission handles, immutable server configuration,
/// and the API-owned chat formatter.
pub(crate) struct CoreState {
    pub(crate) senders: Senders,
    pub(crate) response_buf: usize,
    /// The API key both transports gate on (None = open).
    pub(crate) api_key: Option<String>,
    pub(crate) server_args: Arc<ServerArgs>,
    pub(crate) chat_formatter: Option<ChatFormatter>,
    /// Response heartbeat (bumped per drained ring frame).
    pub(crate) response_activity: ActivityCounter,
}
