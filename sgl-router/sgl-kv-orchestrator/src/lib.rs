//! Session KV orchestration and its versioned Engine control protocol.
mod client;
mod config;
mod feedback;
pub mod models;
pub mod observations;
pub mod operations;
pub mod protocol;
mod runtime;
pub mod session_index;
pub mod target_resolver;
pub use config::OrchestrationConfig;
pub use protocol::{encode_cache_salt, parse_action, Action, Envelope, ParsedAction};
pub use runtime::{KvContext, Runtime};
