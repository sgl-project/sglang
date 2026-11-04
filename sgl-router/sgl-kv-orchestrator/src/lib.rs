//! Versioned KV protocol, bounded session associations, and backend protection.
pub mod backend_protection;
pub mod models;
pub mod observations;
pub mod protocol;
pub mod session_index;
pub mod target_resolver;
pub use protocol::{encode_cache_salt, parse_action, Action, Envelope, ParsedAction};
