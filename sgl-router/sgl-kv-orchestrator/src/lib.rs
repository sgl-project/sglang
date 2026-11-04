//! Pure protocol and bounded logical state. No Router, network, storage, or PIN executor.
pub mod models;
pub mod observations;
pub mod protocol;
pub mod session_index;
pub mod target_resolver;
pub use protocol::{encode_cache_salt, parse_action, Action, Envelope, ParsedAction};
