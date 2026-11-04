//! P1 pure protocol/model foundation. No Router, network, storage, or PIN executor.
pub mod models;
pub mod protocol;
pub use protocol::{encode_cache_salt, parse_action, Action, Envelope, ParsedAction};
