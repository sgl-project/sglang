//! What the transport-neutral handlers (`api_server::handlers`) are built from:
//! the error categories, the semantic [`CoreEvent`]s and their rendering into
//! the generated `api.v1` response types, the DP-rank fan-out and its peers,
//! and media prefetch. Deliberately knows nothing about Axum, SSE framing, or
//! Tonic.

mod error;
pub(crate) mod fanout;
mod frame;
mod peer;
pub(crate) mod prefetch;

pub(crate) use error::{CoreError, CoreErrorKind};
pub(crate) use frame::{CoreEvent, CoreOutput, HealthStatus};
pub(crate) use peer::{FORWARD_PEER_PATH, peer_endpoints};
