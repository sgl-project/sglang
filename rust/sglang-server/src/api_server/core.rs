//! What the transport-neutral handlers (`api_server::handlers`) are built from:
//! the error categories, the semantic [`CoreEvent`]s and their rendering into
//! the generated `api.v1` response types, the DP-rank fan-out and its peers,
//! media prefetch, and a model package's admission check. Deliberately knows
//! nothing about Axum, SSE framing, or Tonic.

mod drain;
mod error;
pub(crate) mod fanout;
mod frame;
mod metrics;
mod peer;
pub(crate) mod prefetch;
mod request_log;
mod validator;

pub(crate) use drain::{DrainEnd, DrainState, InFlight};
pub(crate) use error::{CoreError, CoreErrorKind, STREAM_INVALID_REQUEST_ERROR};
#[cfg(test)]
pub(crate) use frame::frame_metadata_case;
pub(crate) use frame::{CoreEvent, CoreOutput, HealthStatus};
pub use metrics::{ExtensionCounter, ExtensionHistogram};
pub(crate) use metrics::{FLUSH_INTERVAL, FrontendMetrics, RequestMetrics};
pub(crate) use peer::{FORWARD_PEER_PATH, peer_endpoints};
pub(crate) use request_log::{
    BodySnapshot, LoggedHeaders, OpenAiRequest, RawBody, ReceivedAs, RequestLog, RequestRecord,
    flush_request_log,
};
pub use validator::GenerateValidator;
#[cfg(test)]
pub(crate) use validator::TopLogprobsCap;
