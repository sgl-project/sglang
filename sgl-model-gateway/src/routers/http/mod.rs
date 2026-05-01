//! HTTP router implementations

pub mod pd_router;
pub mod pd_types;
pub mod router;

use axum::response::Response;
use bytes::Bytes;
use serde_json::Value;
use tracing::warn;

use crate::{routers::error, sglang_extensions::SglangExtensions};

/// Parse SGLang extension fields from the raw request bytes. A type
/// mismatch (e.g. `return_routed_experts: "yes"`) is surfaced as a 400
/// `invalid_sglang_extension` instead of silently defaulting later. An
/// absent `body_raw` produces the all-defaults extensions, which is the
/// right answer for callers that didn't read the original bytes
/// (e.g. gRPC ingress that arrives as a typed proto).
pub(crate) fn parse_sglang_extensions(
    body_raw: Option<&Bytes>,
) -> Result<SglangExtensions, Response> {
    match body_raw {
        Some(raw) => SglangExtensions::parse(raw).map_err(|e| {
            error::bad_request(
                "invalid_sglang_extension",
                format!("Invalid SGLang extension field: {e}"),
            )
        }),
        None => Ok(SglangExtensions::default()),
    }
}

/// Parse `body_raw` (the original request bytes) into a JSON `Value` so
/// SGLang extension fields the typed deserializer dropped survive when
/// the gateway forwards the request to a backend. Returns `None` when
/// `body_raw` is absent, or — with a `warn!` — when the bytes don't
/// parse as JSON; callers fall back to typed serialization.
///
/// In production the parse-failure path is unreachable: HTTP ingress at
/// `server.rs` only invokes routers after `serde_json::from_slice` has
/// already produced a typed request from the same bytes. The `warn!`
/// exists to surface that impossible state if a future internal caller
/// passes raw bytes through a different path.
pub(crate) fn body_raw_to_value(body_raw: Option<&Bytes>) -> Option<Value> {
    let raw = body_raw?;
    match serde_json::from_slice(raw) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!(
                "body_raw is not valid JSON ({}); falling back to typed serialization",
                e
            );
            None
        }
    }
}
