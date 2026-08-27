//! HTTP error mapping around the protocol-neutral frontend submission handle.

use axum::{http::StatusCode, response::Response};
use tokio::sync::mpsc;

use super::app::AppState;
use super::native_api::native_error;
use crate::message::ids::Rid;
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;

/// Submit one request and map an unavailable frontend to the native HTTP/SSE
/// error shape. Identity, response registration, and FSM intake live in
/// [`crate::frontend::FrontendHandle`].
pub(super) async fn submit(
    state: &AppState,
    kind: RequestKind,
    // `stream`: the client is reading an SSE stream, so it expects 200 plus an
    // error frame rather than a 4xx — `utils::response::error_response`'s rule.
    stream: bool,
) -> Result<(Rid, mpsc::Receiver<ResponseItem>), Response> {
    state.frontend.submit(kind).await.map_err(|_| {
        native_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            stream,
        )
    })
}
