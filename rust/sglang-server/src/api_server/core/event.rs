//! The typed item of a core response stream: a response chunk, or a
//! request-scoped failure the transport shapes itself (an SSE error frame on
//! HTTP; a status or error message on gRPC). Transport framing — SSE `data:`
//! lines, the `[DONE]` sentinel — never appears here.

use crate::api_server::core::error::ApiError;

pub(crate) enum CoreEvent<T> {
    Item(T),
    /// One request in the stream failed; the stream itself continues (other
    /// batch items may still be running).
    ItemError(ApiError),
}
