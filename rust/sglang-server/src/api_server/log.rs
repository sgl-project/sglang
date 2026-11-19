//! What the API server logs: one access line per HTTP request, as uvicorn
//! writes it, and Python's request log and crash-dump record, taken at each
//! request's boundaries.

pub(crate) mod access;
mod request;

pub(crate) use request::{
    BodySnapshot, LoggedHeaders, OpenAiRequest, RawBody, ReceivedAs, RequestLog, RequestRecord,
    flush_request_log,
};
