//! The HTTP transport (axum + tower / tokio). I/O-bound; own pinned multi-thread
//! runtime. Only this module knows HTTP: endpoint logic lives in
//! `api_server::core`, whose shared `CoreState` another transport can mount
//! unchanged. `/generate` submits a `Request` then awaits one `Done` (unary) or
//! relays SSE frames (`data: {json}` … `[DONE]`), byte-compatible with Python
//! `http_server.generate_request`; `/server_info` reuses it for one control result.

pub(crate) mod app;
mod common;
mod disaggregation;
mod native_api;
mod openai;
pub(crate) mod response;
