//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime. Only
//! this module knows HTTP, so other protocols can mount the same `AppState`.
//! `/generate` submits a `Request` then awaits one `Done` (unary) or relays SSE
//! frames (`data: {json}` … `[DONE]`), byte-compatible with Python
//! `http_server.generate_request`; `/server_info` reuses it for one control result.
pub mod app;
mod common;
mod disaggregation;
mod frame;
mod guard;
mod log;
mod native_api;
mod openai;
mod prefetch;
mod submit;
