//! Transport-neutral API core. Endpoint logic lives here as plain typed
//! functions; the transport modules (`api_server` today, a gRPC server later)
//! adapt requests in and shape responses/errors out. Nothing in this tree may
//! depend on axum or tonic.

pub(crate) mod control;
pub(crate) mod error;
pub(crate) mod event;
pub(crate) mod frame;
pub(crate) mod generate;
pub(crate) mod guard;
pub(crate) mod health;
pub(crate) mod openai;
pub(crate) mod prefetch;
pub(crate) mod state;
pub(crate) mod submit;
