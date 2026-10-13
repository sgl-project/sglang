//! Tonic adapter for the native `sglang.api.v1` service.
//!
//! The service is the gRPC rendering of the contract the native HTTP endpoints
//! serve: `Generate` is `/generate`, `HealthCheck` is `/health_generate`, and
//! `GetModelInfo` / `GetServerInfo` are their HTTP namesakes. Requests decode
//! through the shared `/generate` fan-out and responses stream through the
//! shared [`CoreHandle`], which owns preprocessing, admission, runtime
//! communication, and request cancellation.
//!
//! The thin Tonic-service structure, streamed response approach.

pub(crate) mod app;
mod info;
mod native_api;
pub(crate) mod service;
