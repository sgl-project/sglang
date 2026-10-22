//! Transport-neutral core shared by the API adapters: the entry point into the Rust request pipeline.
//!
//! Wire adapters normalize their protocol into the in-process
//! [`GenerateRequest`](crate::message::request::GenerateRequest), submit it
//! through [`CoreHandle`], and render semantic [`CoreEvent`]s for their
//! own transport. This module owns shared preprocessing, runtime translation,
//! capabilities, request lifetime, and the rendering of outputs into the
//! generated `api.v1` response types that both adapters put on the wire; it
//! deliberately knows nothing about Axum, SSE framing, or Tonic.

mod error;
mod frame;
mod handle;
mod prefetch;

pub(crate) use error::{CoreError, CoreErrorKind};
pub(crate) use frame::{CoreEvent, CoreOutput, HealthStatus};
pub(crate) use handle::{CoreCall, CoreConfig, CoreHandle, CoreMetadata, recv_indexed};
