//! Transport-neutral core shared by the API adapters: the entry point into the Rust request pipeline.
//!
//! Wire adapters normalize their protocol into the in-process
//! [`GenerateRequest`](crate::message::request::GenerateRequest), submit it
//! through [`CoreHandle`], and render semantic [`CoreEvent`]s for their
//! own transport. This module owns shared preprocessing, runtime translation,
//! capabilities, and request lifetime; it deliberately knows nothing about
//! Axum, HTTP response shapes, Tonic, or protobuf.

mod error;
mod event;
mod handle;
mod prefetch;

pub(crate) use error::{CoreError, CoreErrorKind};
pub(crate) use event::{CoreEvent, CoreOutput, HealthStatus};
pub(crate) use handle::{CoreCall, CoreConfig, CoreHandle, CoreMetadata, recv_indexed};
