//! HTTP clients for external service communication
//!
//! This module contains HTTP REST API clients used for communicating with
//! external services that don't use gRPC.

mod encode;

pub use encode::{EncodeError, EncodeHttpClient, EncodeRequest, EncodeResponse};
