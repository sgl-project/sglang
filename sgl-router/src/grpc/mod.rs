//! gRPC client module for communicating with SGLang scheduler
//!
//! This module provides a gRPC client implementation for the SGLang router.

pub mod client;

// Re-export the client
pub use client::{proto, SglangSchedulerClient};
