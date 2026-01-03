//! Shared responses module.
//!
//! This module provides shared utilities for the Responses API that can be
//! used across different router implementations (OpenAI, gRPC regular, gRPC harmony).

mod mcp_connection;

pub use mcp_connection::ensure_request_mcp_client;
