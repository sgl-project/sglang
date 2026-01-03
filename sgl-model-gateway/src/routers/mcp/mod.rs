//! Shared MCP (Model Context Protocol) utilities for routers.
//!
//! This module provides shared MCP-related functionality that can be
//! used across different router implementations (OpenAI, gRPC regular, gRPC harmony).

mod mcp_connection;

pub use mcp_connection::ensure_request_mcp_client;
