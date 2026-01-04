//! MCP namespacing and identity management.
//!
//! This module provides structures and functions for managing server identities
//! and creating qualified names for tools and prompts to avoid collisions between
//! different MCP servers.

/// Holds the distinct identifiers for an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ServerIdentity {
    /// The unique key for connection pooling and eviction (e.g., a URL).
    pub key: String,
    /// The user-facing label for qualifying tool names (e.g., a custom label).
    pub label: String,
}

/// Qualify a name with a label to create a unique name across all servers.
pub fn qualify_name(label: &str, name: &str) -> String {
    format!("{}_{}", label, name)
}
