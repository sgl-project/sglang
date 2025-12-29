//! Authentication and authorization module for control plane APIs.
//!
//! This module provides:
//! - JWT/OIDC authentication for external IDP integration
//! - API key authentication with role-based access
//! - Audit logging for control plane operations
//! - Middleware for securing admin and worker routes

mod audit;
mod config;
mod jwks;
mod jwt;
mod middleware;

pub use audit::{AuditEvent, AuditLogger, AuditOutcome};
pub use config::{ApiKeyEntry, ControlPlaneAuthConfig, JwtConfig, Role};
pub use jwt::{JwtValidator, JwtValidatorError};
pub use middleware::{
    control_plane_auth_middleware, ControlPlaneAuthState, Principal, PrincipalExt,
};
