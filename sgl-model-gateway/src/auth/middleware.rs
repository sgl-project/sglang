//! Control plane authentication middleware.
//!
//! Provides middleware for authenticating and authorizing access to control plane APIs.
//! Supports both JWT/OIDC tokens and API keys.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use tracing::{debug, error, info, warn};

use super::{
    audit::{AuditContext, AuditLogger},
    config::{ControlPlaneAuthConfig, Role},
    jwt::JwtValidator,
};
use crate::middleware::RequestId;

/// Authenticated principal information.
#[derive(Debug, Clone)]
pub struct Principal {
    /// Subject identifier (user ID, email, or API key ID)
    pub id: String,

    /// Display name if available
    pub name: Option<String>,

    /// Authentication method used
    pub auth_method: AuthMethod,

    /// Assigned role
    pub role: Role,
}

/// Authentication method used to authenticate the principal.
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// JWT/OIDC token from external IDP
    Jwt { issuer: String },
    /// API key for service accounts
    ApiKey { key_id: String },
}

impl std::fmt::Display for AuthMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthMethod::Jwt { issuer } => write!(f, "jwt:{}", issuer),
            AuthMethod::ApiKey { key_id } => write!(f, "api_key:{}", key_id),
        }
    }
}

/// Extension trait for extracting Principal from request extensions.
pub trait PrincipalExt {
    fn principal(&self) -> Option<&Principal>;
}

impl<B> PrincipalExt for http::Request<B> {
    fn principal(&self) -> Option<&Principal> {
        self.extensions().get::<Principal>()
    }
}

/// State for the control plane authentication middleware.
#[derive(Clone)]
pub struct ControlPlaneAuthState {
    /// Authentication configuration
    pub config: ControlPlaneAuthConfig,

    /// JWT validator (if JWT auth is configured)
    pub jwt_validator: Option<Arc<JwtValidator>>,

    /// Audit logger
    pub audit_logger: AuditLogger,
}

impl ControlPlaneAuthState {
    /// Create a new control plane auth state.
    pub fn new(config: ControlPlaneAuthConfig, jwt_validator: Option<Arc<JwtValidator>>) -> Self {
        let audit_logger = AuditLogger::new(config.audit_enabled);
        Self {
            config,
            jwt_validator,
            audit_logger,
        }
    }

    /// Create from config, initializing JWT validator if needed.
    pub async fn from_config(
        config: ControlPlaneAuthConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let jwt_validator = if let Some(jwt_config) = &config.jwt {
            Some(Arc::new(
                JwtValidator::from_config(jwt_config.clone()).await?,
            ))
        } else {
            None
        };

        Ok(Self::new(config, jwt_validator))
    }

    /// Try to initialize control plane auth from config.
    ///
    /// Returns `Some(state)` if auth is configured and initialized successfully.
    /// Returns `None` if auth is not configured or initialization fails (with error logged).
    pub async fn try_init(config: Option<&ControlPlaneAuthConfig>) -> Option<Self> {
        let config = config.filter(|c| c.is_enabled())?;

        info!("Initializing control plane authentication...");
        match Self::from_config(config.clone()).await {
            Ok(state) => {
                if config.has_jwt() {
                    info!("Control plane JWT/OIDC authentication enabled");
                }
                if config.has_api_keys() {
                    info!(
                        "Control plane API key authentication enabled ({} keys)",
                        config.api_keys.len()
                    );
                }
                if config.audit_enabled {
                    info!("Control plane audit logging enabled");
                }
                Some(state)
            }
            Err(e) => {
                error!(
                    "Failed to initialize control plane auth: {}. Falling back to simple API key auth.",
                    e
                );
                None
            }
        }
    }

    /// Check if authentication is required.
    pub fn is_auth_required(&self) -> bool {
        self.config.is_enabled()
    }
}

/// Check admin role and log denial if not admin.
/// Returns Some(Response) if denied, None if allowed.
fn check_admin_role(
    principal_id: &str,
    auth_method: &str,
    role: Role,
    method: &str,
    path: &str,
    request_id: Option<&str>,
    audit_logger: &AuditLogger,
) -> Option<Response> {
    if role.is_admin() {
        return None;
    }

    warn!(
        "{} {} has role {:?} but admin is required for control plane access",
        auth_method, principal_id, role
    );
    let ctx = AuditContext::new(principal_id, auth_method, role, method, path, request_id);
    audit_logger.log_denied(&ctx, "Admin role required for control plane access");

    Some(
        (
            StatusCode::FORBIDDEN,
            "Admin role required for control plane access",
        )
            .into_response(),
    )
}

/// Log successful authentication.
fn log_auth_success(
    principal: &Principal,
    auth_method: &str,
    method: &str,
    path: &str,
    request_id: Option<&str>,
    audit_logger: &AuditLogger,
) {
    debug!(
        "{} authentication successful for {} with role {:?}",
        auth_method, principal.id, principal.role
    );
    let ctx = AuditContext::new(
        &principal.id,
        auth_method,
        principal.role,
        method,
        path,
        request_id,
    );
    audit_logger.log_success(&ctx, None);
}

/// Control plane authentication middleware.
///
/// This middleware:
/// 1. Extracts the Bearer token from the Authorization header
/// 2. Attempts JWT validation first (if configured)
/// 3. Falls back to API key validation (if configured)
/// 4. Checks if the authenticated principal has admin role
/// 5. Logs audit events for control plane access
///
/// Returns 401 Unauthorized if authentication fails.
/// Returns 403 Forbidden if the user doesn't have admin role.
pub async fn control_plane_auth_middleware(
    State(auth_state): State<ControlPlaneAuthState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // If no authentication is configured, allow through (backward compatibility)
    if !auth_state.is_auth_required() {
        return next.run(request).await;
    }

    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let request_id = request.extensions().get::<RequestId>().map(|r| r.0.clone());

    // Extract Bearer token from Authorization header
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));

    let Some(token) = token else {
        debug!("Missing or invalid Authorization header for control plane API");
        auth_state.audit_logger.log_auth_failure(
            &method,
            &path,
            "Missing or invalid Authorization header",
            request_id.as_deref(),
        );
        return (
            StatusCode::UNAUTHORIZED,
            [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
            "Missing or invalid Authorization header",
        )
            .into_response();
    };

    // Try JWT validation first
    if let Some(jwt_validator) = &auth_state.jwt_validator {
        match jwt_validator.validate(token).await {
            Ok(validated_token) => {
                if let Some(resp) = check_admin_role(
                    &validated_token.subject,
                    "jwt",
                    validated_token.role,
                    &method,
                    &path,
                    request_id.as_deref(),
                    &auth_state.audit_logger,
                ) {
                    return resp;
                }

                let principal = Principal {
                    id: validated_token.subject.clone(),
                    name: validated_token.name.clone(),
                    auth_method: AuthMethod::Jwt {
                        issuer: validated_token.issuer.clone(),
                    },
                    role: validated_token.role,
                };

                log_auth_success(
                    &principal,
                    "jwt",
                    &method,
                    &path,
                    request_id.as_deref(),
                    &auth_state.audit_logger,
                );
                request.extensions_mut().insert(principal);
                return next.run(request).await;
            }
            Err(e) => {
                // If the token looks like a JWT (3 parts separated by dots), it's likely
                // an invalid JWT, not an API key. Fail fast with a specific error
                // instead of silently falling back. This provides better feedback.
                if token.split('.').count() == 3 {
                    warn!("Invalid JWT provided: {}. Not falling back to API key.", e);
                    auth_state.audit_logger.log_auth_failure(
                        &method,
                        &path,
                        &format!("Invalid JWT: {}", e),
                        request_id.as_deref(),
                    );
                    return (
                        StatusCode::UNAUTHORIZED,
                        [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
                        format!("Invalid JWT: {}", e),
                    )
                        .into_response();
                }
                debug!("JWT validation failed: {}, trying API key", e);
            }
        }
    }

    // Try API key validation
    if let Some(api_key_entry) = auth_state.config.find_api_key(token) {
        if let Some(resp) = check_admin_role(
            &api_key_entry.id,
            "api_key",
            api_key_entry.role,
            &method,
            &path,
            request_id.as_deref(),
            &auth_state.audit_logger,
        ) {
            return resp;
        }

        let principal = Principal {
            id: api_key_entry.id.clone(),
            name: Some(api_key_entry.name.clone()),
            auth_method: AuthMethod::ApiKey {
                key_id: api_key_entry.id.clone(),
            },
            role: api_key_entry.role,
        };

        log_auth_success(
            &principal,
            "api_key",
            &method,
            &path,
            request_id.as_deref(),
            &auth_state.audit_logger,
        );
        request.extensions_mut().insert(principal);
        return next.run(request).await;
    }

    // Authentication failed
    debug!("Control plane authentication failed: invalid token");
    auth_state.audit_logger.log_auth_failure(
        &method,
        &path,
        "Invalid token",
        request_id.as_deref(),
    );

    (
        StatusCode::UNAUTHORIZED,
        [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
        "Invalid authentication token",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_method_display() {
        let jwt = AuthMethod::Jwt {
            issuer: "https://example.com".to_string(),
        };
        assert_eq!(jwt.to_string(), "jwt:https://example.com");

        let api_key = AuthMethod::ApiKey {
            key_id: "key-123".to_string(),
        };
        assert_eq!(api_key.to_string(), "api_key:key-123");
    }

    #[test]
    fn test_control_plane_auth_state_no_config() {
        let config = ControlPlaneAuthConfig::default();
        let state = ControlPlaneAuthState::new(config, None);
        assert!(!state.is_auth_required());
    }

    #[test]
    fn test_control_plane_auth_state_with_api_keys() {
        use super::super::config::ApiKeyEntry;

        let config = ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![ApiKeyEntry::new("test", "Test Key", "secret", Role::Admin)],
            audit_enabled: true,
        };
        let state = ControlPlaneAuthState::new(config, None);
        assert!(state.is_auth_required());
    }
}
