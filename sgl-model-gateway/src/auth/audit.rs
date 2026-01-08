//! Audit logging for control plane operations.
//!
//! Provides structured audit events for security monitoring and compliance.
//!
//! Security features:
//! - Input sanitization to prevent log injection attacks
//! - Structured logging for safe parsing

use chrono::{DateTime, Utc};
use serde::Serialize;
use tracing::{info, span, Level};

use super::config::Role;

/// Maximum length for sanitized strings to prevent log flooding
const MAX_SANITIZED_LENGTH: usize = 1024;

/// Sanitize a string for safe logging.
/// Removes/escapes control characters and newlines that could be used for log injection.
fn sanitize_for_log(input: &str) -> String {
    let mut result = String::with_capacity(input.len().min(MAX_SANITIZED_LENGTH));

    for ch in input.chars().take(MAX_SANITIZED_LENGTH) {
        match ch {
            // Replace newlines and carriage returns with escaped versions
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            // Replace tabs with escaped version
            '\t' => result.push_str("\\t"),
            // Escape backslashes
            '\\' => result.push_str("\\\\"),
            // Remove other control characters (ASCII 0-31, 127)
            c if c.is_control() => {
                result.push_str(&format!("\\x{:02x}", c as u32));
            }
            // Keep printable characters as-is
            c => result.push(c),
        }
    }

    // Indicate truncation if input was too long
    if input.len() > MAX_SANITIZED_LENGTH {
        result.push_str("...[TRUNCATED]");
    }

    result
}

/// Outcome of an audited operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditOutcome {
    /// Operation completed successfully
    Success,
    /// Operation denied (authorization failure)
    Denied,
}

impl std::fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditOutcome::Success => write!(f, "success"),
            AuditOutcome::Denied => write!(f, "denied"),
        }
    }
}

/// Audit event for control plane operations.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,

    /// Principal who performed the action (subject ID or API key ID)
    pub principal: String,

    /// Authentication method used (jwt, api_key)
    pub auth_method: String,

    /// Role of the principal
    pub role: Role,

    /// HTTP method
    pub method: String,

    /// Request path
    pub path: String,

    /// Resource being accessed (e.g., worker ID, wasm module ID)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,

    /// Operation outcome
    pub outcome: AuditOutcome,

    /// Request ID for correlation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Additional details or error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Context for audit logging containing request and principal information.
#[derive(Debug, Clone)]
pub struct AuditContext<'a> {
    /// Principal who performed the action
    pub principal: &'a str,
    /// Authentication method used (jwt, api_key)
    pub auth_method: &'a str,
    /// Role of the principal
    pub role: Role,
    /// HTTP method
    pub method: &'a str,
    /// Request path
    pub path: &'a str,
    /// Request ID for correlation
    pub request_id: Option<&'a str>,
}

impl<'a> AuditContext<'a> {
    /// Create a new audit context.
    pub fn new(
        principal: &'a str,
        auth_method: &'a str,
        role: Role,
        method: &'a str,
        path: &'a str,
        request_id: Option<&'a str>,
    ) -> Self {
        Self {
            principal,
            auth_method,
            role,
            method,
            path,
            request_id,
        }
    }
}

impl AuditEvent {
    /// Create a new audit event from context.
    fn from_context(
        ctx: &AuditContext<'_>,
        outcome: AuditOutcome,
        resource: Option<&str>,
        details: Option<&str>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            principal: sanitize_for_log(ctx.principal),
            auth_method: sanitize_for_log(ctx.auth_method),
            role: ctx.role,
            method: sanitize_for_log(ctx.method),
            path: sanitize_for_log(ctx.path),
            resource: resource.map(sanitize_for_log),
            outcome,
            request_id: ctx.request_id.map(sanitize_for_log),
            details: details.map(sanitize_for_log),
        }
    }

    /// Create an audit event for unauthenticated requests.
    fn unauthenticated(method: &str, path: &str, reason: &str, request_id: Option<&str>) -> Self {
        Self {
            timestamp: Utc::now(),
            principal: "unauthenticated".to_string(),
            auth_method: "none".to_string(),
            role: Role::User,
            method: sanitize_for_log(method),
            path: sanitize_for_log(path),
            resource: None,
            outcome: AuditOutcome::Denied,
            request_id: request_id.map(sanitize_for_log),
            details: Some(sanitize_for_log(reason)),
        }
    }
}

/// Audit logger that emits structured audit events.
#[derive(Clone, Default)]
pub struct AuditLogger {
    enabled: bool,
}

impl AuditLogger {
    /// Create a new audit logger.
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Check if audit logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Log an audit event.
    pub fn log(&self, event: &AuditEvent) {
        if !self.enabled {
            return;
        }

        // Create a span for structured logging
        let _span = span!(
            Level::INFO,
            "audit",
            principal = %event.principal,
            auth_method = %event.auth_method,
            role = %event.role,
            method = %event.method,
            path = %event.path,
            outcome = %event.outcome,
        )
        .entered();

        // Log the event
        info!(
            target: "smg::audit",
            timestamp = %event.timestamp.to_rfc3339(),
            principal = %event.principal,
            auth_method = %event.auth_method,
            role = %event.role,
            method = %event.method,
            path = %event.path,
            resource = ?event.resource,
            outcome = %event.outcome,
            request_id = ?event.request_id,
            details = ?event.details,
            "control_plane_audit"
        );
    }

    /// Log a successful operation.
    pub fn log_success(&self, ctx: &AuditContext<'_>, resource: Option<&str>) {
        self.log(&AuditEvent::from_context(
            ctx,
            AuditOutcome::Success,
            resource,
            None,
        ));
    }

    /// Log a denied operation (authorization failure).
    pub fn log_denied(&self, ctx: &AuditContext<'_>, reason: &str) {
        self.log(&AuditEvent::from_context(
            ctx,
            AuditOutcome::Denied,
            None,
            Some(reason),
        ));
    }

    /// Log an authentication failure (before principal is known).
    pub fn log_auth_failure(
        &self,
        method: &str,
        path: &str,
        reason: &str,
        request_id: Option<&str>,
    ) {
        self.log(&AuditEvent::unauthenticated(
            method, path, reason, request_id,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_from_context() {
        let ctx = AuditContext::new(
            "user@example.com",
            "jwt",
            Role::Admin,
            "POST",
            "/workers",
            Some("req-abc"),
        );
        let event = AuditEvent::from_context(&ctx, AuditOutcome::Success, Some("worker-123"), None);

        assert_eq!(event.principal, "user@example.com");
        assert_eq!(event.auth_method, "jwt");
        assert_eq!(event.role, Role::Admin);
        assert_eq!(event.method, "POST");
        assert_eq!(event.path, "/workers");
        assert_eq!(event.resource, Some("worker-123".to_string()));
        assert_eq!(event.outcome, AuditOutcome::Success);
        assert_eq!(event.request_id, Some("req-abc".to_string()));
    }

    #[test]
    fn test_audit_outcome_display() {
        assert_eq!(AuditOutcome::Success.to_string(), "success");
        assert_eq!(AuditOutcome::Denied.to_string(), "denied");
    }

    #[test]
    fn test_sanitize_for_log_normal_input() {
        assert_eq!(sanitize_for_log("normal string"), "normal string");
        assert_eq!(sanitize_for_log("user@example.com"), "user@example.com");
        assert_eq!(sanitize_for_log("/api/v1/workers"), "/api/v1/workers");
    }

    #[test]
    fn test_sanitize_for_log_newlines() {
        assert_eq!(sanitize_for_log("line1\nline2"), "line1\\nline2");
        assert_eq!(sanitize_for_log("line1\r\nline2"), "line1\\r\\nline2");
    }

    #[test]
    fn test_sanitize_for_log_control_chars() {
        assert_eq!(sanitize_for_log("test\x00null"), "test\\x00null");
        assert_eq!(sanitize_for_log("test\x1bescape"), "test\\x1bescape");
    }

    #[test]
    fn test_sanitize_for_log_backslashes() {
        assert_eq!(sanitize_for_log("path\\to\\file"), "path\\\\to\\\\file");
    }

    #[test]
    fn test_sanitize_for_log_truncation() {
        let long_string = "a".repeat(2000);
        let sanitized = sanitize_for_log(&long_string);
        assert!(sanitized.len() < 2000);
        assert!(sanitized.ends_with("...[TRUNCATED]"));
    }

    #[test]
    fn test_audit_event_sanitizes_inputs() {
        let ctx = AuditContext::new(
            "user\ninjected",
            "jwt",
            Role::User,
            "GET",
            "/workers\r\nfake_log_entry",
            None,
        );
        let event = AuditEvent::from_context(
            &ctx,
            AuditOutcome::Denied,
            None,
            Some("error\x00with\x1bnull"),
        );

        // Verify sanitization was applied
        assert!(!event.principal.contains('\n'));
        assert!(event.principal.contains("\\n"));
        assert!(!event.path.contains('\r'));
        assert!(event.path.contains("\\r"));
        assert!(event.details.as_ref().unwrap().contains("\\x00"));
    }
}
