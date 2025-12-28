//! Configuration types for control plane authentication.
//!
//! Security features:
//! - API keys are hashed at load time (never stored in plaintext in memory)
//! - Constant-time comparison for API key validation

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// Role-based access control for control plane APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Full access to all control plane APIs (workers, wasm, tokenizers, etc.)
    Admin,
    /// Access to inference/data plane APIs only (default for backward compatibility)
    #[default]
    User,
}

impl Role {
    /// Check if this role has admin privileges for control plane APIs.
    pub fn is_admin(&self) -> bool {
        matches!(self, Role::Admin)
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::Admin => write!(f, "admin"),
            Role::User => write!(f, "user"),
        }
    }
}

impl std::str::FromStr for Role {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "admin" => Ok(Role::Admin),
            "user" => Ok(Role::User),
            _ => Err(format!("Invalid role: {}. Valid roles: admin, user", s)),
        }
    }
}

/// JWT/OIDC configuration for external identity provider integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    /// OIDC issuer URL (e.g., "https://login.microsoftonline.com/{tenant}/v2.0")
    /// Used to discover JWKS endpoint via .well-known/openid-configuration
    pub issuer: String,

    /// Expected audience claim (usually the client ID or API identifier)
    pub audience: String,

    /// Optional explicit JWKS URI. If not provided, discovered from issuer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwks_uri: Option<String>,

    /// Claim name containing the role (default: "role" or "roles")
    #[serde(default = "default_role_claim")]
    pub role_claim: String,

    /// Mapping from IDP role/group names to gateway roles.
    /// Example: {"Gateway.Admin": "admin", "Gateway.User": "user"}
    #[serde(default)]
    pub role_mapping: HashMap<String, Role>,

    /// Clock skew tolerance in seconds (default: 30)
    #[serde(default = "default_leeway_secs")]
    pub leeway_secs: u64,

    /// JWKS cache TTL in seconds (default: 3600 = 1 hour)
    #[serde(default = "default_jwks_cache_ttl_secs")]
    pub jwks_cache_ttl_secs: u64,
}

fn default_role_claim() -> String {
    "roles".to_string()
}

fn default_leeway_secs() -> u64 {
    30
}

fn default_jwks_cache_ttl_secs() -> u64 {
    3600
}

impl JwtConfig {
    /// Create a new JWT config with required fields.
    pub fn new(issuer: impl Into<String>, audience: impl Into<String>) -> Self {
        Self {
            issuer: issuer.into(),
            audience: audience.into(),
            jwks_uri: None,
            role_claim: default_role_claim(),
            role_mapping: HashMap::new(),
            leeway_secs: default_leeway_secs(),
            jwks_cache_ttl_secs: default_jwks_cache_ttl_secs(),
        }
    }

    /// Set explicit JWKS URI instead of using OIDC discovery.
    pub fn with_jwks_uri(mut self, jwks_uri: impl Into<String>) -> Self {
        self.jwks_uri = Some(jwks_uri.into());
        self
    }

    /// Add a role mapping from IDP role to gateway role.
    pub fn with_role_mapping(mut self, idp_role: impl Into<String>, gateway_role: Role) -> Self {
        self.role_mapping.insert(idp_role.into(), gateway_role);
        self
    }
}

/// Hash an API key using SHA-256.
/// Returns the hash as a fixed-size byte array.
fn hash_api_key(key: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hasher.finalize().into()
}

/// API key entry for service account authentication.
/// The key is hashed at construction time and never stored in plaintext.
#[derive(Clone, Serialize, Deserialize)]
pub struct ApiKeyEntry {
    /// Unique identifier for this key
    pub id: String,

    /// Human-readable name/description
    pub name: String,

    /// SHA-256 hash of the API key (never store plaintext)
    #[serde(skip)]
    key_hash: [u8; 32],

    /// Role assigned to this API key
    #[serde(default)]
    pub role: Role,
}

impl std::fmt::Debug for ApiKeyEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyEntry")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("key_hash", &"[REDACTED]")
            .field("role", &self.role)
            .finish()
    }
}

impl ApiKeyEntry {
    /// Create a new API key entry.
    /// The key is immediately hashed and the plaintext is not stored.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        key: impl Into<String>,
        role: Role,
    ) -> Self {
        let key_str = key.into();
        Self {
            id: id.into(),
            name: name.into(),
            key_hash: hash_api_key(&key_str),
            role,
        }
    }

    /// Check if the provided key matches this entry.
    /// Uses constant-time comparison to prevent timing attacks.
    pub fn verify(&self, key: &str) -> bool {
        let provided_hash = hash_api_key(key);
        self.key_hash.ct_eq(&provided_hash).into()
    }
}

/// Complete control plane authentication configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlPlaneAuthConfig {
    /// JWT/OIDC configuration for external IDP.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwt: Option<JwtConfig>,

    /// API keys for service accounts (parsed from CLI).
    /// Format: "role:key" where role is "admin" or "user"
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub api_keys: Vec<ApiKeyEntry>,

    /// Enable audit logging for control plane operations
    #[serde(default = "default_audit_enabled")]
    pub audit_enabled: bool,
}

fn default_audit_enabled() -> bool {
    true
}

impl ControlPlaneAuthConfig {
    /// Check if any authentication method is configured.
    pub fn is_enabled(&self) -> bool {
        self.jwt.is_some() || !self.api_keys.is_empty()
    }

    /// Check if JWT authentication is configured.
    pub fn has_jwt(&self) -> bool {
        self.jwt.is_some()
    }

    /// Check if any API keys are configured.
    pub fn has_api_keys(&self) -> bool {
        !self.api_keys.is_empty()
    }

    /// Find an API key by its value and return the entry if found.
    /// Uses constant-time hash comparison to prevent timing attacks.
    pub fn find_api_key(&self, key: &str) -> Option<&ApiKeyEntry> {
        // Iterate through all keys to prevent timing leaks about key existence
        // We use a variable to track the match to ensure constant-time behavior
        let mut found: Option<&ApiKeyEntry> = None;
        for entry in &self.api_keys {
            if entry.verify(key) {
                found = Some(entry);
                // Don't break early to maintain constant-time behavior
            }
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_parsing() {
        assert_eq!("admin".parse::<Role>().unwrap(), Role::Admin);
        assert_eq!("ADMIN".parse::<Role>().unwrap(), Role::Admin);
        assert_eq!("user".parse::<Role>().unwrap(), Role::User);
        assert_eq!("USER".parse::<Role>().unwrap(), Role::User);
        assert!("invalid".parse::<Role>().is_err());
    }

    #[test]
    fn test_role_display() {
        assert_eq!(Role::Admin.to_string(), "admin");
        assert_eq!(Role::User.to_string(), "user");
    }

    #[test]
    fn test_role_is_admin() {
        assert!(Role::Admin.is_admin());
        assert!(!Role::User.is_admin());
    }

    #[test]
    fn test_jwt_config_builder() {
        let config = JwtConfig::new("https://issuer.example.com", "api://my-app")
            .with_jwks_uri("https://issuer.example.com/.well-known/jwks.json")
            .with_role_mapping("Gateway.Admin", Role::Admin)
            .with_role_mapping("Gateway.User", Role::User);

        assert_eq!(config.issuer, "https://issuer.example.com");
        assert_eq!(config.audience, "api://my-app");
        assert!(config.jwks_uri.is_some());
        assert_eq!(config.role_mapping.len(), 2);
    }

    #[test]
    fn test_control_plane_auth_config() {
        let mut config = ControlPlaneAuthConfig::default();
        assert!(!config.is_enabled());
        assert!(!config.has_jwt());
        assert!(!config.has_api_keys());

        config.jwt = Some(JwtConfig::new("https://issuer.example.com", "api://test"));
        assert!(config.is_enabled());
        assert!(config.has_jwt());

        config.api_keys.push(ApiKeyEntry::new(
            "test-key",
            "Test Key",
            "secret123",
            Role::Admin,
        ));
        assert!(config.has_api_keys());
    }

    #[test]
    fn test_find_api_key() {
        let config = ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![
                ApiKeyEntry::new("key1", "Key 1", "secret1", Role::Admin),
                ApiKeyEntry::new("key2", "Key 2", "secret2", Role::User),
            ],
            audit_enabled: true,
        };

        let found = config.find_api_key("secret1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "key1");
        assert_eq!(found.unwrap().role, Role::Admin);

        let found = config.find_api_key("secret2");
        assert!(found.is_some());
        assert_eq!(found.unwrap().role, Role::User);

        assert!(config.find_api_key("invalid").is_none());
    }

    #[test]
    fn test_api_key_hashing() {
        let entry = ApiKeyEntry::new("test", "Test Key", "my-secret-key", Role::Admin);

        // Verify the correct key works
        assert!(entry.verify("my-secret-key"));

        // Verify wrong keys don't work
        assert!(!entry.verify("wrong-key"));
        assert!(!entry.verify("my-secret-ke")); // prefix
        assert!(!entry.verify("my-secret-keyy")); // suffix
        assert!(!entry.verify("")); // empty
    }

    #[test]
    fn test_api_key_debug_redacts_hash() {
        let entry = ApiKeyEntry::new("test", "Test Key", "secret", Role::Admin);
        let debug_str = format!("{:?}", entry);

        // Should contain REDACTED, not the actual hash
        assert!(debug_str.contains("[REDACTED]"));
        // Should not contain the key or hash bytes
        assert!(!debug_str.contains("secret"));
    }
}
