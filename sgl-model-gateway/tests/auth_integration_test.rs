//! Integration tests for control plane authentication.
//!
//! Tests the full authentication flow including:
//! - JWT token validation with mock JWKS server
//! - API key authentication
//! - Role-based access control
//! - Token expiration and replay protection

use std::{
    net::SocketAddr,
    sync::LazyLock,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{routing::get, Json, Router};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use jsonwebtoken::{encode, EncodingKey, Header};
use rsa::{traits::PublicKeyParts, RsaPrivateKey};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sgl_model_gateway::auth::{
    ApiKeyEntry, ControlPlaneAuthConfig, ControlPlaneAuthState, JwtConfig, Role,
};
use tokio::net::TcpListener;

const TEST_KEY_ID: &str = "test-key-1";

/// Test RSA key pair - generates matching public/private key components
struct TestKeyPair {
    private_key_pem: String,
    n_base64url: String,
    e_base64url: String,
}

impl TestKeyPair {
    fn generate() -> Self {
        // Generate a new 2048-bit RSA key pair for testing
        use rsa::{pkcs8::EncodePrivateKey, rand_core::OsRng};

        let mut rng = OsRng;
        let private_key =
            RsaPrivateKey::new(&mut rng, 2048).expect("Failed to generate RSA key pair");

        // Get the public key components
        let n_bytes = private_key.n().to_bytes_be();
        let e_bytes = private_key.e().to_bytes_be();

        // Encode to base64url (no padding) for JWKS
        let n_base64url = URL_SAFE_NO_PAD.encode(&n_bytes);
        let e_base64url = URL_SAFE_NO_PAD.encode(&e_bytes);

        // Export private key as PEM
        let private_key_pem = private_key
            .to_pkcs8_pem(rsa::pkcs8::LineEnding::LF)
            .expect("Failed to export private key")
            .to_string();

        Self {
            private_key_pem,
            n_base64url,
            e_base64url,
        }
    }
}

/// Lazily initialized test key pair (shared across all tests)
static TEST_KEYS: LazyLock<TestKeyPair> = LazyLock::new(TestKeyPair::generate);

/// Claims structure for test tokens
#[derive(Debug, Serialize, Deserialize)]
struct TestClaims {
    sub: String,
    iss: String,
    aud: String,
    exp: u64,
    iat: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    jti: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    roles: Option<Vec<String>>,
}

/// Create a test JWT token using the generated test key pair
fn create_test_token(claims: &TestClaims) -> String {
    let mut header = Header::new(jsonwebtoken::Algorithm::RS256);
    header.kid = Some(TEST_KEY_ID.to_string());

    let key = EncodingKey::from_rsa_pem(TEST_KEYS.private_key_pem.as_bytes())
        .expect("Failed to create encoding key");

    encode(&header, claims, &key).expect("Failed to encode token")
}

/// Create claims with default values
fn create_claims(sub: &str, roles: Vec<&str>) -> TestClaims {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    TestClaims {
        sub: sub.to_string(),
        iss: "https://test-issuer.example.com".to_string(),
        aud: "test-gateway".to_string(),
        exp: now + 3600, // 1 hour from now
        iat: now,
        jti: Some(uuid::Uuid::new_v4().to_string()),
        name: Some("Test User".to_string()),
        roles: Some(roles.into_iter().map(String::from).collect()),
    }
}

/// Start a mock JWKS server using the generated test key pair
async fn start_mock_jwks_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    // Use the generated key pair components
    let jwks_response = json!({
        "keys": [{
            "kty": "RSA",
            "use": "sig",
            "kid": TEST_KEY_ID,
            "alg": "RS256",
            "n": TEST_KEYS.n_base64url,
            "e": TEST_KEYS.e_base64url
        }]
    });

    let app = Router::new().route(
        "/.well-known/jwks.json",
        get(move || {
            let jwks = jwks_response.clone();
            async move { Json(jwks) }
        }),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, handle)
}

// ============================================================================
// JWT Authentication Tests
// ============================================================================

#[tokio::test]
async fn test_jwt_valid_token_with_admin_role() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ))
        .with_role_mapping("admin", Role::Admin);

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    // Create a valid token with admin role
    let claims = create_claims("admin@example.com", vec!["admin"]);
    let token = create_test_token(&claims);

    // Validate the token
    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");
    let result = jwt_validator.validate(&token).await;

    assert!(
        result.is_ok(),
        "Token validation failed: {:?}",
        result.err()
    );
    let validated = result.unwrap();
    assert_eq!(validated.subject, "admin@example.com");
    assert_eq!(validated.role, Role::Admin);
}

#[tokio::test]
async fn test_jwt_valid_token_with_user_role() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ))
        .with_role_mapping("admin", Role::Admin);

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    // Create a valid token with user role (not admin)
    let claims = create_claims("user@example.com", vec!["user", "viewer"]);
    let token = create_test_token(&claims);

    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");
    let result = jwt_validator.validate(&token).await;

    assert!(result.is_ok());
    let validated = result.unwrap();
    assert_eq!(validated.subject, "user@example.com");
    assert_eq!(validated.role, Role::User); // Not admin
}

#[tokio::test]
async fn test_jwt_expired_token() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ));

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    // Create an expired token
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let claims = TestClaims {
        sub: "user@example.com".to_string(),
        iss: "https://test-issuer.example.com".to_string(),
        aud: "test-gateway".to_string(),
        exp: now - 3600, // Expired 1 hour ago
        iat: now - 7200,
        jti: None,
        name: None,
        roles: Some(vec!["admin".to_string()]),
    };
    let token = create_test_token(&claims);

    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");
    let result = jwt_validator.validate(&token).await;

    assert!(result.is_err(), "Expired token should fail validation");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("exp") || err.to_string().contains("ExpiredSignature"),
        "Error should mention expiration: {}",
        err
    );
}

#[tokio::test]
async fn test_jwt_wrong_audience() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config =
        JwtConfig::new("https://test-issuer.example.com", "correct-audience").with_jwks_uri(
            format!("http://127.0.0.1:{}/.well-known/jwks.json", addr.port()),
        );

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    // Create a token with wrong audience
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let claims = TestClaims {
        sub: "user@example.com".to_string(),
        iss: "https://test-issuer.example.com".to_string(),
        aud: "wrong-audience".to_string(), // Wrong audience
        exp: now + 3600,
        iat: now,
        jti: None,
        name: None,
        roles: Some(vec!["admin".to_string()]),
    };
    let token = create_test_token(&claims);

    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");
    let result = jwt_validator.validate(&token).await;

    assert!(result.is_err(), "Wrong audience should fail validation");
}

#[tokio::test]
async fn test_jwt_wrong_issuer() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config =
        JwtConfig::new("https://correct-issuer.example.com", "test-gateway").with_jwks_uri(
            format!("http://127.0.0.1:{}/.well-known/jwks.json", addr.port()),
        );

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    // Create a token with wrong issuer
    let claims = create_claims("user@example.com", vec!["admin"]);
    // claims has issuer "https://test-issuer.example.com" which doesn't match
    let token = create_test_token(&claims);

    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");
    let result = jwt_validator.validate(&token).await;

    assert!(result.is_err(), "Wrong issuer should fail validation");
}

// ============================================================================
// API Key Authentication Tests
// ============================================================================

#[tokio::test]
async fn test_api_key_valid_admin() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new(
            "admin-key-1",
            "Admin Service Account",
            "sk-test-admin-key-12345",
            Role::Admin,
        )],
        audit_enabled: false,
    };

    // Verify we can find the API key
    let found = config.find_api_key("sk-test-admin-key-12345");
    assert!(found.is_some(), "API key should be found");

    let entry = found.unwrap();
    assert_eq!(entry.id, "admin-key-1");
    assert_eq!(entry.role, Role::Admin);
}

#[tokio::test]
async fn test_api_key_valid_user() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new(
            "user-key-1",
            "User Service Account",
            "sk-test-user-key-12345",
            Role::User,
        )],
        audit_enabled: false,
    };

    let found = config.find_api_key("sk-test-user-key-12345");
    assert!(found.is_some());

    let entry = found.unwrap();
    assert_eq!(entry.role, Role::User);
}

#[tokio::test]
async fn test_api_key_invalid() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new(
            "admin-key-1",
            "Admin Service Account",
            "sk-correct-key",
            Role::Admin,
        )],
        audit_enabled: false,
    };

    // Try with wrong key
    let found = config.find_api_key("sk-wrong-key");
    assert!(found.is_none(), "Wrong API key should not be found");
}

#[tokio::test]
async fn test_api_key_timing_attack_resistance() {
    // Verify that key comparison uses constant-time comparison
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new(
            "key-1",
            "Test Key",
            "sk-abcdefghijklmnop",
            Role::Admin,
        )],
        audit_enabled: false,
    };

    // These should all take roughly the same time due to constant-time comparison
    // (We can't easily test timing in unit tests, but we verify the function works)
    assert!(config.find_api_key("sk-abcdefghijklmnop").is_some());
    assert!(config.find_api_key("sk-abcdefghijklmnox").is_none()); // One char different
    assert!(config.find_api_key("sk-xxxxxxxxxxxxxxxx").is_none()); // All different
    assert!(config.find_api_key("sk-a").is_none()); // Much shorter
}

// ============================================================================
// Combined Auth Tests (JWT + API Key fallback)
// ============================================================================

#[tokio::test]
async fn test_combined_auth_jwt_and_api_keys() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ))
        .with_role_mapping("admin", Role::Admin);

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![ApiKeyEntry::new(
            "backup-key",
            "Backup Admin Key",
            "sk-backup-admin-key",
            Role::Admin,
        )],
        audit_enabled: true,
    };

    let state = ControlPlaneAuthState::from_config(config.clone())
        .await
        .expect("Failed to create auth state");

    // Both JWT and API key should work
    assert!(state.jwt_validator.is_some());
    assert!(config.find_api_key("sk-backup-admin-key").is_some());

    // Verify JWT works
    let claims = create_claims("jwt-user@example.com", vec!["admin"]);
    let token = create_test_token(&claims);
    let result = state.jwt_validator.as_ref().unwrap().validate(&token).await;
    assert!(result.is_ok());
}

// ============================================================================
// Role-Based Access Control Tests
// ============================================================================

#[tokio::test]
async fn test_role_admin_has_access() {
    assert!(Role::Admin.is_admin());
}

#[tokio::test]
async fn test_role_user_no_admin_access() {
    assert!(!Role::User.is_admin());
}

#[tokio::test]
async fn test_role_parsing() {
    assert_eq!("admin".parse::<Role>().unwrap(), Role::Admin);
    assert_eq!("Admin".parse::<Role>().unwrap(), Role::Admin);
    assert_eq!("ADMIN".parse::<Role>().unwrap(), Role::Admin);
    assert_eq!("user".parse::<Role>().unwrap(), Role::User);
    assert_eq!("User".parse::<Role>().unwrap(), Role::User);
    assert_eq!("USER".parse::<Role>().unwrap(), Role::User);
}

// ============================================================================
// try_init Helper Tests
// ============================================================================

#[tokio::test]
async fn test_try_init_with_no_config() {
    let result = ControlPlaneAuthState::try_init(None).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn test_try_init_with_disabled_config() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![], // No auth configured = disabled
        audit_enabled: false,
    };

    let result = ControlPlaneAuthState::try_init(Some(&config)).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn test_try_init_with_api_keys_only() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new("key-1", "Test", "sk-test", Role::Admin)],
        audit_enabled: false,
    };

    let result = ControlPlaneAuthState::try_init(Some(&config)).await;
    assert!(result.is_some());
}

// ============================================================================
// Audit Logging Tests
// ============================================================================

#[tokio::test]
async fn test_audit_logging_enabled() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new("key-1", "Test", "sk-test", Role::Admin)],
        audit_enabled: true,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    assert!(state.audit_logger.is_enabled());
}

#[tokio::test]
async fn test_audit_logging_disabled() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new("key-1", "Test", "sk-test", Role::Admin)],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    assert!(!state.audit_logger.is_enabled());
}

// ============================================================================
// JTI Replay Protection Tests
// ============================================================================

#[tokio::test]
async fn test_jwt_jti_replay_protection() {
    use sgl_model_gateway::auth::JwtValidator;

    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ))
        .with_role_mapping("admin", Role::Admin);

    // Create validator with JTI replay protection enabled
    let validator = JwtValidator::from_config_with_options(jwt_config, true)
        .await
        .expect("Failed to create JWT validator");

    // Create a token with a specific JTI
    let claims = create_claims("user@example.com", vec!["admin"]);
    let token = create_test_token(&claims);

    // First validation should succeed
    let result1 = validator.validate(&token).await;
    assert!(result1.is_ok(), "First validation should succeed");

    // Second validation with same token should fail (replay)
    let result2 = validator.validate(&token).await;
    assert!(
        result2.is_err(),
        "Second validation should fail (replay detected)"
    );
    let err = result2.unwrap_err();
    assert!(
        err.to_string().contains("replay") || err.to_string().contains("already been used"),
        "Error should mention replay: {}",
        err
    );
}

#[tokio::test]
async fn test_jwt_different_tokens_no_replay() {
    use sgl_model_gateway::auth::JwtValidator;

    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ))
        .with_role_mapping("admin", Role::Admin);

    // Create validator with JTI replay protection enabled
    let validator = JwtValidator::from_config_with_options(jwt_config, true)
        .await
        .expect("Failed to create JWT validator");

    // Create two different tokens (different JTIs)
    let claims1 = create_claims("user1@example.com", vec!["admin"]);
    let claims2 = create_claims("user2@example.com", vec!["admin"]);
    let token1 = create_test_token(&claims1);
    let token2 = create_test_token(&claims2);

    // Both should succeed since they have different JTIs
    let result1 = validator.validate(&token1).await;
    assert!(result1.is_ok(), "First token validation should succeed");

    let result2 = validator.validate(&token2).await;
    assert!(
        result2.is_ok(),
        "Second token validation should succeed (different JTI)"
    );
}

// ============================================================================
// Malformed Token Tests
// ============================================================================

#[tokio::test]
async fn test_jwt_malformed_token() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ));

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");

    // Test various malformed tokens
    let malformed_tokens = [
        "not-a-jwt",
        "only.two.parts.is.not.enough",
        "",
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9",  // Only header
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.", // Header with dot
        ".....",
    ];

    for token in malformed_tokens {
        let result = jwt_validator.validate(token).await;
        assert!(
            result.is_err(),
            "Malformed token '{}' should fail validation",
            token
        );
    }
}

#[tokio::test]
async fn test_jwt_missing_kid_in_header() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ));

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    // Create a token WITHOUT kid in header
    let claims = create_claims("user@example.com", vec!["admin"]);
    let mut header = Header::new(jsonwebtoken::Algorithm::RS256);
    header.kid = None; // Explicitly no kid

    let key = EncodingKey::from_rsa_pem(TEST_KEYS.private_key_pem.as_bytes())
        .expect("Failed to create encoding key");
    let token_without_kid = encode(&header, &claims, &key).expect("Failed to encode token");

    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");
    let result = jwt_validator.validate(&token_without_kid).await;

    assert!(result.is_err(), "Token without kid should fail validation");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("kid") || err.to_string().contains("Missing"),
        "Error should mention missing kid: {}",
        err
    );
}

// ============================================================================
// Edge Cases and Security Tests
// ============================================================================

#[tokio::test]
async fn test_jwt_role_extraction_from_groups_claim() {
    let (addr, _server) = start_mock_jwks_server().await;

    // JWT validator checks alternate claims ("groups", "roles", etc.) as fallback
    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ))
        .with_role_mapping("administrators", Role::Admin);

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    // Create token with 'groups' claim instead of 'roles'
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    #[derive(Serialize)]
    struct GroupsClaims {
        sub: String,
        iss: String,
        aud: String,
        exp: u64,
        iat: u64,
        jti: String,
        groups: Vec<String>, // Use 'groups' instead of 'roles'
    }

    let claims = GroupsClaims {
        sub: "group-user@example.com".to_string(),
        iss: "https://test-issuer.example.com".to_string(),
        aud: "test-gateway".to_string(),
        exp: now + 3600,
        iat: now,
        jti: uuid::Uuid::new_v4().to_string(),
        groups: vec!["administrators".to_string(), "developers".to_string()],
    };

    let mut header = Header::new(jsonwebtoken::Algorithm::RS256);
    header.kid = Some(TEST_KEY_ID.to_string());
    let key = EncodingKey::from_rsa_pem(TEST_KEYS.private_key_pem.as_bytes())
        .expect("Failed to create encoding key");
    let token = encode(&header, &claims, &key).expect("Failed to encode token");

    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");
    let result = jwt_validator.validate(&token).await;

    assert!(
        result.is_ok(),
        "Token with groups claim should validate: {:?}",
        result.err()
    );
    let validated = result.unwrap();
    assert_eq!(
        validated.role,
        Role::Admin,
        "Should map 'administrators' to Admin role"
    );
}

#[tokio::test]
async fn test_jwt_no_role_defaults_to_user() {
    let (addr, _server) = start_mock_jwks_server().await;

    let jwt_config = JwtConfig::new("https://test-issuer.example.com", "test-gateway")
        .with_jwks_uri(format!(
            "http://127.0.0.1:{}/.well-known/jwks.json",
            addr.port()
        ));
    // No role mapping configured

    let config = ControlPlaneAuthConfig {
        jwt: Some(jwt_config),
        api_keys: vec![],
        audit_enabled: false,
    };

    let state = ControlPlaneAuthState::from_config(config)
        .await
        .expect("Failed to create auth state");

    // Create token without any role claims
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    #[derive(Serialize)]
    struct MinimalClaims {
        sub: String,
        iss: String,
        aud: String,
        exp: u64,
        iat: u64,
        jti: String,
        // No roles, groups, or role claims
    }

    let claims = MinimalClaims {
        sub: "minimal-user@example.com".to_string(),
        iss: "https://test-issuer.example.com".to_string(),
        aud: "test-gateway".to_string(),
        exp: now + 3600,
        iat: now,
        jti: uuid::Uuid::new_v4().to_string(),
    };

    let mut header = Header::new(jsonwebtoken::Algorithm::RS256);
    header.kid = Some(TEST_KEY_ID.to_string());
    let key = EncodingKey::from_rsa_pem(TEST_KEYS.private_key_pem.as_bytes())
        .expect("Failed to create encoding key");
    let token = encode(&header, &claims, &key).expect("Failed to encode token");

    let jwt_validator = state.jwt_validator.as_ref().expect("JWT validator not set");
    let result = jwt_validator.validate(&token).await;

    assert!(result.is_ok(), "Token without role should still validate");
    let validated = result.unwrap();
    assert_eq!(
        validated.role,
        Role::User,
        "Should default to User role when no role found"
    );
}

#[tokio::test]
async fn test_multiple_api_keys() {
    let config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![
            ApiKeyEntry::new("admin-1", "Admin Key 1", "sk-admin-key-1", Role::Admin),
            ApiKeyEntry::new("admin-2", "Admin Key 2", "sk-admin-key-2", Role::Admin),
            ApiKeyEntry::new("user-1", "User Key 1", "sk-user-key-1", Role::User),
        ],
        audit_enabled: false,
    };

    // All keys should be findable
    assert!(config.find_api_key("sk-admin-key-1").is_some());
    assert!(config.find_api_key("sk-admin-key-2").is_some());
    assert!(config.find_api_key("sk-user-key-1").is_some());

    // Verify correct roles
    assert_eq!(
        config.find_api_key("sk-admin-key-1").unwrap().role,
        Role::Admin
    );
    assert_eq!(
        config.find_api_key("sk-user-key-1").unwrap().role,
        Role::User
    );

    // Non-existent key should not be found
    assert!(config.find_api_key("sk-nonexistent").is_none());
}

#[tokio::test]
async fn test_config_is_enabled_checks() {
    // Empty config is not enabled
    let empty_config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![],
        audit_enabled: false,
    };
    assert!(!empty_config.is_enabled());

    // Config with only API keys is enabled
    let api_key_config = ControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![ApiKeyEntry::new("key-1", "Test", "sk-test", Role::Admin)],
        audit_enabled: false,
    };
    assert!(api_key_config.is_enabled());

    // Config with only JWT is enabled
    let jwt_config = ControlPlaneAuthConfig {
        jwt: Some(JwtConfig::new("https://issuer.example.com", "audience")),
        api_keys: vec![],
        audit_enabled: false,
    };
    assert!(jwt_config.is_enabled());

    // Config with both is enabled
    let full_config = ControlPlaneAuthConfig {
        jwt: Some(JwtConfig::new("https://issuer.example.com", "audience")),
        api_keys: vec![ApiKeyEntry::new("key-1", "Test", "sk-test", Role::Admin)],
        audit_enabled: true,
    };
    assert!(full_config.is_enabled());
}
