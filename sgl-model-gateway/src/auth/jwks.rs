//! JWKS (JSON Web Key Set) fetching and caching.
//!
//! Handles:
//! - OIDC discovery to find JWKS endpoint
//! - Fetching and parsing JWKS
//! - Caching with TTL and automatic refresh
//!
//! Security features:
//! - SSRF protection: only HTTPS URLs allowed, private IPs blocked
//! - Response size limits to prevent DoS
//! - Redirect prevention

use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    time::{Duration, Instant},
};

use jsonwebtoken::jwk::{Jwk, JwkSet};
use parking_lot::RwLock;
use tracing::{debug, info, warn};
use url::Url;

/// Maximum allowed JWKS response size (1 MB)
const MAX_JWKS_RESPONSE_SIZE: u64 = 1024 * 1024;

/// Error types for JWKS operations.
#[derive(Debug, thiserror::Error)]
pub enum JwksError {
    #[error("Failed to fetch OIDC discovery document: {0}")]
    DiscoveryFetch(String),

    #[error("Failed to parse OIDC discovery document: {0}")]
    DiscoveryParse(String),

    #[error("JWKS URI not found in discovery document")]
    JwksUriNotFound,

    #[error("Failed to fetch JWKS: {0}")]
    JwksFetch(String),

    #[error("Failed to parse JWKS: {0}")]
    JwksParse(String),

    #[error("Key not found for kid: {0}")]
    KeyNotFound(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("SSRF protection: {0}")]
    SsrfBlocked(String),

    #[error("Response too large: {0} bytes (max: {1})")]
    ResponseTooLarge(u64, u64),

    #[error("Failed to create HTTP client: {0}")]
    HttpClientError(String),
}

/// Check if an IP address is private/internal (SSRF protection).
fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => {
            ipv4.is_private()                           // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                || ipv4.is_loopback()                   // 127.0.0.0/8
                || ipv4.is_link_local()                 // 169.254.0.0/16
                || ipv4.is_broadcast()                  // 255.255.255.255
                || ipv4.is_unspecified()                // 0.0.0.0
                || is_shared_address(ipv4)              // 100.64.0.0/10 (CGNAT)
                || is_documentation_v4(ipv4)            // 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24
                || is_cloud_metadata(ipv4) // 169.254.169.254
        }
        IpAddr::V6(ipv6) => {
            ipv6.is_loopback()                          // ::1
                || ipv6.is_unspecified()                // ::
                || is_unique_local(ipv6)                // fc00::/7
                || is_link_local_v6(ipv6) // fe80::/10
        }
    }
}

/// Check for CGNAT shared address space (100.64.0.0/10)
fn is_shared_address(ip: &Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 100 && (octets[1] & 0xC0) == 64
}

/// Check for IPv4 documentation addresses
fn is_documentation_v4(ip: &Ipv4Addr) -> bool {
    let octets = ip.octets();
    // 192.0.2.0/24 (TEST-NET-1)
    (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
    // 198.51.100.0/24 (TEST-NET-2)
    || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
    // 203.0.113.0/24 (TEST-NET-3)
    || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
}

/// Check for cloud metadata endpoint (169.254.169.254)
fn is_cloud_metadata(ip: &Ipv4Addr) -> bool {
    ip.octets() == [169, 254, 169, 254]
}

/// Check for IPv6 unique local addresses (fc00::/7)
fn is_unique_local(ip: &Ipv6Addr) -> bool {
    let segments = ip.segments();
    (segments[0] & 0xfe00) == 0xfc00
}

/// Check for IPv6 link-local addresses (fe80::/10)
fn is_link_local_v6(ip: &Ipv6Addr) -> bool {
    let segments = ip.segments();
    (segments[0] & 0xffc0) == 0xfe80
}

/// Validate a URL for SSRF protection.
/// Returns the validated URL or an error.
///
/// For testing purposes, HTTP is allowed for localhost/127.0.0.1 only.
/// In production, only HTTPS should be used.
pub(crate) fn validate_url(url_str: &str) -> Result<Url, JwksError> {
    let url = Url::parse(url_str)
        .map_err(|e| JwksError::InvalidUrl(format!("Failed to parse URL: {}", e)))?;

    // Check if this is a localhost URL (allowed for testing with HTTP)
    let is_localhost = match url.host_str() {
        Some("localhost") | Some("127.0.0.1") => true,
        Some(host) => host == "::1" || host == "[::1]",
        None => false,
    };

    // Only allow HTTPS (except for localhost in tests)
    match url.scheme() {
        "https" => {}
        "http" => {
            if !is_localhost {
                return Err(JwksError::SsrfBlocked(
                    "Only HTTPS URLs are allowed for JWKS endpoints".to_string(),
                ));
            }
            // HTTP localhost is allowed - skip further checks and return early
            return Ok(url);
        }
        scheme => {
            return Err(JwksError::SsrfBlocked(format!(
                "Invalid URL scheme '{}'. Only HTTPS is allowed",
                scheme
            )));
        }
    }

    // For HTTPS URLs, check for private/internal addresses
    if let Some(host) = url.host() {
        match host {
            url::Host::Ipv4(ip) => {
                if is_private_ip(&IpAddr::V4(ip)) {
                    return Err(JwksError::SsrfBlocked(format!(
                        "Private/internal IP addresses are not allowed: {}",
                        ip
                    )));
                }
            }
            url::Host::Ipv6(ip) => {
                if is_private_ip(&IpAddr::V6(ip)) {
                    return Err(JwksError::SsrfBlocked(format!(
                        "Private/internal IP addresses are not allowed: {}",
                        ip
                    )));
                }
            }
            url::Host::Domain(domain) => {
                // Block common internal hostnames (except localhost which was handled above)
                let lower = domain.to_lowercase();
                if lower == "metadata"
                    || lower == "metadata.google.internal"
                    || lower.ends_with(".internal")
                    || lower.ends_with(".local")
                {
                    return Err(JwksError::SsrfBlocked(format!(
                        "Internal hostnames are not allowed: {}",
                        domain
                    )));
                }
            }
        }
    }

    Ok(url)
}

/// OIDC discovery document (subset of fields we need).
#[derive(Debug, serde::Deserialize)]
struct OidcDiscovery {
    jwks_uri: String,
    #[allow(dead_code)]
    issuer: String,
}

/// Cached JWKS with expiration tracking.
struct CachedJwks {
    jwks: JwkSet,
    fetched_at: Instant,
    ttl: Duration,
}

impl CachedJwks {
    fn is_expired(&self) -> bool {
        self.fetched_at.elapsed() > self.ttl
    }
}

/// JWKS provider with caching and automatic refresh.
pub(crate) struct JwksProvider {
    /// HTTP client for fetching JWKS
    client: reqwest::Client,
    /// JWKS endpoint URL (validated)
    jwks_uri: String,
    /// Cached JWKS
    cache: RwLock<Option<CachedJwks>>,
    /// Cache TTL
    ttl: Duration,
}

impl JwksProvider {
    /// Create a new JWKS provider with explicit JWKS URI.
    /// The URL is validated for SSRF protection.
    pub fn new(jwks_uri: impl Into<String>, ttl: Duration) -> Result<Self, JwksError> {
        let jwks_uri = jwks_uri.into();

        // Validate URL for SSRF protection
        validate_url(&jwks_uri)?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none()) // Prevent SSRF via redirects
            .build()
            .map_err(|e| JwksError::HttpClientError(e.to_string()))?;

        Ok(Self {
            client,
            jwks_uri,
            cache: RwLock::new(None),
            ttl,
        })
    }

    /// Create a new JWKS provider using OIDC discovery.
    /// Both issuer and discovered JWKS URI are validated for SSRF protection.
    pub async fn from_issuer(issuer: &str, ttl: Duration) -> Result<Self, JwksError> {
        // Normalize and validate issuer URL
        let issuer = issuer.trim_end_matches('/');
        let discovery_url = format!("{}/.well-known/openid-configuration", issuer);

        // Validate discovery URL for SSRF protection
        validate_url(&discovery_url)?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| JwksError::HttpClientError(e.to_string()))?;

        info!("Fetching OIDC discovery from: {}", discovery_url);

        let response = client
            .get(&discovery_url)
            .send()
            .await
            .map_err(|e| JwksError::DiscoveryFetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(JwksError::DiscoveryFetch(format!(
                "HTTP {}",
                response.status()
            )));
        }

        // Check response size before parsing
        if let Some(content_length) = response.content_length() {
            if content_length > MAX_JWKS_RESPONSE_SIZE {
                return Err(JwksError::ResponseTooLarge(
                    content_length,
                    MAX_JWKS_RESPONSE_SIZE,
                ));
            }
        }

        let discovery: OidcDiscovery = response
            .json()
            .await
            .map_err(|e| JwksError::DiscoveryParse(e.to_string()))?;

        // Validate discovered JWKS URI for SSRF protection
        validate_url(&discovery.jwks_uri)?;

        info!("Discovered JWKS URI: {}", discovery.jwks_uri);

        Ok(Self {
            client,
            jwks_uri: discovery.jwks_uri,
            cache: RwLock::new(None),
            ttl,
        })
    }

    /// Get the JWKS URI.
    #[allow(dead_code)]
    pub fn jwks_uri(&self) -> &str {
        &self.jwks_uri
    }

    /// Fetch JWKS from the endpoint.
    /// Response size is limited to prevent DoS attacks.
    async fn fetch_jwks(&self) -> Result<JwkSet, JwksError> {
        debug!("Fetching JWKS from: {}", self.jwks_uri);

        let response = self
            .client
            .get(&self.jwks_uri)
            .send()
            .await
            .map_err(|e| JwksError::JwksFetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(JwksError::JwksFetch(format!("HTTP {}", response.status())));
        }

        // Check response size before reading body
        if let Some(content_length) = response.content_length() {
            if content_length > MAX_JWKS_RESPONSE_SIZE {
                return Err(JwksError::ResponseTooLarge(
                    content_length,
                    MAX_JWKS_RESPONSE_SIZE,
                ));
            }
        }

        // Read body with size limit
        let bytes = response
            .bytes()
            .await
            .map_err(|e| JwksError::JwksFetch(e.to_string()))?;

        if bytes.len() as u64 > MAX_JWKS_RESPONSE_SIZE {
            return Err(JwksError::ResponseTooLarge(
                bytes.len() as u64,
                MAX_JWKS_RESPONSE_SIZE,
            ));
        }

        let jwks: JwkSet =
            serde_json::from_slice(&bytes).map_err(|e| JwksError::JwksParse(e.to_string()))?;

        debug!("Fetched JWKS with {} keys", jwks.keys.len());
        Ok(jwks)
    }

    /// Get the cached JWKS, refreshing if expired or not present.
    pub async fn get_jwks(&self) -> Result<JwkSet, JwksError> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.as_ref() {
                if !cached.is_expired() {
                    return Ok(cached.jwks.clone());
                }
            }
        }

        // Cache miss or expired, fetch new JWKS
        let jwks = self.fetch_jwks().await?;

        // Update cache
        {
            let mut cache = self.cache.write();
            *cache = Some(CachedJwks {
                jwks: jwks.clone(),
                fetched_at: Instant::now(),
                ttl: self.ttl,
            });
        }

        Ok(jwks)
    }

    /// Get a specific key by kid (key ID).
    pub async fn get_key(&self, kid: &str) -> Result<Jwk, JwksError> {
        let jwks = self.get_jwks().await?;

        // First try to find by kid
        if let Some(key) = jwks.find(kid) {
            return Ok(key.clone());
        }

        // Key not found - try refreshing the cache in case keys were rotated
        warn!("Key {} not found in cached JWKS, refreshing...", kid);
        let jwks = self.fetch_jwks().await?;

        // Update cache
        {
            let mut cache = self.cache.write();
            *cache = Some(CachedJwks {
                jwks: jwks.clone(),
                fetched_at: Instant::now(),
                ttl: self.ttl,
            });
        }

        jwks.find(kid)
            .cloned()
            .ok_or_else(|| JwksError::KeyNotFound(kid.to_string()))
    }

    /// Force refresh the JWKS cache.
    #[allow(dead_code)]
    pub async fn refresh(&self) -> Result<(), JwksError> {
        let jwks = self.fetch_jwks().await?;
        let mut cache = self.cache.write();
        *cache = Some(CachedJwks {
            jwks,
            fetched_at: Instant::now(),
            ttl: self.ttl,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_jwks_expiration() {
        let jwks = JwkSet { keys: vec![] };
        let cached = CachedJwks {
            jwks,
            fetched_at: Instant::now() - Duration::from_secs(100),
            ttl: Duration::from_secs(60),
        };
        assert!(cached.is_expired());

        let jwks = JwkSet { keys: vec![] };
        let cached = CachedJwks {
            jwks,
            fetched_at: Instant::now(),
            ttl: Duration::from_secs(60),
        };
        assert!(!cached.is_expired());
    }

    #[test]
    fn test_validate_url_https_required() {
        // Valid HTTPS URL
        assert!(validate_url("https://example.com/.well-known/jwks.json").is_ok());

        // HTTP not allowed for non-localhost
        assert!(validate_url("http://example.com/.well-known/jwks.json").is_err());

        // HTTP allowed for localhost (testing)
        assert!(validate_url("http://localhost:8080/jwks").is_ok());
        assert!(validate_url("http://127.0.0.1:8080/jwks").is_ok());

        // Invalid schemes
        assert!(validate_url("ftp://example.com/jwks").is_err());
        assert!(validate_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn test_validate_url_blocks_private_ips() {
        // Private IPv4 ranges
        assert!(validate_url("https://10.0.0.1/jwks").is_err());
        assert!(validate_url("https://172.16.0.1/jwks").is_err());
        assert!(validate_url("https://192.168.1.1/jwks").is_err());

        // Loopback
        assert!(validate_url("https://127.0.0.1/jwks").is_err());

        // Link-local
        assert!(validate_url("https://169.254.1.1/jwks").is_err());

        // Cloud metadata endpoint
        assert!(validate_url("https://169.254.169.254/jwks").is_err());

        // CGNAT
        assert!(validate_url("https://100.64.0.1/jwks").is_err());
    }

    #[test]
    fn test_validate_url_blocks_internal_hostnames() {
        assert!(validate_url("https://metadata/jwks").is_err());
        assert!(validate_url("https://metadata.google.internal/jwks").is_err());
        assert!(validate_url("https://internal.example.internal/jwks").is_err());
        assert!(validate_url("https://printer.local/jwks").is_err());
    }

    #[test]
    fn test_is_private_ip() {
        use std::net::Ipv4Addr;

        // Private ranges
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))));
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1))));
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));

        // Loopback
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));

        // Cloud metadata
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(
            169, 254, 169, 254
        ))));

        // Public IPs should not be blocked
        assert!(!is_private_ip(&IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
        assert!(!is_private_ip(&IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
    }
}
