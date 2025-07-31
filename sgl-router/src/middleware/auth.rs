use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};
use subtle::ConstantTimeEq;

#[derive(Clone)]
pub struct AuthConfig {
    pub api_key: Option<String>,
}

/// Middleware to validate Bearer token against configured API key
/// Only active when router has an API key configured
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(expected_key) = &auth_config.api_key {
        // Extract Authorization header
        let auth_header = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok());

        match auth_header {
            Some(header_value) if header_value.starts_with("Bearer ") => {
                let token = &header_value[7..]; // Skip "Bearer "
                                                // Use constant-time comparison to prevent timing attacks
                let token_bytes = token.as_bytes();
                let expected_bytes = expected_key.as_bytes();

                // Check if lengths match first (this is not constant-time but necessary)
                if token_bytes.len() != expected_bytes.len() {
                    return Err(StatusCode::UNAUTHORIZED);
                }

                // Constant-time comparison of the actual values
                if token_bytes.ct_eq(expected_bytes).unwrap_u8() != 1 {
                    return Err(StatusCode::UNAUTHORIZED);
                }
            }
            _ => return Err(StatusCode::UNAUTHORIZED),
        }
    }

    Ok(next.run(request).await)
}
