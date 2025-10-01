//! WASM Guest Auth Example for sgl-router
//!
//! This example demonstrates API key authentication middleware
//! for sgl-router using the WebAssembly Component Model (WIT).
//!
//! Features:
//! - API Key authentication

wit_bindgen::generate!({
    path: "../../../src/wasm/wit",
    world: "sgl-router",
});

use exports::sgl::router::middleware_on_request::Guest as OnRequestGuest;
use exports::sgl::router::middleware_on_response::Guest as OnResponseGuest;
use sgl::router::middleware_types::{Request, Response, Action};

/// Expected API Key (in production, this should be passed as configuration)
const EXPECTED_API_KEY: &str = "secret-api-key-12345";

/// Main middleware implementation
struct Middleware;

// Helper function to find header value
fn find_header_value(headers: &[sgl::router::middleware_types::Header], name: &str) -> Option<String> {
    headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case(name))
        .map(|h| h.value.clone())
}

// Implement on-request interface
impl OnRequestGuest for Middleware {
    fn on_request(req: Request) -> Action {
        // API Key Authentication
        // Check for API key in Authorization header for /api routes
        if req.path.starts_with("/api") || req.path.starts_with("/v1") {
            let auth_header = find_header_value(&req.headers, "authorization");
            let api_key = auth_header
                .and_then(|h| {
                    if h.starts_with("Bearer ") {
                        Some(h[7..].to_string())
                    } else if h.starts_with("ApiKey ") {
                        Some(h[7..].to_string())
                    } else {
                        None
                    }
                })
                .or_else(|| find_header_value(&req.headers, "x-api-key"));

            if let Some(key) = api_key {
                if key != EXPECTED_API_KEY {
                    // Invalid API key - reject with 401 Unauthorized
                    return Action::Reject(401);
                }
            } else {
                // Missing API key - reject with 401 Unauthorized
                return Action::Reject(401);
            }
        }

        // Authentication passed, continue processing
        Action::Continue
    }
}

// Implement on-response interface (empty - not used for auth)
impl OnResponseGuest for Middleware {
    fn on_response(_resp: Response) -> Action {
        Action::Continue
    }
}

// Export the component
export!(Middleware);

