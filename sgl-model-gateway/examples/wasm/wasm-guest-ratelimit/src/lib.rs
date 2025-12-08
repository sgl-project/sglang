//! WASM Guest Rate Limit Example for sgl-model-gateway
//!
//! This example demonstrates rate limiting middleware
//! for sgl-model-gateway using the WebAssembly Component Model.
//!
//! Features:
//! - Rate limiting based on API Key or IP address
//! - Fixed time window (e.g., 60 requests per minute)
//! - Returns 429 Too Many Requests when limit exceeded
//!
//! Note: This is a simplified implementation. Since WASM components are stateless,
//! each instance maintains its own counters. For production use, consider
//! implementing rate limiting at the host/router level with shared state.

wit_bindgen::generate!({
    path: "../../../src/wasm/interface",
    world: "sgl-model-gateway",
});

use std::cell::RefCell;

use exports::sgl::router::{
    middleware_on_request::Guest as OnRequestGuest,
    middleware_on_response::Guest as OnResponseGuest,
};
use sgl::router::middleware_types::{Action, Request, Response};

/// Main middleware implementation
struct Middleware;

// Rate limit configuration
const RATE_LIMIT_REQUESTS: u64 = 60; // Maximum requests per window
const RATE_LIMIT_WINDOW_MS: u64 = 60_000; // Time window in milliseconds (1 minute)

// Simple in-memory counter (per WASM instance)
// In a real implementation, this would be shared across all instances
// This is a simplified example for demonstration purposes
struct RateLimitState {
    requests: Vec<(String, u64)>, // (identifier, timestamp_ms)
}

impl RateLimitState {
    fn new() -> Self {
        Self {
            requests: Vec::new(),
        }
    }

    // Clean up old entries outside the time window
    fn cleanup(&mut self, current_time_ms: u64) {
        let cutoff = current_time_ms.saturating_sub(RATE_LIMIT_WINDOW_MS);
        self.requests.retain(|(_, timestamp)| *timestamp > cutoff);
    }

    // Check if identifier has exceeded rate limit
    fn check_limit(&mut self, identifier: &str, current_time_ms: u64) -> bool {
        self.cleanup(current_time_ms);

        // Count requests in current window for this identifier
        let count = self
            .requests
            .iter()
            .filter(|(id, timestamp)| {
                id == identifier
                    && *timestamp > current_time_ms.saturating_sub(RATE_LIMIT_WINDOW_MS)
            })
            .count() as u64;

        if count >= RATE_LIMIT_REQUESTS {
            return false; // Limit exceeded
        }

        // Add new request
        self.requests
            .push((identifier.to_string(), current_time_ms));
        true // Within limit
    }
}

// Thread-local state (per WASM instance thread)
// Using thread_local! is safer than static mut as it avoids unsafe blocks
// and provides separate state for each thread automatically
thread_local! {
    static RATE_LIMIT_STATE: RefCell<RateLimitState> = RefCell::new(RateLimitState::new());
}

fn get_identifier(req: &Request) -> String {
    // Helper function to find header value
    let find_header_value =
        |headers: &[sgl::router::middleware_types::Header], name: &str| -> Option<String> {
            headers
                .iter()
                .find(|h| h.name.eq_ignore_ascii_case(name))
                .map(|h| h.value.clone())
        };

    // Prefer API Key as identifier (more stable than IP)
    if let Some(auth_header) = find_header_value(&req.headers, "authorization") {
        if auth_header.starts_with("Bearer ") {
            return format!("api_key:{}", &auth_header[7..]);
        } else if auth_header.starts_with("ApiKey ") {
            return format!("api_key:{}", &auth_header[7..]);
        }
    }

    if let Some(api_key) = find_header_value(&req.headers, "x-api-key") {
        return format!("api_key:{}", api_key);
    }

    // Fall back to IP address from forwarded headers
    if let Some(forwarded_for) = find_header_value(&req.headers, "x-forwarded-for") {
        // Take first IP from comma-separated list
        let ip = forwarded_for.split(',').next().unwrap_or("").trim();
        if !ip.is_empty() {
            return format!("ip:{}", ip);
        }
    }

    if let Some(real_ip) = find_header_value(&req.headers, "x-real-ip") {
        return format!("ip:{}", real_ip);
    }

    // Last resort: use request ID (not ideal, but better than nothing)
    format!("req_id:{}", req.request_id)
}

// Implement on-request interface
impl OnRequestGuest for Middleware {
    fn on_request(req: Request) -> Action {
        let identifier = get_identifier(&req);
        let current_time_ms = req.now_epoch_ms;

        // Access thread-local state safely without unsafe blocks
        // Each thread gets its own RateLimitState instance
        RATE_LIMIT_STATE.with(|state| {
            let mut state = state.borrow_mut();
            if !state.check_limit(&identifier, current_time_ms) {
                // Rate limit exceeded
                return Action::Reject(429);
            }
            // Within rate limit, continue processing
            Action::Continue
        })
    }
}

// Implement on-response interface (empty - not used for rate limiting)
impl OnResponseGuest for Middleware {
    fn on_response(_resp: Response) -> Action {
        Action::Continue
    }
}

// Export the component
export!(Middleware);
