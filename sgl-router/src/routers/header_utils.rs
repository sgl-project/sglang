use axum::body::Body;
use axum::extract::Request;
use axum::http::HeaderMap;

/// Copy request headers to a Vec of name-value string pairs
/// Used for forwarding headers to backend workers
pub fn copy_request_headers(req: &Request<Body>) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            // Convert header value to string, skipping non-UTF8 headers
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

/// Convert headers from reqwest Response to axum HeaderMap
/// Filters out hop-by-hop headers that shouldn't be forwarded
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    for (name, value) in reqwest_headers.iter() {
        // Skip hop-by-hop headers that shouldn't be forwarded
        let name_str = name.as_str().to_lowercase();
        if should_forward_header(&name_str) {
            // The original name and value are already valid, so we can just clone them
            headers.insert(name.clone(), value.clone());
        }
    }

    headers
}

/// Determine if a header should be forwarded from backend to client
fn should_forward_header(name: &str) -> bool {
    // List of headers that should NOT be forwarded (hop-by-hop headers)
    !matches!(
        name,
        "connection" |
        "keep-alive" |
        "proxy-authenticate" |
        "proxy-authorization" |
        "te" |
        "trailers" |
        "transfer-encoding" |
        "upgrade" |
        "content-encoding" | // Let axum/hyper handle encoding
        "host" // Should not forward the backend's host header
    )
}
