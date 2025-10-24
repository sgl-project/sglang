use axum::{body::Body, extract::Request, http::HeaderMap};

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

/// Apply headers to a reqwest request builder, filtering out headers that shouldn't be forwarded
/// or that will be set automatically by reqwest
pub fn apply_request_headers(
    headers: &HeaderMap,
    mut request_builder: reqwest::RequestBuilder,
    skip_content_headers: bool,
) -> reqwest::RequestBuilder {
    // Always forward Authorization header first if present
    if let Some(auth) = headers
        .get("authorization")
        .or_else(|| headers.get("Authorization"))
    {
        request_builder = request_builder.header("Authorization", auth.clone());
    }

    // Forward other headers, filtering out problematic ones
    for (key, value) in headers.iter() {
        let key_str = key.as_str().to_lowercase();

        // Skip headers that:
        // - Are set automatically by reqwest (content-type, content-length for POST/PUT)
        // - We already handled (authorization)
        // - Are hop-by-hop headers (connection, transfer-encoding)
        // - Should not be forwarded (host)
        let should_skip = key_str == "authorization" || // Already handled above
            key_str == "host" ||
            key_str == "connection" ||
            key_str == "transfer-encoding" ||
            key_str == "keep-alive" ||
            key_str == "te" ||
            key_str == "trailers" ||
            key_str == "accept-encoding" ||
            key_str == "upgrade" ||
            (skip_content_headers && (key_str == "content-type" || key_str == "content-length"));

        if !should_skip {
            request_builder = request_builder.header(key.clone(), value.clone());
        }
    }

    request_builder
}
