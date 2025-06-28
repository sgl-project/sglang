/// Build the full API path for a given route
pub fn api_path(base_url: &str, route: &str) -> String {
    if route.starts_with('/') {
        format!("{}{}", base_url, route)
    } else {
        format!("{}/{}", base_url, route)
    }
}

/// Get the hostname from a worker URL
pub fn hostname(url: &str) -> String {
    let url = url
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    url.split(':').next().unwrap_or("localhost").to_string()
}