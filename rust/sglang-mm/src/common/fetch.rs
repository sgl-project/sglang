//! Stage 1 of the server MM pipeline: resolve one media source to raw bytes.
//!
//! Mirrors the Python `get_image_bytes` source handling (and its precedence):
//! raw bytes, `http(s)://` (bounded download, `REQUEST_TIMEOUT` and the proxy
//! env vars like Python), `file://` / absolute path, `data:` URL, else bare
//! base64.

use std::io::Read;
use std::sync::OnceLock;

use base64::Engine;

/// Cap on a fetched payload so a bad URL can't buffer unboundedly (the Python
/// path has no such cap; oversized responses reject the request here).
const MAX_FETCH_BYTES: u64 = 64 << 20;

/// Resolve one string-typed image source into raw encoded-image bytes.
/// An `Err` rejects the request, matching the Python per-request
/// exception → 400.
pub fn fetch_bytes(src: &str) -> Result<Vec<u8>, String> {
    if src.starts_with("http://") || src.starts_with("https://") {
        return http_get(src);
    }
    if let Some(path) = src.strip_prefix("file://") {
        return std::fs::read(path).map_err(|e| format!("media fetch: {path}: {e}"));
    }
    if src.starts_with('/') {
        return std::fs::read(src).map_err(|e| format!("media fetch: {src}: {e}"));
    }
    if let Some(rest) = src.strip_prefix("data:") {
        let encoded = rest
            .split_once(',')
            .ok_or_else(|| "media fetch: malformed data: URL".to_string())?
            .1;
        return b64(encoded);
    }
    // Python treats any other string as bare base64.
    b64(src)
}

fn b64(encoded: &str) -> Result<Vec<u8>, String> {
    // Slightly laxer than Python's `pybase64.b64decode(validate=True)`:
    // surrounding whitespace (e.g. a trailing newline) is trimmed here.
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim().as_bytes())
        .map_err(|e| format!("media fetch: base64 decode: {e}"))
}

/// Shared pooled agent honoring `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`, as the
/// Python `requests` session does.
fn http_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| ureq::AgentBuilder::new().try_proxy_from_env(true).build())
}

/// Companion agent that ignores the proxy env vars, for hosts matched by
/// `NO_PROXY`. ureq has no `NO_PROXY` support of its own, and silently sending
/// an internal image host through a corporate proxy breaks deployments that
/// work on the Python path, so the match is applied here.
fn direct_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| ureq::AgentBuilder::new().build())
}

/// The host component of an `http(s)://` URL, lowercased and without userinfo
/// or port — what `NO_PROXY` entries are matched against.
fn host_of(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    // Bracketed IPv6 literal, else strip a trailing `:port`.
    let host = match host.strip_prefix('[') {
        Some(v6) => v6.split_once(']').map_or(v6, |(h, _)| h),
        None => host.split_once(':').map_or(host, |(h, _)| h),
    };
    (!host.is_empty()).then(|| host.to_ascii_lowercase())
}

fn bypasses_proxy(host: &str) -> bool {
    ["no_proxy", "NO_PROXY"]
        .iter()
        .find_map(|key| std::env::var(key).ok())
        .is_some_and(|list| no_proxy_matches(&list, host))
}

/// `NO_PROXY` semantics as `requests` implements them: comma-separated
/// entries, `*` matches everything, and an entry matches a host that equals it
/// or is a subdomain of it (leading dots ignored). Kept pure so it is testable
/// without mutating process-global env.
fn no_proxy_matches(no_proxy: &str, host: &str) -> bool {
    no_proxy.split(',').any(|entry| {
        let entry = entry.trim().trim_start_matches('.').to_ascii_lowercase();
        !entry.is_empty() && (entry == "*" || host == entry || host.ends_with(&format!(".{entry}")))
    })
}

fn http_get(url: &str) -> Result<Vec<u8>, String> {
    // Python: `int(os.getenv("REQUEST_TIMEOUT", "3"))` seconds per image GET.
    let timeout = std::env::var("REQUEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(3);
    let agent = match host_of(url) {
        Some(host) if bypasses_proxy(&host) => direct_agent(),
        _ => http_agent(),
    };
    let resp = agent
        .get(url)
        .timeout(std::time::Duration::from_secs(timeout))
        .call()
        .map_err(|e| format!("media fetch: GET {url}: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .take(MAX_FETCH_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("media fetch: read {url}: {e}"))?;
    if buf.len() as u64 > MAX_FETCH_BYTES {
        return Err(format!(
            "media fetch: {url}: response exceeds {MAX_FETCH_BYTES} bytes"
        ));
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_url_and_bare_base64_decode() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"hello");
        assert_eq!(
            fetch_bytes(&format!("data:image/png;base64,{b64}")).unwrap(),
            b"hello"
        );
        assert_eq!(fetch_bytes(&b64).unwrap(), b"hello");
    }

    #[test]
    fn bad_base64_fails() {
        assert!(fetch_bytes("!!not-base64!!").is_err());
    }

    #[test]
    fn missing_file_fails() {
        assert!(fetch_bytes("file:///definitely/not/here.jpg").is_err());
        assert!(fetch_bytes("/definitely/not/here.jpg").is_err());
    }

    #[test]
    fn host_parsing_strips_userinfo_port_and_path() {
        assert_eq!(
            host_of("http://Example.COM/a/b.png").unwrap(),
            "example.com"
        );
        assert_eq!(
            host_of("https://u:p@images.internal:8443/x").unwrap(),
            "images.internal"
        );
        assert_eq!(host_of("http://[::1]:8080/x.png").unwrap(), "::1");
        assert_eq!(host_of("http://host?q=1").unwrap(), "host");
        assert!(host_of("data:image/png;base64,AAA").is_none());
    }

    /// `NO_PROXY` must bypass the proxy for exact hosts and subdomains but not
    /// for lookalike suffixes — sending an internal host to a corporate proxy
    /// is a silent failure that works fine on the Python path.
    #[test]
    fn no_proxy_matches_host_and_subdomains_only() {
        let list = " .internal ,localhost";
        assert!(no_proxy_matches(list, "images.internal"));
        assert!(no_proxy_matches(list, "internal"));
        assert!(no_proxy_matches(list, "localhost"));
        assert!(!no_proxy_matches(list, "notinternal"));
        assert!(!no_proxy_matches(list, "example.com"));
        assert!(no_proxy_matches("*", "anything.example.com"));
        // An empty or all-empty list must not bypass everything.
        assert!(!no_proxy_matches("", "example.com"));
        assert!(!no_proxy_matches(" , ", "example.com"));
    }
}
