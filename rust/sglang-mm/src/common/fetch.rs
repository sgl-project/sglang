//! Stage 1 of the server MM pipeline: resolve one media source to raw bytes.
//!
//! Mirrors the Python `get_image_bytes` source handling (and its precedence):
//! raw bytes, `http(s)://` (bounded download, `REQUEST_TIMEOUT` and the proxy
//! env vars like Python), `file://` / absolute path, `data:` URL, else bare
//! base64.

use std::io::Read;
use std::sync::OnceLock;

use base64::Engine;

/// Cap on any single resolved payload — HTTP, file, or base64 — so no source
/// form can exhaust memory (the Python path has no such cap; oversized
/// payloads reject the request here).
pub const MAX_FETCH_BYTES: u64 = 64 << 20;

/// Resolve one string-typed image source into raw encoded-image bytes.
/// An `Err` rejects the request, matching the Python per-request
/// exception → 400.
pub fn fetch_bytes(src: &str) -> Result<Vec<u8>, String> {
    if src.starts_with("http://") || src.starts_with("https://") {
        return http_get(src);
    }
    if let Some(path) = src.strip_prefix("file://") {
        return read_file(path);
    }
    if src.starts_with('/') {
        return read_file(src);
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

/// Bounded read: never trusts metadata, so huge and non-regular files
/// (`/dev/zero`) hit the cap instead of exhausting memory.
fn read_file(path: &str) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("media fetch: {path}: {e}"))?;
    read_capped(file, path)
}

fn read_capped(reader: impl Read, what: &str) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    reader
        .take(MAX_FETCH_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("media fetch: read {what}: {e}"))?;
    if buf.len() as u64 > MAX_FETCH_BYTES {
        return Err(format!(
            "media fetch: {what}: exceeds {MAX_FETCH_BYTES} bytes"
        ));
    }
    Ok(buf)
}

fn b64(encoded: &str) -> Result<Vec<u8>, String> {
    // Slightly laxer than Python's `pybase64.b64decode(validate=True)`:
    // surrounding whitespace (e.g. a trailing newline) is trimmed here.
    let encoded = encoded.trim();
    // Reject by encoded length before allocating the decode buffer.
    if encoded.len() as u64 / 4 * 3 > MAX_FETCH_BYTES {
        return Err(format!(
            "media fetch: base64 payload exceeds {MAX_FETCH_BYTES} bytes"
        ));
    }
    base64::engine::general_purpose::STANDARD
        .decode(encoded.as_bytes())
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

/// The host component of an `http(s)://` URL (lowercased, without userinfo)
/// plus its explicit port, if any — what `NO_PROXY` entries are matched
/// against.
fn host_port_of(url: &str) -> Option<(String, Option<u16>)> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    // Bracketed IPv6 literal, else split off a trailing `:port`.
    let (host, port) = match host.strip_prefix('[') {
        Some(v6) => match v6.split_once(']') {
            Some((h, p)) => (h, p.strip_prefix(':')),
            None => (v6, None),
        },
        None => match host.split_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (host, None),
        },
    };
    let port = port.and_then(|p| p.parse().ok());
    (!host.is_empty()).then(|| (host.to_ascii_lowercase(), port))
}

fn bypasses_proxy(host: &str, port: Option<u16>) -> bool {
    ["no_proxy", "NO_PROXY"]
        .iter()
        .find_map(|key| std::env::var(key).ok())
        .is_some_and(|list| no_proxy_matches(&list, host, port))
}

/// `NO_PROXY` semantics as `requests` implements them: comma-separated
/// entries; `*` matches everything; an IPv4 CIDR entry matches an IPv4 host in
/// that network (requests supports IPv4 networks only); a `host:port` entry
/// matches only that explicit port; otherwise an entry matches a host that
/// equals it or is a subdomain of it (leading dots ignored). Kept pure so it
/// is testable without mutating process-global env.
fn no_proxy_matches(no_proxy: &str, host: &str, port: Option<u16>) -> bool {
    no_proxy.split(',').any(|entry| {
        let entry = entry.trim().trim_start_matches('.').to_ascii_lowercase();
        if entry.is_empty() {
            return false;
        }
        if entry == "*" {
            return true;
        }
        if let Some((net, bits)) = parse_ipv4_cidr(&entry) {
            return host
                .parse::<std::net::Ipv4Addr>()
                .is_ok_and(|ip| in_ipv4_network(ip, net, bits));
        }
        let (entry_host, entry_port) = match entry.rsplit_once(':') {
            Some((h, p)) if p.bytes().all(|b| b.is_ascii_digit()) => (h, p.parse::<u16>().ok()),
            _ => (entry.as_str(), None),
        };
        if entry_port.is_some() && entry_port != port {
            return false;
        }
        host == entry_host || host.ends_with(&format!(".{entry_host}"))
    })
}

fn parse_ipv4_cidr(entry: &str) -> Option<(std::net::Ipv4Addr, u32)> {
    let (net, bits) = entry.split_once('/')?;
    Some((net.parse().ok()?, bits.parse().ok().filter(|b| *b <= 32)?))
}

fn in_ipv4_network(ip: std::net::Ipv4Addr, net: std::net::Ipv4Addr, bits: u32) -> bool {
    let mask = u32::MAX.checked_shl(32 - bits).unwrap_or(0);
    u32::from(ip) & mask == u32::from(net) & mask
}

fn http_get(url: &str) -> Result<Vec<u8>, String> {
    // Python: `int(os.getenv("REQUEST_TIMEOUT", "3"))` seconds per image GET.
    let timeout = std::env::var("REQUEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(3);
    let agent = match host_port_of(url) {
        Some((host, port)) if bypasses_proxy(&host, port) => direct_agent(),
        _ => http_agent(),
    };
    let resp = agent
        .get(url)
        .timeout(std::time::Duration::from_secs(timeout))
        .call()
        .map_err(|e| format!("media fetch: GET {url}: {e}"))?;
    read_capped(resp.into_reader(), url)
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

    /// A non-regular file must hit the byte cap, not exhaust memory.
    #[test]
    fn unbounded_file_capped() {
        let err = fetch_bytes("/dev/zero").err().unwrap();
        assert!(err.contains("exceeds"), "{err}");
    }

    /// Oversized base64 is rejected from its encoded length, before decoding.
    #[test]
    fn oversized_base64_rejected() {
        let encoded = "A".repeat((MAX_FETCH_BYTES / 3 * 4 + 8) as usize);
        let err = fetch_bytes(&encoded).err().unwrap();
        assert!(err.contains("exceeds"), "{err}");
    }

    #[test]
    fn host_parsing_strips_userinfo_and_path() {
        assert_eq!(
            host_port_of("http://Example.COM/a/b.png").unwrap(),
            ("example.com".into(), None)
        );
        assert_eq!(
            host_port_of("https://u:p@images.internal:8443/x").unwrap(),
            ("images.internal".into(), Some(8443))
        );
        assert_eq!(
            host_port_of("http://[::1]:8080/x.png").unwrap(),
            ("::1".into(), Some(8080))
        );
        assert_eq!(
            host_port_of("http://host?q=1").unwrap(),
            ("host".into(), None)
        );
        assert!(host_port_of("data:image/png;base64,AAA").is_none());
    }

    /// `NO_PROXY` must bypass the proxy for exact hosts and subdomains but not
    /// for lookalike suffixes — sending an internal host to a corporate proxy
    /// is a silent failure that works fine on the Python path.
    #[test]
    fn no_proxy_matches_host_and_subdomains_only() {
        let list = " .internal ,localhost";
        assert!(no_proxy_matches(list, "images.internal", None));
        assert!(no_proxy_matches(list, "internal", None));
        assert!(no_proxy_matches(list, "localhost", None));
        assert!(!no_proxy_matches(list, "notinternal", None));
        assert!(!no_proxy_matches(list, "example.com", None));
        assert!(no_proxy_matches("*", "anything.example.com", None));
        // An empty or all-empty list must not bypass everything.
        assert!(!no_proxy_matches("", "example.com", None));
        assert!(!no_proxy_matches(" , ", "example.com", None));
    }

    /// IPv4 CIDR entries match IP-literal hosts, as `requests` does.
    #[test]
    fn no_proxy_matches_ipv4_cidr() {
        assert!(no_proxy_matches("10.0.0.0/8", "10.1.2.3", None));
        assert!(!no_proxy_matches("10.0.0.0/8", "11.1.2.3", None));
        assert!(no_proxy_matches("192.168.1.0/24", "192.168.1.77", Some(80)));
        assert!(!no_proxy_matches("192.168.1.0/24", "192.168.2.1", None));
        assert!(no_proxy_matches("0.0.0.0/0", "8.8.8.8", None));
        // CIDR entries never match hostnames.
        assert!(!no_proxy_matches("10.0.0.0/8", "example.com", None));
    }

    /// `host:port` entries match only that explicit port.
    #[test]
    fn no_proxy_matches_host_with_port() {
        assert!(no_proxy_matches("internal:8443", "internal", Some(8443)));
        assert!(!no_proxy_matches("internal:8443", "internal", Some(80)));
        assert!(!no_proxy_matches("internal:8443", "internal", None));
        assert!(no_proxy_matches(
            "internal:8443",
            "img.internal",
            Some(8443)
        ));
        // A port-free entry matches any port.
        assert!(no_proxy_matches("internal", "internal", Some(8443)));
    }

    /// End-to-end HTTP download against a local one-shot server, and the
    /// capped rejection of an oversized response.
    #[test]
    fn http_download_and_cap() {
        let serve = |body: Vec<u8>, content_length: u64| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let handle = std::thread::spawn(move || {
                use std::io::{BufRead, Write};
                let (stream, _) = listener.accept().unwrap();
                let mut reader = std::io::BufReader::new(stream);
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 2 {
                    line.clear(); // headers until the blank line
                }
                let mut stream = reader.into_inner();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {content_length}\r\n\r\n"
                )
                .unwrap();
                stream.write_all(&body).unwrap();
            });
            (addr, handle)
        };

        let (addr, handle) = serve(b"tiny image".to_vec(), 10);
        assert_eq!(
            fetch_bytes(&format!("http://{addr}/img.png")).unwrap(),
            b"tiny image"
        );
        handle.join().unwrap();

        // A response over the cap is rejected without buffering it all.
        let over = MAX_FETCH_BYTES + 2;
        let (addr, handle) = serve(vec![0u8; over as usize], over);
        let err = fetch_bytes(&format!("http://{addr}/big.png"))
            .err()
            .unwrap();
        assert!(err.contains("exceeds"), "{err}");
        drop(handle); // server thread may die on the closed socket; don't join
    }
}
