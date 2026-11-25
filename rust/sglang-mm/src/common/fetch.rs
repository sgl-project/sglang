//! Stage 1 of the server MM pipeline: resolve one media source to raw bytes.
//!
//! Mirrors the Python `get_image_bytes` source handling (and its precedence):
//! raw bytes, `http(s)://` (bounded direct download), local files, `data:` URL, else bare
//! base64.

use base64::Engine;
use std::io::Read;
use std::net::{IpAddr, ToSocketAddrs};
use std::time::{Duration, Instant};
use url::Url;

#[derive(Clone, Debug)]
pub struct MediaPolicy {
    pub domains: Vec<String>,
    pub allow_private: bool,
    pub allow_local: bool,
    pub max_file_bytes: u64,
    pub timeout: Option<Duration>,
}

impl Default for MediaPolicy {
    fn default() -> Self {
        Self {
            domains: Vec::new(),
            allow_private: false,
            allow_local: true,
            max_file_bytes: MAX_FETCH_BYTES,
            timeout: None,
        }
    }
}

#[cfg(feature = "python")]
pub(crate) fn python_media_policy(py: pyo3::Python<'_>) -> pyo3::PyResult<MediaPolicy> {
    use pyo3::prelude::*;
    let config = py
        .import("sglang.srt.multimodal.media_io")?
        .call_method0("media_url_security_config")?;
    let (domains, megabytes, seconds, allow_local, allow_private): (
        Vec<String>,
        u64,
        Option<f64>,
        bool,
        bool,
    ) = config.extract()?;
    let timeout = seconds
        .map(Duration::try_from_secs_f64)
        .transpose()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid media timeout"))?;
    Ok(MediaPolicy {
        domains,
        allow_private,
        allow_local,
        max_file_bytes: megabytes.saturating_mul(1 << 20),
        timeout,
    })
}

#[derive(serde::Deserialize)]
struct AddressPolicy {
    exceptions: Vec<ipnet::IpNet>,
    excluded: Vec<ipnet::IpNet>,
    unicast: Vec<ipnet::IpNet>,
}

fn public_address(ip: IpAddr) -> bool {
    static NETWORKS: std::sync::LazyLock<AddressPolicy> = std::sync::LazyLock::new(|| {
        serde_json::from_str(include_str!(
            "../../../../python/sglang/srt/multimodal/media_networks.json"
        ))
        .expect("valid bundled media address policy")
    });
    let ip = ip.to_canonical();
    let matches = |nets: &[ipnet::IpNet]| nets.iter().any(|net| net.contains(&ip));
    matches(&NETWORKS.exceptions) || (matches(&NETWORKS.unicast) && !matches(&NETWORKS.excluded))
}

/// Default byte limit for one remote or local media file.
pub const MAX_FETCH_BYTES: u64 = 64 << 20;

/// A byte allowance shared by every source of one request, charged *as they
/// stream*, so concurrent fetches stop at their combined size rather than each
/// stopping at [`MAX_FETCH_BYTES`].
#[derive(Debug)]
pub struct ByteBudget(std::sync::atomic::AtomicU64);

impl ByteBudget {
    pub fn new(total: u64) -> Self {
        Self(std::sync::atomic::AtomicU64::new(total))
    }

    /// Claim `n` bytes, or `Err` once the allowance is spent.
    fn claim(&self, n: u64) -> Result<(), ()> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        self.0
            .fetch_update(AcqRel, Acquire, |left| left.checked_sub(n))
            .map(|_| ())
            .map_err(|_| ())
    }

    fn remaining(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Charge bytes which were already materialized by an earlier pipeline
    /// stage. This lets the consumer apply one whole-request bound across
    /// prefetched I/O and inline payloads without reading the source twice.
    pub fn charge_existing(&self, n: usize, what: &str) -> Result<(), String> {
        self.claim(n as u64).map_err(|()| over_budget(what))
    }

    /// Give back bytes claimed for a chunk but not filled by the read.
    fn release(&self, n: u64) {
        self.0.fetch_add(n, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Resolve one string-typed image source into raw encoded-image bytes.
/// An `Err` rejects the request, matching the Python per-request
/// exception → 400.
pub fn fetch_bytes(src: &str) -> Result<Vec<u8>, String> {
    fetch_bytes_budgeted(src, &ByteBudget::new(MAX_FETCH_BYTES))
}

fn fetch_local_with_policy(
    src: &str,
    budget: &ByteBudget,
    policy: &MediaPolicy,
) -> Result<Vec<u8>, String> {
    if !policy.allow_local {
        return Err("Local media files are not allowed".into());
    }
    let path = if src.starts_with("file://") {
        Url::parse(src)
            .ok()
            .and_then(|url| url.to_file_path().ok())
            .ok_or("Invalid local media URL")?
    } else {
        std::path::PathBuf::from(src)
    };
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|e| format!("media fetch: open local file: {}", e.kind()))?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("media fetch: stat local file: {e}"))?;
    if !metadata.is_file() {
        return Err("media fetch: not a regular file".into());
    }
    let expected = metadata.len();
    if policy.max_file_bytes != 0 && expected > policy.max_file_bytes {
        return Err("Local media exceeds the file size limit".into());
    }
    budget
        .claim(expected)
        .map_err(|()| over_budget("local file"))?;
    let mut buf = Vec::with_capacity(usize::try_from(expected).unwrap_or(usize::MAX));
    let read = file
        .take(expected.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| format!("media fetch: read local file: {e}"))? as u64;
    if read > expected {
        return Err("media fetch: local file changed size while reading".into());
    }
    budget.release(expected - read);
    Ok(buf)
}

/// [`fetch_bytes`] against a caller-owned allowance, for resolving several
/// sources under one whole-request bound. [`MAX_FETCH_BYTES`] still caps I/O
/// streams; already-resident base64 is bounded by `budget`.
pub fn fetch_bytes_budgeted(src: &str, budget: &ByteBudget) -> Result<Vec<u8>, String> {
    fetch_with_policy(src, budget, &MediaPolicy::default())
}

pub fn fetch_with_policy(
    src: &str,
    budget: &ByteBudget,
    policy: &MediaPolicy,
) -> Result<Vec<u8>, String> {
    if src.starts_with("http://") || src.starts_with("https://") {
        return http_get(src, budget, policy);
    }
    if src.starts_with("file://")
        || src.starts_with('/') && policy.allow_local && std::path::Path::new(src).exists()
    {
        return fetch_local_with_policy(src, budget, policy);
    }
    if let Some(rest) = src.strip_prefix("data:") {
        let encoded = rest
            .split_once(',')
            .ok_or_else(|| "media fetch: malformed data: URL".to_string())?
            .1;
        return decode_base64_budgeted(encoded, budget);
    }
    // Python treats any other string as bare base64.
    decode_base64_budgeted(src, budget)
}

/// Reserve the maximum decoded size before allocating. The reservation is
/// reconciled with the exact size afterwards because trailing padding can
/// reduce the result by up to two bytes.
fn decode_base64_budgeted(encoded: &str, budget: &ByteBudget) -> Result<Vec<u8>, String> {
    let encoded = encoded.trim();
    let padding = encoded
        .as_bytes()
        .iter()
        .rev()
        .take_while(|&&byte| byte == b'=')
        .take(2)
        .count() as u64;
    let estimate = (encoded.len() as u64)
        .checked_add(3)
        .and_then(|n| n.checked_div(4))
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_sub(padding))
        .ok_or_else(|| over_budget("base64 payload"))?;
    let remaining = budget.remaining();
    budget.claim(estimate).map_err(|()| {
        format!(
            "{} (decoded size {estimate} bytes, {remaining} bytes remaining)",
            over_budget("base64 payload")
        )
    })?;
    match base64::engine::general_purpose::STANDARD.decode(encoded.as_bytes()) {
        Ok(decoded) => {
            budget.release(estimate - decoded.len() as u64);
            Ok(decoded)
        }
        Err(error) => {
            budget.release(estimate);
            Err(format!("media fetch: base64 decode: {error}"))
        }
    }
}

fn over_budget(what: &str) -> String {
    format!("media fetch: {what}: exceeds the request media byte budget")
}

fn read_capped(mut reader: impl Read, budget: &ByteBudget, limit: u64) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 64 << 10];
    loop {
        let n = reader
            .read(&mut chunk)
            .map_err(|e| format!("media fetch: response read: {e}"))?;
        if n == 0 {
            return Ok(buf);
        }
        if limit != 0 && (buf.len() + n) as u64 > limit {
            return Err(format!("media fetch: response exceeds {limit} bytes"));
        }
        budget.charge_existing(n, "response")?;
        buf.extend_from_slice(&chunk[..n]);
    }
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

fn http_get(source: &str, budget: &ByteBudget, policy: &MediaPolicy) -> Result<Vec<u8>, String> {
    let mut url = Url::parse(source).map_err(|_| "Invalid media URL")?;
    let timeout = policy.timeout.unwrap_or_else(|| {
        Duration::from_secs(
            std::env::var("REQUEST_TIMEOUT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(3),
        )
    });
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or("Invalid media timeout")?;
    for _ in 0..=5 {
        if !matches!(url.scheme(), "http" | "https") {
            return Err("Expected an HTTP(S) media URL".into());
        }
        let host = url
            .host_str()
            .ok_or("Missing media host")?
            .trim_matches(['[', ']'])
            .trim_end_matches('.');
        let listed = policy.domains.iter().any(|d| d == host);
        if !policy.domains.is_empty() && !listed {
            return Err("Media URL domain is not allowed".into());
        }
        let proxy_key = format!("{}_proxy", url.scheme());
        if !bypasses_proxy(host, url.port())
            && [
                &proxy_key[..],
                &proxy_key.to_ascii_uppercase(),
                "all_proxy",
                "ALL_PROXY",
            ]
            .iter()
            .any(|key| std::env::var(key).is_ok_and(|v| !v.is_empty()))
        {
            return Err("Pinned media connections require a direct route; configure NO_PROXY for the media origin".into());
        }
        let port = url.port_or_known_default().ok_or("Missing media port")?;
        let addresses: Vec<_> = (host, port)
            .to_socket_addrs()
            .map_err(|_| "Could not resolve media host")?
            .collect();
        if addresses.is_empty()
            || (!listed
                && !policy.allow_private
                && addresses.iter().any(|addr| !public_address(addr.ip())))
        {
            return Err("Media address is not permitted".into());
        }
        // This agent serves exactly one hop. TLS and Host use the original
        // URL while the connector receives only the approved socket addresses.
        let agent = ureq::AgentBuilder::new()
            .try_proxy_from_env(false)
            .redirects(0)
            .resolver(move |_: &str| Ok(addresses.clone()))
            .build();
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or("Media fetch timed out")?;
        let response = agent
            .get(url.as_str())
            .timeout(remaining)
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(code, _) => format!("media fetch: HTTP {code}"),
                ureq::Error::Transport(t) => format!("media fetch: {}", t.kind()),
            })?;
        if matches!(response.status(), 301 | 302 | 303 | 307 | 308)
            && let Some(location) = response.header("Location")
        {
            url = url.join(location).map_err(|_| "Invalid media redirect")?;
            continue;
        }
        return read_capped(response.into_reader(), budget, policy.max_file_bytes);
    }
    Err("Too many media redirects".into())
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
    fn trusted_local_reader_rejects_non_regular_files() {
        let err = fetch_with_policy("/dev/zero", &ByteBudget::new(1024), &MediaPolicy::default())
            .err()
            .unwrap();
        assert!(err.contains("not a regular file"), "{err}");
    }

    /// The convenience API keeps its 64 MiB budget, while a server request may
    /// supply a larger bounded allowance for already-resident inline media.
    #[test]
    fn inline_base64_uses_the_supplied_request_budget() {
        let encoded = "A".repeat((MAX_FETCH_BYTES / 3 * 4 + 8) as usize);
        let err = fetch_bytes(&encoded).err().unwrap();
        assert!(err.contains("exceeds"), "{err}");
        let decoded = fetch_bytes_budgeted(&encoded, &ByteBudget::new(MAX_FETCH_BYTES + 16))
            .expect("larger request budget admits inline media over the remote-fetch cap");
        assert!(decoded.len() as u64 > MAX_FETCH_BYTES);
    }

    #[test]
    fn base64_uses_the_callers_exact_budget() {
        let encoded = base64::engine::general_purpose::STANDARD.encode(b"a");
        assert_eq!(
            fetch_bytes_budgeted(&encoded, &ByteBudget::new(1)).unwrap(),
            b"a"
        );
        let err = fetch_bytes_budgeted(&encoded, &ByteBudget::new(0))
            .err()
            .unwrap();
        assert!(err.contains("request media byte budget"), "{err}");
    }

    /// One budget spans sources: each fits alone, the set does not.
    #[test]
    fn shared_budget_spans_sources() {
        let payload = base64::engine::general_purpose::STANDARD.encode([7u8; 4096]);
        let budget = ByteBudget::new(6144);
        assert_eq!(fetch_bytes_budgeted(&payload, &budget).unwrap().len(), 4096);
        let err = fetch_bytes_budgeted(&payload, &budget).err().unwrap();
        assert!(err.contains("request media byte budget"), "{err}");
    }

    /// Small local files consume exactly their size.
    #[test]
    fn exact_local_sizes_share_the_budget() {
        let path = std::env::temp_dir().join(format!("sglang-budget-{}", std::process::id()));
        std::fs::write(&path, [0u8; 1024]).unwrap();
        let src = path.display().to_string();
        let budget = ByteBudget::new(4 * 1024);
        for _ in 0..4 {
            assert_eq!(fetch_bytes_budgeted(&src, &budget).unwrap().len(), 1024);
        }
        std::fs::remove_file(&path).ok();
    }

    /// The per-source cap holds even under a larger shared budget.
    #[test]
    fn per_source_cap_survives_a_large_budget() {
        let budget = ByteBudget::new(MAX_FETCH_BYTES * 4);
        let err = read_capped(std::io::repeat(0), &budget, MAX_FETCH_BYTES)
            .err()
            .unwrap();
        assert!(err.contains(&format!("exceeds {MAX_FETCH_BYTES}")), "{err}");
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
            fetch_with_policy(
                &format!("http://{addr}/img.png"),
                &ByteBudget::new(MAX_FETCH_BYTES),
                &MediaPolicy {
                    allow_private: true,
                    ..Default::default()
                }
            )
            .unwrap(),
            b"tiny image"
        );
        handle.join().unwrap();

        // A response over the cap is rejected without buffering it all.
        let over = MAX_FETCH_BYTES + 2;
        let (addr, handle) = serve(vec![0u8; over as usize], over);
        let err = fetch_with_policy(
            &format!("http://{addr}/big.png"),
            &ByteBudget::new(MAX_FETCH_BYTES + 10),
            &MediaPolicy {
                allow_private: true,
                ..Default::default()
            },
        )
        .err()
        .unwrap();
        assert!(err.contains("exceeds"), "{err}");
        drop(handle); // server thread may die on the closed socket; don't join
    }
}
