//! Stage 1 of the server MM pipeline: resolve one media source to raw bytes.
//!
//! Mirrors the Python `get_image_bytes` source handling (and its precedence):
//! raw bytes, `http(s)://` (bounded download, `REQUEST_TIMEOUT` and the proxy
//! env vars like Python), `file://` / absolute path, `data:` URL, else bare
//! base64.

use std::io::Read;
use std::sync::OnceLock;

use base64::Engine;

/// Cap on one remotely fetched payload. Inline base64 and trusted local files
/// use their caller's whole-request budget instead.
pub const MAX_FETCH_BYTES: u64 = 64 << 20;

/// Charge granularity of a streaming read: the most an in-flight source can
/// over-charge a shared [`ByteBudget`] by.
const CHUNK_BYTES: u64 = 256 << 10;

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

    /// Atomically reserve up to `max` available bytes. Unlike loading
    /// [`Self::remaining`] and then calling [`Self::claim`], this never races
    /// another source for the same allowance.
    fn claim_up_to(&self, max: u64) -> u64 {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let mut left = self.0.load(Acquire);
        loop {
            let claimed = left.min(max);
            if claimed == 0 {
                return 0;
            }
            match self
                .0
                .compare_exchange_weak(left, left - claimed, AcqRel, Acquire)
            {
                Ok(_) => return claimed,
                Err(actual) => left = actual,
            }
        }
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

/// Read a trusted local media path without applying the per-source remote cap.
///
/// Python's media security limit is specifically a URL-download limit. Local
/// video fixtures and mounted production assets are commonly larger than 64
/// MiB, so applying [`MAX_FETCH_BYTES`] to them breaks requests which the Python
/// frontend accepts. They still consume the caller's whole-request budget.
/// Reject non-regular files and charge their size before reading so a request
/// cannot turn devices or a huge sparse file into an unbounded allocation.
pub fn fetch_local_file_budgeted(src: &str, budget: &ByteBudget) -> Result<Vec<u8>, String> {
    let path = src.strip_prefix("file://").unwrap_or(src);
    let file = std::fs::File::open(path).map_err(|e| format!("media fetch: {path}: {e}"))?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("media fetch: stat {path}: {e}"))?;
    if !metadata.is_file() {
        return Err(format!("media fetch: {path}: not a regular file"));
    }
    let expected = metadata.len();
    budget.claim(expected).map_err(|()| over_budget(path))?;
    let mut buf = Vec::with_capacity(usize::try_from(expected).unwrap_or(usize::MAX));
    let read = file
        .take(expected.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| format!("media fetch: read {path}: {e}"))? as u64;
    if read > expected {
        return Err(format!("media fetch: {path}: changed size while reading"));
    }
    budget.release(expected - read);
    Ok(buf)
}

/// [`fetch_bytes`] against a caller-owned allowance, for resolving several
/// sources under one whole-request bound. [`MAX_FETCH_BYTES`] still caps I/O
/// streams; already-resident base64 is bounded by `budget`.
pub fn fetch_bytes_budgeted(src: &str, budget: &ByteBudget) -> Result<Vec<u8>, String> {
    if src.starts_with("http://") || src.starts_with("https://") {
        return http_get(src, budget);
    }
    if let Some(path) = src.strip_prefix("file://") {
        return read_file(path, budget);
    }
    if src.starts_with('/') {
        return read_file(src, budget);
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

/// Bounded read: never trusts metadata, so huge and non-regular files
/// (`/dev/zero`) hit the cap instead of exhausting memory.
fn read_file(path: &str, budget: &ByteBudget) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("media fetch: {path}: {e}"))?;
    read_capped(file, path, budget)
}

/// Read to EOF, charging `budget` per chunk, so an oversized source stops
/// mid-stream instead of going fully resident first.
fn read_capped(mut reader: impl Read, what: &str, budget: &ByteBudget) -> Result<Vec<u8>, String> {
    let too_big = || format!("media fetch: {what}: exceeds {MAX_FETCH_BYTES} bytes");
    let mut buf = Vec::new();
    'read: loop {
        // `+ 1`: reserve room for one byte past the per-source cap, so an
        // oversized source is rejected rather than silently truncated.
        let want = CHUNK_BYTES.min(MAX_FETCH_BYTES + 1 - buf.len() as u64);
        let reserved = budget.claim_up_to(want);
        if reserved == 0 {
            // The budget is payload capacity, so an exact-fit payload can
            // leave no room for another reservation. Probe one stack byte to
            // distinguish EOF from overflow. A returned byte is charged
            // before appending, and the transient probe is bounded to one
            // byte per active reader.
            let mut probe = [0u8; 1];
            loop {
                match reader.read(&mut probe) {
                    Ok(0) => return Ok(buf),
                    Ok(_) => {
                        if buf.len() as u64 >= MAX_FETCH_BYTES {
                            return Err(too_big());
                        }
                        budget.claim(1).map_err(|()| over_budget(what))?;
                        buf.push(probe[0]);
                        continue 'read;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(format!("media fetch: read {what}: {error}")),
                }
            }
        }
        let read = reader
            .by_ref()
            .take(reserved)
            .read_to_end(&mut buf)
            .map_err(|e| format!("media fetch: read {what}: {e}"))? as u64;
        // Read errors retain the reservation until this request fails, matching
        // the existing budget lifetime. Successful short reads refund the part
        // that never became payload.
        budget.release(reserved - read);
        if buf.len() as u64 > MAX_FETCH_BYTES {
            return Err(too_big());
        }
        if read < reserved {
            return Ok(buf); // short read == EOF
        }
    }
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

fn http_get(url: &str, budget: &ByteBudget) -> Result<Vec<u8>, String> {
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
    read_capped(resp.into_reader(), url, budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ShortReader {
        inner: std::io::Cursor<Vec<u8>>,
        max_read: usize,
    }

    impl Read for ShortReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = buf.len().min(self.max_read);
            self.inner.read(&mut buf[..n])
        }
    }

    struct ErrorAfterPartialRead {
        sent: bool,
    }

    impl Read for ErrorAfterPartialRead {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.sent {
                return Err(std::io::Error::other("injected read failure"));
            }
            self.sent = true;
            let n = buf.len().min(3);
            buf[..n].fill(7);
            Ok(n)
        }
    }

    struct BarrierReader {
        inner: std::io::Cursor<Vec<u8>>,
        barrier: std::sync::Arc<std::sync::Barrier>,
        waited: bool,
    }

    impl Read for BarrierReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if !self.waited {
                self.waited = true;
                self.barrier.wait();
            }
            self.inner.read(buf)
        }
    }

    struct TempFile(std::path::PathBuf);

    impl TempFile {
        fn new(bytes: &[u8]) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("sglang-mm-budget-{}-{id}", std::process::id()));
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

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
        let err = fetch_local_file_budgeted("/dev/zero", &ByteBudget::new(1024))
            .err()
            .unwrap();
        assert!(err.contains("not a regular file"), "{err}");
    }

    /// A non-regular file must hit the byte cap, not exhaust memory.
    #[test]
    fn unbounded_file_capped() {
        let err = fetch_bytes("/dev/zero").err().unwrap();
        assert!(err.contains("exceeds"), "{err}");
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

    #[test]
    fn read_capped_accepts_empty_input_with_zero_budget() {
        assert!(
            read_capped(std::io::Cursor::new([]), "empty", &ByteBudget::new(0))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn read_capped_rejects_nonempty_input_with_zero_budget() {
        let err =
            read_capped(std::io::Cursor::new(b"x"), "nonempty", &ByteBudget::new(0)).unwrap_err();
        assert!(err.contains("request media byte budget"), "{err}");
    }

    #[test]
    fn read_capped_accepts_small_input_at_exact_budget() {
        assert_eq!(
            read_capped(std::io::Cursor::new(b"hello"), "small", &ByteBudget::new(5)).unwrap(),
            b"hello"
        );
    }

    #[test]
    fn read_capped_rejects_one_byte_over_shared_budget() {
        let err = read_capped(
            std::io::Cursor::new(b"123456"),
            "over-budget",
            &ByteBudget::new(5),
        )
        .unwrap_err();
        assert!(err.contains("request media byte budget"), "{err}");
    }

    #[test]
    fn read_capped_accepts_exact_budgets_at_chunk_boundaries() {
        for size in [CHUNK_BYTES - 1, CHUNK_BYTES, CHUNK_BYTES + 1] {
            let bytes = read_capped(
                std::io::repeat(7).take(size),
                "chunk-boundary",
                &ByteBudget::new(size),
            )
            .unwrap_or_else(|err| panic!("{size} bytes at exact budget: {err}"));
            assert_eq!(bytes.len() as u64, size);
        }
    }

    #[test]
    fn read_capped_accepts_exact_per_source_cap_at_eof() {
        let bytes = read_capped(
            std::io::repeat(7).take(MAX_FETCH_BYTES),
            "at-source-cap",
            &ByteBudget::new(MAX_FETCH_BYTES),
        )
        .expect("exact per-source-cap EOF is accepted");
        assert_eq!(bytes.len() as u64, MAX_FETCH_BYTES);
    }

    #[test]
    fn read_capped_rejects_one_byte_over_per_source_cap() {
        let err = read_capped(
            std::io::repeat(7).take(MAX_FETCH_BYTES + 1),
            "over-source-cap",
            &ByteBudget::new(MAX_FETCH_BYTES + 1),
        )
        .unwrap_err();
        assert!(
            err.contains(&format!("exceeds {MAX_FETCH_BYTES} bytes")),
            "{err}"
        );
    }

    #[test]
    fn shared_budget_accepts_multiple_sources_that_exactly_fit() {
        let first = TempFile::new(b"ab");
        let second = TempFile::new(b"cde");
        let budget = ByteBudget::new(5);

        assert_eq!(
            fetch_bytes_budgeted(first.path().to_str().unwrap(), &budget).unwrap(),
            b"ab"
        );
        assert_eq!(
            fetch_bytes_budgeted(second.path().to_str().unwrap(), &budget).unwrap(),
            b"cde"
        );
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn read_capped_short_reads_fit_the_exact_budget() {
        let reader = ShortReader {
            inner: std::io::Cursor::new(b"hello".to_vec()),
            max_read: 1,
        };
        assert_eq!(
            read_capped(reader, "short-reader", &ByteBudget::new(5)).unwrap(),
            b"hello"
        );
    }

    #[test]
    fn read_capped_propagates_injected_read_errors() {
        let err = read_capped(
            ErrorAfterPartialRead { sent: false },
            "error-reader",
            &ByteBudget::new(16),
        )
        .unwrap_err();
        assert!(err.contains("injected read failure"), "{err}");
    }

    #[test]
    fn read_capped_contention_observes_atomic_chunk_reservations() {
        let budget = std::sync::Arc::new(ByteBudget::new(2));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let mut threads = Vec::new();
        for payload in [b"ab".as_slice(), b"c".as_slice()] {
            let budget = budget.clone();
            let barrier = barrier.clone();
            let payload = payload.to_vec();
            threads.push(std::thread::spawn(move || {
                read_capped(
                    BarrierReader {
                        inner: std::io::Cursor::new(payload),
                        barrier,
                        waited: false,
                    },
                    "contended-source",
                    &budget,
                )
            }));
        }

        let results = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        let err = results
            .iter()
            .find_map(|result| result.as_ref().err())
            .unwrap();
        assert!(err.contains("request media byte budget"), "{err}");
    }

    #[test]
    fn read_capped_accepts_probe_byte_after_another_reader_refunds() {
        use std::sync::mpsc::{self, Receiver, Sender};
        use std::time::Duration;

        const TIMEOUT: Duration = Duration::from_secs(5);

        struct FirstReader {
            started: Sender<()>,
            wait_for_probe: Receiver<()>,
            sent_byte: bool,
        }

        impl Read for FirstReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.sent_byte {
                    return Ok(0);
                }
                self.started.send(()).map_err(std::io::Error::other)?;
                self.wait_for_probe
                    .recv_timeout(TIMEOUT)
                    .map_err(std::io::Error::other)?;
                self.sent_byte = true;
                buf[0] = b'a';
                Ok(1)
            }
        }

        struct ProbeReader {
            entered: Sender<()>,
            notify_first_reader: Sender<()>,
            wait_for_first_reader: Receiver<()>,
            sent_byte: bool,
        }

        impl Read for ProbeReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.sent_byte {
                    return Ok(0);
                }
                self.entered.send(()).map_err(std::io::Error::other)?;
                self.notify_first_reader
                    .send(())
                    .map_err(std::io::Error::other)?;
                self.wait_for_first_reader
                    .recv_timeout(TIMEOUT)
                    .map_err(std::io::Error::other)?;
                self.sent_byte = true;
                buf[0] = b'b';
                Ok(1)
            }
        }

        let budget = std::sync::Arc::new(ByteBudget::new(2));
        let (first_started_tx, first_started_rx) = mpsc::channel();
        let (probe_entered_tx, probe_entered_rx) = mpsc::channel();
        let (probe_to_first_tx, probe_to_first_rx) = mpsc::channel();
        let (first_done_tx, first_done_rx) = mpsc::channel();

        let first_budget = budget.clone();
        let first = std::thread::spawn(move || {
            let result = read_capped(
                FirstReader {
                    started: first_started_tx,
                    wait_for_probe: probe_to_first_rx,
                    sent_byte: false,
                },
                "first-source",
                &first_budget,
            );
            first_done_tx.send(()).unwrap();
            result
        });

        // The first read starts only after A has reserved both request bytes.
        first_started_rx.recv_timeout(TIMEOUT).unwrap();

        let second_budget = budget.clone();
        let second = std::thread::spawn(move || {
            read_capped(
                ProbeReader {
                    entered: probe_entered_tx,
                    notify_first_reader: probe_to_first_tx,
                    wait_for_first_reader: first_done_rx,
                    sent_byte: false,
                },
                "second-source",
                &second_budget,
            )
        });

        // B has failed to reserve while A holds both bytes and is now in its
        // one-byte probe. Let A reach EOF, refund one byte, and finish before
        // allowing B's probe read to return its byte.
        probe_entered_rx.recv_timeout(TIMEOUT).unwrap();

        assert_eq!(first.join().unwrap().unwrap(), b"a");
        assert_eq!(second.join().unwrap().unwrap(), b"b");
        assert_eq!(budget.remaining(), 0);
    }

    /// Unused claims come back, so small sources fit in a budget their
    /// worst-case sizes would have exhausted.
    #[test]
    fn short_reads_release_their_claim() {
        let path = std::env::temp_dir().join(format!("sglang-budget-{}", std::process::id()));
        std::fs::write(&path, [0u8; 1024]).unwrap();
        let src = path.display().to_string();
        let budget = ByteBudget::new(CHUNK_BYTES + 4096);
        for _ in 0..4 {
            assert_eq!(fetch_bytes_budgeted(&src, &budget).unwrap().len(), 1024);
        }
        std::fs::remove_file(&path).ok();
    }

    /// The per-source cap holds even under a larger shared budget.
    #[test]
    fn per_source_cap_survives_a_large_budget() {
        let budget = ByteBudget::new(MAX_FETCH_BYTES * 4);
        let err = fetch_bytes_budgeted("/dev/zero", &budget).err().unwrap();
        assert!(err.contains(&format!("exceeds {MAX_FETCH_BYTES}")), "{err}");
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
            fetch_bytes_budgeted(&format!("http://{addr}/img.png"), &ByteBudget::new(10)).unwrap(),
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
