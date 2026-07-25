//! Stage 1 of the server MM pipeline: resolve one media source to raw bytes.
//!
//! Mirrors the Python `get_image_bytes` source handling (and its precedence):
//! raw bytes, `http(s)://` (bounded download, `REQUEST_TIMEOUT` env like
//! Python), `file://` / absolute path, `data:` URL, else bare base64.

use base64::Engine;

/// Cap on a fetched payload so a bad URL can't buffer unboundedly.
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
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim().as_bytes())
        .map_err(|e| format!("media fetch: base64 decode: {e}"))
}

fn http_get(url: &str) -> Result<Vec<u8>, String> {
    // Python: `int(os.getenv("REQUEST_TIMEOUT", "3"))` seconds per image GET.
    let timeout = std::env::var("REQUEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(3);
    let resp = ureq::get(url)
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

use std::io::Read;

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
}
