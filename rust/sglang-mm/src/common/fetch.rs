//! Stage 1 of the native MM pipeline: resolve one media source to raw bytes.
//!
//! Mirrors the Python `get_image_bytes` source handling (and its precedence):
//! raw bytes, `http(s)://` (bounded download, `REQUEST_TIMEOUT` env like
//! Python), `file://` / absolute path, `data:` URL, else bare base64.

use base64::Engine;

/// Cap on a fetched payload so a bad URL can't buffer unboundedly.
const MAX_FETCH_BYTES: u64 = 64 << 20;

/// Why a source couldn't be resolved natively.
#[derive(Debug)]
pub enum FetchError {
    /// Recognized-but-unsupported shape (e.g. a precomputed-feature dict):
    /// route the whole request to the Python fallback, don't fail it.
    Unsupported(String),
    /// A real failure (bad URL, network/file error, oversized payload):
    /// reject the request, matching the Python per-request exception → 400.
    Failed(String),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::Unsupported(s) => write!(f, "unsupported media source: {s}"),
            FetchError::Failed(s) => write!(f, "media fetch failed: {s}"),
        }
    }
}

/// Resolve one string-typed image source into raw encoded-image bytes.
pub fn fetch_bytes(src: &str) -> Result<Vec<u8>, FetchError> {
    if src.starts_with("http://") || src.starts_with("https://") {
        return http_get(src);
    }
    if let Some(path) = src.strip_prefix("file://") {
        return std::fs::read(path).map_err(|e| FetchError::Failed(format!("{path}: {e}")));
    }
    if src.starts_with('/') {
        return std::fs::read(src).map_err(|e| FetchError::Failed(format!("{src}: {e}")));
    }
    if let Some(rest) = src.strip_prefix("data:") {
        let encoded = rest
            .split_once(',')
            .ok_or_else(|| FetchError::Failed("malformed data: URL".into()))?
            .1;
        return b64(encoded);
    }
    // Python treats any other string as bare base64.
    b64(src)
}

fn b64(encoded: &str) -> Result<Vec<u8>, FetchError> {
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim().as_bytes())
        .map_err(|e| FetchError::Failed(format!("base64 decode: {e}")))
}

fn http_get(url: &str) -> Result<Vec<u8>, FetchError> {
    // Python: `int(os.getenv("REQUEST_TIMEOUT", "3"))` seconds per image GET.
    let timeout = std::env::var("REQUEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(3);
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(timeout))
        .call()
        .map_err(|e| FetchError::Failed(format!("GET {url}: {e}")))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .take(MAX_FETCH_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| FetchError::Failed(format!("read {url}: {e}")))?;
    if buf.len() as u64 > MAX_FETCH_BYTES {
        return Err(FetchError::Failed(format!(
            "{url}: response exceeds {MAX_FETCH_BYTES} bytes"
        )));
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
        assert!(matches!(
            fetch_bytes("!!not-base64!!"),
            Err(FetchError::Failed(_))
        ));
    }

    #[test]
    fn missing_file_fails() {
        assert!(matches!(
            fetch_bytes("file:///definitely/not/here.jpg"),
            Err(FetchError::Failed(_))
        ));
        assert!(matches!(
            fetch_bytes("/definitely/not/here.jpg"),
            Err(FetchError::Failed(_))
        ));
    }
}
