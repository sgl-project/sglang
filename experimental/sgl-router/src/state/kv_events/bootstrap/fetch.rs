//! The HTTP client side of peer bootstrap: URLs, a bounded read and the gzip or identity decode.

use std::io::Read;
use std::time::Duration;

use flate2::read::GzDecoder;
use tracing::debug;

use super::{PeerSnapshot, CURSORS_ONLY_PARAM, MAX_AGE_PARAM, SNAPSHOT_PATH};

/// Ceiling on a peer snapshot body, both as buffered off the wire and as
/// inflated, as a resource guard rather than a format bound.
///
/// The producing route is unauthenticated, so without it one response could
/// demand an arbitrarily large allocation — an identity body directly, a gzip
/// body from a small response — and an OOM kills a booting router before it
/// can log. A legitimate body is bounded by the fleet's total KV capacity in
/// blocks, orders of magnitude below this. Generous on purpose: rejecting a
/// real snapshot costs a cold boot, so this should only ever catch a body no
/// honest peer would send.
const MAX_INFLATED_SNAPSHOT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// The URL a snapshot fetch goes to, carrying the caller's freshness
/// requirement when it has one.
fn snapshot_url(peer_base_url: &str, max_age: Option<Duration>) -> String {
    let base = format!("{}{}", peer_base_url.trim_end_matches('/'), SNAPSHOT_PATH);
    match max_age {
        // Saturating rather than wrapping: an absurd duration must read as "any
        // age will do", never as a near-zero age that forces a needless rebuild.
        Some(d) => {
            let ms = u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
            format!("{base}?{MAX_AGE_PARAM}={ms}")
        }
        None => base,
    }
}

/// The URL a cursors-only fetch goes to. Carries no freshness requirement: the
/// producer reads its cursors live on this path, so there is no cached
/// generation to negotiate against. An empty base yields a path-relative URI,
/// usable against an in-process router.
fn cursors_url(peer_base_url: &str) -> String {
    format!(
        "{}{}?{CURSORS_ONLY_PARAM}=true",
        peer_base_url.trim_end_matches('/'),
        SNAPSHOT_PATH,
    )
}

/// What a snapshot fetch got back.
pub enum FetchAnswer {
    Body(PeerSnapshot),
    /// The peer answered non-success. The status distinguishes "an older
    /// router image that does not serve this route" (404) from a sick peer
    /// (5xx) — both retriable, but the first-occurrence log should name which.
    NoBody(reqwest::StatusCode),
}

/// Fetch one peer's snapshot. [`FetchAnswer::NoBody`] means "peer reachable
/// but has no snapshot to give" (non-200, including an older image's 404).
///
/// `max_age` is the oldest export the caller accepts (see
/// [`PRODUCER_CACHE_TTL`]); `None` sends no requirement, as an older consumer
/// does. An older producer ignores the parameter.
///
/// [`PRODUCER_CACHE_TTL`]: super::PRODUCER_CACHE_TTL
pub async fn fetch_snapshot(
    http: &reqwest::Client,
    peer_base_url: &str,
    max_age: Option<Duration>,
) -> Result<FetchAnswer, anyhow::Error> {
    fetch_body(http, &snapshot_url(peer_base_url, max_age), peer_base_url).await
}

/// Fetch one peer's cursor table (`?cursors_only=true`). A producer that
/// ignores the parameter answers with a full snapshot, whose cursor table
/// lists only ranks still carrying nodes; both decode to [`PeerSnapshot`].
/// `None` on non-success.
pub async fn fetch_cursors(
    http: &reqwest::Client,
    peer_base_url: &str,
) -> Result<Option<PeerSnapshot>, anyhow::Error> {
    Ok(
        match fetch_body(http, &cursors_url(peer_base_url), peer_base_url).await? {
            FetchAnswer::Body(snap) => Some(snap),
            FetchAnswer::NoBody(_) => None,
        },
    )
}

/// Shared transport for both fetch shapes: send, branch on what the peer
/// actually encoded, and decode off the runtime.
///
/// `peer_base_url` is carried separately from `url` only so the log names the
/// peer rather than a URL with a query string on it.
///
/// Asks for gzip on this request alone — the snapshot is the largest body the
/// router transfers — rather than via reqwest's `gzip` feature, which is
/// crate-wide and would make every client, including the SSE proxy, advertise
/// and auto-decode it.
async fn fetch_body(
    http: &reqwest::Client,
    url: &str,
    peer_base_url: &str,
) -> Result<FetchAnswer, anyhow::Error> {
    let resp = http
        .get(url)
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await?;
    if !resp.status().is_success() {
        debug!(
            peer = %peer_base_url,
            status = %resp.status(),
            "kv-bootstrap: peer returned no usable body for the snapshot request",
        );
        return Ok(FetchAnswer::NoBody(resp.status()));
    }
    // Branch on what the peer actually sent, not on what we asked for: a router
    // image that predates route compression answers identity, and must keep
    // working. reqwest leaves both the header and the body untouched here because
    // its `gzip` feature is off.
    //
    // Exact-match on purpose. The only legitimate producer is another router of
    // this codebase, which sends the bare token, so a compound or non-canonical
    // coding means an intermediary rewrote the body — better surfaced as a decode
    // failure than guessed at.
    let gzipped = resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"gzip"));
    // Bounded before buffering; see [`MAX_INFLATED_SNAPSHOT_BYTES`].
    let body = read_bounded(resp, peer_base_url).await?;
    // Inflating and parsing a large tree is seconds of CPU with no await point,
    // on the runtime that also proxies requests (including SSE), so it runs on
    // the blocking pool. A cursors-only body is cheap, but the full-snapshot
    // answer an older producer gives to the same request is not, and one
    // transport path is worth more than saving a task hop on the cheap case.
    tokio::task::spawn_blocking(move || decode_snapshot(&body, gzipped))
        .await?
        .map(FetchAnswer::Body)
}

/// Most a declared `Content-Length` may pre-allocate for the response buffer.
/// The header is peer-controlled, so it sizes the first allocation only up to
/// this; a larger body grows the buffer as it streams in.
const READ_PREALLOC_CAP: u64 = 64 * 1024 * 1024;

/// Most a gzip trailer's ISIZE may pre-allocate for the inflate buffer. The
/// trailer is peer-controlled and records the size modulo 2^32, so it is only
/// a hint; the `take` bound in [`decode_snapshot`] still enforces the ceiling.
const INFLATE_PREALLOC_CAP: u64 = 256 * 1024 * 1024;

/// The error for a body that would pass [`MAX_INFLATED_SNAPSHOT_BYTES`].
fn past_ceiling(what: std::fmt::Arguments<'_>) -> anyhow::Error {
    anyhow::anyhow!(
        "{what} past the {MAX_INFLATED_SNAPSHOT_BYTES}-byte ceiling; refusing to buffer it"
    )
}

/// Buffer a response body, refusing one that grows past
/// [`MAX_INFLATED_SNAPSHOT_BYTES`].
async fn read_bounded(
    mut resp: reqwest::Response,
    peer_base_url: &str,
) -> Result<bytes::Bytes, anyhow::Error> {
    let declared = resp.content_length();
    if declared.is_some_and(|n| n > MAX_INFLATED_SNAPSHOT_BYTES) {
        return Err(past_ceiling(format_args!(
            "peer {peer_base_url} declares a snapshot"
        )));
    }
    let prealloc = declared.map_or(0, |n| n.min(READ_PREALLOC_CAP)) as usize;
    let mut buf = bytes::BytesMut::with_capacity(prealloc);
    // Counted across chunks as well, because `Content-Length` is absent on a
    // chunked response.
    while let Some(chunk) = resp.chunk().await? {
        if buf.len() as u64 + chunk.len() as u64 > MAX_INFLATED_SNAPSHOT_BYTES {
            return Err(past_ceiling(format_args!("peer {peer_base_url} streamed")));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf.freeze())
}

/// Inflate (when gzipped) and parse one snapshot body. Blocking and CPU-bound —
/// callers run it off the async runtime.
fn decode_snapshot(body: &[u8], gzipped: bool) -> Result<PeerSnapshot, anyhow::Error> {
    if !gzipped {
        return Ok(serde_json::from_slice(body)?);
    }
    // The gzip trailer's last four bytes are ISIZE, the inflated length
    // (little-endian, modulo 2^32).
    let isize_hint = body
        .last_chunk::<4>()
        .map_or(0, |b| u64::from(u32::from_le_bytes(*b)));
    let prealloc = isize_hint
        .min(MAX_INFLATED_SNAPSHOT_BYTES)
        .min(INFLATE_PREALLOC_CAP) as usize;
    let mut inflated = Vec::with_capacity(prealloc);
    // Bounded before buffering; see [`MAX_INFLATED_SNAPSHOT_BYTES`].
    let read = GzDecoder::new(body)
        .take(MAX_INFLATED_SNAPSHOT_BYTES + 1)
        .read_to_end(&mut inflated)?;
    if read as u64 > MAX_INFLATED_SNAPSHOT_BYTES {
        return Err(past_ceiling(format_args!("peer snapshot inflates")));
    }
    // Parse from the buffered slice rather than streaming the decoder into
    // serde: serde_json parses a slice much faster than a reader.
    Ok(serde_json::from_slice(&inflated)?)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::sample_snapshot;
    use super::*;

    /// A caller with no freshness requirement must send a bare path, so an older
    /// router image sees the request it has always seen.
    #[test]
    fn snapshot_url_omits_the_parameter_when_any_age_will_do() {
        assert_eq!(
            snapshot_url("http://peer:3000", None),
            format!("http://peer:3000{SNAPSHOT_PATH}"),
        );
        // Trailing slash on the base must not produce a doubled separator.
        assert_eq!(
            snapshot_url("http://peer:3000/", None),
            format!("http://peer:3000{SNAPSHOT_PATH}"),
        );
    }

    /// A cursors-only fetch carries the parameter that makes the producer skip
    /// its tree.
    #[test]
    fn cursors_url_asks_for_the_cursor_table_alone() {
        assert_eq!(
            cursors_url("http://peer:3000"),
            format!("http://peer:3000{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
        // Trailing slash on the base must not produce a doubled separator.
        assert_eq!(
            cursors_url("http://peer:3000/"),
            format!("http://peer:3000{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
        // An empty base yields the path-relative URI, for in-process routers.
        assert_eq!(
            cursors_url(""),
            format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
    }

    #[test]
    fn snapshot_url_states_the_requirement_in_milliseconds() {
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::from_millis(1500))),
            format!("http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=1500"),
        );
        // Zero is the strictest request, not an absent one.
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::ZERO)),
            format!("http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=0"),
        );
    }

    /// Saturate rather than wrap. A wrapped duration would read as a near-zero
    /// age and force the peer into a fleet-wide rebuild it was never asked for.
    #[test]
    fn snapshot_url_saturates_an_absurd_age() {
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::MAX)),
            format!(
                "http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}={}",
                u64::MAX
            ),
        );
    }

    /// The consumer decodes both encodings a producer can send: identity from
    /// an image that predates route compression, gzip from one that has it.
    #[test]
    fn decode_snapshot_reads_identity_and_gzip_bodies() {
        use std::io::Write;

        let snap = sample_snapshot();
        let json = serde_json::to_vec(&snap).unwrap();
        assert_eq!(decode_snapshot(&json, false).unwrap(), snap);

        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(&json).unwrap();
        let gz = enc.finish().unwrap();
        assert_eq!(decode_snapshot(&gz, true).unwrap(), snap);

        assert!(
            decode_snapshot(&gz, false).is_err(),
            "gzip bytes labelled identity must fail to parse, not half-decode",
        );
        assert!(
            decode_snapshot(&json, true).is_err(),
            "identity bytes labelled gzip must fail to inflate",
        );
    }
}
