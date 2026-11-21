//! Request-body decompression: the port of Python
//! `http_request_decompression.py`. Gated on
//! `SGLANG_ENABLE_REQUEST_DECOMPRESSION`; a request whose `x-body-compressed`
//! header names the method (`x-body-compressed: zstd`) reaches every layer and
//! handler inside it with the decompressed body.

use std::io::Read;
use std::sync::Arc;

use axum::Router;
use axum::body::{Body, HttpBody as _};
use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::Response;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::message::config::ServerArgs;

const COMPRESSION_HEADER: &str = "x-body-compressed";

/// The most a zstd frame can expand: an RLE block, 4 bytes, decodes to at most
/// 128 KiB.
const MAX_ZSTD_EXPANSION: u64 = 32 * 1024;

/// Compressed bytes of known length buffered at once. Arbitrary; no evidence
/// this is the right budget.
const BUFFERED_BUDGET_KIB: u32 = 1 << 20;

/// Install the decompression middleware when `server_args` enables it;
/// identity otherwise.
pub(super) fn apply(app: Router, server_args: &ServerArgs) -> Router {
    if server_args.enable_request_decompression {
        with_decompression(
            app,
            Limits::new(python_executor_workers(), BUFFERED_BUDGET_KIB),
        )
    } else {
        app
    }
}

/// Python decompresses on asyncio's default executor, so at most
/// `min(32, cpu_count + 4)` bodies at once; the rest wait for a thread.
fn python_executor_workers() -> usize {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .saturating_add(4)
        .min(32)
}

/// Bodies decompressing at once, and compressed bytes buffered at once, in KiB
/// because `acquire_many` takes a `u32`.
#[derive(Clone)]
struct Limits {
    decompressions: Arc<Semaphore>,
    buffered_kib: Arc<Semaphore>,
    budget_kib: u32,
}

impl Limits {
    fn new(decompressions: usize, budget_kib: u32) -> Self {
        Self {
            decompressions: Arc::new(Semaphore::new(decompressions)),
            buffered_kib: Arc::new(Semaphore::new(budget_kib as usize)),
            budget_kib,
        }
    }

    async fn decompression(&self) -> Result<OwnedSemaphorePermit, String> {
        self.decompressions
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| error.to_string())
    }

    /// A body over the whole budget takes all of it rather than waiting forever.
    async fn reserve(&self, length: u64) -> Result<OwnedSemaphorePermit, String> {
        let kib = u32::try_from(length.div_ceil(1024)).unwrap_or(u32::MAX);
        self.buffered_kib
            .clone()
            .acquire_many_owned(kib.min(self.budget_kib))
            .await
            .map_err(|error| error.to_string())
    }
}

fn with_decompression(app: Router, limits: Limits) -> Router {
    app.layer(axum::middleware::from_fn(
        move |request: Request, next: Next| decompress_request_body(limits.clone(), request, next),
    ))
}

/// Like Python, only zstd is supported, the decompressed size is not capped
/// (libzstd's default decoder refuses only frames whose window exceeds
/// 128 MiB), an empty body stays empty, and the body is decompressed off the
/// async workers. Python's rejections are 400s with a plain-text body and no
/// content type.
async fn decompress_request_body(limits: Limits, request: Request, next: Next) -> Response {
    let Some(method) = request.headers().get(COMPRESSION_HEADER) else {
        return next.run(request).await;
    };
    if method != "zstd" {
        let method = String::from_utf8_lossy(method.as_bytes());
        return bad_request(format!(
            "unsupported x-body-compressed '{method}'; supported: ['zstd']"
        ));
    }

    let (mut parts, body) = request.into_parts();
    let body = match decompress_zstd(body, &limits).await {
        Ok(body) => body,
        Err(error) => {
            tracing::warn!("request body decompress failed: {error}");
            return bad_request("decompress failed".to_owned());
        }
    };
    parts.headers.remove(COMPRESSION_HEADER);
    parts
        .headers
        .insert(header::CONTENT_LENGTH, HeaderValue::from(body.len()));
    next.run(Request::from_parts(parts, Body::from(body))).await
}

/// A body of known length (hyper's, from `Content-Length`) reserves its bytes
/// and is read before it waits for a permit. One of unknown length cannot
/// reserve, so it is read only under a permit, which bounds those instead.
async fn decompress_zstd(body: Body, limits: &Limits) -> Result<Vec<u8>, String> {
    let length = body.size_hint().exact();
    if length == Some(0) {
        return Ok(Vec::new());
    }
    let (reserved, permit) = match length {
        Some(length) => (Some(limits.reserve(length).await?), None),
        None => (None, Some(limits.decompression().await?)),
    };
    let compressed = axum::body::to_bytes(body, usize::MAX)
        .await
        .map_err(|error| format!("reading the body: {error}"))?;
    if compressed.is_empty() {
        return Ok(Vec::new());
    }
    let permit = match permit {
        Some(permit) => permit,
        None => limits.decompression().await?,
    };
    tokio::task::spawn_blocking(move || {
        let _held = (reserved, permit);
        decode_zstd(&compressed)
    })
    .await
    .map_err(|error| error.to_string())?
    .map_err(|error| error.to_string())
}

/// Sized to the frame's declared content size, as far as the compressed bytes
/// could expand, and trimmed after: the body holds no spare capacity.
///
/// Like Python's `stream_reader(...).read()`, a frame cut short yields what
/// decoded before the cut: a frame missing only its checksum is the whole
/// body, and a cut payload is left for the route to reject.
fn decode_zstd(compressed: &[u8]) -> std::io::Result<Vec<u8>> {
    let declared = zstd::zstd_safe::get_frame_content_size(compressed)
        .ok()
        .flatten()
        .unwrap_or(0);
    let bound = (compressed.len() as u64).saturating_mul(MAX_ZSTD_EXPANSION);
    let mut body = Vec::with_capacity(usize::try_from(declared.min(bound)).unwrap_or(0));
    // The decoder's only `UnexpectedEof` is the input ending mid-frame, and
    // `read_to_end` keeps the bytes decoded before it.
    if let Err(error) = zstd::stream::read::Decoder::with_buffer(compressed)?.read_to_end(&mut body)
        && error.kind() != std::io::ErrorKind::UnexpectedEof
    {
        return Err(error);
    }
    body.shrink_to_fit();
    Ok(body)
}

fn bad_request(message: String) -> Response {
    let mut response = Response::new(Body::from(message));
    *response.status_mut() = StatusCode::BAD_REQUEST;
    response
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::Infallible;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};
    use std::time::Duration;

    use axum::routing::post;
    use bytes::Bytes;
    use http_body::{Frame, SizeHint};
    use tower::ServiceExt;

    use super::*;

    const GENERATE: &[u8] =
        br#"{"input_ids":[1,2,3],"stream":false,"sampling_params":{"max_new_tokens":8}}"#;
    /// Python `zstandard` 0.25: `ZstdCompressor(write_checksum=True).compress(GENERATE)`.
    const GENERATE_WITH_CHECKSUM: &[u8] = b"\
        \x28\xb5\x2f\xfd\x24\x4b\x59\x02\x00\x7b\x22\x69\x6e\x70\x75\x74\x5f\x69\x64\x73\x22\
        \x3a\x5b\x31\x2c\x32\x2c\x33\x5d\x2c\x22\x73\x74\x72\x65\x61\x6d\x22\x3a\x66\x61\x6c\
        \x73\x65\x2c\x22\x73\x61\x6d\x70\x6c\x69\x6e\x67\x5f\x70\x61\x72\x61\x6d\x73\x22\x3a\
        \x7b\x22\x6d\x61\x78\x5f\x6e\x65\x77\x5f\x74\x6f\x6b\x65\x6e\x73\x22\x3a\x38\x7d\x7d\
        \xe2\x70\x1d\xda";
    /// `ZstdCompressor(write_checksum=False).compress(GENERATE)`.
    const GENERATE_WITHOUT_CHECKSUM: &[u8] = b"\
        \x28\xb5\x2f\xfd\x20\x4b\x59\x02\x00\x7b\x22\x69\x6e\x70\x75\x74\x5f\x69\x64\x73\x22\
        \x3a\x5b\x31\x2c\x32\x2c\x33\x5d\x2c\x22\x73\x74\x72\x65\x61\x6d\x22\x3a\x66\x61\x6c\
        \x73\x65\x2c\x22\x73\x61\x6d\x70\x6c\x69\x6e\x67\x5f\x70\x61\x72\x61\x6d\x73\x22\x3a\
        \x7b\x22\x6d\x61\x78\x5f\x6e\x65\x77\x5f\x74\x6f\x6b\x65\x6e\x73\x22\x3a\x38\x7d\x7d";
    /// `ZstdCompressor(write_checksum=True).compress(b'{"text":"' + b"x" * (3 << 17) + b'"}')`:
    /// three 128 KiB blocks, then the last block holding the 11 bytes left.
    const THREE_BLOCKS_WITH_CHECKSUM: &[u8] = b"\
        \x28\xb5\x2f\xfd\xa4\x0b\x00\x06\x00\x94\x00\x00\x50\x7b\x22\x74\x65\x78\x74\x22\x3a\
        \x22\x78\x01\x00\xf3\xff\x39\x40\x02\x02\x00\x10\x78\x02\x00\x10\x78\x59\x00\x00\x78\
        \x78\x78\x78\x78\x78\x78\x78\x78\x22\x7d\x61\x7d\xfe\x93";

    /// Each Python-produced frame decodes to what Python's middleware hands
    /// the route (`stream_reader(...).read()` on the same bytes), or fails with
    /// its `decompress failed` 400. A frame cut short keeps what decoded before
    /// the cut: without its checksum it is the whole body, and a cut payload is
    /// left for the route to reject as JSON.
    #[tokio::test]
    async fn cut_frames_decode_like_python() {
        let cut = |frame: &[u8], bytes: usize| frame[..frame.len() - bytes].to_vec();
        let three_blocks = [&br#"{"text":""#[..], &[b'x'; 3 << 17], br#""}"#].concat();
        let corrupt_checksum = [
            &GENERATE_WITH_CHECKSUM[..GENERATE_WITH_CHECKSUM.len() - 4],
            &[0; 4],
        ]
        .concat();
        // The frame, and the body Python's route sees (`None`: a 400).
        type Case<'a> = (&'a str, Vec<u8>, Option<&'a [u8]>);
        let cases: [Case; 14] = [
            ("checksum", GENERATE_WITH_CHECKSUM.to_vec(), Some(GENERATE)),
            (
                "no checksum",
                GENERATE_WITHOUT_CHECKSUM.to_vec(),
                Some(GENERATE),
            ),
            (
                "checksum cut",
                cut(GENERATE_WITH_CHECKSUM, 4),
                Some(GENERATE),
            ),
            (
                "half the checksum cut",
                cut(GENERATE_WITH_CHECKSUM, 2),
                Some(GENERATE),
            ),
            (
                "content cut",
                GENERATE_WITH_CHECKSUM[..GENERATE_WITH_CHECKSUM.len() / 2].to_vec(),
                Some(&GENERATE[..35]),
            ),
            (
                "header only",
                GENERATE_WITH_CHECKSUM[..6].to_vec(),
                Some(b""),
            ),
            (
                "second frame's checksum cut",
                [GENERATE_WITH_CHECKSUM, &cut(GENERATE_WITH_CHECKSUM, 4)].concat(),
                Some(&[GENERATE, GENERATE].concat()),
            ),
            (
                "second frame's header cut",
                [GENERATE_WITH_CHECKSUM, &GENERATE_WITH_CHECKSUM[..4]].concat(),
                Some(GENERATE),
            ),
            ("corrupt checksum", corrupt_checksum, None),
            (
                "trailing junk",
                [GENERATE_WITH_CHECKSUM, b"junk"].concat(),
                None,
            ),
            (
                "checksum cut, then junk",
                [&cut(GENERATE_WITH_CHECKSUM, 4), &b"junk"[..]].concat(),
                None,
            ),
            (
                "three blocks, checksum cut",
                cut(THREE_BLOCKS_WITH_CHECKSUM, 4),
                Some(&three_blocks),
            ),
            (
                "three blocks, last block cut",
                // The last block is stored raw: 5 of its 11 bytes arrive.
                cut(THREE_BLOCKS_WITH_CHECKSUM, 10),
                Some(&three_blocks[..(3 << 17) + 5]),
            ),
            (
                "three blocks, third block cut",
                cut(THREE_BLOCKS_WITH_CHECKSUM, 20),
                Some(&three_blocks[..2 << 17]),
            ),
        ];
        let app = echo_app(Limits::new(1, BUFFERED_BUDGET_KIB));
        for (case, frame, python) in cases {
            let request = Request::post("/echo")
                .header(COMPRESSION_HEADER, "zstd")
                .body(Body::from(frame))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let (expected_status, expected_body) = match python {
                Some(body) => (StatusCode::OK, body),
                None => (StatusCode::BAD_REQUEST, &b"decompress failed"[..]),
            };
            assert_eq!(status, expected_status, "{case}");
            assert!(body == expected_body, "{case}: {} bytes", body.len());
        }
    }

    fn echo_app(limits: Limits) -> Router {
        with_decompression(
            Router::new().route("/echo", post(|body: Bytes| async move { body })),
            limits,
        )
    }

    /// Hands out its chunks only when polled, like a socket, counting the bytes
    /// read; `exact` declares the length, as hyper does from `Content-Length`.
    struct LazyBody {
        chunks: VecDeque<Bytes>,
        remaining: u64,
        exact: bool,
        read: Arc<AtomicUsize>,
    }

    impl http_body::Body for LazyBody {
        type Data = Bytes;
        type Error = Infallible;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
            let chunk = self.chunks.pop_front();
            Poll::Ready(chunk.map(|chunk| {
                self.remaining -= chunk.len() as u64;
                self.read.fetch_add(chunk.len(), Ordering::Relaxed);
                Ok(Frame::data(chunk))
            }))
        }

        fn size_hint(&self) -> SizeHint {
            if self.exact {
                SizeHint::with_exact(self.remaining)
            } else {
                SizeHint::default()
            }
        }
    }

    /// A lazily read zstd request, and the count of its body bytes read.
    fn lazy_request(compressed: &[u8], exact: bool) -> (Request, Arc<AtomicUsize>) {
        let read = Arc::new(AtomicUsize::new(0));
        let body = LazyBody {
            chunks: compressed.chunks(256).map(Bytes::copy_from_slice).collect(),
            remaining: compressed.len() as u64,
            exact,
            read: read.clone(),
        };
        let request = Request::post("/echo")
            .header(COMPRESSION_HEADER, "zstd")
            .body(Body::new(body))
            .unwrap();
        (request, read)
    }

    async fn still_waiting(
        response: &mut tokio::task::JoinHandle<Result<Response, Infallible>>,
    ) -> bool {
        tokio::time::timeout(Duration::from_millis(200), response)
            .await
            .is_err()
    }

    async fn echoed(response: tokio::task::JoinHandle<Result<Response, Infallible>>) -> Bytes {
        let response = response.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
    }

    /// A decompression past the limit waits for a permit instead of failing;
    /// a body of unknown length is not read until it has one. A body declaring
    /// its content size decodes without spare capacity.
    #[tokio::test]
    async fn decompressions_past_the_limit_wait() {
        assert!((5..=32).contains(&python_executor_workers()));
        let payload = br#"{"text":"hello"}"#;
        let compressed = zstd::bulk::compress(payload, 3).unwrap();
        let decoded = decode_zstd(&compressed).unwrap();
        assert_eq!(
            (decoded.as_slice(), decoded.capacity()),
            (&payload[..], payload.len())
        );

        let limits = Limits::new(1, BUFFERED_BUDGET_KIB);
        let app = echo_app(limits.clone());
        let held = limits.decompression().await.unwrap();
        let (request, read) = lazy_request(&compressed, false);
        let mut response = tokio::spawn(app.oneshot(request));
        assert!(
            still_waiting(&mut response).await,
            "decompressed without a permit"
        );
        assert_eq!(
            read.load(Ordering::Relaxed),
            0,
            "read the body while queued"
        );
        drop(held);
        assert_eq!(&echoed(response).await[..], payload);
        assert_eq!(read.load(Ordering::Relaxed), compressed.len());
    }

    /// A body of known length is read while it waits for a permit, but only
    /// within the byte budget; one larger than the whole budget takes all of it.
    #[tokio::test]
    async fn known_lengths_wait_on_the_byte_budget() {
        // About 1.5 KiB that zstd cannot shrink, against a 1 KiB budget.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let large: Vec<u8> = (0..1500)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        let small = br#"{"text":"hello"}"#;
        let large_frame = zstd::bulk::compress(&large, 3).unwrap();
        let small_frame = zstd::bulk::compress(small, 3).unwrap();
        assert!(large_frame.len() > 1024);

        let limits = Limits::new(1, 1);
        let app = echo_app(limits.clone());
        let held = limits.decompression().await.unwrap();
        let (request, large_read) = lazy_request(&large_frame, true);
        let mut large_response = tokio::spawn(app.clone().oneshot(request));
        assert!(still_waiting(&mut large_response).await);
        assert_eq!(large_read.load(Ordering::Relaxed), large_frame.len());

        let (request, small_read) = lazy_request(&small_frame, true);
        let mut small_response = tokio::spawn(app.oneshot(request));
        assert!(still_waiting(&mut small_response).await);
        assert_eq!(
            small_read.load(Ordering::Relaxed),
            0,
            "read past the budget"
        );

        drop(held);
        assert_eq!(&echoed(large_response).await[..], &large[..]);
        assert_eq!(&echoed(small_response).await[..], small);
    }
}
