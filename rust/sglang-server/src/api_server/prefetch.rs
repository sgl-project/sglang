//! Resolve I/O-backed media sources on the API runtime, before MM dispatch.
//!
//! The MM worker pool is fixed and core-pinned CPU capacity: a slow image
//! host — or a file on a hanging network mount — must never occupy it, and
//! images within a request must download concurrently (not
//! `n * REQUEST_TIMEOUT`). URLs and file paths resolve here — bounded
//! globally, via `sglang-mm`'s `fetch_bytes` so proxy/timeout/cap semantics
//! have one owner — and ride out-of-band as [`MmData::prefetched`];
//! [`crate::message::mm_payload::parse`] swaps them back in.

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::message::mm_payload::io_sources;
use crate::message::{GenerateRequest, MmData};

/// Global bound on concurrent media fetches across all in-flight requests;
/// excess acquisitions queue on the semaphore without holding a thread.
static PERMITS: Semaphore = Semaphore::const_new(32);

/// Fill [`MmData::prefetched`] for every request; every fetch across the
/// batch runs concurrently. Any failure rejects the call (fetch errors are
/// per-request 400s, as on the Python path).
pub async fn prefetch_all(requests: &mut [GenerateRequest]) -> Result<(), String> {
    let sources_of = |mm: &Option<Box<MmData>>| {
        mm.as_deref()
            .and_then(|m| m.image_data.as_ref())
            .map(io_sources)
            .unwrap_or_default()
    };
    let fetches = requests.iter().map(|r| fetch_ordered(sources_of(&r.mm)));
    let fetched = futures::future::try_join_all(fetches).await?;
    for (req, bytes) in requests.iter_mut().zip(fetched) {
        if !bytes.is_empty() {
            req.mm.as_mut().expect("sources came from mm").prefetched = bytes;
        }
    }
    Ok(())
}

/// Resolve every source concurrently (globally bounded), preserving order.
async fn fetch_ordered(sources: Vec<String>) -> Result<Vec<Bytes>, String> {
    futures::future::try_join_all(sources.into_iter().map(|src| async move {
        let _permit = PERMITS.acquire().await.expect("semaphore never closed");
        // `fetch_bytes` is blocking I/O; tokio's blocking pool threads are
        // unpinned and lazily spawned, so they never contend with CPU stages.
        tokio::task::spawn_blocking(move || sglang_mm::common::fetch::fetch_bytes(&src))
            .await
            .map_err(|e| format!("media prefetch: {e}"))?
            .map(Bytes::from)
    }))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmpv::Value;

    fn serve(bodies: Vec<Vec<u8>>) -> std::net::SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for body in bodies {
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
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        addr
    }

    fn mm_request(image_data: Value) -> GenerateRequest {
        GenerateRequest {
            mm: Some(Box::new(MmData {
                image_data: Some(image_data),
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    /// URLs and file paths resolve concurrently and land in `prefetched` in
    /// source order; CPU-only sources and mm-free requests are untouched.
    #[tokio::test]
    async fn resolves_io_sources() {
        let addr = serve(vec![b"one".to_vec(), b"two".to_vec()]);
        let path = std::env::temp_dir().join(format!("sglang-prefetch-{}", std::process::id()));
        std::fs::write(&path, b"zzz").unwrap();
        let mut requests = vec![
            mm_request(Value::Array(vec![
                Value::from(format!("http://{addr}/a.png")),
                Value::from("data:image/png;base64,x"),
                Value::from(format!("http://{addr}/b.png")),
                Value::from(path.display().to_string()),
            ])),
            GenerateRequest::default(),
        ];
        prefetch_all(&mut requests).await.unwrap();
        std::fs::remove_file(&path).ok();
        let fetched = &requests[0].mm.as_ref().unwrap().prefetched;
        // The one-shot server answers in accept order, so contents may swap
        // between the two URLs; all three bodies must arrive.
        let mut got: Vec<&[u8]> = fetched.iter().map(|b| b.as_ref()).collect();
        got.sort();
        assert_eq!(got, vec![b"one".as_ref(), b"two".as_ref(), b"zzz".as_ref()]);
        assert!(requests[1].mm.is_none());
    }

    #[tokio::test]
    async fn failed_download_rejects() {
        let mut requests = vec![mm_request(Value::from("http://127.0.0.1:1/nope.png"))];
        let err = prefetch_all(&mut requests).await.err().unwrap();
        assert!(err.contains("media fetch"), "{err}");
    }
}
