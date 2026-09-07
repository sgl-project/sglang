//! Resolve I/O-backed media sources on the API runtime, before MM dispatch.
//!
//! The MM worker pool is fixed, core-pinned CPU capacity: a slow image host — or
//! a file on a hanging network mount — must never occupy it, and a request's
//! images must download concurrently, not in `n * REQUEST_TIMEOUT`. Remote and
//! inline sources resolve through `sglang-mm`'s `fetch_bytes_budgeted` (one
//! owner for proxy/timeout/cap semantics); trusted local files skip the remote
//! per-source cap but share the whole-request budget. Resolved bytes ride out-of-band as
//! [`crate::message::request::MmData::prefetched`], which
//! [`crate::multi_modality::payload::to_mm_input`] swaps back in.

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use sglang_mm::common::fetch::{ByteBudget, fetch_bytes_budgeted, fetch_local_file_budgeted};
use sglang_mm::driver::{MAX_ITEMS_PER_REQUEST, MAX_REQUEST_BYTES};
use tokio::sync::Semaphore;

use crate::message::request::{GenerateRequest, MmData};
use crate::multi_modality::payload::{io_sources, item_count};

/// Global bound on concurrent media fetches across all in-flight requests;
/// excess acquisitions queue on the semaphore without holding a thread.
static PERMITS: Semaphore = Semaphore::const_new(32);

/// Fill [`MmData::prefetched`] for every request, all fetches across the batch
/// concurrent. Any failure rejects the call (a 400, as on the Python path).
///
/// The driver's budgets ([`MAX_ITEMS_PER_REQUEST`], [`MAX_REQUEST_BYTES`]) are
/// enforced *here* rather than in `sglang_mm::driver::process`, where 64 sources
/// of 64 MiB would already be resident. The driver keeps its own checks as the
/// backstop for callers without a prefetch layer.
pub async fn prefetch_all(
    requests: &mut [GenerateRequest],
    modality_limits: &BTreeMap<String, usize>,
) -> Result<(), String> {
    // The item budget rejects before a single byte is fetched.
    let plan = |mm: &Option<Box<MmData>>| -> Result<Vec<String>, String> {
        let Some(mm) = mm.as_deref() else {
            return Ok(Vec::new());
        };
        let values = [
            ("image", mm.image_data.as_ref()),
            ("video", mm.video_data.as_ref()),
            ("audio", mm.audio_data.as_ref()),
        ];
        let items = values
            .iter()
            .filter_map(|(_, value)| *value)
            .map(item_count)
            .sum::<usize>();
        if items > MAX_ITEMS_PER_REQUEST {
            return Err(format!(
                "multimodal request exceeds {MAX_ITEMS_PER_REQUEST} media items"
            ));
        }
        for (modality, value) in values {
            let count = value.map(item_count).unwrap_or_default();
            if let Some(limit) = modality_limits.get(modality)
                && count > *limit
            {
                let display = modality[..1].to_uppercase() + &modality[1..];
                return Err(format!(
                    "{display} count {count} exceeds limit {limit} per request."
                ));
            }
        }
        Ok(values
            .iter()
            .filter_map(|(_, value)| *value)
            .flat_map(io_sources)
            .collect())
    };
    let plans = requests
        .iter()
        .map(|r| plan(&r.mm))
        .collect::<Result<Vec<_>, String>>()?;
    let fetches = plans
        .into_iter()
        .map(|sources| fetch_ordered(sources, MAX_REQUEST_BYTES));
    let fetched = futures::future::try_join_all(fetches).await?;
    for (req, bytes) in requests.iter_mut().zip(fetched) {
        if !bytes.is_empty() {
            req.mm.as_mut().expect("sources came from mm").prefetched = bytes;
        }
    }
    Ok(())
}

/// Resolve one request's sources concurrently (globally bounded), in order.
/// All inputs share `total_bytes`; trusted local files skip only the remote
/// per-source cap, matching Python's URL-only security limit. Overflow rejects
/// before or during I/O and `try_join_all` drops the rest, so queued sources
/// never start.
async fn fetch_ordered(sources: Vec<String>, total_bytes: u64) -> Result<Vec<Bytes>, String> {
    let budget = Arc::new(ByteBudget::new(total_bytes));
    futures::future::try_join_all(sources.into_iter().map(|src| {
        let budget = Arc::clone(&budget);
        async move {
            let _permit = PERMITS.acquire().await.expect("semaphore never closed");
            // Blocking I/O: parks a lazily-spawned blocking-pool thread, never
            // an API worker. Those threads are pinned round-robin over the api
            // core set (see `on_thread_start` in `runtime::start`) — off the
            // CPU-bound stages, and mostly I/O-parked, so sharing is fine.
            tokio::task::spawn_blocking(move || {
                if src.starts_with('/') || src.starts_with("file://") {
                    fetch_local_file_budgeted(&src, &budget)
                } else {
                    fetch_bytes_budgeted(&src, &budget)
                }
            })
            .await
            .map_err(|e| format!("media prefetch: {e}"))?
            .map(Bytes::from)
        }
    }))
    .await
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;

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

    #[tokio::test]
    async fn mixed_modalities_preserve_image_video_audio_order() {
        let base = std::env::temp_dir().join(format!("sglang-prefetch-mm-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let paths = [base.join("image"), base.join("video"), base.join("audio")];
        for (path, body) in
            paths
                .iter()
                .zip([b"image".as_ref(), b"video".as_ref(), b"audio".as_ref()])
        {
            std::fs::write(path, body).unwrap();
        }
        let mut requests = vec![GenerateRequest {
            mm: Some(Box::new(MmData {
                image_data: Some(Value::from(paths[0].display().to_string())),
                video_data: Some(Value::from(paths[1].display().to_string())),
                audio_data: Some(Value::from(paths[2].display().to_string())),
                ..Default::default()
            })),
            ..Default::default()
        }];
        prefetch_all(&mut requests, &BTreeMap::new()).await.unwrap();
        std::fs::remove_dir_all(base).ok();
        let fetched = &requests[0].mm.as_ref().unwrap().prefetched;
        assert_eq!(
            fetched.iter().map(Bytes::as_ref).collect::<Vec<_>>(),
            vec![b"image".as_ref(), b"video".as_ref(), b"audio".as_ref()]
        );
    }

    /// URLs and file paths resolve concurrently into `prefetched` in source
    /// order; CPU-only sources and mm-free requests are untouched.
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
        prefetch_all(&mut requests, &BTreeMap::new()).await.unwrap();
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
        let err = prefetch_all(&mut requests, &BTreeMap::new())
            .await
            .err()
            .unwrap();
        assert!(err.contains("media fetch"), "{err}");
    }

    /// The item budget rejects before any source is touched: all of these would
    /// fail to fetch, so a fetch error would prove fetching started.
    #[tokio::test]
    async fn item_budget_rejects_before_fetching() {
        let sources: Vec<Value> = (0..=MAX_ITEMS_PER_REQUEST)
            .map(|i| Value::from(format!("/definitely/not/here-{i}.png")))
            .collect();
        let mut requests = vec![mm_request(Value::Array(sources))];
        let err = prefetch_all(&mut requests, &BTreeMap::new())
            .await
            .err()
            .unwrap();
        assert_eq!(
            err,
            format!("multimodal request exceeds {MAX_ITEMS_PER_REQUEST} media items")
        );
        assert!(requests[0].mm.as_ref().unwrap().prefetched.is_empty());
    }

    #[tokio::test]
    async fn per_modality_budget_rejects_before_fetching() {
        let mut requests = vec![GenerateRequest {
            mm: Some(Box::new(MmData {
                image_data: Some(Value::Array(vec![
                    Value::from("/definitely/not/here-0.png"),
                    Value::from("/definitely/not/here-1.png"),
                ])),
                video_data: Some(Value::Array(vec![Value::from("/definitely/not/here.mp4")])),
                ..Default::default()
            })),
            ..Default::default()
        }];
        let limits = BTreeMap::from([("image".to_owned(), 1), ("video".to_owned(), 1)]);
        let err = prefetch_all(&mut requests, &limits).await.err().unwrap();
        assert_eq!(err, "Image count 2 exceeds limit 1 per request.");
        assert!(requests[0].mm.as_ref().unwrap().prefetched.is_empty());
    }

    /// Sources legal alone but collectively over the limit are rejected while
    /// downloading, not once every body is resident.
    #[tokio::test]
    async fn byte_budget_is_shared_across_sources() {
        let addr = serve(vec![vec![b'a'; 4096], vec![b'b'; 4096]]);
        let sources = vec![
            format!("http://{addr}/a.png"),
            format!("http://{addr}/b.png"),
        ];
        // Room for one body, not both.
        let err = fetch_ordered(sources, 6144).await.err().unwrap();
        assert!(err.contains("request media byte budget"), "{err}");
    }

    /// ...and a fitting set still fetches: the budget never over-rejects.
    #[tokio::test]
    async fn byte_budget_admits_a_fitting_request() {
        let addr = serve(vec![vec![b'a'; 4096], vec![b'b'; 4096]]);
        let sources = vec![
            format!("http://{addr}/a.png"),
            format!("http://{addr}/b.png"),
        ];
        let fetched = fetch_ordered(sources, MAX_REQUEST_BYTES).await.unwrap();
        assert_eq!(fetched.iter().map(|b| b.len()).sum::<usize>(), 8192);
    }

    #[tokio::test]
    async fn local_files_share_the_request_budget() {
        let base = std::env::temp_dir().join(format!(
            "sglang-prefetch-local-budget-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let first = base.join("first.mp4");
        let second = base.join("second.mp4");
        std::fs::write(&first, b"first").unwrap();
        std::fs::write(&second, b"second").unwrap();

        let sources = vec![first.display().to_string(), second.display().to_string()];
        let fetched = fetch_ordered(sources.clone(), 11).await.unwrap();
        let error = fetch_ordered(sources, 10).await.err().unwrap();
        std::fs::remove_dir_all(base).ok();
        assert_eq!(fetched.len(), 2);
        assert!(error.contains("request media byte budget"), "{error}");
    }
}
