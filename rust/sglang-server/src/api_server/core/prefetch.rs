//! Resolve I/O-backed media sources on the async API runtime, before MM dispatch.
//!
//! The MM worker pool is fixed, core-pinned CPU capacity: a slow image host — or
//! a file on a hanging network mount — must never occupy it, and a request's
//! images must download concurrently, not in `n * REQUEST_TIMEOUT`. Remote and
//! inline sources resolve through `sglang-mm`'s `fetch_bytes_budgeted` (one
//! owner for proxy/timeout/cap semantics); trusted local files skip the remote
//! per-source cap but share the whole-request budget. Resolved bytes ride out-of-band as
//! [`crate::message::request::MmData::prefetched`], which
//! [`crate::multi_modality::payload::to_mm_input`] swaps back in.

use std::sync::{Arc, LazyLock};

use bytes::Bytes;
use sglang_mm::common::fetch::{ByteBudget, fetch_bytes_budgeted, fetch_local_file_budgeted};
use sglang_mm::driver::MAX_REQUEST_BYTES;
use tokio::sync::Semaphore;

use crate::message::request::{GenerateRequest, MmData};
use crate::multi_modality::payload::io_sources;

/// Global bound on concurrent media fetches across all in-flight requests;
/// excess acquisitions queue on the semaphore without holding a thread.
static PERMITS: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(32)));

/// Fill [`MmData::prefetched`] for every request, all fetches across the batch
/// concurrent. Any failure rejects the call (a 400, as on the Python path).
///
/// `modality_limits` is `--limit-mm-data-per-request`, checked before a single
/// byte is fetched, in its configured order and with Python's message
/// (`TokenizerManager._validate_mm_limits`). Unset means no per-modality limit,
/// as in Python. Python has no fixed item cap, and neither does this.
///
/// [`MAX_REQUEST_BYTES`] is enforced while fetching, here rather than in
/// `sglang_mm::driver::process`, where every source would already be resident.
/// It bounds buffered media bytes regardless of the item count. The driver
/// keeps its own byte check as the backstop for callers without a prefetch
/// layer.
pub async fn prefetch_all(
    requests: &mut [GenerateRequest],
    modality_limits: &[(String, usize)],
) -> Result<(), String> {
    let plan = |mm: &Option<Box<MmData>>| -> Result<Vec<String>, String> {
        let Some(mm) = mm.as_deref() else {
            return Ok(Vec::new());
        };
        let modalities = [
            ("image", &mm.image_data),
            ("video", &mm.video_data),
            ("audio", &mm.audio_data),
        ];
        for (modality, limit) in modality_limits {
            let Some((name, items)) = modalities.iter().find(|(name, _)| name == modality) else {
                continue;
            };
            let count = items.len();
            if count > *limit {
                let display = name[..1].to_uppercase() + &name[1..];
                return Err(format!(
                    "{display} count {count} exceeds limit {limit} per request."
                ));
            }
        }
        Ok(modalities
            .iter()
            .flat_map(|(_, items)| io_sources(items))
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
            // Blocking I/O: parks a lazily-spawned blocking-pool thread, never
            // an API worker. Those threads are pinned round-robin over the api
            // core set (see `on_thread_start` in `runtime::start`) — off the
            // CPU-bound stages, and mostly I/O-parked, so sharing is fine.
            run_blocking_bounded(Arc::clone(&PERMITS), move || {
                if src.starts_with('/') || src.starts_with("file://") {
                    fetch_local_file_budgeted(&src, &budget)
                } else {
                    fetch_bytes_budgeted(&src, &budget)
                }
            })
            .await?
            .map(Bytes::from)
        }
    }))
    .await
}

async fn run_blocking_bounded<F, T>(permits: Arc<Semaphore>, work: F) -> Result<T, String>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let permit = permits
        .acquire_owned()
        .await
        .expect("semaphore never closed");
    tokio::task::spawn_blocking(move || {
        // Keep the permit with the blocking operation: dropping the async
        // caller cannot make uncancellable work disappear from the bound.
        let _permit = permit;
        work()
    })
    .await
    .map_err(|e| format!("media prefetch: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::multimodal::MmItem;

    fn src(s: impl Into<String>) -> MmItem {
        MmItem::Source(s.into())
    }

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

    fn mm_request(image_data: Vec<MmItem>) -> GenerateRequest {
        GenerateRequest {
            mm: Some(Box::new(MmData {
                image_data,
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
                image_data: vec![src(paths[0].display().to_string())],
                video_data: vec![src(paths[1].display().to_string())],
                audio_data: vec![src(paths[2].display().to_string())],
                ..Default::default()
            })),
            ..Default::default()
        }];
        prefetch_all(&mut requests, &[]).await.unwrap();
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
            mm_request(vec![
                src(format!("http://{addr}/a.png")),
                src("data:image/png;base64,x"),
                src(format!("http://{addr}/b.png")),
                src(path.display().to_string()),
            ]),
            GenerateRequest::default(),
        ];
        prefetch_all(&mut requests, &[]).await.unwrap();
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
        let mut requests = vec![mm_request(vec![src("http://127.0.0.1:1/nope.png")])];
        let err = prefetch_all(&mut requests, &[]).await.err().unwrap();
        assert!(err.contains("media fetch"), "{err}");
    }

    /// Without `--limit-mm-data-per-request`, a request with far more items
    /// than the old fixed cap (64) is fetched in full, as Python serves it.
    #[tokio::test]
    async fn items_beyond_the_old_cap_are_fetched_without_a_limit() {
        let n = 200;
        let path =
            std::env::temp_dir().join(format!("sglang-prefetch-many-{}", std::process::id()));
        std::fs::write(&path, b"img").unwrap();
        let mut requests = vec![mm_request(
            (0..n).map(|_| src(path.display().to_string())).collect(),
        )];
        let fetched = prefetch_all(&mut requests, &[]).await;
        std::fs::remove_file(&path).ok();
        fetched.unwrap();
        assert_eq!(requests[0].mm.as_ref().unwrap().prefetched.len(), n);
    }

    /// `--limit-mm-data-per-request` matches Python `_validate_mm_limits`: a
    /// count at the limit passes, the first over-limit modality in configured
    /// order is reported with Python's message, and nothing is fetched.
    #[tokio::test]
    async fn per_modality_limits_match_python_before_fetching() {
        let request = || GenerateRequest {
            mm: Some(Box::new(MmData {
                image_data: vec![
                    src("/definitely/not/here-0.png"),
                    src("/definitely/not/here-1.png"),
                ],
                video_data: vec![
                    src("/definitely/not/here-0.mp4"),
                    src("/definitely/not/here-1.mp4"),
                ],
                ..Default::default()
            })),
            ..Default::default()
        };
        let owned = |pairs: &[(&str, usize)]| -> Vec<(String, usize)> {
            pairs.iter().map(|(m, n)| ((*m).to_owned(), *n)).collect()
        };
        for (limits, expected) in [
            (
                owned(&[("video", 1), ("image", 1)]),
                "Video count 2 exceeds limit 1 per request.",
            ),
            (
                owned(&[("audio", 0), ("image", 1), ("video", 1)]),
                "Image count 2 exceeds limit 1 per request.",
            ),
        ] {
            let mut requests = vec![request()];
            let err = prefetch_all(&mut requests, &limits).await.err().unwrap();
            assert_eq!(err, expected);
            assert!(requests[0].mm.as_ref().unwrap().prefetched.is_empty());
        }

        // At the limit the request passes the count check and reaches fetching.
        let mut requests = vec![request()];
        let err = prefetch_all(&mut requests, &owned(&[("image", 2)]))
            .await
            .err()
            .unwrap();
        assert!(err.contains("media fetch"), "{err}");
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

    #[tokio::test]
    async fn cancelled_caller_does_not_release_blocking_permit_early() {
        let permits = Arc::new(Semaphore::new(1));
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let task_permits = Arc::clone(&permits);
        let task = tokio::spawn(async move {
            run_blocking_bounded(task_permits, move || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
            .await
        });

        tokio::task::spawn_blocking(move || started_rx.recv().unwrap())
            .await
            .unwrap();
        task.abort();
        tokio::task::yield_now().await;
        assert_eq!(
            permits.available_permits(),
            0,
            "the blocking operation must retain its permit after caller cancellation"
        );

        release_tx.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while permits.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(permits.available_permits(), 1);
    }
}
