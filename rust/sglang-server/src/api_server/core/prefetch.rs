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

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use sglang_mm::common::fetch::{ByteBudget, fetch_bytes_budgeted, fetch_local_file_budgeted};
use sglang_mm::driver::{MAX_ITEMS_PER_REQUEST, MAX_REQUEST_BYTES};
use tokio::sync::Semaphore;

use crate::message::request::{GenerateRequest, MmData};
use crate::multi_modality::payload::io_sources;

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
        let modalities = [
            ("image", &mm.image_data),
            ("video", &mm.video_data),
            ("audio", &mm.audio_data),
        ];
        let items = modalities
            .iter()
            .map(|(_, items)| items.len())
            .sum::<usize>();
        if items > MAX_ITEMS_PER_REQUEST {
            return Err(format!(
                "multimodal request exceeds {MAX_ITEMS_PER_REQUEST} media items"
            ));
        }
        for (modality, items) in modalities {
            let count = items.len();
            if let Some(limit) = modality_limits.get(modality)
                && count > *limit
            {
                let display = modality[..1].to_uppercase() + &modality[1..];
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
        .map(|sources| fetch_ordered(sources, MAX_REQUEST_BYTES, &PERMITS));
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
async fn fetch_ordered(
    sources: Vec<String>,
    total_bytes: u64,
    permits: &'static Semaphore,
) -> Result<Vec<Bytes>, String> {
    let budget = Arc::new(ByteBudget::new(total_bytes));
    futures::future::try_join_all(sources.into_iter().map(|src| {
        let budget = Arc::clone(&budget);
        async move {
            let permit = permits.acquire().await.expect("semaphore never closed");
            // Blocking I/O: parks a lazily-spawned blocking-pool thread, never
            // an API worker. Those threads are pinned round-robin over the api
            // core set (see `on_thread_start` in `runtime::start`) — off the
            // CPU-bound stages, and mostly I/O-parked, so sharing is fine.
            tokio::task::spawn_blocking(move || {
                // A cancelled waiter cannot stop blocking I/O. Keep its slot
                // occupied until the fetch itself returns or unwinds.
                let _permit = permit;
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
            mm_request(vec![
                src(format!("http://{addr}/a.png")),
                src("data:image/png;base64,x"),
                src(format!("http://{addr}/b.png")),
                src(path.display().to_string()),
            ]),
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
        let mut requests = vec![mm_request(vec![src("http://127.0.0.1:1/nope.png")])];
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
        let sources: Vec<MmItem> = (0..=MAX_ITEMS_PER_REQUEST)
            .map(|i| src(format!("/definitely/not/here-{i}.png")))
            .collect();
        let mut requests = vec![mm_request(sources)];
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
                image_data: vec![
                    src("/definitely/not/here-0.png"),
                    src("/definitely/not/here-1.png"),
                ],
                video_data: vec![src("/definitely/not/here.mp4")],
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
        let err = fetch_ordered(sources, 6144, &PERMITS).await.err().unwrap();
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
        let fetched = fetch_ordered(sources, MAX_REQUEST_BYTES, &PERMITS)
            .await
            .unwrap();
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
        let fetched = fetch_ordered(sources.clone(), 11, &PERMITS).await.unwrap();
        let error = fetch_ordered(sources, 10, &PERMITS).await.err().unwrap();
        std::fs::remove_dir_all(base).ok();
        assert_eq!(fetched.len(), 2);
        assert!(error.contains("request media byte budget"), "{error}");
    }

    struct GatedFetch {
        url: String,
        started: tokio::sync::oneshot::Receiver<()>,
        release: std::sync::mpsc::Sender<()>,
        server: std::thread::JoinHandle<()>,
    }

    fn gated_fetch(status: u16) -> GatedFetch {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (started_tx, started) = tokio::sync::oneshot::channel();
        let (release, release_rx) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            use std::io::{BufRead, Write};

            let (stream, _) = listener.accept().unwrap();
            let mut reader = std::io::BufReader::new(stream);
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 2 {
                line.clear();
            }
            started_tx.send(()).unwrap();
            // Bound cleanup if the assertion path fails; the test never sleeps
            // to infer whether a download has started or finished.
            let _ = release_rx.recv_timeout(std::time::Duration::from_secs(5));
            let _ = write!(
                reader.into_inner(),
                "HTTP/1.1 {status} Test\r\nContent-Length: 2\r\n\r\nok"
            );
        });
        GatedFetch {
            url: format!("http://{address}/image"),
            started,
            release,
            server,
        }
    }

    async fn check_cancelled_fetch_permit(permits: &'static Semaphore, reject_sibling: bool) {
        use std::time::Duration;
        use tokio::time::timeout;

        let slow = gated_fetch(200);
        let rejected = reject_sibling.then(|| gated_fetch(503));
        let mut sources = vec![slow.url];
        if let Some(rejected) = &rejected {
            sources.push(rejected.url.clone());
        }
        let fetch = tokio::spawn(fetch_ordered(sources, MAX_REQUEST_BYTES, permits));
        timeout(Duration::from_secs(2), slow.started)
            .await
            .expect("slow download did not start")
            .unwrap();

        if let Some(rejected) = rejected {
            timeout(Duration::from_secs(2), rejected.started)
                .await
                .expect("failing download did not start")
                .unwrap();
            rejected.release.send(()).unwrap();
            let error = timeout(Duration::from_secs(2), fetch)
                .await
                .expect("failed batch did not return")
                .unwrap()
                .unwrap_err();
            assert!(error.contains("503"), "{error}");
            rejected.server.join().unwrap();
        } else {
            fetch.abort();
            assert!(fetch.await.unwrap_err().is_cancelled());
        }

        let available_while_downloading = permits.available_permits();
        // The cancelled download occupies one slot, leaving the second usable
        // by a live request before the slow server has sent any response.
        let live_address = serve(vec![b"live".to_vec()]);
        let live = timeout(
            Duration::from_secs(2),
            fetch_ordered(
                vec![format!("http://{live_address}/live")],
                MAX_REQUEST_BYTES,
                permits,
            ),
        )
        .await
        .expect("live request could not use the remaining fetch slot")
        .unwrap();
        assert_eq!(live[0].as_ref(), b"live");
        slow.release.send(()).unwrap();
        let _all_permits = timeout(Duration::from_secs(2), permits.acquire_many(2))
            .await
            .expect("finished download did not release its permit")
            .unwrap();
        slow.server.join().unwrap();
        assert_eq!(
            available_while_downloading, 1,
            "cancelled caller released the permit before blocking I/O finished"
        );
    }

    #[tokio::test]
    async fn cancelled_prefetch_keeps_ongoing_download_permit() {
        static TEST_PERMITS: Semaphore = Semaphore::const_new(2);
        check_cancelled_fetch_permit(&TEST_PERMITS, false).await;
    }

    #[tokio::test]
    async fn rejected_batch_keeps_ongoing_download_permit() {
        static TEST_PERMITS: Semaphore = Semaphore::const_new(2);
        check_cancelled_fetch_permit(&TEST_PERMITS, true).await;
    }
}
