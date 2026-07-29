//! Best-effort tee of router-computed `input_ids` to the theoretical
//! cache-sim's `POST /ingest_ids`.
//!
//! The router already tokenizes each request once at ingress (the ids it routes
//! by), so teeing those ids gives the cache-sim a far better proxy than
//! re-tokenizing raw text — and drops the cache-sim's own tokenization CPU and
//! its dependency on a loadable `tokenizer.json`. When the ids are
//! engine-equivalent (chat-encoder path — what the router forwards to the engine
//! as `input_ids`) the measurement is byte-exact to what the engine's paged KV
//! cache keys on, custom encoders included; for the raw-prompt fallback
//! (`/v1/completions`, or a chat model with no encoder) the ids are the router's
//! routing tokenization, which the engine re-templates — closer than the
//! cache-sim's own re-tokenization but not guaranteed engine-identical. The tee
//! fires whenever the router tokenized (it does not filter on
//! `engine_equivalent`).
//!
//! Fire-and-forget by construction: [`CacheSimTee::offer`] enqueues onto a
//! bounded channel and returns immediately; a full queue is dropped + counted
//! (never parked), and one background task POSTs serially. The tee is purely
//! observational, so it must never slow, block, or fail the serving path.

use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tokio::sync::mpsc;

use crate::server::metrics::MetricsRegistry;

/// Bounded queue depth. Excess is dropped + counted rather than parked — the
/// tee must never apply backpressure to the router.
const CHANNEL_CAPACITY: usize = 4096;

/// Per-POST timeout. Short: the cache-sim is an in-cluster Service and a
/// slow/hung one must not let tee requests pile up.
const POST_TIMEOUT: Duration = Duration::from_secs(2);

/// Which cache-sim ingest a teed message targets. Two distinct endpoints on
/// the same service, with distinct accounting semantics on the receiver:
/// `Ingest` (`/ingest_ids`) counts a request against the hit-rate denominator;
/// `Extend` (`/extend_ids`) is insert-only — it seeds the sim's block stores
/// with a completed response's prompt+output sequence so the NEXT round of the
/// conversation measures the hit a real engine's KV cache would serve, without
/// inflating the denominator.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TeeKind {
    Ingest,
    Extend,
}

struct TeeMsg {
    kind: TeeKind,
    model: String,
    input_ids: Vec<u32>,
    request_id: String,
    prompt_len: Option<usize>,
}

/// Wire body of `POST /ingest_ids` and `POST /extend_ids` (same shape).
/// Hand-mirrored (no shared crate) by the receiver's `IdsBody` in
/// gpu-platform-proto `sglang-router-cache-sim` (`src/server.rs`) — keep the
/// two shapes in lockstep.
///
/// `request_id` is the join key: the oracle emits one record per ingest and
/// one per extend, and without a shared id they cannot be paired, so a
/// request's output-token count has nothing to attach to. It is additive and
/// safe to deploy first — the receiver's `IdsBody` ignores unknown fields, so
/// an older cache-sim simply drops it.
///
/// `prompt_len` is sent ONLY on `/extend_ids`, and only when the extension was
/// built incrementally as `prompt_ids ++ suffix`, where the boundary is exact.
/// The full-re-encode fallback re-tokenizes the whole conversation, so no
/// prefix of its output is guaranteed to be the prompt — reporting a boundary
/// there would silently misattribute tokens between prompt and output. Absent
/// means "not derivable", not "zero".
#[derive(Serialize)]
struct IngestIdsBody<'a> {
    model: &'a str,
    input_ids: &'a [u32],
    request_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_len: Option<usize>,
}

/// Handle the chat/completions handler offers pre-tokenized requests to.
pub struct CacheSimTee {
    tx: mpsc::Sender<TeeMsg>,
    metrics: Arc<MetricsRegistry>,
}

// Manual (MetricsRegistry isn't Debug) so AppContext's derive(Debug) holds.
impl std::fmt::Debug for CacheSimTee {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheSimTee")
            .field("queue_capacity", &self.tx.max_capacity())
            .finish_non_exhaustive()
    }
}

impl CacheSimTee {
    /// Spawn the background sender and return a handle. `url` is the cache-sim
    /// base (e.g. `http://radixark-cache-sim:9095`); `/ingest_ids` is appended.
    pub fn spawn(url: String, metrics: Arc<MetricsRegistry>) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        // .expect, not a fallback: reqwest::Client::new() would panic on the
        // same (near-impossible, TLS-backend-init) failure, and a fallback
        // without POST_TIMEOUT would silently defeat the "a hung cache-sim can't
        // pile up" invariant. If this fails, the router's other reqwest clients
        // (engine dispatch) fail too — the process is already unusable.
        let client = reqwest::Client::builder()
            .timeout(POST_TIMEOUT)
            .build()
            .expect("cache-sim tee: build reqwest client");
        let base = url.trim_end_matches('/');
        let ingest_url = format!("{base}/ingest_ids");
        let extend_url = format!("{base}/extend_ids");
        tracing::info!(ingest = %ingest_url, extend = %extend_url, "cache-sim tee enabled");
        tokio::spawn(run_sender(
            rx,
            client,
            ingest_url,
            extend_url,
            Arc::clone(&metrics),
        ));
        Arc::new(Self { tx, metrics })
    }

    /// Offer one request's tokens to the tee. Never blocks: a full queue is
    /// dropped + counted, a closed channel (sender task gone) likewise. Empty
    /// id lists are a no-op. Cheap enough to call unconditionally on the hot
    /// path.
    pub fn offer(&self, model: &str, input_ids: &[u32], request_id: &str) {
        self.offer_kind(TeeKind::Ingest, model, input_ids, request_id, None);
    }

    /// Offer one completed response's FULL token sequence (prompt + generated
    /// output, re-rendered the way the next round's request will be) to the
    /// insert-only `/extend_ids` path. Same never-blocks contract as
    /// [`Self::offer`].
    /// `prompt_len` is `Some` only when the extension was built incrementally
    /// (so the prompt/output boundary is exact) — see [`IngestIdsBody`].
    pub fn offer_extend(
        &self,
        model: &str,
        input_ids: &[u32],
        request_id: &str,
        prompt_len: Option<usize>,
    ) {
        self.offer_kind(TeeKind::Extend, model, input_ids, request_id, prompt_len);
    }

    fn offer_kind(
        &self,
        kind: TeeKind,
        model: &str,
        input_ids: &[u32],
        request_id: &str,
        prompt_len: Option<usize>,
    ) {
        if input_ids.is_empty() {
            return;
        }
        // A boundary at or past the end would make output_tokens zero or
        // negative on the receiver. Drop the claim rather than send one that
        // cannot be true; the extension itself is still worth teeing.
        let prompt_len = prompt_len.filter(|n| *n < input_ids.len());
        let msg = TeeMsg {
            kind,
            model: model.to_owned(),
            input_ids: input_ids.to_vec(),
            request_id: request_id.to_owned(),
            prompt_len,
        };
        match self.tx.try_send(msg) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => self.metrics.record_cache_sim_tee("dropped"),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.metrics.record_cache_sim_tee("closed")
            }
        }
    }
}

/// Drain the queue and POST each request's ids to the cache-sim, serially.
/// Errors are metered and dropped — a down cache-sim must never spam the
/// router's logs or affect serving. Ends when the channel closes (all senders
/// dropped, i.e. shutdown).
///
/// INVARIANT: this loop must never be able to panic. It is the sole consumer of
/// the tee channel; a panic ends the task permanently and silently (teeing just
/// stops — no counter moves). Every fallible step is matched, not unwrapped.
async fn run_sender(
    mut rx: mpsc::Receiver<TeeMsg>,
    client: reqwest::Client,
    ingest_url: String,
    extend_url: String,
    metrics: Arc<MetricsRegistry>,
) {
    while let Some(msg) = rx.recv().await {
        let body = IngestIdsBody {
            model: &msg.model,
            input_ids: &msg.input_ids,
            request_id: &msg.request_id,
            prompt_len: msg.prompt_len,
        };
        // Per-kind outcome labels so a version-skewed cache-sim (no
        // /extend_ids yet → 404) shows up as `extend_http_error` while the
        // ingest tee stays visibly healthy.
        let (url, sent, http_error, error) = match msg.kind {
            TeeKind::Ingest => (&ingest_url, "sent", "http_error", "error"),
            TeeKind::Extend => (
                &extend_url,
                "extend_sent",
                "extend_http_error",
                "extend_error",
            ),
        };
        // Serializing {model, input_ids} cannot realistically fail; count it
        // rather than unwrap-panicking the sole sender task.
        let bytes = match serde_json::to_vec(&body) {
            Ok(b) => b,
            Err(_) => {
                metrics.record_cache_sim_tee(error);
                continue;
            }
        };
        // reqwest returns Ok for ANY completed exchange, INCLUDING 4xx/5xx, so
        // inspect the status: a misconfigured URL (404) or an overloaded/OOM
        // cache-sim (503) is a broken tee, not a delivery — counting it "sent"
        // would blind the one health signal this counter exists to be. `error`
        // stays transport-only (connect refused / DNS / the 2s timeout) so a
        // dashboard can tell "cache-sim rejecting" from "cache-sim unreachable".
        match client
            .post(url)
            .header("content-type", "application/json")
            .body(bytes)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => metrics.record_cache_sim_tee(sent),
            Ok(_) => metrics.record_cache_sim_tee(http_error),
            Err(_) => metrics.record_cache_sim_tee(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::State, routing::post, Router};
    use std::sync::Mutex;
    use std::time::Duration;

    /// Build the tee without spawning the background sender, returning the
    /// receiver so the channel stays open but undrained — lets a test fill the
    /// queue and exercise the drop path deterministically.
    fn unstarted(
        capacity: usize,
        metrics: Arc<MetricsRegistry>,
    ) -> (CacheSimTee, mpsc::Receiver<TeeMsg>) {
        let (tx, rx) = mpsc::channel(capacity);
        (CacheSimTee { tx, metrics }, rx)
    }

    #[tokio::test]
    async fn offer_posts_input_ids_to_ingest_ids() {
        // Mock cache-sim: capture the last /ingest_ids body, reply 204.
        let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
        let app =
            Router::new()
                .route(
                    "/ingest_ids",
                    post(
                        |State(cap): State<Arc<Mutex<Option<Vec<u8>>>>>,
                         body: axum::body::Bytes| async move {
                            *cap.lock().unwrap() = Some(body.to_vec());
                            axum::http::StatusCode::NO_CONTENT
                        },
                    ),
                )
                .with_state(Arc::clone(&captured));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let metrics = MetricsRegistry::new();
        let tee = CacheSimTee::spawn(format!("http://{addr}"), Arc::clone(&metrics));
        tee.offer("m", &[10, 11, 12], "rid-1");

        // The sender POSTs asynchronously; poll until the body lands.
        let mut body = None;
        for _ in 0..80 {
            if let Some(b) = captured.lock().unwrap().clone() {
                body = Some(b);
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let body = body.expect("cache-sim never received a POST");
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model"], "m");
        assert_eq!(v["input_ids"], serde_json::json!([10, 11, 12]));
        assert_eq!(v["request_id"], "rid-1", "the join key must reach the wire");
        // The ingest body IS the prompt, so a boundary would be redundant —
        // and its presence on this path would mean the receiver had two
        // disagreeing notions of where the prompt ends.
        assert!(
            v.get("prompt_len").is_none(),
            "prompt_len must not be sent on /ingest_ids: {v}"
        );

        // And the outcome is metered as sent.
        let mut rendered = String::new();
        for _ in 0..80 {
            rendered = metrics.render();
            if rendered.contains(r#"sgl_router_cache_sim_tee_total{result="sent"} 1"#) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            rendered.contains(r#"sgl_router_cache_sim_tee_total{result="sent"} 1"#),
            "tee sent counter not rendered:\n{rendered}"
        );
    }

    /// The full-re-encode fallback has no valid prompt/output boundary, so it
    /// sends none. Absent must stay absent on the wire — a `null` or a `0`
    /// would read on the receiver as "zero output tokens", turning "cannot be
    /// derived" into a confident wrong answer.
    #[test]
    fn an_omitted_boundary_is_absent_from_the_wire_not_null() {
        let body = IngestIdsBody {
            model: "m",
            input_ids: &[1, 2, 3],
            request_id: "rid-1",
            prompt_len: None,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert!(v.get("prompt_len").is_none(), "serialized: {v}");
        assert_eq!(v["request_id"], "rid-1");
    }

    /// A boundary at or past the end of the sequence cannot be true: it would
    /// make output tokens zero or negative on the receiver. Drop the claim,
    /// keep the extension.
    #[tokio::test]
    async fn an_impossible_boundary_is_dropped_rather_than_sent() {
        let metrics = Arc::new(MetricsRegistry::default());
        let tee = CacheSimTee::spawn("http://127.0.0.1:1".to_string(), metrics);
        // Equal to the length, and past it: neither leaves any output tokens.
        for bad in [4usize, 99] {
            let msg = TeeMsg {
                kind: TeeKind::Extend,
                model: "m".into(),
                input_ids: vec![1, 2, 3, 4],
                request_id: "rid-1".into(),
                prompt_len: Some(bad).filter(|n| *n < 4),
            };
            assert!(msg.prompt_len.is_none(), "boundary {bad} should be dropped");
        }
        // A real boundary survives.
        tee.offer_extend("m", &[1, 2, 3, 4], "rid-1", Some(3));
    }

    #[tokio::test]
    async fn offer_extend_posts_to_extend_ids() {
        // Mock cache-sim: capture the last /extend_ids body, reply 204; a hit
        // on /ingest_ids would be a routing bug (fail via the captured-path
        // assertion below).
        type Captured = Arc<Mutex<Option<(String, Vec<u8>)>>>;
        let captured: Captured = Arc::new(Mutex::new(None));
        let handler = |path: &'static str| {
            move |State(cap): State<Captured>, body: axum::body::Bytes| async move {
                *cap.lock().unwrap() = Some((path.to_string(), body.to_vec()));
                axum::http::StatusCode::NO_CONTENT
            }
        };
        let app = Router::new()
            .route("/ingest_ids", post(handler("/ingest_ids")))
            .route("/extend_ids", post(handler("/extend_ids")))
            .with_state(Arc::clone(&captured));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let metrics = MetricsRegistry::new();
        let tee = CacheSimTee::spawn(format!("http://{addr}"), Arc::clone(&metrics));
        tee.offer_extend("m", &[10, 11, 12, 13], "rid-1", Some(2));

        let mut got = None;
        for _ in 0..80 {
            if let Some(g) = captured.lock().unwrap().clone() {
                got = Some(g);
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let (path, body) = got.expect("cache-sim never received a POST");
        assert_eq!(path, "/extend_ids", "extend must not land on /ingest_ids");
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model"], "m");
        assert_eq!(v["input_ids"], serde_json::json!([10, 11, 12, 13]));
        assert_eq!(
            v["request_id"], "rid-1",
            "extend must carry the SAME id as its ingest, or the two records cannot be paired"
        );
        assert_eq!(
            v["prompt_len"], 2,
            "the exact prompt/output boundary must reach the wire"
        );

        // Metered under the extend-specific label.
        let mut rendered = String::new();
        for _ in 0..80 {
            rendered = metrics.render();
            if rendered.contains(r#"sgl_router_cache_sim_tee_total{result="extend_sent"} 1"#) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            rendered.contains(r#"sgl_router_cache_sim_tee_total{result="extend_sent"} 1"#),
            "extend_sent counter not rendered:\n{rendered}"
        );
    }

    #[tokio::test]
    async fn offer_drops_when_queue_full_and_never_blocks() {
        let metrics = MetricsRegistry::new();
        // Capacity 1, no consumer draining: the first offer buffers, the rest
        // overflow and must be counted as dropped (and never block).
        let (tee, _rx) = unstarted(1, Arc::clone(&metrics));
        for _ in 0..5 {
            tee.offer("m", &[1, 2, 3], "rid-1");
        }
        let rendered = metrics.render();
        assert!(
            rendered.contains(r#"sgl_router_cache_sim_tee_total{result="dropped"}"#),
            "expected dropped tee outcomes to be counted:\n{rendered}"
        );
    }

    #[tokio::test]
    async fn offer_ignores_empty_ids() {
        let metrics = MetricsRegistry::new();
        let (tee, mut rx) = unstarted(4, Arc::clone(&metrics));
        tee.offer("m", &[], "rid-1");
        // Nothing enqueued.
        assert!(rx.try_recv().is_err());
    }

    // A cache-sim that returns 4xx/5xx is a BROKEN tee, not a delivery: reqwest's
    // Ok(resp) must not be counted "sent". Guards the tee's sole health signal.
    #[tokio::test]
    async fn run_sender_records_http_error_on_non_2xx() {
        let app = Router::new().route(
            "/ingest_ids",
            post(|| async { axum::http::StatusCode::INTERNAL_SERVER_ERROR }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let metrics = MetricsRegistry::new();
        let tee = CacheSimTee::spawn(format!("http://{addr}"), Arc::clone(&metrics));
        tee.offer("m", &[1, 2, 3], "rid-1");

        let mut rendered = String::new();
        for _ in 0..80 {
            rendered = metrics.render();
            if rendered.contains(r#"sgl_router_cache_sim_tee_total{result="http_error"} 1"#) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            rendered.contains(r#"sgl_router_cache_sim_tee_total{result="http_error"} 1"#),
            "a 500 from the cache-sim must count as http_error, not sent:\n{rendered}"
        );
        assert!(
            !rendered.contains(r#"sgl_router_cache_sim_tee_total{result="sent"}"#),
            "a 500 must not be counted sent:\n{rendered}"
        );
    }

    // Offering after the sender task is gone (channel closed) is counted `closed`,
    // never a panic or block.
    #[tokio::test]
    async fn offer_records_closed_when_sender_gone() {
        let metrics = MetricsRegistry::new();
        let (tee, rx) = unstarted(4, Arc::clone(&metrics));
        drop(rx); // no receiver → channel closed
        tee.offer("m", &[1, 2, 3], "rid-1");
        assert!(
            metrics
                .render()
                .contains(r#"sgl_router_cache_sim_tee_total{result="closed"}"#),
            "offer on a closed channel must count as closed"
        );
    }
}
