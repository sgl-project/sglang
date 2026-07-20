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

struct TeeMsg {
    model: String,
    input_ids: Vec<u32>,
}

/// Wire body of `POST /ingest_ids`. Hand-mirrored (no shared crate) by the
/// receiver's `IdsBody` in gpu-platform-proto `sglang-router-cache-sim`
/// (`src/server.rs`) — keep the two `{model, input_ids}` shapes in lockstep.
#[derive(Serialize)]
struct IngestIdsBody<'a> {
    model: &'a str,
    input_ids: &'a [u32],
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
        let ingest_url = format!("{}/ingest_ids", url.trim_end_matches('/'));
        tracing::info!(url = %ingest_url, "cache-sim tee enabled");
        tokio::spawn(run_sender(rx, client, ingest_url, Arc::clone(&metrics)));
        Arc::new(Self { tx, metrics })
    }

    /// Offer one request's tokens to the tee. Never blocks: a full queue is
    /// dropped + counted, a closed channel (sender task gone) likewise. Empty
    /// id lists are a no-op. Cheap enough to call unconditionally on the hot
    /// path.
    pub fn offer(&self, model: &str, input_ids: &[u32]) {
        if input_ids.is_empty() {
            return;
        }
        let msg = TeeMsg {
            model: model.to_owned(),
            input_ids: input_ids.to_vec(),
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
    url: String,
    metrics: Arc<MetricsRegistry>,
) {
    while let Some(msg) = rx.recv().await {
        let body = IngestIdsBody {
            model: &msg.model,
            input_ids: &msg.input_ids,
        };
        // Serializing {model, input_ids} cannot realistically fail; count it
        // rather than unwrap-panicking the sole sender task.
        let bytes = match serde_json::to_vec(&body) {
            Ok(b) => b,
            Err(_) => {
                metrics.record_cache_sim_tee("error");
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
            .post(&url)
            .header("content-type", "application/json")
            .body(bytes)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => metrics.record_cache_sim_tee("sent"),
            Ok(_) => metrics.record_cache_sim_tee("http_error"),
            Err(_) => metrics.record_cache_sim_tee("error"),
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
        tee.offer("m", &[10, 11, 12]);

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

    #[tokio::test]
    async fn offer_drops_when_queue_full_and_never_blocks() {
        let metrics = MetricsRegistry::new();
        // Capacity 1, no consumer draining: the first offer buffers, the rest
        // overflow and must be counted as dropped (and never block).
        let (tee, _rx) = unstarted(1, Arc::clone(&metrics));
        for _ in 0..5 {
            tee.offer("m", &[1, 2, 3]);
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
        tee.offer("m", &[]);
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
        tee.offer("m", &[1, 2, 3]);

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
        tee.offer("m", &[1, 2, 3]);
        assert!(
            metrics
                .render()
                .contains(r#"sgl_router_cache_sim_tee_total{result="closed"}"#),
            "offer on a closed channel must count as closed"
        );
    }
}
