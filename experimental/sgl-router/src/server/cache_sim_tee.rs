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
//! (never parked), and a background sender drains it with a bounded fan-out of
//! concurrent POSTs (see [`run_sender`]). The tee is purely observational, so it
//! must never slow, block, or fail the serving path.

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

/// Per-request attribution an upstream gateway stamps on the request it
/// dispatches (the `x-radixark-*` headers), read off the incoming request here
/// and RE-ATTACHED verbatim to the cache-sim tee POST. Pure pass-through: the
/// router never mints these — it relays what the upstream resolved (endpoint /
/// key / slug at auth, correlation server-minted). The cache-sim receiver reads
/// them back off the headers into each record, so the record is self-describing:
/// it can be partitioned by `slug` and deduped / correlated by `correlation_id`
/// without a separate lookup.
///
/// All optional: the upstream sets only non-empty values (a shared-bearer request
/// has no `key_id`; a direct-to-router request with no such gateway has none),
/// and each absent field is simply not sent — never as an empty header, which
/// would look like a real value downstream.
#[derive(Clone, Default)]
pub struct Attribution {
    /// `x-radixark-correlation-id` — the upstream's server-minted join key,
    /// distinct from the tee body's `request_id` (the router's own rid). Used
    /// downstream as the record's dedup + correlation key.
    pub correlation_id: Option<String>,
    pub endpoint_id: Option<String>,
    pub key_id: Option<String>,
    pub slug: Option<String>,
}

struct TeeMsg {
    kind: TeeKind,
    model: String,
    input_ids: Vec<u32>,
    output_tokens: Option<u64>,
    request_id: String,
    prompt_len: Option<usize>,
    choice: Option<Choice>,
    attr: Attribution,
}

/// Which of an `n > 1` response's alternative continuations an extension is.
///
/// One request produces ONE ingest record but N extends — every choice is a
/// separate continuation and each should seed the block store. Without a
/// discriminator those N records share a single `request_id`, so a receiver
/// joining on that key fans out N:1 and sums N times the real output; and N
/// rows under one key is also exactly what a duplicate-delivery bug looks
/// like, so the consumer cannot even detect the condition. `count` lets a
/// receiver verify it got them all; `index` lets it pick one for accounting
/// while still ingesting all of them for simulation.
#[derive(Clone, Copy, Serialize)]
pub struct Choice {
    pub index: usize,
    pub count: usize,
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
    /// Absent on `/ingest_ids` (a request has one prompt) and on a
    /// single-choice response; present whenever a response fanned out.
    #[serde(skip_serializing_if = "Option::is_none")]
    choice_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    choice_count: Option<usize>,
    /// The engine's own completion-token count, when the response reported
    /// one. Authoritative — unlike `input_ids.len() - prompt_len`, which
    /// measures the RE-RENDERED history turn (template envelope, EOS,
    /// re-serialized tool calls, reasoning the render drops) and drifts from
    /// what was actually generated on exactly the tool-calling traffic this
    /// oracle exists to measure.
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens: Option<u64>,
}

/// Handle the chat/completions handler offers pre-tokenized requests to.
pub struct CacheSimTee {
    tx: mpsc::Sender<TeeMsg>,
    metrics: Arc<MetricsRegistry>,
    /// Router-global budget on concurrent streaming response captures. Each
    /// armed [`crate::proxy::sse::StreamCapture`] holds one permit for its
    /// stream's lifetime, so aggregate capture memory is hard-bounded at
    /// `permits × MAX_EXTEND_CAPTURE_BYTES` regardless of traffic. When the
    /// budget is exhausted the request simply isn't captured (its extend tee is
    /// skipped) — the capture is observational, so shedding it is always safe.
    capture_sem: Arc<tokio::sync::Semaphore>,
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
    /// `max_captures` caps concurrent streaming captures (see `capture_sem`);
    /// `0` is clamped to `1` so the semaphore is always constructible.
    /// `send_concurrency` is how many teed POSTs the background sender keeps in
    /// flight at once — see [`run_sender`] for why serial draining was the tee's
    /// throughput ceiling; `0` is clamped to `1` (serial).
    pub fn spawn(
        url: String,
        metrics: Arc<MetricsRegistry>,
        max_captures: usize,
        send_concurrency: usize,
    ) -> Arc<Self> {
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
            send_concurrency,
        ));
        Arc::new(Self {
            tx,
            metrics,
            capture_sem: Arc::new(tokio::sync::Semaphore::new(max_captures.max(1))),
        })
    }

    /// Try to reserve one concurrent-capture slot. `Some(permit)` — the caller
    /// owns a slot until the permit drops (RAII, on the pump's end); `None` —
    /// the budget is exhausted, so the caller must NOT capture this stream
    /// (skip it, counted as `capture_capped`). Never blocks.
    pub fn try_acquire_capture_permit(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        Arc::clone(&self.capture_sem).try_acquire_owned().ok()
    }

    /// Offer one request's tokens to the tee. Never blocks: a full queue is
    /// dropped + counted, a closed channel (sender task gone) likewise. Empty
    /// id lists are a no-op. Cheap enough to call unconditionally on the hot
    /// path.
    pub fn offer(&self, model: &str, input_ids: &[u32], request_id: &str, attr: Attribution) {
        self.offer_kind(
            TeeKind::Ingest,
            model,
            input_ids,
            request_id,
            None,
            None,
            None,
            attr,
        );
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
        choice: Option<Choice>,
        output_tokens: Option<u64>,
    ) {
        // The extend (response-completion, insert-only) path runs in a detached
        // task that no longer holds the ingress request headers, so it carries no
        // attribution for now. The attributed path is the ingest tee above;
        // threading attribution here is a follow-up (capture it at ingress
        // alongside the extend prompt).
        self.offer_kind(
            TeeKind::Extend,
            model,
            input_ids,
            request_id,
            prompt_len,
            choice,
            output_tokens,
            Attribution::default(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn offer_kind(
        &self,
        kind: TeeKind,
        model: &str,
        input_ids: &[u32],
        request_id: &str,
        prompt_len: Option<usize>,
        choice: Option<Choice>,
        output_tokens: Option<u64>,
        attr: Attribution,
    ) {
        if input_ids.is_empty() {
            return;
        }
        // A boundary must be strictly inside the sequence: 0 would claim the
        // whole thing is output, >= len would claim none of it is. Neither can
        // be true on the incremental path (`prompt_ids ++ non_empty_suffix`),
        // so if one shows up the concat invariant that path rests on has
        // broken. Drop the claim — the extension is still worth teeing — but
        // COUNT it, because otherwise the record is indistinguishable on the
        // wire from a legitimate full-re-encode fallback, and the one signal
        // saying "your incremental path is producing garbage" reads as normal
        // traffic.
        let prompt_len = match prompt_len {
            Some(n) if (1..input_ids.len()).contains(&n) => Some(n),
            Some(_) => {
                // Its OWN counter, not a `result` label on the tee counter.
                // Those labels partition delivery attempts, and this message
                // still goes on to record `extend_sent` — sharing the counter
                // would make one attempt bump two labels, so
                // sum(cache_sim_tee_total) would stop being the attempt count
                // and any sent/total ratio would skew.
                self.metrics.record_cache_sim_boundary_impossible();
                None
            }
            None => None,
        };
        let msg = TeeMsg {
            kind,
            model: model.to_owned(),
            input_ids: input_ids.to_vec(),
            request_id: request_id.to_owned(),
            prompt_len,
            choice,
            output_tokens,
            attr,
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

/// Drain the queue, POSTing each request's ids to the cache-sim with up to
/// `concurrency` POSTs in flight at once. Errors are metered and dropped — a
/// down cache-sim must never spam the router's logs or affect serving. Ends when
/// the channel closes (all senders dropped, i.e. shutdown).
///
/// Why concurrent: the sender used to `.await` each POST serially, so tee
/// throughput was capped at ~`1 / POST-latency` (≈110 req/s at a ~9 ms
/// in-cluster POST). Any sustained arrival above that filled the bounded tee
/// channel and dropped the excess
/// (`sgl_router_cache_sim_tee_total{result="dropped"}`) — the oracle then
/// undercounted vs the engine at peak. Fanning out to `concurrency` tasks raises
/// the ceiling to ~`concurrency / POST-latency`; the per-POST timeout still
/// bounds a hung cache-sim.
///
/// A `concurrency`-permit semaphore backpressures the DRAIN, never the
/// offer/serving path (that's the non-blocking `try_send` in `offer`): when all
/// permits are out the recv loop parks until a POST finishes, so we never spawn
/// unbounded tasks. The queue keeps absorbing offers meanwhile, dropping on
/// overflow exactly as before.
///
/// INVARIANT: the recv loop must never panic — it is the sole consumer of the
/// tee channel, and a panic ends teeing permanently and silently. Each POST now
/// runs in its own task, so a (nonexistent — every step is matched) panic there
/// can't kill the consumer, and its permit is released on unwind.
async fn run_sender(
    mut rx: mpsc::Receiver<TeeMsg>,
    client: reqwest::Client,
    ingest_url: String,
    extend_url: String,
    metrics: Arc<MetricsRegistry>,
    concurrency: usize,
) {
    let ingest_url = Arc::new(ingest_url);
    let extend_url = Arc::new(extend_url);
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
    while let Some(msg) = rx.recv().await {
        // Acquire BEFORE spawning so at most `concurrency` POSTs are in flight;
        // this await parks the drain (not serving) when saturated.
        // `acquire_owned` errors only on a closed semaphore, which we never
        // close (we hold an Arc) — treat the impossible case as shutdown rather
        // than unwrap-panicking the sole consumer.
        let permit = match Arc::clone(&sem).acquire_owned().await {
            Ok(p) => p,
            Err(_) => break,
        };
        // reqwest::Client is Arc-backed — cloning shares the connection pool.
        let client = client.clone();
        let ingest_url = Arc::clone(&ingest_url);
        let extend_url = Arc::clone(&extend_url);
        let metrics = Arc::clone(&metrics);
        tokio::spawn(async move {
            let _permit = permit; // released when the POST finishes (or unwinds)
            post_one(msg, &client, &ingest_url, &extend_url, &metrics).await;
        });
    }
}

/// POST one teed message to the cache-sim and meter the outcome. Split out of
/// [`run_sender`] so the concurrent fan-out can spawn it directly. Never panics:
/// every fallible step is matched, not unwrapped.
async fn post_one(
    msg: TeeMsg,
    client: &reqwest::Client,
    ingest_url: &str,
    extend_url: &str,
    metrics: &MetricsRegistry,
) {
    let body = IngestIdsBody {
        model: &msg.model,
        input_ids: &msg.input_ids,
        request_id: &msg.request_id,
        prompt_len: msg.prompt_len,
        // A single-choice response carries no discriminator: absent means
        // "not a fan-out", which is the common case and the cheapest wire.
        choice_index: msg.choice.map(|c| c.index),
        choice_count: msg.choice.map(|c| c.count),
        output_tokens: msg.output_tokens,
    };
    // Per-kind outcome labels so a version-skewed cache-sim (no /extend_ids yet
    // → 404) shows up as `extend_http_error` while the ingest tee stays visibly
    // healthy.
    let (url, sent, http_error, error) = match msg.kind {
        TeeKind::Ingest => (ingest_url, "sent", "http_error", "error"),
        TeeKind::Extend => (
            extend_url,
            "extend_sent",
            "extend_http_error",
            "extend_error",
        ),
    };
    // Serializing {model, input_ids} cannot realistically fail; count it rather
    // than unwrap-panicking the sender task.
    let bytes = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(_) => {
            metrics.record_cache_sim_tee(error);
            return;
        }
    };
    // reqwest returns Ok for ANY completed exchange, INCLUDING 4xx/5xx, so
    // inspect the status: a misconfigured URL (404) or an overloaded/OOM
    // cache-sim (503) is a broken tee, not a delivery — counting it "sent"
    // would blind the one health signal this counter exists to be. `error`
    // stays transport-only (connect refused / DNS / the 2s timeout) so a
    // dashboard can tell "cache-sim rejecting" from "cache-sim unreachable".
    // Re-attach the upstream's per-request attribution as x-radixark-* headers
    // so the cache-sim receiver reads them into each record (it reads headers,
    // not the JSON body). Non-empty values only — an empty header would look
    // like a real value and collapse a downstream group-by / join. Pure
    // pass-through: the router relays what the upstream stamped, minting nothing.
    let mut rb = client.post(url).header("content-type", "application/json");
    for (name, value) in [
        ("x-radixark-correlation-id", &msg.attr.correlation_id),
        ("x-radixark-endpoint-id", &msg.attr.endpoint_id),
        ("x-radixark-key-id", &msg.attr.key_id),
        ("x-radixark-endpoint-slug", &msg.attr.slug),
    ] {
        if let Some(v) = value.as_deref().filter(|s| !s.is_empty()) {
            rb = rb.header(name, v);
        }
    }
    match rb.body(bytes).send().await {
        Ok(r) if r.status().is_success() => metrics.record_cache_sim_tee(sent),
        Ok(_) => metrics.record_cache_sim_tee(http_error),
        Err(_) => metrics.record_cache_sim_tee(error),
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
        (
            CacheSimTee {
                tx,
                metrics,
                capture_sem: Arc::new(tokio::sync::Semaphore::new(64)),
            },
            rx,
        )
    }

    #[tokio::test]
    async fn offer_posts_input_ids_to_ingest_ids() {
        // Mock cache-sim: capture the last /ingest_ids body + headers, reply 204.
        type Captured = Arc<Mutex<Option<(Vec<u8>, axum::http::HeaderMap)>>>;
        let captured: Captured = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route(
                "/ingest_ids",
                post(
                    |State(cap): State<Captured>,
                     headers: axum::http::HeaderMap,
                     body: axum::body::Bytes| async move {
                        *cap.lock().unwrap() = Some((body.to_vec(), headers));
                        axum::http::StatusCode::NO_CONTENT
                    },
                ),
            )
            .with_state(Arc::clone(&captured));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let metrics = MetricsRegistry::new();
        let tee = CacheSimTee::spawn(format!("http://{addr}"), Arc::clone(&metrics), 64, 8);
        // Full attribution — plus an empty key_id, which must be OMITTED
        // (not sent as a blank header).
        tee.offer(
            "m",
            &[10, 11, 12],
            "rid-1",
            Attribution {
                correlation_id: Some("corr-abc".into()),
                endpoint_id: Some("ep-1".into()),
                key_id: Some(String::new()),
                slug: Some("demo-slug".into()),
            },
        );

        // The sender POSTs asynchronously; poll until the body lands.
        let mut captured_pair = None;
        for _ in 0..80 {
            if let Some(p) = captured.lock().unwrap().clone() {
                captured_pair = Some(p);
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let (body, headers) = captured_pair.expect("cache-sim never received a POST");
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model"], "m");
        assert_eq!(v["input_ids"], serde_json::json!([10, 11, 12]));
        assert_eq!(v["request_id"], "rid-1", "the join key must reach the wire");
        // Per-request attribution is forwarded as x-radixark-* headers
        // (pass-through) so the cache-sim receiver can self-describe each record.
        let hdr = |n: &str| headers.get(n).and_then(|v| v.to_str().ok());
        assert_eq!(hdr("x-radixark-correlation-id"), Some("corr-abc"));
        assert_eq!(hdr("x-radixark-endpoint-id"), Some("ep-1"));
        assert_eq!(hdr("x-radixark-endpoint-slug"), Some("demo-slug"));
        assert!(
            headers.get("x-radixark-key-id").is_none(),
            "an empty attribution value must be omitted, never sent blank"
        );
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

    /// The sender POSTs concurrently, not serially. A mock that holds each POST
    /// briefly while tracking the live in-flight count proves the fan-out: with
    /// the old serial sender only ONE POST was ever in flight (peak == 1); the
    /// concurrent sender drives the peak above 1. This is the regression guard
    /// for the throughput ceiling that made the oracle undercount at peak.
    #[tokio::test]
    async fn sender_posts_concurrently_not_serially() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        #[derive(Clone, Default)]
        struct Live {
            cur: Arc<AtomicUsize>,
            max: Arc<AtomicUsize>,
        }
        let live = Live::default();
        let app = Router::new()
            .route(
                "/ingest_ids",
                post(|State(l): State<Live>, _b: axum::body::Bytes| async move {
                    // Track peak concurrency: bump on entry, hold, drop on exit.
                    let now = l.cur.fetch_add(1, Ordering::SeqCst) + 1;
                    l.max.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    l.cur.fetch_sub(1, Ordering::SeqCst);
                    axum::http::StatusCode::NO_CONTENT
                }),
            )
            .with_state(live.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let metrics = MetricsRegistry::new();
        // Concurrency 8, and 8 requests that each hold the mock 100 ms: a serial
        // sender would take ~800 ms with peak in-flight 1; the concurrent sender
        // runs them together.
        let tee = CacheSimTee::spawn(format!("http://{addr}"), Arc::clone(&metrics), 64, 8);
        for i in 0..8u32 {
            tee.offer("m", &[1, 2, 3], &format!("rid-{i}"), Attribution::default());
        }

        // Wait until all 8 land (or time out), then assert the peak in-flight
        // exceeded 1 — impossible under a serial sender.
        for _ in 0..80 {
            if metrics
                .render()
                .contains(r#"sgl_router_cache_sim_tee_total{result="sent"} 8"#)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let peak = live.max.load(Ordering::SeqCst);
        assert!(
            peak > 1,
            "sender must POST concurrently; peak in-flight was {peak} (a serial sender pins it at 1)"
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
            choice_index: None,
            choice_count: None,
            output_tokens: None,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert!(v.get("prompt_len").is_none(), "serialized: {v}");
        assert_eq!(v["request_id"], "rid-1");
        // Same reasoning for the fan-out and usage fields: a single-choice
        // response and an engine that reported no usage must be ABSENT, not
        // null or 0 — the receiver reads 0 as a real measurement.
        for k in ["choice_index", "choice_count", "output_tokens"] {
            assert!(v.get(k).is_none(), "{k} must be absent: {v}");
        }
    }

    /// A boundary at or past the end of the sequence cannot be true: it would
    /// leave zero or negative output tokens. Drop the claim, keep the
    /// extension.
    ///
    /// Asserts on what actually lands in the channel, via `unstarted()`. An
    /// earlier version of this test re-implemented the filter in its own body
    /// and asserted on that — which passed with the production guard deleted,
    /// i.e. it certified an unguarded path.
    #[tokio::test]
    async fn an_impossible_boundary_is_dropped_rather_than_sent() {
        let metrics = Arc::new(MetricsRegistry::default());
        let (tee, mut rx) = unstarted(8, Arc::clone(&metrics));

        // 0 claims the whole sequence is output; 4 and 99 claim none of it is.
        for bad in [0usize, 4, 99] {
            tee.offer_extend("m", &[1, 2, 3, 4], "rid-1", Some(bad), None, None);
            let msg = rx
                .try_recv()
                .expect("the extension itself must still be teed");
            assert!(msg.prompt_len.is_none(), "boundary {bad} must be dropped");
            assert_eq!(msg.input_ids, vec![1, 2, 3, 4]);
        }

        // A boundary strictly inside the sequence survives untouched.
        tee.offer_extend("m", &[1, 2, 3, 4], "rid-1", Some(3), None, None);
        assert_eq!(rx.try_recv().unwrap().prompt_len, Some(3));
    }

    /// The ingest leg never carries a boundary — its body IS the prompt.
    #[tokio::test]
    async fn ingest_never_carries_a_boundary() {
        let metrics = Arc::new(MetricsRegistry::default());
        let (tee, mut rx) = unstarted(4, Arc::clone(&metrics));
        tee.offer("m", &[1, 2, 3], "rid-1", Attribution::default());
        let msg = rx.try_recv().unwrap();
        assert!(msg.prompt_len.is_none());
        assert_eq!(msg.request_id, "rid-1");
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
        let tee = CacheSimTee::spawn(format!("http://{addr}"), Arc::clone(&metrics), 64, 8);
        tee.offer_extend(
            "m",
            &[10, 11, 12, 13],
            "rid-1",
            Some(2),
            Some(Choice { index: 1, count: 3 }),
            Some(42),
        );

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
        assert_eq!(
            v["choice_index"], 1,
            "fan-out discriminator must reach the wire"
        );
        assert_eq!(v["choice_count"], 3);
        assert_eq!(
            v["output_tokens"], 42,
            "the engine's own completion count must reach the wire"
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
            tee.offer("m", &[1, 2, 3], "rid-1", Attribution::default());
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
        tee.offer("m", &[], "rid-1", Attribution::default());
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
        let tee = CacheSimTee::spawn(format!("http://{addr}"), Arc::clone(&metrics), 64, 8);
        tee.offer("m", &[1, 2, 3], "rid-1", Attribution::default());

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
        tee.offer("m", &[1, 2, 3], "rid-1", Attribution::default());
        assert!(
            metrics
                .render()
                .contains(r#"sgl_router_cache_sim_tee_total{result="closed"}"#),
            "offer on a closed channel must count as closed"
        );
    }

    /// The capture budget is a hard bound: the first `N` permits are granted,
    /// the `N+1`th is refused (the request skips its capture), and dropping a
    /// permit frees a slot. This is what caps aggregate capture memory at
    /// `N × 16 MiB` under a streaming flood.
    #[tokio::test]
    async fn capture_permits_are_bounded_and_released_on_drop() {
        let metrics = MetricsRegistry::new();
        let (tee, _rx) = unstarted(4, Arc::clone(&metrics));
        // unstarted() gives a 64-permit budget; drain it to prove the bound and
        // the release both hold without depending on the exact N.
        let mut held = Vec::new();
        while let Some(p) = tee.try_acquire_capture_permit() {
            held.push(p);
            if held.len() > 1000 {
                panic!("capture budget is not bounded");
            }
        }
        assert_eq!(
            held.len(),
            64,
            "exactly the budget many permits are granted"
        );
        assert!(
            tee.try_acquire_capture_permit().is_none(),
            "an exhausted budget must refuse further captures",
        );
        held.pop(); // release one
        assert!(
            tee.try_acquire_capture_permit().is_some(),
            "a released permit must free a slot for the next capture",
        );
    }
}
