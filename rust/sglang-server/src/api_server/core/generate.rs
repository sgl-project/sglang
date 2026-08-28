//! The transport-neutral half of `/generate`: request fan-out and submission
//! ([`generate_start`]), the unary fold ([`drain_unary`]), and the multiplexed
//! generation frame stream ([`generation_event_stream`]). Transports pick the
//! response shape; frame JSON shaping itself lives in the sibling `frame`.

use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::api_server::core::error::ApiError;
use crate::api_server::core::frame::{
    OutputAccumulator, cumulative_frame_string, frame_value, stream_frame_string,
    stream_frame_value, tag_value,
};
use crate::api_server::core::guard::AbortGuard;
use crate::api_server::core::state::CoreState;
use crate::api_server::core::submit::submit;
use crate::message::ids::Rid;
use crate::message::request::{GenerateBody, RequestKind};
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::utils::response::error_value;

/// API-local timing for one request.
///
/// Python records time-to-first-token on the first output batch and end-to-end
/// latency when that request finishes. Keep both measurements here even though
/// `/generate` currently exposes only `e2e_latency`; this avoids putting
/// API-only timestamps onto scheduler messages.
#[derive(Clone, Debug)]
pub(crate) struct RequestTiming {
    // TODO: Move request lifecycle timing into a dedicated tracing/metrics
    // module and align its design with Python's APIServerReqTimeStats.
    created_at: Instant,
    time_to_first_token: Option<Duration>,
    e2e_latency: Option<Duration>,
}

impl RequestTiming {
    fn new() -> Self {
        Self {
            created_at: Instant::now(),
            time_to_first_token: None,
            e2e_latency: None,
        }
    }

    fn observe_first_output(&mut self) {
        self.time_to_first_token
            .get_or_insert_with(|| self.created_at.elapsed());
    }

    fn finish(&mut self) {
        self.e2e_latency
            .get_or_insert_with(|| self.created_at.elapsed());
    }

    fn terminal_latencies(&self) -> Option<(Duration, Duration)> {
        Some((self.time_to_first_token?, self.e2e_latency?))
    }
}

/// Everything `/generate` submits before a response shape is chosen: one
/// receiver + timing per fanned-out request, the abort guard armed for all of
/// them, and the flags the response shaping needs.
pub(crate) struct GeneratePlan {
    pub(crate) receivers: Vec<(Rid, mpsc::Receiver<ResponseItem>, RequestTiming)>,
    pub(crate) guard: AbortGuard,
    /// List-form body — the unary response is a JSON array and every stream
    /// frame carries its `index`.
    pub(crate) is_batch: bool,
    /// Cumulative frames (SGLang default) vs per-step deltas.
    pub(crate) incremental: bool,
}

/// The transport-neutral front half of `/generate`: fan the body into
/// per-request payloads, prefetch media, and submit them all (the scheduler
/// runs a batch together). The guard covers every submitted rid, so dropping
/// the plan aborts them.
pub(crate) async fn generate_start(
    state: &CoreState,
    body: GenerateBody,
) -> Result<GeneratePlan, ApiError> {
    // Fan `text`/`input_ids`/`sampling_params` (scalar or list) into per-request
    // payloads; the error carries its own status (a bad batch is `Validation` → 400).
    let (mut payloads, is_batch) = body
        .into_requests()
        .map_err(|e| ApiError::from_pipeline(&e))?;
    // Python starts APIServerReqTimeStats after request normalization and before
    // tokenization / multimodal preprocessing / scheduler dispatch. Start at the
    // equivalent boundary: into_requests() has normalized the body, while prefetch
    // and every downstream stage are still ahead of us.
    let timing = RequestTiming::new();
    // Media I/O (URL downloads, file reads) happens here, on the API runtime
    // — never on the MM worker pool (see `prefetch`).
    crate::api_server::core::prefetch::prefetch_all(&mut payloads)
        .await
        .map_err(ApiError::bad_request)?;
    // No cross-item rid collision to worry about: `into_requests` rejected duplicate
    // rids within this batch, and `Rid::from_client` made each one unique against
    // every other in-flight request. `return_text_in_logprobs` is decoded on the
    // detok shard into `*_txt`, so frame shaping never needs a tokenizer here.
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut receivers = Vec::with_capacity(payloads.len());
    for req in payloads {
        let (rid, rx) = submit(state, RequestKind::Generate(Box::new(req))).await?;
        guard.arm(rid.clone());
        receivers.push((rid, rx, timing.clone()));
    }
    Ok(GeneratePlan {
        receivers,
        guard,
        is_batch,
        incremental: state.server_args.incremental_streaming_output,
    })
}

/// One folded unary result: the HTTP status code to answer with, the result or
/// `{"error": …}` body, and whether a real terminal item arrived (`false` =
/// truncation; the caller keeps the abort guard armed).
pub(crate) struct NativeUnary {
    pub(crate) code: u16,
    pub(crate) body: serde_json::Value,
    pub(crate) terminal: bool,
}

/// Fold a unary request to its terminal [`NativeUnary`]. Shared by single + batch.
pub(crate) async fn drain_unary(
    rx: &mut mpsc::Receiver<ResponseItem>,
    rid_str: &str,
    mut timing: RequestTiming,
) -> NativeUnary {
    let mut acc = OutputAccumulator::default();
    while let Some(item) = rx.recv().await {
        match item {
            ResponseItem::Frame(out) => {
                timing.observe_first_output();
                acc.fold(&out);
            }
            ResponseItem::Done(out) => {
                timing.observe_first_output();
                timing.finish();
                acc.fold(&out);
                let final_out = acc.into_output();
                // A validation abort carries its own HTTP status + diagnostic.
                if let Some((code, message)) = final_out
                    .finish_reason
                    .as_ref()
                    .and_then(|f| f.abort_status())
                {
                    return NativeUnary {
                        code,
                        body: error_value(code, message),
                        terminal: true,
                    };
                }
                let mut value = frame_value(&final_out, rid_str);
                add_e2e_latency(&mut value, &timing);
                return NativeUnary {
                    code: 200,
                    body: value,
                    terminal: true,
                };
            }
            ResponseItem::Error(e) => {
                timing.finish();
                let code = e.http_status();
                return NativeUnary {
                    code,
                    body: error_value(code, &e.to_string()),
                    terminal: true,
                };
            }
            ResponseItem::Control(_) | ResponseItem::Data(_) => continue, // never on `/generate`
        }
    }
    // Sender dropped without a terminal item: the shard dropped this request (a
    // truncation — a client disconnect would have dropped the handler future).
    NativeUnary {
        code: 500,
        body: error_value(500, "response truncated before completion"),
        terminal: false,
    }
}

/// Await the next item from `rx`, then drain whatever queued behind it (so the caller
/// can coalesce a backlog, as Python's `state.out_list` does), handing the receiver
/// back for `FuturesUnordered` to re-poll. Empty result = channel closed.
async fn recv_indexed(
    index: usize,
    mut rx: mpsc::Receiver<ResponseItem>,
) -> (usize, mpsc::Receiver<ResponseItem>, Vec<ResponseItem>) {
    let mut items = Vec::new();
    match rx.recv().await {
        Some(item) => items.push(item),
        None => return (index, rx, items), // closed
    }
    while let Ok(item) = rx.try_recv() {
        items.push(item);
    }
    (index, rx, items)
}

/// One transport's rendering of the multiplexed generation stream: the JSON
/// string frames the HTTP SSE encoder ships, or (gRPC) typed stream items.
/// The state machine in [`generation_event_stream_with`] is shared; only the
/// frame rendering differs per transport.
pub(crate) trait FrameShaper {
    type Frame: Send + 'static;
    /// An incremental step frame (only under `incremental`).
    fn delta(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        rid: &str,
        index: Option<usize>,
    ) -> Self::Frame;
    /// The coalesced cumulative frame for a drained backlog (only under
    /// cumulative streaming).
    fn coalesced(
        &mut self,
        acc: &OutputAccumulator,
        rid: &str,
        index: Option<usize>,
    ) -> Self::Frame;
    /// The terminal frame (never an abort — aborts became `item_error`).
    fn terminal(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        incremental: bool,
        rid: &str,
        index: Option<usize>,
        timing: &RequestTiming,
    ) -> Self::Frame;
    /// One request's failure (pipeline error, scheduler abort, truncation);
    /// the stream continues for the other batch items.
    fn item_error(&mut self, code: u16, message: &str, index: Option<usize>) -> Self::Frame;
}

/// The HTTP rendering: complete JSON frame strings, delegating 1:1 to the
/// byte-pinned frame.rs paths (incl. the memoized cumulative fast path).
pub(crate) struct JsonFrameShaper;

impl FrameShaper for JsonFrameShaper {
    type Frame = String;
    fn delta(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        rid: &str,
        index: Option<usize>,
    ) -> String {
        stream_frame_string(out, acc, true, rid, index)
    }
    fn coalesced(&mut self, acc: &OutputAccumulator, rid: &str, index: Option<usize>) -> String {
        cumulative_frame_string(acc, rid, index)
    }
    fn terminal(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        incremental: bool,
        rid: &str,
        index: Option<usize>,
        timing: &RequestTiming,
    ) -> String {
        terminal_stream_frame_string(out, acc, incremental, rid, index, timing)
    }
    fn item_error(&mut self, code: u16, message: &str, index: Option<usize>) -> String {
        tag_value(error_value(code, message), index)
    }
}

/// The gRPC rendering: typed `GenerateStreamItem`s, derived from the same
/// byte-pinned JSON shaping (`stream_frame_value` etc.) and converted through
/// the generated type, so both transports shape frames from one source.
// TODO(perf): hand-build the typed frames if the per-frame Value round-trip
// ever shows up in gRPC streaming profiles.
pub(crate) struct PbFrameShaper;

impl PbFrameShaper {
    fn typed_frame(
        mut value: serde_json::Value,
        index: Option<usize>,
    ) -> sglang_api_types::api::v1::GenerateStreamItem {
        use sglang_api_types::api::v1 as genapi;
        if let Some(i) = index {
            value["index"] = serde_json::json!(i);
        }
        let frame: genapi::GenerateResponse = serde_json::from_value(value)
            .expect("frame_value output parses into the generated GenerateResponse");
        genapi::GenerateStreamItem {
            item: Some(genapi::generate_stream_item::Item::Frame(frame)),
        }
    }
}

impl FrameShaper for PbFrameShaper {
    type Frame = sglang_api_types::api::v1::GenerateStreamItem;
    fn delta(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        rid: &str,
        index: Option<usize>,
    ) -> Self::Frame {
        Self::typed_frame(stream_frame_value(out, acc, true, rid), index)
    }
    fn coalesced(
        &mut self,
        acc: &OutputAccumulator,
        rid: &str,
        index: Option<usize>,
    ) -> Self::Frame {
        Self::typed_frame(frame_value(acc.snapshot(), rid), index)
    }
    fn terminal(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        incremental: bool,
        rid: &str,
        index: Option<usize>,
        timing: &RequestTiming,
    ) -> Self::Frame {
        let mut value = stream_frame_value(out, acc, incremental, rid);
        add_e2e_latency(&mut value, timing);
        Self::typed_frame(value, index)
    }
    fn item_error(&mut self, code: u16, message: &str, index: Option<usize>) -> Self::Frame {
        use sglang_api_types::api::v1 as genapi;
        genapi::GenerateStreamItem {
            item: Some(genapi::generate_stream_item::Item::Error(
                genapi::GenerateStreamError {
                    error: Some(genapi::ErrorBody {
                        message: message.to_owned(),
                        code: u32::from(code),
                    }),
                    index: index.map(|i| u32::try_from(i).unwrap_or(u32::MAX)),
                },
            )),
        }
    }
}

/// Multiplex `receivers` (one per request) into complete JSON frame strings;
/// `with_index` tags each frame (batch only), `incremental` = delta vs cumulative,
/// `guard` aborts unfinished on drop. The stream ends when every request has
/// terminated — the `[DONE]` sentinel is SSE framing, appended by `sse_encode`.
pub(crate) fn generation_event_stream(
    receivers: Vec<(Rid, mpsc::Receiver<ResponseItem>, RequestTiming)>,
    guard: AbortGuard,
    incremental: bool,
    with_index: bool,
) -> impl futures::Stream<Item = String> {
    generation_event_stream_with(receivers, guard, incremental, with_index, JsonFrameShaper)
}

/// The shared multiplex/coalesce/abort state machine, rendered by `shaper`.
pub(crate) fn generation_event_stream_with<S: FrameShaper>(
    receivers: Vec<(Rid, mpsc::Receiver<ResponseItem>, RequestTiming)>,
    mut guard: AbortGuard,
    incremental: bool,
    with_index: bool,
    mut shaper: S,
) -> impl futures::Stream<Item = S::Frame> {
    async_stream::stream! {
        use futures::StreamExt;

        let n = receivers.len();
        let rid_strs: Vec<Rid> = receivers
            .iter()
            .map(|(rid, _, _)| rid.clone())
            .collect();
        let mut timings: Vec<RequestTiming> = receivers
            .iter()
            .map(|(_, _, timing)| timing.clone())
            .collect();
        let mut accs: Vec<OutputAccumulator> =
            (0..n).map(|_| OutputAccumulator::default()).collect();

        // Batch position, tagged onto every frame (a single request omits it).
        let idx = |i: usize| with_index.then_some(i);

        // Poll all receivers concurrently; re-arm a receiver's future after each
        // non-terminal frame so its stream keeps flowing.
        let mut futs = futures::stream::FuturesUnordered::new();
        for (i, (_, rx, _)) in receivers.into_iter().enumerate() {
            futs.push(recv_indexed(i, rx));
        }

        while let Some((i, rx, items)) = futs.next().await {
            if items.is_empty() {
                // Channel closed with no terminal → truncation for this item;
                // leave its rid armed so the scheduler work is aborted.
                yield shaper.item_error(500, "response truncated before completion", idx(i));
                continue;
            }

            // Cumulative frames supersede one another, so a drained backlog collapses
            // to its last (Python's `out_list[-1]`); deltas can't be dropped.
            let mut coalesced = false; // a cumulative frame is pending
            let mut terminal = None;   // (finish_reason) of a `Done` in this batch
            let mut failed = None;     // an `Error` in this batch

            for item in items {
                match item {
                    ResponseItem::Frame(out) => {
                        timings[i].observe_first_output();
                        accs[i].fold(&out);
                        if incremental {
                            yield shaper.delta(out, &accs[i], rid_strs[i].client_facing(), idx(i));
                        } else {
                            coalesced = true;
                        }
                    }
                    ResponseItem::Done(out) => {
                        timings[i].observe_first_output();
                        timings[i].finish();
                        accs[i].fold(&out);
                        terminal = Some(out);
                    }
                    ResponseItem::Error(e) => {
                        timings[i].finish();
                        failed = Some(e);
                    }
                    ResponseItem::Control(_) | ResponseItem::Data(_) => {} // never on /generate
                }
            }

            if let Some(e) = failed {
                yield shaper.item_error(e.http_status(), &e.to_string(), idx(i));
                guard.disarm(&rid_strs[i]);
            } else if let Some(out) = terminal {
                // A validation abort → an error item, not a frame. The final frame
                // carries the full cumulative state, so any coalesced ones are moot.
                yield match out.finish_reason.as_ref().and_then(|f| f.abort_status()) {
                    Some((code, message)) => {
                        let (code, message) = (code, message.to_owned());
                        shaper.item_error(code, &message, idx(i))
                    }
                    None => shaper.terminal(
                        out,
                        &accs[i],
                        incremental,
                        rid_strs[i].client_facing(),
                        idx(i),
                        &timings[i],
                    ),
                };
                guard.disarm(&rid_strs[i]); // terminal → not re-pushed
            } else {
                if coalesced {
                    yield shaper.coalesced(&accs[i], rid_strs[i].client_facing(), idx(i));
                }
                futs.push(recv_indexed(i, rx)); // keep this item flowing
            }
        }
    }
}

/// Python's `e2e_latency` is `finished_time - created_time`, in seconds, and is
/// attached only when the request finishes. The Rust native API owns the same
/// lifecycle boundary, so it adds the value while handling the terminal egress
/// item rather than putting API-only timing onto every scheduler `ChunkEvent`.
fn add_e2e_latency(value: &mut serde_json::Value, timing: &RequestTiming) {
    let (time_to_first_token, e2e_latency) = timing
        .terminal_latencies()
        .expect("a successful terminal output has complete request timing");
    debug_assert!(time_to_first_token <= e2e_latency);
    value["meta_info"]["e2e_latency"] = serde_json::json!(e2e_latency.as_secs_f64());
}

/// Render a terminal streaming frame. Intermediate cumulative frames keep the
/// memoized fast path; the one terminal frame uses the Value path so it can carry
/// the request-local `e2e_latency`, exactly as Python does.
fn terminal_stream_frame_string(
    out: ChunkEvent,
    acc: &OutputAccumulator,
    incremental: bool,
    rid_str: &str,
    index: Option<usize>,
    timing: &RequestTiming,
) -> String {
    let mut value =
        crate::api_server::core::frame::stream_frame_value(out, acc, incremental, rid_str);
    add_e2e_latency(&mut value, timing);
    tag_value(value, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::response::ChunkEvent;
    use crate::tokenizer_manager::wiring::Senders;
    use crate::utils::error::Error;
    use futures::StreamExt;
    use std::time::Duration;
    fn senders() -> Senders {
        Senders {
            tok_manager_tx: flume::unbounded().0,
            abort_tx: flume::unbounded().0,
            tokenizer_tx: flume::unbounded().0,
            detokenizer_tx: vec![],
        }
    }

    fn frame(rid: u64, text: &str) -> ResponseItem {
        ResponseItem::Frame(ChunkEvent {
            rid: Rid::from(rid.to_string()),
            text: text.into(),
            completion_tokens: 1,
            ..Default::default()
        })
    }
    fn done(rid: u64, text: &str) -> ResponseItem {
        ResponseItem::Done(ChunkEvent {
            rid: Rid::from(rid.to_string()),
            text: text.into(),
            completion_tokens: 1,
            // Parsed from the wire map Python emits, not a hand-built enum.
            finish_reason: Some(
                serde_json::from_value(serde_json::json!({"type": "length", "length": 1}))
                    .expect("finish reason must parse"),
            ),
            ..Default::default()
        })
    }
    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("frame is JSON")
    }

    fn timed_receiver(
        rid: u64,
        rx: mpsc::Receiver<ResponseItem>,
    ) -> (Rid, mpsc::Receiver<ResponseItem>, RequestTiming) {
        (
            Rid::from(rid.to_string()),
            rx,
            RequestTiming {
                created_at: Instant::now() - Duration::from_millis(10),
                time_to_first_token: None,
                e2e_latency: None,
            },
        )
    }

    #[test]
    fn request_timing_records_ttft_once_and_e2e_on_finish() {
        let mut timing = RequestTiming {
            created_at: Instant::now() - Duration::from_millis(10),
            time_to_first_token: None,
            e2e_latency: None,
        };
        assert!(timing.terminal_latencies().is_none());

        timing.observe_first_output();
        let time_to_first_token = timing.time_to_first_token.unwrap();
        timing.observe_first_output();
        assert_eq!(
            timing.time_to_first_token,
            Some(time_to_first_token),
            "later output must not overwrite TTFT"
        );

        timing.finish();
        let (recorded_ttft, e2e_latency) = timing.terminal_latencies().unwrap();
        assert_eq!(recorded_ttft, time_to_first_token);
        assert!(e2e_latency >= recorded_ttft);

        timing.finish();
        assert_eq!(
            timing.terminal_latencies(),
            Some((recorded_ttft, e2e_latency)),
            "later terminal handling must not overwrite E2E latency"
        );
    }

    /// The native unary response uses the same names and meanings as Python's
    /// TokenizerManager metadata, and adds e2e_latency only on the terminal
    /// result. The timer is seconds from normalized-request acceptance through
    /// terminal-output handling.
    #[tokio::test]
    async fn unary_terminal_meta_info_matches_python_semantics() {
        let (tx, mut rx) = mpsc::channel(2);
        tx.send(ResponseItem::Done(ChunkEvent {
            rid: "internal-rid".into(),
            text: "ok".into(),
            token_ids: vec![7, 8],
            prompt_tokens: 5,
            completion_tokens: 2,
            finish_reason: serde_json::from_value(serde_json::json!({
                "type": "length",
                "length": 2
            }))
            .expect("finish reason must parse"),
            ..Default::default()
        }))
        .await
        .unwrap();

        let timing = RequestTiming {
            created_at: Instant::now() - Duration::from_millis(20),
            time_to_first_token: None,
            e2e_latency: None,
        };
        let unary = drain_unary(&mut rx, "client-rid", timing).await;
        assert_eq!(unary.code, 200);
        assert!(unary.terminal);
        let value = unary.body;
        assert_eq!(value["meta_info"]["id"], "client-rid");
        assert_eq!(value["meta_info"]["prompt_tokens"], 5);
        assert_eq!(value["meta_info"]["completion_tokens"], 2);
        assert_eq!(
            value["meta_info"]["finish_reason"],
            serde_json::json!({"type": "length", "length": 2})
        );
        assert!(
            value["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.020,
            "latency is expressed in seconds from request creation"
        );
        assert!(
            value["meta_info"].get("ttft").is_none()
                && value["meta_info"].get("time_to_first_token").is_none(),
            "TTFT is recorded internally but is not part of this PR's API"
        );
    }

    /// Two sub-requests' frames interleave into one stream, each tagged with its
    /// batch `index`; text accumulates per item; `[DONE]` comes only after both
    /// terminate, then the stream ends.
    #[tokio::test]
    async fn interleaves_indexes_and_accumulates() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let receivers = vec![timed_receiver(10, rx0), timed_receiver(11, rx1)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, true);
        futures::pin_mut!(stream);

        // Drive deterministically: exactly one channel has data before each poll.
        tx0.send(frame(10, "a")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["text"], "a");

        tx1.send(frame(11, "b")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);
        assert_eq!(v["text"], "b");

        tx0.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["text"], "a!", "cumulative per item");
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
        assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

        tx1.send(done(11, "?")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);
        assert_eq!(v["text"], "b?");
        assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

        // `[DONE]` is SSE framing appended by `sse_encode`, not a stream item.
        assert!(stream.next().await.is_none());
    }

    /// A per-item error is surfaced with its `index` and doesn't end the batch;
    /// `[DONE]` still waits for the other item.
    #[tokio::test]
    async fn per_item_error_carries_index() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let receivers = vec![timed_receiver(10, rx0), timed_receiver(11, rx1)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, true);
        futures::pin_mut!(stream);

        tx0.send(ResponseItem::Error(Error::Validation("bad".into())))
            .await
            .unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["error"]["code"], 400);

        tx1.send(done(11, "ok")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);

        assert!(stream.next().await.is_none());
    }

    /// `incremental=true`: each frame carries this step's **delta** text/output_ids,
    /// but `meta_info.completion_tokens` stays cumulative (matching Python).
    #[tokio::test]
    async fn incremental_emits_deltas_with_cumulative_count() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![timed_receiver(10, rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), true, true);
        futures::pin_mut!(stream);

        tx.send(frame(10, "Hello")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "Hello");
        assert_eq!(v["meta_info"]["completion_tokens"], 1);
        assert!(v["meta_info"].get("e2e_latency").is_none());

        tx.send(frame(10, " world")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], " world", "delta, not cumulative 'Hello world'");
        assert_eq!(
            v["meta_info"]["completion_tokens"], 2,
            "count stays cumulative"
        );
        assert!(v["meta_info"].get("e2e_latency").is_none());

        tx.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "!");
        assert_eq!(v["meta_info"]["completion_tokens"], 3);
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
        assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

        assert!(stream.next().await.is_none());
    }

    /// The single-request shape (`with_index=false`, one receiver) omits the
    /// `index` field entirely.
    #[tokio::test]
    async fn single_shape_omits_index() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![timed_receiver(10, rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, false);
        futures::pin_mut!(stream);

        tx.send(done(10, "hi")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "hi");
        assert!(v.get("index").is_none(), "single response has no index");
        assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

        assert!(stream.next().await.is_none());
    }

    /// A backlog of cumulative chunks collapses to a single frame carrying the latest
    /// state — each cumulative frame supersedes the last, so emitting the intermediate
    /// ones ships the full O(T) payload again for nothing. Mirrors the Python waiter's
    /// `out = out_list[-1]`. This is the whole point of draining in `recv_indexed`.
    #[tokio::test]
    async fn cumulative_backlog_coalesces_to_latest() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![timed_receiver(10, rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, false);
        futures::pin_mut!(stream);

        // Three chunks queued before the stream is ever polled (a client falling behind).
        tx.send(frame(10, "a")).await.unwrap();
        tx.send(frame(10, "b")).await.unwrap();
        tx.send(frame(10, "c")).await.unwrap();

        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "abc", "one frame, full cumulative text");
        assert_eq!(v["meta_info"]["completion_tokens"], 3, "no tokens lost");

        // The terminal frame still carries everything, and only then does the
        // stream end.
        tx.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "abc!");
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
        assert!(stream.next().await.is_none());
    }

    /// The typed (gRPC) shaper renders exactly what the JSON shaper renders:
    /// identical scripted streams through both, frames compared as JSON trees
    /// (e2e_latency normalized — the two runs time independently) and errors
    /// compared field-for-field with their batch index.
    #[tokio::test]
    async fn typed_shaper_matches_json_shaper() {
        use sglang_api_types::api::v1::generate_stream_item::Item;

        for (incremental, with_index) in [(false, false), (true, true), (false, true)] {
            let script = |txs: &[tokio::sync::mpsc::Sender<ResponseItem>]| {
                let a = txs[0].clone();
                let b = txs[1].clone();
                async move {
                    a.send(frame(10, "He")).await.unwrap();
                    a.send(frame(10, "llo")).await.unwrap();
                    a.send(done(10, "!")).await.unwrap();
                    b.send(ResponseItem::Error(crate::utils::error::Error::Validation(
                        "bad".into(),
                    )))
                    .await
                    .unwrap();
                }
            };
            let run = |shaperless: bool| {
                let (tx0, rx0) = mpsc::channel(8);
                let (tx1, rx1) = mpsc::channel(8);
                let receivers = vec![timed_receiver(10, rx0), timed_receiver(11, rx1)];
                let guard = AbortGuard::new_empty(senders());
                (vec![tx0, tx1], receivers, guard, shaperless)
            };

            let (txs, receivers, guard, _) = run(true);
            script(&txs).await;
            drop(txs);
            let json_frames: Vec<String> =
                generation_event_stream(receivers, guard, incremental, with_index)
                    .collect()
                    .await;

            let (txs, receivers, guard, _) = run(false);
            script(&txs).await;
            drop(txs);
            let typed_frames: Vec<sglang_api_types::api::v1::GenerateStreamItem> =
                generation_event_stream_with(
                    receivers,
                    guard,
                    incremental,
                    with_index,
                    PbFrameShaper,
                )
                .collect()
                .await;

            assert_eq!(json_frames.len(), typed_frames.len(), "frame counts");
            for (json, typed) in json_frames.iter().zip(&typed_frames) {
                let mut want: serde_json::Value = serde_json::from_str(json).unwrap();
                match typed.item.as_ref().expect("typed item present") {
                    Item::Frame(f) => {
                        let mut got = serde_json::to_value(f).unwrap();
                        // The two runs time independently; pin presence, drop value.
                        let w = want["meta_info"]["e2e_latency"].take();
                        let g = got["meta_info"]["e2e_latency"].take();
                        assert_eq!(w.is_null(), g.is_null(), "e2e_latency presence");
                        assert_eq!(got, want, "frame diverged (incremental={incremental})");
                    }
                    Item::Error(e) => {
                        let body = e.error.as_ref().expect("error body");
                        assert_eq!(want["error"]["message"], body.message.as_str());
                        assert_eq!(want["error"]["code"], body.code);
                        assert_eq!(
                            want.get("index").and_then(serde_json::Value::as_u64),
                            e.index.map(u64::from),
                            "error index"
                        );
                    }
                }
            }
        }
    }

    /// Incremental frames are *deltas*, so a backlog must emit every one — dropping
    /// any would silently lose tokens. Only the cumulative protocol may coalesce.
    #[tokio::test]
    async fn incremental_backlog_emits_every_delta() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![timed_receiver(10, rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), true, false);
        futures::pin_mut!(stream);

        tx.send(frame(10, "a")).await.unwrap();
        tx.send(frame(10, "b")).await.unwrap();
        tx.send(frame(10, "c")).await.unwrap();

        for (n, expect) in [(1, "a"), (2, "b"), (3, "c")] {
            let v = parse(&stream.next().await.unwrap());
            assert_eq!(v["text"], expect, "delta {n} must not be dropped");
            assert_eq!(
                v["meta_info"]["completion_tokens"], n,
                "count stays cumulative"
            );
        }
    }
}
