//! The transport-neutral half of `/generate`: request fan-out and submission
//! ([`generate_start`]), the unary fold ([`drain_unary`]), and the multiplexed
//! generation frame stream ([`generation_event_stream`]). Transports pick the
//! response shape; frame JSON shaping itself lives in the sibling `frame`.

use std::time::{Duration, Instant};

use futures::StreamExt;
use tokio::sync::mpsc;

use crate::api_server::core::error::ApiError;
use crate::api_server::core::frame::{
    OutputAccumulator, cumulative_frame_string, error_value, frame_typed, stream_frame_typed,
    tag_value, typed_frame_string,
};
use crate::api_server::core::guard::AbortGuard;
use crate::api_server::core::state::CoreState;
use crate::api_server::core::submit::submit;
use crate::message::ids::Rid;
use crate::message::request::{GenerateBody, RequestKind};
use crate::message::response::{ChunkEvent, ResponseItem};

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
    pub(crate) fn new() -> Self {
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
    let mut plan = start_generate_plan_with_timing(state, payloads, timing).await?;
    plan.is_batch = is_batch;
    Ok(plan)
}

/// Submit protocol-neutral generation requests through the common lifecycle.
/// This is the only generation submission entry point adapters should call.
pub(crate) async fn start_generate_plan(
    state: &CoreState,
    requests: Vec<crate::message::request::GenerateRequest>,
) -> Result<GeneratePlan, ApiError> {
    start_generate_plan_with_timing(state, requests, RequestTiming::new()).await
}

async fn start_generate_plan_with_timing(
    state: &CoreState,
    requests: Vec<crate::message::request::GenerateRequest>,
    timing: RequestTiming,
) -> Result<GeneratePlan, ApiError> {
    if requests.is_empty() {
        return Err(ApiError::bad_request("generation request list is empty"));
    }

    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let is_batch = requests.len() > 1;
    let mut receivers = Vec::with_capacity(requests.len());
    for request in requests {
        let (rid, rx) = submit(state, RequestKind::Generate(Box::new(request))).await?;
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

/// One folded unary result, transport-neutral: each transport renders
/// `Complete` in its own frame shape, maps `Error`'s HTTP-numbered code into
/// its own status space, and keeps the abort guard armed on `Truncated`.
pub(crate) enum UnaryOutcome {
    /// A real terminal item: the folded output and its finished timing.
    Complete(ChunkEvent, RequestTiming),
    /// Validation abort or pipeline error.
    Error { code: u16, message: String },
    /// Sender dropped without a terminal item: the shard dropped this request
    /// (a truncation — a client disconnect would have dropped the handler
    /// future).
    Truncated,
}

/// Fold a unary request to its terminal [`UnaryOutcome`]. Shared by single + batch.
pub(crate) async fn drain_unary(
    rx: &mut mpsc::Receiver<ResponseItem>,
    mut timing: RequestTiming,
) -> UnaryOutcome {
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
                    return UnaryOutcome::Error {
                        code,
                        message: message.to_string(),
                    };
                }
                return UnaryOutcome::Complete(final_out, timing);
            }
            ResponseItem::Error(e) => {
                timing.finish();
                return UnaryOutcome::Error {
                    code: e.http_status(),
                    message: e.to_string(),
                };
            }
            ResponseItem::Control(_) => continue, // never on `/generate`
        }
    }
    UnaryOutcome::Truncated
}

pub(crate) enum UnaryDrainPolicy {
    /// Drain every item and preserve one outcome per input. Native batch uses
    /// this because its public contract is per-item errors in a 200 response.
    PerItem,
    /// Stop at the first failed/truncated item. OpenAI fan-out uses this
    /// because several scheduler requests represent one logical HTTP request.
    AggregateFailFast,
}

fn outcome_error(outcome: &UnaryOutcome) -> Option<ApiError> {
    match outcome {
        UnaryOutcome::Complete(_, _) => None,
        UnaryOutcome::Error { code, message } => Some(ApiError::new(*code, message.clone())),
        UnaryOutcome::Truncated => Some(ApiError::internal("response truncated before completion")),
    }
}

/// Convert the one successful unary outcome used by protocol adapters. The
/// shared driver owns failure classification; adapters only need this one
/// common success/error boundary.
pub(crate) fn unary_output(outcome: UnaryOutcome) -> Result<ChunkEvent, ApiError> {
    match outcome {
        UnaryOutcome::Complete(output, _) => Ok(output),
        UnaryOutcome::Error { code, message } => Err(ApiError::new(code, message)),
        UnaryOutcome::Truncated => Err(ApiError::internal("response truncated before completion")),
    }
}

/// Drain all planned items through the same fold used by native unary output.
/// The policy controls only how a multi-item logical request treats failures;
/// receiver ownership and guard disarming stay in this shared function.
pub(crate) async fn drain_plan_unary(
    plan: GeneratePlan,
    policy: UnaryDrainPolicy,
) -> Result<Vec<(Rid, UnaryOutcome)>, ApiError> {
    let GeneratePlan {
        receivers,
        mut guard,
        ..
    } = plan;

    let mut futures = futures::stream::FuturesUnordered::new();
    for (order, (rid, mut rx, timing)) in receivers.into_iter().enumerate() {
        futures.push(async move {
            let outcome = drain_unary(&mut rx, timing).await;
            (order, rid, outcome)
        });
    }

    let mut drained = Vec::new();
    while let Some((order, rid, outcome)) = futures.next().await {
        if !matches!(outcome, UnaryOutcome::Truncated) {
            guard.disarm(&rid);
        }
        if matches!(policy, UnaryDrainPolicy::AggregateFailFast)
            && let Some(error) = outcome_error(&outcome)
        {
            // Dropping the guard here aborts every unfinished sibling.
            return Err(error);
        }
        drained.push((order, rid, outcome));
    }
    drained.sort_unstable_by_key(|(order, _, _)| *order);
    Ok(drained
        .into_iter()
        .map(|(_, rid, outcome)| (rid, outcome))
        .collect())
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
    /// One opaque renderer output; adapters own expansion and rendering failures.
    type Frame: Send + 'static;
    /// An incremental step frame (only under `incremental`).
    /// Only the completion count is retained in `acc` for incremental streams.
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
    /// As with `delta`, incremental `acc` contains only the running count.
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
    /// Protocol-only trailer, e.g. OpenAI's final usage chunk.
    fn finish(&mut self) -> Option<Self::Frame> {
        None
    }
}

/// The HTTP rendering: complete JSON frame strings — the serialized typed
/// frame, except cumulative intermediate frames, which keep the memoized fast
/// path (byte-pinned against the typed serialization).
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
        typed_frame_string(stream_frame_typed(out, acc, true, rid), index)
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

/// The gRPC rendering: the same typed frames HTTP serializes
/// (`frame_typed` / `stream_frame_typed`), shipped as `GenerateStreamItem`s —
/// one shaping source, so the two transports cannot drift.
pub(crate) struct PbFrameShaper;

impl PbFrameShaper {
    fn item(
        mut frame: sglang_api_types::api::v1::GenerateResponse,
        index: Option<usize>,
    ) -> sglang_api_types::api::v1::GenerateStreamItem {
        use sglang_api_types::api::v1 as genapi;
        frame.index = index.map(|i| u32::try_from(i).unwrap_or(u32::MAX));
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
        Self::item(stream_frame_typed(out, acc, true, rid), index)
    }
    fn coalesced(
        &mut self,
        acc: &OutputAccumulator,
        rid: &str,
        index: Option<usize>,
    ) -> Self::Frame {
        Self::item(frame_typed(acc.snapshot(), rid), index)
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
        let mut frame = stream_frame_typed(out, acc, incremental, rid);
        add_e2e_latency(&mut frame, timing);
        Self::item(frame, index)
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

/// Multiplex a generation plan into JSON frames. The [DONE] sentinel is
/// transport framing, appended by sse_encode.
pub(crate) fn generation_event_stream(plan: GeneratePlan) -> impl futures::Stream<Item = String> {
    generation_event_stream_with(plan, JsonFrameShaper)
}

/// The shared multiplex/coalesce/abort state machine, rendered by shaper.
pub(crate) fn generation_event_stream_with<S: FrameShaper>(
    plan: GeneratePlan,
    mut shaper: S,
) -> impl futures::Stream<Item = S::Frame> {
    async_stream::stream! {
        let GeneratePlan { receivers, mut guard, incremental, is_batch: with_index } = plan;
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

        let mut has_successful_terminal = false;

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
                        if incremental {
                            accs[i].count_tokens(&out);
                        } else {
                            accs[i].fold(&out);
                        }
                        if incremental {
                            yield shaper.delta(out, &accs[i], rid_strs[i].client_facing(), idx(i));
                        } else {
                            coalesced = true;
                        }
                    }
                    ResponseItem::Done(out) => {
                        timings[i].observe_first_output();
                        timings[i].finish();
                        if incremental {
                            accs[i].count_tokens(&out);
                        } else {
                            accs[i].fold(&out);
                        }
                        terminal = Some(out);
                        break;
                    }
                    ResponseItem::Error(e) => {
                        timings[i].finish();
                        failed = Some(e);
                        break;
                    }
                    ResponseItem::Control(_) => {} // never on /generate
                }
            }

            if let Some(e) = failed {
                guard.disarm(&rid_strs[i]);
                yield shaper.item_error(e.http_status(), &e.to_string(), idx(i));
            } else if let Some(out) = terminal {
                // A validation abort → an error item, not a frame. The final frame
                // carries the full cumulative state, so any coalesced ones are moot.
                // Disarm before rendering/yielding, even if rendering fails or the client drops.
                guard.disarm(&rid_strs[i]);
                yield match out.finish_reason.as_ref().and_then(|f| f.abort_status()) {
                    Some((code, message)) => {
                        let (code, message) = (code, message.to_owned());
                        shaper.item_error(code, &message, idx(i))
                    }
                    None => {
                        has_successful_terminal = true;
                        shaper.terminal(
                            out,
                            &accs[i],
                            incremental,
                            rid_strs[i].client_facing(),
                            idx(i),
                            &timings[i],
                        )
                    }
                };
            } else {
                if coalesced {
                    yield shaper.coalesced(&accs[i], rid_strs[i].client_facing(), idx(i));
                }
                futs.push(recv_indexed(i, rx)); // keep this item flowing
            }
        }
        if has_successful_terminal
            && let Some(frame) = shaper.finish()
        {
            yield frame;
        }
    }
}

/// Python's `e2e_latency` is `finished_time - created_time`, in seconds, and is
/// attached only when the request finishes. The Rust native API owns the same
/// lifecycle boundary, so it adds the value while handling the terminal egress
/// item rather than putting API-only timing onto every scheduler `ChunkEvent`.
pub(crate) fn add_e2e_latency(
    frame: &mut sglang_api_types::api::v1::GenerateResponse,
    timing: &RequestTiming,
) {
    let (time_to_first_token, e2e_latency) = timing
        .terminal_latencies()
        .expect("a successful terminal output has complete request timing");
    debug_assert!(time_to_first_token <= e2e_latency);
    frame
        .meta_info
        .get_or_insert_with(Default::default)
        .e2e_latency = Some(e2e_latency.as_secs_f64());
}

/// Render a terminal streaming frame. Intermediate cumulative frames keep the
/// memoized fast path; the one terminal frame serializes the typed frame so it
/// can carry the request-local `e2e_latency`, exactly as Python does.
fn terminal_stream_frame_string(
    out: ChunkEvent,
    acc: &OutputAccumulator,
    incremental: bool,
    rid_str: &str,
    index: Option<usize>,
    timing: &RequestTiming,
) -> String {
    let mut frame = stream_frame_typed(out, acc, incremental, rid_str);
    add_e2e_latency(&mut frame, timing);
    typed_frame_string(frame, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::core::frame::frame_typed;
    use crate::api_server::core::state::CoreState;
    use crate::api_server::core::test_utils::{TestReceiver, plan, senders};
    use crate::message::config::ServerArgs;
    use crate::message::request::GenerateRequest;
    use crate::message::response::ChunkEvent;
    use crate::tokenizer_manager::wiring::Senders;
    use crate::utils::error::Error;
    use futures::{FutureExt, StreamExt};
    use std::sync::{Arc, atomic::AtomicU64};
    use std::time::Duration;

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

    fn timed_receiver(rid: u64, rx: mpsc::Receiver<ResponseItem>) -> TestReceiver {
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

    fn abort_senders() -> (
        Senders,
        flume::Receiver<crate::tokenizer_manager::wiring::AbortSource>,
    ) {
        let (abort_tx, abort_rx) = flume::unbounded();
        (
            Senders {
                tok_manager_tx: flume::unbounded().0,
                abort_tx,
                tokenizer_tx: flume::unbounded().0,
                detokenizer_tx: vec![],
            },
            abort_rx,
        )
    }

    fn aborted_rids(
        abort_rx: &flume::Receiver<crate::tokenizer_manager::wiring::AbortSource>,
    ) -> Vec<String> {
        use crate::tokenizer_manager::wiring::AbortSource;

        let mut rids = abort_rx
            .try_iter()
            .filter_map(|source| match source {
                AbortSource::Guard(rid) => Some(rid.as_str().to_owned()),
                _ => None,
            })
            .collect::<Vec<_>>();
        rids.sort();
        rids
    }

    fn submission_state(
        tok_manager_tx: flume::Sender<crate::tokenizer_manager::wiring::TmEvent>,
        abort_tx: flume::Sender<crate::tokenizer_manager::wiring::AbortSource>,
    ) -> CoreState {
        CoreState {
            senders: Senders {
                tok_manager_tx,
                abort_tx,
                tokenizer_tx: flume::unbounded().0,
                detokenizer_tx: vec![],
            },
            response_buf: 4,
            api_key: None,
            server_args: Arc::new(ServerArgs::default()),
            chat_formatter: None,
            response_activity: Arc::new(AtomicU64::new(0)),
        }
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
        let UnaryOutcome::Complete(out, timing) = drain_unary(&mut rx, timing).await else {
            panic!("a Done item folds to Complete");
        };
        let mut frame = frame_typed(&out, "client-rid");
        add_e2e_latency(&mut frame, &timing);
        let value = serde_json::to_value(&frame).expect("a generated frame serializes");
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

    #[tokio::test]
    async fn partial_submission_failure_aborts_exactly_submitted_requests() {
        let (tok_manager_tx, tok_manager_rx) = flume::bounded(1);
        let (abort_tx, abort_rx) = flume::unbounded();
        let state = Arc::new(submission_state(tok_manager_tx, abort_tx));
        let requests = vec![
            GenerateRequest {
                rid: "submitted".into(),
                ..Default::default()
            },
            GenerateRequest {
                rid: "not-submitted".into(),
                ..Default::default()
            },
        ];

        let task = tokio::spawn({
            let state = Arc::clone(&state);
            async move { start_generate_plan(&state, requests).await }
        });
        while tok_manager_rx.is_empty() {
            tokio::task::yield_now().await;
        }
        drop(tok_manager_rx);

        assert!(task.await.unwrap().is_err());
        assert_eq!(aborted_rids(&abort_rx), ["submitted"]);
    }

    #[tokio::test]
    async fn submission_cancellation_aborts_exactly_submitted_requests() {
        let (tok_manager_tx, tok_manager_rx) = flume::bounded(1);
        let (abort_tx, abort_rx) = flume::unbounded();
        let state = Arc::new(submission_state(tok_manager_tx, abort_tx));
        let requests = vec![
            GenerateRequest {
                rid: "submitted".into(),
                ..Default::default()
            },
            GenerateRequest {
                rid: "pending".into(),
                ..Default::default()
            },
        ];

        let task = tokio::spawn({
            let state = Arc::clone(&state);
            async move { start_generate_plan(&state, requests).await }
        });
        while tok_manager_rx.is_empty() {
            tokio::task::yield_now().await;
        }
        task.abort();
        assert!(matches!(task.await, Err(error) if error.is_cancelled()));
        drop(tok_manager_rx);

        assert_eq!(aborted_rids(&abort_rx), ["submitted"]);
    }

    #[tokio::test]
    async fn per_item_plan_drain_preserves_order() {
        let (tx0, rx0) = mpsc::channel(2);
        let (tx1, rx1) = mpsc::channel(2);
        tx1.send(done(11, "second")).await.unwrap();

        let drain = drain_plan_unary(
            plan(
                vec![timed_receiver(10, rx0), timed_receiver(11, rx1)],
                senders(),
            ),
            UnaryDrainPolicy::PerItem,
        );
        futures::pin_mut!(drain);
        assert!(drain.as_mut().now_or_never().is_none());
        tx0.send(done(10, "first")).await.unwrap();
        let result = drain.await.expect("both planned items complete");

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0.as_str(), "10");
        assert_eq!(result[1].0.as_str(), "11");
    }

    #[tokio::test]
    async fn per_item_plan_drain_disarms_terminal_items_and_aborts_truncation() {
        use crate::tokenizer_manager::wiring::AbortSource;

        let (abort_tx, abort_rx) = flume::unbounded();
        let (done_tx, done_rx) = mpsc::channel(2);
        let (error_tx, error_rx) = mpsc::channel(2);
        let (truncated_tx, truncated_rx) = mpsc::channel(2);
        done_tx.send(done(10, "ok")).await.unwrap();
        error_tx
            .send(ResponseItem::Error(Error::Validation("bad".into())))
            .await
            .unwrap();
        drop(truncated_tx);

        let result = drain_plan_unary(
            plan(
                vec![
                    timed_receiver(10, done_rx),
                    timed_receiver(11, error_rx),
                    timed_receiver(12, truncated_rx),
                ],
                Senders {
                    tok_manager_tx: flume::unbounded().0,
                    abort_tx,
                    tokenizer_tx: flume::unbounded().0,
                    detokenizer_tx: vec![],
                },
            ),
            UnaryDrainPolicy::PerItem,
        )
        .await
        .expect("per-item mode keeps draining after one item fails");

        assert_eq!(result.len(), 3);
        assert!(matches!(result[0].1, UnaryOutcome::Complete(..)));
        assert!(matches!(result[1].1, UnaryOutcome::Error { .. }));
        assert!(matches!(result[2].1, UnaryOutcome::Truncated));
        assert!(matches!(
            abort_rx.try_recv().expect("truncation remains armed"),
            AbortSource::Guard(rid) if rid.as_str() == "12"
        ));
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn aggregate_plan_drain_fail_fast_aborts_unfinished_siblings() {
        use crate::tokenizer_manager::wiring::AbortSource;

        let (abort_tx, abort_rx) = flume::unbounded();
        let (error_tx, error_rx) = mpsc::channel(2);
        let (_pending_tx, pending_rx) = mpsc::channel(2);
        error_tx
            .send(ResponseItem::Error(Error::Validation("bad choice".into())))
            .await
            .unwrap();

        let result = drain_plan_unary(
            plan(
                vec![timed_receiver(10, error_rx), timed_receiver(11, pending_rx)],
                Senders {
                    tok_manager_tx: flume::unbounded().0,
                    abort_tx,
                    tokenizer_tx: flume::unbounded().0,
                    detokenizer_tx: vec![],
                },
            ),
            UnaryDrainPolicy::AggregateFailFast,
        )
        .await;

        assert!(result.is_err(), "the logical request fails fast");
        assert!(matches!(
            abort_rx.try_recv().expect("unfinished sibling is aborted"),
            AbortSource::Guard(rid) if rid.as_str() == "11"
        ));
        assert!(abort_rx.try_recv().is_err(), "the failed item was disarmed");
    }

    #[tokio::test]
    async fn dropping_native_stream_after_terminal_aborts_only_pending_item() {
        let (senders, abort_rx) = abort_senders();
        let (tx0, rx0) = mpsc::channel(4);
        let (_tx1, rx1) = mpsc::channel(4);
        tx0.send(done(10, "complete")).await.unwrap();

        {
            let stream = generation_event_stream(GeneratePlan {
                incremental: true,
                ..plan(
                    vec![timed_receiver(10, rx0), timed_receiver(11, rx1)],
                    senders,
                )
            });
            futures::pin_mut!(stream);
            assert_eq!(parse(&stream.next().await.unwrap())["text"], "complete");
        }

        assert_eq!(aborted_rids(&abort_rx), ["11"]);
    }

    /// Two sub-requests' frames interleave into one stream, each tagged with its
    /// batch `index`; text accumulates per item; `[DONE]` comes only after both
    /// terminate, then the stream ends.
    #[tokio::test]
    async fn interleaves_indexes_and_accumulates() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let receivers = vec![timed_receiver(10, rx0), timed_receiver(11, rx1)];
        let stream = generation_event_stream(GeneratePlan {
            incremental: false,
            is_batch: true,
            ..plan(receivers, senders())
        });
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
        let stream = generation_event_stream(GeneratePlan {
            incremental: false,
            is_batch: true,
            ..plan(receivers, senders())
        });
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
        let stream = generation_event_stream(GeneratePlan {
            incremental: true,
            is_batch: true,
            ..plan(receivers, senders())
        });
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
        let stream = generation_event_stream(GeneratePlan {
            incremental: false,
            is_batch: false,
            ..plan(receivers, senders())
        });
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
        let stream = generation_event_stream(GeneratePlan {
            incremental: false,
            is_batch: false,
            ..plan(receivers, senders())
        });
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
    ///
    /// Each send is followed by one poll, so nonterminal cumulative frames are
    /// consumed before `Done` arrives and the coalesced branch actually runs
    /// (a fully queued fixture bypasses it).
    #[tokio::test]
    async fn typed_shaper_matches_json_shaper() {
        use sglang_api_types::api::v1::generate_stream_item::Item;

        let channels = || {
            let (tx0, rx0) = mpsc::channel(8);
            let (tx1, rx1) = mpsc::channel(8);
            let receivers = vec![timed_receiver(10, rx0), timed_receiver(11, rx1)];
            (vec![tx0, tx1], receivers)
        };
        let bad = || ResponseItem::Error(crate::utils::error::Error::Validation("bad".into()));

        for (incremental, with_index) in [(false, false), (true, true), (false, true)] {
            let json_frames = {
                let (txs, receivers) = channels();
                let stream = generation_event_stream(GeneratePlan {
                    incremental,
                    is_batch: with_index,
                    ..plan(receivers, senders())
                });
                futures::pin_mut!(stream);
                let mut frames = Vec::new();
                txs[0].send(frame(10, "He")).await.unwrap();
                frames.push(parse(&stream.next().await.expect("first frame")));
                txs[0].send(frame(10, "llo")).await.unwrap();
                frames.push(parse(&stream.next().await.expect("second frame")));
                txs[0].send(done(10, "!")).await.unwrap();
                frames.push(parse(&stream.next().await.expect("terminal frame")));
                txs[1].send(bad()).await.unwrap();
                frames.push(parse(&stream.next().await.expect("error frame")));
                drop(txs);
                assert!(
                    stream.next().await.is_none(),
                    "stream ends after the script"
                );
                frames
            };

            // Intermediate content, running counts, and the batch index are all
            // real, not just the terminal frame. Incremental frames are deltas;
            // cumulative frames carry the folded text.
            let expected_text = if incremental {
                ["He", "llo", "!"]
            } else {
                ["He", "Hello", "Hello!"]
            };
            assert_eq!(json_frames[0]["text"], expected_text[0]);
            assert_eq!(json_frames[0]["meta_info"]["completion_tokens"], 1);
            assert_eq!(json_frames[1]["text"], expected_text[1]);
            assert_eq!(json_frames[1]["meta_info"]["completion_tokens"], 2);
            assert_eq!(json_frames[2]["text"], expected_text[2]);
            assert_eq!(json_frames[2]["meta_info"]["completion_tokens"], 3);
            assert_eq!(
                json_frames[2]["meta_info"]["finish_reason"]["type"],
                "length"
            );
            assert_eq!(
                json_frames[0]
                    .get("index")
                    .and_then(serde_json::Value::as_u64),
                with_index.then_some(0u64)
            );

            let typed_frames = {
                let (txs, receivers) = channels();
                let stream = generation_event_stream_with(
                    GeneratePlan {
                        incremental,
                        is_batch: with_index,
                        ..plan(receivers, senders())
                    },
                    PbFrameShaper,
                );
                futures::pin_mut!(stream);
                let mut frames = Vec::new();
                txs[0].send(frame(10, "He")).await.unwrap();
                frames.push(stream.next().await.expect("first frame"));
                txs[0].send(frame(10, "llo")).await.unwrap();
                frames.push(stream.next().await.expect("second frame"));
                txs[0].send(done(10, "!")).await.unwrap();
                frames.push(stream.next().await.expect("terminal frame"));
                txs[1].send(bad()).await.unwrap();
                frames.push(stream.next().await.expect("error frame"));
                drop(txs);
                assert!(
                    stream.next().await.is_none(),
                    "stream ends after the script"
                );
                frames
            };

            assert_eq!(json_frames.len(), typed_frames.len(), "frame counts");
            for (json, typed) in json_frames.iter().zip(&typed_frames) {
                match typed.item.as_ref().expect("typed item present") {
                    Item::Frame(f) => {
                        let mut want = json.clone();
                        let mut got = serde_json::to_value(f).unwrap();
                        // The two runs time independently; pin presence, drop value.
                        let w = want["meta_info"]["e2e_latency"].take();
                        let g = got["meta_info"]["e2e_latency"].take();
                        assert_eq!(w.is_null(), g.is_null(), "e2e_latency presence");
                        assert_eq!(got, want, "frame diverged (incremental={incremental})");
                    }
                    Item::Error(e) => {
                        let body = e.error.as_ref().expect("error body");
                        assert_eq!(json["error"]["message"], body.message.as_str());
                        assert_eq!(json["error"]["code"], body.code);
                        assert_eq!(
                            json.get("index").and_then(serde_json::Value::as_u64),
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
        let stream = generation_event_stream(GeneratePlan {
            incremental: true,
            is_batch: false,
            ..plan(receivers, senders())
        });
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
