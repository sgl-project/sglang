//! Detokenizer shards — CPU-bound, one pinned thread per shard.
//!
//! Each shard owns a *local* `rid -> DetokState` map. There is no lock: a given
//! rid is routed to exactly one shard (by `Rid::shard`) for both its
//! `Register` and all its `Chunk`s, so the map has a single accessor.
//!
//! The hash PARTITIONS, the rid IDENTIFIES. Keying the map by the hash meant two
//! distinct rids that happened to collide became one entry: `Register` evicted the
//! first client's sink and their tokens were then written to the second client's
//! connection. Chunks carry the rid string (moved out of the frame header, which
//! owns it and would otherwise drop it), so a collision now only co-locates.
//!
//! Real detokenization keeps a rolling token window with the same prompt-context
//! subtraction and partial UTF-8 recovery as Python's DetokenizerManager.
//! Each request owns its decoder. When no tokenizer is configured (or
//! `skip_tokenizer_init` is set) the backend is `Skip`: no decoding, the raw
//! `output_ids` are emitted instead of text.
//!
//! Per-chunk response flow (no FSM state change inside Streaming):
//!   ChunkEvent{finish:None}  -> step ids -> delta -> Server frame
//!   ChunkEvent{finish:Some}  -> step ids -> delta -> final frame

use std::collections::HashMap;
use std::sync::Arc;

use crate::message::detok::DetokMsg;
use crate::message::finish_reason::Matched;
use crate::message::ids::Rid;
use crate::message::response::{
    ChunkEvent, PROMPT_CONTEXT_TOKENS, ResponseItem, ResponseSink, SchedulerAbort, SinkError,
    TokenCounts,
};
use crate::message::types::TokenIds;
use crate::tokenizer_manager::wiring::LifecycleEvent;
use crate::utils::runtime::Runnable;
use crate::utils::{
    error::Error,
    fsm::{Event, RequestState},
};

/// Per-request incremental decoder. `step` feeds the new token ids for one chunk
/// and returns the newly decoded text delta (empty if the ids only produced a
/// partial/incomplete multi-byte sequence that needs more tokens).
pub trait StreamDecoder: Send {
    fn step(&mut self, token_ids: &[i32], finished: bool) -> Result<String, Error>;
}

/// Decode the previous committed chunk together with the new tokens. Keeping
/// that context preserves leading spaces and byte-fallback boundaries without
/// decoding the request's entire output on every step.
struct DynamoDecoder {
    tokenizer: dynamo_tokenizers::Tokenizer,
    skip_special_tokens: bool,
    vocab_size: Option<u64>,
    ids: Vec<u32>,
    read_offset: usize,
    /// Printable characters already emitted from an incomplete UTF-8 tail.
    pending_chars: usize,
}

impl StreamDecoder for DynamoDecoder {
    fn step(&mut self, token_ids: &[i32], finished: bool) -> Result<String, Error> {
        self.ids.extend(
            token_ids
                .iter()
                .map(|&id| clamp_decode_id(id, self.vocab_size)),
        );
        let decode = |ids: &[u32]| {
            self.tokenizer
                .decode(ids, self.skip_special_tokens)
                .map(String::from)
                .map_err(|error| Error::Detokenize(error.to_string()))
        };
        let context = decode(&self.ids[..self.read_offset])?;
        let text = decode(&self.ids)?;
        // Python slices by Unicode characters, including when an incomplete
        // prompt suffix changes after the generated tokens complete it.
        let new_text = text
            .strip_prefix(&context)
            .unwrap_or_else(|| skip_chars(&text, context.chars().count()));
        if finished {
            // A terminal incomplete sequence is visible as U+FFFD in Python.
            return Ok(skip_chars(new_text, self.pending_chars).to_owned());
        }
        if !new_text.is_empty() && !new_text.ends_with('\u{fffd}') {
            let delta = skip_chars(new_text, self.pending_chars).to_owned();
            self.ids.drain(..self.read_offset);
            self.read_offset = self.ids.len();
            self.pending_chars = 0;
            Ok(delta)
        } else {
            let printable = printable_prefix(new_text);
            let delta = skip_chars(printable, self.pending_chars).to_owned();
            self.pending_chars = self.pending_chars.max(printable.chars().count());
            Ok(delta)
        }
    }
}

fn clamp_decode_id(id: i32, vocab_size: Option<u64>) -> u32 {
    if id >= 0
        && vocab_size
            .filter(|&size| size > 0)
            .is_none_or(|size| (id as u64) < size)
    {
        id as u32
    } else {
        0
    }
}

fn skip_chars(text: &str, count: usize) -> &str {
    text.char_indices()
        .nth(count)
        .map_or("", |(offset, _)| &text[offset..])
}

/// Python's `find_printable_text`: withhold a possibly changing word suffix,
/// but permit complete words, newlines and CJK characters through immediately.
fn printable_prefix(text: &str) -> &str {
    let is_cjk = |ch: char| {
        matches!(ch as u32,
            0x4e00..=0x9fff | 0x3400..=0x4dbf | 0x20000..=0x2a6df |
            0x2a700..=0x2b73f | 0x2b740..=0x2b81f | 0x2b820..=0x2ceaf |
            0xf900..=0xfaff | 0x2f800..=0x2fa1f
        )
    };
    let mut chars = text.char_indices().rev();
    if let Some((last_offset, last)) = chars.next() {
        if last == '\n' || is_cjk(last) {
            return text;
        }
        if chars.next().is_some_and(|(_, ch)| is_cjk(ch)) {
            return &text[..last_offset];
        }
    }
    text.rfind(' ').map_or("", |offset| &text[..offset + 1])
}

/// Shard-wide detok backend. Cloned per shard; mints a fresh per-request decoder
/// when the first chunk supplies its prompt context.
#[derive(Clone)]
pub enum DetokenizerBackend {
    Dynamo {
        tokenizer: dynamo_tokenizers::Tokenizer,
        vocab_size: Option<u64>,
    },
    /// No decoding at all — the shard emits each chunk's raw output token ids as
    /// `output_ids` (no decoder, no accumulation). Used for
    /// `skip_tokenizer_init` and when no tokenizer is configured.
    Skip,
}

impl DetokenizerBackend {
    /// Mint a per-request decoder, or `None` in skip mode (the shard passes the
    /// token ids through untouched instead of decoding text).
    fn new_decoder(
        &self,
        prompt_context: &[i32],
        skip_special_tokens: bool,
    ) -> Option<Box<dyn StreamDecoder>> {
        match self {
            DetokenizerBackend::Dynamo {
                tokenizer,
                vocab_size,
            } => {
                let prompt_context =
                    &prompt_context[prompt_context.len().saturating_sub(PROMPT_CONTEXT_TOKENS)..];
                Some(Box::new(DynamoDecoder {
                    tokenizer: tokenizer.clone(),
                    skip_special_tokens,
                    vocab_size: *vocab_size,
                    ids: prompt_context
                        .iter()
                        .map(|&id| clamp_decode_id(id, *vocab_size))
                        .collect(),
                    read_offset: prompt_context.len(),
                    pending_chars: 0,
                }))
            }
            DetokenizerBackend::Skip => None,
        }
    }

    /// Decode one complete sequence without creating request-scoped streaming
    /// state. This runs on a pinned detokenizer worker, never on an API runtime
    /// thread.
    fn decode_once(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        match self {
            DetokenizerBackend::Dynamo { tokenizer, .. } => tokenizer
                .decode(token_ids, skip_special_tokens)
                .map(String::from)
                .map_err(|error| Error::Detokenize(error.to_string())),
            DetokenizerBackend::Skip => Err(Error::Validation(
                "echo for token-ID prompts is unavailable when skip_tokenizer_init=True".into(),
            )),
        }
    }

    /// Decode each logprob token id to its own text (one id at a time, matching
    /// Python's `batch_decode([[id] for id in ids])`). Runs on this CPU-bound
    /// shard, not the api-server I/O threads. `Skip` mode (no tokenizer) yields
    /// no text, so the `[logprob, token_id, text]` tuple's text slot stays null.
    fn decode_logprob_texts(&self, idxs: &[i32]) -> Vec<String> {
        match self {
            DetokenizerBackend::Dynamo { tokenizer, .. } => idxs
                .iter()
                .map(|&id| {
                    tokenizer
                        .decode(&[id as u32], false)
                        .map(String::from)
                        .unwrap_or_default()
                })
                .collect(),
            DetokenizerBackend::Skip => Vec::new(),
        }
    }
}

struct DetokState {
    sink: ResponseSink,
    /// `return_text_in_logprobs`: whether to decode this request's logprob token
    /// ids to text (in this shard) for the `[logprob, token_id, text]` tuples.
    decode_logprob_text: bool,
    skip_special_tokens: bool,
    /// `SamplingParams.no_stop_trim`: keep the matched stop in the output.
    no_stop_trim: bool,
    stop_buffer: StopBuffer,
    logprobs: Option<crate::message::response::LogprobOptions>,
    /// Created from the first scheduler chunk's prompt context. Stays `None`
    /// in `skip_tokenizer_init` mode.
    /// The decoder keeps a rolling token window; `stop_buffer` holds a possible
    /// stop prefix. The api-server's drain loop reassembles emitted deltas into a
    /// cumulative view where a consumer needs it (every unary response and the
    /// cumulative SGLang `/generate` stream); OpenAI streaming forwards deltas.
    decoder: Option<Box<dyn StreamDecoder>>,
    /// The last scheduler snapshot is needed if a later queue abort has no
    /// token batch of its own. Text, ids and logprobs stay with the API consumer.
    counts: Arc<TokenCounts>,
    prompt_tokens: u32,
    prompt_token_ids: Option<Arc<[i32]>>,
    dispatch_finished_ts: Option<f64>,
    response_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    control_results: Vec<Option<bytes::Bytes>>,
    metrics: Option<Box<crate::metrics::RequestMetrics>>,
    response_processor: Option<Box<dyn crate::ResponseProcessor>>,
    /// Response half of the lifecycle FSM. Lives here because the `Request` (and
    /// its FSM) was handed to the scheduler when queued; the shard is the sole
    /// owner of the response state, so no lock.
    fsm: RequestState,
}

/// One detokenizer shard: owns a *local* `rid -> DetokState` map (single accessor,
/// no lock) and the detokenizer backend.
pub struct DetokenizerWorker {
    shard: usize,
    rx: flume::Receiver<DetokMsg>,
    backend: DetokenizerBackend,
    /// Completion releases intake state; terminal delivery failures also stop
    /// any remaining scheduler work.
    lifecycle: flume::Sender<LifecycleEvent>,
    http_extension: Option<Arc<dyn crate::HttpExtension>>,
}

impl DetokenizerWorker {
    pub fn new(
        shard: usize,
        rx: flume::Receiver<DetokMsg>,
        backend: DetokenizerBackend,
        lifecycle: flume::Sender<LifecycleEvent>,
    ) -> Self {
        Self {
            shard,
            rx,
            backend,
            lifecycle,
            http_extension: None,
        }
    }

    pub fn with_http_extension(mut self, extension: Option<Arc<dyn crate::HttpExtension>>) -> Self {
        self.http_extension = extension;
        self
    }
}

impl Runnable for DetokenizerWorker {
    fn run(self) {
        let mut table: HashMap<Rid, DetokState> = HashMap::new();
        tracing::debug!(shard = self.shard, "detokenizer worker started");

        // Plain `recv`: exits when the `DetokMsg` channel closes (every `Senders`
        // clone gone). On shutdown that happens once the API runtime drop cancels
        // in-flight handlers (their `AbortGuard`s release the last clones) and
        // to-scheduler/from-scheduler exit — no shutdown signal needed here.
        while let Ok(msg) = self.rx.recv() {
            match msg {
                DetokMsg::Register {
                    rid,
                    sink,
                    decode_logprob_text,
                    skip_special_tokens,
                    no_stop_trim,
                    stop_texts,
                    logprobs,
                    control_replies,
                    metrics,
                } => {
                    table.insert(
                        rid.clone(),
                        DetokState {
                            sink,
                            decode_logprob_text,
                            skip_special_tokens,
                            no_stop_trim,
                            stop_buffer: StopBuffer::new(stop_texts),
                            logprobs,
                            decoder: None,
                            counts: Default::default(),
                            prompt_tokens: 0,
                            prompt_token_ids: None,
                            dispatch_finished_ts: None,
                            response_metadata: None,
                            control_results: vec![None; control_replies],
                            metrics,
                            response_processor: self
                                .http_extension
                                .as_ref()
                                .and_then(|extension| extension.new_response_processor()),
                            // Registered == handed to the scheduler == Queued.
                            fsm: RequestState::Queued,
                        },
                    );
                }
                DetokMsg::Prepared {
                    rid,
                    prompt_token_ids,
                    sampling_params,
                    dispatch_finished_ts,
                    response_metadata,
                } => {
                    if let Some(state) = table.get_mut(&rid) {
                        state.prompt_token_ids = prompt_token_ids;
                        state.dispatch_finished_ts = dispatch_finished_ts;
                        state.response_metadata = response_metadata;
                        if let Some(processor) = &mut state.response_processor {
                            processor.prepare(&sampling_params);
                        }
                    }
                }
                // One decode step's chunks for this shard, batched by from-scheduler.
                DetokMsg::Chunks(evs) => {
                    for ev in evs {
                        handle_chunk(&mut table, ev, &self.backend, &self.lifecycle);
                    }
                }
                DetokMsg::Decode {
                    rid,
                    token_ids,
                    skip_special_tokens,
                } => {
                    handle_decode(
                        &mut table,
                        &rid,
                        &token_ids,
                        skip_special_tokens,
                        &self.backend,
                    );
                    let _ = self.lifecycle.send(LifecycleEvent::Finished(rid));
                }
                DetokMsg::Result { rid, payload } => {
                    handle_result(&mut table, rid, None, payload, &self.lifecycle);
                }
                DetokMsg::ResultPart {
                    rid,
                    dp_rank,
                    payload,
                } => {
                    handle_result(&mut table, rid, Some(dp_rank), payload, &self.lifecycle);
                }
                DetokMsg::Abort { rid, outcome } => {
                    handle_abort(&mut table, rid, outcome, &self.backend, &self.lifecycle);
                }
                DetokMsg::Fail { rid, message } => {
                    handle_fail(&mut table, &rid, message, &self.lifecycle)
                }
                DetokMsg::Deregister { rid } => {
                    table.remove(&rid);
                }
            }
        }
    }
}

/// The `RequestKind::Detokenize` backend stage: to-scheduler queued this rid's
/// `Register` just before on this same channel, so the entry exists — deliver
/// the decoded text (or the error) through the registered sink and drop it,
/// like a one-result control request. No scheduler abort on failure: this kind
/// never reaches the ring, so there is nothing to stop.
fn handle_decode(
    table: &mut HashMap<Rid, DetokState>,
    rid: &Rid,
    token_ids: &[u32],
    skip_special_tokens: bool,
    backend: &DetokenizerBackend,
) {
    if let Some(mut st) = table.remove(rid) {
        let item = match backend.decode_once(token_ids, skip_special_tokens) {
            Ok(text) => ResponseItem::Data(text.into()),
            Err(e) => ResponseItem::Error(e),
        };
        let _ = st.sink.try_send(item);
        st.fsm = RequestState::Completed;
    }
}

/// Complete a control only after every rank has replied. Cancellation removes
/// this same entry, so late replies cannot retain aggregation state.
fn handle_result(
    table: &mut HashMap<Rid, DetokState>,
    rid: Rid,
    dp_rank: Option<u32>,
    payload: bytes::Bytes,
    lifecycle: &flume::Sender<LifecycleEvent>,
) {
    let Some(st) = table.get_mut(&rid) else {
        return;
    };
    let rank = match dp_rank {
        Some(rank) if (rank as usize) < st.control_results.len() => rank as usize,
        None if st.control_results.len() == 1 => 0,
        _ => {
            handle_fail(
                table,
                &rid,
                "invalid control response rank".into(),
                lifecycle,
            );
            return;
        }
    };
    if st.control_results[rank].is_some() {
        return;
    }
    st.control_results[rank] = Some(payload);
    if st.control_results.iter().any(Option::is_none) {
        return;
    }
    if let Some(st) = table.remove(&rid) {
        let results = st.control_results.into_iter().flatten().collect();
        let _ = st.sink.try_send(ResponseItem::Control(results));
        let _ = lifecycle.send(LifecycleEvent::Finished(rid));
    }
}

/// Terminal per-request failure (bad request header): send an `Error` to the sink
/// (the api-server turns it into an HTTP 400) and drop the request.
/// Terminal per-request failure. `Internal` (500), not `Validation` (400): the
/// producers of this message are server faults — a malformed scheduler output
/// frame — not bad client input. Also aborts the request on the scheduler, which
/// otherwise keeps generating tokens for a connection that will never read them.
fn handle_fail(
    table: &mut HashMap<Rid, DetokState>,
    rid: &Rid,
    message: String,
    lifecycle: &flume::Sender<LifecycleEvent>,
) {
    if let Some(mut st) = table.remove(rid) {
        // Abort first: `try_send` on the sink can release the handler, which frees
        // the rid for reuse (same ordering hazard as the disconnect path).
        let _ = lifecycle.send(LifecycleEvent::DetokAbort(rid.clone()));
        let _ = st
            .sink
            .try_send(ResponseItem::Error(Error::Internal(message)));
        st.fsm = RequestState::Completed;
    }
}

fn handle_abort(
    table: &mut HashMap<Rid, DetokState>,
    rid: Rid,
    outcome: SchedulerAbort,
    backend: &DetokenizerBackend,
    lifecycle: &flume::Sender<LifecycleEvent>,
) {
    let Some(state) = table.get_mut(&rid) else {
        return;
    };
    // Python's AbortReq handler closes the request without collect_metrics;
    // only scheduler batch outputs contribute completion observations.
    state.metrics.take();
    let mut counts = state.counts.as_ref().clone();
    counts.speculative = None;
    if outcome.weight_version.is_some() {
        counts.weight_version = outcome.weight_version;
    }
    if outcome.weight_versions.is_some() {
        counts.weight_versions = outcome.weight_versions;
    }
    let event = ChunkEvent {
        rid,
        finish_reason: Some(outcome.finished_reason),
        prompt_tokens: state.prompt_tokens,
        counts: Arc::new(counts),
        ..Default::default()
    };
    handle_chunk(table, event, backend, lifecycle);
}

fn handle_chunk(
    table: &mut HashMap<Rid, DetokState>,
    mut ev: ChunkEvent,
    backend: &DetokenizerBackend,
    lifecycle: &flume::Sender<LifecycleEvent>,
) {
    // Copied once: `ev` is moved into the sink below, but the rid is still
    // needed to look the request up and to remove it.
    let rid = ev.rid.clone();

    let Some(st) = table.get_mut(&rid) else {
        // Late chunk after completion/abort — drop.
        return;
    };
    let decode_logprob_text = st.decode_logprob_text;
    let finished = ev.finish_reason.is_some();

    if finished && let Some(metadata) = st.response_metadata.take() {
        ev.extras
            .get_or_insert_with(Default::default)
            .metadata
            .fields
            .extend(metadata);
    }

    if let Some(ids) = &st.prompt_token_ids {
        ev.extras
            .get_or_insert_with(Default::default)
            .prompt_token_ids = Some(ids.clone());
    }

    if let Some(options) = st.logprobs {
        use crate::message::response::FlatTopLogprobs;
        let extras = ev.extras.get_or_insert_with(Default::default);
        extras.logprobs = Some(options);
        if options.flat && options.top_k > 0 {
            let flat = extras.flat_input_top_logprobs.take().or_else(|| {
                if !extras.in_top_lens.is_empty() {
                    FlatTopLogprobs::from_nested(
                        &extras.in_top_val,
                        &extras.in_top_idx,
                        &extras.in_top_lens,
                        options.top_k,
                    )
                } else {
                    matches!(st.fsm, RequestState::Queued)
                        .then(|| FlatTopLogprobs::empty(options.top_k))
                }
            });
            if let Some(flat) = flat {
                extras.in_top_val.clear();
                extras.in_top_idx.clear();
                extras.in_top_lens.clear();
                extras
                    .metadata
                    .fields
                    .extend(flat.into_fields(options.base64));
            }
        }
    }

    if let Some(extras) = ev.extras.as_deref_mut() {
        use base64::Engine;
        for (key, bytes) in [
            ("routed_experts", extras.routed_experts.take()),
            ("indexer_topk", extras.indexer_topk.take()),
        ] {
            if let Some(bytes) = bytes {
                extras.metadata.fields.insert(
                    key.into(),
                    base64::engine::general_purpose::STANDARD
                        .encode(bytes)
                        .into(),
                );
            }
        }
        if finished && let Some(beams) = &mut extras.beam_output {
            beams.scheduler_metadata.clone_from(&extras.metadata);
        }
    }

    if finished && let Some(timestamp) = st.dispatch_finished_ts {
        ev.extras
            .get_or_insert_with(Default::default)
            .metadata
            .fields
            .insert("api_server_dispatch_finish_ts".into(), timestamp.into());
    }
    if st.decoder.is_none() {
        let context = ev
            .extras
            .as_deref()
            .map_or(&[][..], |extras| extras.prompt_context.as_slice());
        st.decoder = backend.new_decoder(context, st.skip_special_tokens);
    }

    if let Some(processor) = st.response_processor.as_mut() {
        let extras = ev.extras.get_or_insert_with(Default::default);
        if let Err(message) = processor.process(&mut extras.metadata, finished) {
            handle_fail(table, &rid, message, lifecycle);
            return;
        }
    }

    // Queued → Streaming on the first chunk (the scheduler picked it).
    if matches!(st.fsm, RequestState::Queued) {
        let _ = st.fsm.apply(Event::SchedulerPicked);
    }

    let (n_tok, delta_text) = match decode_output(st, &mut ev, backend) {
        Ok(decoded) => decoded,
        Err(e) => {
            let _ = st.fsm.apply(Event::Error(e.clone()));
            let _ = lifecycle.send(LifecycleEvent::DetokAbort(rid.clone()));
            let _ = st.sink.try_send(ResponseItem::Error(e));
            table.remove(&rid);
            return;
        }
    };

    // Streaming → Streaming (finish:false) or Streaming → Finalizing (finish:true).
    let _ = st.fsm.apply(Event::Chunk { finish: finished });

    // `return_text_in_logprobs`: decode each logprob token id to text HERE (this
    // CPU-bound shard) rather than on the api-server I/O threads. Flat text columns
    // stay parallel to the `idx` buffers, so `sglang_frame` just reads them. Only the
    // logprob-carrying frames have an `extras` box; a plain token frame skips this.
    if decode_logprob_text && let Some(ex) = ev.extras.as_deref_mut() {
        ex.out_lp_txt = backend.decode_logprob_texts(&ex.out_lp_idx);
        ex.in_lp_txt = backend.decode_logprob_texts(&ex.in_lp_idx);
        ex.out_top_txt = backend.decode_logprob_texts(&ex.out_top_idx);
        ex.in_top_txt = backend.decode_logprob_texts(&ex.in_top_idx);
        ex.out_tid_txt = backend.decode_logprob_texts(&ex.out_tid_idx);
        ex.in_tid_txt = backend.decode_logprob_texts(&ex.in_tid_idx);
    }

    // Fill the decode outputs in place; the pre-decode columns (boxed logprobs/hidden,
    // token_ids, prompt_tokens, finish_reason) already ride in `ev`. The API handler
    // formats this delta (and accumulates for the cumulative view).
    ev.text = delta_text;
    ev.completion_tokens = n_tok;
    if let Some(metrics) = st.metrics.as_mut() {
        metrics.observe(&ev);
    }
    st.counts.clone_from(&ev.counts);
    st.prompt_tokens = ev.prompt_tokens;

    if finished {
        // The Done frame *is* the final frame: Finalizing → Completed.
        let sent = st.sink.try_send(ResponseItem::Done(ev)).is_ok();
        let _ = st.fsm.apply(if sent {
            Event::FinalFrameSent
        } else {
            Event::Disconnect
        });
        table.remove(&rid);
        let _ = lifecycle.send(LifecycleEvent::Finished(rid));
    } else {
        // Every intermediate chunk emits its delta frame. A failed send means the
        // client can't receive it — `Closed` (gone) or `Full` (backpressure: not
        // reading fast enough). Either way we can't buffer unboundedly, and
        // silently dropping the frame would truncate the response and still look
        // like success at EOS. So treat both as terminal: drop the request AND
        // abort scheduler work for it.
        if let Err(e) = st.sink.try_send(ResponseItem::Frame(ev)) {
            match e {
                SinkError::Full => {
                    tracing::warn!(
                        rid = %rid,
                        "detok: sink full; aborting (client backpressure)"
                    )
                }
                SinkError::Closed => {
                    tracing::debug!(rid = %rid, "detok: sink closed; aborting (client gone)")
                }
            }
            let _ = st.fsm.apply(Event::Disconnect);
            // Abort ONLY when the sink is full. `Closed` means the handler future is
            // already gone, so its `AbortGuard` has run: it aborted and released the
            // rid. A second abort from here is unordered with respect to that
            // release, so it lands after a resubmit of the same rid has registered
            // and deregisters the NEW request — the cross-wiring the rid registry
            // exists to prevent, reached through the one abort producer that
            // bypasses the guard's ordering.
            if matches!(e, SinkError::Full) {
                let _ = lifecycle.send(LifecycleEvent::DetokAbort(rid.clone()));
            }
            table.remove(&rid);
        }
    }
}

fn decode_output(
    state: &mut DetokState,
    output: &mut ChunkEvent,
    backend: &DetokenizerBackend,
) -> Result<(u64, String), Error> {
    let finished = output.finish_reason.is_some();
    if finished
        && let Some(beams) = output
            .extras
            .as_mut()
            .and_then(|extras| extras.beam_output.as_mut())
        && !beams.sequences.is_empty()
    {
        let total_tokens = beams
            .sequences
            .iter()
            .map(|beam| beam.token_ids.len() as u64)
            .sum();
        for beam in &mut beams.sequences {
            // Candidates have independent stop reasons and no prompt context.
            // Python preserves their original IDs while trimming decoded text.
            let matched = beam
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.matched())
                .cloned();
            let mut tokens = beam.token_ids.clone();
            trim_stop_token(&mut tokens, &matched, state.no_stop_trim);
            beam.text = backend
                .new_decoder(&[], state.skip_special_tokens)
                .map(|mut decoder| decoder.step(&tokens, true))
                .transpose()?;
        }
        let first = &beams.sequences[0];
        output.token_ids.clone_from(&first.token_ids);
        output.finish_reason.clone_from(&first.finish_reason);
        return Ok((total_tokens, first.text.clone().unwrap_or_default()));
    }

    let matched = finished
        .then(|| {
            output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.matched())
        })
        .flatten()
        .cloned();
    // Count generated tokens, including a matched stop token, before trimming.
    let tokens = output.token_ids.len() as u64;
    trim_stop_token(&mut output.token_ids, &matched, state.no_stop_trim);
    let text = match &mut state.decoder {
        Some(decoder) => decoder.step(&output.token_ids, finished)?,
        None => String::new(),
    };
    Ok((
        tokens,
        state
            .stop_buffer
            .step(text, finished, &matched, state.no_stop_trim),
    ))
}

/// Drop a matched stop TOKEN from the final chunk (Python `trim_matched_stop`,
/// token branch); `no_stop_trim` / non-token match keeps it.
fn trim_stop_token(token_ids: &mut TokenIds, matched: &Option<Matched>, no_stop_trim: bool) {
    // Token id 0 is NOT a match: Python guards with `if not matched`, and 0 is
    // falsy there, so it trims nothing. Trimming on 0 drops a real generated token
    // for any model whose stop id happens to be 0.
    if !no_stop_trim && matches!(matched, Some(Matched::Token(t)) if *t != 0) {
        token_ids.pop();
    }
}

/// Remove the matched stop string from the decoded final chunk (Python
/// `trim_matched_stop`, string branch); `no_stop_trim` keeps it. Truncates at
/// the FIRST occurrence.
fn trim_stop_str(text: &mut String, stop: &str, no_stop_trim: bool) {
    if stop.is_empty() {
        return;
    }
    if let Some(pos) = text.find(stop) {
        text.truncate(if no_stop_trim { pos + stop.len() } else { pos });
    }
}

#[derive(Default)]
struct StopBuffer {
    stops: Vec<String>,
    pending: String,
}

impl StopBuffer {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
        }
    }

    fn step(
        &mut self,
        delta: String,
        finished: bool,
        matched: &Option<Matched>,
        no_stop_trim: bool,
    ) -> String {
        let mut text = if self.pending.is_empty() {
            delta
        } else {
            let mut text = std::mem::take(&mut self.pending);
            text.push_str(&delta);
            text
        };
        if finished {
            if let Some(Matched::Str(stop)) = matched {
                trim_stop_str(&mut text, stop, no_stop_trim);
            }
            return text;
        }

        // Python trims the literal `matched` string, including regex patterns.
        // Retain an occurrence (and any speculative over-generation), or the
        // longest suffix that could complete one in the next chunk.
        let mut keep_from = text.len();
        for stop in self.stops.iter().filter(|stop| !stop.is_empty()) {
            if let Some(start) = text.find(stop) {
                keep_from = keep_from.min(start);
            } else {
                for (end, _) in stop.char_indices().skip(1) {
                    if text.ends_with(&stop[..end]) {
                        keep_from = keep_from.min(text.len() - end);
                    }
                }
            }
        }
        self.pending = text.split_off(keep_from);
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn routing_output_columns_match_python_base64_including_empty_tensors() {
        use crate::message::response::{BatchHeader, for_each_chunk, frame_decode_batch_cols};
        use base64::Engine;

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/routing_outputs_python.json"))
                .unwrap();
        let cases = fixture["outputs"].as_array().unwrap();
        let mut header = BatchHeader {
            rids: (0..cases.len()).map(|i| i.to_string()).collect(),
            prompt_tokens: vec![2; cases.len()],
            tok_lens: vec![1; cases.len()],
            finish_reasons: vec![
                Some(
                    serde_json::from_value(serde_json::json!({
                        "type": "length", "length": 1
                    }))
                    .unwrap()
                );
                cases.len()
            ],
            ..Default::default()
        };
        let mut data: Vec<u8> = (0..cases.len()).flat_map(|_| 3_i32.to_le_bytes()).collect();
        for (key, lengths) in [
            ("routed_experts", &mut header.routed_experts_bytes),
            ("indexer_topk", &mut header.indexer_topk_bytes),
        ] {
            for case in cases {
                let bytes = case["expected"][key].as_str().map(|value| {
                    base64::engine::general_purpose::STANDARD
                        .decode(value)
                        .unwrap()
                });
                lengths.push(bytes.as_ref().map(|bytes| bytes.len() as u32));
                data.extend(bytes.into_iter().flatten());
            }
        }
        let mut table = HashMap::new();
        let mut receivers = Vec::new();
        for rid in &header.rids {
            let (tx, rx) = mpsc::channel(1);
            receivers.push(rx);
            table.insert(
                rid.clone().into(),
                DetokState {
                    sink: ResponseSink::Local(tx),
                    decode_logprob_text: false,
                    skip_special_tokens: true,
                    no_stop_trim: false,
                    stop_buffer: StopBuffer::default(),
                    logprobs: None,
                    decoder: None,
                    counts: Default::default(),
                    prompt_tokens: 0,
                    prompt_token_ids: None,
                    dispatch_finished_ts: None,
                    response_metadata: None,
                    control_results: Vec::new(),
                    metrics: None,
                    response_processor: None,
                    fsm: RequestState::Queued,
                },
            );
        }
        let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
        let (lifecycle, _) = flume::unbounded();
        let result = for_each_chunk(&frame[1..], |event| {
            handle_chunk(&mut table, event, &DetokenizerBackend::Skip, &lifecycle)
        });
        assert!(result.rids.is_empty());
        for (case, rx) in cases.iter().zip(&mut receivers) {
            let ResponseItem::Done(event) = rx.try_recv().unwrap() else {
                panic!("expected final output");
            };
            assert_eq!(event.token_ids, vec![3]);
            let fields = event
                .extras
                .map_or_else(Default::default, |extras| extras.metadata.fields);
            assert_eq!(serde_json::Value::Object(fields), case["expected"]);
        }
        assert!(table.is_empty());

        // Bad lengths must reject the entire batch before any partial delivery.
        for truncated in [false, true] {
            let mut malformed: BatchHeader =
                rmp_serde::from_slice(&rmp_serde::to_vec(&header).unwrap()).unwrap();
            if truncated {
                malformed.routed_experts_bytes[0] = Some(u32::MAX);
            } else {
                malformed.indexer_topk_bytes.pop();
            }
            let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&malformed).unwrap(), &[&data]);
            let result = for_each_chunk(&frame[1..], |_| panic!("malformed frame was delivered"));
            assert_eq!(result.rids.len(), cases.len());
        }
    }

    #[test]
    fn stop_chunks_match_python_without_emitting_trimmed_prefixes() {
        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            stops: Vec<String>,
            chunks: Vec<String>,
            finish: crate::message::finish_reason::FinishReason,
            trimmed: String,
            kept: String,
        }
        struct FixtureDecoder(std::collections::VecDeque<String>);
        impl StreamDecoder for FixtureDecoder {
            fn step(&mut self, _: &[i32], _: bool) -> Result<String, Error> {
                Ok(self.0.pop_front().unwrap())
            }
        }
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../../testdata/stop_text_python.json")).unwrap();
        for case in cases {
            for keep in [false, true] {
                let (tx, mut rx) = mpsc::channel(1);
                let rid = Rid::from(case.name.clone());
                let mut table = HashMap::from([(
                    rid.clone(),
                    DetokState {
                        sink: ResponseSink::Local(tx),
                        decode_logprob_text: false,
                        skip_special_tokens: false,
                        no_stop_trim: keep,
                        stop_buffer: StopBuffer::new(case.stops.clone()),
                        decoder: Some(Box::new(FixtureDecoder(case.chunks.clone().into()))),
                        counts: Default::default(),
                        prompt_tokens: 0,
                        prompt_token_ids: None,
                        dispatch_finished_ts: None,
                        response_metadata: None,
                        control_results: Vec::new(),
                        metrics: None,
                        logprobs: None,
                        response_processor: None,
                        fsm: RequestState::Queued,
                    },
                )]);
                let expected = if keep { &case.kept } else { &case.trimmed };
                let mut text = String::new();
                let (lifecycle, _) = flume::unbounded();
                for (index, delta) in case.chunks.iter().enumerate() {
                    let finished = index + 1 == case.chunks.len();
                    let ids = if delta.is_empty() {
                        vec![]
                    } else {
                        vec![index as i32]
                    };
                    if finished && case.finish.abort_status().is_some() {
                        handle_abort(
                            &mut table,
                            rid.clone(),
                            SchedulerAbort {
                                finished_reason: case.finish.clone(),
                                weight_version: None,
                                weight_versions: None,
                            },
                            &DetokenizerBackend::Skip,
                            &lifecycle,
                        );
                    } else {
                        handle_chunk(
                            &mut table,
                            ChunkEvent {
                                rid: rid.clone(),
                                token_ids: ids.clone(),
                                finish_reason: finished.then(|| case.finish.clone()),
                                ..Default::default()
                            },
                            &DetokenizerBackend::Skip,
                            &lifecycle,
                        );
                    }
                    let item = rx.try_recv().unwrap();
                    assert_eq!(matches!(item, ResponseItem::Done(_)), finished);
                    let (ResponseItem::Frame(event) | ResponseItem::Done(event)) = item else {
                        panic!("unexpected response {item:?}");
                    };
                    assert_eq!(event.token_ids, ids, "{}", case.name);
                    assert_eq!(event.completion_tokens, ids.len() as u64);
                    text.push_str(&event.text);
                    assert!(
                        expected.starts_with(&text),
                        "{}: leaked {text:?}",
                        case.name
                    );
                }
                assert_eq!(&text, expected, "{}", case.name);
                assert!(table.is_empty());
            }
        }
    }

    #[test]
    fn flat_prompt_logprobs_match_python_raw_arrays_and_base64() {
        use crate::message::response::{
            BatchHeader, FlatTopLogprobShape, LogprobOptions, for_each_chunk,
            frame_decode_batch_cols,
        };
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/flat_logprobs_python.json")).unwrap();
        for case in cases["cases"].as_array().unwrap() {
            let shape: FlatTopLogprobShape = serde_json::from_value(case["shape"].clone()).unwrap();
            let values: Vec<f32> = serde_json::from_value(case["values"].clone()).unwrap();
            let indices: Vec<i32> = serde_json::from_value(case["indices"].clone()).unwrap();
            for base64 in [false, true] {
                for nested in [false, true] {
                    let mut header = BatchHeader {
                        rids: vec!["absent".into(), "flat".into()],
                        finish_reasons: vec![
                            Some(
                                serde_json::from_value(serde_json::json!({
                                    "type": "length", "length": 1
                                }))
                                .unwrap()
                            );
                            2
                        ],
                        prompt_tokens: vec![4; 2],
                        tok_lens: vec![1; 2],
                        flat_top_logprob_shapes: vec![None, Some(shape)],
                        ..Default::default()
                    };
                    if nested {
                        header.flat_top_logprob_shapes.clear();
                        header.in_top_reqlens = vec![0, shape.rows + shape.null_prefix];
                        header.in_top_poslens = std::iter::repeat_n(0, shape.null_prefix as usize)
                            .chain(std::iter::repeat_n(shape.top_k as u32, shape.rows as usize))
                            .collect();
                    }
                    let data: Vec<u8> = [8i32, 9]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .chain(values.iter().flat_map(|value| value.to_le_bytes()))
                        .chain(indices.iter().flat_map(|index| index.to_le_bytes()))
                        .collect();
                    let frame =
                        frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
                    let (lifecycle, _events) = flume::unbounded();
                    let mut table = HashMap::new();
                    let mut receivers = Vec::new();
                    for rid in &header.rids {
                        let (sink, receive) = mpsc::channel(1);
                        table.insert(
                            Rid::from(rid.clone()),
                            DetokState {
                                sink: ResponseSink::Local(sink),
                                decode_logprob_text: false,
                                skip_special_tokens: true,
                                no_stop_trim: false,
                                stop_buffer: StopBuffer::default(),
                                logprobs: (rid == "flat").then_some(LogprobOptions {
                                    top_k: shape.top_k,
                                    token_ids: false,
                                    flat: true,
                                    base64,
                                }),
                                decoder: None,
                                counts: Default::default(),
                                prompt_tokens: 0,
                                prompt_token_ids: None,
                                dispatch_finished_ts: None,
                                response_metadata: None,
                                control_results: Vec::new(),
                                metrics: None,
                                response_processor: None,
                                fsm: RequestState::Queued,
                            },
                        );
                        receivers.push(receive);
                    }
                    assert!(
                        for_each_chunk(&frame[1..], |event| {
                            handle_chunk(&mut table, event, &DetokenizerBackend::Skip, &lifecycle)
                        })
                        .ok
                    );
                    for (i, mut receive) in receivers.into_iter().enumerate() {
                        let Some(ResponseItem::Done(event)) = receive.blocking_recv() else {
                            panic!("no final event")
                        };
                        if i == 0 {
                            assert!(event.extras.is_none());
                        } else {
                            let extras = event.extras.unwrap();
                            assert!(extras.flat_input_top_logprobs.is_none());
                            assert_eq!(
                                serde_json::json!(extras.metadata.fields),
                                case[if base64 { "base64" } else { "json" }]
                            );
                        }
                    }
                    header.flat_top_logprob_shapes = vec![
                        None,
                        Some(FlatTopLogprobShape {
                            rows: u32::MAX,
                            top_k: u64::MAX,
                            null_prefix: 0,
                        }),
                    ];
                    let frame =
                        frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
                    assert!(
                        !for_each_chunk(&frame[1..], |_| panic!("invalid shape was routed")).ok
                    );
                }
            }
        }
    }

    /// A non-terminal chunk that can't be delivered (sink full → client
    /// backpressure) drops the request AND aborts scheduler work — it does not
    /// silently keep state, which would later read as a clean completion at EOS.
    #[test]
    fn full_sink_drops_request_and_aborts_scheduler() {
        // Capacity-1 sink, pre-filled so the next send hits `Full`.
        let (tx, _rx) = mpsc::channel::<ResponseItem>(1);
        tx.try_send(ResponseItem::Frame(ChunkEvent::default()))
            .unwrap();

        let mut table = HashMap::new();
        table.insert(
            Rid::from("1"),
            DetokState {
                sink: ResponseSink::Local(tx),
                decode_logprob_text: false,
                skip_special_tokens: true,
                no_stop_trim: false,
                stop_buffer: StopBuffer::default(),
                decoder: None,
                counts: Default::default(),
                prompt_tokens: 0,
                prompt_token_ids: None,
                dispatch_finished_ts: None,
                response_metadata: None,
                control_results: Vec::new(),
                metrics: None,
                logprobs: None,
                response_processor: None,
                fsm: RequestState::Queued,
            },
        );

        let (tm_tx, tm_rx) = flume::unbounded::<LifecycleEvent>();
        let ev = ChunkEvent {
            rid: Rid::from("1"),
            token_ids: vec![5],
            ..Default::default() // finish_reason None → non-terminal
        };
        handle_chunk(&mut table, ev, &DetokenizerBackend::Skip, &tm_tx);

        // Request removed (no lingering state to be mistaken for success)...
        assert!(!table.contains_key(&Rid::from("1")));
        // ...and the scheduler was told to abort it.
        assert!(matches!(
            tm_rx.try_recv(),
            Ok(LifecycleEvent::DetokAbort(rid)) if rid == Rid::from("1")
        ));
    }

    /// `trim_stop_str` reproduces the base's stop-string semantics: `stop: "3"` on
    /// output " 1, 2, 3" yields " 1, 2, " by default and " 1, 2, 3" with
    /// `no_stop_trim`.
    #[test]
    fn trim_stop_str_matches_base() {
        let mut t = " 1, 2, 3".to_string();
        trim_stop_str(&mut t, "3", false);
        assert_eq!(t, " 1, 2, ");

        let mut t = " 1, 2, 3".to_string();
        trim_stop_str(&mut t, "3", true);
        assert_eq!(t, " 1, 2, 3");

        // Empty / absent stop is a no-op.
        let mut t = "abc".to_string();
        trim_stop_str(&mut t, "", false);
        assert_eq!(t, "abc");

        // The stop can occur twice in the final chunk
        let mut t = "a STOP b STOP".to_string();
        trim_stop_str(&mut t, "STOP", false);
        assert_eq!(t, "a ");

        let mut t = "a STOP b STOP".to_string();
        trim_stop_str(&mut t, "STOP", true);
        assert_eq!(t, "a STOP");
    }

    #[test]
    fn decode_once_rejects_skip_mode() {
        let error = DetokenizerBackend::Skip
            .decode_once(&[1], true)
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("skip_tokenizer_init=True"));
    }

    /// A `Decode` job answers through the REGISTERED sink and consumes the
    /// entry.
    #[test]
    fn decode_answers_via_registered_sink_and_consumes_the_entry() {
        let (tx, mut rx) = mpsc::channel::<ResponseItem>(4);
        let mut table = HashMap::new();
        table.insert(
            Rid::from("d1"),
            DetokState {
                sink: ResponseSink::Local(tx),
                decode_logprob_text: false,
                skip_special_tokens: true,
                no_stop_trim: false,
                stop_buffer: StopBuffer::default(),
                decoder: None,
                counts: Default::default(),
                prompt_tokens: 0,
                prompt_token_ids: None,
                dispatch_finished_ts: None,
                response_metadata: None,
                control_results: Vec::new(),
                metrics: None,
                logprobs: None,
                response_processor: None,
                fsm: RequestState::Queued,
            },
        );

        handle_decode(
            &mut table,
            &Rid::from("d1"),
            &[1],
            true,
            &DetokenizerBackend::Skip,
        );

        let Ok(ResponseItem::Error(err)) = rx.try_recv() else {
            panic!("the decode error must reach the sink, not vanish");
        };
        assert!(matches!(err, Error::Validation(_)));
        assert!(!table.contains_key(&Rid::from("d1")), "entry consumed");

        // Unregistered rid (raced with an abort's Deregister): nothing to
        // answer to — must be a no-op, not a panic.
        handle_decode(
            &mut table,
            &Rid::from("d2"),
            &[1],
            true,
            &DetokenizerBackend::Skip,
        );
        assert!(rx.try_recv().is_err());
    }

    /// Two requests on the SAME shard keep separate entries. This is what a
    /// A shard-hash collision now degrades to: the hash partitions, the rid
    /// identifies. Keying the table by the hash made colliding rids one entry, so
    /// `Register` evicted the first client's sink and their tokens were written to
    /// the second client's connection. A single shard forces co-location
    /// deterministically, without needing to find a real 64-bit collision.
    #[test]
    fn co_located_requests_keep_their_own_sinks() {
        let (tx_a, mut rx_a) = mpsc::channel::<ResponseItem>(4);
        let (tx_b, mut rx_b) = mpsc::channel::<ResponseItem>(4);
        let mut table = HashMap::new();
        let state = |tx| DetokState {
            sink: ResponseSink::Local(tx),
            decode_logprob_text: false,
            skip_special_tokens: true,
            no_stop_trim: false,
            stop_buffer: StopBuffer::default(),
            decoder: None,
            counts: Default::default(),
            prompt_tokens: 0,
            prompt_token_ids: None,
            dispatch_finished_ts: None,
            response_metadata: None,
            control_results: Vec::new(),
            metrics: None,
            logprobs: None,
            response_processor: None,
            fsm: RequestState::Queued,
        };
        table.insert(Rid::from("alice"), state(tx_a));
        table.insert(Rid::from("bob"), state(tx_b));
        let (tm_tx, _tm_rx) = flume::unbounded::<LifecycleEvent>();

        let chunk = |rid: &str, id: i32| ChunkEvent {
            rid: Rid::from(rid.to_string()),
            token_ids: vec![id],
            ..Default::default()
        };
        handle_chunk(
            &mut table,
            chunk("alice", 11),
            &DetokenizerBackend::Skip,
            &tm_tx,
        );
        handle_chunk(
            &mut table,
            chunk("bob", 22),
            &DetokenizerBackend::Skip,
            &tm_tx,
        );

        let ids = |rx: &mut mpsc::Receiver<ResponseItem>| match rx.try_recv() {
            Ok(ResponseItem::Frame(ev)) => ev.token_ids,
            other => panic!("expected a frame, got {other:?}"),
        };
        assert_eq!(
            ids(&mut rx_a),
            vec![11],
            "alice must not receive bob's tokens"
        );
        assert_eq!(
            ids(&mut rx_b),
            vec![22],
            "bob must not receive alice's tokens"
        );
        assert_eq!(table.len(), 2, "neither registration evicted the other");
    }

    /// Drive a final (`finish_reason`) chunk through `handle_chunk` in skip mode and
    /// return the emitted `Done` event.
    fn final_chunk(
        no_stop_trim: bool,
        finish_reason: serde_json::Value,
        ids: Vec<i32>,
    ) -> ChunkEvent {
        let (tx, mut rx) = mpsc::channel::<ResponseItem>(4);
        let mut table = HashMap::new();
        table.insert(
            Rid::from("1"),
            DetokState {
                sink: ResponseSink::Local(tx),
                decode_logprob_text: false,
                skip_special_tokens: true,
                no_stop_trim,
                stop_buffer: StopBuffer::default(),
                decoder: None, // skip mode → output_ids passthrough
                counts: Default::default(),
                prompt_tokens: 0,
                prompt_token_ids: None,
                dispatch_finished_ts: None,
                response_metadata: None,
                control_results: Vec::new(),
                metrics: None,
                logprobs: None,
                response_processor: None,
                fsm: RequestState::Queued,
            },
        );
        let (tm_tx, _tm_rx) = flume::unbounded::<LifecycleEvent>();
        let ev = ChunkEvent {
            rid: Rid::from("1"),
            token_ids: ids,
            // Parsed from the wire map, so the trim paths are driven by the same
            // shape Python emits rather than a hand-built enum.
            finish_reason: Some(
                serde_json::from_value(finish_reason).expect("finish reason must parse"),
            ),
            ..Default::default()
        };
        handle_chunk(&mut table, ev, &DetokenizerBackend::Skip, &tm_tx);
        match rx.try_recv() {
            Ok(ResponseItem::Done(out)) => out,
            other => panic!("expected Done, got {other:?}"),
        }
    }

    /// Token id 0 is not a match: Python's `trim_matched_stop` guards with
    /// `if not matched`, and 0 is falsy there. Trimming on it drops a real
    /// generated token for any model whose stop id is 0.
    #[test]
    fn matched_token_zero_does_not_trim() {
        let mut ids = vec![1, 2, 0];
        trim_stop_token(&mut ids, &Some(Matched::Token(0)), false);
        assert_eq!(ids, vec![1, 2, 0], "id 0 is not a matched stop");
        // A real stop id still trims.
        let mut ids = vec![1, 2, 3];
        trim_stop_token(&mut ids, &Some(Matched::Token(3)), false);
        assert_eq!(ids, vec![1, 2]);
    }

    /// A matched stop TOKEN is dropped from the surfaced `output_ids` by default
    /// (but still counted in `completion_tokens`); `no_stop_trim` keeps it.
    #[test]
    fn stop_token_trimmed_from_output_ids() {
        let fr = serde_json::json!({ "type": "stop", "matched": 3 });
        let out = final_chunk(false, fr.clone(), vec![1, 2, 3]);
        assert_eq!(out.token_ids, vec![1, 2], "matched stop token dropped");
        assert_eq!(
            out.completion_tokens, 3,
            "generated count still includes it"
        );

        let out = final_chunk(true, fr, vec![1, 2, 3]);
        assert_eq!(out.token_ids, vec![1, 2, 3], "no_stop_trim keeps it");
    }

    /// A non-stop finish (`length`, no `matched`) never trims.
    #[test]
    fn length_finish_keeps_all_tokens() {
        let fr = serde_json::json!({ "type": "length", "length": 3 });
        let out = final_chunk(false, fr, vec![1, 2, 3]);
        assert_eq!(out.token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn response_processors_own_each_request_and_abort_on_invalid_output() {
        use crate::{HttpExtension, OutputMetadata, ResponseProcessor};
        use serde_json::json;

        #[derive(Debug, Default)]
        struct Sum(f64, Option<f64>);
        impl ResponseProcessor for Sum {
            fn prepare(&mut self, parameters: &crate::SamplingParams) {
                self.1 = Some(parameters.temperature);
            }
            fn process(
                &mut self,
                metadata: &mut OutputMetadata,
                _finished: bool,
            ) -> Result<(), String> {
                for value in metadata.customized_info.remove("value").unwrap_or_default() {
                    self.0 += value.as_f64().ok_or("invalid model output")?;
                }
                metadata.fields.insert("sum".into(), json!(self.0));
                metadata.fields.insert("temperature".into(), json!(self.1));
                Ok(())
            }
        }
        #[derive(Debug)]
        struct Extension;
        impl HttpExtension for Extension {
            fn apply(&self, router: axum::Router) -> axum::Router {
                router
            }
            fn new_response_processor(&self) -> Option<Box<dyn ResponseProcessor>> {
                Some(Box::<Sum>::default())
            }
        }
        let (input, receive) = flume::unbounded();
        let (lifecycle, events) = flume::unbounded();
        let worker = DetokenizerWorker::new(0, receive, DetokenizerBackend::Skip, lifecycle)
            .with_http_extension(Some(Arc::new(Extension)));
        let thread = std::thread::spawn(move || worker.run());
        let mut outputs = Vec::new();
        for rid in ["a", "b", "invalid"] {
            let (sink, receive) = mpsc::channel(8);
            input
                .send(DetokMsg::Register {
                    rid: rid.into(),
                    sink: ResponseSink::Local(sink),
                    decode_logprob_text: false,
                    skip_special_tokens: true,
                    no_stop_trim: false,
                    stop_texts: Vec::new(),
                    control_replies: 0,
                    metrics: None,
                    logprobs: None,
                })
                .unwrap();
            input
                .send(DetokMsg::Prepared {
                    rid: rid.into(),
                    prompt_token_ids: (rid == "a").then(|| Arc::from([2, 3, 4])),
                    dispatch_finished_ts: (rid == "a").then_some(1700000000.),
                    response_metadata: (rid == "a").then(|| {
                        serde_json::Map::from_iter([(
                            "media_stats".into(),
                            json!({"preprocess_e2e_ms": 17}),
                        )])
                    }),
                    sampling_params: Box::new(crate::SamplingParams {
                        temperature: if rid == "a" { 0.5 } else { 0.9 },
                        ..Default::default()
                    }),
                })
                .unwrap();
            outputs.push(receive);
        }
        for (rid, value, finished) in [
            ("a", json!(1), false),
            ("b", json!(10), false),
            ("a", json!(2), true),
            ("b", json!(20), true),
            ("invalid", json!("bad"), false),
        ] {
            input
                .send(DetokMsg::Chunks(vec![ChunkEvent {
                    rid: rid.into(),
                    token_ids: vec![11],
                    finish_reason: finished.then(|| {
                        serde_json::from_value(json!({"type": "length", "length": 2})).unwrap()
                    }),
                    extras: Some(Box::new(crate::message::response::ChunkExtras {
                        metadata: OutputMetadata {
                            customized_info: [("value".into(), vec![value])].into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    })),
                    ..Default::default()
                }]))
                .unwrap();
        }
        drop(input);
        thread.join().unwrap();
        for (index, expected) in [[1.0, 3.0], [10.0, 30.0]].into_iter().enumerate() {
            for (i, expected) in expected.into_iter().enumerate() {
                let item = outputs[index].try_recv().unwrap();
                assert_eq!(matches!(&item, ResponseItem::Done(_)), i == 1);
                let out = match item {
                    ResponseItem::Frame(out) | ResponseItem::Done(out) => out,
                    _ => panic!("generation output"),
                };
                let metadata = &out.extras.as_ref().unwrap().metadata;
                assert_eq!(
                    metadata.fields.get("api_server_dispatch_finish_ts"),
                    (index == 0 && i == 1).then_some(&serde_json::json!(1700000000.))
                );
                assert_eq!(metadata.fields["sum"], expected);
                assert_eq!(
                    metadata.fields.get("media_stats"),
                    (index == 0 && i == 1).then_some(&json!({"preprocess_e2e_ms": 17}))
                );
                assert_eq!(
                    metadata.fields["temperature"],
                    if index == 0 { 0.5 } else { 0.9 }
                );
                assert_eq!(
                    out.extras.as_ref().unwrap().prompt_token_ids.as_deref(),
                    (index == 0).then_some(&[2, 3, 4][..])
                );
                assert!(metadata.customized_info.is_empty());
            }
        }
        assert!(
            matches!(outputs[2].try_recv().unwrap(), ResponseItem::Error(Error::Internal(message)) if message == "invalid model output")
        );
        assert!(events.try_iter().any(
            |event| matches!(event, LifecycleEvent::DetokAbort(rid) if rid == Rid::from("invalid"))
        ));
    }

    #[test]
    fn incremental_decoding_matches_python_prompt_context_and_unicode_fixtures() {
        #[derive(serde::Deserialize)]
        struct Case {
            name: String,
            prompt_context: Vec<i32>,
            chunks: Vec<Vec<i32>>,
            skip_special_tokens: bool,
            expected: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct Fixture {
            name: String,
            tokenizer: serde_json::Value,
            vocab_size: u64,
            cases: Vec<Case>,
        }
        let fixtures: Vec<Fixture> =
            serde_json::from_str(include_str!("../../testdata/decoder_python.json")).unwrap();
        for fixture in fixtures {
            let path =
                std::env::temp_dir().join(format!("sglang-decoder-{}.json", uuid::Uuid::new_v4()));
            std::fs::write(&path, serde_json::to_vec(&fixture.tokenizer).unwrap()).unwrap();
            let tokenizer =
                dynamo_tokenizers::Tokenizer::from_file(path.to_str().unwrap()).unwrap();
            std::fs::remove_file(path).unwrap();
            let (input, receive) = flume::unbounded();
            let (lifecycle, events) = flume::unbounded();
            let worker = DetokenizerWorker::new(
                0,
                receive,
                DetokenizerBackend::Dynamo {
                    tokenizer,
                    vocab_size: Some(fixture.vocab_size),
                },
                lifecycle,
            );
            let thread = std::thread::spawn(move || worker.run());
            let mut outputs = Vec::new();
            for case in &fixture.cases {
                let (sink, receive) = mpsc::channel(case.chunks.len());
                input
                    .send(DetokMsg::Register {
                        rid: Rid::from(case.name.clone()),
                        sink: ResponseSink::Local(sink),
                        decode_logprob_text: false,
                        skip_special_tokens: case.skip_special_tokens,
                        no_stop_trim: false,
                        stop_texts: Vec::new(),
                        control_replies: 0,
                        metrics: None,
                        logprobs: None,
                    })
                    .unwrap();
                outputs.push(receive);
            }
            for step in 0..fixture
                .cases
                .iter()
                .map(|case| case.chunks.len())
                .max()
                .unwrap()
            {
                input.send(DetokMsg::Chunks(fixture.cases.iter().filter_map(|case| {
                    let tokens = case.chunks.get(step)?;
                    Some(ChunkEvent {
                        rid: Rid::from(case.name.clone()), token_ids: tokens.clone(),
                        finish_reason: (step + 1 == case.chunks.len()).then(|| serde_json::from_value(serde_json::json!({
                            "type": "length", "length": case.chunks.iter().map(Vec::len).sum::<usize>()
                        })).unwrap()),
                        extras: (step == 0).then(|| Box::new(crate::message::response::ChunkExtras {
                            prompt_context: case.prompt_context.clone(), ..Default::default()
                        })),
                        ..Default::default()
                    })
                }).collect())).unwrap();
            }
            drop(input);
            thread.join().unwrap();
            for (case, mut receive) in fixture.cases.iter().zip(outputs) {
                for (step, expected) in case.expected.iter().enumerate() {
                    let output = receive.try_recv().unwrap();
                    assert_eq!(
                        matches!(output, ResponseItem::Done(_)),
                        step + 1 == case.chunks.len()
                    );
                    let output = match output {
                        ResponseItem::Frame(output) | ResponseItem::Done(output) => output,
                        _ => panic!("expected generation output"),
                    };
                    assert_eq!(
                        &output.text, expected,
                        "{}: {} chunk {step}",
                        fixture.name, case.name
                    );
                    assert_eq!(output.token_ids, case.chunks[step]);
                    assert_eq!(output.completion_tokens, case.chunks[step].len() as u64);
                }
            }
            assert_eq!(
                events
                    .try_iter()
                    .filter(|event| matches!(event, LifecycleEvent::Finished(_)))
                    .count(),
                fixture.cases.len()
            );
        }
    }

    #[test]
    fn mixed_requests_preserve_special_tokens_and_stop_settings_with_real_tokenizer() {
        let backend = DetokenizerBackend::Dynamo {
            tokenizer: crate::tokenizer_manager::tokenizer::test_tokenizer().decoder(),
            vocab_size: Some(7),
        };
        let (input, receive) = flume::unbounded();
        let (lifecycle, _events) = flume::unbounded();
        let worker = DetokenizerWorker::new(0, receive, backend.clone(), lifecycle);
        let thread = std::thread::spawn(move || worker.run());
        let mut requests = Vec::new();
        for skip_special_tokens in [true, false] {
            for no_stop_trim in [true, false] {
                let rid = Rid::from(format!("{skip_special_tokens}-{no_stop_trim}"));
                let (sink, receive) = mpsc::channel(4);
                input
                    .send(DetokMsg::Register {
                        rid: rid.clone(),
                        sink: ResponseSink::Local(sink),
                        decode_logprob_text: false,
                        skip_special_tokens,
                        no_stop_trim,
                        stop_texts: Vec::new(),
                        control_replies: 0,
                        metrics: None,
                        logprobs: None,
                    })
                    .unwrap();
                requests.push((rid, skip_special_tokens, no_stop_trim, receive));
            }
        }
        for (tokens, finished) in [(vec![0, 3], false), (vec![4], false), (vec![1], true)] {
            input
                .send(DetokMsg::Chunks(
                    requests
                        .iter()
                        .map(|(rid, ..)| ChunkEvent {
                            rid: rid.clone(),
                            token_ids: tokens.clone(),
                            finish_reason: finished.then(|| {
                                serde_json::from_value(serde_json::json!({
                                    "type": "stop", "matched": 1
                                }))
                                .unwrap()
                            }),
                            ..Default::default()
                        })
                        .collect(),
                ))
                .unwrap();
        }
        drop(input);
        thread.join().unwrap();
        for (_, skip_special_tokens, no_stop_trim, mut receive) in requests {
            let mut text = String::new();
            let mut ids = Vec::new();
            let mut completion_tokens = 0;
            for index in 0..3 {
                let output = receive.try_recv().unwrap();
                assert_eq!(matches!(output, ResponseItem::Done(_)), index == 2);
                let output = match output {
                    ResponseItem::Frame(output) | ResponseItem::Done(output) => output,
                    _ => panic!("expected generation output"),
                };
                text.push_str(&output.text);
                ids.extend(output.token_ids);
                completion_tokens += output.completion_tokens;
            }
            let expected_ids = if no_stop_trim {
                vec![0, 3, 4, 1]
            } else {
                vec![0, 3, 4]
            };
            assert_eq!(ids, expected_ids);
            assert_eq!(completion_tokens, 4);
            assert_eq!(text.contains("<s>"), !skip_special_tokens);
            assert_eq!(text.ends_with("</s>"), !skip_special_tokens && no_stop_trim);
            assert_eq!(
                text,
                backend
                    .decode_once(
                        &ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                        skip_special_tokens,
                    )
                    .unwrap(),
            );
        }
    }
}
