//! Detokenizer shards — CPU-bound, one pinned thread per shard.
//!
//! Each shard owns a *local* `id -> DetokState` map. There is no lock: a given
//! `RequestId` is routed to exactly one shard (by `RequestId::shard`) for both
//! its `Register` and all its `Chunk`s, so the map has a single accessor.
//!
//! Real detokenization uses dynamo-tokenizers' `DecodeStream`, a stateful
//! incremental decoder (TGI/vLLM-style: it buffers partial UTF-8 / byte-fallback
//! tokens and only emits text once a valid boundary is reached). Each request
//! gets its own `DecodeStream`. When no tokenizer is configured (or
//! `skip_tokenizer_init` is set) the backend is `Skip`: no decoding, the raw
//! `output_ids` are emitted instead of text.
//!
//! Per-chunk egress flow (no FSM state change inside Streaming):
//!   ChunkEvent{finish:None}  -> step ids -> delta -> Server frame
//!   ChunkEvent{finish:Some}  -> step ids -> delta -> final frame

use std::collections::HashMap;

use crate::error::Error;
use crate::fsm::{Event, RequestState};
use crate::ids::RequestId;
use crate::message::{ChunkEvent, EgressItem, EgressSink, GenerationOutput};
use crate::runtime::Runnable;
use crate::runtime::channels::DetokMsg;

/// Default for `skip_special_tokens` (SGLang's SamplingParams default). The
/// per-request value isn't available on the egress side yet; see the note in
/// `DetokenizerBackend::new_decoder`.
const SKIP_SPECIAL_TOKENS: bool = true;

/// Per-request incremental decoder. `step` feeds the new token ids for one chunk
/// and returns the newly decoded text delta (empty if the ids only produced a
/// partial/incomplete multi-byte sequence that needs more tokens).
pub trait StreamDecoder: Send {
    fn step(&mut self, token_ids: &[i32]) -> Result<String, Error>;
}

/// Real decoder wrapping a dynamo-tokenizers `DecodeStream`.
struct DynamoDecoder {
    stream: dynamo_tokenizers::DecodeStream,
}

impl StreamDecoder for DynamoDecoder {
    fn step(&mut self, token_ids: &[i32]) -> Result<String, Error> {
        let mut out = String::new();
        for &id in token_ids {
            if let Some(chunk) = self
                .stream
                .step(id as u32)
                .map_err(|e| Error::Detokenize(e.to_string()))?
            {
                out.push_str(&chunk);
            }
        }
        Ok(out)
    }
}

/// Shard-wide detok backend. Cloned per shard; mints a fresh per-request decoder
/// on each `Register`.
#[derive(Clone)]
pub enum DetokenizerBackend {
    Dynamo(dynamo_tokenizers::Tokenizer),
    /// No decoding at all — the shard accumulates the raw output token ids and
    /// emits them as `output_ids` (no `DecodeStream`). Used for
    /// `skip_tokenizer_init` and when no tokenizer is configured.
    Skip,
}

impl DetokenizerBackend {
    /// Mint a per-request decoder, or `None` in skip mode (the shard passes the
    /// token ids through untouched instead of decoding text).
    fn new_decoder(&self) -> Option<Box<dyn StreamDecoder>> {
        match self {
            // NOTE: the stream is seeded with an empty prompt context, which is
            // correct for the common case. Seeding with the prompt's trailing
            // tokens (for perfect first-token spacing) would require Register to
            // carry input_ids — deferred.
            DetokenizerBackend::Dynamo(t) => Some(Box::new(DynamoDecoder {
                stream: t.decode_stream(&[], SKIP_SPECIAL_TOKENS),
            })),
            DetokenizerBackend::Skip => None,
        }
    }
}

struct DetokState {
    sink: EgressSink,
    stream: bool,
    /// Cumulative decoded text (SGLang stream frames carry cumulative text).
    text: String,
    /// Cumulative output token ids, emitted as `output_ids` in
    /// `skip_tokenizer_init` mode; stays empty when a decoder is present.
    output_ids: Vec<i32>,
    /// Cumulative output token count, reported as `meta_info.completion_tokens`
    /// (clients like bench_serving diff successive frames to get per-step tokens).
    completion_tokens: u64,
    /// Per-request incremental decoder; `None` in `skip_tokenizer_init` mode.
    decoder: Option<Box<dyn StreamDecoder>>,
    /// Egress half of the lifecycle FSM. Lives here because the ingress
    /// `Request` (and its FSM) was handed to the scheduler when queued; the
    /// shard is the sole owner of the request's egress state, so no lock.
    fsm: RequestState,
}

/// One detokenizer shard: owns a *local* `id -> DetokState` map (single
/// accessor, no lock) and the egress backend. Spawned (pinned) per shard as a
/// [`Runnable`]; a given `RequestId` is routed to exactly one shard.
pub struct DetokenizerWorker {
    shard: usize,
    rx: flume::Receiver<DetokMsg>,
    backend: DetokenizerBackend,
}

impl DetokenizerWorker {
    pub fn new(shard: usize, rx: flume::Receiver<DetokMsg>, backend: DetokenizerBackend) -> Self {
        Self { shard, rx, backend }
    }
}

impl Runnable for DetokenizerWorker {
    fn run(self) {
        let mut table: HashMap<RequestId, DetokState> = HashMap::new();
        tracing::debug!(shard = self.shard, "detokenizer worker started");

        while let Ok(msg) = self.rx.recv() {
            match msg {
                DetokMsg::Register { id, sink, stream } => {
                    table.insert(
                        id,
                        DetokState {
                            sink,
                            stream,
                            text: String::new(),
                            output_ids: Vec::new(),
                            completion_tokens: 0,
                            decoder: self.backend.new_decoder(),
                            // Registered == handed to the scheduler == Queued.
                            fsm: RequestState::Queued,
                        },
                    );
                }
                DetokMsg::Chunk(ev) => handle_chunk(&mut table, ev),
                DetokMsg::Result { id, payload } => handle_result(&mut table, id, payload),
            }
        }
    }
}

/// Control-request result: deliver the JSON payload to the sink verbatim as a
/// single `Done` frame — no detokenization, no streaming.
fn handle_result(table: &mut HashMap<RequestId, DetokState>, id: RequestId, payload: bytes::Bytes) {
    if let Some(mut st) = table.remove(&id) {
        let _ = st.sink.try_send(EgressItem::Control(payload));
        // Egress FSM: a control request goes straight to Completed (no Streaming
        // / Finalizing states — single response, never streamed).
        st.fsm = RequestState::Completed;
    }
}

fn handle_chunk(table: &mut HashMap<RequestId, DetokState>, ev: ChunkEvent) {
    let id = match ev.rid.parse::<u64>() {
        Ok(v) => RequestId(v),
        Err(_) => {
            tracing::warn!(rid = %ev.rid, "detok: unparsable rid");
            return;
        }
    };

    let Some(st) = table.get_mut(&id) else {
        // Late chunk after completion/abort — drop.
        return;
    };

    // Queued → Streaming on the first chunk (the scheduler picked it).
    if matches!(st.fsm, RequestState::Queued) {
        let _ = st.fsm.apply(Event::SchedulerPicked);
    }

    st.completion_tokens += ev.token_ids.len() as u64;
    match &mut st.decoder {
        // Normal path: incrementally decode to cumulative text.
        Some(decoder) => match decoder.step(&ev.token_ids) {
            Ok(delta) => st.text.push_str(&delta),
            Err(e) => {
                let _ = st.fsm.apply(Event::Error(e.clone()));
                let _ = st.sink.try_send(EgressItem::Error(e));
                table.remove(&id);
                return;
            }
        },
        // skip_tokenizer_init: pass token ids through, no decode.
        None => st.output_ids.extend_from_slice(&ev.token_ids),
    }

    let finished = ev.finish_reason.is_some();
    // Streaming → Streaming (finish:false) or Streaming → Finalizing (finish:true).
    let _ = st.fsm.apply(Event::Chunk { finish: finished });

    // Protocol-neutral cumulative snapshot; the API handler formats it. `text`
    // and `output_ids` are cumulative so we clone (the entry persists across
    // streamed frames); `rid`/`finish_reason` are moved (this `ev` is done).
    let output = GenerationOutput {
        rid: ev.rid,
        text: st.text.clone(),
        output_ids: st.output_ids.clone(),
        prompt_tokens: ev.prompt_tokens,
        completion_tokens: st.completion_tokens,
        finish_reason: ev.finish_reason,
    };

    if finished {
        // The Done frame *is* the final frame: Finalizing → Completed.
        let sent = st.sink.try_send(EgressItem::Done(output)).is_ok();
        let _ = st.fsm.apply(if sent {
            Event::FinalFrameSent
        } else {
            Event::Disconnect
        });
        table.remove(&id);
    } else if st.stream {
        // Only emit intermediate frames for streaming requests; unary requests
        // just accumulate until the final Done. A failed send == client gone.
        if st.sink.try_send(EgressItem::Frame(output)).is_err() {
            let _ = st.fsm.apply(Event::Disconnect);
            table.remove(&id);
            // TODO(abort): signal the scheduler to abort this rid.
        }
    }
}
