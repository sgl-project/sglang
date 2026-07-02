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
    /// No decoding at all — the shard emits each chunk's raw output token ids as
    /// `output_ids` (no `DecodeStream`, no accumulation). Used for
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

    /// Decode each logprob token id to its own text (one id at a time, matching
    /// Python's `batch_decode([[id] for id in ids])`). Runs on this CPU-bound
    /// shard, not the api-server I/O threads. `Skip` mode (no tokenizer) yields
    /// no text, so the `[logprob, token_id, text]` tuple's text slot stays null.
    fn decode_logprob_texts(&self, idxs: &[i32]) -> Vec<String> {
        match self {
            DetokenizerBackend::Dynamo(t) => idxs
                .iter()
                .map(|&id| {
                    t.decode(&[id as u32], false)
                        .map(String::from)
                        .unwrap_or_default()
                })
                .collect(),
            DetokenizerBackend::Skip => Vec::new(),
        }
    }
}

struct DetokState {
    sink: EgressSink,
    /// `return_text_in_logprobs`: whether to decode this request's logprob token
    /// ids to text (in this shard) for the `[logprob, token_id, text]` tuples.
    decode_logprob_text: bool,
    /// Per-request incremental decoder; `None` in `skip_tokenizer_init` mode.
    /// This is the *only* per-request accumulation the shard keeps: the decoder's
    /// internal byte/UTF-8 buffer. Decoded **text deltas** are emitted per chunk
    /// (no cumulative buffer here) — the api-server's drain loop reassembles the
    /// cumulative view where a consumer needs it (every unary response and the
    /// cumulative SGLang `/generate` stream); OpenAI streaming forwards deltas.
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
                DetokMsg::Register {
                    id,
                    sink,
                    decode_logprob_text,
                } => {
                    table.insert(
                        id,
                        DetokState {
                            sink,
                            decode_logprob_text,
                            decoder: self.backend.new_decoder(),
                            // Registered == handed to the scheduler == Queued.
                            fsm: RequestState::Queued,
                        },
                    );
                }
                DetokMsg::Chunk(ev) => handle_chunk(&mut table, ev, &self.backend),
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

fn handle_chunk(
    table: &mut HashMap<RequestId, DetokState>,
    mut ev: ChunkEvent,
    backend: &DetokenizerBackend,
) {
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
    let decode_lp_text = st.decode_logprob_text;

    // Queued → Streaming on the first chunk (the scheduler picked it).
    if matches!(st.fsm, RequestState::Queued) {
        let _ = st.fsm.apply(Event::SchedulerPicked);
    }

    // Fully incremental: decode just this chunk's delta (or, in skip mode, take
    // its raw ids). Nothing cumulative is kept here — the api-server's drain loop
    // reassembles it where needed.
    let n_tok = ev.token_ids.len() as u64;
    let (delta_text, delta_ids) = match &mut st.decoder {
        Some(decoder) => match decoder.step(&ev.token_ids) {
            Ok(delta) => (delta, Vec::new()),
            Err(e) => {
                let _ = st.fsm.apply(Event::Error(e.clone()));
                let _ = st.sink.try_send(EgressItem::Error(e));
                table.remove(&id);
                return;
            }
        },
        // skip_tokenizer_init: pass the chunk's token ids through, no decode.
        None => (String::new(), std::mem::take(&mut ev.token_ids)),
    };

    let finished = ev.finish_reason.is_some();
    // Streaming → Streaming (finish:false) or Streaming → Finalizing (finish:true).
    let _ = st.fsm.apply(Event::Chunk { finish: finished });

    // `return_text_in_logprobs`: decode each logprob token id to text HERE (this
    // CPU-bound shard) rather than on the api-server I/O threads. Flat text
    // columns stay parallel to the `idx` buffers, so `sglang_frame` just reads
    // them. Read `ev.*_idx` before the `mem::take`s below move them.
    let (out_lp_txt, in_lp_txt, out_top_txt, in_top_txt, out_tid_txt, in_tid_txt) =
        if decode_lp_text {
            (
                backend.decode_logprob_texts(&ev.out_lp_idx),
                backend.decode_logprob_texts(&ev.in_lp_idx),
                backend.decode_logprob_texts(&ev.out_top_idx),
                backend.decode_logprob_texts(&ev.in_top_idx),
                backend.decode_logprob_texts(&ev.out_tid_idx),
                backend.decode_logprob_texts(&ev.in_tid_idx),
            )
        } else {
            Default::default()
        };

    // Protocol-neutral **delta** snapshot for this chunk; the API handler formats
    // it (and accumulates when a cumulative view is required). `text`/`output_ids`
    // are this chunk's delta; `completion_tokens` is this chunk's token count.
    let output = GenerationOutput {
        rid: ev.rid,
        text: delta_text,
        output_ids: delta_ids,
        prompt_tokens: ev.prompt_tokens,
        completion_tokens: n_tok,
        finish_reason: ev.finish_reason,
        out_lp_val: std::mem::take(&mut ev.out_lp_val),
        out_lp_idx: std::mem::take(&mut ev.out_lp_idx),
        in_lp_val: std::mem::take(&mut ev.in_lp_val),
        in_lp_idx: std::mem::take(&mut ev.in_lp_idx),
        out_top_val: std::mem::take(&mut ev.out_top_val),
        out_top_idx: std::mem::take(&mut ev.out_top_idx),
        out_top_lens: std::mem::take(&mut ev.out_top_lens),
        in_top_val: std::mem::take(&mut ev.in_top_val),
        in_top_idx: std::mem::take(&mut ev.in_top_idx),
        in_top_lens: std::mem::take(&mut ev.in_top_lens),
        out_tid_val: std::mem::take(&mut ev.out_tid_val),
        out_tid_idx: std::mem::take(&mut ev.out_tid_idx),
        out_tid_lens: std::mem::take(&mut ev.out_tid_lens),
        in_tid_val: std::mem::take(&mut ev.in_tid_val),
        in_tid_idx: std::mem::take(&mut ev.in_tid_idx),
        in_tid_lens: std::mem::take(&mut ev.in_tid_lens),
        hidden_val: std::mem::take(&mut ev.hidden_val),
        hidden_lens: std::mem::take(&mut ev.hidden_lens),
        out_lp_txt,
        in_lp_txt,
        out_top_txt,
        in_top_txt,
        out_tid_txt,
        in_tid_txt,
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
    } else {
        // Every intermediate chunk emits its delta frame (fully-delta egress; the
        // egress shards exist to carry exactly this per-token fan-out at high
        // concurrency). A failed send == client gone.
        if st.sink.try_send(EgressItem::Frame(output)).is_err() {
            let _ = st.fsm.apply(Event::Disconnect);
            table.remove(&id);
            // TODO(abort): signal the scheduler to abort this rid.
        }
    }
}
