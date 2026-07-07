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
use crate::message::{ChunkEvent, EgressItem, EgressSink, SinkError};
use crate::runtime::Runnable;
use crate::runtime::channels::{DetokMsg, TmEvent};

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
    /// `SamplingParams.no_stop_trim`: keep the matched stop in the output. Default
    /// (`false`) trims it off the final chunk (see [`trim_stop_str`]).
    no_stop_trim: bool,
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
    /// TokenizerManager inbox, used to abort a request the shard had to drop
    /// (client backpressure/disconnect) so the scheduler stops generating for it.
    tm: flume::Sender<TmEvent>,
}

impl DetokenizerWorker {
    pub fn new(
        shard: usize,
        rx: flume::Receiver<DetokMsg>,
        backend: DetokenizerBackend,
        tm: flume::Sender<TmEvent>,
    ) -> Self {
        Self {
            shard,
            rx,
            backend,
            tm,
        }
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
                    no_stop_trim,
                } => {
                    table.insert(
                        id,
                        DetokState {
                            sink,
                            decode_logprob_text,
                            no_stop_trim,
                            decoder: self.backend.new_decoder(),
                            // Registered == handed to the scheduler == Queued.
                            fsm: RequestState::Queued,
                        },
                    );
                }
                DetokMsg::Chunk(ev) => handle_chunk(&mut table, ev, &self.backend, &self.tm),
                DetokMsg::Result { id, payload } => handle_result(&mut table, id, payload),
                DetokMsg::Fail { id, message } => handle_fail(&mut table, id, message),
                DetokMsg::Deregister { id } => {
                    table.remove(&id);
                }
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

/// Terminal per-request failure (bad request header): send an `Error` to the sink
/// (the api-server turns it into an HTTP 400) and drop the request.
fn handle_fail(table: &mut HashMap<RequestId, DetokState>, id: RequestId, message: String) {
    if let Some(mut st) = table.remove(&id) {
        let _ = st
            .sink
            .try_send(EgressItem::Error(Error::Validation(message)));
        st.fsm = RequestState::Completed;
    }
}

fn handle_chunk(
    table: &mut HashMap<RequestId, DetokState>,
    mut ev: ChunkEvent,
    backend: &DetokenizerBackend,
    tm: &flume::Sender<TmEvent>,
) {
    let id = RequestId(ev.rid); // raw u64 from the wire — no parse

    let Some(st) = table.get_mut(&id) else {
        // Late chunk after completion/abort — drop.
        return;
    };
    let decode_lp_text = st.decode_logprob_text;
    let no_stop_trim = st.no_stop_trim;

    // Queued → Streaming on the first chunk (the scheduler picked it).
    if matches!(st.fsm, RequestState::Queued) {
        let _ = st.fsm.apply(Event::SchedulerPicked);
    }

    let finished = ev.finish_reason.is_some();
    // Matched-stop trim (Python `trim_matched_stop`): the final chunk's finish
    // reason names the stop it matched — a stop STRING or a stop TOKEN id. By
    // default that stop is removed from the output; `no_stop_trim` keeps it.
    let matched = finished
        .then(|| ev.finish_reason.as_ref().and_then(|fr| fr.get("matched")))
        .flatten()
        .cloned();
    // Count generated tokens (incl. a matched stop token) *before* trimming.
    let n_tok = ev.token_ids.len() as u64;
    // Stop TOKEN: drop it before decode, so it reaches neither `text` nor `output_ids`.
    trim_stop_token(&mut ev.token_ids, &matched, no_stop_trim);

    // Fully incremental: decode just this chunk's delta. `token_ids` stays in the
    // event — it's ALSO surfaced as the `/generate` response's `output_ids` (the
    // Python server returns them by default alongside `text`), in both normal and
    // `skip_tokenizer_init` mode. Nothing cumulative is kept here — the api-server's
    // drain loop reassembles it where needed.
    let mut delta_text = match &mut st.decoder {
        Some(decoder) => match decoder.step(&ev.token_ids) {
            Ok(delta) => delta,
            Err(e) => {
                let _ = st.fsm.apply(Event::Error(e.clone()));
                let _ = st.sink.try_send(EgressItem::Error(e));
                table.remove(&id);
                return;
            }
        },
        // skip_tokenizer_init: no decode; the token ids pass through in `ev`.
        None => String::new(),
    };
    // Stop STRING: trim it (and anything after) from the decoded delta's tail.
    if let Some(serde_json::Value::String(stop)) = &matched {
        trim_stop_str(&mut delta_text, stop, no_stop_trim);
    }

    // Streaming → Streaming (finish:false) or Streaming → Finalizing (finish:true).
    let _ = st.fsm.apply(Event::Chunk { finish: finished });

    // `return_text_in_logprobs`: decode each logprob token id to text HERE (this
    // CPU-bound shard) rather than on the api-server I/O threads. Flat text columns
    // stay parallel to the `idx` buffers, so `sglang_frame` just reads them.
    if decode_lp_text {
        ev.out_lp_txt = backend.decode_logprob_texts(&ev.out_lp_idx);
        ev.in_lp_txt = backend.decode_logprob_texts(&ev.in_lp_idx);
        ev.out_top_txt = backend.decode_logprob_texts(&ev.out_top_idx);
        ev.in_top_txt = backend.decode_logprob_texts(&ev.in_top_idx);
        ev.out_tid_txt = backend.decode_logprob_texts(&ev.out_tid_idx);
        ev.in_tid_txt = backend.decode_logprob_texts(&ev.in_tid_idx);
    }

    // Fill the decode outputs in place; the pre-decode columns (logprobs/hidden,
    // token_ids, prompt_tokens, finish_reason) already ride in `ev`. The API
    // handler formats this delta (and accumulates for the cumulative view).
    ev.text = delta_text;
    ev.completion_tokens = n_tok;

    if finished {
        // The Done frame *is* the final frame: Finalizing → Completed.
        let sent = st.sink.try_send(EgressItem::Done(ev)).is_ok();
        let _ = st.fsm.apply(if sent {
            Event::FinalFrameSent
        } else {
            Event::Disconnect
        });
        table.remove(&id);
    } else {
        // Every intermediate chunk emits its delta frame. A failed send means the
        // client can't receive it — `Closed` (gone) or `Full` (backpressure: not
        // reading fast enough). Either way we can't buffer unboundedly, and
        // silently dropping the frame would truncate the response and still look
        // like success at EOS. So treat both as terminal: drop the request AND
        // abort scheduler work for it.
        if let Err(e) = st.sink.try_send(EgressItem::Frame(ev)) {
            match e {
                SinkError::Full => {
                    tracing::warn!(
                        rid = id.0,
                        "detok: sink full; aborting (client backpressure)"
                    )
                }
                SinkError::Closed => {
                    tracing::debug!(rid = id.0, "detok: sink closed; aborting (client gone)")
                }
            }
            let _ = st.fsm.apply(Event::Disconnect);
            table.remove(&id);
            // Best-effort: stop the scheduler generating for a request no client
            // can receive. A full/closed tm inbox just drops it (shutdown).
            let _ = tm.try_send(TmEvent::Abort(id));
        }
    }
}

/// Drop a matched stop TOKEN from the final chunk (Python `trim_matched_stop`,
/// token branch); `no_stop_trim` / non-token match keeps it.
fn trim_stop_token(
    token_ids: &mut Vec<i32>,
    matched: &Option<serde_json::Value>,
    no_stop_trim: bool,
) {
    if !no_stop_trim && matches!(matched, Some(serde_json::Value::Number(_))) {
        token_ids.pop();
    }
}

/// Remove the matched stop string from the decoded final chunk (Python
/// `trim_matched_stop`, string branch); `no_stop_trim` keeps it.
fn trim_stop_str(text: &mut String, stop: &str, no_stop_trim: bool) {
    if stop.is_empty() {
        return;
    }
    if let Some(pos) = text.rfind(stop) {
        text.truncate(if no_stop_trim { pos + stop.len() } else { pos });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    /// A non-terminal chunk that can't be delivered (sink full → client
    /// backpressure) drops the request AND aborts scheduler work — it does not
    /// silently keep state, which would later read as a clean completion at EOS.
    #[test]
    fn full_sink_drops_request_and_aborts_scheduler() {
        // Capacity-1 sink, pre-filled so the next send hits `Full`.
        let (tx, _rx) = mpsc::channel::<EgressItem>(1);
        tx.try_send(EgressItem::Frame(ChunkEvent::default()))
            .unwrap();

        let mut table = HashMap::new();
        table.insert(
            RequestId(1),
            DetokState {
                sink: EgressSink::Local(tx),
                decode_logprob_text: false,
                no_stop_trim: false,
                decoder: None,
                fsm: RequestState::Queued,
            },
        );

        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
        let ev = ChunkEvent {
            rid: 1,
            token_ids: vec![5],
            ..Default::default() // finish_reason None → non-terminal
        };
        handle_chunk(&mut table, ev, &DetokenizerBackend::Skip, &tm_tx);

        // Request removed (no lingering state to be mistaken for success)...
        assert!(table.get(&RequestId(1)).is_none());
        // ...and the scheduler was told to abort it.
        assert!(matches!(
            tm_rx.try_recv(),
            Ok(TmEvent::Abort(id)) if id == RequestId(1)
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
    }

    /// Drive a final (`finish_reason`) chunk through `handle_chunk` in skip mode and
    /// return the emitted `Done` event.
    fn final_chunk(
        no_stop_trim: bool,
        finish_reason: serde_json::Value,
        ids: Vec<i32>,
    ) -> ChunkEvent {
        let (tx, mut rx) = mpsc::channel::<EgressItem>(4);
        let mut table = HashMap::new();
        table.insert(
            RequestId(1),
            DetokState {
                sink: EgressSink::Local(tx),
                decode_logprob_text: false,
                no_stop_trim,
                decoder: None, // skip mode → output_ids passthrough
                fsm: RequestState::Queued,
            },
        );
        let (tm_tx, _tm_rx) = flume::unbounded::<TmEvent>();
        let ev = ChunkEvent {
            rid: 1,
            token_ids: ids,
            finish_reason: Some(finish_reason),
            ..Default::default()
        };
        handle_chunk(&mut table, ev, &DetokenizerBackend::Skip, &tm_tx);
        match rx.try_recv() {
            Ok(EgressItem::Done(out)) => out,
            other => panic!("expected Done, got {other:?}"),
        }
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
}
