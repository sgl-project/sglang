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
//! Real detokenization uses dynamo-tokenizers' `DecodeStream`, a stateful
//! incremental decoder (TGI/vLLM-style: it buffers partial UTF-8 / byte-fallback
//! tokens and only emits text once a valid boundary is reached). Each request
//! gets its own `DecodeStream`. When no tokenizer is configured (or
//! `skip_tokenizer_init` is set) the backend is `Skip`: no decoding, the raw
//! `output_ids` are emitted instead of text.
//!
//! Per-chunk response flow (no FSM state change inside Streaming):
//!   ChunkEvent{finish:None}  -> step ids -> delta -> Server frame
//!   ChunkEvent{finish:Some}  -> step ids -> delta -> final frame

use std::collections::HashMap;

use crate::message::detok::DetokMsg;
use crate::message::finish_reason::Matched;
use crate::message::ids::Rid;
use crate::message::response::{ChunkEvent, ResponseItem, ResponseSink, SinkError};
use crate::message::types::TokenIds;
use crate::tokenizer_manager::wiring::AbortSource;
use crate::utils::runtime::Runnable;
use crate::utils::{
    error::Error,
    fsm::{Event, RequestState},
};

/// Default for `skip_special_tokens` (SGLang's SamplingParams default). The
/// per-request value isn't available on the response yet; see the note in
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

    /// Decode one complete sequence without creating request-scoped streaming
    /// state. This runs on a pinned detokenizer worker, never on an API runtime
    /// thread.
    fn decode_once(&self, token_ids: &[u32]) -> Result<String, Error> {
        match self {
            DetokenizerBackend::Dynamo(tokenizer) => tokenizer
                .decode(token_ids, true)
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
    sink: ResponseSink,
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
    /// Unbounded abort lane, used to abort a request the shard had to drop
    /// (client backpressure) so the scheduler stops generating for it.
    abort: flume::Sender<AbortSource>,
}

impl DetokenizerWorker {
    pub fn new(
        shard: usize,
        rx: flume::Receiver<DetokMsg>,
        backend: DetokenizerBackend,
        abort: flume::Sender<AbortSource>,
    ) -> Self {
        Self {
            shard,
            rx,
            backend,
            abort,
        }
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
                    no_stop_trim,
                } => {
                    table.insert(
                        rid.clone(),
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
                // One decode step's chunks for this shard, batched by from-scheduler.
                DetokMsg::Chunks(evs) => {
                    for ev in evs {
                        handle_chunk(&mut table, ev, &self.backend, &self.abort);
                    }
                }
                DetokMsg::Decode { rid, token_ids } => {
                    handle_decode(&mut table, &rid, &token_ids, &self.backend)
                }
                DetokMsg::Result { rid, payload } => handle_result(&mut table, &rid, payload),
                DetokMsg::Fail { rid, message } => {
                    handle_fail(&mut table, &rid, message, &self.abort)
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
    backend: &DetokenizerBackend,
) {
    if let Some(mut st) = table.remove(rid) {
        let item = match backend.decode_once(token_ids) {
            Ok(text) => ResponseItem::Data(text.into()),
            Err(e) => ResponseItem::Error(e),
        };
        let _ = st.sink.try_send(item);
        st.fsm = RequestState::Completed;
    }
}

/// Control-request result: deliver the JSON payload to the sink verbatim as a
/// single `Done` frame — no detokenization, no streaming.
fn handle_result(table: &mut HashMap<Rid, DetokState>, rid: &Rid, payload: bytes::Bytes) {
    if let Some(mut st) = table.remove(rid) {
        let _ = st.sink.try_send(ResponseItem::Control(payload));
        // Response FSM: a control request goes straight to Completed (no Streaming
        // / Finalizing states — single response, never streamed).
        st.fsm = RequestState::Completed;
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
    abort: &flume::Sender<AbortSource>,
) {
    if let Some(mut st) = table.remove(rid) {
        // Abort first: `try_send` on the sink can release the handler, which frees
        // the rid for reuse (same ordering hazard as the disconnect path).
        let _ = abort.send(AbortSource::Detok(rid.clone()));
        let _ = st
            .sink
            .try_send(ResponseItem::Error(Error::Internal(message)));
        st.fsm = RequestState::Completed;
    }
}

fn handle_chunk(
    table: &mut HashMap<Rid, DetokState>,
    mut ev: ChunkEvent,
    backend: &DetokenizerBackend,
    abort: &flume::Sender<AbortSource>,
) {
    // Copied once: `ev` is moved into the sink below, but the rid is still
    // needed to look the request up and to remove it.
    let rid = ev.rid.clone();

    let Some(st) = table.get_mut(&rid) else {
        // Late chunk after completion/abort — drop.
        return;
    };
    let decode_logprob_text = st.decode_logprob_text;
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
        .then(|| ev.finish_reason.as_ref().and_then(|fr| fr.matched()))
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
                // Abort too: this is terminal for the request, and without it the
                // scheduler keeps generating for a connection that is already gone
                // — the other two terminal paths (disconnect, fail) both abort.
                let _ = st.fsm.apply(Event::Error(e.clone()));
                let _ = abort.send(AbortSource::Detok(rid.clone()));
                let _ = st.sink.try_send(ResponseItem::Error(e));
                table.remove(&rid);
                return;
            }
        },
        // skip_tokenizer_init: no decode; the token ids pass through in `ev`.
        None => String::new(),
    };
    // Stop STRING: trim it (and anything after) from the decoded delta's tail.
    if let Some(Matched::Str(stop)) = &matched {
        trim_stop_str(&mut delta_text, stop, no_stop_trim);
    }

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

    if finished {
        // The Done frame *is* the final frame: Finalizing → Completed.
        let sent = st.sink.try_send(ResponseItem::Done(ev)).is_ok();
        let _ = st.fsm.apply(if sent {
            Event::FinalFrameSent
        } else {
            Event::Disconnect
        });
        table.remove(&rid);
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
                let _ = abort.send(AbortSource::Detok(rid.clone()));
            }
            table.remove(&rid);
        }
    }
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
        let (tx, _rx) = mpsc::channel::<ResponseItem>(1);
        tx.try_send(ResponseItem::Frame(ChunkEvent::default()))
            .unwrap();

        let mut table = HashMap::new();
        table.insert(
            Rid::from("1"),
            DetokState {
                sink: ResponseSink::Local(tx),
                decode_logprob_text: false,
                no_stop_trim: false,
                decoder: None,
                fsm: RequestState::Queued,
            },
        );

        let (tm_tx, tm_rx) = flume::unbounded::<AbortSource>();
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
            Ok(AbortSource::Detok(rid)) if rid == Rid::from("1")
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
        let error = DetokenizerBackend::Skip.decode_once(&[1]).unwrap_err();
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
                no_stop_trim: false,
                decoder: None,
                fsm: RequestState::Queued,
            },
        );

        handle_decode(
            &mut table,
            &Rid::from("d1"),
            &[1],
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
            no_stop_trim: false,
            decoder: None,
            fsm: RequestState::Queued,
        };
        table.insert(Rid::from("alice"), state(tx_a));
        table.insert(Rid::from("bob"), state(tx_b));
        let (tm_tx, _tm_rx) = flume::unbounded::<AbortSource>();

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
                no_stop_trim,
                decoder: None, // skip mode → output_ids passthrough
                fsm: RequestState::Queued,
            },
        );
        let (tm_tx, _tm_rx) = flume::unbounded::<AbortSource>();
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
}
