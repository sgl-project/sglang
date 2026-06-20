//! Detokenizer shards — CPU-bound, one pinned thread per shard.
//!
//! Each shard owns a *local* `id -> DetokState` map. There is no lock: a given
//! `RequestId` is routed to exactly one shard (by `RequestId::shard`) for both
//! its `Register` and all its `Chunk`s, so the map has a single accessor.
//!
//! Real detokenization uses dynamo-tokenizers' `DecodeStream`, a stateful
//! incremental decoder (TGI/vLLM-style: it buffers partial UTF-8 / byte-fallback
//! tokens and only emits text once a valid boundary is reached). Each request
//! gets its own `DecodeStream`. `StubDecoder` (bytes) is the fallback when no
//! tokenizer is configured.
//!
//! Per-chunk egress flow (no FSM state change inside Streaming):
//!   ChunkEvent{finish:None}  -> step ids -> delta -> Server frame
//!   ChunkEvent{finish:Some}  -> step ids -> delta -> final frame

use std::collections::HashMap;

use bytes::Bytes;

use crate::error::Error;
use crate::fsm::{Event, RequestState};
use crate::ids::RequestId;
use crate::message::{ChunkEvent, EgressItem, EgressSink};
use crate::runtime::channels::DetokMsg;

/// Default for `skip_special_tokens` (SGLang's SamplingParams default). The
/// per-request value isn't available on the egress side yet; see the note in
/// `DetokBackend::new_stream`.
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

/// Byte fallback: inverse of `tokenizer::StubTokenizer`.
struct StubDecoder;

impl StreamDecoder for StubDecoder {
    fn step(&mut self, token_ids: &[i32]) -> Result<String, Error> {
        let bytes: Vec<u8> = token_ids.iter().map(|&id| id as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Shard-wide detok backend. Cloned per shard; mints a fresh per-request decoder
/// on each `Register`.
#[derive(Clone)]
pub enum DetokBackend {
    Dynamo(dynamo_tokenizers::Tokenizer),
    Stub,
}

impl DetokBackend {
    fn new_stream(&self) -> Box<dyn StreamDecoder> {
        match self {
            // NOTE: the stream is seeded with an empty prompt context, which is
            // correct for the common case. Seeding with the prompt's trailing
            // tokens (for perfect first-token spacing) would require Register to
            // carry input_ids — deferred.
            DetokBackend::Dynamo(t) => Box::new(DynamoDecoder {
                stream: t.decode_stream(&[], SKIP_SPECIAL_TOKENS),
            }),
            DetokBackend::Stub => Box::new(StubDecoder),
        }
    }
}

struct DetokState {
    sink: EgressSink,
    stream: bool,
    /// Cumulative decoded text (SGLang stream frames carry cumulative text).
    text: String,
    decoder: Box<dyn StreamDecoder>,
    /// Egress half of the lifecycle FSM. Lives here because the ingress
    /// `Request` (and its FSM) was handed to the scheduler when queued; the
    /// shard is the sole owner of the request's egress state, so no lock.
    fsm: RequestState,
}

pub fn run_shard(shard: usize, rx: flume::Receiver<DetokMsg>, backend: DetokBackend) {
    let mut table: HashMap<RequestId, DetokState> = HashMap::new();
    tracing::debug!(shard, "detok shard started");

    while let Ok(msg) = rx.recv() {
        match msg {
            DetokMsg::Register { id, sink, stream } => {
                table.insert(
                    id,
                    DetokState {
                        sink,
                        stream,
                        text: String::new(),
                        decoder: backend.new_stream(),
                        // Registered == handed to the scheduler == Queued.
                        fsm: RequestState::Queued,
                    },
                );
            }
            DetokMsg::Chunk(ev) => handle_chunk(&mut table, ev),
        }
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

    match st.decoder.step(&ev.token_ids) {
        Ok(delta) => st.text.push_str(&delta),
        Err(e) => {
            let _ = st.fsm.apply(Event::Error(e.clone()));
            let _ = st.sink.try_send(EgressItem::Error(e));
            table.remove(&id);
            return;
        }
    }

    let finished = ev.finish_reason.is_some();
    // Streaming → Streaming (finish:false) or Streaming → Finalizing (finish:true).
    let _ = st.fsm.apply(Event::Chunk { finish: finished });
    let frame = build_frame(&ev.rid, &st.text, ev.finish_reason.as_deref());

    if finished {
        // The Done frame *is* the final frame: Finalizing → Completed.
        let sent = st.sink.try_send(EgressItem::Done(frame)).is_ok();
        let _ = st.fsm.apply(if sent {
            Event::FinalFrameSent
        } else {
            Event::Disconnect
        });
        table.remove(&id);
    } else if st.stream {
        // Only emit intermediate frames for streaming requests; unary requests
        // just accumulate until the final Done. A failed send == client gone.
        if st.sink.try_send(EgressItem::Frame(frame)).is_err() {
            let _ = st.fsm.apply(Event::Disconnect);
            table.remove(&id);
            // TODO(abort): signal the scheduler to abort this rid.
        }
    }
}

/// Serialize one SGLang-style `/generate` output frame.
fn build_frame(rid: &str, text: &str, finish_reason: Option<&str>) -> Bytes {
    let v = serde_json::json!({
        "text": text,
        "meta_info": {
            "id": rid,
            "finish_reason": finish_reason.map(|r| serde_json::json!({ "type": r })),
        },
    });
    // to_vec never fails for this shape.
    Bytes::from(serde_json::to_vec(&v).unwrap_or_default())
}
