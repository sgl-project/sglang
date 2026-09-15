//! Messages to a Detokenizer shard.

use super::ids::Rid;
use super::response::{ChunkEvent, ResponseSink, SchedulerAbort};
use super::sampling::SamplingParams;

/// Messages to a Detokenizer shard. `Register` carries the per-request sink for
/// the shard's local `rid -> sink` map. The rid STRING is the identity: `Rid::hash`
/// picks the shard (collisions there merely co-locate, which is harmless), but two
/// distinct rids that hash alike must not be the same map entry — that evicted one
/// client's sink and delivered their tokens to the other's connection. Equal rids
/// cannot reach here from different requests: `Rid::from_client` uniquifies every
/// client-supplied one.
pub enum DetokMsg {
    /// Final prompt and normalized parameters, queued before scheduler admission.
    Prepared {
        rid: Rid,
        prompt_token_ids: Option<std::sync::Arc<[i32]>>,
        sampling_params: Box<SamplingParams>,
        dispatch_finished_ts: Option<f64>,
        response_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    },
    Register {
        /// Client-visible rid string — kept in `DetokState` so the shard can
        /// emit `TmEvent::Abort(rid)` (the wire needs the string, not the hash).
        rid: Rid,
        sink: ResponseSink,
        /// Decode logprob token ids to text here (CPU-bound) not on the api threads.
        decode_logprob_text: bool,
        /// Preserve special tokens when requested by `SamplingParams`.
        skip_special_tokens: bool,
        /// `SamplingParams.no_stop_trim`: keep the matched stop; default trims it.
        no_stop_trim: bool,
        /// Literal strings used by Python's final matched-stop trimming.
        stop_texts: Vec<String>,
        logprobs: Option<super::response::LogprobOptions>,
        /// One reply per DP worker for scheduler controls; zero for generation.
        control_replies: usize,
        metrics: Option<Box<crate::metrics::RequestMetrics>>,
    },
    /// One decode step's chunks for *this shard*. Batched because `from-scheduler` blocks
    /// per send.
    Chunks(Vec<ChunkEvent>),
    /// Decode a complete token-id sequence — the backend of
    /// [`RequestKind::Detokenize`](super::RequestKind::Detokenize), the one
    /// request kind the detok stage itself answers (it never reaches the
    /// scheduler ring). Sent by to-scheduler right after the same rid's `Register`
    /// on the same channel (FIFO), so the shard delivers the text through the
    /// registered sink like a control `Result` and drops the entry.
    Decode {
        rid: Rid,
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
    },
    /// Control result: one already-serialized payload delivered to the sink verbatim.
    Result { rid: Rid, payload: bytes::Bytes },
    ResultPart {
        rid: Rid,
        dp_rank: u32,
        payload: bytes::Bytes,
    },
    /// Scheduler-side rejection or abort, distinct from an internal wire error.
    Abort { rid: Rid, outcome: SchedulerAbort },
    /// Terminal per-request failure → an `Error` to the sink (a 400, not a crash).
    Fail { rid: Rid, message: String },
    /// Drop the `rid -> sink` entry for a request rejected before the scheduler
    /// (the rejecting stage already answered the client); else `Register` leaks one
    /// entry.
    Deregister { rid: Rid },
}
