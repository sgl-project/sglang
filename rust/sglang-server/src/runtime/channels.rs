//! Inter-stage channel topology.
//!
//! All edges are `flume` MPMC channels carrying *moved* values (zero copy).
//! `Sender`s are cheap to clone and are bundled in [`Senders`] for producers;
//! each stage loop owns its `Receiver` exclusively (handed out at spawn time),
//! so there is exactly one consumer identity per logical queue and no lock is
//! needed to protect per-request state.

use crate::ids::RequestId;
use crate::message::{ChunkEvent, EgressSink, Request};

/// Events delivered to the TokenizerManager ingress loop. Both the API server
/// (fresh requests) and the Tokenizer pool (returned requests) feed this one
/// inbox, which keeps the loop a single consumer with no `select`.
pub enum TmEvent {
    /// A freshly received request from the API server.
    Ingress(Request),
    /// A request returned from the Tokenizer pool with `input_ids` filled in.
    Tokenized(Request),
    /// Abort an in-flight request (the HTTP client disconnected). The ingress
    /// loop forwards it to the scheduler as an `AbortReq` so generation stops
    /// instead of running to EOS for a client that's gone.
    Abort(RequestId),
}

/// Messages to a Detokenizer shard. Registration carries the per-request sink
/// so the shard owns a local `id -> sink` map (no shared, no lock); chunks are
/// routed to the same shard purely by `RequestId::shard`.
pub enum DetokMsg {
    Register {
        id: RequestId,
        sink: EgressSink,
        /// `return_text_in_logprobs`: decode logprob token ids to text in this
        /// shard (CPU-bound pool) rather than on the api-server I/O threads.
        decode_logprob_text: bool,
    },
    Chunk(ChunkEvent),
    /// Control-request result: a single already-serialized JSON payload to
    /// deliver to the request's sink verbatim (no detokenization, no
    /// streaming). Used by `/server_info` and other control endpoints.
    Result {
        id: RequestId,
        payload: bytes::Bytes,
    },
    /// Terminal per-request failure (e.g. the scheduler ingress couldn't decode a
    /// malformed request header): deliver an `Error` to the sink and drop the
    /// request, so the client gets a 400 instead of the scheduler crashing.
    Fail {
        id: RequestId,
        message: String,
    },
}

/// Producer-side handles, cloned into every stage that needs to emit.
#[derive(Clone)]
pub struct Senders {
    /// → TokenizerManager ingress loop.
    pub tm: flume::Sender<TmEvent>,
    /// → Tokenizer pool (CPU-bound, pinned threads).
    pub tok: flume::Sender<Request>,
    /// → Detokenizer shards, indexed by `RequestId::shard(detok.len())`.
    pub detok: Vec<flume::Sender<DetokMsg>>,
}

impl Senders {
    #[inline]
    pub fn detok_for(&self, id: RequestId) -> &flume::Sender<DetokMsg> {
        &self.detok[id.shard(self.detok.len())]
    }
}
