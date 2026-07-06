//! Inter-stage channel topology.
//!
//! All edges are `flume` MPMC channels carrying *moved* values (zero copy).
//! `Sender`s are cheap to clone and are bundled in [`Senders`] for producers;
//! each stage loop owns its `Receiver` exclusively (handed out at spawn time),
//! so there is exactly one consumer identity per logical queue and no lock is
//! needed to protect per-request state.

use crate::ids::RequestId;
use crate::message::{ChunkEvent, EgressSink, Request};

/// Events into the TokenizerManager ingress loop. API server + tokenizer pool
/// share this one inbox, keeping the loop a single consumer (no `select`).
pub enum TmEvent {
    /// A freshly received request from the API server.
    Ingress(Request),
    /// A request back from the tokenizer pool: `Queued` (ids filled) on success,
    /// or `Failed` on a tokenize error. `drive` handles both.
    Tokenized(Request),
    /// Client disconnected: forwarded to the scheduler as an `AbortReq` so
    /// generation stops instead of running to EOS.
    Abort(RequestId),
}

/// Messages to a Detokenizer shard. `Register` carries the per-request sink for
/// the shard's local `id -> sink` map; everything routes by `RequestId::shard`.
pub enum DetokMsg {
    Register {
        id: RequestId,
        sink: EgressSink,
        /// Decode logprob token ids to text on this (CPU-bound) shard rather than
        /// the api-server I/O threads.
        decode_logprob_text: bool,
    },
    Chunk(ChunkEvent),
    /// Control result: one already-serialized payload delivered to the sink
    /// verbatim (e.g. `/server_info`).
    Result {
        id: RequestId,
        payload: bytes::Bytes,
    },
    /// Terminal per-request failure (e.g. an undecodable header): deliver an
    /// `Error` to the sink and drop it, so the client gets a 400, not a crash.
    Fail {
        id: RequestId,
        message: String,
    },
    /// Drop a registration for a request rejected before the scheduler. The
    /// rejecting stage already notified the client, so this only removes the
    /// `id -> sink` entry (else `Register` leaks one entry per rejected request).
    Deregister {
        id: RequestId,
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
