//! Inter-stage channel topology.
//!
//! All edges are `flume` MPMC channels carrying *moved* values (zero copy).
//! `Sender`s are cheap to clone and are bundled in [`Senders`] for producers;
//! each stage loop owns its `Receiver` exclusively (handed out at spawn time),
//! so there is exactly one consumer identity per logical queue and no lock is
//! needed to protect per-request state.

use crate::ids::RidHash;
use crate::message::{ChunkEvent, EgressSink, Request};

/// Blocking receive that also wakes on shutdown: returns `None` when `rx` closes
/// *or* the `shutdown` sender is dropped.
pub fn recv<T>(rx: &flume::Receiver<T>, shutdown: &flume::Receiver<()>) -> Option<T> {
    flume::Selector::new()
        .recv(rx, |r| r.ok())
        .recv(shutdown, |_| None)
        .wait()
}

/// Events into the TokenizerManager ingress loop. API server + tokenizer pool
/// share this one inbox, keeping the loop a single consumer (no `select`).
pub enum TmEvent {
    /// A freshly received request from the API server.
    Ingress(Request),
    /// A request back from the tokenizer pool: `Queued` (ids filled) on success,
    /// or `Failed` on a tokenize error. `drive` handles both.
    Tokenized(Request),
    /// Client disconnected: forwarded to the scheduler as an `AbortReq` so
    /// generation stops instead of running to EOS. Carries the rid *string* —
    /// the scheduler wire needs it and it can't be recovered from the hashed
    /// `RidHash` (which `on_abort` re-derives via `RidHash::from_rid`).
    Abort(String),
}

/// Messages to a Detokenizer shard. `Register` carries the per-request sink for
/// the shard's local `rid_hash -> sink` map; everything routes by `RidHash::shard`.
pub enum DetokMsg {
    Register {
        rid_hash: RidHash,
        /// Client-visible rid string — kept in `DetokState` so the shard can
        /// emit `TmEvent::Abort(rid)` (the wire needs the string, not the hash).
        rid: String,
        sink: EgressSink,
        /// Decode logprob token ids to text here (CPU-bound) not on the api threads.
        decode_logprob_text: bool,
        /// `SamplingParams.no_stop_trim`: keep the matched stop; default trims it.
        no_stop_trim: bool,
    },
    /// One decode step's chunks for *this shard*. Batched because `tm-egress` blocks
    /// per send, so one message per request cost ~1.3 µs × batch (5.1x at 4096).
    Chunks(Vec<ChunkEvent>),
    /// Control result: one already-serialized payload delivered to the sink verbatim.
    Result {
        rid_hash: RidHash,
        payload: bytes::Bytes,
    },
    /// Terminal per-request failure → an `Error` to the sink (a 400, not a crash).
    Fail { rid_hash: RidHash, message: String },
    /// Drop the `rid_hash -> sink` entry for a request rejected before the scheduler
    /// (the rejecting stage already answered the client); else `Register` leaks one
    /// entry.
    Deregister { rid_hash: RidHash },
}

/// Producer-side handles, cloned into every stage that needs to emit.
#[derive(Clone)]
pub struct Senders {
    /// → TokenizerManager ingress loop.
    pub tm: flume::Sender<TmEvent>,
    /// → Tokenizer pool (CPU-bound, pinned threads).
    pub tok: flume::Sender<Request>,
    /// → Detokenizer shards, indexed by `RidHash::shard(detok.len())`.
    pub detok: Vec<flume::Sender<DetokMsg>>,
}

impl Senders {
    #[inline]
    pub fn detok_for(&self, id: RidHash) -> &flume::Sender<DetokMsg> {
        &self.detok[id.shard(self.detok.len())]
    }
}
