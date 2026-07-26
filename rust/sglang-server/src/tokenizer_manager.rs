//! TokenizerManager — owns the request lifecycle across two isolated threads:
//!
//!   * [`ingress`] — drives the ingress FSM (Received → Validating →
//!     Normalizing → {Tokenizing | Queued}) and pushes tokenized requests to the
//!     scheduler ring.
//!   * [`egress`] — drains the scheduler-output ring and routes each chunk to
//!     the owning detokenizer shard.
//!
//! The two run on separate pinned threads with no shared state, connected to
//! the rest of the pipeline only through `flume` channels: [`TmEvent`] into the
//! ingress loop, [`Senders`] fanning out to the pools.

mod egress;
mod ingress;

pub use egress::{ActivityCounter, Egress};
pub use ingress::Ingress;

use crate::ids::RidHash;
use crate::message::{DetokMsg, Request};

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
    /// An MM worker finished a request parked in `Encoding`: `input_ids` are
    /// the final (placeholder-expanded) prompt ids from the native pipeline.
    /// The mm buffers ride the Rust sidecar (rid-keyed, popped by
    /// `RustServer.drain` via `Server.take_mm`), not this event.
    MmEncoded { rid: String, input_ids: Vec<i32> },
    /// An MM worker rejected a request parked in `Encoding` (bad media URL,
    /// unsupported modality, preprocess error, …) — reject it back to the
    /// client as a 400.
    MmFailed { rid: String, message: String },
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
