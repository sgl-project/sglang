//! TokenizerManager — owns the request lifecycle across two isolated threads:
//!
//!   * [`ingress`] — drives the ingress FSM (Received → Validating →
//!     Normalizing → {Tokenizing | PreSendValidating}) and pushes tokenized requests to the
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
pub use ingress::{Ingress, Limits};

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
    /// A request back from the tokenizer pool: `PreSendValidating` (ids filled) on success,
    /// or `Failed` on a tokenize error. `drive` handles both.
    Tokenized(Request),
}

/// Producer-side handles, cloned into every stage that needs to emit.
#[derive(Clone)]
pub struct Senders {
    /// → TokenizerManager ingress loop.
    pub tm: flume::Sender<TmEvent>,
    /// → the same loop, but UNBOUNDED and abort-only.
    ///
    /// Aborts cannot share the bounded inbox. `try_send` there drops them exactly
    /// when they matter most — under overload — leaving the scheduler generating
    /// for a dead connection; and the caller then faces a false choice between
    /// releasing the rid (a live entry can be overwritten by a resubmit) and
    /// holding it (a permanent leak). An unbounded lane removes the dilemma: an
    /// abort is a small `String` and is always accepted, so releases can be
    /// unconditional again. It cannot grow without bound in practice — one entry
    /// per in-flight request, each already bounded by the inbox that admitted it.
    pub abort: flume::Sender<String>,
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
