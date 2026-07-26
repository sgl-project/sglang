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

use crate::ids::Rid;
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
    /// An MM worker finished a request parked in `Encoding`: `input_ids` are
    /// the final (placeholder-expanded) prompt ids from the native pipeline.
    /// The mm buffers ride the Rust sidecar (rid-keyed, popped by
    /// `RustServer.drain` via `Server.take_mm`), not this event.
    MmEncoded { rid: Rid, input_ids: Vec<i32> },
    /// An MM worker rejected a request parked in `Encoding` (bad media URL,
    /// unsupported modality, preprocess error, …) — reject it back to the
    /// client as a 400.
    MmFailed { rid: Rid, message: String },
}

/// Producer-side handles, cloned into every stage that needs to emit.
/// Who asked for an abort. Both variants do the same work in
/// [`Ingress::on_abort`](crate::tokenizer_manager::ingress::Ingress) — deregister
/// the detok entry, tell the scheduler to stop — and the source is kept for
/// diagnostics.
///
/// There is no in-flight rid registry to keep consistent, and so no release
/// ordering to get wrong: [`Rid::from_client`] makes every client-supplied rid
/// internally unique, so a resubmit of the "same" rid is a different `Rid` and
/// cannot be tangled up with an abort still in flight for the original.
#[derive(Clone, Debug)]
pub enum AbortSource {
    /// From an `AbortGuard` drop. Owns the release.
    Guard(Rid),
    /// From a detokenizer terminal path. Aborts the scheduler work.
    Detok(Rid),
}

impl AbortSource {
    pub fn rid(&self) -> &Rid {
        match self {
            Self::Guard(rid) | Self::Detok(rid) => rid,
        }
    }
}

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
    pub abort: flume::Sender<AbortSource>,
    /// → Tokenizer pool (CPU-bound, pinned threads).
    pub tok: flume::Sender<Request>,
    /// → Detokenizer shards, indexed by `Rid::shard(detok.len())`.
    pub detok: Vec<flume::Sender<DetokMsg>>,
}

impl Senders {
    #[inline]
    pub fn detok_for(&self, rid: &Rid) -> &flume::Sender<DetokMsg> {
        &self.detok[rid.shard(self.detok.len())]
    }
}
