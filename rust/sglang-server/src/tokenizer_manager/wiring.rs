//! The flume fabric between stages: the request-loop inbox ([`TmEvent`]), the
//! abort lane ([`LifecycleEvent`]), the producer-side handles ([`Senders`]), and
//! the shutdown-aware [`recv`].

use crate::message::detok::DetokMsg;
use crate::message::ids::Rid;
use crate::message::request::Request;

/// Blocking receive that also wakes on shutdown: returns `None` when `rx` closes
/// *or* the `shutdown` sender is dropped.
pub fn recv<T>(rx: &flume::Receiver<T>, shutdown: &flume::Receiver<()>) -> Option<T> {
    flume::Selector::new()
        .recv(rx, |r| r.ok())
        .recv(shutdown, |_| None)
        .wait()
}

/// Events into the TokenizerManager request loop. API server + tokenizer pool
/// share this one inbox, keeping the loop a single consumer (no `select`).
pub enum TmEvent {
    /// A freshly received request from the API server.
    Intake(Request),
    /// Ordered behind earlier submissions so cancellation also covers requests
    /// that have not left the intake queue yet.
    Abort {
        rid_prefix: String,
        abort_all: bool,
        reply: tokio::sync::oneshot::Sender<bool>,
    },
    /// A request back from the tokenizer pool: `PreSendValidating` (ids filled)
    /// on success, or `Failed` on a tokenize error. `drive` handles both.
    Tokenized(Request),
    /// An MM worker finished a request parked in `Encoding`: `input_ids` are the
    /// final placeholder-expanded prompt ids. The buffers ride the rid-keyed
    /// result store (`Server.take_mm_result`), not this event.
    MmEncoded {
        rid: Rid,
        input_ids: Vec<i32>,
        response_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    },
    /// An MM worker rejected a request parked in `Encoding` (bad media URL,
    /// unsupported modality, preprocess error, …).
    MmFailed { rid: Rid, message: String },
}

/// Terminal notifications use a separate lane so generation backpressure
/// cannot prevent cancellation or release of request state.
#[derive(Clone, Debug)]
pub enum LifecycleEvent {
    /// From an `AbortGuard` drop. Owns the release.
    GuardAbort(Rid),
    /// From a detokenizer terminal path. Aborts the scheduler work.
    DetokAbort(Rid),
    Finished(Rid),
}

impl LifecycleEvent {
    pub fn rid(&self) -> &Rid {
        match self {
            Self::GuardAbort(rid) | Self::DetokAbort(rid) | Self::Finished(rid) => rid,
        }
    }
}

/// Producer-side handles, cloned into every stage that needs to emit.
#[derive(Clone)]
pub struct Senders {
    /// → TokenizerManager loop.
    pub tok_manager_tx: flume::Sender<TmEvent>,
    /// → the same loop, with cancellation and terminal notifications.
    pub lifecycle_tx: flume::Sender<LifecycleEvent>,
    /// → Tokenizer pool (CPU-bound, pinned threads).
    pub tokenizer_tx: flume::Sender<Request>,
    /// → Detokenizer shards, indexed by `Rid::shard(detok.len())`.
    pub detokenizer_tx: Vec<flume::Sender<DetokMsg>>,
}

impl Senders {
    #[inline]
    pub fn detok_for(&self, rid: &Rid) -> &flume::Sender<DetokMsg> {
        &self.detokenizer_tx[rid.shard(self.detokenizer_tx.len())]
    }
}
