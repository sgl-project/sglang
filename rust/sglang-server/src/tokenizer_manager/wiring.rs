//! The flume fabric between stages: the request-loop inbox ([`TmEvent`]), the
//! abort lane ([`AbortSource`]), the producer-side handles ([`Senders`]), and
//! the shutdown-aware [`recv`].

use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

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

/// Events into the TokenizerManager request loop. API server, tokenizer pool
/// and MM pool share this one inbox, keeping the loop a single consumer (no
/// `select`). Both pools take the whole `Request`, advance its FSM, and hand
/// it back; `drive` re-dispatches on the state.
pub enum TmEvent {
    /// A freshly received request from the API server.
    Intake {
        request: Request,
        admission: RequestAdmission,
    },
    /// A request back from the tokenizer pool: `PreSendValidating` (ids filled)
    /// or `Encoding` (multimodal prompt) on success, `Failed` on a tokenize
    /// error.
    Tokenized(Request),
    /// A request back from the MM pool: `PreSendValidating` with the final
    /// placeholder-expanded `input_ids` and the feature buffers (inline or
    /// shm) set on it, or `Failed` (bad media URL, unsupported modality,
    /// preprocess error, ...).
    Encoded(Request),
}

const ADMISSION_PENDING: u8 = 0;
const ADMISSION_ACCEPTED: u8 = 1;
const ADMISSION_CANCELLED: u8 = 2;
const ADMISSION_COMPLETED: u8 = 3;

/// Cancellation-safe ownership handoff between a core call and Intake.
///
/// A bounded `send_async` can enqueue a request and wake its sender without the
/// sender being polled again. If that task is then cancelled, this token decides
/// exactly one outcome: either Intake observes the prior cancellation and drops
/// the request, or the core observes prior admission and emits an abort.
#[derive(Clone)]
pub struct RequestAdmission {
    state: Arc<AtomicU8>,
}

impl RequestAdmission {
    pub fn pending() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(ADMISSION_PENDING)),
        }
    }

    /// Claim the request for Intake. `false` means cancellation won the race.
    pub fn try_accept(&self) -> bool {
        self.state
            .compare_exchange(
                ADMISSION_PENDING,
                ADMISSION_ACCEPTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Cancel the request. `true` means Intake already owns it and needs an abort.
    pub fn cancel_requires_abort(&self) -> bool {
        self.state.swap(ADMISSION_CANCELLED, Ordering::AcqRel) == ADMISSION_ACCEPTED
    }

    /// Cancel the request only if Intake has not claimed it. `true` means it
    /// will never start.
    pub fn cancel_if_pending(&self) -> bool {
        self.state
            .compare_exchange(
                ADMISSION_PENDING,
                ADMISSION_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Whether Intake owns the request and no terminal response or
    /// cancellation has ended it.
    pub fn is_accepted(&self) -> bool {
        self.state.load(Ordering::Acquire) == ADMISSION_ACCEPTED
    }

    /// Prevent an explicit abort racing with an observed terminal response
    /// from sending a redundant scheduler abort.
    pub fn complete(&self) {
        let _ = self.state.compare_exchange(
            ADMISSION_ACCEPTED,
            ADMISSION_COMPLETED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

/// The source of the abort request. [`Intake::on_abort`] deregisters the
/// detok entry for every source and tells the scheduler to stop for all but
/// [`Self::Local`]; the source is otherwise kept for diagnostics. A
/// [`Self::Request`] for a request the scheduler has keeps the entry instead.
///
/// [`Rid::from_client`] makes every client-supplied rid internally unique, so
/// an abort already in flight cannot affect a later reuse of the public ID.
/// Explicit cancellation resolves public IDs in the core before using this lane.
#[derive(Clone, Debug)]
pub enum AbortSource {
    /// From a core call drop or explicit core cancellation. Owns the release.
    Guard(Rid),
    /// From a detokenizer terminal path. Aborts the scheduler work.
    Detok(Rid),
    /// From a dropped call that never reaches the scheduler (tokenize,
    /// detokenize): only its Rust-side state is released.
    Local(Rid),
    /// From `/abort_request`: the scheduler aborts the request and its own
    /// terminal output, partial generation included, still reaches the
    /// client, as Python's `TokenizerManager.abort_request` leaves it.
    Request(Rid),
}

impl AbortSource {
    pub fn rid(&self) -> &Rid {
        match self {
            Self::Guard(rid) | Self::Detok(rid) | Self::Local(rid) | Self::Request(rid) => rid,
        }
    }
}

/// Producer-side handles, cloned into every stage that needs to emit.
#[derive(Clone)]
pub struct Senders {
    /// → TokenizerManager loop.
    pub tok_manager_tx: flume::Sender<TmEvent>,
    /// → the same loop, but UNBOUNDED and abort-only.
    pub abort_tx: flume::Sender<AbortSource>,
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
