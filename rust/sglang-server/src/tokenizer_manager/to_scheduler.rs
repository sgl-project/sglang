//! TokenizerManager — to_scheduler side.

use std::collections::HashMap;

use bytes::Bytes;

use crate::message::detok::DetokMsg;
use crate::message::ids::Rid;
use crate::message::io_struct::{AbortReq, ControlRequest};
use crate::message::request::{MmRequest, Request, RequestKind, SchedulerRequest};
use crate::message::response::ResponseItem;
use crate::runtime::Runnable;
use crate::tokenizer_manager::channel::ToSchedulerTx;
pub use crate::tokenizer_manager::to_scheduler_types::{Limits, Mm};
use crate::tokenizer_manager::to_scheduler_validation::{check_total_tokens, validate};
use crate::tokenizer_manager::wiring::{AbortSource, Senders, TmEvent};
use crate::utils::{
    error::Error,
    fsm::{Event, RequestState, ValidationOutcome},
};

/// Longest client-supplied rid accepted. It keys the detok table and travels on
/// every chunk, so its length is a recurring cost; Python mints 32-byte uuid hex.
pub(super) const MAX_RID_LEN: usize = 128;

/// Intake FSM dispatcher stage. Owns its inbox + downstream handles, so the
/// runtime spawns it as a [`Runnable`] rather than calling a free `run_*` fn
/// with positional arguments.
pub struct Intake {
    tok_manager_rx: flume::Receiver<TmEvent>,
    /// Unbounded abort lane (see [`Senders::abort`]). Selected against `rx` so an
    /// abort is handled promptly even while the bounded inbox is saturated.
    abort_rx: flume::Receiver<AbortSource>,
    senders: Senders,
    to_scheduler_tx: ToSchedulerTx,
    limits: Limits,
    mm: Mm,
    /// Requests parked in `Encoding` while an MM worker processes their media;
    /// resumed by `MmEncoded` / `MmFailed`. Only this thread touches it, so no
    /// lock.
    pending_mm: HashMap<Rid, Request>,
    shutdown: flume::Receiver<()>,
}

impl Intake {
    pub fn new(
        tok_manager_rx: flume::Receiver<TmEvent>,
        abort_rx: flume::Receiver<AbortSource>,
        senders: Senders,
        to_scheduler_tx: ToSchedulerTx,
        limits: Limits,
        mm: Mm,
        shutdown: flume::Receiver<()>,
    ) -> Self {
        Self {
            tok_manager_rx,
            abort_rx,
            senders,
            to_scheduler_tx,
            limits,
            mm,
            pending_mm: HashMap::new(),
            shutdown,
        }
    }
}

/// Which lane produced the next item.
enum Lane {
    Abort(AbortSource),
    Event(TmEvent),
}

impl Runnable for Intake {
    fn run(mut self) {
        loop {
            // Select, not a drain-then-block: an abort arriving while the inbox is
            // idle must still be handled at once.
            let next = flume::Selector::new()
                .recv(&self.abort_rx, |r| r.ok().map(Lane::Abort))
                .recv(&self.tok_manager_rx, |r| r.ok().map(Lane::Event))
                .recv(&self.shutdown, |_| None)
                .wait();
            match next {
                Some(Lane::Abort(rid)) => self.on_abort(rid),
                // A fresh request and one returning from the tokenizer pool.
                Some(Lane::Event(TmEvent::Intake(req) | TmEvent::Tokenized(req))) => {
                    self.drive(req)
                }
                Some(Lane::Event(TmEvent::MmEncoded { rid, input_ids })) => {
                    self.on_mm_encoded(rid, input_ids)
                }
                Some(Lane::Event(TmEvent::MmFailed { rid, message })) => {
                    self.on_mm_failed(rid, message)
                }
                None => {
                    // Shutdown, or the inbox closed. Drain whatever is still queued
                    // on the abort lane first: those requests are in flight on the
                    // scheduler, and the selector may report the closed inbox before
                    // it ever looks at a pending abort.
                    while let Ok(source) = self.abort_rx.try_recv() {
                        self.on_abort(source);
                    }
                    return;
                }
            }
        }
    }
}

impl Intake {
    /// Reject a request: → `Failed`, notify the client, deregister (unconditional
    /// — a no-op when nothing was registered).
    /// `registered` says whether this request ever reached `register_detok`. It
    /// must: `Deregister`'s handler is a bare `table.remove(&rid)`, so a
    /// request rejected BEFORE registering would evict whatever entry currently
    /// holds that key — a concurrent request's sink — leaving that client with no
    /// terminal frame and a hung connection. Python cannot hit this because it
    /// validates before `rid_to_state[obj.rid] = state`.
    fn fail(&self, req: &mut Request, err: Error, registered: bool) {
        // Log only server faults (500); 4xx/499/503 are expected and would spam.
        if err.http_status() == 500 {
            tracing::error!(rid = %req.rid, error = %err, "intake rejected request");
        }
        // A rejected request never reaches the scheduler drain, so purge any
        // parked MM result (no-op for the common non-mm request).
        self.mm.sidecar.purge(req.rid.as_str());
        let _ = req.state.apply(Event::Error(err.clone()));
        let _ = req.sink.try_send(ResponseItem::Error(err)); // client may be gone
        if registered {
            let _ = self.senders.detok_for(&req.rid).send(DetokMsg::Deregister {
                rid: req.rid.clone(),
            });
        }
    }

    /// Drive a request through its intake states until it terminates (failed or
    /// pushed to the ring), is handed to the tokenizer pool (re-entering as a
    /// `Tokenized` event), or is parked in `pending_mm` awaiting an MM worker
    /// (re-entering via `MmEncoded` / `MmFailed`). Each arm acts and advances
    /// the FSM; the loop re-dispatches. The arms are the design table's states,
    /// `Failed` the single reject path.
    fn drive(&mut self, mut req: Request) {
        // Flipped once `register_detok` succeeds; `fail` must not deregister before
        // that (see `fail`). A pool return re-enters `drive` already registered.
        let mut registered = !matches!(req.state, RequestState::Received);
        loop {
            match req.state.clone() {
                // Validate, then register the sink before the request leaves Rust.
                // Failures move to `Failed` and fall through to the reject arm.
                RequestState::Received => {
                    if let Err(e) = validate(&mut req, &self.limits) {
                        let _ = req.state.apply(Event::Error(e)); // → Failed
                        continue;
                    }
                    if !self.register_detok(&req) {
                        let _ = req
                            .state
                            .apply(Event::Error(Error::Internal("detok shard gone".into())));
                        continue;
                    }
                    registered = true;
                    // `validate` advanced Received → Validating; keep driving.
                }
                // Control and detokenize skip normalization (no sampling params)
                // straight to the pre-send checks; generate goes to Normalizing.
                RequestState::Validating => match &req.kind {
                    RequestKind::Control(_) | RequestKind::Detokenize { .. } => {
                        let _ = req
                            .state
                            .apply(Event::Validated(ValidationOutcome::AlreadyTokenized));
                    }
                    RequestKind::Generate(_) => {
                        let _ = req.state.apply(Event::NeedsNormalize);
                    }
                },
                // Normalize + verify sampling params (off the scheduler loop), then
                // pick the branch; a bad param becomes `Failed`.
                RequestState::Normalizing => {
                    let outcome = {
                        let RequestKind::Generate(g) = &mut req.kind else {
                            // Unreachable (control/detokenize never reach here);
                            // reject so a bug can't leak/hang a registered request.
                            self.fail(
                                &mut req,
                                Error::Internal("non-generate request in Normalizing".into()),
                                registered,
                            );
                            return;
                        };
                        match g
                            .sampling_params
                            .normalize(self.limits.skip_tokenizer_init, self.limits.vocab_size)
                        {
                            Err(e) => Err(e),
                            // The Rust MM pipeline produces the final input_ids,
                            // so it wins even over a pre-tokenized prompt (which
                            // still needs placeholder expansion) — the same
                            // precedence as the Python TokenizerManager.
                            Ok(()) if self.mm.enabled && g.has_multimodal() => {
                                Ok(ValidationOutcome::HasMultimodal)
                            }
                            // Client ids skip the pool; text goes to the tokenizer.
                            Ok(()) if g.already_tokenized() => {
                                Ok(ValidationOutcome::AlreadyTokenized)
                            }
                            Ok(()) => Ok(ValidationOutcome::NeedsTokenize),
                        }
                    };
                    match outcome {
                        Err(e) => {
                            let _ = req.state.apply(Event::Error(e)); // → Failed
                        }
                        Ok(o) => {
                            // AlreadyTokenized → Queued, NeedsTokenize → Tokenizing.
                            let _ = req.state.apply(Event::Validated(o));
                        }
                    }
                }
                // Hand off to the MM worker pool and park the request; it
                // re-enters via `MmEncoded` (→ PreSendValidating) or `MmFailed`
                // (→ reject). Doesn't loop.
                RequestState::Encoding => {
                    let work = {
                        let RequestKind::Generate(g) = &mut req.kind else {
                            self.fail(
                                &mut req,
                                Error::Internal("non-generate request in Encoding".into()),
                                registered,
                            );
                            return;
                        };
                        g.take_mm_work()
                    };
                    let msg = MmRequest {
                        rid: req.rid.clone(),
                        work,
                    };
                    // Full = the pool can't keep up, so back-pressure like a full
                    // to_scheduler channel. Disconnected = pool gone.
                    if let Err(e) = self.mm.tx.try_send(msg) {
                        let err = match e {
                            flume::TrySendError::Full(_) => Error::QueueFull,
                            flume::TrySendError::Disconnected(_) => {
                                Error::Internal("mm worker pool gone".into())
                            }
                        };
                        self.fail(&mut req, err, registered);
                        return;
                    }
                    self.pending_mm.insert(req.rid.clone(), req);
                    return;
                }
                // Hand off to the tokenizer pool; it returns the request as a
                // `Tokenized` event (PreSendValidating, or Failed on error).
                // Doesn't loop.
                RequestState::Tokenizing => {
                    if let Err(err) = self.senders.tokenizer_tx.send(req) {
                        // Pool gone (workers exited); flume hands the request back.
                        let mut req = err.into_inner();
                        // Past `Received`, so registration happened.
                        self.fail(
                            &mut req,
                            Error::Internal("tokenizer pool gone".into()),
                            true,
                        );
                    }
                    return;
                }
                // The checks that need the final `input_ids`: every branch
                // converges here (client ids arrive directly, text arrives from
                // the tokenizer pool), so they run once per request regardless of
                // how it was tokenized. `validate` runs too early — at `Received`
                // a text request has no ids yet.
                RequestState::PreSendValidating => {
                    if let RequestKind::Generate(g) = &mut req.kind
                        && let Err(e) = check_total_tokens(g, &self.limits)
                    {
                        let _ = req.state.apply(Event::Error(e)); // → Failed
                        continue;
                    }
                    let _ = req.state.apply(Event::PreSendValidated); // → Queued
                }
                // Hand the request to the stage that answers it: the scheduler
                // ring (generate payload or control frame), or — for detokenize
                // — the detok shard itself.
                RequestState::Queued => {
                    // The patterns bind nothing, so the match reads only the
                    // discriminant and `req` can be moved into each push.
                    match req.kind {
                        RequestKind::Generate(_) => self.push_to_ring(req),
                        RequestKind::Control(_) => self.push_control_to_ring(req),
                        RequestKind::Detokenize { .. } => self.push_detokenize_to_shard(req),
                    }
                    return;
                }
                // The single reject path for every post-register failure.
                RequestState::Failed(e) => {
                    self.fail(&mut req, e, registered);
                    return;
                }
                // Unreachable (request states never reach here). Reject via `fail`/
                // return (not apply + continue, which would spin on a terminal state).
                other => {
                    self.fail(
                        &mut req,
                        Error::Internal(format!("unexpected state: {other:?}")),
                        registered,
                    );
                    return;
                }
            }
        }
    }

    /// Register the response sink with the owning detok shard (by id) so the response
    /// has a home. Carries the per-request detok flags — `return_text_in_logprobs`
    /// (decode logprob text on this shard) and `no_stop_trim` (keep the matched
    /// stop in the output) — so the shard needs no back-reference to the request.
    /// Returns `false` if the shard is gone.
    fn register_detok(&self, req: &Request) -> bool {
        let (decode_logprob_text, no_stop_trim) = match &req.kind {
            RequestKind::Generate(g) => (
                g.return_text_in_logprobs.unwrap_or(false),
                g.sampling_params.no_stop_trim,
            ),
            RequestKind::Control(_) | RequestKind::Detokenize { .. } => (false, false),
        };
        self.senders
            .detok_for(&req.rid)
            .send(DetokMsg::Register {
                rid: req.rid.clone(),
                sink: req.sink.clone(),
                decode_logprob_text,
                no_stop_trim,
            })
            .is_ok()
    }

    /// Hand a `Detokenize` request to its owning detok shard — the stage that
    /// answers this kind (it never touches the scheduler ring). The shard
    /// already holds this rid's sink: `register_detok` queued `Register` on the
    /// same channel from this same thread, so FIFO gives Register → Decode.
    fn push_detokenize_to_shard(&self, mut req: Request) {
        let RequestKind::Detokenize { token_ids } = &req.kind else {
            self.fail(
                &mut req,
                Error::Internal("non-detokenize request reached push_detokenize_to_shard".into()),
                true,
            );
            return;
        };
        // Infallible: `validate` rejected out-of-range ids at `Received`.
        let token_ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
        if self
            .senders
            .detok_for(&req.rid)
            .send(DetokMsg::Decode {
                rid: req.rid.clone(),
                token_ids,
            })
            .is_err()
        {
            self.fail(&mut req, Error::Internal("detok shard gone".into()), true);
        }
    }

    /// Push a bare control request (`[tag, rid, nil]`) onto the to_scheduler channel. The
    /// scheduler dispatches it (e.g. `GetInternalStateReq`) and replies via the
    /// from_scheduler channel as a single `Result`.
    fn push_control_to_ring(&self, mut req: Request) {
        let encode = match &req.kind {
            RequestKind::Control(control) => control.encode(),
            _ => Err(Error::Internal(
                "non-control request reached push_control_to_ring".into(),
            )),
        };
        let header = match encode {
            Ok(b) => b,
            Err(e) => {
                self.fail(&mut req, e, true); // on the push path: registered
                return;
            }
        };
        // Control requests carry no tensor cell — empty `ids`.
        if !self.to_scheduler_tx.try_push(SchedulerRequest {
            header,
            ids: Bytes::new(),
        }) {
            self.fail(&mut req, Error::QueueFull, true); // registered
        }
    }

    /// An MM worker finished a parked request: fill in the final expanded
    /// `input_ids`, advance `Encoding → PreSendValidating`, and resume driving
    /// (pre-send checks → ring). No pending entry means the request was already
    /// rejected or aborted, so the result is dropped.
    fn on_mm_encoded(&mut self, rid: Rid, input_ids: Vec<i32>) {
        let Some(mut req) = self.pending_mm.remove(&rid) else {
            tracing::debug!(rid = %rid, "mm result for unknown/finished request; dropped");
            // It will never reach the scheduler drain, so purge or leak.
            self.mm.sidecar.purge(rid.as_str());
            return;
        };
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(input_ids);
        }
        let _ = req.state.apply(Event::EncodeDone); // Encoding → PreSendValidating
        self.drive(req);
    }

    /// An MM worker failed a parked request (bad URL, processor error): reject it
    /// back to the client, as Python turns a per-request exception into a 400.
    fn on_mm_failed(&mut self, rid: Rid, message: String) {
        let Some(mut req) = self.pending_mm.remove(&rid) else {
            tracing::debug!(rid = %rid, "mm failure for unknown/finished request; dropped");
            return;
        };
        self.fail(&mut req, Error::Encode(message), true); // parked ⇒ registered
    }

    /// Client disconnected (or a detok terminal): deregister the sink, then push an
    /// `AbortReq(rid)` so the scheduler stops generating for it.
    ///
    /// A failed push is logged, not retried: the scheduler keeps generating and the
    /// chunks arrive for a rid no longer in the detok table, where they are dropped.
    /// That wastes GPU work until the request finishes on its own, but it cannot be
    /// misdelivered — the rid is unique to this request for the process's lifetime
    /// ([`Rid::from_client`]), so no later request can ever answer to it.
    ///
    /// A request parked in `pending_mm` is cancelled here, so the worker's late
    /// result lands in `on_mm_encoded`'s no-entry branch and purges the sidecar —
    /// no generation runs for output nobody will read.
    fn on_abort(&mut self, source: AbortSource) {
        let rid = source.rid().clone();
        if self.pending_mm.remove(&rid).is_some() {
            tracing::debug!(rid = %rid, "abort cancelled request parked for MM");
        }
        let _ = self
            .senders
            .detok_for(&rid)
            .send(DetokMsg::Deregister { rid: rid.clone() });

        // The ring is BOUNDED and drops pushes under exactly the load this matters
        // for, so report the miss rather than assuming the scheduler was told.
        match ControlRequest::AbortReq(AbortReq::new(rid.as_str().to_string(), false)).encode() {
            Ok(header) => {
                if !self.to_scheduler_tx.try_push(SchedulerRequest {
                    header,
                    ids: Bytes::new(),
                }) {
                    tracing::error!(
                        rid = %rid,
                        "abort dropped: to_scheduler channel is full; the scheduler keeps generating \
                         for this request until it finishes on its own"
                    );
                }
            }
            Err(e) => tracing::error!(rid = %rid, error = %e, "abort encode failed"),
        }
    }

    /// Serialize the tokenized request to its `TokenizedGenerateReqInput` wire and
    /// push it onto the to_scheduler channel for the scheduler. On backpressure, fail it.
    fn push_to_ring(&self, mut req: Request) {
        // Only generate requests reach here (control uses `push_control_to_ring`).
        // Validate + serialize while borrowing `g` immutably; the resulting `Bytes`
        // own their data, so the borrow ends before any `fail(&mut req)`.
        let serialized = match &req.kind {
            RequestKind::Generate(g) if g.already_tokenized() => g
                .encode_header()
                .map(|header| (header, g.encode_data_buf())),
            RequestKind::Generate(_) => Err(Error::Tokenize("empty input_ids".into())),
            _ => Err(Error::Internal(
                "non-generate request reached push_to_ring".into(),
            )),
        };
        let (header, ids) = match serialized {
            Ok(v) => v,
            Err(e) => {
                self.fail(&mut req, e, true); // on the push path: registered
                return;
            }
        };

        if !self
            .to_scheduler_tx
            .try_push(SchedulerRequest { header, ids })
        {
            self.fail(&mut req, Error::QueueFull, true); // registered
        }
        // On success the scheduler owns the request (response arrives by rid); we
        // drop our `Request` here — the detok shard holds the sink.
    }
}

#[cfg(test)]
#[path = "to_scheduler_tests.rs"]
mod tests;
