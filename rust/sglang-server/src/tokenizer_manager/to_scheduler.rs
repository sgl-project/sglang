//! TokenizerManager — to_scheduler side.

use std::collections::HashMap;

use crate::message::detok::DetokMsg;
use crate::message::ids::Rid;
use crate::message::io_struct::{AbortReq, ControlRequest};
use crate::message::request::{Request, RequestKind, SchedulerRequest};
use crate::message::response::ResponseItem;
use crate::runtime::Runnable;
use crate::tokenizer_manager::channel::ToSchedulerTx;
pub use crate::tokenizer_manager::to_scheduler_types::{Limits, MmDispatch};
use crate::tokenizer_manager::to_scheduler_validation::{
    check_total_tokens, validate, validate_input_ids,
};
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
    mm: MmDispatch,
    /// Track request states when intake does not hold it.
    request_states: HashMap<Rid, RequestState>,
    shutdown: flume::Receiver<()>,
}

impl Intake {
    pub fn new(
        tok_manager_rx: flume::Receiver<TmEvent>,
        abort_rx: flume::Receiver<AbortSource>,
        senders: Senders,
        to_scheduler_tx: ToSchedulerTx,
        limits: Limits,
        mm: MmDispatch,
        shutdown: flume::Receiver<()>,
    ) -> Self {
        Self {
            tok_manager_rx,
            abort_rx,
            senders,
            to_scheduler_tx,
            limits,
            mm,
            request_states: HashMap::new(),
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
                // A fresh request and one returning from either pool.
                Some(Lane::Event(
                    TmEvent::Intake(req) | TmEvent::Tokenized(req) | TmEvent::Encoded(req),
                )) => self.drive(req),
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
        let _ = req.state.apply(Event::Error(err.clone()));
        let _ = req.sink.try_send(ResponseItem::Error(err)); // client may be gone
        if registered {
            let _ = self.senders.detok_for(&req.rid).send(DetokMsg::Deregister {
                rid: req.rid.clone(),
            });
        }
    }

    /// Drive a request through its intake states until it terminates (failed or
    /// pushed to the channel) or is handed to a pool — the tokenizer pool
    /// (re-entering as `Tokenized`) or the MM pool (re-entering as `Encoded`).
    /// Each arm acts and advances the FSM; the loop re-dispatches. The arms
    /// are the design table's states, `Failed` the single reject path.
    fn drive(&mut self, mut req: Request) {
        // A pool return whose client left while it was out: the sink is already
        // deregistered and the scheduler never saw the rid, so nothing is
        // pushed or failed. Dropping the request releases any shm it carries.
        if matches!(
            self.request_states.remove(&req.rid),
            Some(RequestState::Aborted)
        ) {
            tracing::debug!(rid = %req.rid, "dropping request aborted");
            return;
        }
        // Flipped once `register_detok` succeeds; `fail` must not deregister before
        // that (see `fail`). A pool return re-enters `drive` already registered.
        let mut registered = !matches!(req.state, RequestState::Received);
        loop {
            // One clone per pass: the hand-off arms store it as the state the
            // request leaves in.
            let state = req.state.clone();
            match state {
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
                            // precedence as the Python TokenizerManager. The MM
                            // worker expands placeholders in ids only, so the
                            // route is Tokenizing → Encoding; a prompt that
                            // already has ids skips the pool hop, not the state.
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
                            // AlreadyTokenized → PreSendValidating; NeedsTokenize
                            // and HasMultimodal → Tokenizing (then PreSend / Encode).
                            let _ = req.state.apply(Event::Validated(o));
                        }
                    }
                }
                // Hand off to the MM worker pool, which returns the request as
                // an `Encoded` event (PreSendValidating with the expanded ids and
                // feature buffers set — or Failed on error). Doesn't loop. The
                // request is out of reach while there; an abort in that window
                // is recorded in `request_states` and honored when it returns.
                RequestState::Encoding => {
                    let rid = req.rid.clone();
                    // Full = the pool can't keep up, so back-pressure like a full
                    // to_scheduler channel. Disconnected = pool gone. Either way
                    // flume hands the request back.
                    match self.mm.tx.try_send(req) {
                        Ok(()) => {
                            self.request_states.insert(rid, state);
                        }
                        Err(e) => {
                            let (err, mut req) = match e {
                                flume::TrySendError::Full(req) => (Error::QueueFull, req),
                                flume::TrySendError::Disconnected(req) => {
                                    (Error::Internal("mm worker pool gone".into()), req)
                                }
                            };
                            self.fail(&mut req, err, registered);
                        }
                    }
                    return;
                }
                // Hand off to the tokenizer pool; it returns the request as a
                // `Tokenized` event (`then`: PreSendValidating or Encoding — or
                // Failed on error). Doesn't loop — except for a prompt that
                // already carries ids (a pre-tokenized multimodal request), which
                // has nothing for the pool: apply `TokenizeDone` here and keep
                // driving, so the state walks the same fixed order without the hop.
                RequestState::Tokenizing { .. } => {
                    if let RequestKind::Generate(g) = &req.kind
                        && g.already_tokenized()
                    {
                        let _ = req.state.apply(Event::TokenizeDone);
                        continue;
                    }
                    let rid = req.rid.clone();
                    match self.senders.tokenizer_tx.send(req) {
                        Ok(()) => {
                            self.request_states.insert(rid, state);
                        }
                        Err(err) => {
                            // Pool gone (workers exited); flume hands the request back.
                            let mut req = err.into_inner();
                            // Past `Received`, so registration happened.
                            self.fail(
                                &mut req,
                                Error::Internal("tokenizer pool gone".into()),
                                true,
                            );
                        }
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
                        && let Err(e) = validate_input_ids(g, self.limits.vocab_size)
                            .and_then(|()| check_total_tokens(g, &self.limits))
                    {
                        let _ = req.state.apply(Event::Error(e)); // → Failed
                        continue;
                    }
                    let _ = req.state.apply(Event::PreSendValidated); // → Queued
                }
                // Hand the request to the stage that answers it: the scheduler
                // channel (generate payload or control frame), or — for detokenize
                // — the detok shard itself.
                RequestState::Queued => {
                    // The patterns bind nothing, so the match reads only the
                    // discriminant and `req` can be moved into each push.
                    match req.kind {
                        RequestKind::Generate(_) => self.push_to_channel(req),
                        RequestKind::Control(_) => self.push_control_to_channel(req),
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
    /// answers this kind (it never touches the scheduler channel). The shard
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
    fn push_control_to_channel(&self, mut req: Request) {
        let encode = match &req.kind {
            RequestKind::Control(control) => control.encode(),
            _ => Err(Error::Internal(
                "non-control request reached push_control_to_channel".into(),
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
            buffers: Vec::new(),
        }) {
            self.fail(&mut req, Error::QueueFull, true); // registered
        }
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
    /// A request out in a pool never reached the scheduler, so there is nothing to
    /// abort there yet: it is marked `Aborted` and dropped when it returns.
    fn on_abort(&mut self, source: AbortSource) {
        let rid = source.rid().clone();
        let _ = self
            .senders
            .detok_for(&rid)
            .send(DetokMsg::Deregister { rid: rid.clone() });
        if let Some(state) = self.request_states.get_mut(&rid) {
            *state = RequestState::Aborted;
            tracing::debug!(rid = %rid, "abort recorded for request");
            return;
        }

        // The channel is BOUNDED and drops pushes under exactly the load this matters
        // for, so report the miss rather than assuming the scheduler was told.
        match ControlRequest::AbortReq(AbortReq::new(rid.as_str().to_string(), false)).encode() {
            Ok(header) => {
                if !self.to_scheduler_tx.try_push(SchedulerRequest {
                    header,
                    buffers: Vec::new(),
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
    fn push_to_channel(&self, mut req: Request) {
        // Only generate requests reach here (control uses `push_control_to_channel`).
        // Validate + serialize the header first (borrowing `g`), then move the
        // buffers out; the resulting values own their data, so no borrow
        // outlives a `fail(&mut req)`.
        let serialized = match &mut req.kind {
            RequestKind::Generate(g) if g.already_tokenized() => {
                g.encode_header().map(|header| (header, g.take_buffers()))
            }
            RequestKind::Generate(_) => Err(Error::Tokenize("empty input_ids".into())),
            _ => Err(Error::Internal(
                "non-generate request reached push_to_channel".into(),
            )),
        };
        let (header, buffers) = match serialized {
            Ok(v) => v,
            Err(e) => {
                self.fail(&mut req, e, true); // on the push path: registered
                return;
            }
        };

        if !self
            .to_scheduler_tx
            .try_push(SchedulerRequest { header, buffers })
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
