//! TokenizerManager — to_scheduler side.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use bytes::Bytes;

use crate::message::detok::DetokMsg;
use crate::message::finish_reason::{AbortReason, FinishKind};
use crate::message::ids::Rid;
use crate::message::io_struct::{AbortReq, ControlRequest};
use crate::message::request::{MmRequest, Request, RequestKind, SchedulerRequest};
use crate::message::response::{ChunkEvent, ResponseItem, ResponseSink};
use crate::runtime::Runnable;
use crate::tokenizer_manager::channel::ToSchedulerTx;
pub use crate::tokenizer_manager::to_scheduler_types::{Limits, MmDispatch};
use crate::tokenizer_manager::to_scheduler_validation::{
    check_total_tokens, validate, validate_delimiter_indices, validate_input_ids,
};
use crate::tokenizer_manager::wiring::{LifecycleEvent, Senders, TmEvent};
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
    /// Terminal notifications remain available while the work inbox is full.
    lifecycle_rx: flume::Receiver<LifecycleEvent>,
    senders: Senders,
    to_scheduler_tx: ToSchedulerTx,
    limits: Limits,
    mm: MmDispatch,
    /// Requests parked in `Encoding` while an MM worker processes their media;
    /// resumed by `MmEncoded` / `MmFailed`. Only this thread touches it, so no
    /// lock.
    pending_mm: HashMap<Rid, Request>,
    in_flight: HashMap<Rid, InFlight>,
    /// At most one abort per admitted generation. Retry without admitting more
    /// work when the reserved control slots are temporarily occupied.
    pending_aborts: VecDeque<SchedulerRequest>,
    pending_deregistrations: VecDeque<Rid>,
    shutdown: flume::Receiver<()>,
    metrics: Option<std::sync::Arc<crate::metrics::FrontendMetrics>>,
}

struct InFlight {
    sink: ResponseSink,
    is_generation: bool,
    sent_to_scheduler: bool,
    abort_queued: bool,
}

impl Intake {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tok_manager_rx: flume::Receiver<TmEvent>,
        lifecycle_rx: flume::Receiver<LifecycleEvent>,
        senders: Senders,
        to_scheduler_tx: ToSchedulerTx,
        limits: Limits,
        mm: MmDispatch,
        shutdown: flume::Receiver<()>,
        metrics: Option<std::sync::Arc<crate::metrics::FrontendMetrics>>,
    ) -> Self {
        Self {
            tok_manager_rx,
            lifecycle_rx,
            senders,
            to_scheduler_tx,
            limits,
            mm,
            pending_mm: HashMap::new(),
            in_flight: HashMap::new(),
            pending_aborts: VecDeque::new(),
            pending_deregistrations: VecDeque::new(),
            shutdown,
            metrics,
        }
    }
}

/// Which lane produced the next item.
enum Lane {
    Lifecycle(LifecycleEvent),
    Event(TmEvent),
    Retry,
}

impl Runnable for Intake {
    fn run(mut self) {
        loop {
            self.flush_pending_aborts();
            self.flush_pending_deregistrations();
            // Select, not a drain-then-block: an abort arriving while the inbox is
            // idle must still be handled at once.
            let selector = flume::Selector::new()
                .recv(&self.lifecycle_rx, |r| r.ok().map(Lane::Lifecycle))
                .recv(&self.tok_manager_rx, |r| r.ok().map(Lane::Event))
                .recv(&self.shutdown, |_| None);
            let next = if self.pending_aborts.is_empty() && self.pending_deregistrations.is_empty()
            {
                selector.wait()
            } else {
                selector
                    .wait_timeout(Duration::from_millis(1))
                    .unwrap_or(Some(Lane::Retry))
            };
            match next {
                Some(Lane::Lifecycle(event)) => self.on_lifecycle(event),
                Some(Lane::Retry) => {}
                // A fresh request and one returning from the tokenizer pool.
                Some(Lane::Event(TmEvent::Intake(req) | TmEvent::Tokenized(req))) => {
                    self.drive(req)
                }
                Some(Lane::Event(TmEvent::Abort {
                    rid_prefix,
                    abort_all,
                    reply,
                })) => {
                    let dispatched = self.on_client_abort(&rid_prefix, abort_all);
                    let _ = reply.send(dispatched);
                }
                Some(Lane::Event(TmEvent::MmEncoded {
                    rid,
                    input_ids,
                    response_metadata,
                })) => self.on_mm_encoded(rid, input_ids, response_metadata),
                Some(Lane::Event(TmEvent::MmFailed { rid, message })) => {
                    self.on_mm_failed(rid, message)
                }
                None => {
                    // Shutdown, or the inbox closed. Drain whatever is still queued
                    // on the abort lane first: those requests are in flight on the
                    // scheduler, and the selector may report the closed inbox before
                    // it ever looks at a pending abort.
                    while let Ok(source) = self.lifecycle_rx.try_recv() {
                        self.on_lifecycle(source);
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
    fn fail(&mut self, req: &mut Request, err: Error, registered: bool) {
        // Log only server faults (500); 4xx/499/503 are expected and would spam.
        if err.http_status() == 500 {
            tracing::error!(rid = %req.rid, error = %err, "intake rejected request");
        }
        // A rejected request never reaches the scheduler drain, so purge any
        // parked MM result (no-op for the common non-mm request).
        self.mm.results.purge(req.rid.as_str());
        let _ = req.state.apply(Event::Error(err.clone()));
        let _ = req.sink.try_send(ResponseItem::Error(err)); // client may be gone
        if registered {
            self.in_flight.remove(&req.rid);
            self.deregister(req.rid.clone());
        }
    }

    /// Drive a request through its intake states until it terminates (failed or
    /// pushed to the ring), is handed to the tokenizer pool (re-entering as a
    /// `Tokenized` event), or is parked in `pending_mm` awaiting an MM worker
    /// (re-entering via `MmEncoded` / `MmFailed`). Each arm acts and advances
    /// the FSM; the loop re-dispatches. The arms are the design table's states,
    /// `Failed` the single reject path.
    fn drive(&mut self, mut req: Request) {
        // A worker may finish after its caller cancelled. Its result cannot
        // resurrect the generation after the abort has already been delivered.
        if (!matches!(req.state, RequestState::Received) && !self.in_flight.contains_key(&req.rid))
            || req.sink.is_closed()
        {
            let registered = self.in_flight.contains_key(&req.rid);
            self.fail(&mut req, Error::Disconnected, registered);
            return;
        }
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
                    if matches!(req.kind, RequestKind::Tokenize { .. }) {
                        self.push_to_tokenizer(req, false);
                        return;
                    }
                    if let Err(error) = self.register_detok(&mut req) {
                        let _ = req.state.apply(Event::Error(error));
                        continue;
                    }
                    registered = true;
                    self.in_flight.insert(
                        req.rid.clone(),
                        InFlight {
                            sink: req.sink.clone(),
                            is_generation: matches!(req.kind, RequestKind::Generate(_)),
                            sent_to_scheduler: false,
                            abort_queued: false,
                        },
                    );
                    // `validate` advanced Received → Validating; keep driving.
                }
                // Control and detokenize skip normalization (no sampling params)
                // straight to the pre-send checks; generate goes to Normalizing.
                RequestState::Validating => match &req.kind {
                    RequestKind::Control(_)
                    | RequestKind::Detokenize { .. }
                    | RequestKind::Tokenize { .. } => {
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
                    self.push_to_tokenizer(req, true);
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
                            .and_then(|()| validate_delimiter_indices(g))
                            .and_then(|()| {
                                g.positional_embed_overrides
                                    .as_ref()
                                    .map_or(Ok(()), |embeds| {
                                        embeds.validate(
                                            g.input_ids.as_ref().map_or(0, Vec::len),
                                            self.limits.hidden_size,
                                        )
                                    })
                            })
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
                        RequestKind::Tokenize { .. } => self.fail(
                            &mut req,
                            Error::Internal("tokenize service reached scheduler queue".into()),
                            registered,
                        ),
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
    /// has a home. Carries the request's decoding flags so the shard needs no
    /// back-reference to the request.
    fn register_detok(&self, req: &mut Request) -> Result<(), Error> {
        let (decode_logprob_text, skip_special_tokens, no_stop_trim) = match &req.kind {
            RequestKind::Generate(g) => (
                g.return_text_in_logprobs.unwrap_or(false),
                g.sampling_params.skip_special_tokens,
                g.sampling_params.no_stop_trim,
            ),
            RequestKind::Control(_)
            | RequestKind::Detokenize { .. }
            | RequestKind::Tokenize { .. } => (false, true, false),
        };
        self.senders
            .detok_for(&req.rid)
            .try_send(DetokMsg::Register {
                rid: req.rid.clone(),
                sink: req.sink.clone(),
                decode_logprob_text,
                skip_special_tokens,
                no_stop_trim,
                stop_texts: match &req.kind {
                    RequestKind::Generate(request) => {
                        use crate::message::types::OneOrMany;
                        let params = &request.sampling_params;
                        // Registration precedes sampling normalization. OpenAI
                        // callers may already have normalized the aliases.
                        [
                            (&params.stop, &params.stop_strs),
                            (&params.stop_regex, &params.stop_regex_strs),
                        ]
                        .into_iter()
                        .flat_map(|(raw, normalized)| match raw {
                            Some(OneOrMany::One(stop)) => std::slice::from_ref(stop),
                            Some(OneOrMany::Many(stops)) => stops.as_slice(),
                            None => normalized.as_slice(),
                        })
                        .filter(|stop| !stop.is_empty())
                        .cloned()
                        .collect()
                    }
                    _ => Vec::new(),
                },
                logprobs: match &req.kind {
                    RequestKind::Generate(request) if request.return_logprob => {
                        Some(crate::message::response::LogprobOptions {
                            top_k: request.top_logprobs_num.max(0) as u64,
                            token_ids: request.token_ids_logprob.is_some(),
                            flat: request.return_flat_raw_top_logprobs,
                            base64: request.return_flat_raw_top_logprobs_b64,
                        })
                    }
                    _ => None,
                },
                metrics: match &mut req.kind {
                    RequestKind::Generate(request) => request.metric_state.take(),
                    _ => None,
                },
                control_replies: if matches!(req.kind, RequestKind::Control(_)) {
                    self.limits.dp_size
                } else {
                    0
                },
            })
            .map_err(|error| match error {
                flume::TrySendError::Full(_) => Error::QueueFull,
                flume::TrySendError::Disconnected(_) => Error::Internal("detok shard gone".into()),
            })
    }

    fn push_to_tokenizer(&mut self, req: Request, registered: bool) {
        if let Err(error) = self.senders.tokenizer_tx.try_send(req) {
            let (mut req, error) = match error {
                flume::TrySendError::Full(req) => (req, Error::QueueFull),
                flume::TrySendError::Disconnected(req) => {
                    (req, Error::Internal("tokenizer pool gone".into()))
                }
            };
            self.fail(&mut req, error, registered);
        }
    }

    /// Hand a `Detokenize` request to its owning detok shard — the stage that
    /// answers this kind (it never touches the scheduler ring). The shard
    /// already holds this rid's sink: `register_detok` queued `Register` on the
    /// same channel from this same thread, so FIFO gives Register → Decode.
    fn push_detokenize_to_shard(&mut self, mut req: Request) {
        let RequestKind::Detokenize {
            token_ids,
            skip_special_tokens,
        } = &req.kind
        else {
            self.fail(
                &mut req,
                Error::Internal("non-detokenize request reached push_detokenize_to_shard".into()),
                true,
            );
            return;
        };
        // Infallible: `validate` rejected out-of-range ids at `Received`.
        let token_ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
        if let Err(error) = self.senders.detok_for(&req.rid).try_send(DetokMsg::Decode {
            rid: req.rid.clone(),
            token_ids,
            skip_special_tokens: *skip_special_tokens,
        }) {
            let error = match error {
                flume::TrySendError::Full(_) => Error::QueueFull,
                flume::TrySendError::Disconnected(_) => Error::Internal("detok shard gone".into()),
            };
            self.fail(&mut req, error, true);
        }
    }

    /// Push a bare control request (`[tag, rid, nil]`) onto the to_scheduler channel. The
    /// scheduler dispatches it (e.g. `GetInternalStateReq`) and replies via the
    /// from_scheduler channel as a single `Result`.
    fn push_control_to_ring(&mut self, mut req: Request) {
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
        if self
            .to_scheduler_tx
            .try_push_control(SchedulerRequest {
                header,
                ids: Bytes::new(),
            })
            .is_err()
        {
            self.fail(&mut req, Error::QueueFull, true); // registered
        }
    }

    /// An MM worker finished a parked request: fill in the final expanded
    /// `input_ids`, advance `Encoding → PreSendValidating`, and resume driving
    /// (pre-send checks → ring). No pending entry means the request was already
    /// rejected or aborted, so the result is dropped.
    fn on_mm_encoded(
        &mut self,
        rid: Rid,
        input_ids: Vec<i32>,
        response_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        let Some(mut req) = self.pending_mm.remove(&rid) else {
            tracing::debug!(rid = %rid, "mm result for unknown/finished request; dropped");
            // It will never reach the scheduler drain, so purge or leak.
            self.mm.results.purge(rid.as_str());
            return;
        };
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(input_ids);
            if let Some(metadata) = response_metadata {
                g.response_metadata.get_or_insert_default().extend(metadata);
            }
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

    fn on_lifecycle(&mut self, event: LifecycleEvent) {
        if let LifecycleEvent::Finished(rid) = event {
            self.in_flight.remove(&rid);
        } else {
            self.on_abort(event);
        }
    }

    fn on_client_abort(&mut self, rid_prefix: &str, abort_all: bool) -> bool {
        if rid_prefix.is_empty() && !abort_all {
            return false;
        }
        let rids: Vec<_> = self
            .in_flight
            .iter()
            .filter_map(|(rid, request)| {
                (request.is_generation
                    && !request.abort_queued
                    && (abort_all || rid.client_facing().starts_with(rid_prefix)))
                .then_some(rid.clone())
            })
            .collect();
        let dispatched = abort_all || !rids.is_empty();
        for rid in rids {
            let Some(request) = self.in_flight.get_mut(&rid) else {
                continue;
            };
            if request.sent_to_scheduler {
                // Keep the response registration: the scheduler's terminal
                // output carries the tokens already generated and finish reason.
                request.abort_queued = true;
                self.queue_scheduler_abort(&rid);
            } else {
                // CPU work has not reached the scheduler. Complete locally and
                // discard any late tokenizer/media result for this internal rid.
                let _ = request.sink.try_send(ResponseItem::Done(ChunkEvent {
                    rid: rid.clone(),
                    finish_reason: Some(
                        FinishKind::Abort(Box::new(AbortReason {
                            message: Some("Aborted".into()),
                            status_code: None,
                            err_type: None,
                        }))
                        .into(),
                    ),
                    ..Default::default()
                }));
                self.in_flight.remove(&rid);
                self.pending_mm.remove(&rid);
                self.mm.results.purge(rid.as_str());
                self.deregister(rid);
            }
        }
        dispatched
    }

    /// Remove local work immediately, and retain scheduler aborts until their
    /// FIFO delivery succeeds. Finished/duplicate notifications are harmless.
    fn on_abort(&mut self, source: LifecycleEvent) {
        let rid = source.rid().clone();
        let request = self.in_flight.remove(&rid);
        self.pending_mm.remove(&rid);
        self.mm.results.purge(rid.as_str());
        self.deregister(rid.clone());

        let Some(request) = request else {
            return;
        };
        if request.is_generation
            && !request.abort_queued
            && let Some(metrics) = &self.metrics
        {
            metrics.aborted();
        }
        // The guard's client is already gone; a detokenizer failure owns its
        // terminal response. Do not race that response with another error here.
        if !request.is_generation || !request.sent_to_scheduler || request.abort_queued {
            return;
        }
        self.queue_scheduler_abort(&rid);
    }

    fn queue_scheduler_abort(&mut self, rid: &Rid) {
        match ControlRequest::AbortReq(AbortReq::new(rid.as_str().to_string(), false)).encode() {
            Ok(header) => {
                self.pending_aborts.push_back(SchedulerRequest {
                    header,
                    ids: Bytes::new(),
                });
                self.flush_pending_aborts();
            }
            Err(e) => tracing::error!(rid = %rid, error = %e, "abort encode failed"),
        }
    }

    fn flush_pending_aborts(&mut self) {
        while let Some(request) = self.pending_aborts.pop_front() {
            match self.to_scheduler_tx.try_push_control(request) {
                Ok(()) => {}
                Err(Some(request)) => {
                    self.pending_aborts.push_front(request);
                    break;
                }
                Err(None) => {
                    // The scheduler has exited, so no generation remains to stop.
                    self.pending_aborts.clear();
                    break;
                }
            }
        }
    }

    fn deregister(&mut self, rid: Rid) {
        if matches!(
            self.senders
                .detok_for(&rid)
                .try_send(DetokMsg::Deregister { rid: rid.clone() }),
            Err(flume::TrySendError::Full(_))
        ) {
            self.pending_deregistrations.push_back(rid);
        }
    }

    fn flush_pending_deregistrations(&mut self) {
        // A stalled shard must not prevent another shard's cleanup, nor block
        // delivery of scheduler aborts on the next turn of the intake loop.
        for _ in 0..self.pending_deregistrations.len() {
            let Some(rid) = self.pending_deregistrations.pop_front() else {
                break;
            };
            self.deregister(rid);
        }
    }

    /// Serialize the tokenized request to its `TokenizedGenerateReqInput` wire and
    /// push it onto the to_scheduler channel for the scheduler. On backpressure, fail it.
    fn push_to_ring(&mut self, mut req: Request) {
        if !self.pending_aborts.is_empty() {
            self.fail(&mut req, Error::QueueFull, true);
            return;
        }
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

        if let RequestKind::Generate(g) = &mut req.kind {
            // The wire buffers now own the scheduler input. Move response-only
            // state to the detokenizer before the scheduler can emit a chunk.
            let prepared = DetokMsg::Prepared {
                rid: req.rid.clone(),
                response_metadata: g.response_metadata.take(),
                prompt_token_ids: if g.return_prompt_token_ids {
                    g.input_ids.take().map(Into::into)
                } else {
                    None
                },
                sampling_params: Box::new(std::mem::take(&mut g.sampling_params)),
                dispatch_finished_ts: self
                    .metrics
                    .as_ref()
                    .map(|_| crate::metrics::realtime_seconds()),
            };
            if let Err(error) = self.senders.detok_for(&req.rid).try_send(prepared) {
                let error = match error {
                    flume::TrySendError::Full(_) => Error::QueueFull,
                    flume::TrySendError::Disconnected(_) => {
                        Error::Internal("detok shard gone".into())
                    }
                };
                self.fail(&mut req, error, true);
                return;
            }
        }

        if !self
            .to_scheduler_tx
            .try_push(SchedulerRequest { header, ids })
        {
            self.fail(&mut req, Error::QueueFull, true); // registered
        } else if let Some(request) = self.in_flight.get_mut(&req.rid) {
            request.sent_to_scheduler = true;
        }
        // On success the scheduler owns the request (response arrives by rid); we
        // drop our `Request` here — the detok shard holds the sink.
    }
}

#[cfg(test)]
#[path = "to_scheduler_tests.rs"]
mod tests;
