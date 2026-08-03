//! TokenizerManager — ingress side.
//!
//! [`Ingress`] is a single-consumer stage draining one inbox fed by both the API
//! server (fresh requests) and the Tokenizer pool (returned requests). It owns
//! the request while driving the ingress FSM and hands it off by *moving* it to
//! the next stage; nothing here is shared, so no locks.
//!
//! Edges driven here (from the design table):
//!   Received      → Validating
//!   Validating    → Normalizing   (generate: sampling-param normalize/verify)
//!   Validating    → PreSendValidating   (control: no tokenize, no sampling params)
//!   Normalizing   → {Encoding | Tokenizing | PreSendValidating}  (by ValidationOutcome)
//!   Tokenizing    → PreSendValidating   (on TokenizeDone, when the request returns)
//!   PreSendValidating → Queued          (checks needing the tokenized length)
//!   Queued        → ring                (handed to the scheduler)
//!
//! The egress edges (Streaming/Finalizing/Completed) are driven on the egress
//! side (see `egress` + `detokenizer`).

use std::collections::HashMap;

use bytes::Bytes;

use crate::error::Error;
use crate::fsm::{Event, RequestState, ValidationOutcome};
use crate::ids::Rid;

use crate::message::{
    AbortReq, ControlRequest, DetokMsg, EgressItem, GenerateRequest, IngressMsg, MmRequest,
    Request, RequestKind,
};
use crate::ring::IngressProducer;
use crate::runtime::{Runnable, ServerArgs};
use crate::tokenizer_manager::{AbortSource, Senders, TmEvent};

/// Ingress FSM dispatcher stage. Owns its inbox + downstream handles, so the
/// runtime spawns it as a [`Runnable`] rather than calling a free `run_*` fn
/// with positional arguments.
pub struct Ingress {
    rx: flume::Receiver<TmEvent>,
    /// Unbounded abort lane (see [`Senders::abort`]). Selected against `rx` so an
    /// abort is handled promptly even while the bounded inbox is saturated.
    abort_rx: flume::Receiver<AbortSource>,
    senders: Senders,
    ingress: IngressProducer,
    limits: Limits,
    /// Whether the model is multimodal (an MM worker pool consumes `mm_tx`).
    /// When false, mm fields on a request are silently ignored — the exact
    /// Python `TokenizerManager` behavior (`mm_processor is None` skips the MM
    /// block without error).
    mm_enabled: bool,
    /// → MM worker pool (spawned via `Server.start_mm_workers`).
    mm_tx: flume::Sender<MmRequest>,
    /// Requests parked in `Encoding` while an MM worker processes their
    /// media; resumed by `MmEncoded` / `MmFailed`. Only this (single) thread
    /// touches it, so no lock.
    pending_mm: HashMap<Rid, Request>,
    /// Native MM results sidecar — purged here when a late result arrives for
    /// a request that is no longer parked (it would otherwise leak: only the
    /// scheduler drain pops entries).
    mm_sidecar: crate::mm::Sidecar,
    shutdown: flume::Receiver<()>,
}

/// Longest client-supplied rid accepted. It keys the detok table and travels on
/// every chunk, so its length is a recurring cost; Python mints 32-byte uuid hex.
const MAX_RID_LEN: usize = 128;

/// What ingress admits, resolved once at boot from the scheduler's `server_args`.
/// A struct rather than more positional `new` arguments — these grew from two to
/// six, and every one of them is a `u64`/`bool` that would be trivial to swap at
/// a call site.
///
/// NOT `Default`-able on purpose. `vocab_size` and `context_len` are mandatory,
/// and their zero value is the most restrictive setting there is — a derived
/// `Default` would silently build limits that reject every request rather than
/// failing loudly. Tests construct these explicitly (see `test_limits`).
#[derive(Clone, Debug)]
pub struct Limits {
    /// Token-ids-in mode: a generate request must arrive already tokenized.
    pub skip_tokenizer_init: bool,
    /// `model_config.vocab_size`; bounds client-supplied token ids. Mandatory —
    /// [`ServerArgs::validate_mandatory`](crate::runtime::ServerArgs) rejects a
    /// boot without it, so ingress can check unconditionally.
    pub vocab_size: u64,
    /// `model_config.context_len`, the ceiling for input + `max_new_tokens`.
    /// Mandatory, as above.
    pub context_len: u64,
    /// Output slots reserved on top of the input (eagle draft tokens).
    pub num_reserved_tokens: u64,
    /// Clamp `max_new_tokens` to what fits instead of rejecting the request.
    pub allow_auto_truncate: bool,
    /// Whether the server can produce hidden states at all.
    pub enable_return_hidden_states: bool,
}

impl TryFrom<&ServerArgs> for Limits {
    type Error = Error;

    fn try_from(sa: &ServerArgs) -> Result<Self, Self::Error> {
        Ok(Self {
            skip_tokenizer_init: sa.skip_tokenizer_init,
            vocab_size: sa
                .model_config
                .vocab_size
                .ok_or_else(|| Error::Validation("vocab_size missing".into()))?,
            context_len: sa
                .model_config
                .context_len
                .ok_or_else(|| Error::Validation("context_len missing".into()))?,
            num_reserved_tokens: sa.num_reserved_tokens,
            allow_auto_truncate: sa.allow_auto_truncate,
            enable_return_hidden_states: sa.enable_return_hidden_states,
        })
    }
}

impl Ingress {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rx: flume::Receiver<TmEvent>,
        abort_rx: flume::Receiver<AbortSource>,
        senders: Senders,
        ingress: IngressProducer,
        limits: Limits,
        mm_enabled: bool,
        mm_tx: flume::Sender<MmRequest>,
        mm_sidecar: crate::mm::Sidecar,
        shutdown: flume::Receiver<()>,
    ) -> Self {
        Self {
            rx,
            abort_rx,
            senders,
            ingress,
            limits,
            mm_enabled,
            mm_tx,
            pending_mm: HashMap::new(),
            mm_sidecar,
            shutdown,
        }
    }
}

/// Which lane produced the next item.
enum Lane {
    Abort(AbortSource),
    Event(TmEvent),
}

impl Runnable for Ingress {
    fn run(mut self) {
        loop {
            // Select, not a drain-then-block: an abort arriving while the inbox is
            // idle must still be handled at once.
            let next = flume::Selector::new()
                .recv(&self.abort_rx, |r| r.ok().map(Lane::Abort))
                .recv(&self.rx, |r| r.ok().map(Lane::Event))
                .recv(&self.shutdown, |_| None)
                .wait();
            match next {
                Some(Lane::Abort(rid)) => self.on_abort(rid),
                // A fresh request and one returning from the tokenizer pool.
                Some(Lane::Event(TmEvent::Ingress(req) | TmEvent::Tokenized(req))) => {
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

impl Ingress {
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
            tracing::error!(rid = %req.rid, error = %err, "ingress rejected request");
        }
        // A rejected request never reaches the scheduler drain: purge any
        // parked MM result (no-op for the common non-mm request).
        self.mm_sidecar.purge(req.rid.as_str());
        let _ = req.state.apply(Event::Error(err.clone()));
        let _ = req.sink.try_send(EgressItem::Error(err)); // client may be gone
        if registered {
            let _ = self.senders.detok_for(&req.rid).send(DetokMsg::Deregister {
                rid: req.rid.clone(),
            });
        }
    }

    /// Drive a request through its ingress states until it terminates (failed or
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
                            // Multimodal (and MM workers are up): the native MM
                            // pipeline produces the final input_ids — even for
                            // pre-tokenized prompts (placeholder expansion), the
                            // same precedence as the Python TokenizerManager.
                            Ok(()) if self.mm_enabled && g.has_multimodal() => {
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
                    // Full channel = the MM pool can't keep up → backpressure,
                    // same as a full ingress ring. Disconnected = pool gone.
                    if let Err(e) = self.mm_tx.try_send(msg) {
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
                    if let Err(err) = self.senders.tok.send(req) {
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
                // Unreachable (egress states never reach here). Reject via `fail`/
                // return (not apply + continue, which would spin on a terminal state).
                other => {
                    self.fail(
                        &mut req,
                        Error::Internal(format!("unexpected ingress state: {other:?}")),
                        registered,
                    );
                    return;
                }
            }
        }
    }

    /// Register the egress sink with the owning detok shard (by id) so the response
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

    /// Push a bare control request (`[tag, rid, nil]`) onto the ingress ring. The
    /// scheduler dispatches it (e.g. `GetInternalStateReq`) and replies via the
    /// egress ring as a single `Result`.
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
        if !self.ingress.try_push(IngressMsg {
            header,
            ids: Bytes::new(),
        }) {
            self.fail(&mut req, Error::QueueFull, true); // registered
        }
    }

    /// An MM worker finished a parked request: fill the final expanded
    /// `input_ids`, advance `Encoding → PreSendValidating`, and resume driving
    /// (pre-send checks → ring): expanded image tokens count against the same
    /// input + `max_new_tokens` ceiling as tokenized text.
    /// No pending entry means the request was already rejected/aborted — the
    /// result is dropped (its sidecar entry is popped by the abort path /
    /// never attached).
    fn on_mm_encoded(&mut self, rid: Rid, input_ids: Vec<i32>) {
        let Some(mut req) = self.pending_mm.remove(&rid) else {
            tracing::debug!(rid = %rid, "mm result for unknown/finished request; dropped");
            // The request will never reach the scheduler drain, so its
            // sidecar entry (if any) must be purged here or it leaks.
            self.mm_sidecar.purge(rid.as_str());
            return;
        };
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(input_ids);
        }
        let _ = req.state.apply(Event::EncodeDone); // Encoding → PreSendValidating
        self.drive(req);
    }

    /// An MM worker failed a parked request (bad URL, processor error):
    /// reject it back to the client, mirroring the Python TokenizerManager's
    /// per-request exception → 400 behavior.
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
    /// A request parked in `pending_mm` is cancelled here: the entry is
    /// removed, so the worker's late result lands in `on_mm_encoded`'s
    /// no-entry branch, which purges the sidecar — no generation runs for
    /// output nobody will read.
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
                if !self.ingress.try_push(IngressMsg {
                    header,
                    ids: Bytes::new(),
                }) {
                    tracing::error!(
                        rid = %rid,
                        "abort dropped: ingress ring full; the scheduler keeps generating \
                         for this request until it finishes on its own"
                    );
                }
            }
            Err(e) => tracing::error!(rid = %rid, error = %e, "abort encode failed"),
        }
    }

    /// Serialize the tokenized request to its `TokenizedGenerateReqInput` wire and
    /// push it onto the ingress ring for the scheduler. On backpressure, fail it.
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

        if !self.ingress.try_push(IngressMsg { header, ids }) {
            self.fail(&mut req, Error::QueueFull, true); // registered
        }
        // On success the scheduler owns the request (egress arrives by rid); we
        // drop our `Request` here — the detok shard holds the sink.
    }
}

/// `Received → Validating` + admissibility check. Under `skip_tokenizer_init` a
/// generate request must already carry token ids (no tokenizer to byte-encode
/// text); control requests carry none and are exempt.
fn validate(req: &mut Request, limits: &Limits) -> Result<(), Error> {
    let (skip_tokenizer_init, vocab_size) = (limits.skip_tokenizer_init, limits.vocab_size);
    let _ = req
        .state
        .apply(Event::Validated(ValidationOutcome::NeedsTokenize));

    // The rid is the request's identity everywhere downstream: it keys the detok
    // table, and it rides on EVERY chunk of EVERY decode step. An unbounded
    // client-supplied rid is therefore a per-step cost, not a one-off. Python's is
    // a 32-byte uuid hex, so this is generous.
    // Measured on the CLIENT-facing form: the uniquifier `Rid::from_client` appends
    // is this server's own overhead, and charging the client for bytes it did not
    // send would reject a rid exactly at the documented limit.
    let client_rid_len = req.rid.client_facing().len();
    if client_rid_len > MAX_RID_LEN {
        return Err(Error::Validation(format!(
            "rid is {client_rid_len} bytes, over the {MAX_RID_LEN}-byte limit"
        )));
    }
    if skip_tokenizer_init
        && matches!(&req.kind, RequestKind::Generate(g) if !g.already_tokenized())
    {
        // `Validation` (400), not `Tokenize` (500): the client sent a request this
        // server cannot serve, which is their error to fix — Python 400s it too.
        return Err(Error::Validation(
            "skip_tokenizer_init is set: request must provide input_ids".into(),
        ));
    }

    // Client-supplied token ids must be in-vocabulary: an out-of-range id
    // reaches the embedding lookup and kills the scheduler process, so 400
    // here instead — mirroring the Python `TokenizerManager` validation.
    if let RequestKind::Generate(g) = &req.kind {
        if let Some(ids) = &g.input_ids {
            for &id in ids {
                if id < 0 || id as u64 >= vocab_size {
                    return Err(Error::Validation(format!(
                        "input_ids contains out-of-vocabulary token id {id}; \
                         valid range is [0, {vocab_size})"
                    )));
                }
            }
        }
        if let Some(ids) = &g.token_ids_logprob {
            for &id in ids {
                if id < 0 || id as u64 >= vocab_size {
                    return Err(Error::Validation(format!(
                        "token_ids_logprob contains out-of-vocabulary token id \
                         {id}; valid range is [0, {vocab_size})"
                    )));
                }
            }
        }
    }

    // Detokenize ids must fit the shard's `&[u32]` decode domain. No vocab
    // bound — parity with the retired direct decode service: an unknown id is
    // the tokenizer's error to report, and nothing here reaches the scheduler's
    // embedding lookup.
    if let RequestKind::Detokenize { token_ids } = &req.kind {
        for &id in token_ids {
            if u32::try_from(id).is_err() {
                return Err(Error::Validation(format!("Token ID {id} is out of range")));
            }
        }
    }

    // The scheduler only computes hidden states when launched for it, so without
    // this the request would 200 with `meta_info.hidden_states` silently absent
    // (Python `TokenizerManager._validate_one_request`).
    if !limits.enable_return_hidden_states
        && matches!(&req.kind, RequestKind::Generate(g) if g.return_hidden_states)
    {
        return Err(Error::Validation(
            "The server is not configured to return the hidden states. \
             Please set `--enable-return-hidden-states` to enable this feature."
                .into(),
        ));
    }

    Ok(())
}

/// The context-window checks that need the tokenized length, mirroring Python
/// `TokenizerManager._validate_one_request`: the input alone must fit, and then
/// input + `max_new_tokens` must fit. Without them the scheduler silently clamps
/// and the client gets a 200 with a truncated completion instead of an actionable
/// 400.
///
/// Under `allow_auto_truncate` both clamp instead of rejecting — the launch flag
/// opted into that.
fn check_total_tokens(g: &mut GenerateRequest, limits: &Limits) -> Result<(), Error> {
    let max_req_len = limits.context_len;
    // Python counts the reserved slots as part of the input, so a request can be
    // rejected for them even when the prompt alone fits.
    let input_len =
        g.input_ids.as_ref().map_or(0, |ids| ids.len()) as u64 + limits.num_reserved_tokens;

    // Input length first, and unconditionally: `max_new_tokens: null` means "no
    // cap", which must not disable this. Python's comparison is `>=` — a prompt
    // that exactly fills the window leaves no room to generate.
    if input_len >= max_req_len {
        if !limits.allow_auto_truncate {
            return Err(Error::Validation(format!(
                "The input ({input_len} tokens) is longer than the model's context \
                 length ({max_req_len} tokens)."
            )));
        }
        if let Some(ids) = &mut g.input_ids {
            ids.truncate(max_req_len as usize);
        }
    }
    let input_len =
        g.input_ids.as_ref().map_or(0, |ids| ids.len()) as u64 + limits.num_reserved_tokens;

    let Some(max_new_tokens) = g.sampling_params.max_new_tokens else {
        return Ok(()); // no cap requested → nothing to add to the input length
    };
    let total = input_len.saturating_add(max_new_tokens.max(0) as u64);
    if total <= max_req_len {
        return Ok(());
    }
    if !limits.allow_auto_truncate {
        return Err(Error::Validation(format!(
            "Requested token count exceeds the model's maximum context length of \
             {max_req_len} tokens. You requested a total of {total} tokens: {input_len} \
             tokens from the input messages and {max_new_tokens} tokens for the \
             completion. Please reduce the number of tokens in the input messages or \
             the completion to fit within the limit."
        )));
    }
    let clamped = max_req_len.saturating_sub(input_len) as i64;
    // Re-check what the clamp can break. `verify` already ran (in Normalizing), so
    // lowering `max_new_tokens` here can leave `min_new_tokens > max_new_tokens` —
    // and `is_normalized: true` stops the scheduler from re-verifying, so nothing
    // downstream would catch it. Python validates before it verifies; we can't
    // reorder the FSM, so we re-assert the one invariant the clamp can violate.
    if g.sampling_params.min_new_tokens > clamped {
        return Err(Error::Validation(format!(
            "min_new_tokens must be in [0, max_new_tokens({clamped})], got {}",
            g.sampling_params.min_new_tokens
        )));
    }
    g.sampling_params.max_new_tokens = Some(clamped);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::RequestState;
    use crate::message::{EgressSink, GenerateRequest, SamplingParams};
    use crate::ring::{IngressConsumer, ingress_ring};
    use tokio::sync::mpsc;

    /// An `Ingress` plus its detok-shard receiver, ring consumer (keep alive —
    /// dropping it closes the ring → false QueueFull), tm inbox sender, and the
    /// mm-bridge receiver (keep alive — dropping it makes mm submits fail).
    fn make_ingress() -> (
        Ingress,
        flume::Receiver<DetokMsg>,
        IngressConsumer,
        flume::Sender<TmEvent>,
        flume::Receiver<MmRequest>,
    ) {
        make_ingress_with(test_limits())
    }

    fn make_ingress_with_abort(
        abort_rx: flume::Receiver<AbortSource>,
    ) -> (
        Ingress,
        flume::Receiver<DetokMsg>,
        IngressConsumer,
        flume::Sender<TmEvent>,
        flume::Receiver<MmRequest>,
    ) {
        make_ingress_inner(test_limits(), abort_rx)
    }

    fn make_ingress_with(
        limits: Limits,
    ) -> (
        Ingress,
        flume::Receiver<DetokMsg>,
        IngressConsumer,
        flume::Sender<TmEvent>,
        flume::Receiver<MmRequest>,
    ) {
        let (abort_tx, abort_rx) = flume::unbounded::<AbortSource>();
        std::mem::forget(abort_tx); // keep the lane open; tests end by dropping tm_tx
        make_ingress_inner(limits, abort_rx)
    }

    fn make_ingress_inner(
        limits: Limits,
        abort_rx: flume::Receiver<AbortSource>,
    ) -> (
        Ingress,
        flume::Receiver<DetokMsg>,
        IngressConsumer,
        flume::Sender<TmEvent>,
        flume::Receiver<MmRequest>,
    ) {
        let (tok_tx, _tok_rx) = flume::unbounded();
        let (detok_tx, detok_rx) = flume::unbounded();
        let senders = Senders {
            tm: flume::unbounded().0,
            abort: flume::unbounded().0,
            tok: tok_tx,
            detok: vec![detok_tx],
        };
        let (ingress_producer, consumer) = ingress_ring(16);
        let (tm_tx, tm_rx) = flume::unbounded();
        let (mm_tx, mm_rx) = flume::unbounded();
        // Keep the shutdown sender alive (leak) so its branch never fires — tests
        // end `run` by dropping `tm_tx`, not by shutdown.
        let (sd_tx, sd_rx) = flume::unbounded::<()>();
        std::mem::forget(sd_tx);
        let ingress = Ingress::new(
            tm_rx,
            abort_rx,
            senders,
            ingress_producer,
            limits,
            true,
            mm_tx,
            Default::default(),
            sd_rx,
        );
        (ingress, detok_rx, consumer, tm_tx, mm_rx)
    }

    /// Both abort sources do the same two things: drop the detok entry so no
    /// further chunk can be delivered, and tell the scheduler to stop generating.
    ///
    /// Neither releases anything, and nothing needs them to. Release ordering used
    /// to be the delicate part here — `AbortGuard::drop` releasing a rid right
    /// after enqueuing the abort ordered the SEND, not the EFFECT, so a retry of
    /// the same rid could `Register` ahead of the stale abort and be torn down by
    /// it. `Rid::from_client` removes the premise: a retry carries a different
    /// `Rid`, so no abort in flight can name it.
    #[test]
    fn every_abort_source_deregisters_and_stops_the_scheduler() {
        for source in [
            AbortSource::Guard("x".into()),
            AbortSource::Detok("x".into()),
        ] {
            let (detok_tx, detok_rx) = flume::unbounded::<DetokMsg>();
            let (ingress_producer, consumer) = ingress_ring(16);
            let (sd_tx, sd_rx) = flume::unbounded::<()>();
            std::mem::forget(sd_tx);
            let mut ingress = Ingress::new(
                flume::unbounded().1,
                flume::unbounded().1,
                Senders {
                    tm: flume::unbounded().0,
                    abort: flume::unbounded().0,
                    tok: flume::unbounded().0,
                    detok: vec![detok_tx],
                },
                ingress_producer,
                test_limits(),
                true,
                flume::unbounded().0,
                Default::default(),
                sd_rx,
            );

            ingress.on_abort(source.clone());

            assert!(
                matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "x"),
                "{source:?} must drop the detok entry",
            );
            assert_eq!(
                consumer.drain(8).headers.len(),
                1,
                "{source:?} must push an AbortReq so the scheduler stops",
            );
        }
    }

    /// A context ceiling high enough that only a test which sets one on purpose
    /// can reach it. `context_len` is mandatory now, so "no ceiling" has to be a
    /// large number rather than `None`; kept well below `u64::MAX` so the
    /// `as i64` in the auto-truncate clamp cannot go negative if a future test
    /// does reach this path.
    const NO_CONTEXT_CEILING: u64 = 1 << 40;

    /// The default test limits: a real tokenizer, vocab 1000, no context ceiling.
    /// Spelled out rather than `..Default::default()` — `Limits` deliberately has
    /// no `Default`, because a zero `vocab_size`/`context_len` would reject every
    /// request instead of behaving like "unset".
    fn test_limits() -> Limits {
        Limits {
            skip_tokenizer_init: false,
            vocab_size: 1000,
            context_len: NO_CONTEXT_CEILING,
            num_reserved_tokens: 0,
            allow_auto_truncate: false,
            enable_return_hidden_states: false,
        }
    }

    fn generate_req(id: u64, sampling_params: SamplingParams) -> Request {
        let (tx, _rx) = mpsc::channel(8);
        Request {
            rid: id.to_string().into(),
            state: RequestState::Received,
            sink: EgressSink::Local(tx),
            kind: RequestKind::Generate(Box::new(GenerateRequest {
                rid: id.to_string().into(),
                input_ids: Some(vec![1, 2, 3]),
                sampling_params,
                ..Default::default()
            })),
        }
    }

    /// `input + max_new_tokens` past the context window is an actionable 400, not a
    /// silently truncated 200 (Python `TokenizerManager._validate_one_request`).
    /// The message names both halves so the client can fix the right one.
    #[test]
    fn total_tokens_over_context_is_rejected() {
        let limits = Limits {
            context_len: 10,
            ..test_limits()
        };
        let mut g = GenerateRequest {
            input_ids: Some(vec![1, 2, 3]),
            sampling_params: SamplingParams {
                max_new_tokens: Some(100),
                ..Default::default()
            },
            ..Default::default()
        };
        let err = check_total_tokens(&mut g, &limits).unwrap_err();
        let msg = err.to_string();
        assert_eq!(err.http_status(), 400);
        assert!(msg.contains("total of 103 tokens"), "{msg}");
        assert!(msg.contains("3 tokens from the input"), "{msg}");
        assert!(msg.contains("100 tokens for the completion"), "{msg}");
        // Exactly filling the window is allowed (Python compares with `>`).
        g.sampling_params.max_new_tokens = Some(7);
        assert!(check_total_tokens(&mut g, &limits).is_ok());
        assert_eq!(g.sampling_params.max_new_tokens, Some(7), "left alone");
    }

    /// The reserved slots (eagle draft tokens) count as input, so a request can be
    /// rejected for them even when the prompt alone would fit.
    #[test]
    fn reserved_tokens_count_toward_the_limit() {
        let limits = Limits {
            context_len: 10,
            num_reserved_tokens: 5,
            ..test_limits()
        };
        let mut g = GenerateRequest {
            input_ids: Some(vec![1, 2, 3]),
            sampling_params: SamplingParams {
                max_new_tokens: Some(3), // 3 + 3 fits, but 3 + 5 + 3 does not
                ..Default::default()
            },
            ..Default::default()
        };
        let msg = check_total_tokens(&mut g, &limits).unwrap_err().to_string();
        assert!(msg.contains("8 tokens from the input"), "{msg}");
    }

    /// `--allow-auto-truncate` opts into clamping instead of rejecting; with no
    /// context length, or no `max_new_tokens` cap, there is nothing to check.
    #[test]
    fn auto_truncate_clamps_and_unknowns_skip() {
        let sp = |max_new_tokens| SamplingParams {
            max_new_tokens,
            ..Default::default()
        };
        let mut g = GenerateRequest {
            input_ids: Some(vec![1, 2, 3]),
            sampling_params: sp(Some(100)),
            ..Default::default()
        };
        let truncating = Limits {
            context_len: 10,
            allow_auto_truncate: true,
            ..test_limits()
        };
        assert!(check_total_tokens(&mut g, &truncating).is_ok());
        assert_eq!(g.sampling_params.max_new_tokens, Some(7), "clamped to fit");

        // Unknown context length → no ceiling to enforce.
        g.sampling_params = sp(Some(100));
        assert!(check_total_tokens(&mut g, &test_limits()).is_ok());
        assert_eq!(g.sampling_params.max_new_tokens, Some(100), "untouched");

        // No cap requested → nothing to add to the input length, but the input
        // itself is still checked (see `input_length_is_checked_unconditionally`).
        g.sampling_params = sp(None);
        let roomy = Limits {
            context_len: 100,
            ..test_limits()
        };
        assert!(check_total_tokens(&mut g, &roomy).is_ok());
    }

    /// `max_new_tokens: null` means "no cap", NOT "skip the checks" — the input
    /// alone must still fit. Gating the whole function on `max_new_tokens` let an
    /// over-long prompt through to the scheduler with no ingress error at all.
    /// Python compares with `>=`: a prompt that exactly fills the window leaves no
    /// room to generate.
    #[test]
    fn input_length_is_checked_unconditionally() {
        let limits = Limits {
            context_len: 3,
            ..test_limits()
        };
        let req = |max_new_tokens| GenerateRequest {
            input_ids: Some(vec![1, 2, 3]), // exactly fills a 3-token window
            sampling_params: SamplingParams {
                max_new_tokens,
                ..Default::default()
            },
            ..Default::default()
        };
        for max_new_tokens in [None, Some(1)] {
            let err = check_total_tokens(&mut req(max_new_tokens), &limits)
                .expect_err("input == context_len must be rejected (Python uses >=)");
            assert_eq!(err.http_status(), 400);
            assert!(err.to_string().contains("longer than the model's context"));
        }
        // One token shorter fits, with or without a cap.
        let mut g = GenerateRequest {
            input_ids: Some(vec![1, 2]),
            ..Default::default()
        };
        g.sampling_params.max_new_tokens = None;
        assert!(check_total_tokens(&mut g, &limits).is_ok());

        // Under auto-truncate the input is cut to fit instead of rejected.
        let truncating = Limits {
            allow_auto_truncate: true,
            ..limits.clone()
        };
        let mut g = req(None);
        assert!(check_total_tokens(&mut g, &truncating).is_ok());
        assert_eq!(
            g.input_ids.as_deref(),
            Some(&[1, 2, 3][..]),
            "fits at the cap"
        );
    }

    /// The clamp runs AFTER `verify` (which happens in `Normalizing`), so lowering
    /// `max_new_tokens` can leave `min_new_tokens > max_new_tokens`. Nothing
    /// downstream re-checks — `is_normalized: true` makes the scheduler's own
    /// verify early-return — so the clamp has to re-assert it here.
    #[test]
    fn auto_truncate_cannot_invert_min_and_max_new_tokens() {
        let limits = Limits {
            context_len: 10,
            allow_auto_truncate: true,
            ..test_limits()
        };
        let mut g = GenerateRequest {
            input_ids: Some(vec![1, 2, 3]), // clamps max_new_tokens to 7
            sampling_params: SamplingParams {
                max_new_tokens: Some(100),
                min_new_tokens: 50, // …which is below min_new_tokens
                ..Default::default()
            },
            ..Default::default()
        };
        let err = check_total_tokens(&mut g, &limits)
            .expect_err("a clamp that inverts min/max must 400, not ride the wire");
        assert_eq!(err.http_status(), 400);
        assert!(err.to_string().contains("min_new_tokens"), "{err}");

        // A clamp that keeps the invariant still clamps.
        g.sampling_params.min_new_tokens = 2;
        g.sampling_params.max_new_tokens = Some(100);
        assert!(check_total_tokens(&mut g, &limits).is_ok());
        assert_eq!(g.sampling_params.max_new_tokens, Some(7));
    }

    /// `return_hidden_states` on a server not launched for it is a 400: the
    /// scheduler never computes them, so the request would otherwise 200 with
    /// `meta_info.hidden_states` silently missing.
    #[test]
    fn hidden_states_gated_on_server_support() {
        let req = |want| {
            let mut r = generate_req(31, SamplingParams::default());
            if let RequestKind::Generate(g) = &mut r.kind {
                g.return_hidden_states = want;
            }
            r
        };
        let disabled = test_limits();
        let err = validate(&mut req(true), &disabled).unwrap_err();
        assert_eq!(err.http_status(), 400);
        assert!(
            err.to_string().contains("--enable-return-hidden-states"),
            "message must name the flag: {err}"
        );
        // Not asking for them (the client sent `false`, or sent nothing and
        // `into_requests` resolved the default), or asking on a server that
        // supports them, is fine.
        assert!(validate(&mut req(false), &disabled).is_ok());
        let enabled = Limits {
            enable_return_hidden_states: true,
            ..test_limits()
        };
        assert!(validate(&mut req(true), &enabled).is_ok());
    }

    /// End-to-end through `drive`: an over-context request is rejected on the way
    /// to the ring, after registration — so it must be deregistered, not leaked.
    #[test]
    fn over_context_request_deregisters_and_never_reaches_the_ring() {
        let (mut ingress, detok_rx, consumer, _tm_tx, _mm_rx) = make_ingress_with(Limits {
            context_len: 4,
            ..test_limits()
        });
        ingress.drive(generate_req(
            33,
            SamplingParams {
                max_new_tokens: Some(64),
                ..Default::default()
            },
        ));
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid, .. }) if rid.as_str() == "33"),
            "registered before the check",
        );
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "33"),
            "must deregister on reject",
        );
        assert!(
            consumer.drain(16).headers.is_empty(),
            "must not reach the scheduler"
        );
    }

    /// A `Detokenize` request terminates at the detok stage, and the shard must
    /// see its `Register` BEFORE its `Decode` — the shard delivers the result
    /// through the sink registered under that rid, so a `Decode` that arrives
    /// unregistered is silently dropped and the caller waits forever. Both
    /// messages ride one channel from this one thread, which is the FIFO this
    /// pins. Nothing may reach the scheduler ring.
    #[test]
    fn detokenize_flows_register_then_decode_and_skips_the_ring() {
        let (mut ingress, detok_rx, consumer, _tm_tx, _mm_rx) = make_ingress();
        let (tx, mut rx) = mpsc::channel(8);
        ingress.drive(Request {
            rid: "41".into(),
            state: RequestState::Received,
            sink: EgressSink::Local(tx),
            kind: RequestKind::Detokenize {
                token_ids: vec![7, 8, 9],
            },
        });
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid, .. }) if rid.as_str() == "41"),
            "the sink must be registered before the decode job",
        );
        assert!(
            matches!(
                detok_rx.try_recv(),
                Ok(DetokMsg::Decode { rid, token_ids })
                    if rid.as_str() == "41" && token_ids == [7, 8, 9]
            ),
            "the decode job follows, ids intact",
        );
        assert!(
            consumer.drain(16).headers.is_empty(),
            "must never reach the scheduler"
        );
        assert!(rx.try_recv().is_err(), "no egress until the shard answers");
    }

    /// Negative ids cannot decode (the shard's domain is `&[u32]`): rejected by
    /// `validate` at `Received` — an `Error` to the sink, and the shard sees
    /// NOTHING (validation runs before registration, so there is no entry to
    /// leak and no decode job to drop).
    #[test]
    fn detokenize_negative_ids_reject_before_registration() {
        let (mut ingress, detok_rx, consumer, _tm_tx, _mm_rx) = make_ingress();
        let (tx, mut rx) = mpsc::channel(8);
        ingress.drive(Request {
            rid: "43".into(),
            state: RequestState::Received,
            sink: EgressSink::Local(tx),
            kind: RequestKind::Detokenize {
                token_ids: vec![1, -1],
            },
        });
        let Ok(EgressItem::Error(err)) = rx.try_recv() else {
            panic!("sink must receive the validation error");
        };
        assert_eq!(err.http_status(), 400);
        assert!(err.to_string().contains("out of range"), "{err}");
        assert!(detok_rx.try_recv().is_err(), "shard never hears of it");
        assert!(consumer.drain(16).headers.is_empty());
    }

    /// A dropped ring push is survivable, and this pins WHY. The ring is bounded,
    /// so under load the scheduler never learns to stop and keeps generating; its
    /// chunks then arrive for a rid the detok table no longer holds and are
    /// dropped. That wastes GPU work but cannot MISDELIVER, because
    /// `Rid::from_client` guarantees no later request ever answers to that rid.
    /// The detok entry is dropped either way — that is the half that must not
    /// depend on the ring.
    ///
    /// Ring capacity 1: the first abort pushes, the second finds it full.
    #[test]
    fn abort_deregisters_even_when_the_ring_push_is_dropped() {
        let (tok_tx, _tok_rx) = flume::unbounded();
        let (detok_tx, detok_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded::<AbortSource>();
        let senders = Senders {
            tm: flume::unbounded().0,
            abort: abort_tx,
            tok: tok_tx,
            detok: vec![detok_tx],
        };
        let (producer, _consumer) = ingress_ring(1);
        let (_tm_tx, tm_rx) = flume::unbounded();
        let (sd_tx, sd_rx) = flume::unbounded::<()>();
        std::mem::forget(sd_tx);
        let mut ingress = Ingress::new(
            tm_rx,
            abort_rx,
            senders,
            producer,
            test_limits(),
            true,
            flume::unbounded().0,
            Default::default(),
            sd_rx,
        );

        ingress.on_abort(AbortSource::Guard("pushed".into()));
        ingress.on_abort(AbortSource::Guard("dropped".into()));

        // Both deregisters land regardless of whether the ring accepted the push.
        for expected in ["pushed", "dropped"] {
            assert!(
                matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == expected),
                "{expected}: the detok entry must be dropped even when the ring is full",
            );
        }
    }

    /// The rid keys the detok table and rides on every chunk of every decode step,
    /// so an unbounded client-supplied one is a recurring cost, not a one-off.
    #[test]
    fn oversized_rid_is_rejected() {
        let mut req = generate_req(51, SamplingParams::default());
        req.rid = "x".repeat(MAX_RID_LEN + 1).into();
        let err = validate(&mut req, &test_limits()).expect_err("must be rejected");
        assert_eq!(err.http_status(), 400);
        assert!(err.to_string().contains("over the"), "{err}");

        // A uuid-sized rid — what Python mints — is nowhere near the cap.
        let mut req = generate_req(52, SamplingParams::default());
        req.rid = "0123456789abcdef0123456789abcdef".into();
        assert!(validate(&mut req, &test_limits()).is_ok());
    }

    /// A request rejected BEFORE `register_detok` must not send `Deregister`: the
    /// handler is a bare `table.remove(&rid)`, so it would evict whatever entry
    /// holds that key — a concurrent request's sink — leaving that client hung with
    /// no terminal frame. Python validates before it inserts, so it cannot hit this.
    #[test]
    fn pre_registration_failure_does_not_deregister() {
        // Rejected inside `validate` (out-of-vocab id), which runs before registration.
        let (mut ingress, detok_rx, _consumer, _tm_tx, _mm_rx) = make_ingress();
        let mut req = generate_req(41, SamplingParams::default());
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(vec![2_000_000_000]);
        }
        ingress.drive(req);
        assert!(
            detok_rx.try_recv().is_err(),
            "a pre-registration reject must send NOTHING to the shard — a Deregister \
             here removes a live request's sink"
        );

        // A post-registration reject still deregisters (the leak fix stays fixed).
        let (mut ingress, detok_rx, _consumer, _tm_tx, _mm_rx) = make_ingress();
        ingress.drive(generate_req(
            42,
            SamplingParams {
                top_p: 2.0, // rejected by `normalize`, after registration
                ..Default::default()
            },
        ));
        assert!(matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })));
        assert!(matches!(
            detok_rx.try_recv(),
            Ok(DetokMsg::Deregister { .. })
        ));
    }

    /// A request rejected at normalization (post-register) must not leak: the shard
    /// sees `Register` then `Deregister`. Regression for RSS growth on bad input.
    #[test]
    fn rejected_request_deregisters_from_shard() {
        let (mut ingress, detok_rx, _consumer, _tm_tx, _mm_rx) = make_ingress();
        // top_p = 2.0 is outside (0, 1], so `SamplingParams::normalize` rejects it.
        let bad = SamplingParams {
            top_p: 2.0,
            ..Default::default()
        };
        ingress.drive(generate_req(7, bad));

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid, .. }) if rid.as_str() == "7"),
            "expected Register for rid 7",
        );
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "7"),
            "expected Deregister for rid 7 (leak fix)",
        );
        assert!(
            detok_rx.try_recv().is_err(),
            "no further shard messages — registration fully cleaned up",
        );
    }

    /// Regression: an out-of-vocabulary client token id must be rejected at
    /// ingress with a 400 — passed through, it reaches the embedding lookup
    /// and kills the scheduler process (`make_ingress` bounds vocab at 1000).
    #[test]
    fn out_of_vocab_input_ids_rejected() {
        let (mut ingress, detok_rx, _consumer, _tm_tx, _mm_rx) = make_ingress();
        let mut req = generate_req(21, SamplingParams::default());
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(vec![1, 2_000_000_000]);
        }
        ingress.drive(req);
        // Rejected before registration: the only shard message is nothing at
        // all, or a Deregister if registration happened first — never a push.
        match detok_rx.try_recv() {
            Err(_) => {}
            Ok(DetokMsg::Deregister { .. }) => {}
            Ok(_) => panic!("out-of-vocab request must not be admitted"),
        }
    }

    /// Same guard for negative ids and for `token_ids_logprob` entries.
    #[test]
    fn negative_and_logprob_token_ids_rejected() {
        let (mut ingress, detok_rx, _consumer, _tm_tx, _mm_rx) = make_ingress();
        let mut req = generate_req(22, SamplingParams::default());
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(vec![-1]);
        }
        ingress.drive(req);
        match detok_rx.try_recv() {
            Err(_) | Ok(DetokMsg::Deregister { .. }) => {}
            Ok(_) => panic!("negative token id must not be admitted"),
        }

        let (mut ingress, detok_rx, _consumer, _tm_tx, _mm_rx) = make_ingress();
        let mut req = generate_req(23, SamplingParams::default());
        if let RequestKind::Generate(g) = &mut req.kind {
            g.token_ids_logprob = Some(vec![999_999]);
        }
        ingress.drive(req);
        match detok_rx.try_recv() {
            Err(_) | Ok(DetokMsg::Deregister { .. }) => {}
            Ok(_) => panic!("out-of-vocab token_ids_logprob must not be admitted"),
        }
    }

    /// A valid request is registered and handed onward — never deregistered.
    #[test]
    fn admitted_request_keeps_registration() {
        let (mut ingress, detok_rx, _consumer, _tm_tx, _mm_rx) = make_ingress();
        // Empty map → all sampling defaults, passes normalization.
        ingress.drive(generate_req(9, SamplingParams::default()));

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid, .. }) if rid.as_str() == "9"),
            "expected Register for rid 9",
        );
        assert!(
            detok_rx.try_recv().is_err(),
            "admitted request must not be deregistered",
        );
    }

    /// A pool return in `Failed` state (failed encode) is rejected via the same
    /// path and deregistered, not leaked.
    #[test]
    fn tokenize_failure_deregisters_via_ingress() {
        let (ingress, detok_rx, _consumer, tm_tx, _mm_rx) = make_ingress();
        // The pool marks a failed encode as `Failed(err)` before returning it.
        let mut req = generate_req(11, SamplingParams::default());
        let _ = req
            .state
            .apply(Event::Error(Error::Tokenize("boom".into())));
        tm_tx.send(TmEvent::Tokenized(req)).unwrap();
        // Close the inbox so the run loop returns after draining the one event.
        drop(tm_tx);
        ingress.run();

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "11"),
            "tokenize failure must deregister rid 11",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }

    /// An abort deregisters (by the id hashed from the rid string), so a request
    /// aborted before any terminal chunk can't leak.
    #[test]
    fn abort_deregisters_from_shard() {
        // Aborts arrive on their own unbounded lane now, not the request inbox.
        let (abort_tx, abort_rx) = flume::unbounded::<AbortSource>();
        let (ingress, detok_rx, _consumer, tm_tx, _mm_rx) = make_ingress_with_abort(abort_rx);
        abort_tx.send(AbortSource::Guard("rid-13".into())).unwrap();
        drop(abort_tx);
        drop(tm_tx);
        ingress.run();

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "rid-13"),
            "abort must deregister by rid",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }

    /// A successful pool return (Queued, ids filled) is pushed to the ring, not
    /// rejected; its registration is untouched.
    #[test]
    fn tokenized_return_pushes_without_deregister() {
        let (ingress, detok_rx, _consumer, tm_tx, _mm_rx) = make_ingress();
        let mut req = generate_req(15, SamplingParams::default());
        // Simulate a successful pool return: ids filled, PreSendValidating.
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(vec![1, 2, 3]);
        }
        req.state = RequestState::PreSendValidating;
        tm_tx.send(TmEvent::Tokenized(req)).unwrap();
        drop(tm_tx);
        ingress.run();

        // Pushed to the ring; the shard sees nothing.
        assert!(
            detok_rx.try_recv().is_err(),
            "a queued pool-return must be pushed, not touch the shard",
        );
    }

    /// If the pool is gone, a request needing tokenization is rejected +
    /// deregistered, not silently dropped.
    #[test]
    fn tokenize_pool_gone_deregisters() {
        // `make_ingress` drops the tok receiver, so `tok.send` fails.
        let (mut ingress, detok_rx, _consumer, _tm_tx, _mm_rx) = make_ingress();
        // No ids → NeedsTokenize → Tokenizing branch.
        let mut req = generate_req(21, SamplingParams::default());
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = None;
        }
        ingress.drive(req);

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid, .. }) if rid.as_str() == "21"),
            "expected Register for rid 21",
        );
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "21"),
            "pool-gone hand-off must deregister rid 21",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }

    /// Build a generate request carrying an image. The parked entry and the
    /// `MmEncoded` resume path agree on identity via the rid string.
    fn mm_generate_req(rid: &str) -> Request {
        let (tx, _rx) = mpsc::channel(8);
        Request {
            rid: rid.to_string().into(),
            state: RequestState::Received,
            sink: EgressSink::Local(tx),
            kind: RequestKind::Generate(Box::new(GenerateRequest {
                rid: rid.to_string().into(),
                text: Some("<image> hi".into()),
                mm: Some(Box::new(crate::message::MmData {
                    image_data: Some(rmpv::Value::from("data:image/jpeg;base64,xxxx")),
                    ..Default::default()
                })),
                ..Default::default()
            })),
        }
    }

    /// An abort while the request is parked for MM cancels it: the pending
    /// entry is removed, the worker's late result is dropped, and its parked
    /// sidecar entry is purged — no scheduler work runs for a dead client.
    #[test]
    fn abort_cancels_parked_mm_request() {
        let (mut ingress, _detok_rx, consumer, _tm_tx, mm_rx) = make_ingress();
        ingress.drive(mm_generate_req("mm-gone"));
        mm_rx.try_recv().expect("parked to mm pool");

        // The worker parks its result, as it always does before MmEncoded.
        ingress.mm_sidecar.park(
            "mm-gone".into(),
            crate::mm::MmSidecarEntry {
                features: crate::mm::FeatureStore::Inline(vec![]),
                grids: vec![],
                hashes: vec![],
                offsets: vec![],
                mrope: vec![],
                mrope_delta: 0,
            },
        );
        ingress.on_abort(AbortSource::Guard("mm-gone".to_string().into()));
        assert_eq!(consumer.drain(16).headers.len(), 1, "only the AbortReq");

        // The late result must be dropped, not queued, and the sidecar purged.
        ingress.on_mm_encoded("mm-gone".to_string().into(), vec![5, 6]);
        assert!(consumer.drain(16).headers.is_empty(), "cancelled, not queued");
        assert!(ingress.mm_sidecar.take("mm-gone").is_none(), "entry purged");
    }

    /// A multimodal request parks in `Encoding` (submitted to the mm worker
    /// pool, not the tokenizer pool, not the ring) until `MmEncoded` resumes
    /// it → ring.
    #[test]
    fn mm_request_parks_then_mm_encoded_pushes_to_ring() {
        let (mut ingress, _detok_rx, consumer, _tm_tx, mm_rx) = make_ingress();
        ingress.drive(mm_generate_req("mm-1"));

        // Submitted to the mm pool with the typed work item; nothing on the ring yet.
        let sub = mm_rx.try_recv().expect("mm pool must receive the request");
        assert_eq!(sub.rid.as_str(), "mm-1");
        assert_eq!(sub.work.text.as_deref(), Some("<image> hi"));
        assert!(sub.work.input_ids.is_none(), "no client input_ids");
        assert_eq!(
            sub.work.image_data.as_ref().and_then(|v| v.as_str()),
            Some("data:image/jpeg;base64,xxxx")
        );
        assert!(consumer.drain(16).headers.is_empty(), "parked, not queued");

        // Bridge returns the final expanded ids → pushed to the ring.
        ingress.on_mm_encoded("mm-1".to_string().into(), vec![5, 6, 7, 8]);
        let batch = consumer.drain(16);
        assert_eq!(batch.headers.len(), 1);
        assert_eq!(
            batch.lengths,
            vec![4],
            "expanded ids ride the columnar cell"
        );
    }

    /// A bridge failure rejects the parked request (deregister, no ring push).
    #[test]
    fn mm_failure_rejects_parked_request() {
        let (mut ingress, detok_rx, consumer, _tm_tx, _mm_rx) = make_ingress();
        ingress.drive(mm_generate_req("mm-2"));
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })),
            "registered before parking",
        );

        ingress.on_mm_failed("mm-2".to_string().into(), "bad image".into());
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid })
                if rid.as_str() == "mm-2"),
            "mm failure must deregister",
        );
        assert!(consumer.drain(16).headers.is_empty(), "nothing queued");
    }

    /// With no bridge attached (`mm_enabled == false`, the non-multimodal-model
    /// case), image_data is silently ignored and the request tokenizes as plain
    /// text — the Python TokenizerManager behavior when `mm_processor is None`.
    #[test]
    fn mm_fields_ignored_when_bridge_disabled() {
        let (tok_tx, tok_rx) = flume::unbounded();
        let (detok_tx, _detok_rx) = flume::unbounded();
        let senders = Senders {
            tm: flume::unbounded().0,
            abort: flume::unbounded().0,
            tok: tok_tx,
            detok: vec![detok_tx],
        };
        let (ingress_producer, _consumer) = ingress_ring(16);
        let (_tm_tx, tm_rx) = flume::unbounded();
        let (mm_tx, mm_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded::<AbortSource>();
        std::mem::forget(abort_tx);
        let (sd_tx, sd_rx) = flume::unbounded::<()>();
        std::mem::forget(sd_tx);
        let mut ingress = Ingress::new(
            tm_rx,
            abort_rx,
            senders,
            ingress_producer,
            test_limits(),
            false,
            mm_tx,
            Default::default(),
            sd_rx,
        );

        ingress.drive(mm_generate_req("mm-3"));
        assert!(
            mm_rx.try_recv().is_err(),
            "bridge disabled: nothing submitted to the mm channel",
        );
        assert!(
            tok_rx.try_recv().is_ok(),
            "request must fall through to plain tokenization",
        );
    }

    /// A late mm result for a rid that is no longer parked is dropped without
    /// panicking (e.g. hash-collision overwrite) — regression guard.
    #[test]
    fn late_mm_result_is_dropped() {
        let (mut ingress, _detok_rx, consumer, _tm_tx, _mm_rx) = make_ingress();
        ingress.on_mm_encoded("ghost".to_string().into(), vec![1]);
        ingress.on_mm_failed("ghost".to_string().into(), "boom".into());
        assert!(consumer.drain(16).headers.is_empty());
    }
}
