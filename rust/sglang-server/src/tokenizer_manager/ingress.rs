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

use bytes::Bytes;

use crate::error::Error;
use crate::fsm::{Event, RequestState, ValidationOutcome};

use crate::message::{
    AbortReq, ControlRequest, DetokMsg, EgressItem, GenerateRequest, IngressMsg, Request,
    RequestKind,
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
    pub fn new(
        rx: flume::Receiver<TmEvent>,
        abort_rx: flume::Receiver<AbortSource>,
        senders: Senders,
        ingress: IngressProducer,
        limits: Limits,
        shutdown: flume::Receiver<()>,
    ) -> Self {
        Self {
            rx,
            abort_rx,
            senders,
            ingress,
            limits,
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
    fn run(self) {
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
        let _ = req.state.apply(Event::Error(err.clone()));
        let _ = req.sink.try_send(EgressItem::Error(err)); // client may be gone
        if registered {
            let _ = self.senders.detok_for(&req.rid).send(DetokMsg::Deregister {
                rid: req.rid.clone(),
            });
        }
    }

    /// Drive a request through its ingress states until it terminates (failed or
    /// pushed to the ring) or is handed to the tokenizer pool (re-entering as a
    /// `Tokenized` event). Each arm acts and advances the FSM; the loop
    /// re-dispatches. The arms are the design table's states, `Failed` the single
    /// reject path.
    fn drive(&self, mut req: Request) {
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
                // Control skips normalization (no sampling params) straight to the
                // pre-send checks; generate goes to Normalizing.
                RequestState::Validating => match &req.kind {
                    RequestKind::Control(_) => {
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
                            // Unreachable (control never reaches here); reject so a
                            // bug can't leak/hang a registered request.
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
                // Push the wire message (control frame or generate payload) to the ring.
                RequestState::Queued => {
                    // `matches!` reads the discriminant without holding a borrow,
                    // so `req` can be moved into the push below.
                    if matches!(req.kind, RequestKind::Generate(_)) {
                        self.push_to_ring(req);
                    } else {
                        self.push_control_to_ring(req);
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
            RequestKind::Control(_) => (false, false),
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

    /// Client disconnected (or a detok terminal): deregister the sink, then push an
    /// `AbortReq(rid)` so the scheduler stops generating for it.
    ///
    /// A failed push is logged, not retried: the scheduler keeps generating and the
    /// chunks arrive for a rid no longer in the detok table, where they are dropped.
    /// That wastes GPU work until the request finishes on its own, but it cannot be
    /// misdelivered — the rid is unique to this request for the process's lifetime
    /// ([`Rid::from_client`]), so no later request can ever answer to it.
    fn on_abort(&self, source: AbortSource) {
        let rid = source.rid().clone();
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
