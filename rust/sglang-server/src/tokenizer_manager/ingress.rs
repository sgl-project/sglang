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
use crate::ids::RidHash;
use crate::message::{
    AbortReq, ControlRequest, DetokMsg, EgressItem, GenerateRequest, IngressMsg, Request,
    RequestKind,
};
use crate::ring::IngressProducer;
use crate::runtime::Runnable;
use crate::tokenizer_manager::{Senders, TmEvent, recv};

/// Ingress FSM dispatcher stage. Owns its inbox + downstream handles, so the
/// runtime spawns it as a [`Runnable`] rather than calling a free `run_*` fn
/// with positional arguments.
pub struct Ingress {
    rx: flume::Receiver<TmEvent>,
    senders: Senders,
    ingress: IngressProducer,
    limits: Limits,
    shutdown: flume::Receiver<()>,
}

/// What ingress admits, resolved once at boot from the scheduler's `server_args`.
/// A struct rather than more positional `new` arguments — these grew from two to
/// six, and every one of them is an `Option<u64>`/`bool` that would be trivial to
/// swap at a call site.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    /// Token-ids-in mode: a generate request must arrive already tokenized.
    pub skip_tokenizer_init: bool,
    /// `model_config.vocab_size`; bounds client-supplied token ids
    /// (`None` → unknown, checks skipped).
    pub vocab_size: Option<u64>,
    /// `model_config.context_len`, the ceiling for input + `max_new_tokens`
    /// (`None` → unknown, check skipped).
    pub context_len: Option<u64>,
    /// Output slots reserved on top of the input (eagle draft tokens).
    pub num_reserved_tokens: u64,
    /// Clamp `max_new_tokens` to what fits instead of rejecting the request.
    pub allow_auto_truncate: bool,
    /// Whether the server can produce hidden states at all.
    pub enable_return_hidden_states: bool,
}

impl Limits {
    pub fn from_server_args(sa: &crate::runtime::ServerArgs) -> Self {
        Self {
            skip_tokenizer_init: sa.skip_tokenizer_init,
            vocab_size: sa.model_config.vocab_size,
            context_len: sa.model_config.context_len,
            num_reserved_tokens: sa.num_reserved_tokens,
            allow_auto_truncate: sa.allow_auto_truncate,
            enable_return_hidden_states: sa.enable_return_hidden_states,
        }
    }
}

impl Ingress {
    pub fn new(
        rx: flume::Receiver<TmEvent>,
        senders: Senders,
        ingress: IngressProducer,
        limits: Limits,
        shutdown: flume::Receiver<()>,
    ) -> Self {
        Self {
            rx,
            senders,
            ingress,
            limits,
            shutdown,
        }
    }
}

impl Runnable for Ingress {
    fn run(self) {
        while let Some(ev) = recv(&self.rx, &self.shutdown) {
            match ev {
                // A fresh request and one returning from the tokenizer pool.
                TmEvent::Ingress(req) | TmEvent::Tokenized(req) => self.drive(req),
                TmEvent::Abort(rid) => self.on_abort(rid),
            }
        }
    }
}

impl Ingress {
    /// Reject a request: → `Failed`, notify the client, deregister (unconditional
    /// — a no-op when nothing was registered).
    fn fail(&self, req: &mut Request, err: Error) {
        let id = req.rid_hash;
        // Log only server faults (500); 4xx/499/503 are expected and would spam.
        if err.http_status() == 500 {
            tracing::error!(rid = id.0, error = %err, "ingress rejected request");
        }
        let _ = req.state.apply(Event::Error(err.clone()));
        let _ = req.sink.try_send(EgressItem::Error(err)); // client may be gone
        let _ = self
            .senders
            .detok_for(id)
            .send(DetokMsg::Deregister { rid_hash: id });
    }

    /// Drive a request through its ingress states until it terminates (failed or
    /// pushed to the ring) or is handed to the tokenizer pool (re-entering as a
    /// `Tokenized` event). Each arm acts and advances the FSM; the loop
    /// re-dispatches. The arms are the design table's states, `Failed` the single
    /// reject path.
    fn drive(&self, mut req: Request) {
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
                        self.fail(&mut req, Error::Internal("tokenizer pool gone".into()));
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
                    self.fail(&mut req, e);
                    return;
                }
                // Unreachable (egress states never reach here). Reject via `fail`/
                // return (not apply + continue, which would spin on a terminal state).
                other => {
                    self.fail(
                        &mut req,
                        Error::Internal(format!("unexpected ingress state: {other:?}")),
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
            .detok_for(req.rid_hash)
            .send(DetokMsg::Register {
                rid_hash: req.rid_hash,
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
                self.fail(&mut req, e);
                return;
            }
        };
        // Control requests carry no tensor cell — empty `ids`.
        if !self.ingress.try_push(IngressMsg {
            header,
            ids: Bytes::new(),
        }) {
            self.fail(&mut req, Error::QueueFull);
        }
    }

    /// Client disconnected: deregister, then push an `AbortReq(rid)` so the
    /// scheduler stops generating. Fire-and-forget (a full ring drops the abort;
    /// the request then finishes at EOS).
    fn on_abort(&self, rid: String) {
        let id = RidHash::from_rid(&rid);
        let _ = self
            .senders
            .detok_for(id)
            .send(DetokMsg::Deregister { rid_hash: id });

        match ControlRequest::AbortReq(AbortReq::new(rid.clone(), false)).encode() {
            Ok(header) => {
                if !self.ingress.try_push(IngressMsg {
                    header,
                    ids: Bytes::new(),
                }) {
                    tracing::warn!(rid = %rid, "abort dropped: ingress ring full");
                }
            }
            Err(e) => tracing::warn!(rid = %rid, error = %e, "abort encode failed"),
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
                self.fail(&mut req, e);
                return;
            }
        };

        if !self.ingress.try_push(IngressMsg { header, ids }) {
            self.fail(&mut req, Error::QueueFull);
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
    if skip_tokenizer_init
        && matches!(&req.kind, RequestKind::Generate(g) if !g.already_tokenized())
    {
        return Err(Error::Tokenize(
            "skip_tokenizer_init is set: request must provide input_ids".into(),
        ));
    }

    // Client-supplied token ids must be in-vocabulary: an out-of-range id
    // reaches the embedding lookup and kills the scheduler process, so 400
    // here instead — mirroring the Python `TokenizerManager` validation.
    if let (Some(vs), RequestKind::Generate(g)) = (vocab_size, &req.kind) {
        if let Some(ids) = &g.input_ids {
            for &id in ids {
                if id < 0 || id as u64 >= vs {
                    return Err(Error::Validation(format!(
                        "input_ids contains out-of-vocabulary token id {id}; \
                         valid range is [0, {vs})"
                    )));
                }
            }
        }
        if let Some(ids) = &g.token_ids_logprob {
            for &id in ids {
                if id < 0 || id as u64 >= vs {
                    return Err(Error::Validation(format!(
                        "token_ids_logprob contains out-of-vocabulary token id \
                         {id}; valid range is [0, {vs})"
                    )));
                }
            }
        }
    }

    // The scheduler only computes hidden states when launched for it, so without
    // this the request would 200 with `meta_info.hidden_states` silently absent
    // (Python `TokenizerManager._validate_one_request`).
    if !limits.enable_return_hidden_states
        && matches!(&req.kind, RequestKind::Generate(g) if g.return_hidden_states == Some(true))
    {
        return Err(Error::Validation(
            "The server is not configured to return the hidden states. \
             Please set `--enable-return-hidden-states` to enable this feature."
                .into(),
        ));
    }

    Ok(())
}

/// `input + max_new_tokens` must fit the context window (Python
/// `TokenizerManager._validate_one_request`, "Validate total tokens"). Without it
/// the scheduler silently clamps `max_new_tokens` and the client gets a 200 with a
/// truncated completion instead of an actionable 400.
///
/// Under `allow_auto_truncate` we clamp too — but Python's way, from a value the
/// client can compute, and it is the launch flag that opted into it.
fn check_total_tokens(g: &mut GenerateRequest, limits: &Limits) -> Result<(), Error> {
    let (Some(max_req_len), Some(max_new_tokens)) =
        (limits.context_len, g.sampling_params.max_new_tokens)
    else {
        return Ok(()); // context length unknown, or no cap requested
    };
    // Python counts the reserved slots as part of the input, so a request can be
    // rejected for them even when the prompt alone fits.
    let input_len =
        g.input_ids.as_ref().map_or(0, |ids| ids.len()) as u64 + limits.num_reserved_tokens;
    let total = input_len.saturating_add(max_new_tokens.max(0) as u64);
    if total <= max_req_len {
        return Ok(());
    }
    if limits.allow_auto_truncate {
        g.sampling_params.max_new_tokens = Some(max_req_len.saturating_sub(input_len) as i64);
        return Ok(());
    }
    Err(Error::Validation(format!(
        "Requested token count exceeds the model's maximum context length of \
         {max_req_len} tokens. You requested a total of {total} tokens: {input_len} \
         tokens from the input messages and {max_new_tokens} tokens for the \
         completion. Please reduce the number of tokens in the input messages or \
         the completion to fit within the limit."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::RequestState;
    use crate::message::{EgressSink, GenerateRequest, SamplingParams};
    use crate::ring::{IngressConsumer, ingress_ring};
    use tokio::sync::mpsc;

    /// An `Ingress` plus its detok-shard receiver, ring consumer (keep alive —
    /// dropping it closes the ring → false QueueFull), and tm inbox sender.
    fn make_ingress() -> (
        Ingress,
        flume::Receiver<DetokMsg>,
        IngressConsumer,
        flume::Sender<TmEvent>,
    ) {
        make_ingress_with(test_limits())
    }

    fn make_ingress_with(
        limits: Limits,
    ) -> (
        Ingress,
        flume::Receiver<DetokMsg>,
        IngressConsumer,
        flume::Sender<TmEvent>,
    ) {
        let (tok_tx, _tok_rx) = flume::unbounded();
        let (detok_tx, detok_rx) = flume::unbounded();
        let senders = Senders {
            tm: flume::unbounded().0,
            tok: tok_tx,
            detok: vec![detok_tx],
        };
        let (ingress_producer, consumer) = ingress_ring(16);
        let (tm_tx, tm_rx) = flume::unbounded();
        // Keep the shutdown sender alive (leak) so its branch never fires — tests
        // end `run` by dropping `tm_tx`, not by shutdown.
        let (sd_tx, sd_rx) = flume::unbounded::<()>();
        std::mem::forget(sd_tx);
        let ingress = Ingress::new(tm_rx, senders, ingress_producer, limits, sd_rx);
        (ingress, detok_rx, consumer, tm_tx)
    }

    /// The default test limits: a real tokenizer, vocab 1000, no context ceiling.
    fn test_limits() -> Limits {
        Limits {
            vocab_size: Some(1000),
            ..Default::default()
        }
    }

    fn generate_req(id: u64, sampling_params: SamplingParams) -> Request {
        let (tx, _rx) = mpsc::channel(8);
        Request {
            rid_hash: RidHash(id),
            rid: id.to_string(),
            state: RequestState::Received,
            sink: EgressSink::Local(tx),
            kind: RequestKind::Generate(Box::new(GenerateRequest {
                rid: id.to_string(),
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
            context_len: Some(10),
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
            context_len: Some(10),
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
            context_len: Some(10),
            allow_auto_truncate: true,
            ..test_limits()
        };
        assert!(check_total_tokens(&mut g, &truncating).is_ok());
        assert_eq!(g.sampling_params.max_new_tokens, Some(7), "clamped to fit");

        // Unknown context length → no ceiling to enforce.
        g.sampling_params = sp(Some(100));
        assert!(check_total_tokens(&mut g, &test_limits()).is_ok());
        assert_eq!(g.sampling_params.max_new_tokens, Some(100), "untouched");

        // No cap requested → nothing to add to the input length.
        g.sampling_params = sp(None);
        let capped = Limits {
            context_len: Some(1),
            ..test_limits()
        };
        assert!(check_total_tokens(&mut g, &capped).is_ok());
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
        let err = validate(&mut req(Some(true)), &disabled).unwrap_err();
        assert_eq!(err.http_status(), 400);
        assert!(
            err.to_string().contains("--enable-return-hidden-states"),
            "message must name the flag: {err}"
        );
        // Not asking for them, or asking on a server that supports them, is fine.
        assert!(validate(&mut req(Some(false)), &disabled).is_ok());
        assert!(validate(&mut req(None), &disabled).is_ok());
        let enabled = Limits {
            enable_return_hidden_states: true,
            ..test_limits()
        };
        assert!(validate(&mut req(Some(true)), &enabled).is_ok());
    }

    /// End-to-end through `drive`: an over-context request is rejected on the way
    /// to the ring, after registration — so it must be deregistered, not leaked.
    #[test]
    fn over_context_request_deregisters_and_never_reaches_the_ring() {
        let (ingress, detok_rx, consumer, _tm_tx) = make_ingress_with(Limits {
            context_len: Some(4),
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
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid_hash, .. }) if rid_hash == RidHash(33)),
            "registered before the check",
        );
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid_hash }) if rid_hash == RidHash(33)),
            "must deregister on reject",
        );
        assert!(
            consumer.drain(16).headers.is_empty(),
            "must not reach the scheduler"
        );
    }

    /// A request rejected at normalization (post-register) must not leak: the shard
    /// sees `Register` then `Deregister`. Regression for RSS growth on bad input.
    #[test]
    fn rejected_request_deregisters_from_shard() {
        let (ingress, detok_rx, _consumer, _tm_tx) = make_ingress();
        // top_p = 2.0 is outside (0, 1], so `SamplingParams::normalize` rejects it.
        let bad = SamplingParams {
            top_p: 2.0,
            ..Default::default()
        };
        ingress.drive(generate_req(7, bad));

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid_hash, .. }) if rid_hash == RidHash(7)),
            "expected Register for rid 7",
        );
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid_hash }) if rid_hash == RidHash(7)),
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
        let (ingress, detok_rx, _consumer, _tm_tx) = make_ingress();
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
        let (ingress, detok_rx, _consumer, _tm_tx) = make_ingress();
        let mut req = generate_req(22, SamplingParams::default());
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(vec![-1]);
        }
        ingress.drive(req);
        match detok_rx.try_recv() {
            Err(_) | Ok(DetokMsg::Deregister { .. }) => {}
            Ok(_) => panic!("negative token id must not be admitted"),
        }

        let (ingress, detok_rx, _consumer, _tm_tx) = make_ingress();
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
        let (ingress, detok_rx, _consumer, _tm_tx) = make_ingress();
        // Empty map → all sampling defaults, passes normalization.
        ingress.drive(generate_req(9, SamplingParams::default()));

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid_hash, .. }) if rid_hash == RidHash(9)),
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
        let (ingress, detok_rx, _consumer, tm_tx) = make_ingress();
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
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid_hash }) if rid_hash == RidHash(11)),
            "tokenize failure must deregister rid 11",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }

    /// An abort deregisters (by the id hashed from the rid string), so a request
    /// aborted before any terminal chunk can't leak.
    #[test]
    fn abort_deregisters_from_shard() {
        let (ingress, detok_rx, _consumer, tm_tx) = make_ingress();
        tm_tx.send(TmEvent::Abort("rid-13".to_string())).unwrap();
        drop(tm_tx);
        ingress.run();

        let want = RidHash::from_rid("rid-13");
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid_hash }) if rid_hash == want),
            "abort must deregister the hashed rid",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }

    /// A successful pool return (Queued, ids filled) is pushed to the ring, not
    /// rejected; its registration is untouched.
    #[test]
    fn tokenized_return_pushes_without_deregister() {
        let (ingress, detok_rx, _consumer, tm_tx) = make_ingress();
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
        let (ingress, detok_rx, _consumer, _tm_tx) = make_ingress();
        // No ids → NeedsTokenize → Tokenizing branch.
        let mut req = generate_req(21, SamplingParams::default());
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = None;
        }
        ingress.drive(req);

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid_hash, .. }) if rid_hash == RidHash(21)),
            "expected Register for rid 21",
        );
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid_hash }) if rid_hash == RidHash(21)),
            "pool-gone hand-off must deregister rid 21",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }
}
