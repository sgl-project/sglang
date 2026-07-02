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
//!   Validating    → Queued        (control: no tokenize, no sampling params)
//!   Normalizing   → {Encoding | Tokenizing | Queued}   (by ValidationOutcome)
//!   Tokenizing    → Queued        (on TokenizeDone, when the request returns)
//!   Queued        → ring          (handed to the scheduler)
//!
//! The egress edges (Streaming/Finalizing/Completed) are driven on the egress
//! side (see `egress` + `detokenizer`).

use bytes::Bytes;

use crate::error::Error;
use crate::fsm::{Event, ValidationOutcome};
use crate::ids::RequestId;
use crate::message::{
    EgressItem, GenerateRequest, IngressMsg, Request, RequestKind, TokenizedReqPayload,
    abort_req_msgpack, control_req_msgpack,
};
use crate::runtime::Runnable;
use crate::runtime::channels::{DetokMsg, Senders, TmEvent};
use crate::runtime::ring::IngressProducer;
use crate::tokenizer_manager::sampling::normalize_sampling_params;

/// Ingress FSM dispatcher stage. Owns its inbox + downstream handles, so the
/// runtime spawns it as a [`Runnable`] rather than calling a free `run_*` fn
/// with positional arguments.
pub struct Ingress {
    rx: flume::Receiver<TmEvent>,
    senders: Senders,
    ingress: IngressProducer,
    skip_tokenizer_init: bool,
}

impl Ingress {
    pub fn new(
        rx: flume::Receiver<TmEvent>,
        senders: Senders,
        ingress: IngressProducer,
        skip_tokenizer_init: bool,
    ) -> Self {
        Self {
            rx,
            senders,
            ingress,
            skip_tokenizer_init,
        }
    }
}

impl Runnable for Ingress {
    fn run(self) {
        while let Ok(ev) = self.rx.recv() {
            match ev {
                TmEvent::Ingress(req) => self.on_ingress(req),
                TmEvent::Tokenized(req) => self.on_tokenized(req),
                TmEvent::Abort(id) => self.on_abort(id),
            }
        }
    }
}

impl Ingress {
    /// Validate a fresh request and route it onto the correct ingress branch.
    fn on_ingress(&self, mut req: Request) {
        // Received → Validating, plus payload validation; reject invalid requests.
        if let Err(e) = validate(&mut req, self.skip_tokenizer_init) {
            fail(&mut req, e);
            return;
        }

        // Register the egress sink with the owning detok shard *before* the
        // request leaves Rust, so the response (generate chunks or a control
        // result) has a home. Routing is by id only.
        let shard = self.senders.detok_for(req.id);
        if shard
            .send(DetokMsg::Register {
                id: req.id,
                sink: req.sink.clone(),
            })
            .is_err()
        {
            fail(&mut req, Error::Internal("detok shard gone".into()));
            return;
        }

        // Branch by kind. Copy the control tag out so the borrow of `req.kind`
        // ends before we move `req` downstream.
        let control_tag = match &req.kind {
            RequestKind::Control(c) => Some(c.tag),
            RequestKind::Generate(_) => None,
        };
        if let Some(tag) = control_tag {
            // Control requests skip tokenization entirely: validate straight to
            // Queued and push the bare `[tag, rid, nil]` control message.
            let _ = req
                .state
                .apply(Event::Validated(ValidationOutcome::AlreadyTokenized)); // → Queued
            self.push_control_to_ring(req, tag);
            return;
        }

        // Validating → Normalizing: normalize + verify the sampling params here
        // (the Rust server replaces the Python TokenizerManager, where this runs)
        // so the work stays off the scheduler's latency-critical loop. Sets
        // `is_normalized=true` on the wire; the scheduler then skips its pass.
        let _ = req.state.apply(Event::Normalized);
        if let RequestKind::Generate(g) = &mut req.kind
            && let Err(e) = normalize_sampling_params(&mut g.payload.sampling_params)
        {
            fail(&mut req, e);
            return;
        }

        self.route_generate(req);
    }

    /// Route a validated generate request: queue directly when it already carries
    /// token ids, else hand it to the tokenizer pool.
    fn route_generate(&self, mut req: Request) {
        let RequestKind::Generate(g) = &req.kind else {
            return; // unreachable: control is handled by the caller
        };
        match classify(g) {
            ValidationOutcome::AlreadyTokenized => {
                if let RequestKind::Generate(g) = &mut req.kind {
                    g.input_ids = g.payload.input_ids.clone();
                }
                let _ = req
                    .state
                    .apply(Event::Validated(ValidationOutcome::AlreadyTokenized)); // → Queued
                self.push_to_ring(req); // no tokenize hop
            }
            ValidationOutcome::NeedsTokenize => {
                let _ = req
                    .state
                    .apply(Event::Validated(ValidationOutcome::NeedsTokenize)); // → Tokenizing
                if self.senders.tok.send(req).is_err() {
                    tracing::error!("tokenizer pool gone");
                }
            }
            ValidationOutcome::HasMultimodal => {
                // Encoder deferred this iteration: treat as a plain tokenize.
                let _ = req
                    .state
                    .apply(Event::Validated(ValidationOutcome::NeedsTokenize));
                if self.senders.tok.send(req).is_err() {
                    tracing::error!("tokenizer pool gone");
                }
            }
        }
    }

    /// Push a bare control request (`[tag, rid, nil]`) onto the ingress ring. The
    /// scheduler dispatches it (e.g. `GetInternalStateReq`) and replies via the
    /// egress ring as a single `Result`.
    fn push_control_to_ring(&self, mut req: Request, tag: &str) {
        let header = match control_req_msgpack(tag, &req.id.0.to_string()) {
            Ok(b) => b,
            Err(e) => {
                fail(&mut req, e);
                return;
            }
        };
        // Control requests carry no tensor cell — empty `ids`.
        if !self.ingress.try_push(IngressMsg {
            header,
            ids: Bytes::new(),
        }) {
            fail(&mut req, Error::QueueFull);
        }
    }

    /// Client disconnected: push an `AbortReq(rid)` onto the ingress ring so the
    /// scheduler stops generating. Fire-and-forget — there's no sink left (the
    /// client is gone); a full ring just drops the abort (the request will still
    /// finish at EOS, only later).
    fn on_abort(&self, id: RequestId) {
        match abort_req_msgpack(&id.0.to_string()) {
            Ok(header) => {
                if !self.ingress.try_push(IngressMsg {
                    header,
                    ids: Bytes::new(),
                }) {
                    tracing::warn!(rid = id.0, "abort dropped: ingress ring full");
                }
            }
            Err(e) => tracing::warn!(rid = id.0, error = %e, "abort encode failed"),
        }
    }

    /// A request returned from the Tokenizer pool with `input_ids` filled in.
    fn on_tokenized(&self, mut req: Request) {
        // Tokenizing → Queued
        let _ = req.state.apply(Event::TokenizeDone);
        self.push_to_ring(req);
    }

    /// Build the msgpack `TokenizedGenerateReqInput` and push it onto the ingress
    /// ring for the scheduler. On backpressure, fail the request.
    fn push_to_ring(&self, mut req: Request) {
        // Only generate requests reach here (control uses `push_control_to_ring`).
        let RequestKind::Generate(g) = &mut req.kind else {
            fail(
                &mut req,
                Error::Internal("non-generate request reached push_to_ring".into()),
            );
            return;
        };
        // Move (not clone) the generate fields out; `take` leaves valid empties so
        // the borrow of `req.kind` ends and `req` is free for the `fail` path.
        let input_ids = g.input_ids.take();
        let input_text = g.payload.text.take();
        let sampling_params = g.payload.sampling_params.take();
        let stream = g.stream;
        // Replicate TokenizerManager's scalar normalization of the logprob
        // options (this path bypasses it): absent → the scheduler defaults.
        let return_logprob = g.payload.return_logprob.unwrap_or(false);
        let logprob_start_len = g.payload.logprob_start_len.unwrap_or(-1);
        let top_logprobs_num = g.payload.top_logprobs_num.unwrap_or(0);
        let token_ids_logprob = g.payload.token_ids_logprob.take();
        let return_hidden_states = g.payload.return_hidden_states.unwrap_or(false);

        let input_ids = match input_ids {
            Some(ids) if !ids.is_empty() => ids,
            _ => {
                fail(&mut req, Error::Tokenize("empty input_ids".into()));
                return;
            }
        };

        let payload = TokenizedReqPayload {
            rid: req.id.0.to_string(),
            input_text,
            input_ids,
            sampling_params,
            return_logprob,
            logprob_start_len,
            top_logprobs_num,
            token_ids_logprob,
            return_hidden_states,
            stream,
        };

        // Columnar split: scalar header through msgpack, the ids tensor as a raw
        // int64 buffer alongside (concatenated across the batch in `recv_requests`).
        let header: Bytes = match payload.to_header_msgpack() {
            Ok(b) => b,
            Err(e) => {
                fail(&mut req, e);
                return;
            }
        };
        let ids = payload.input_ids_i64_le();

        if !self.ingress.try_push(IngressMsg { header, ids }) {
            fail(&mut req, Error::QueueFull);
        }
        // On success the request is now owned by the scheduler; egress will
        // arrive by rid. We intentionally drop our `Request` here (state ==
        // Queued); the detok shard holds the sink.
    }
}

/// Validating phase: drive `Received → Validating` and check the payload is
/// admissible. `Err` rejects the request (it never reaches a branch).
///
/// `skip_tokenizer_init` means no tokenizer is loaded, so a generate request
/// *must* already carry token ids; a text-only request is rejected here rather
/// than being silently byte-encoded by the stub tokenizer. Control requests
/// carry no token ids and are exempt.
fn validate(req: &mut Request, skip_tokenizer_init: bool) -> Result<(), Error> {
    // Received → Validating
    let _ = req
        .state
        .apply(Event::Validated(ValidationOutcome::NeedsTokenize));
    // The skip check is generate-only: control requests carry no token ids, so
    // matching `Generate` naturally exempts them.
    if skip_tokenizer_init
        && matches!(&req.kind, RequestKind::Generate(g) if !g.payload.already_tokenized())
    {
        return Err(Error::Tokenize(
            "skip_tokenizer_init is set: request must provide input_ids".into(),
        ));
    }

    Ok(())
}

/// Pick the ingress branch for a validated generate request.
fn classify(g: &GenerateRequest) -> ValidationOutcome {
    if g.payload.has_multimodal() {
        ValidationOutcome::HasMultimodal
    } else if g.payload.already_tokenized() {
        ValidationOutcome::AlreadyTokenized
    } else {
        ValidationOutcome::NeedsTokenize
    }
}

fn fail(req: &mut Request, err: Error) {
    let _ = req.state.apply(Event::Error(err.clone()));
    // Best-effort notify the client; sink may already be closed.
    let _ = req.sink.try_send(EgressItem::Error(err));
}
