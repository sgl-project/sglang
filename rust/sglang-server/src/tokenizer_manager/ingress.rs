//! TokenizerManager — ingress side.
//!
//! Single-consumer loop draining one inbox fed by both the API server (fresh
//! requests) and the Tokenizer pool (returned requests). It owns the request
//! while driving the ingress FSM and hands it off by *moving* it to the next
//! stage; nothing here is shared, so no locks.
//!
//! Edges driven here (from the design table):
//!   Received      → Validating
//!   Validating    → {Encoding | Tokenizing | Queued}   (by ValidationOutcome)
//!   Tokenizing    → Queued        (on TokenizeDone, when the request returns)
//!   Queued        → ring          (handed to the scheduler)
//!
//! The egress edges (Streaming/Finalizing/Completed) are driven on the egress
//! side (see `egress` + `detokenizer`).

use bytes::Bytes;

use crate::error::Error;
use crate::fsm::{Event, ValidationOutcome};
use crate::message::{
    EgressItem, GenerateRequest, Request, RequestKind, TokenizedReqPayload, control_req_msgpack,
};
use crate::runtime::channels::{DetokMsg, Senders, TmEvent};
use crate::runtime::ring::IngressProducer;

pub fn run_ingress(
    rx: flume::Receiver<TmEvent>,
    senders: Senders,
    ingress: IngressProducer,
    skip_tokenizer_init: bool,
) {
    while let Ok(ev) = rx.recv() {
        match ev {
            TmEvent::Ingress(req) => on_ingress(req, &senders, &ingress, skip_tokenizer_init),
            TmEvent::Tokenized(req) => on_tokenized(req, &ingress),
        }
    }
}

/// Validate a fresh request and route it onto the correct ingress branch.
fn on_ingress(
    mut req: Request,
    senders: &Senders,
    ingress: &IngressProducer,
    skip_tokenizer_init: bool,
) {
    // Received → Validating, plus payload validation; reject invalid requests.
    if let Err(e) = validate(&mut req, skip_tokenizer_init) {
        fail(&mut req, e);
        return;
    }

    // Register the egress sink with the owning detok shard *before* the request
    // leaves Rust, so the response (generate chunks or a control result) has a
    // home. Routing is by id only.
    let shard = senders.detok_for(req.id);
    if shard
        .send(DetokMsg::Register {
            id: req.id,
            sink: req.sink.clone(),
            stream: req.kind.is_stream(),
        })
        .is_err()
    {
        fail(&mut req, Error::Internal("detok shard gone".into()));
        return;
    }

    // Branch by kind. Copy the control tag out so the borrow of `req.kind` ends
    // before we move `req` downstream.
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
        push_control_to_ring(req, ingress, tag);
        return;
    }

    route_generate(req, senders, ingress);
}

/// Route a validated generate request: queue directly when it already carries
/// token ids, else hand it to the tokenizer pool.
fn route_generate(mut req: Request, senders: &Senders, ingress: &IngressProducer) {
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
            push_to_ring(req, ingress); // no tokenize hop
        }
        ValidationOutcome::NeedsTokenize => {
            let _ = req
                .state
                .apply(Event::Validated(ValidationOutcome::NeedsTokenize)); // → Tokenizing
            if senders.tok.send(req).is_err() {
                tracing::error!("tokenizer pool gone");
            }
        }
        ValidationOutcome::HasMultimodal => {
            // Encoder deferred this iteration: treat as a plain tokenize.
            let _ = req
                .state
                .apply(Event::Validated(ValidationOutcome::NeedsTokenize));
            if senders.tok.send(req).is_err() {
                tracing::error!("tokenizer pool gone");
            }
        }
    }
}

/// Push a bare control request (`[tag, rid, nil]`) onto the ingress ring. The
/// scheduler dispatches it (e.g. `GetInternalStateReq`) and replies via the
/// egress ring as a single `Result`.
fn push_control_to_ring(mut req: Request, ingress: &IngressProducer, tag: &str) {
    let bytes = match control_req_msgpack(tag, &req.id.0.to_string()) {
        Ok(b) => b,
        Err(e) => {
            fail(&mut req, e);
            return;
        }
    };
    if !ingress.try_push(bytes) {
        fail(&mut req, Error::QueueFull);
    }
}

/// A request returned from the Tokenizer pool with `input_ids` filled in.
fn on_tokenized(mut req: Request, ingress: &IngressProducer) {
    // Tokenizing → Queued
    let _ = req.state.apply(Event::TokenizeDone);
    push_to_ring(req, ingress);
}

/// Build the msgpack `TokenizedGenerateReqInput` and push it onto the ingress
/// ring for the scheduler. On backpressure, fail the request.
fn push_to_ring(mut req: Request, ingress: &IngressProducer) {
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
        stream,
    };

    let bytes: Bytes = match payload.to_msgpack() {
        Ok(b) => b,
        Err(e) => {
            fail(&mut req, e);
            return;
        }
    };

    if !ingress.try_push(bytes) {
        fail(&mut req, Error::QueueFull);
    }
    // On success the request is now owned by the scheduler; egress will arrive
    // by rid. We intentionally drop our `Request` here (state == Queued); the
    // detok shard holds the sink.
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
    if skip_tokenizer_init {
        match &req.kind {
            RequestKind::Generate(g) => {
                if !g.payload.already_tokenized() {
                    return Err(Error::Tokenize(
                        "skip_tokenizer_init is set: request must provide input_ids".into(),
                    ));
                }
            }
            _ => {} // exempt
        }
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
