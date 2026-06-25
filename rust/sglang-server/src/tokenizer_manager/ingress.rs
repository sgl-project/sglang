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
use crate::message::{EgressItem, Request, RequestKind, TokenizedReqPayload, control_req_msgpack};
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
    // Received → Validating
    let _ = req
        .state
        .apply(Event::Validated(ValidationOutcome::NeedsTokenize));

    // Register the egress sink with the owning detok shard *before* the request
    // leaves Rust, so the response (generate chunks or a control result) has a
    // home. Routing is by id only.
    let shard = senders.detok_for(req.id);
    if shard
        .send(DetokMsg::Register {
            id: req.id,
            sink: req.sink.clone(),
            stream: req.stream,
        })
        .is_err()
    {
        fail(&mut req, Error::Internal("detok shard gone".into()));
        return;
    }

    // Control requests reuse this FSM but skip tokenization entirely: validate
    // straight to Queued and push the bare `[tag, rid, nil]` control message.
    if let RequestKind::Control(tag) = req.kind {
        let _ = req
            .state
            .apply(Event::Validated(ValidationOutcome::AlreadyTokenized)); // → Queued
        push_control_to_ring(req, ingress, tag);
        return;
    }

    // skip_tokenizer_init: there is no tokenizer — the client must send token
    // ids. Treat every generate request as already-tokenized (no tokenize hop);
    // a missing/empty `input_ids` is a client error rather than a silent
    // byte-encode by the stub tokenizer.
    if skip_tokenizer_init {
        if !req.payload.already_tokenized() {
            fail(
                &mut req,
                Error::Tokenize(
                    "skip_tokenizer_init is set: request must provide input_ids".into(),
                ),
            );
            return;
        }
        req.input_ids = req.payload.input_ids.clone();
        let _ = req
            .state
            .apply(Event::Validated(ValidationOutcome::AlreadyTokenized)); // → Queued
        push_to_ring(req, ingress);
        return;
    }

    let outcome = classify(&req);
    match outcome {
        ValidationOutcome::AlreadyTokenized => {
            req.input_ids = req.payload.input_ids.clone();
            let _ = req.state.apply(Event::Validated(outcome)); // → Queued
            push_to_ring(req, ingress); // no tokenize hop
        }
        ValidationOutcome::NeedsTokenize => {
            let _ = req.state.apply(Event::Validated(outcome)); // → Tokenizing
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
    let input_ids = match req.input_ids.take() {
        Some(ids) if !ids.is_empty() => ids,
        _ => {
            fail(&mut req, Error::Tokenize("empty input_ids".into()));
            return;
        }
    };

    // Move (not clone) out of the owned request: we never read `req.payload`
    // again — the only later use of `req` is the `fail` error path, which
    // touches `req.state` / `req.sink`. `take` leaves valid empties behind.
    let payload = TokenizedReqPayload {
        rid: req.id.0.to_string(),
        input_text: req.payload.text.take(),
        input_ids,
        sampling_params: req.payload.sampling_params.take(),
        stream: req.stream,
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

fn classify(req: &Request) -> ValidationOutcome {
    if req.payload.has_multimodal() {
        ValidationOutcome::HasMultimodal
    } else if req.payload.already_tokenized() {
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
