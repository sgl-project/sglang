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
use crate::fsm::{Event, RequestState, ValidationOutcome};
use crate::ids::RequestId;
use crate::message::{
    EgressItem, IngressMsg, Request, RequestKind, TokenizedReqPayload, abort_req_msgpack,
    control_req_msgpack,
};
use crate::runtime::Runnable;
use crate::runtime::channels::{DetokMsg, Senders, TmEvent, recv};
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
    shutdown: flume::Receiver<()>,
}

impl Ingress {
    pub fn new(
        rx: flume::Receiver<TmEvent>,
        senders: Senders,
        ingress: IngressProducer,
        skip_tokenizer_init: bool,
        shutdown: flume::Receiver<()>,
    ) -> Self {
        Self {
            rx,
            senders,
            ingress,
            skip_tokenizer_init,
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
                TmEvent::Abort(id) => self.on_abort(id),
            }
        }
    }
}

impl Ingress {
    /// Reject a request: → `Failed`, notify the client, deregister (unconditional
    /// — a no-op when nothing was registered).
    fn fail(&self, req: &mut Request, err: Error) {
        let id = req.id;
        // Log only server faults (500); 4xx/499/503 are expected and would spam.
        if err.http_status() == 500 {
            tracing::error!(rid = id.0, error = %err, "ingress rejected request");
        }
        let _ = req.state.apply(Event::Error(err.clone()));
        let _ = req.sink.try_send(EgressItem::Error(err)); // client may be gone
        let _ = self.senders.detok_for(id).send(DetokMsg::Deregister { id });
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
                    if let Err(e) = validate(&mut req, self.skip_tokenizer_init) {
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
                // Control skips straight to Queued; generate goes to Normalizing.
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
                        match normalize_sampling_params(&mut g.payload.sampling_params) {
                            Err(e) => Err(e),
                            // Client ids skip the pool (→ Queued); text is tokenized.
                            Ok(()) if g.payload.already_tokenized() => {
                                g.input_ids = g.payload.input_ids.clone();
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
                // `Tokenized` event (Queued, or Failed on error). Doesn't loop.
                RequestState::Tokenizing => {
                    if let Err(err) = self.senders.tok.send(req) {
                        // Pool gone (workers exited); flume hands the request back.
                        let mut req = err.into_inner();
                        self.fail(&mut req, Error::Internal("tokenizer pool gone".into()));
                    }
                    return;
                }
                // Push the wire message (control frame or generate payload) to the ring.
                RequestState::Queued => {
                    match &req.kind {
                        RequestKind::Control(c) => {
                            let tag = c.tag;
                            self.push_control_to_ring(req, tag);
                        }
                        RequestKind::Generate(_) => self.push_to_ring(req),
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
    /// has a home. Carries `return_text_in_logprobs` so the shard, not the I/O
    /// threads, decodes logprob token text. Returns `false` if the shard is gone.
    fn register_detok(&self, req: &Request) -> bool {
        let decode_logprob_text = match &req.kind {
            RequestKind::Generate(g) => g.payload.return_text_in_logprobs.unwrap_or(false),
            RequestKind::Control(_) => false,
        };
        self.senders
            .detok_for(req.id)
            .send(DetokMsg::Register {
                id: req.id,
                sink: req.sink.clone(),
                decode_logprob_text,
            })
            .is_ok()
    }

    /// Push a bare control request (`[tag, rid, nil]`) onto the ingress ring. The
    /// scheduler dispatches it (e.g. `GetInternalStateReq`) and replies via the
    /// egress ring as a single `Result`.
    fn push_control_to_ring(&self, mut req: Request, tag: &str) {
        let header = match control_req_msgpack(tag, &req.id.0.to_string()) {
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
    fn on_abort(&self, id: RequestId) {
        let _ = self.senders.detok_for(id).send(DetokMsg::Deregister { id });

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

    /// Build the msgpack `TokenizedGenerateReqInput` and push it onto the ingress
    /// ring for the scheduler. On backpressure, fail the request.
    fn push_to_ring(&self, mut req: Request) {
        // Only generate requests reach here (control uses `push_control_to_ring`).
        let RequestKind::Generate(g) = &mut req.kind else {
            self.fail(
                &mut req,
                Error::Internal("non-generate request reached push_to_ring".into()),
            );
            return;
        };
        // Move the fields out (`take` frees the `req.kind` borrow for `fail`).
        let input_ids = g.input_ids.take();
        let input_text = g.payload.text.take();
        let sampling_params = g.payload.sampling_params.take();
        let stream = g.stream;
        // Scalar logprob normalization the TokenizerManager would do: absent →
        // scheduler defaults.
        let return_logprob = g.payload.return_logprob.unwrap_or(false);
        let logprob_start_len = g.payload.logprob_start_len.unwrap_or(-1);
        let top_logprobs_num = g.payload.top_logprobs_num.unwrap_or(0);
        let token_ids_logprob = g.payload.token_ids_logprob.take();
        let return_hidden_states = g.payload.return_hidden_states.unwrap_or(false);
        let is_health_check = g.is_health_check;

        let input_ids = match input_ids {
            Some(ids) if !ids.is_empty() => ids,
            _ => {
                self.fail(&mut req, Error::Tokenize("empty input_ids".into()));
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
            is_health_check,
            stream,
        };

        // Columnar split: scalar header via msgpack, ids as a raw int64 buffer.
        let header: Bytes = match payload.to_header_msgpack() {
            Ok(b) => b,
            Err(e) => {
                self.fail(&mut req, e);
                return;
            }
        };
        let ids = payload.input_ids_i64_le();

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
fn validate(req: &mut Request, skip_tokenizer_init: bool) -> Result<(), Error> {
    let _ = req
        .state
        .apply(Event::Validated(ValidationOutcome::NeedsTokenize));
    if skip_tokenizer_init
        && matches!(&req.kind, RequestKind::Generate(g) if !g.payload.already_tokenized())
    {
        return Err(Error::Tokenize(
            "skip_tokenizer_init is set: request must provide input_ids".into(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::RequestState;
    use crate::message::{EgressSink, GeneratePayload, GenerateRequest};
    use crate::runtime::ring::{IngressConsumer, ingress_ring};
    use tokio::sync::mpsc;

    /// An `Ingress` plus its detok-shard receiver, ring consumer (keep alive —
    /// dropping it closes the ring → false QueueFull), and tm inbox sender.
    fn make_ingress() -> (
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
        let ingress = Ingress::new(tm_rx, senders, ingress_producer, false, sd_rx);
        (ingress, detok_rx, consumer, tm_tx)
    }

    fn generate_req(id: u64, sampling_params: rmpv::Value) -> Request {
        let payload = GeneratePayload {
            input_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(sampling_params),
            ..Default::default()
        };
        let (tx, _rx) = mpsc::channel(8);
        Request {
            id: RequestId(id),
            state: RequestState::Received,
            sink: EgressSink::Local(tx),
            kind: RequestKind::Generate(GenerateRequest {
                payload,
                ..Default::default()
            }),
        }
    }

    /// A request rejected at normalization (post-register) must not leak: the shard
    /// sees `Register` then `Deregister`. Regression for RSS growth on bad input.
    #[test]
    fn rejected_request_deregisters_from_shard() {
        let (ingress, detok_rx, _consumer, _tm_tx) = make_ingress();
        // top_p = 2.0 is outside (0, 1], so `normalize_sampling_params` rejects it.
        let bad = rmpv::Value::Map(vec![(rmpv::Value::from("top_p"), rmpv::Value::F64(2.0))]);
        ingress.drive(generate_req(7, bad));

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { id, .. }) if id == RequestId(7)),
            "expected Register for rid 7",
        );
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { id }) if id == RequestId(7)),
            "expected Deregister for rid 7 (leak fix)",
        );
        assert!(
            detok_rx.try_recv().is_err(),
            "no further shard messages — registration fully cleaned up",
        );
    }

    /// A valid request is registered and handed onward — never deregistered.
    #[test]
    fn admitted_request_keeps_registration() {
        let (ingress, detok_rx, _consumer, _tm_tx) = make_ingress();
        // Empty map → all sampling defaults, passes normalization.
        ingress.drive(generate_req(9, rmpv::Value::Map(vec![])));

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { id, .. }) if id == RequestId(9)),
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
        let mut req = generate_req(11, rmpv::Value::Map(vec![]));
        let _ = req
            .state
            .apply(Event::Error(Error::Tokenize("boom".into())));
        tm_tx.send(TmEvent::Tokenized(req)).unwrap();
        // Close the inbox so the run loop returns after draining the one event.
        drop(tm_tx);
        ingress.run();

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { id }) if id == RequestId(11)),
            "tokenize failure must deregister rid 11",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }

    /// An abort deregisters, so a request aborted before any terminal chunk
    /// can't leak.
    #[test]
    fn abort_deregisters_from_shard() {
        let (ingress, detok_rx, _consumer, tm_tx) = make_ingress();
        tm_tx.send(TmEvent::Abort(RequestId(13))).unwrap();
        drop(tm_tx);
        ingress.run();

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { id }) if id == RequestId(13)),
            "abort must deregister rid 13",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }

    /// A successful pool return (Queued, ids filled) is pushed to the ring, not
    /// rejected; its registration is untouched.
    #[test]
    fn tokenized_return_pushes_without_deregister() {
        let (ingress, detok_rx, _consumer, tm_tx) = make_ingress();
        let mut req = generate_req(15, rmpv::Value::Map(vec![]));
        // Simulate a successful pool return: ids filled, Queued.
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = Some(vec![1, 2, 3]);
        }
        req.state = RequestState::Queued;
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
        let mut req = generate_req(21, rmpv::Value::Map(vec![]));
        if let RequestKind::Generate(g) = &mut req.kind {
            g.payload.input_ids = None;
        }
        ingress.drive(req);

        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { id, .. }) if id == RequestId(21)),
            "expected Register for rid 21",
        );
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { id }) if id == RequestId(21)),
            "pool-gone hand-off must deregister rid 21",
        );
        assert!(detok_rx.try_recv().is_err(), "no further shard messages");
    }
}
