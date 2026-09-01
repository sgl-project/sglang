//! Tests for scheduler intake.

use super::*;
use crate::message::request::GenerateRequest;
use crate::message::response::ResponseSink;
use crate::message::sampling::SamplingParams;
use crate::tokenizer_manager::channel::{ToSchedulerRx, to_scheduler};
use crate::utils::fsm::RequestState;
use tokio::sync::mpsc;

/// An `Intake` plus its detok-shard receiver, to_scheduler channel consumer (keep alive —
/// dropping it closes the channel → false QueueFull), tm inbox sender, and the
/// mm-pool receiver (keep alive — dropping it makes mm submits fail).
fn make_intake() -> (
    Intake,
    flume::Receiver<DetokMsg>,
    ToSchedulerRx,
    flume::Sender<TmEvent>,
    flume::Receiver<MmRequest>,
) {
    make_intake_with(test_limits())
}

fn make_intake_with_abort(
    abort_rx: flume::Receiver<AbortSource>,
) -> (
    Intake,
    flume::Receiver<DetokMsg>,
    ToSchedulerRx,
    flume::Sender<TmEvent>,
    flume::Receiver<MmRequest>,
) {
    make_intake_inner(test_limits(), abort_rx)
}

fn make_intake_with(
    limits: Limits,
) -> (
    Intake,
    flume::Receiver<DetokMsg>,
    ToSchedulerRx,
    flume::Sender<TmEvent>,
    flume::Receiver<MmRequest>,
) {
    let (abort_tx, abort_rx) = flume::unbounded::<AbortSource>();
    std::mem::forget(abort_tx); // keep the lane open; tests end by dropping tm_tx
    make_intake_inner(limits, abort_rx)
}

fn make_intake_inner(
    limits: Limits,
    abort_rx: flume::Receiver<AbortSource>,
) -> (
    Intake,
    flume::Receiver<DetokMsg>,
    ToSchedulerRx,
    flume::Sender<TmEvent>,
    flume::Receiver<MmRequest>,
) {
    let (tok_tx, _tok_rx) = flume::unbounded();
    let (detok_tx, detok_rx) = flume::unbounded();
    let senders = Senders {
        tok_manager_tx: flume::unbounded().0,
        abort_tx: flume::unbounded().0,
        tokenizer_tx: tok_tx,
        detokenizer_tx: vec![detok_tx],
    };
    let (to_scheduler_tx, consumer) = to_scheduler(16);
    let (tm_tx, tm_rx) = flume::unbounded();
    let (mm_tx, mm_rx) = flume::unbounded();
    // Keep the shutdown sender alive (leak) so its branch never fires — tests
    // end `run` by dropping `tm_tx`, not by shutdown.
    let (sd_tx, sd_rx) = flume::unbounded::<()>();
    std::mem::forget(sd_tx);
    let intake = Intake::new(
        tm_rx,
        abort_rx,
        senders,
        to_scheduler_tx,
        limits,
        test_mm(mm_tx, true),
        sd_rx,
    );
    (intake, detok_rx, consumer, tm_tx, mm_rx)
}

/// An [`Mm`] over `tx` with a fresh sidecar.
fn test_mm(tx: flume::Sender<MmRequest>, enabled: bool) -> Mm {
    Mm {
        enabled,
        tx,
        sidecar: Default::default(),
    }
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
        let (to_scheduler_tx, consumer) = to_scheduler(16);
        let (sd_tx, sd_rx) = flume::unbounded::<()>();
        std::mem::forget(sd_tx);
        let mut intake = Intake::new(
            flume::unbounded().1,
            flume::unbounded().1,
            Senders {
                tok_manager_tx: flume::unbounded().0,
                abort_tx: flume::unbounded().0,
                tokenizer_tx: flume::unbounded().0,
                detokenizer_tx: vec![detok_tx],
            },
            to_scheduler_tx,
            test_limits(),
            test_mm(flume::unbounded().0, true),
            sd_rx,
        );

        intake.on_abort(source.clone());

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
        sink: ResponseSink::Local(tx),
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
/// over-long prompt through to the scheduler with no error at all.
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
    let (mut intake, detok_rx, consumer, _tm_tx, _mm_rx) = make_intake_with(Limits {
        context_len: 4,
        ..test_limits()
    });
    intake.drive(generate_req(
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
    let (mut intake, detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
    let (tx, mut rx) = mpsc::channel(8);
    intake.drive(Request {
        rid: "41".into(),
        state: RequestState::Received,
        sink: ResponseSink::Local(tx),
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
    assert!(
        rx.try_recv().is_err(),
        "no response until the shard answers"
    );
}

/// Negative ids cannot decode (the shard's domain is `&[u32]`): rejected by
/// `validate` at `Received` — an `Error` to the sink, and the shard sees
/// NOTHING (validation runs before registration, so there is no entry to
/// leak and no decode job to drop).
#[test]
fn detokenize_negative_ids_reject_before_registration() {
    let (mut intake, detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
    let (tx, mut rx) = mpsc::channel(8);
    intake.drive(Request {
        rid: "43".into(),
        state: RequestState::Received,
        sink: ResponseSink::Local(tx),
        kind: RequestKind::Detokenize {
            token_ids: vec![1, -1],
        },
    });
    let Ok(ResponseItem::Error(err)) = rx.try_recv() else {
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
        tok_manager_tx: flume::unbounded().0,
        abort_tx,
        tokenizer_tx: tok_tx,
        detokenizer_tx: vec![detok_tx],
    };
    let (producer, _consumer) = to_scheduler(1);
    let (_tm_tx, tm_rx) = flume::unbounded();
    let (sd_tx, sd_rx) = flume::unbounded::<()>();
    std::mem::forget(sd_tx);
    let mut intake = Intake::new(
        tm_rx,
        abort_rx,
        senders,
        producer,
        test_limits(),
        test_mm(flume::unbounded().0, true),
        sd_rx,
    );

    intake.on_abort(AbortSource::Guard("pushed".into()));
    intake.on_abort(AbortSource::Guard("dropped".into()));

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
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    let mut req = generate_req(41, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = Some(vec![2_000_000_000]);
    }
    intake.drive(req);
    assert!(
        detok_rx.try_recv().is_err(),
        "a pre-registration reject must send NOTHING to the shard — a Deregister \
             here removes a live request's sink"
    );

    // A post-registration reject still deregisters (the leak fix stays fixed).
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    intake.drive(generate_req(
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
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    // top_p = 2.0 is outside (0, 1], so `SamplingParams::normalize` rejects it.
    let bad = SamplingParams {
        top_p: 2.0,
        ..Default::default()
    };
    intake.drive(generate_req(7, bad));

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
/// with a 400 — passed through, it reaches the embedding lookup
/// and kills the scheduler process (`make_intake` bounds vocab at 1000).
#[test]
fn out_of_vocab_input_ids_rejected() {
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    let mut req = generate_req(21, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = Some(vec![1, 2_000_000_000]);
    }
    intake.drive(req);
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
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    let mut req = generate_req(22, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = Some(vec![-1]);
    }
    intake.drive(req);
    match detok_rx.try_recv() {
        Err(_) | Ok(DetokMsg::Deregister { .. }) => {}
        Ok(_) => panic!("negative token id must not be admitted"),
    }

    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    let mut req = generate_req(23, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.token_ids_logprob = Some(vec![999_999]);
    }
    intake.drive(req);
    match detok_rx.try_recv() {
        Err(_) | Ok(DetokMsg::Deregister { .. }) => {}
        Ok(_) => panic!("out-of-vocab token_ids_logprob must not be admitted"),
    }
}

/// A valid request is registered and handed onward — never deregistered.
#[test]
fn admitted_request_keeps_registration() {
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    // Empty map → all sampling defaults, passes normalization.
    intake.drive(generate_req(9, SamplingParams::default()));

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
fn tokenize_failure_deregisters_via_intake() {
    let (intake, detok_rx, _consumer, tm_tx, _mm_rx) = make_intake();
    // The pool marks a failed encode as `Failed(err)` before returning it.
    let mut req = generate_req(11, SamplingParams::default());
    let _ = req
        .state
        .apply(Event::Error(Error::Tokenize("boom".into())));
    tm_tx.send(TmEvent::Tokenized(req)).unwrap();
    // Close the inbox so the run loop returns after draining the one event.
    drop(tm_tx);
    intake.run();

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
    let (intake, detok_rx, _consumer, tm_tx, _mm_rx) = make_intake_with_abort(abort_rx);
    abort_tx.send(AbortSource::Guard("rid-13".into())).unwrap();
    drop(abort_tx);
    drop(tm_tx);
    intake.run();

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
    let (intake, detok_rx, _consumer, tm_tx, _mm_rx) = make_intake();
    let mut req = generate_req(15, SamplingParams::default());
    // Simulate a successful pool return: ids filled, PreSendValidating.
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = Some(vec![1, 2, 3]);
    }
    req.state = RequestState::PreSendValidating;
    tm_tx.send(TmEvent::Tokenized(req)).unwrap();
    drop(tm_tx);
    intake.run();

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
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    let mut req = generate_req(21, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = None;
    }
    intake.drive(req);

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
        sink: ResponseSink::Local(tx),
        kind: RequestKind::Generate(Box::new(GenerateRequest {
            rid: rid.to_string().into(),
            text: Some("<image> hi".into()),
            mm: Some(Box::new(crate::message::request::MmData {
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
    let (mut intake, _detok_rx, consumer, _tm_tx, mm_rx) = make_intake();
    intake.drive(mm_generate_req("mm-gone"));
    mm_rx.try_recv().expect("parked to mm pool");

    // The worker parks its result, as it always does before MmEncoded.
    intake.mm.sidecar.park(
        "mm-gone".into(),
        crate::multi_modality::sidecar::MmSidecarEntry {
            features: crate::multi_modality::sidecar::FeatureStore::Inline(vec![]),
            grids: vec![],
            hashes: vec![],
            offsets: vec![],
            mrope: vec![],
            mrope_delta: 0,
        },
    );
    intake.on_abort(AbortSource::Guard("mm-gone".to_string().into()));
    assert_eq!(consumer.drain(16).headers.len(), 1, "only the AbortReq");

    // The late result must be dropped, not queued, and the sidecar purged.
    intake.on_mm_encoded("mm-gone".to_string().into(), vec![5, 6]);
    assert!(
        consumer.drain(16).headers.is_empty(),
        "cancelled, not queued"
    );
    assert!(intake.mm.sidecar.take("mm-gone").is_none(), "entry purged");
}

/// A multimodal request parks in `Encoding` (submitted to the mm worker
/// pool, not the tokenizer pool, not the ring) until `MmEncoded` resumes
/// it → ring.
#[test]
fn mm_request_parks_then_mm_encoded_pushes_to_ring() {
    let (mut intake, _detok_rx, consumer, _tm_tx, mm_rx) = make_intake();
    intake.drive(mm_generate_req("mm-1"));

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

    // The worker returns the final expanded ids → pushed to the ring.
    intake.on_mm_encoded("mm-1".to_string().into(), vec![5, 6, 7, 8]);
    let batch = consumer.drain(16);
    assert_eq!(batch.headers.len(), 1);
    assert_eq!(
        batch.lengths,
        vec![4],
        "expanded ids ride the columnar cell"
    );
}

/// A worker failure rejects the parked request (deregister, no ring push).
#[test]
fn mm_failure_rejects_parked_request() {
    let (mut intake, detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
    intake.drive(mm_generate_req("mm-2"));
    assert!(
        matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })),
        "registered before parking",
    );

    intake.on_mm_failed("mm-2".to_string().into(), "bad image".into());
    assert!(
        matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid })
                if rid.as_str() == "mm-2"),
        "mm failure must deregister",
    );
    assert!(consumer.drain(16).headers.is_empty(), "nothing queued");
}

/// On a non-multimodal model (`Mm::enabled == false`), image_data is silently
/// ignored and the request tokenizes as plain text — the Python
/// TokenizerManager behavior when `mm_processor is None`.
#[test]
fn mm_fields_ignored_when_disabled() {
    let (tok_tx, tok_rx) = flume::unbounded();
    let (detok_tx, _detok_rx) = flume::unbounded();
    let senders = Senders {
        tok_manager_tx: flume::unbounded().0,
        abort_tx: flume::unbounded().0,
        tokenizer_tx: tok_tx,
        detokenizer_tx: vec![detok_tx],
    };
    let (to_scheduler_tx, _consumer) = to_scheduler(16);
    let (_tm_tx, tm_rx) = flume::unbounded();
    let (mm_tx, mm_rx) = flume::unbounded();
    let (abort_tx, abort_rx) = flume::unbounded::<AbortSource>();
    std::mem::forget(abort_tx);
    let (sd_tx, sd_rx) = flume::unbounded::<()>();
    std::mem::forget(sd_tx);
    let mut intake = Intake::new(
        tm_rx,
        abort_rx,
        senders,
        to_scheduler_tx,
        test_limits(),
        test_mm(mm_tx, false),
        sd_rx,
    );

    intake.drive(mm_generate_req("mm-3"));
    assert!(
        mm_rx.try_recv().is_err(),
        "mm disabled: nothing submitted to the mm channel",
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
    let (mut intake, _detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
    intake.on_mm_encoded("ghost".to_string().into(), vec![1]);
    intake.on_mm_failed("ghost".to_string().into(), "boom".into());
    assert!(consumer.drain(16).headers.is_empty());
}
