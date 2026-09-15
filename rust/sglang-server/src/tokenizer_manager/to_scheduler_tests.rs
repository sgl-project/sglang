//! Tests for scheduler intake.

use super::*;
use crate::message::request::{GenerateBody, GenerateRequest};
use crate::message::response::ResponseSink;
use crate::message::sampling::SamplingParams;
use crate::message::types::HiddenStatesMode;
use crate::tokenizer_manager::channel::{ToSchedulerRx, to_scheduler};
use crate::utils::fsm::RequestState;
use tokio::sync::mpsc;

#[test]
fn canonical_media_pads_do_not_exempt_other_ids_from_vocabulary_validation() {
    use crate::message::request::MmPadSpan;

    let mut request = GenerateRequest {
        text: Some("original prompt".into()),
        ..Default::default()
    };
    assert!(request.set_canonical_mm_input(vec![], vec![]).is_err());
    assert_eq!(request.text.as_deref(), Some("original prompt"));
    assert!(request.input_ids.is_none());
    request
        .set_canonical_mm_input(
            vec![1, 1_000_005, 1_000_005, 2],
            vec![
                MmPadSpan {
                    start: 1,
                    end: 2,
                    pad_value: 1_000_005,
                },
                MmPadSpan {
                    start: 1,
                    end: 1,
                    pad_value: 1_000_005,
                },
            ],
        )
        .unwrap();
    assert!(validate_input_ids(&request, 128).is_ok());
    assert!(request.text.is_none());
    assert!(request.already_tokenized());
    assert_eq!(
        request.mm_pad_spans.len(),
        1,
        "overlapping equal pads normalize once"
    );
    request.input_ids.as_mut().unwrap()[0] = 1_000_005;
    assert!(validate_input_ids(&request, 128).is_err());
    request.input_ids.as_mut().unwrap()[0] = 1;
    request.input_ids.as_mut().unwrap()[1] = 1_000_006;
    assert!(validate_input_ids(&request, 128).is_err());
    for span in [
        MmPadSpan {
            start: 2,
            end: 1,
            pad_value: 5,
        },
        MmPadSpan {
            start: 0,
            end: 9,
            pad_value: 5,
        },
        MmPadSpan {
            start: 0,
            end: 0,
            pad_value: -1,
        },
        MmPadSpan {
            start: 0,
            end: 0,
            pad_value: 6,
        },
    ] {
        assert!(request.set_canonical_mm_input(vec![5], vec![span]).is_err());
    }
}

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
    lifecycle_rx: flume::Receiver<LifecycleEvent>,
) -> (
    Intake,
    flume::Receiver<DetokMsg>,
    ToSchedulerRx,
    flume::Sender<TmEvent>,
    flume::Receiver<MmRequest>,
) {
    make_intake_inner(test_limits(), lifecycle_rx)
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
    let (lifecycle_tx, lifecycle_rx) = flume::unbounded::<LifecycleEvent>();
    std::mem::forget(lifecycle_tx); // keep the lane open; tests end by dropping tm_tx
    make_intake_inner(limits, lifecycle_rx)
}

fn make_intake_inner(
    limits: Limits,
    lifecycle_rx: flume::Receiver<LifecycleEvent>,
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
        lifecycle_tx: flume::unbounded().0,
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
        lifecycle_rx,
        senders,
        to_scheduler_tx,
        limits,
        test_mm(mm_tx, true),
        sd_rx,
        None,
    );
    (intake, detok_rx, consumer, tm_tx, mm_rx)
}

/// An [`MmDispatch`] over `tx` with a fresh result store.
fn test_mm(tx: flume::Sender<MmRequest>, enabled: bool) -> MmDispatch {
    MmDispatch {
        enabled,
        tx,
        results: Default::default(),
    }
}

/// Both terminal error paths stop admitted scheduler work exactly once.
#[test]
fn every_abort_source_deregisters_and_stops_the_scheduler() {
    for source in [
        LifecycleEvent::GuardAbort("1".into()),
        LifecycleEvent::DetokAbort("1".into()),
    ] {
        let (mut intake, detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
        let (req, mut rx) = generate_req(1, SamplingParams::default());
        intake.drive(req);
        assert!(matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })));
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Prepared { rid, .. }) if rid.as_str() == "1")
        );
        assert_eq!(consumer.drain(16).headers.len(), 1);

        intake.on_lifecycle(source.clone());
        assert!(
            matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "1")
        );
        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
        let headers = consumer.drain(16).headers;
        assert_eq!(headers.len(), 1);
        let wire: Vec<rmpv::Value> = rmp_serde::from_slice(&headers[0]).unwrap();
        assert_eq!(wire[0].as_str(), Some("AbortReq"));
        assert_eq!(wire[1].as_str(), Some("1"));
        assert!(intake.in_flight.is_empty());
        intake.on_lifecycle(source);
        assert!(consumer.drain(16).headers.is_empty(), "duplicate abort");
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
        dp_size: 1,
        skip_tokenizer_init: false,
        vocab_size: 1000,
        context_len: NO_CONTEXT_CEILING,
        num_reserved_tokens: 0,
        allow_auto_truncate: false,
        max_return_hidden_states: HiddenStatesMode::Off,
        enable_custom_logit_processor: false,
        enable_strict_thinking: false,
        disable_radix_cache: false,
        hidden_size: 2,
    }
}

fn generate_req(
    id: u64,
    sampling_params: SamplingParams,
) -> (Request, mpsc::Receiver<ResponseItem>) {
    let (tx, rx) = mpsc::channel(8);
    let request = Request {
        rid: id.to_string().into(),
        state: RequestState::Received,
        sink: ResponseSink::Local(tx),
        kind: RequestKind::Generate(Box::new(GenerateRequest {
            rid: id.to_string().into(),
            input_ids: Some(vec![1, 2, 3]),
            sampling_params,
            ..Default::default()
        })),
    };
    (request, rx)
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

#[test]
fn hidden_states_match_python_modes_and_server_maximum() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../testdata/hidden_states_python.json")).unwrap();
    for case in fixtures["requests"].as_array().unwrap() {
        let body: GenerateBody = serde_json::from_value(case["body"].clone()).unwrap();
        let (requests, _) = body.into_requests().unwrap();
        let expected = case["modes"].as_array().unwrap();
        assert_eq!(requests.len(), expected.len());
        for (request, expected) in requests.into_iter().zip(expected) {
            let forwarded: GenerateBody =
                serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
            let (forwarded, _) = forwarded.into_requests().unwrap();
            for request in std::iter::once(request).chain(forwarded) {
                assert_eq!(
                    serde_json::to_value(request.return_hidden_states).unwrap(),
                    *expected
                );
                let header: serde_json::Value =
                    rmp_serde::from_slice(&request.encode_header().unwrap()).unwrap();
                assert_eq!(header[16], *expected);
                for maximum in [
                    HiddenStatesMode::Off,
                    HiddenStatesMode::Last,
                    HiddenStatesMode::Full,
                ] {
                    let (mut candidate, _) = generate_req(31, SamplingParams::default());
                    candidate.kind = RequestKind::Generate(Box::new(request.clone()));
                    let result = validate(
                        &mut candidate,
                        &Limits {
                            max_return_hidden_states: maximum,
                            ..test_limits()
                        },
                    );
                    if request.return_hidden_states > maximum {
                        let error = result.unwrap_err();
                        assert_eq!(error.http_status(), 400);
                        assert!(error.to_string().contains("--return-hidden-states-mode"));
                    } else {
                        result.unwrap();
                    }
                }
            }
        }
    }
    for invalid in fixtures["invalid_modes"].as_array().unwrap() {
        assert!(
            serde_json::from_value::<GenerateBody>(serde_json::json!({
                "input_ids": [1], "return_hidden_states": invalid
            }))
            .is_err()
        );
    }
}

#[test]
fn sampling_mask_options_match_python_and_survive_dp_forwarding() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../testdata/sampling_masks_python.json")).unwrap();
    for case in fixtures["requests"].as_array().unwrap() {
        let body: GenerateBody = serde_json::from_value(case["body"].clone()).unwrap();
        let (requests, _) = body.into_requests().unwrap();
        let expected = case["masks"].as_array().unwrap();
        assert_eq!(requests.len(), expected.len());
        for (request, expected) in requests.into_iter().zip(expected) {
            let forwarded: GenerateBody =
                serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
            let (forwarded, _) = forwarded.into_requests().unwrap();
            for request in std::iter::once(request).chain(forwarded) {
                let header: serde_json::Value =
                    rmp_serde::from_slice(&request.encode_header().unwrap()).unwrap();
                assert_eq!(header[14], *expected);
            }
        }
    }
    let body: GenerateBody = serde_json::from_value(serde_json::json!({
        "input_ids": [[1], [2]], "return_sampling_mask": [true, false],
        "sampling_params": {"n": 2},
    }))
    .unwrap();
    assert!(
        body.into_requests()
            .unwrap_err()
            .to_string()
            .contains("Cannot use list return_sampling_mask")
    );
}

#[test]
fn input_embeddings_preserve_python_shapes_through_dp_and_scheduler_intake() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../testdata/input_embeddings_python.json")).unwrap();
    for case in fixtures.as_array().unwrap() {
        let body: GenerateBody = serde_json::from_value(case["body"].clone()).unwrap();
        let (requests, _) = body.into_requests().unwrap();
        let expected = case["expected"].as_array().unwrap();
        assert_eq!(requests.len(), expected.len());
        for (request, expected) in requests.into_iter().zip(expected) {
            let forwarded: GenerateBody =
                serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
            let (forwarded, _) = forwarded.into_requests().unwrap();
            for request in std::iter::once(request).chain(forwarded) {
                for disable_radix_cache in [false, true] {
                    let (mut intake, _detok, consumer, _tm, _mm) = make_intake_with(Limits {
                        skip_tokenizer_init: true,
                        disable_radix_cache,
                        ..test_limits()
                    });
                    let (sink, mut response) = mpsc::channel(8);
                    intake.drive(Request {
                        rid: request.rid.clone(),
                        state: RequestState::Received,
                        sink: ResponseSink::Local(sink),
                        kind: RequestKind::Generate(Box::new(request.clone())),
                    });
                    let batch = consumer.drain(16);
                    if disable_radix_cache {
                        assert_eq!(batch.headers.len(), 1);
                        let header: serde_json::Value =
                            rmp_serde::from_slice(&batch.headers[0]).unwrap();
                        assert_eq!(header[5], *expected);
                        let rows = expected.as_array().unwrap().len();
                        assert_eq!(batch.lengths, vec![rows as u32]);
                        assert_eq!(
                            batch.ids[0].as_ref(),
                            [1i64.to_le_bytes()].repeat(rows).concat()
                        );
                    } else {
                        let ResponseItem::Error(error) = response.try_recv().unwrap() else {
                            panic!("embedding inputs require the radix-cache capability check");
                        };
                        assert_eq!(error.http_status(), 400);
                        assert!(error.to_string().contains("--disable-radix-cache"));
                        assert!(batch.headers.is_empty());
                    }
                }
            }
        }
    }
}

#[test]
fn delimiter_scoring_validates_final_prompt_positions_before_admission() {
    for (indices, truncate, should_pass) in [
        (vec![0, 3], false, true),
        (vec![0, 3], true, true),
        (vec![], false, false),
        (vec![-1, 3], false, false),
        (vec![1, 4], false, false),
        (vec![1, 4], true, false),
    ] {
        let (mut intake, _detok, consumer, _tm, _mm) = make_intake_with(Limits {
            skip_tokenizer_init: true,
            context_len: if truncate { 4 } else { 5 },
            allow_auto_truncate: truncate,
            ..test_limits()
        });
        let (sink, mut response) = mpsc::channel(8);
        let request = GenerateRequest {
            input_ids: Some(if truncate { vec![1; 5] } else { vec![1; 4] }),
            multi_item_delimiter_indices: Some(indices),
            sampling_params: SamplingParams {
                max_new_tokens: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        intake.drive(Request {
            rid: request.rid.clone(),
            state: RequestState::Received,
            sink: ResponseSink::Local(sink),
            kind: RequestKind::Generate(Box::new(request)),
        });
        let batch = consumer.drain(16);
        if should_pass {
            assert_eq!(batch.lengths, vec![4]);
        } else {
            let ResponseItem::Error(error) = response.try_recv().unwrap() else {
                panic!("invalid delimiter indices must return a validation error");
            };
            assert_eq!(error.http_status(), 400);
            assert!(error.to_string().contains("multi_item_delimiter_indices"));
            assert!(batch.headers.is_empty());
        }
    }
    for body in [
        serde_json::json!({"input_ids": [[1, 2], [3, 4]], "multi_item_delimiter_indices": [0, 1]}),
        serde_json::json!({"input_ids": [[1, 2], [3, 4]], "multi_item_delimiter_indices": [[0, 1]]}),
        serde_json::json!({"input_ids": [1, 2], "multi_item_delimiter_indices": [0, 1],
            "return_flat_raw_top_logprobs": true, "top_logprobs_num": 2}),
    ] {
        assert!(
            serde_json::from_value::<GenerateBody>(body)
                .unwrap()
                .into_requests()
                .is_err()
        );
    }
}

#[test]
fn positional_embeddings_match_python_normalization_owned_tensor_wire_and_validation() {
    use crate::message::embeddings::PositionalEmbeds;
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../testdata/positional_embeds_python.json")).unwrap();
    let index = fixture["field_index"].as_u64().unwrap() as usize;
    for case in fixture["accepted"].as_array().unwrap() {
        let (requests, _) = serde_json::from_value::<GenerateBody>(case["body"].clone())
            .unwrap()
            .into_requests()
            .unwrap();
        let expected = case["expected"].as_array().unwrap();
        assert_eq!(requests.len(), expected.len());
        for (request, expected) in requests.into_iter().zip(expected) {
            // Exercise the scalar body sent from DP ingress to a worker too.
            let (mut forwarded, _) =
                serde_json::from_value::<GenerateBody>(serde_json::to_value(&request).unwrap())
                    .unwrap()
                    .into_requests()
                    .unwrap();
            assert_eq!(forwarded.len(), 1);
            let request = forwarded.pop().unwrap();
            let (mut intake, _detok, consumer, _tm, _mm) = make_intake_with(Limits {
                skip_tokenizer_init: true,
                hidden_size: fixture["hidden_size"].as_u64().unwrap(),
                ..test_limits()
            });
            let (sink, _response) = mpsc::channel(8);
            intake.drive(Request {
                rid: request.rid.clone(),
                state: RequestState::Received,
                sink: ResponseSink::Local(sink),
                kind: RequestKind::Generate(Box::new(request)),
            });
            let batch = consumer.drain(16);
            assert_eq!(batch.headers.len(), 1, "{}", case["body"]);
            let header: rmpv::Value = rmp_serde::from_slice(&batch.headers[0]).unwrap();
            let wire = rmp_serde::to_vec(&header[index]).unwrap();
            let hex: String = wire.iter().map(|byte| format!("{byte:02x}")).collect();
            assert_eq!(hex, expected["wire_hex"].as_str().unwrap());
        }
    }
    for body in fixture["rejected"].as_array().unwrap() {
        let Ok(body) = serde_json::from_value::<GenerateBody>(body.clone()) else {
            continue;
        };
        let Ok((requests, _)) = body.into_requests() else {
            continue;
        };
        for request in requests {
            let (mut intake, _detok, consumer, _tm, _mm) = make_intake_with(Limits {
                skip_tokenizer_init: true,
                hidden_size: fixture["hidden_size"].as_u64().unwrap(),
                ..test_limits()
            });
            let (sink, mut response) = mpsc::channel(8);
            intake.drive(Request {
                rid: request.rid.clone(),
                state: RequestState::Received,
                sink: ResponseSink::Local(sink),
                kind: RequestKind::Generate(Box::new(request)),
            });
            assert!(consumer.drain(16).headers.is_empty());
            let ResponseItem::Error(error) = response.try_recv().unwrap() else {
                panic!("invalid positional overrides must return a validation error");
            };
            assert_eq!(error.http_status(), 400);
            assert!(error.to_string().contains("positional_embed_overrides"));
        }
    }
    for (position, value, should_pass) in
        [(3, 1.0, true), (4, 1.0, false), (1, f32::INFINITY, false)]
    {
        let (mut intake, _detok, consumer, _tm, _mm) = make_intake_with(Limits {
            skip_tokenizer_init: true,
            context_len: 4,
            allow_auto_truncate: true,
            ..test_limits()
        });
        let request = GenerateRequest {
            input_ids: Some(vec![1; 5]),
            positional_embed_overrides: Some(PositionalEmbeds {
                embeds: vec![vec![value, 2.0]],
                positions: vec![position],
            }),
            ..Default::default()
        };
        let (sink, mut response) = mpsc::channel(8);
        intake.drive(Request {
            rid: request.rid.clone(),
            state: RequestState::Received,
            sink: ResponseSink::Local(sink),
            kind: RequestKind::Generate(Box::new(request)),
        });
        assert_eq!(consumer.drain(16).headers.len(), usize::from(should_pass));
        if !should_pass {
            assert!(
                matches!(response.try_recv(), Ok(ResponseItem::Error(error)) if error.http_status() == 400 && error.to_string().contains("positional_embed_overrides"))
            );
        }
    }
}

#[test]
fn invalid_embedding_shapes_and_context_limits_do_not_reach_the_scheduler() {
    for (embeddings, truncate, should_pass) in [
        (vec![], false, false),
        (vec![vec![]], false, false),
        (vec![vec![1.0, 2.0], vec![3.0]], false, false),
        (vec![vec![f32::INFINITY, 2.0]], false, false),
        (vec![vec![1.0, 2.0]; 5], false, false),
        (vec![vec![1.0, 2.0]; 5], true, true),
    ] {
        let (mut intake, detok, consumer, _tm, _mm) = make_intake_with(Limits {
            skip_tokenizer_init: true,
            disable_radix_cache: true,
            context_len: 4,
            allow_auto_truncate: truncate,
            ..test_limits()
        });
        let (sink, mut response) = mpsc::channel(8);
        let request = GenerateRequest {
            input_embeds: Some(embeddings),
            return_prompt_token_ids: true,
            sampling_params: SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            },
            ..Default::default()
        };
        intake.drive(Request {
            rid: request.rid.clone(),
            state: RequestState::Received,
            sink: ResponseSink::Local(sink),
            kind: RequestKind::Generate(Box::new(request)),
        });
        let batch = consumer.drain(16);
        if should_pass {
            assert!(matches!(detok.try_recv(), Ok(DetokMsg::Register { .. })));
            assert!(
                matches!(detok.try_recv(), Ok(DetokMsg::Prepared { prompt_token_ids, .. })
                if prompt_token_ids.as_deref() == Some(&[1, 1, 1, 1][..]))
            );
            let header: serde_json::Value = rmp_serde::from_slice(&batch.headers[0]).unwrap();
            assert_eq!(header[5].as_array().unwrap().len(), 4);
            assert_eq!(batch.lengths, vec![4]);
            assert_eq!(header[8][0], 0);
        } else {
            let ResponseItem::Error(error) = response.try_recv().unwrap() else {
                panic!("invalid embeddings must return a validation error");
            };
            assert_eq!(error.http_status(), 400);
            assert!(batch.headers.is_empty());
        }
    }
}

#[test]
fn generation_policy_matches_python_and_requires_server_capabilities() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../testdata/generation_policy_python.json")).unwrap();
    for fixture in fixtures.as_array().unwrap() {
        let body: GenerateBody = serde_json::from_value(fixture["body"].clone()).unwrap();
        let (requests, _) = body.into_requests().unwrap();
        let expected = fixture["expected"].as_array().unwrap();
        assert_eq!(requests.len(), expected.len());
        for (request, expected) in requests.into_iter().zip(expected) {
            let forwarded: GenerateBody =
                serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
            let (forwarded, _) = forwarded.into_requests().unwrap();
            for request in std::iter::once(request).chain(forwarded) {
                for enabled in [false, true] {
                    let (mut intake, _detok, consumer, _tm, _mm) = make_intake_with(Limits {
                        enable_custom_logit_processor: enabled,
                        enable_strict_thinking: enabled,
                        ..test_limits()
                    });
                    let needs_capability = request
                        .custom_logit_processor
                        .as_ref()
                        .is_some_and(|processor| !processor.is_empty())
                        || request.max_thinking_tokens.is_some();
                    let (sink, mut response) = mpsc::channel(8);
                    intake.drive(Request {
                        rid: request.rid.clone(),
                        state: RequestState::Received,
                        sink: ResponseSink::Local(sink),
                        kind: RequestKind::Generate(Box::new(request.clone())),
                    });
                    let batch = consumer.drain(16);
                    if needs_capability && !enabled {
                        let ResponseItem::Error(error) = response.try_recv().unwrap() else {
                            panic!("unsupported policy must fail before scheduler admission");
                        };
                        assert_eq!(error.http_status(), 400);
                        assert!(error.to_string().contains("--enable-"));
                        assert!(batch.headers.is_empty());
                    } else {
                        assert_eq!(batch.headers.len(), 1);
                        let header: serde_json::Value =
                            rmp_serde::from_slice(&batch.headers[0]).unwrap();
                        assert_eq!(
                            serde_json::json!([header[23], header[33], header[8][25]]),
                            *expected
                        );
                    }
                }
            }
        }
    }
}

/// End-to-end through `drive`: an over-context request is rejected on the way
/// to the ring, after registration — so it must be deregistered, not leaked.
#[test]
fn over_context_request_deregisters_and_never_reaches_the_ring() {
    let (mut intake, detok_rx, consumer, _tm_tx, _mm_rx) = make_intake_with(Limits {
        context_len: 4,
        ..test_limits()
    });
    let (req, _rx) = generate_req(
        33,
        SamplingParams {
            max_new_tokens: Some(64),
            ..Default::default()
        },
    );
    intake.drive(req);
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
            skip_special_tokens: false,
        },
    });
    assert!(
        matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { rid, .. }) if rid.as_str() == "41"),
        "the sink must be registered before the decode job",
    );
    assert!(
        matches!(
            detok_rx.try_recv(),
            Ok(DetokMsg::Decode { rid, token_ids, skip_special_tokens: false })
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
            skip_special_tokens: true,
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

/// Saturating the control reserve cannot lose an abort. Pending cancellation
/// blocks new GPU work until every admitted generation has its abort queued.
#[test]
fn aborts_are_retried_until_the_scheduler_can_drain_them() {
    let (mut intake, _detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
    let mut clients = Vec::new();
    for id in 0..65 {
        let (req, rx) = generate_req(id, SamplingParams::default());
        clients.push(rx);
        intake.drive(req);
        assert_eq!(consumer.drain(1).headers.len(), 1);
    }
    for id in 0..65 {
        intake.on_lifecycle(LifecycleEvent::GuardAbort(id.to_string().into()));
    }
    assert!(
        !intake.pending_aborts.is_empty(),
        "control reserve saturated"
    );
    assert!(intake.in_flight.is_empty());

    let (req, mut rejected) = generate_req(100, SamplingParams::default());
    intake.drive(req);
    assert!(matches!(
        rejected.try_recv(),
        Ok(ResponseItem::Error(Error::QueueFull))
    ));

    let mut aborted = Vec::new();
    while !intake.pending_aborts.is_empty() {
        aborted.extend(consumer.drain(128).headers);
        intake.flush_pending_aborts();
    }
    aborted.extend(consumer.drain(128).headers);
    let ids: Vec<_> = aborted
        .iter()
        .map(|header| {
            let wire: Vec<rmpv::Value> = rmp_serde::from_slice(header).unwrap();
            assert_eq!(wire[0].as_str(), Some("AbortReq"));
            wire[1].as_str().unwrap().to_owned()
        })
        .collect();
    assert_eq!(ids, (0..65).map(|id| id.to_string()).collect::<Vec<_>>());
}

#[test]
fn disconnect_before_admission_and_late_tokenizer_result_do_not_schedule_work() {
    let (mut intake, detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
    let (req, rx) = generate_req(1, SamplingParams::default());
    drop(rx);
    intake.drive(req);
    assert!(intake.in_flight.is_empty());
    assert!(detok_rx.try_recv().is_err());

    let (tokenizer_tx, tokenizer_rx) = flume::unbounded();
    intake.senders.tokenizer_tx = tokenizer_tx;
    let (mut req, _rx) = generate_req(2, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = None;
        g.text = Some("hello".into());
    }
    intake.drive(req);
    let mut req = tokenizer_rx.try_recv().unwrap();
    assert!(matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })));
    intake.on_lifecycle(LifecycleEvent::GuardAbort(req.rid.clone()));
    req.state = RequestState::PreSendValidating;
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = Some(vec![1]);
    }
    intake.drive(req);
    assert!(intake.in_flight.is_empty());
    assert!(consumer.drain(16).headers.is_empty());
    assert!(matches!(
        detok_rx.try_recv(),
        Ok(DetokMsg::Deregister { .. })
    ));
    assert!(detok_rx.try_recv().is_err());
}

#[test]
fn finished_requests_release_tracking_without_scheduling_an_abort() {
    let (mut intake, _detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
    let (req, _rx) = generate_req(1, SamplingParams::default());
    intake.drive(req);
    assert_eq!(consumer.drain(16).headers.len(), 1);
    intake.on_lifecycle(LifecycleEvent::Finished("1".into()));
    assert!(intake.in_flight.is_empty());
    intake.on_lifecycle(LifecycleEvent::GuardAbort("1".into()));
    assert!(consumer.drain(16).headers.is_empty());
}

#[test]
fn full_worker_queues_cannot_block_scheduler_cancellation() {
    let (mut intake, _detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
    let (tok_tx, _tok_rx) = flume::bounded(0);
    intake.senders.tokenizer_tx = tok_tx;
    let (mut req, mut rx) = generate_req(1, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = None;
        g.text = Some("hello".into());
    }
    intake.drive(req);
    assert!(matches!(
        rx.try_recv(),
        Ok(ResponseItem::Error(Error::QueueFull))
    ));
    assert!(intake.in_flight.is_empty());

    let (detok_tx, detok_rx) = flume::bounded(2);
    intake.senders.detokenizer_tx = vec![detok_tx];
    let (req, _rx) = generate_req(2, SamplingParams::default());
    intake.drive(req);
    assert_eq!(consumer.drain(16).headers.len(), 1);
    // Registration and preparation fill the queue; cancellation still reaches Python.
    intake.on_lifecycle(LifecycleEvent::GuardAbort("2".into()));
    assert_eq!(consumer.drain(16).headers.len(), 1);
    assert_eq!(intake.pending_deregistrations.len(), 1);
    assert!(matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })));
    assert!(matches!(detok_rx.try_recv(), Ok(DetokMsg::Prepared { .. })));
    intake.flush_pending_deregistrations();
    assert!(intake.pending_deregistrations.is_empty());
    assert!(matches!(
        detok_rx.try_recv(),
        Ok(DetokMsg::Deregister { .. })
    ));
}

#[test]
fn full_preparation_queue_rejects_before_scheduler_admission() {
    let (mut intake, _detok, consumer, _tm, _mm) = make_intake();
    let (detok_tx, detok_rx) = flume::bounded(1);
    intake.senders.detokenizer_tx = vec![detok_tx];
    let (request, mut response) = generate_req(1, SamplingParams::default());
    intake.drive(request);
    assert!(matches!(
        response.try_recv(),
        Ok(ResponseItem::Error(Error::QueueFull))
    ));
    assert!(consumer.drain(16).headers.is_empty());
    assert!(intake.in_flight.is_empty());
    assert!(matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })));
    intake.flush_pending_deregistrations();
    assert!(matches!(
        detok_rx.try_recv(),
        Ok(DetokMsg::Deregister { .. })
    ));
    assert!(intake.pending_deregistrations.is_empty());
}

#[test]
fn explicit_abort_matches_client_ids_and_preserves_scheduler_response_registration() {
    let (mut intake, detok_rx, consumer, _tm_tx, mm_rx) = make_intake();
    let mut clients = Vec::new();
    let mut expected = Vec::new();
    for client_id in ["batch-1", "batch-1", "other"] {
        let (mut req, rx) = generate_req(1, SamplingParams::default());
        req.rid = Rid::from_client(client_id);
        if let RequestKind::Generate(g) = &mut req.kind {
            g.rid = req.rid.clone();
        }
        if client_id == "batch-1" {
            expected.push(req.rid.to_string());
        }
        clients.push(rx);
        intake.drive(req);
    }
    let (req, mut mm_response) = mm_generate_req("batch-2");
    intake.drive(req);
    assert_eq!(consumer.drain(16).headers.len(), 3);
    assert!(mm_rx.try_recv().is_ok());
    let mut registered = 0;
    let mut prepared = 0;
    for message in detok_rx.drain() {
        match message {
            DetokMsg::Register { .. } => registered += 1,
            DetokMsg::Prepared { .. } => prepared += 1,
            _ => panic!("request unexpectedly terminated"),
        }
    }
    assert_eq!((registered, prepared), (4, 3));

    intake.on_client_abort("", false);
    assert!(
        consumer.drain(16).headers.is_empty(),
        "empty rid is a no-op"
    );
    intake.on_client_abort("batch-", false);
    let mut actual: Vec<_> = consumer
        .drain(16)
        .headers
        .iter()
        .map(|header| {
            let wire: Vec<rmpv::Value> = rmp_serde::from_slice(header).unwrap();
            assert_eq!(wire[0].as_str(), Some("AbortReq"));
            wire[1].as_str().unwrap().to_owned()
        })
        .collect();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
    let Ok(ResponseItem::Done(output)) = mm_response.try_recv() else {
        panic!("CPU-only request must complete locally");
    };
    let reason = output.finish_reason.unwrap();
    assert_eq!(reason.kind_name(), Some("abort"));
    assert_eq!(reason.abort_status(), None);
    assert!(intake.pending_mm.is_empty());
    assert_eq!(
        intake.in_flight.len(),
        3,
        "GPU requests still await their final output"
    );
    assert!(
        matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "batch-2")
    );
    assert!(detok_rx.try_recv().is_err());
    intake.on_client_abort("batch-", false);
    assert!(
        consumer.drain(16).headers.is_empty(),
        "duplicate abort is idempotent"
    );
    intake.on_client_abort("", true);
    assert_eq!(
        consumer.drain(16).headers.len(),
        1,
        "abort-all includes the other request"
    );
}

/// The rid keys the detok table and rides on every chunk of every decode step,
/// so an unbounded client-supplied one is a recurring cost, not a one-off.
#[test]
fn oversized_rid_is_rejected() {
    let (mut req, _rx) = generate_req(51, SamplingParams::default());
    req.rid = "x".repeat(MAX_RID_LEN + 1).into();
    let err = validate(&mut req, &test_limits()).expect_err("must be rejected");
    assert_eq!(err.http_status(), 400);
    assert!(err.to_string().contains("over the"), "{err}");

    // A uuid-sized rid — what Python mints — is nowhere near the cap.
    let (mut req, _rx) = generate_req(52, SamplingParams::default());
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
    let (mut req, _rx) = generate_req(41, SamplingParams::default());
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
    let (req, _rx) = generate_req(
        42,
        SamplingParams {
            top_p: 2.0, // rejected by `normalize`, after registration
            ..Default::default()
        },
    );
    intake.drive(req);
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
    let (req, _rx) = generate_req(7, bad);
    intake.drive(req);

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
    let (mut req, _rx) = generate_req(21, SamplingParams::default());
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
    let (mut req, _rx) = generate_req(22, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.input_ids = Some(vec![-1]);
    }
    intake.drive(req);
    match detok_rx.try_recv() {
        Err(_) | Ok(DetokMsg::Deregister { .. }) => {}
        Ok(_) => panic!("negative token id must not be admitted"),
    }

    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    let (mut req, _rx) = generate_req(23, SamplingParams::default());
    if let RequestKind::Generate(g) = &mut req.kind {
        g.token_ids_logprob = Some(vec![999_999]);
    }
    intake.drive(req);
    match detok_rx.try_recv() {
        Err(_) | Ok(DetokMsg::Deregister { .. }) => {}
        Ok(_) => panic!("out-of-vocab token_ids_logprob must not be admitted"),
    }
}

#[test]
fn multimodal_sentinel_is_validated_after_expansion() {
    let (mut req, _rx) = generate_req(24, SamplingParams::default());
    let RequestKind::Generate(g) = &mut req.kind else {
        unreachable!()
    };
    g.input_ids = Some(vec![1, -103, 2]);
    g.mm = Some(Box::new(crate::message::request::MmData {
        audio_data: vec![crate::message::multimodal::MmItem::Source(
            "data:audio/wav;base64,xxxx".into(),
        )],
        ..Default::default()
    }));

    assert!(validate(&mut req, &test_limits()).is_ok());
    let RequestKind::Generate(g) = &mut req.kind else {
        unreachable!()
    };
    assert!(validate_input_ids(g, test_limits().vocab_size).is_err());
    g.input_ids = Some(vec![1, 103, 2]);
    assert!(validate_input_ids(g, test_limits().vocab_size).is_ok());
}

/// A valid request is registered and handed onward — never deregistered.
#[test]
fn admitted_request_keeps_registration() {
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    for (id, skip_specials, keep_stop) in [(9, true, false), (10, false, true)] {
        let (mut req, _rx) = generate_req(
            id,
            SamplingParams {
                skip_special_tokens: skip_specials,
                no_stop_trim: keep_stop,
                stop: Some(crate::message::types::OneOrMany::Many(vec!["END".into()])),
                stop_regex: Some(crate::message::types::OneOrMany::Many(vec!["E.D".into()])),
                ..Default::default()
            },
        );
        if let RequestKind::Generate(request) = &mut req.kind {
            request.return_prompt_token_ids = true;
            request.return_logprob = true;
            request.top_logprobs_num = 2;
            request.return_flat_raw_top_logprobs = true;
            request.return_flat_raw_top_logprobs_b64 = keep_stop;
            if keep_stop {
                request.sampling_params.normalize(false, 1000).unwrap();
            }
        }
        intake.drive(req);
        assert!(matches!(
            detok_rx.try_recv(),
            Ok(DetokMsg::Register { rid, skip_special_tokens, no_stop_trim, stop_texts, logprobs, .. })
                if rid.as_str() == id.to_string()
                    && skip_special_tokens == skip_specials && no_stop_trim == keep_stop
                    && stop_texts == vec!["END", "E.D"]
                    && logprobs.is_some_and(|options| options.top_k == 2 && options.flat && options.base64 == keep_stop)
        ));
        assert!(matches!(
            detok_rx.try_recv(),
            Ok(DetokMsg::Prepared { rid, prompt_token_ids, sampling_params, .. })
                if rid.as_str() == id.to_string()
                    && prompt_token_ids.as_deref() == Some(&[1, 2, 3][..])
                    && sampling_params.stop_strs == vec!["END"]
                    && sampling_params.stop_regex_strs == vec!["E.D"]
        ));
        assert!(
            detok_rx.try_recv().is_err(),
            "admitted request must not be deregistered",
        );
    }
}

/// Worker success and failure return through the same registered request path.
#[test]
fn tokenizer_return_preserves_or_cleans_up_registration() {
    for success in [false, true] {
        let (mut intake, detok_rx, consumer, _tm_tx, _mm_rx) = make_intake();
        let (tokenizer_tx, tokenizer_rx) = flume::unbounded();
        intake.senders.tokenizer_tx = tokenizer_tx;
        let (mut req, mut rx) = generate_req(11, SamplingParams::default());
        if let RequestKind::Generate(g) = &mut req.kind {
            g.input_ids = None;
            g.text = Some("hello".into());
            g.return_prompt_token_ids = true;
        }
        intake.drive(req);
        assert!(matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })));
        let mut req = tokenizer_rx.try_recv().unwrap();
        if success {
            if let RequestKind::Generate(g) = &mut req.kind {
                g.input_ids = Some(vec![1, 2, 3]);
            }
            req.state = RequestState::PreSendValidating;
        } else {
            req.state
                .apply(Event::Error(Error::Tokenize("boom".into())))
                .unwrap();
        }
        intake.drive(req);
        if success {
            assert_eq!(consumer.drain(16).headers.len(), 1);
            assert!(intake.in_flight.contains_key(&"11".into()));
            assert!(matches!(
                detok_rx.try_recv(),
                Ok(DetokMsg::Prepared { rid, prompt_token_ids, .. })
                    if rid.as_str() == "11"
                        && prompt_token_ids.as_deref() == Some(&[1, 2, 3][..])
            ));
            assert!(detok_rx.try_recv().is_err());
        } else {
            assert!(consumer.drain(16).headers.is_empty());
            assert!(intake.in_flight.is_empty());
            assert!(matches!(
                rx.try_recv(),
                Ok(ResponseItem::Error(Error::Tokenize(_)))
            ));
            assert!(
                matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "11")
            );
        }
    }
}

/// An abort deregisters (by the id hashed from the rid string), so a request
/// aborted before any terminal chunk can't leak.
#[test]
fn abort_deregisters_from_shard() {
    // Aborts arrive on their own unbounded lane now, not the request inbox.
    let (lifecycle_tx, lifecycle_rx) = flume::unbounded::<LifecycleEvent>();
    let (intake, detok_rx, _consumer, tm_tx, _mm_rx) = make_intake_with_abort(lifecycle_rx);
    lifecycle_tx
        .send(LifecycleEvent::GuardAbort("rid-13".into()))
        .unwrap();
    drop(lifecycle_tx);
    drop(tm_tx);
    intake.run();

    assert!(
        matches!(detok_rx.try_recv(), Ok(DetokMsg::Deregister { rid }) if rid.as_str() == "rid-13"),
        "abort must deregister by rid",
    );
    assert!(detok_rx.try_recv().is_err(), "no further shard messages");
}

/// If the pool is gone, a request needing tokenization is rejected +
/// deregistered, not silently dropped.
#[test]
fn tokenize_pool_gone_deregisters() {
    let (mut intake, detok_rx, _consumer, _tm_tx, _mm_rx) = make_intake();
    let (mut req, _rx) = generate_req(21, SamplingParams::default());
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
fn mm_generate_req(rid: &str) -> (Request, mpsc::Receiver<ResponseItem>) {
    let (tx, rx) = mpsc::channel(8);
    let request = Request {
        rid: rid.to_string().into(),
        state: RequestState::Received,
        sink: ResponseSink::Local(tx),
        kind: RequestKind::Generate(Box::new(GenerateRequest {
            rid: rid.to_string().into(),
            text: Some("<image> hi".into()),
            mm: Some(Box::new(crate::message::request::MmData {
                image_data: vec![crate::message::multimodal::MmItem::Source(
                    "data:image/jpeg;base64,xxxx".into(),
                )],
                ..Default::default()
            })),
            ..Default::default()
        })),
    };
    (request, rx)
}

/// An abort while the request is parked for MM cancels it: the pending
/// entry is removed, the worker's late result is dropped, and its parked
/// result-store entry is purged — no scheduler work runs for a dead client.
#[test]
fn abort_cancels_parked_mm_request() {
    let (mut intake, _detok_rx, consumer, _tm_tx, mm_rx) = make_intake();
    let (req, _rx) = mm_generate_req("mm-gone");
    intake.drive(req);
    mm_rx.try_recv().expect("parked to mm pool");

    // The worker parks its result, as it always does before MmEncoded.
    intake.mm.results.park(
        "mm-gone".into(),
        crate::multi_modality::result_store::MmEncodedEntry::Qwen(
            crate::multi_modality::result_store::QwenMmEncodedEntry {
                features: crate::multi_modality::result_store::FeatureStore::Inline(vec![]),
                grids: vec![],
                hashes: vec![],
                offsets: vec![],
                mrope: vec![],
                mrope_delta: 0,
            },
        ),
    );
    intake.on_abort(LifecycleEvent::GuardAbort("mm-gone".to_string().into()));
    assert!(
        consumer.drain(16).headers.is_empty(),
        "never sent to the scheduler"
    );

    // The late result must be dropped, not queued, and the parked result purged.
    intake.on_mm_encoded("mm-gone".to_string().into(), vec![5, 6], None);
    assert!(
        consumer.drain(16).headers.is_empty(),
        "cancelled, not queued"
    );
    assert!(intake.mm.results.take("mm-gone").is_none(), "entry purged");
}

/// A multimodal request parks in `Encoding` (submitted to the mm worker
/// pool, not the tokenizer pool, not the ring) until `MmEncoded` resumes
/// it → ring.
#[test]
fn mm_request_parks_then_mm_encoded_pushes_to_ring() {
    let (mut intake, detok_rx, consumer, _tm_tx, mm_rx) = make_intake();
    let (mut req, _rx) = mm_generate_req("mm-1");
    if let RequestKind::Generate(g) = &mut req.kind {
        g.return_prompt_token_ids = true;
    }
    intake.drive(req);
    assert!(matches!(detok_rx.try_recv(), Ok(DetokMsg::Register { .. })));
    assert!(detok_rx.try_recv().is_err(), "prompt is not ready yet");

    // Submitted to the mm pool with the typed work item; nothing on the ring yet.
    let sub = mm_rx.try_recv().expect("mm pool must receive the request");
    assert_eq!(sub.rid.as_str(), "mm-1");
    assert_eq!(sub.work.text.as_deref(), Some("<image> hi"));
    assert!(sub.work.input_ids.is_none(), "no client input_ids");
    assert_eq!(
        sub.work.image_data.first().and_then(|item| item.source()),
        Some("data:image/jpeg;base64,xxxx")
    );
    assert!(consumer.drain(16).headers.is_empty(), "parked, not queued");

    // The worker returns the final expanded ids → pushed to the ring.
    intake.on_mm_encoded(
        "mm-1".to_string().into(),
        vec![5, 6, 7, 8],
        Some(serde_json::Map::from_iter([(
            "media_stats".into(),
            serde_json::json!({"preprocess_e2e_ms": 17}),
        )])),
    );
    assert!(matches!(
        detok_rx.try_recv(),
        Ok(DetokMsg::Prepared { prompt_token_ids, response_metadata, .. })
            if prompt_token_ids.as_deref() == Some(&[5, 6, 7, 8][..])
                && response_metadata.as_ref().unwrap()["media_stats"]["preprocess_e2e_ms"] == 17
    ));
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
    let (req, mut rx) = mm_generate_req("mm-2");
    intake.drive(req);
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
    let ResponseItem::Error(error) = rx.try_recv().unwrap() else {
        panic!("invalid media must reach the HTTP consumer as an error");
    };
    assert_eq!(error.http_status(), 400);
    assert_eq!(error.to_string(), "encode failed: bad image");
    assert!(!intake.pending_mm.contains_key(&Rid::from("mm-2")));
}

/// On a non-multimodal model (`MmDispatch::enabled == false`), image_data is silently
/// ignored and the request tokenizes as plain text — the Python
/// TokenizerManager behavior when `mm_processor is None`.
#[test]
fn mm_fields_ignored_when_disabled() {
    let (tok_tx, tok_rx) = flume::unbounded();
    let (detok_tx, _detok_rx) = flume::unbounded();
    let senders = Senders {
        tok_manager_tx: flume::unbounded().0,
        lifecycle_tx: flume::unbounded().0,
        tokenizer_tx: tok_tx,
        detokenizer_tx: vec![detok_tx],
    };
    let (to_scheduler_tx, _consumer) = to_scheduler(16);
    let (_tm_tx, tm_rx) = flume::unbounded();
    let (mm_tx, mm_rx) = flume::unbounded();
    let (lifecycle_tx, lifecycle_rx) = flume::unbounded::<LifecycleEvent>();
    std::mem::forget(lifecycle_tx);
    let (sd_tx, sd_rx) = flume::unbounded::<()>();
    std::mem::forget(sd_tx);
    let mut intake = Intake::new(
        tm_rx,
        lifecycle_rx,
        senders,
        to_scheduler_tx,
        test_limits(),
        test_mm(mm_tx, false),
        sd_rx,
        None,
    );

    let (req, _rx) = mm_generate_req("mm-3");
    intake.drive(req);
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
    intake.on_mm_encoded("ghost".to_string().into(), vec![1], None);
    intake.on_mm_failed("ghost".to_string().into(), "boom".into());
    assert!(consumer.drain(16).headers.is_empty());
}
