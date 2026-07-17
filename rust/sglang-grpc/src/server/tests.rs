use super::{
    ChoiceTracker, DEFAULT_GRPC_MAX_MESSAGE_SIZE, GenerationOffsets, TypedTerminal,
    openai_status_code, resolve_max_message_size, terminal_error_status, typed_generation_chunk,
};
use crate::bridge::{ResponseData, TerminalError};
use tonic::Code;

#[test]
fn openai_status_code_uses_forwarded_status_when_present() {
    let meta_info = serde_json::from_value(serde_json::json!({"status_code": 429})).unwrap();
    assert_eq!(openai_status_code(&meta_info, 200), 429);
}

#[test]
fn openai_status_code_falls_back_when_missing_or_invalid() {
    assert_eq!(openai_status_code(&serde_json::Map::new(), 200), 200);

    let meta_info =
        serde_json::from_value(serde_json::json!({"status_code": "not-an-int"})).unwrap();
    assert_eq!(openai_status_code(&meta_info, 200), 200);
}

fn response_data(
    output_ids: Vec<i32>,
    text: &str,
    incremental: bool,
    meta_info: serde_json::Value,
) -> ResponseData {
    ResponseData {
        text: Some(text.to_string()),
        output_ids: Some(output_ids),
        embedding: None,
        choice_index: 1,
        incremental,
        json_bytes: None,
        meta_info: serde_json::from_value(meta_info).unwrap(),
    }
}

#[test]
fn cumulative_and_incremental_chunks_are_normalized_to_deltas() {
    let mut cumulative = GenerationOffsets::default();
    let first = typed_generation_chunk(
        response_data(vec![1, 2], "hé", false, serde_json::json!({})),
        false,
        &mut cumulative,
    )
    .unwrap();
    let second = typed_generation_chunk(
        response_data(vec![1, 2, 3], "héllo", false, serde_json::json!({})),
        false,
        &mut cumulative,
    )
    .unwrap();
    assert_eq!(first.delta_output_ids, vec![1, 2]);
    assert_eq!(first.delta_text, "hé");
    assert_eq!(second.delta_output_ids, vec![3]);
    assert_eq!(second.delta_text, "llo");

    let mut incremental = GenerationOffsets::default();
    let first = typed_generation_chunk(
        response_data(vec![1], "hé", true, serde_json::json!({})),
        false,
        &mut incremental,
    )
    .unwrap();
    let second = typed_generation_chunk(
        response_data(vec![2], "llo", true, serde_json::json!({})),
        false,
        &mut incremental,
    )
    .unwrap();
    assert_eq!(first.delta_output_ids, vec![1]);
    assert_eq!(second.delta_output_ids, vec![2]);
    assert_eq!(second.delta_text, "llo");
}

#[test]
fn choice_tracker_requires_one_terminal_per_choice() {
    let mut choices = ChoiceTracker::new(2);
    assert!(!choices.observe(1, false, false, "Generate").unwrap());
    assert!(!choices.observe(1, true, false, "Generate").unwrap());
    assert!(choices.observe(0, true, true, "Generate").unwrap());

    let mut missing = ChoiceTracker::new(2);
    let error = missing.observe(0, true, true, "Generate").unwrap_err();
    assert!(error.contains("1/2 terminal choices"));

    let mut duplicate = ChoiceTracker::new(2);
    duplicate.observe(0, true, false, "Generate").unwrap();
    assert!(
        duplicate
            .observe(0, true, false, "Generate")
            .unwrap_err()
            .contains("data after terminal")
    );

    let mut out_of_range = ChoiceTracker::new(2);
    assert!(
        out_of_range
            .observe(2, false, false, "Generate")
            .unwrap_err()
            .contains("outside 0..2")
    );
}

#[test]
fn logprobs_are_delta_aligned_and_prompt_logprobs_emit_once() {
    let mut offsets = GenerationOffsets::default();
    let metadata = serde_json::json!({
        "input_token_logprobs": [[-0.1, 10, "prompt"]],
        "input_top_logprobs": [[[-0.1, 10, "prompt"]]],
        "output_token_logprobs": [[-0.2, 20, "a"]],
        "output_top_logprobs": [[[-0.2, 20, "a"], [-1.2, 21, "b"]]]
    });
    let first = typed_generation_chunk(
        response_data(vec![20], "a", false, metadata),
        false,
        &mut offsets,
    )
    .unwrap();
    let logprobs = first.logprobs.unwrap();
    assert_eq!(logprobs.prompt.len(), 1);
    assert_eq!(logprobs.output.len(), first.delta_output_ids.len());
    assert_eq!(logprobs.output[0].top_logprobs.len(), 2);

    let second = typed_generation_chunk(
        response_data(
            vec![20, 22],
            "ac",
            false,
            serde_json::json!({
                "input_token_logprobs": [[-0.1, 10, "prompt"]],
                "output_token_logprobs": [[-0.2, 20, "a"], [-0.3, 22, "c"]]
            }),
        ),
        false,
        &mut offsets,
    )
    .unwrap();
    let logprobs = second.logprobs.unwrap();
    assert!(logprobs.prompt.is_empty());
    assert_eq!(logprobs.output.len(), second.delta_output_ids.len());
    assert_eq!(logprobs.output[0].token_id, 22);
}

#[test]
fn prompt_logprobs_do_not_require_output_logprobs() {
    let mut offsets = GenerationOffsets::default();
    let chunk = typed_generation_chunk(
        response_data(
            vec![],
            "",
            false,
            serde_json::json!({
                "input_token_logprobs": [[-0.1, 10, "prompt"]]
            }),
        ),
        true,
        &mut offsets,
    )
    .unwrap();
    let logprobs = chunk.logprobs.expect("prompt logprobs");
    assert_eq!(logprobs.prompt.len(), 1);
    assert!(logprobs.output.is_empty());
}

#[test]
fn malformed_logprobs_and_routed_experts_are_rejected() {
    let mut offsets = GenerationOffsets::default();
    let error = typed_generation_chunk(
        response_data(
            vec![1],
            "a",
            false,
            serde_json::json!({"output_token_logprobs": "invalid"}),
        ),
        false,
        &mut offsets,
    )
    .err()
    .expect("malformed output logprobs should fail");
    assert!(error.contains("non-array output logprobs"));

    let mut offsets = GenerationOffsets::default();
    let error = typed_generation_chunk(
        response_data(
            vec![1],
            "a",
            false,
            serde_json::json!({"routed_experts": [1, "invalid"]}),
        ),
        false,
        &mut offsets,
    )
    .err()
    .expect("malformed routed experts should fail");
    assert!(error.contains("non-i32"));
}

#[test]
fn finish_reasons_preserve_string_and_token_stops() {
    for (matched, expected_string, expected_token) in [
        (serde_json::json!("END"), Some("END"), None),
        (serde_json::json!(42), None, Some(42)),
    ] {
        let mut offsets = GenerationOffsets::default();
        let chunk = typed_generation_chunk(
            response_data(
                vec![],
                "",
                false,
                serde_json::json!({
                    "finish_reason": {"type": "stop", "matched": matched}
                }),
            ),
            true,
            &mut offsets,
        )
        .unwrap();
        let TypedTerminal::Finish(finish) = chunk.terminal.unwrap() else {
            panic!("expected finish terminal");
        };
        let reason = finish.stop_reason.unwrap().reason.unwrap();
        match reason {
            crate::proto::stop_reason::Reason::MatchedString(value) => {
                assert_eq!(Some(value.as_str()), expected_string);
            }
            crate::proto::stop_reason::Reason::MatchedTokenId(value) => {
                assert_eq!(Some(value), expected_token);
            }
        }
    }
}

#[test]
fn terminal_metadata_is_typed_and_not_duplicated_in_extensions() {
    let mut offsets = GenerationOffsets::default();
    let chunk = typed_generation_chunk(
        response_data(
            vec![7],
            "done",
            false,
            serde_json::json!({
                "finish_reason": {"type": "stop", "matched": "END"},
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "cached_tokens": 3,
                "weight_version": "v2"
            }),
        ),
        true,
        &mut offsets,
    )
    .unwrap();
    let usage = chunk.usage.unwrap();
    assert_eq!(usage.total_tokens, 9);
    assert_eq!(usage.cached_prompt_tokens, 3);
    assert!(matches!(chunk.terminal, Some(TypedTerminal::Finish(_))));
    let fields = chunk.engine_metadata.unwrap().fields;
    assert!(fields.contains_key("weight_version"));
    assert!(!fields.contains_key("finish_reason"));
    assert!(!fields.contains_key("prompt_tokens"));
}

#[test]
fn aborts_become_typed_generation_errors() {
    let mut offsets = GenerationOffsets::default();
    let chunk = typed_generation_chunk(
        response_data(
            vec![],
            "",
            false,
            serde_json::json!({
                "finish_reason": {"type": "abort", "message": "bad request", "status_code": 400}
            }),
        ),
        true,
        &mut offsets,
    )
    .unwrap();
    match chunk.terminal.unwrap() {
        TypedTerminal::Error(error) => {
            assert_eq!(
                error.code,
                crate::proto::GenerationErrorCode::InvalidArgument as i32
            );
            assert!(!error.retryable);
        }
        TypedTerminal::Finish(_) => panic!("expected typed error"),
    }
}

#[test]
fn terminal_error_status_maps_channel_full_to_resource_exhausted() {
    let status = terminal_error_status(TerminalError::ChannelFull {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::ResourceExhausted);
}

#[test]
fn terminal_error_status_maps_abort_to_cancelled() {
    let status = terminal_error_status(TerminalError::Aborted {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::Cancelled);
}

// SAFETY: env vars are process-global; bundle all SGLANG_TONIC_PAYLOAD cases into one
// serial test so they don't race each other under `cargo test`'s default parallelism.
#[test]
fn resolve_max_message_size_honors_env_var() {
    const VAR: &str = "SGLANG_TONIC_PAYLOAD";

    // Unset → default.
    // SAFETY: single-threaded test mutating process env (see note above).
    unsafe {
        std::env::remove_var(VAR);
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Valid override → honored verbatim.
    unsafe {
        std::env::set_var(VAR, "1048576");
    }
    assert_eq!(resolve_max_message_size(), 1_048_576);

    // Invalid string → warn + fall back to default.
    unsafe {
        std::env::set_var(VAR, "not-a-number");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Zero → treated as invalid, fall back to default.
    unsafe {
        std::env::set_var(VAR, "0");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    unsafe {
        std::env::remove_var(VAR);
    }
}
