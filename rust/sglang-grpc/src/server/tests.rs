use super::{
    ChoiceTracker, DEFAULT_GRPC_MAX_MESSAGE_SIZE, TypedTerminal, expected_generation_choices,
    generation_chunk, openai_status_code, resolve_max_message_size, terminal_error_status,
};
use crate::bridge::{ResponseData, ResponseMetadata, TerminalError};
use prost::Message;
use tonic::Code;

#[derive(Clone, PartialEq, Message)]
struct LegacySamplingParams {
    #[prost(string, optional, tag = "14")]
    json_schema: Option<String>,
    #[prost(string, optional, tag = "15")]
    regex: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct LegacyGenerateResponse {
    #[prost(int32, repeated, tag = "1")]
    output_ids: Vec<i32>,
    #[prost(map = "string, string", tag = "2")]
    meta_info: std::collections::HashMap<String, String>,
    #[prost(bool, tag = "3")]
    finished: bool,
}

#[test]
#[allow(deprecated)]
fn legacy_generation_wire_fields_remain_compatible() {
    let legacy_sampling = LegacySamplingParams {
        json_schema: Some("{\"type\":\"string\"}".into()),
        regex: Some("[a-z]+".into()),
    };
    let decoded_sampling =
        crate::proto::SamplingParams::decode(legacy_sampling.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded_sampling.json_schema, legacy_sampling.json_schema);
    assert_eq!(decoded_sampling.regex, legacy_sampling.regex);

    let legacy_response = LegacyGenerateResponse {
        output_ids: vec![1, 2, 3],
        meta_info: [("finish_reason".into(), "\"stop\"".into())]
            .into_iter()
            .collect(),
        finished: true,
    };
    let decoded_response =
        crate::proto::GenerateResponse::decode(legacy_response.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded_response.output_ids, legacy_response.output_ids);
    assert_eq!(decoded_response.meta_info, legacy_response.meta_info);
    assert!(decoded_response.finished);
    assert!(decoded_response.request_id.is_empty());
    assert!(decoded_response.delta_output_ids.is_empty());
    assert!(decoded_response.terminal.is_none());
}

#[test]
fn openai_status_code_uses_forwarded_status_when_present() {
    let meta_info = [("status_code".to_string(), "429".to_string())]
        .into_iter()
        .collect();
    assert_eq!(openai_status_code(&meta_info, 200), 429);
}

#[test]
fn openai_status_code_falls_back_when_missing_or_invalid() {
    assert_eq!(
        openai_status_code(&std::collections::HashMap::new(), 200),
        200
    );

    let meta_info = [("status_code".into(), "not-an-int".into())]
        .into_iter()
        .collect();
    assert_eq!(openai_status_code(&meta_info, 200), 200);
}

fn response_data(output_ids: Vec<i32>, text: &str, meta_info: serde_json::Value) -> ResponseData {
    ResponseData {
        text: Some(text.to_string()),
        legacy_output_ids: Some(output_ids.clone()),
        output_ids: Some(output_ids),
        embedding: None,
        choice_index: 1,
        json_bytes: None,
        meta_info: ResponseMetadata::Generate {
            legacy: std::collections::HashMap::new(),
            typed: serde_json::from_value(meta_info).unwrap(),
        },
    }
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
fn typed_logprobs_are_aligned_with_delta_tokens() {
    let metadata = serde_json::json!({
        "input_token_logprobs": [[-0.1, 10, "prompt"]],
        "input_top_logprobs": [[[-0.1, 10, "prompt"]]],
        "output_token_logprobs": [[-0.2, 20, "a"]],
        "output_top_logprobs": [[[-0.2, 20, "a"], [-1.2, 21, "b"]]]
    });
    let first = generation_chunk(response_data(vec![20], "a", metadata), false).unwrap();
    let logprobs = first.logprobs.unwrap();
    assert_eq!(logprobs.prompt.len(), 1);
    assert_eq!(logprobs.output.len(), first.delta_output_ids.len());
    assert_eq!(logprobs.output[0].top_logprobs.len(), 2);
}

#[test]
fn prompt_logprobs_do_not_require_output_logprobs() {
    let chunk = generation_chunk(
        response_data(
            vec![],
            "",
            serde_json::json!({
                "input_token_logprobs": [[-0.1, 10, "prompt"]]
            }),
        ),
        true,
    )
    .unwrap();
    let logprobs = chunk.logprobs.expect("prompt logprobs");
    assert_eq!(logprobs.prompt.len(), 1);
    assert!(logprobs.output.is_empty());
}

#[test]
fn malformed_logprobs_are_rejected() {
    let error = generation_chunk(
        response_data(
            vec![1],
            "a",
            serde_json::json!({"output_token_logprobs": "invalid"}),
        ),
        false,
    )
    .err()
    .expect("malformed output logprobs should fail");
    assert!(error.contains("non-array output logprobs"));
}

#[test]
fn finish_reasons_preserve_string_and_token_stops() {
    for (matched, expected_string, expected_token) in [
        (serde_json::json!("END"), Some("END"), None),
        (serde_json::json!(42), None, Some(42)),
    ] {
        let chunk = generation_chunk(
            response_data(
                vec![],
                "",
                serde_json::json!({
                    "finish_reason": {"type": "stop", "matched": matched}
                }),
            ),
            true,
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
    let chunk = generation_chunk(
        response_data(
            vec![7],
            "done",
            serde_json::json!({
                "id": "internal-child-rid",
                "finish_reason": {"type": "stop", "matched": "END"},
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "cached_tokens": 3,
                "weight_version": "v2"
            }),
        ),
        true,
    )
    .unwrap();
    let usage = chunk.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 8);
    assert_eq!(usage.completion_tokens, 1);
    assert_eq!(usage.cached_prompt_tokens, 3);
    assert!(matches!(chunk.terminal, Some(TypedTerminal::Finish(_))));
    let fields = chunk.engine_metadata.unwrap().fields;
    assert!(fields.contains_key("weight_version"));
    assert!(!fields.contains_key("id"));
    assert!(!fields.contains_key("finish_reason"));
    assert!(!fields.contains_key("prompt_tokens"));
}

#[test]
fn generate_response_uses_the_rust_owned_public_request_id() {
    let chunk = generation_chunk(
        response_data(
            vec![7],
            "done",
            serde_json::json!({
                "id": "internal-child-rid",
                "finish_reason": {"type": "stop"}
            }),
        ),
        true,
    )
    .unwrap();
    assert!(chunk.usage.is_none());
    let response = chunk.into_response("public-parent-rid".into(), true);

    assert_eq!(response.request_id, "public-parent-rid");
    assert_eq!(response.output_ids, vec![7]);
    assert_eq!(response.delta_output_ids, vec![7]);
    assert!(response.finished);
    assert!(
        !response
            .engine_metadata
            .is_some_and(|metadata| metadata.fields.contains_key("id"))
    );
}

#[test]
fn generation_rejects_invalid_or_unbounded_choice_counts() {
    let request_with_n = |n| crate::proto::GenerateRequest {
        sampling_params: Some(crate::proto::SamplingParams {
            n: Some(n),
            ..Default::default()
        }),
        ..Default::default()
    };

    assert_eq!(expected_generation_choices(&request_with_n(1)).unwrap(), 1);
    assert_eq!(
        expected_generation_choices(&request_with_n(1024)).unwrap(),
        1024
    );
    for invalid in [0, -1, 1025, i32::MAX] {
        let error = expected_generation_choices(&request_with_n(invalid)).unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
    }
}

#[test]
fn aborts_become_typed_generation_errors() {
    let chunk = generation_chunk(
        response_data(
            vec![],
            "",
            serde_json::json!({
                "finish_reason": {"type": "abort", "message": "bad request", "status_code": 400}
            }),
        ),
        true,
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
