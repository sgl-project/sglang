//! Protocol preparation invariants shared by rendering and inference.

use super::protocol::{
    ChatCompletionRequest, CompletionRequest, lower_chat_request, lower_text_completion_request,
    lower_token_ids_completion_request,
};
use super::test_utils::renderer_config;
use crate::SamplingDefaults;

#[test]
fn chat_lowering_preserves_template_controls_and_metadata() {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "rid": "chat-lowering",
        "chat_template_kwargs": {"enable_thinking": false},
        "continue_final_message": true,
        "top_k": 17,
        "min_p": 0.2,
        "min_tokens": 3,
        "stop_regex": "END[0-9]",
        "ignore_eos": true,
        "skip_special_tokens": false,
        "return_meta_info": false,
        "bootstrap_host": "prefill",
        "bootstrap_port": 8998,
        "bootstrap_room": 42
    }))
    .unwrap();

    assert_eq!(request.model, "model");
    assert_eq!(
        request
            .chat_template_kwargs
            .as_ref()
            .and_then(|args| args.get("enable_thinking")),
        Some(&serde_json::Value::Bool(false))
    );
    assert!(request.continue_final_message);
    assert_eq!(request.sampling_overrides.top_k, Some(17));
    assert_eq!(request.sampling_overrides.min_p, Some(0.2));
    assert_eq!(request.sampling_overrides.min_tokens, Some(3));
    assert_eq!(request.sampling_overrides.ignore_eos, Some(true));
    assert_eq!(request.sampling_overrides.skip_special_tokens, Some(false));
    assert_eq!(request.extensions.return_meta_info, Some(false));

    let (response_id, request) = lower_chat_request(&renderer_config(), request).unwrap();

    assert_eq!(response_id, "chat-lowering");
    assert_eq!(request.metadata.bootstrap_host.as_deref(), Some("prefill"));
    assert_eq!(request.metadata.bootstrap_port, Some(8998));
    assert_eq!(request.metadata.bootstrap_room, Some(42));
    assert_eq!(request.sampling_params.top_k, 17);
}

#[test]
fn chat_lowering_rejects_return_meta_info_until_supported() {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "return_meta_info": true
    }))
    .unwrap();

    let error = match lower_chat_request(&renderer_config(), request) {
        Ok(_) => panic!("return_meta_info=true must not be silently ignored"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("return_meta_info"));
}

#[test]
fn completion_sampling_defaults_follow_request_model_terminal_priority() {
    let mut config = renderer_config();
    config.default_sampling_params = SamplingDefaults {
        temperature: Some(0.6),
        top_p: Some(0.9),
        top_k: Some(32),
        min_p: Some(0.1),
        repetition_penalty: Some(1.1),
    };
    let omitted: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "prompt": "hello"
    }))
    .unwrap();
    let (_, requests) = lower_text_completion_request(&config, &omitted).unwrap();
    let sampling = &requests[0].options.sampling_params;
    assert_eq!(sampling.temperature, 0.6);
    assert_eq!(sampling.top_p, 0.9);
    assert_eq!(sampling.top_k, 32);
    assert_eq!(sampling.min_p, 0.1);
    assert_eq!(sampling.repetition_penalty, 1.1);

    let explicit: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "prompt": "hello",
        "temperature": 0.2,
        "top_p": 0.5,
        "top_k": 17,
        "min_p": 0.2,
        "repetition_penalty": 1.2
    }))
    .unwrap();
    let (_, requests) = lower_text_completion_request(&config, &explicit).unwrap();
    let sampling = &requests[0].options.sampling_params;
    assert!((sampling.temperature - 0.2).abs() < 1e-6);
    assert!((sampling.top_p - 0.5).abs() < 1e-6);
    assert_eq!(sampling.top_k, 17);
    assert_eq!(sampling.min_p, 0.2);
    assert_eq!(sampling.repetition_penalty, 1.2);
}

#[test]
fn unsupported_sglang_fields_are_rejected_instead_of_ignored() {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "input_ids": [1, 2, 3],
        "task": "domain"
    }))
    .unwrap();

    let error = lower_chat_request(&renderer_config(), request)
        .unwrap_err()
        .to_string();

    assert_eq!(error, "unsupported request fields: input_ids, task");
}

#[test]
fn chat_modalities_keep_the_typed_openai_contract() {
    for modalities in [serde_json::json!("text"), serde_json::json!(["vision"])] {
        let request = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "modalities": modalities
        });
        assert!(serde_json::from_value::<ChatCompletionRequest>(request).is_err());
    }

    let text_request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["text"]
    }))
    .unwrap();
    lower_chat_request(&renderer_config(), text_request).unwrap();
}

#[test]
fn reasoning_inputs_normalize_with_python_precedence() {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "high",
        "reasoning": {"effort": "none", "enabled": true},
        "chat_template_kwargs": {"thinking": true}
    }))
    .unwrap();
    let (_, request) = lower_chat_request(&renderer_config(), request).unwrap();
    let args = request.chat_template_args.unwrap();

    assert_eq!(
        serde_json::to_value(request.reasoning_effort).unwrap(),
        serde_json::json!("none")
    );
    assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
    assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));

    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "0.5"
    }))
    .unwrap();
    let (_, request) = lower_chat_request(&renderer_config(), request).unwrap();
    assert_eq!(
        serde_json::to_value(request.reasoning_effort).unwrap(),
        serde_json::json!(0.5)
    );
    assert_eq!(
        request
            .chat_template_args
            .as_ref()
            .and_then(|args| args.get("thinking")),
        Some(&serde_json::json!(true))
    );

    for invalid in [serde_json::json!(true), serde_json::json!(1.0)] {
        let request = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": invalid
        });
        assert!(serde_json::from_value::<ChatCompletionRequest>(request).is_err());
    }
}

#[test]
fn text_completion_lowering_attaches_batched_metadata_in_prompt_major_order() {
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "prompt": ["one", "two"],
        "n": 2,
        "rid": ["prompt-a", "prompt-b"],
        "cache_salt": ["tenant-a", "tenant-b"],
        "extra_key": ["", "batch"],
        "bootstrap_host": ["prefill-a", "prefill-b"],
        "bootstrap_port": [8998, null],
        "bootstrap_room": [41, 52],
        "priority": 7,
        "routed_dp_rank": 2
    }))
    .unwrap();
    let (response_id, requests) =
        lower_text_completion_request(&renderer_config(), &request).unwrap();

    assert_eq!(response_id, "prompt-a");
    assert_eq!(
        requests
            .iter()
            .flat_map(|request| request.requests.iter())
            .map(|request| request.rid.as_str())
            .collect::<Vec<_>>(),
        ["prompt-a-0", "prompt-a-1", "prompt-b-0", "prompt-b-1"]
    );
    assert_eq!(
        requests[0].requests[0].metadata.cache_salt.as_deref(),
        Some("tenant-a")
    );
    assert_eq!(requests[0].requests[1].metadata.extra_key, None);
    assert_eq!(
        requests[1].requests[0].metadata.extra_key.as_deref(),
        Some("batch")
    );
    assert_eq!(requests[0].requests[0].metadata.bootstrap_port, Some(8998));
    assert_eq!(requests[1].requests[0].metadata.bootstrap_port, None);
    assert_eq!(requests[0].requests[1].metadata.bootstrap_room, Some(41));
    assert_eq!(requests[1].requests[1].metadata.bootstrap_room, Some(52));
    assert_eq!(requests[1].requests[1].metadata.routed_dp_rank, Some(2));
}

#[test]
fn completion_lowering_validates_metadata_lengths_duplicates_and_scalar_rooms() {
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "prompt": ["one", "two"],
        "rid": ["duplicate", "duplicate"],
        "cache_salt": ["only-one"]
    }))
    .unwrap();
    let error = lower_text_completion_request(&renderer_config(), &request).unwrap_err();
    assert!(error.to_string().contains("duplicate request ID"));

    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "prompt": ["one", "two"],
        "cache_salt": ["only-one"]
    }))
    .unwrap();
    let error = lower_text_completion_request(&renderer_config(), &request).unwrap_err();
    assert!(error.to_string().contains("prompt batch size (2)"));

    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "prompt": ["one", "two"],
        "n": 2,
        "bootstrap_room": 90
    }))
    .unwrap();
    let (_, requests) = lower_text_completion_request(&renderer_config(), &request).unwrap();
    assert_eq!(
        requests
            .iter()
            .flat_map(|request| request.requests.iter())
            .map(|request| request.metadata.bootstrap_room)
            .collect::<Vec<_>>(),
        [Some(90), Some(90), Some(91), Some(91)]
    );
}

#[test]
fn completion_lowering_rejects_zero_max_tokens() {
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "prompt": "hello",
        "max_tokens": 0
    }))
    .unwrap();

    let error = lower_text_completion_request(&renderer_config(), &request).unwrap_err();

    assert_eq!(error.to_string(), "max_tokens must be positive");
}

#[test]
fn token_id_completion_lowering_attaches_batched_metadata() {
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model",
        "prompt": [[1, 2], [3]],
        "n": 2,
        "rid": ["tokens-a", "tokens-b"],
        "bootstrap_host": ["prefill-a", "prefill-b"],
        "bootstrap_port": [8998, 8999],
        "bootstrap_room": [41, 52]
    }))
    .unwrap();
    let (response_id, requests) =
        lower_token_ids_completion_request(&renderer_config(), &request).unwrap();

    assert_eq!(response_id, "tokens-a");
    assert_eq!(requests[2].rid, "tokens-b-0");
    assert_eq!(requests[2].input_ids, [3]);
    assert_eq!(
        requests[2].metadata.bootstrap_host.as_deref(),
        Some("prefill-b")
    );
    assert_eq!(requests[2].metadata.bootstrap_port, Some(8999));
    assert_eq!(requests[3].metadata.bootstrap_room, Some(52));
}
