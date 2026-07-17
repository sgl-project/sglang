use std::collections::HashMap;

use crate::proto;

fn regex_escape_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(
            character,
            '.' | '+' | '*' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// Convert proto SamplingParams to a serde_json map (used as Python dict via PyO3).
fn sampling_params_to_map(
    params: &Option<proto::SamplingParams>,
) -> Result<serde_json::Value, String> {
    match params {
        Some(p) => {
            if p.n.is_some_and(|n| n <= 0) {
                return Err("n must be greater than zero".into());
            }
            let mut map = serde_json::Map::new();
            if let Some(v) = p.temperature {
                map.insert("temperature".into(), serde_json::json!(v));
            }
            if let Some(v) = p.top_p {
                map.insert("top_p".into(), serde_json::json!(v));
            }
            if let Some(v) = p.top_k {
                map.insert("top_k".into(), serde_json::json!(v));
            }
            if let Some(v) = p.min_p {
                map.insert("min_p".into(), serde_json::json!(v));
            }
            if let Some(v) = p.frequency_penalty {
                map.insert("frequency_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.presence_penalty {
                map.insert("presence_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.repetition_penalty {
                map.insert("repetition_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.max_new_tokens {
                map.insert("max_new_tokens".into(), serde_json::json!(v));
            }
            if let Some(v) = p.min_new_tokens {
                map.insert("min_new_tokens".into(), serde_json::json!(v));
            }
            if !p.stop.is_empty() {
                map.insert("stop".into(), serde_json::json!(p.stop));
            }
            if !p.stop_token_ids.is_empty() {
                map.insert("stop_token_ids".into(), serde_json::json!(p.stop_token_ids));
            }
            if let Some(v) = p.ignore_eos {
                map.insert("ignore_eos".into(), serde_json::json!(v));
            }
            if let Some(v) = p.n {
                map.insert("n".into(), serde_json::json!(v));
            }
            if let Some(v) = p.seed {
                map.insert("sampling_seed".into(), serde_json::json!(v));
            }
            if let Some(max_thinking_tokens) = p.max_thinking_tokens {
                map.insert(
                    "custom_params".into(),
                    serde_json::json!({"thinking_budget": max_thinking_tokens}),
                );
            }
            if let Some(guided) = p.guided_decoding.as_ref() {
                use proto::guided_decoding::Constraint;
                match guided.constraint.as_ref() {
                    Some(Constraint::JsonSchema(value)) if !value.is_empty() => {
                        map.insert("json_schema".into(), serde_json::json!(value));
                    }
                    Some(Constraint::Regex(value)) if !value.is_empty() => {
                        map.insert("regex".into(), serde_json::json!(value));
                    }
                    Some(Constraint::Ebnf(value)) if !value.is_empty() => {
                        map.insert("ebnf".into(), serde_json::json!(value));
                    }
                    Some(Constraint::Choice(choice))
                        if !choice.values.is_empty()
                            && choice.values.iter().all(|value| !value.is_empty()) =>
                    {
                        let alternatives = choice
                            .values
                            .iter()
                            .map(|value| regex_escape_literal(value))
                            .collect::<Vec<_>>()
                            .join("|");
                        map.insert(
                            "regex".into(),
                            serde_json::json!(format!("(?:{alternatives})")),
                        );
                    }
                    Some(Constraint::StructuralTag(value)) if !value.is_empty() => {
                        map.insert("structural_tag".into(), serde_json::json!(value));
                    }
                    Some(Constraint::Choice(_)) => {
                        return Err("guided choice must contain only non-empty values".into());
                    }
                    Some(_) => return Err("guided decoding constraint must not be empty".into()),
                    None => return Err("guided decoding constraint must be specified".into()),
                }
            }
            Ok(serde_json::Value::Object(map))
        }
        None => Ok(serde_json::Value::Object(serde_json::Map::new())),
    }
}

fn trace_headers_to_json(headers: &HashMap<String, String>) -> Option<serde_json::Value> {
    if headers.is_empty() {
        None
    } else {
        Some(serde_json::json!(headers))
    }
}

fn insert_disaggregated_params(
    request: &mut HashMap<String, serde_json::Value>,
    params: &Option<proto::DisaggregatedParams>,
) {
    if let Some(params) = params {
        request.insert(
            "bootstrap_host".into(),
            serde_json::json!(params.bootstrap_host),
        );
        request.insert(
            "bootstrap_port".into(),
            serde_json::json!(params.bootstrap_port),
        );
        request.insert(
            "bootstrap_room".into(),
            serde_json::json!(params.bootstrap_room),
        );
    }
}

fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub(crate) fn extract_model_path(json_info: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(json_info) {
        Ok(value) => value
            .get("model_path")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .unwrap_or_default(),
        Err(err) => {
            tracing::warn!("Failed to parse model info JSON: {}", err);
            String::new()
        }
    }
}

/// Build a request dict for GenerateReqInput from proto TextGenerateRequest fields.
pub(crate) fn build_text_generate_dict(
    rid: &str,
    req: &proto::TextGenerateRequest,
) -> Result<HashMap<String, serde_json::Value>, String> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("text".into(), serde_json::json!(req.text));
    d.insert(
        "sampling_params".into(),
        sampling_params_to_map(&req.sampling_params)?,
    );
    d.insert(
        "stream".into(),
        serde_json::json!(req.stream.unwrap_or(false)),
    );
    d.insert(
        "return_logprob".into(),
        serde_json::json!(req.return_logprob.unwrap_or(false)),
    );
    d.insert(
        "top_logprobs_num".into(),
        serde_json::json!(req.top_logprobs_num.unwrap_or(0)),
    );
    d.insert(
        "logprob_start_len".into(),
        serde_json::json!(req.logprob_start_len.unwrap_or(-1)),
    );
    d.insert(
        "return_text_in_logprobs".into(),
        serde_json::json!(req.return_text_in_logprobs.unwrap_or(false)),
    );
    if let Some(ref lp) = req.lora_path {
        d.insert("lora_path".into(), serde_json::json!(lp));
    }
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(rank) = req.routed_dp_rank {
        d.insert("routed_dp_rank".into(), serde_json::json!(rank));
    }
    if let Some(ref session_id) = req.session_id {
        d.insert("session_id".into(), serde_json::json!(session_id));
    }
    if let Some(priority) = req.priority {
        d.insert("priority".into(), serde_json::json!(priority));
    }
    if let Some(params) = req.sampling_params.as_ref() {
        d.insert(
            "require_reasoning".into(),
            serde_json::json!(params.require_reasoning.unwrap_or(false)),
        );
    }
    insert_disaggregated_params(&mut d, &req.disaggregated_params);
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    Ok(d)
}

/// Build a request dict for GenerateReqInput from proto GenerateRequest (tokenized).
pub(crate) fn build_generate_dict(
    rid: &str,
    req: &proto::GenerateRequest,
) -> Result<HashMap<String, serde_json::Value>, String> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("input_ids".into(), serde_json::json!(req.input_ids));
    d.insert(
        "sampling_params".into(),
        sampling_params_to_map(&req.sampling_params)?,
    );
    d.insert(
        "stream".into(),
        serde_json::json!(req.stream.unwrap_or(false)),
    );
    d.insert(
        "return_logprob".into(),
        serde_json::json!(req.return_logprob.unwrap_or(false)),
    );
    d.insert(
        "top_logprobs_num".into(),
        serde_json::json!(req.top_logprobs_num.unwrap_or(0)),
    );
    d.insert(
        "logprob_start_len".into(),
        serde_json::json!(req.logprob_start_len.unwrap_or(-1)),
    );
    if let Some(ref lp) = req.lora_path {
        d.insert("lora_path".into(), serde_json::json!(lp));
    }
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(rank) = req.routed_dp_rank {
        d.insert("routed_dp_rank".into(), serde_json::json!(rank));
    }
    if let Some(ref session_id) = req.session_id {
        d.insert("session_id".into(), serde_json::json!(session_id));
    }
    if let Some(priority) = req.priority {
        d.insert("priority".into(), serde_json::json!(priority));
    }
    if let Some(params) = req.sampling_params.as_ref() {
        d.insert(
            "require_reasoning".into(),
            serde_json::json!(params.require_reasoning.unwrap_or(false)),
        );
    }
    insert_disaggregated_params(&mut d, &req.disaggregated_params);
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    Ok(d)
}

/// Build a request dict for EmbeddingReqInput from proto TextEmbedRequest.
pub(crate) fn build_text_embed_dict(
    rid: &str,
    req: &proto::TextEmbedRequest,
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("text".into(), serde_json::json!(req.text));
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    d
}

/// Build a request dict for EmbeddingReqInput from proto EmbedRequest (tokenized).
pub(crate) fn build_embed_dict(
    rid: &str,
    req: &proto::EmbedRequest,
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("input_ids".into(), serde_json::json!(req.input_ids));
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    d
}

/// Build a request dict for EmbeddingReqInput from proto ClassifyRequest.
pub(crate) fn build_classify_dict(
    rid: &str,
    req: &proto::ClassifyRequest,
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    if !req.text.is_empty() {
        d.insert("text".into(), serde_json::json!(req.text));
    }
    if !req.input_ids.is_empty() {
        d.insert("input_ids".into(), serde_json::json!(req.input_ids));
    }
    if let Some(ref rk) = req.routing_key {
        d.insert("routing_key".into(), serde_json::json!(rk));
    }
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_dicts_include_session_id() {
        let session_id = Some("session-1".to_string());
        let text_req = proto::TextGenerateRequest {
            session_id: session_id.clone(),
            ..Default::default()
        };
        let token_req = proto::GenerateRequest {
            session_id,
            ..Default::default()
        };

        assert_eq!(
            build_text_generate_dict("request-1", &text_req)
                .unwrap()
                .get("session_id"),
            Some(&serde_json::json!("session-1"))
        );
        assert_eq!(
            build_generate_dict("request-2", &token_req)
                .unwrap()
                .get("session_id"),
            Some(&serde_json::json!("session-1"))
        );
    }

    #[test]
    fn generate_dicts_include_disaggregated_params() {
        let disaggregated_params = Some(proto::DisaggregatedParams {
            bootstrap_host: "10.0.0.1".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: i64::MAX,
        });
        let text_req = proto::TextGenerateRequest {
            disaggregated_params: disaggregated_params.clone(),
            ..Default::default()
        };
        let token_req = proto::GenerateRequest {
            disaggregated_params,
            ..Default::default()
        };

        for request in [
            build_text_generate_dict("request-1", &text_req),
            build_generate_dict("request-2", &token_req),
        ] {
            let request = request.unwrap();
            assert_eq!(
                request.get("bootstrap_host"),
                Some(&serde_json::json!("10.0.0.1"))
            );
            assert_eq!(
                request.get("bootstrap_port"),
                Some(&serde_json::json!(8998))
            );
            assert_eq!(
                request.get("bootstrap_room"),
                Some(&serde_json::json!(i64::MAX))
            );
        }
    }

    #[test]
    fn generate_dicts_omit_disaggregated_params_when_absent() {
        let text_request =
            build_text_generate_dict("request-1", &proto::TextGenerateRequest::default()).unwrap();
        let token_request =
            build_generate_dict("request-2", &proto::GenerateRequest::default()).unwrap();

        for request in [text_request, token_request] {
            assert!(!request.contains_key("bootstrap_host"));
            assert!(!request.contains_key("bootstrap_port"));
            assert!(!request.contains_key("bootstrap_room"));
        }
    }

    #[test]
    fn generate_maps_seed_priority_reasoning_and_thinking_budget() {
        let request = proto::GenerateRequest {
            input_ids: vec![1, 2],
            priority: Some(7),
            sampling_params: Some(proto::SamplingParams {
                seed: Some(42),
                require_reasoning: Some(true),
                max_thinking_tokens: Some(128),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mapped = build_generate_dict("request", &request).unwrap();
        assert_eq!(mapped["priority"], serde_json::json!(7));
        assert_eq!(mapped["require_reasoning"], serde_json::json!(true));
        assert_eq!(
            mapped["sampling_params"]["sampling_seed"],
            serde_json::json!(42)
        );
        assert_eq!(
            mapped["sampling_params"]["custom_params"]["thinking_budget"],
            serde_json::json!(128)
        );
    }

    #[test]
    fn generate_maps_all_guided_decoding_constraints() {
        use proto::guided_decoding::Constraint;
        let cases = [
            (
                Constraint::JsonSchema("{\"type\":\"object\"}".into()),
                "json_schema",
                "{\"type\":\"object\"}",
            ),
            (Constraint::Regex("[0-9]+".into()), "regex", "[0-9]+"),
            (
                Constraint::Ebnf("root ::= 'yes'".into()),
                "ebnf",
                "root ::= 'yes'",
            ),
            (
                Constraint::StructuralTag("{\"type\":\"structural_tag\"}".into()),
                "structural_tag",
                "{\"type\":\"structural_tag\"}",
            ),
        ];
        for (constraint, key, expected) in cases {
            let request = proto::GenerateRequest {
                sampling_params: Some(proto::SamplingParams {
                    guided_decoding: Some(proto::GuidedDecoding {
                        constraint: Some(constraint),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let mapped = build_generate_dict("request", &request).unwrap();
            assert_eq!(mapped["sampling_params"][key], serde_json::json!(expected));
        }
    }

    #[test]
    fn guided_choices_are_regex_escaped_and_validated() {
        let request = |values| proto::GenerateRequest {
            sampling_params: Some(proto::SamplingParams {
                guided_decoding: Some(proto::GuidedDecoding {
                    constraint: Some(proto::guided_decoding::Constraint::Choice(
                        proto::ChoiceConstraint { values },
                    )),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mapped =
            build_generate_dict("request", &request(vec!["a+b".into(), "x.y".into()])).unwrap();
        assert_eq!(mapped["sampling_params"]["regex"], "(?:a\\+b|x\\.y)");
        assert!(build_generate_dict("request", &request(vec![])).is_err());
    }
}
