use std::collections::HashMap;

use crate::proto;

/// Convert proto SamplingParams to a serde_json map (used as Python dict via PyO3).
fn sampling_params_to_map(params: &Option<proto::SamplingParams>) -> serde_json::Value {
    match params {
        Some(p) => {
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
            if let Some(ref v) = p.json_schema {
                map.insert("json_schema".into(), serde_json::json!(v));
            }
            if let Some(ref v) = p.regex {
                map.insert("regex".into(), serde_json::json!(v));
            }
            serde_json::Value::Object(map)
        }
        None => serde_json::Value::Object(serde_json::Map::new()),
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
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("text".into(), serde_json::json!(req.text));
    d.insert(
        "sampling_params".into(),
        sampling_params_to_map(&req.sampling_params),
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
    insert_disaggregated_params(&mut d, &req.disaggregated_params);
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    d
}

/// Build a request dict for GenerateReqInput from proto GenerateRequest (tokenized).
pub(crate) fn build_generate_dict(
    rid: &str,
    req: &proto::GenerateRequest,
) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("rid".into(), serde_json::json!(rid));
    d.insert("input_ids".into(), serde_json::json!(req.input_ids));
    d.insert(
        "sampling_params".into(),
        sampling_params_to_map(&req.sampling_params),
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
    insert_disaggregated_params(&mut d, &req.disaggregated_params);
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        d.insert("external_trace_header".into(), trace);
    }
    d.insert("received_time".into(), serde_json::json!(now_timestamp()));
    d
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
            build_text_generate_dict("request-1", &text_req).get("session_id"),
            Some(&serde_json::json!("session-1"))
        );
        assert_eq!(
            build_generate_dict("request-2", &token_req).get("session_id"),
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
            build_text_generate_dict("request-1", &proto::TextGenerateRequest::default());
        let token_request = build_generate_dict("request-2", &proto::GenerateRequest::default());

        for request in [text_request, token_request] {
            assert!(!request.contains_key("bootstrap_host"));
            assert!(!request.contains_key("bootstrap_port"));
            assert!(!request.contains_key("bootstrap_room"));
        }
    }
}
