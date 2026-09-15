//! SGLang's OpenAI extensions share native request validation and fan-out.

use axum::http::HeaderMap;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::message::config::{DefaultSamplingParams, ServerArgs};
use crate::message::request::GenerateBody;
use crate::message::sampling::SamplingParams;
use crate::message::types::HiddenStatesMode;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub(super) struct OutputOptions {
    pub return_token_ids: bool,
    pub return_prompt_token_ids: bool,
    pub return_meta_info: bool,
    pub return_input_ids_in_sglext: bool,
    pub return_output_ids_in_sglext: bool,
    pub return_cached_tokens_details: bool,
    pub return_spec_tokens_details: bool,
    pub return_hidden_states: HiddenStatesMode,
    pub return_routed_experts: bool,
    #[serde(skip)]
    pub enable_cache_report: bool,
    #[serde(skip)]
    pub ids_framed: bool,
    #[serde(skip)]
    pub continuous_usage: bool,
}

pub(super) fn request_options(
    raw: &Value,
    headers: &HeaderMap,
    args: &ServerArgs,
    chat: bool,
) -> Result<(GenerateBody, OutputOptions), String> {
    let mut output: OutputOptions =
        serde_json::from_value(raw.clone()).map_err(|e| e.to_string())?;
    output.enable_cache_report = args.enable_cache_report;
    output.continuous_usage =
        raw.pointer("/stream_options/continuous_usage_stats") == Some(&Value::Bool(true));
    if chat {
        output.ids_framed = headers.get("x-sglext-ids-framed").is_some_and(|v| v == "1");
        output.return_input_ids_in_sglext |= args.return_input_ids;
        output.return_output_ids_in_sglext |= args.return_output_ids;
        output.return_input_ids_in_sglext |= headers
            .get("x-sglext-return-input-ids")
            .is_some_and(|v| v == "1");
        output.return_output_ids_in_sglext |= headers
            .get("x-sglext-return-output-ids")
            .is_some_and(|v| v == "1");
        if raw.get("stream") == Some(&Value::Bool(true)) {
            for (enabled, name) in [
                (output.return_prompt_token_ids, "return_prompt_token_ids"),
                (output.return_token_ids, "return_token_ids"),
                (output.return_meta_info, "return_meta_info"),
            ] {
                if enabled {
                    return Err(format!(
                        "{name} is not supported with streaming. Please set stream=false when using {name}=true."
                    ));
                }
            }
        }
        if raw.get("return_sampling_mask") == Some(&Value::Bool(true)) && !output.return_meta_info {
            return Err("return_sampling_mask requires return_meta_info=true".into());
        }
    }

    // Only fields forwarded by the Python OpenAI adapter enter the native
    // schema. Unknown OpenAI fields retain that adapter's ignored behavior.
    let mut fields = Map::new();
    for name in [
        "rid",
        "bootstrap_host",
        "bootstrap_port",
        "bootstrap_room",
        "routed_dp_rank",
        "data_parallel_rank",
        "disagg_prefill_dp_rank",
        "return_hidden_states",
        "return_routed_experts",
        "routed_experts_start_len",
        "extra_key",
        "cache_salt",
        "priority",
        "custom_logit_processor",
        "session_id",
        "session_params",
        "lora_path",
        "images_config",
    ] {
        if let Some(value) = raw.get(name) {
            fields.insert(name.into(), value.clone());
        }
    }
    if chat {
        if let Some(ids) = raw.get("input_ids").filter(|ids| !ids.is_null()) {
            serde_json::from_value::<Vec<i32>>(ids.clone())
                .map_err(|e| format!("invalid input_ids: {e}"))?;
        }
        for name in [
            "input_ids",
            "return_sampling_mask",
            "video_config",
            "max_dynamic_patch",
            "min_dynamic_patch",
            "use_audio_in_video",
        ] {
            if let Some(value) = raw.get(name) {
                fields.insert(name.into(), value.clone());
            }
        }
    }
    let mut body: GenerateBody =
        serde_json::from_value(Value::Object(fields)).map_err(|e| e.to_string())?;
    body.return_prompt_token_ids = output.return_token_ids
        || (chat && (output.return_prompt_token_ids || output.return_input_ids_in_sglext));
    body.routing_key = headers
        .get("x-smg-routing-key")
        .map(|v| v.to_str().map(str::to_owned))
        .transpose()
        .map_err(|e| e.to_string())?;
    if let Some(rank) = super::super::headers::integer(headers, "x-data-parallel-rank")? {
        body.routed_dp_rank = Some(rank);
    }
    if chat && args.enable_request_header_overrides {
        super::super::headers::apply_overrides(&mut body, headers)?;
    }
    if body
        .routed_dp_rank
        .or(body.data_parallel_rank)
        .is_some_and(|rank| rank < 0 || rank as u64 >= args.dp_size as u64)
    {
        return Err("routed_dp_rank is out of range".into());
    }
    Ok((body, output))
}

pub(super) fn apply_sampling(
    raw: &Value,
    sampling: &mut SamplingParams,
    model_defaults: Option<&DefaultSamplingParams>,
) -> Result<(), String> {
    let mut value = sampling
        .serialize_client(serde_json::value::Serializer)
        .map_err(|e| e.to_string())?;
    if let Some(defaults) = model_defaults {
        for (name, default) in [
            ("top_k", defaults.top_k.map(Value::from)),
            ("min_p", defaults.min_p.map(Value::from)),
            (
                "repetition_penalty",
                defaults.repetition_penalty.map(Value::from),
            ),
        ] {
            if let Some(default) = default {
                value[name] = default;
            }
        }
    }
    for name in [
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "top_k",
        "min_p",
        "repetition_penalty",
        "stop_token_ids",
        "stop_regex",
        "no_stop_trim",
        "ignore_eos",
        "skip_special_tokens",
        "regex",
        "ebnf",
        "json_schema",
        "custom_params",
    ] {
        if name == "json_schema" && model_defaults.is_some() {
            // Chat accepts JSON schemas through response_format only.
            continue;
        }
        if let Some(field) = raw.get(name).filter(|field| !field.is_null()) {
            value[name] = field.clone();
        }
    }
    if let Some(min_tokens) = raw.get("min_tokens") {
        value["min_new_tokens"] = min_tokens.clone();
    }
    if let Some(format) = raw.get("response_format") {
        match format.get("type").and_then(Value::as_str) {
            Some("json_object") => value["json_schema"] = r#"{"type":"object"}"#.into(),
            Some("json_schema") => {
                let schema = format
                    .pointer("/json_schema/schema")
                    .ok_or("schema is required for json_schema response format")?;
                value["json_schema"] = schema.to_string().into();
            }
            Some("structural_tag") => value["structural_tag"] = format.to_string().into(),
            _ => {}
        }
    }
    *sampling = serde_json::from_value(value).map_err(|e| e.to_string())?;
    Ok(())
}
