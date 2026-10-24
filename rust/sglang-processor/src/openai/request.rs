//! Request pieces both endpoints parse as Python's protocol models do.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{OpenAiHeaders, OpenAiSettings, Unsupported};
use crate::py::{py_strip, python_json};

pub(super) fn default_model() -> String {
    "default".into()
}
pub(super) fn one() -> i64 {
    1
}
pub(super) fn yes() -> bool {
    true
}

#[derive(Deserialize, serde::Serialize)]
#[serde(untagged)]
pub(super) enum OneOrList<T> {
    One(T),
    List(Vec<T>),
}

#[derive(Deserialize)]
pub(super) struct StreamOptions {
    include_usage: Option<bool>,
    continuous_usage_stats: Option<bool>,
}

/// `should_include_usage`: `(include_usage, continuous_usage_stats)`.
pub(super) fn usage_flags(
    options: Option<&StreamOptions>,
    settings: &OpenAiSettings,
) -> (bool, bool) {
    let default = settings.stream_response_default_include_usage;
    match options {
        Some(options) => (
            options.include_usage.unwrap_or(false) || default,
            options.continuous_usage_stats.unwrap_or(false),
        ),
        None => (default, false),
    }
}

/// Python validates `json_schema` whatever the `type`. A `structural_tag`
/// format fails to parse, so it goes to the engine.
#[derive(Deserialize)]
pub(super) struct ResponseFormat {
    #[serde(rename = "type")]
    kind: FormatKind,
    json_schema: Option<JsonSchemaFormat>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum FormatKind {
    Text,
    JsonObject,
    JsonSchema,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub(super) struct JsonSchemaFormat {
    name: String,
    description: Option<String>,
    schema: Option<Map<String, Value>>,
    strict: Option<bool>,
}

/// The `json_schema` sampling param a `response_format` sets.
/// `renderer_handles_response_format`: the renderer shows the model a
/// `json_schema` format, so `strict: false` goes unconstrained.
pub(super) fn format_json_schema(
    format: Option<&ResponseFormat>,
    renderer_handles_response_format: bool,
) -> Result<Option<String>, Unsupported> {
    let Some(format) = format else {
        return Ok(None);
    };
    Ok(match format.kind {
        FormatKind::JsonSchema => {
            let json_schema = format.json_schema.as_ref();
            let schema = json_schema
                .and_then(|json_schema| json_schema.schema.clone())
                .ok_or(Unsupported("invalid_request"))?;
            let unconstrained = renderer_handles_response_format
                && json_schema.is_some_and(|json_schema| json_schema.strict == Some(false));
            (!unconstrained).then(|| python_json(&Value::Object(schema), true))
        }
        FormatKind::JsonObject => Some(r#"{"type": "object"}"#.into()),
        FormatKind::Text => None,
    })
}

/// `logit_bias` as Pydantic's `Dict[str, float]`, once its values are checked to be numbers.
pub(super) fn float_logit_bias(bias: &Option<Map<String, Value>>) -> Option<Map<String, Value>> {
    bias.as_ref().map(|bias| {
        bias.iter()
            .map(|(k, v)| (k.clone(), json!(v.as_f64())))
            .collect()
    })
}

/// `_resolve_lora_path`: a `model:adapter` adapter wins over `lora_path`; the
/// exact served name, colons and all, has none.
pub(super) fn lora_path(
    model: &str,
    lora_path: &Option<OneOrList<Option<String>>>,
    served_model_name: Option<&str>,
) -> Value {
    let adapter = match served_model_name {
        Some(served) if model == served => None,
        Some(served) if model.starts_with(&format!("{served}:")) => {
            Some(&model[served.len() + 1..])
        }
        _ => model.split_once(':').map(|(_, adapter)| adapter),
    };
    match adapter.map(py_strip).filter(|adapter| !adapter.is_empty()) {
        Some(adapter) => json!(adapter),
        None => json!(lora_path),
    }
}

/// `OpenAIServingBase.extract_custom_labels`; `Err` when `/generate` would
/// reject them as not `Dict[str, str]`.
pub(super) fn custom_labels(
    headers: &OpenAiHeaders<'_>,
    settings: &OpenAiSettings,
) -> Result<Value, Unsupported> {
    let allowed = settings
        .tokenizer_metrics_allowed_custom_labels
        .as_deref()
        .unwrap_or_default();
    if allowed.is_empty() || settings.tokenizer_metrics_custom_labels_header.is_none() {
        return Ok(Value::Null);
    }
    let Some(Value::Object(labels)) = headers
        .custom_labels
        .and_then(|raw| serde_json::from_str(raw).ok())
    else {
        return Ok(Value::Null);
    };
    let labels: Map<_, _> = labels
        .into_iter()
        .filter(|(label, _)| allowed.contains(label))
        .collect();
    match labels.values().all(Value::is_string) {
        true => Ok(Value::Object(labels)),
        false => Err(Unsupported("invalid_request")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_path_keeps_the_served_name_whole() {
        let served = Some("s3://b:m");
        assert_eq!(lora_path("s3://b:m", &None, served), json!(null));
        assert_eq!(lora_path("s3://b:m:ad", &None, served), json!("ad"));
        assert_eq!(lora_path("base:ad", &None, served), json!("ad"));
    }
}
