//! Request pieces both endpoints parse as Python's protocol models do.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{Map, Value};
use sglang_api_types::api::v1::{
    BoolList, BoolOrList, Int64List, Int64OrList, OptionalInt64, OptionalInt64List,
    OptionalInt64OrList, OptionalString, OptionalStringList, OptionalStringOrList, StringList,
    StringOrList, bool_or_list, int64_or_list, optional_int64_or_list, optional_string_or_list,
    string_or_list,
};

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

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum OneOrList<T> {
    One(T),
    List(Vec<T>),
}

impl From<OneOrList<String>> for StringOrList {
    fn from(value: OneOrList<String>) -> Self {
        Self {
            value: Some(match value {
                OneOrList::One(value) => string_or_list::Value::One(value),
                OneOrList::List(items) => string_or_list::Value::Many(StringList { items }),
            }),
        }
    }
}

impl From<OneOrList<bool>> for BoolOrList {
    fn from(value: OneOrList<bool>) -> Self {
        Self {
            value: Some(match value {
                OneOrList::One(value) => bool_or_list::Value::One(value),
                OneOrList::List(items) => bool_or_list::Value::Many(BoolList { items }),
            }),
        }
    }
}

impl From<OneOrList<i64>> for Int64OrList {
    fn from(value: OneOrList<i64>) -> Self {
        Self {
            value: Some(match value {
                OneOrList::One(value) => int64_or_list::Value::One(value),
                OneOrList::List(items) => int64_or_list::Value::Many(Int64List { items }),
            }),
        }
    }
}

impl<T: Into<Option<String>>> From<OneOrList<T>> for OptionalStringOrList {
    fn from(value: OneOrList<T>) -> Self {
        let wrap = |value: T| OptionalString {
            value: value.into(),
        };
        Self {
            value: Some(match value {
                OneOrList::One(value) => optional_string_or_list::Value::One(wrap(value)),
                OneOrList::List(items) => {
                    optional_string_or_list::Value::Many(OptionalStringList {
                        items: items.into_iter().map(wrap).collect(),
                    })
                }
            }),
        }
    }
}

impl<T: Into<Option<i64>>> From<OneOrList<T>> for OptionalInt64OrList {
    fn from(value: OneOrList<T>) -> Self {
        let wrap = |value: T| OptionalInt64 {
            value: value.into(),
        };
        Self {
            value: Some(match value {
                OneOrList::One(value) => optional_int64_or_list::Value::One(wrap(value)),
                OneOrList::List(items) => optional_int64_or_list::Value::Many(OptionalInt64List {
                    items: items.into_iter().map(wrap).collect(),
                }),
            }),
        }
    }
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

/// `model:adapter` selects a LoRA adapter over `lora_path`.
pub(super) fn lora_path(
    model: &str,
    lora_path: &Option<OneOrList<Option<String>>>,
) -> Option<OptionalStringOrList> {
    match model.split_once(':') {
        Some((_, adapter)) if !py_strip(adapter).is_empty() => {
            Some(OneOrList::One(py_strip(adapter).to_owned()).into())
        }
        _ => lora_path.clone().map(Into::into),
    }
}

/// `OpenAIServingBase.extract_custom_labels`; `Err` when `/generate` would
/// reject them as not `Dict[str, str]`.
pub(super) fn custom_labels(
    headers: &OpenAiHeaders<'_>,
    settings: &OpenAiSettings,
) -> Result<HashMap<String, String>, Unsupported> {
    let allowed = settings
        .tokenizer_metrics_allowed_custom_labels
        .as_deref()
        .unwrap_or_default();
    if allowed.is_empty() || settings.tokenizer_metrics_custom_labels_header.is_none() {
        return Ok(HashMap::new());
    }
    let Some(Value::Object(labels)) = headers
        .custom_labels
        .and_then(|raw| serde_json::from_str(raw).ok())
    else {
        return Ok(HashMap::new());
    };
    labels
        .into_iter()
        .filter(|(label, _)| allowed.contains(label))
        .map(|(label, value)| match value {
            Value::String(value) => Ok((label, value)),
            _ => Err(Unsupported("invalid_request")),
        })
        .collect()
}
