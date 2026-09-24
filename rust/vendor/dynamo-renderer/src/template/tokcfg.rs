// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modified by SGLang: finite f64 formatting for Python-compatible compact tojson output;
// remaining changes are comment-only provenance or spelling metadata.

//based on: https://github.com/EricLBuehler/mistral.rs/blob/d970bb5feb863acf8e8ec90de97e18221fb959f1/mistralrs-core/src/pipeline/chat_template.rs

use std::collections::HashMap;

use chrono::{DateTime, Local};
use either::Either;
use minijinja::{Error, ErrorKind, Value, value::Kwargs};
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct AddedTokensDecoder {
    __type: Option<String>,
    pub content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: Option<bool>,
}

pub fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[derive(Debug, Deserialize)]
pub struct BeginEndUnkTok(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

/// Support older tool use patterns where the tool use template was separate from the default/chat template.
/// Modern patterns use a single template with a `tool_use` key, e.g.
///
/// ```jinja
/// {%- if tools is not none and tool_choice is not none %}
/// ```
#[derive(Debug, Deserialize)]
pub struct ChatTemplateValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, Vec<HashMap<String, String>>>,
);

/// If present, pad_token is usually a single value. Deepseek R1 and it's distill's use a map.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct PadTokenValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default)]
/// Template for chat models including bos/eos/unk as well as the chat template.
pub struct ChatTemplate {
    pub bos_token: Option<BeginEndUnkTok>,
    pub eos_token: Option<BeginEndUnkTok>,
    pub unk_token: Option<BeginEndUnkTok>,

    /// Jinja format [chat templating] for chat completion.
    ///
    /// [chat templating]: https://huggingface.co/docs/transformers/chat_templating
    pub chat_template: Option<ChatTemplateValue>,

    // future
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    added_tokens_decoder: Option<HashMap<String, AddedTokensDecoder>>,
    additional_special_tokens: Option<Vec<String>>,
    clean_up_tokenization_spaces: Option<bool>,
    device_map: Option<String>,
    legacy: Option<bool>,
    model_max_length: Option<f64>,
    pad_token: Option<PadTokenValue>,
    sp_model_kwargs: Option<HashMap<String, String>>,
    spaces_between_special_tokens: Option<bool>,
    tokenizer_class: Option<String>,
    truncation_size: Option<String>,
    use_default_system_prompt: Option<bool>,
}

impl ChatTemplate {
    pub fn eos_tok(&self) -> Option<String> {
        match self.eos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn bos_tok(&self) -> Option<String> {
        match self.bos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn unk_tok(&self) -> Option<String> {
        match self.unk_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    #[serde(with = "either::serde_untagged")]
    bos_token_id: Either<u32, Vec<u32>>,
    #[serde(with = "either::serde_untagged")]
    eos_token_id: Either<u32, Vec<u32>>,
}

/// Formatter matching Python `json.dumps` default separators (`", "` and
/// `": "`). serde_json's `CompactFormatter` writes `","`/`":"` instead, and
/// chat templates embed these strings directly into the prompt, so the
/// separator choice is model-visible.
struct PyJsonFormatter;

/// Preserve Serde's shortest-roundtrip digits and apply Python's float notation.
/// Serde keeps 1e-5 in fixed notation; Python switches below 1e-4.
fn python_float(value: f64) -> String {
    let json = serde_json::to_string(&value).expect("f64 serialization cannot fail");
    if let Some((mantissa, exponent)) = json.split_once('e') {
        let exponent: i32 = exponent.parse().expect("JSON float exponent is numeric");
        return format!("{mantissa}e{exponent:+03}");
    }
    if value != 0.0 && value.abs() < 1e-4 {
        let (sign, unsigned) = json
            .strip_prefix('-')
            .map_or(("", json.as_str()), |digits| ("-", digits));
        let fraction = unsigned
            .strip_prefix("0.")
            .expect("small fixed floats have a fractional part");
        let digits = fraction.trim_start_matches('0');
        let exponent = -((fraction.len() - digits.len() + 1) as i32);
        let (first, rest) = digits.split_at(1);
        let point = if rest.is_empty() { "" } else { "." };
        return format!("{sign}{first}{point}{rest}e{exponent:+03}");
    }
    json
}

impl serde_json::ser::Formatter for PyJsonFormatter { // codespell:ignore ser
    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(python_float(value).as_bytes())
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}

/// Mirrors HF transformers' `tojson` filter, not stock Jinja2's. Transformers
/// overrides Jinja's HTML-safe `tojson` with plain
/// `json.dumps(x, ensure_ascii=False)` in its chat-template environment, and
/// vLLM/SGLang render through that — so chat templates (and the models trained
/// on their output) expect Python separators and **no** HTML escaping
/// (`'`, `<`, `>`, `&` stay literal). serde_json leaves non-ASCII unescaped by
/// default, matching `ensure_ascii=False`.
pub fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let mut buf = Vec::new();
    let result = if let Ok(indent) = kwargs.get("indent") {
        // Python `json.dumps(indent=n)` separators are `(",", ": ")` with the
        // item separator followed by newline + indent — PrettyFormatter matches.
        let repeat = b" ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&repeat); // codespell:ignore ser
        let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut serializer)
    } else {
        let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
        value.serialize(&mut serializer)
    };
    result.map_err(|err| {
        Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
    })?;
    String::from_utf8(buf)
        .map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
        .map(Value::from_safe_string)
}

/// Parse a JSON string into a structured value.
///
/// HuggingFace/transformers chat-template environments expose this filter, and several
/// published templates depend on it — e.g. Step-3.7-Flash's `tool_use` block applies it to
/// a tool call's `arguments`, which arrive as a JSON *string*, to iterate the decoded
/// object. Without it minijinja aborts the render with `unknown filter: fromjson`, which
/// fails every multi-turn tool-call request.
///
/// Values that are not strings pass through untouched, so templates that apply the filter
/// defensively to an already-decoded value keep rendering.
pub fn fromjson(value: Value) -> Result<Value, Error> {
    let Some(text) = value.as_str() else {
        return Ok(value);
    };
    let parsed: serde_json::Value = serde_json::from_str(text).map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot parse JSON").with_source(err)
    })?;
    Ok(Value::from_serialize(&parsed))
}

pub fn strftime_now(format_str: &str) -> Result<Value, Error> {
    let local: DateTime<Local> = Local::now();
    Ok(Value::from_safe_string(
        local.format(format_str).to_string(),
    ))
}

#[cfg(test)]
mod fromjson_tests {
    use super::*;

    #[test]
    fn parses_json_object_string() {
        let out = fromjson(Value::from(r#"{"location":"San Francisco","n":3}"#)).unwrap();
        assert_eq!(
            out.get_attr("location").unwrap().as_str(),
            Some("San Francisco")
        );
        assert_eq!(out.get_attr("n").unwrap().to_string(), "3");
    }

    #[test]
    fn parses_json_array_string() {
        let out = fromjson(Value::from(r#"[1,2,3]"#)).unwrap();
        assert_eq!(out.len(), Some(3));
    }

    #[test]
    fn passes_through_non_string() {
        // Already-decoded values must survive a defensive `| fromjson`.
        let already = Value::from_serialize(serde_json::json!({"a": 1}));
        let out = fromjson(already).unwrap();
        assert_eq!(out.get_attr("a").unwrap().to_string(), "1");
    }

    #[test]
    fn errors_on_malformed_json() {
        assert!(fromjson(Value::from("{not json")).is_err());
    }

    /// Regression: renders the shape of Step-3.7-Flash's `tool_use` block, where a tool
    /// call's `arguments` arrive as a JSON string. Before the filter existed this failed
    /// with `unknown filter: fromjson`, 500-ing every multi-turn tool-call request.
    #[test]
    fn renders_tool_use_block_with_json_string_arguments() {
        let mut env = minijinja::Environment::new();
        env.add_filter("fromjson", fromjson);
        env.add_template(
            "tool_use",
            "{% for tc in tool_calls %}{% set a = tc.function.arguments | fromjson %}\
CALL {{ tc.function.name }} loc={{ a.location }} unit={{ a.unit }}{% endfor %}",
        )
        .unwrap();
        let rendered = env
            .get_template("tool_use")
            .unwrap()
            .render(minijinja::context! { tool_calls => serde_json::json!([{
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"San Francisco\",\"unit\":\"F\"}"
                }
            }])})
            .expect("tool_use template must render once `fromjson` is registered");
        assert_eq!(rendered, "CALL get_weather loc=San Francisco unit=F");
    }
}
