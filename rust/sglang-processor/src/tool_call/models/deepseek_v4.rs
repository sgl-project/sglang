//! DeepSeek-V4 (`DeepSeekV4Detector` in `function_call/deepseekv4_detector.py`):
//! SGLang's DSML detector (`deepseekv32_detector.py`), with
//! `FunctionCallParser`'s wrapping.
//!
//! Arguments JSON round-trips as Python's would, except integers beyond 64 bits,
//! `-0`, non-finite numbers, lone surrogates and nesting past 128 levels, which
//! serde_json parses differently or rejects. `SGLANG_FORWARD_UNKNOWN_TOOLS` is
//! the engine's env and is taken as unset.

use regex::Regex;
use serde_json::{Map, Value};

use crate::py::{py_rstrip, py_strip, python_json};
use crate::tool_call::{Parsed, ToolCallItem, ToolDetector};

/// The DSML tag names a `--tool-call-parser` configures.
pub(super) struct DsmlTags {
    block: &'static str,
    invoke: &'static str,
    parameter: &'static str,
}

pub(super) const DSML_TAGS: DsmlTags = DsmlTags {
    block: "tool_calls",
    invoke: "invoke",
    parameter: "parameter",
};

const DSML: &str = "｜DSML｜";
/// Python's `re` `\s`, which also matches U+001C..U+001F.
const S: &str = r"[\s\x1c-\x1f]";

pub(super) struct DsmlDetector {
    tool_names: Vec<String>,
    bot_token: String,
    eot_token: String,
    invoke_start_token: String,
    invoke_end_token: String,
    parameter_regex: Regex,
    calls_regex: Regex,
    invoke_regex: Regex,
    buffer: String,
    current_tool_id: i64,
}

impl DsmlDetector {
    pub(super) fn new(tags: DsmlTags, tool_names: Vec<String>) -> Self {
        let (block, invoke, parameter) = (
            format!("{DSML}{}", tags.block),
            format!("{DSML}{}", tags.invoke),
            format!("{DSML}{}", tags.parameter),
        );
        let regex = |pattern: String| Regex::new(&pattern).expect("valid DSML regex");
        Self {
            tool_names,
            bot_token: format!("<{block}>"),
            eot_token: format!("</{block}>"),
            invoke_start_token: format!("<{invoke}"),
            invoke_end_token: format!("</{invoke}>"),
            parameter_regex: regex(format!(
                r#"(?s)<{parameter}{S}+name="([^"]+)"{S}+string="([^"]+)"{S}*>(.*?)</{parameter}>"#
            )),
            calls_regex: regex(format!(r"(?s)<{block}>(.*?)</{block}>")),
            invoke_regex: regex(format!(
                r#"(?s)<{invoke}{S}+name="(?P<name>[^"]+)"{S}*(?:(?P<self_close>/>)|>(?P<body>.*?)(?P<end>(?:</{invoke}>|$)))"#
            )),
            buffer: String::new(),
            current_tool_id: -1,
        }
    }

    fn detect_and_parse(&self, text: &str) -> Parsed {
        let (normal, sections): (String, Vec<&str>) = match text.find(&self.bot_token) {
            Some(idx) => (
                remove_suffix(&text[..idx], "\n\n").to_owned(),
                self.calls_regex
                    .captures_iter(text)
                    .map(|c| c.get(1).unwrap().as_str())
                    .collect(),
            ),
            None => match text.find(&self.invoke_start_token) {
                None => return (text.to_owned(), Vec::new()),
                Some(start) => (self.text_before_dsml(text), vec![&text[start..]]),
            },
        };
        let mut calls = Vec::new();
        for section in sections {
            for invoke in self.invoke_regex.captures_iter(section) {
                let (name, body, complete) = unpack_invoke(&invoke);
                let Some(arguments) = complete.then(|| self.parameters(body).ok()).flatten() else {
                    continue;
                };
                calls.extend(self.parse_base_json(name, &arguments));
            }
        }
        (normal, calls)
    }

    /// `BaseFormatDetector.parse_base_json`: unknown tools are dropped and the
    /// arguments re-serialized.
    fn parse_base_json(&self, name: &str, arguments: &str) -> Option<ToolCallItem> {
        let tool_index = self.tool_names.iter().rposition(|tool| tool == name)?;
        let parameters: Value = serde_json::from_str(arguments).ok()?;
        Some(ToolCallItem {
            tool_index: tool_index as i64,
            name: name.to_owned(),
            parameters: python_json(&parameters, false),
        })
    }

    fn parse_streaming_increment(&mut self, text: &str) -> Parsed {
        self.buffer.push_str(text);
        let mut current = self.buffer.clone();
        let bar = DSML.chars().next().expect("non-empty");
        let markers = [DSML.to_owned(), format!("<{bar}"), format!("</{bar}")];
        let potentially_dsml = markers
            .iter()
            .any(|marker| current.contains(marker.as_str()));
        let trimmed = py_rstrip(&current);
        let prefixes = [
            "<".to_owned(),
            format!("<{bar}"),
            "</".to_owned(),
            format!("</{bar}"),
        ];
        let ends_with_prefix = prefixes
            .iter()
            .any(|prefix| trimmed.ends_with(prefix.as_str()));
        if !self.has_tool_call(&current) && !potentially_dsml && !ends_with_prefix {
            // Trailing whitespace may precede a DSML block; hold it back.
            if trimmed.is_empty() {
                return Default::default();
            }
            self.buffer = current[trimmed.len()..].to_owned();
            let mut normal = trimmed.to_owned();
            for token in [&self.eot_token, &self.invoke_end_token] {
                normal = normal.replace(token.as_str(), "");
            }
            return (normal, Vec::new());
        }

        let mut calls = Vec::new();
        let mut preamble = String::new();
        loop {
            let Some(invoke) = self.invoke_regex.captures(&current) else {
                break;
            };
            let (name, body, complete) = unpack_invoke(&invoke);
            if !complete {
                break;
            }
            let whole = invoke.get(0).expect("match");
            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                let call_start = current[..whole.start()]
                    .rfind(&self.bot_token)
                    .unwrap_or(whole.start());
                preamble = remove_suffix(&current[..call_start], "\n\n").to_owned();
            }
            let parameters = self.parameters(body);
            let (name, rest) = (name.to_owned(), current[whole.end()..].to_owned());
            if let Ok(parameters) = parameters {
                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id,
                    name,
                    parameters,
                });
                self.current_tool_id += 1;
            }
            self.buffer = rest;
            current = self.buffer.clone();
        }
        (preamble, calls)
    }

    fn finish(&mut self) -> Parsed {
        if self.buffer.is_empty() {
            return Default::default();
        }
        let buffered = std::mem::take(&mut self.buffer);
        if self.current_tool_id != -1 {
            return Default::default();
        }
        (self.text_before_dsml(&buffered), Vec::new())
    }

    /// Prose before the first DSML tag, minus the blank line before a call.
    fn text_before_dsml(&self, text: &str) -> String {
        let Some(mut idx) = text.find(DSML) else {
            return text.to_owned();
        };
        if text[..idx].ends_with("</") {
            idx -= 2;
        } else if text[..idx].ends_with('<') {
            idx -= 1;
        }
        remove_suffix(&text[..idx], "\n\n").to_owned()
    }

    /// `_parse_parameters_from_xml`: the call's arguments as a JSON string.
    fn parameters(&self, body: &str) -> Result<String, ()> {
        let stripped = py_strip(body);
        if stripped.starts_with('{') {
            return match serde_json::from_str::<Value>(stripped) {
                Ok(Value::Object(_)) => Ok(stripped.to_owned()),
                _ => Err(()),
            };
        }
        let mut parameters = Map::new();
        let mut leftover = String::new();
        let mut last_end = 0;
        let mut matched = false;
        for param in self.parameter_regex.captures_iter(body) {
            let whole = param.get(0).expect("match");
            leftover.push_str(&body[last_end..whole.start()]);
            last_end = whole.end();
            matched = true;
            let (name, kind, value) = (&param[1], &param[2], &param[3]);
            let value = match kind {
                "true" => Value::String(py_strip(value).to_owned()),
                _ => serde_json::from_str(py_strip(value))
                    .unwrap_or_else(|_| Value::String(py_strip(value).to_owned())),
            };
            parameters.insert(name.to_owned(), value);
        }
        leftover.push_str(&body[last_end..]);
        if leftover.contains(DSML) || (!matched && !py_strip(&leftover).is_empty()) {
            return Err(());
        }
        Ok(python_json(&Value::Object(parameters), false))
    }
}

impl ToolDetector for DsmlDetector {
    fn has_tool_call(&self, text: &str) -> bool {
        text.contains(&self.bot_token) || text.contains(&self.invoke_start_token)
    }

    /// `FunctionCallParser.parse_non_stream`.
    fn parse_non_stream(&self, text: &str) -> Parsed {
        let has_tool_call = self.has_tool_call(text);
        let (normal, calls) = self.detect_and_parse(text);
        match !calls.is_empty() || has_tool_call {
            true => (normal, calls),
            false => (text.to_owned(), Vec::new()),
        }
    }

    /// `parse_stream_chunk`, plus `parse_stream_end` when `flush`.
    fn parse_stream(&mut self, text: &str, flush: bool) -> Parsed {
        let (mut normal, mut calls) = self.parse_streaming_increment(text);
        if flush {
            let (end_normal, end_calls) = self.finish();
            normal.push_str(&end_normal);
            calls.extend(end_calls);
        }
        (normal, calls)
    }
}

/// `(name, body, complete)` of an invoke match; self-closing invokes are complete.
fn unpack_invoke<'a>(invoke: &regex::Captures<'a>) -> (&'a str, &'a str, bool) {
    let name = py_strip(invoke.name("name").expect("name group").as_str());
    if invoke.name("self_close").is_some() {
        return (name, "", true);
    }
    let body = invoke.name("body").map_or("", |m| m.as_str());
    let complete = invoke.name("end").is_some_and(|m| !m.as_str().is_empty());
    (name, body, complete)
}

fn remove_suffix<'a>(text: &'a str, suffix: &str) -> &'a str {
    text.strip_suffix(suffix).unwrap_or(text)
}
