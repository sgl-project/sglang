//! Kimi-K3 (`KimiK3Detector` in `function_call/kimik3_detector.py`, markers in
//! `kimik3_format.py`), with `FunctionCallParser`'s wrapping.
//!
//! Calls are numbered in output order and kept whatever their tool name. Typed
//! argument values round-trip as Python's `json.loads` would, except integers
//! beyond 64 bits, `-0`, `NaN`/`Infinity` and lone surrogates, which serde_json
//! parses differently or rejects. `_ATTR_RE`'s `\w` is the regex crate's, which
//! differs from Python's only outside ASCII.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::kimi_k3_format::{
    CALL_CLOSE, CALL_OPEN, MESSAGE_CLOSE, RESPONSE_CLOSE, RESPONSE_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
    partial_suffix_len, strip_partial_marker_suffix, strip_response_wrappers,
};
use crate::py::python_json;
use crate::tool_call::{Parsed, ToolCallItem, ToolDetector};

const SEP: &str = "<|sep|>";
const ARGUMENT_OPEN: &str = "<|open|>argument";
const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";

/// `_ATTR_RE`.
static ATTR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(\w+)="([^"]*)""#).expect("valid attribute regex"));

#[derive(Default)]
pub(super) struct KimiK3Detector {
    buffer: String,
    /// `_sent_normal_idx`, a byte offset into `buffer`.
    sent_normal_idx: usize,
    /// `current_tool_id + 1`: calls already streamed.
    streamed_calls: usize,
}

impl KimiK3Detector {
    /// `detect_and_parse`, as `parse_non_stream` sees it once a tools channel is
    /// present: the text after the channel is dropped.
    fn detect_and_parse(&self, text: &str, open_idx: usize) -> Parsed {
        let before = strip_response_wrappers(&text[..open_idx]);
        let section = &text[open_idx + TOOLS_OPEN.len()..];
        let section = section.find(TOOLS_CLOSE).map_or(section, |c| &section[..c]);
        let calls = parse_calls(section)
            .into_iter()
            .enumerate()
            .map(|(i, (name, parameters))| ToolCallItem {
                tool_index: i as i64,
                name,
                parameters,
            });
        (before, calls.collect())
    }

    /// `parse_streaming_increment`: the section is not cut at `TOOLS_CLOSE`.
    fn parse_streaming_increment(&mut self, text: &str) -> Parsed {
        self.buffer.push_str(text);
        let Some(open_idx) = self.buffer.find(TOOLS_OPEN) else {
            return (self.emit_normal_text(None), Vec::new());
        };
        let normal = self.emit_normal_text(Some(open_idx));
        let parsed = parse_calls(&self.buffer[open_idx + TOOLS_OPEN.len()..]);
        let mut calls = Vec::new();
        for (name, parameters) in parsed.into_iter().skip(self.streamed_calls) {
            calls.push(ToolCallItem {
                tool_index: self.streamed_calls as i64,
                name,
                parameters,
            });
            self.streamed_calls += 1;
        }
        (normal, calls)
    }

    /// `finish`: an open tools channel drops what is left; otherwise the held-back
    /// tail is released, minus a cut marker.
    fn finish(&mut self) -> Parsed {
        if self.buffer.contains(TOOLS_OPEN) {
            return Default::default();
        }
        let pending = self.emit_normal_text(Some(self.buffer.len()));
        (strip_partial_marker_suffix(&pending).to_owned(), Vec::new())
    }

    /// `_emit_normal_text`: unsent text up to `limit`, or up to a possible marker
    /// prefix at the end, minus the response/message wrappers.
    fn emit_normal_text(&mut self, limit: Option<usize>) -> String {
        let limit = limit.unwrap_or_else(|| {
            let markers = [TOOLS_OPEN, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE];
            self.buffer.len() - partial_suffix_len(&self.buffer, &markers)
        });
        if limit <= self.sent_normal_idx {
            return String::new();
        }
        let mut pending = self.buffer[self.sent_normal_idx..limit].to_owned();
        for marker in [RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE] {
            pending = pending.replace(marker, "");
        }
        self.sent_normal_idx = limit;
        pending
    }
}

impl ToolDetector for KimiK3Detector {
    fn has_tool_call(&self, text: &str) -> bool {
        text.contains(TOOLS_OPEN)
    }

    /// `FunctionCallParser.parse_non_stream`: without a tools channel the text
    /// comes back untouched, wrappers included.
    fn parse_non_stream(&self, text: &str) -> Parsed {
        match text.find(TOOLS_OPEN) {
            Some(open_idx) => self.detect_and_parse(text, open_idx),
            None => (text.to_owned(), Vec::new()),
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

/// `_parse_calls`: `(name, arguments JSON)` of each complete call with a tool name.
fn parse_calls(section: &str) -> Vec<(String, String)> {
    blocks(section, CALL_OPEN, CALL_CLOSE)
        .into_iter()
        .filter_map(|(attrs, body)| decode_call(attrs, body))
        .collect()
}

/// `_decode_call`: `type="string"` values stay raw text, the rest are JSON with
/// the raw text as fallback.
fn decode_call(attrs: &str, body: &str) -> Option<(String, String)> {
    let name = attr(attrs, "tool").unwrap_or_default();
    if name.is_empty() {
        return None;
    }
    let mut arguments = Map::new();
    for (arg_attrs, raw) in blocks(body, ARGUMENT_OPEN, ARGUMENT_CLOSE) {
        let key = attr(arg_attrs, "key").unwrap_or_default();
        let value = match attr(arg_attrs, "type").as_deref().unwrap_or("string") {
            "string" => Value::String(raw.to_owned()),
            _ => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned())),
        };
        arguments.insert(key, value);
    }
    Some((name, python_json(&Value::Object(arguments), false)))
}

/// `(attrs, body)` of each `_CALL_RE` / `_ARG_RE` match: `open`, whitespace,
/// attributes up to the first `<|sep|>`, then a body up to the first `close`.
fn blocks<'a>(text: &'a str, open: &str, close: &str) -> Vec<(&'a str, &'a str)> {
    let mut found = Vec::new();
    let mut pos = 0;
    while let Some(start) = text[pos..].find(open).map(|i| pos + i + open.len()) {
        let rest = &text[start..];
        let attrs_start = rest.len() - rest.trim_start_matches(py_space).len();
        if attrs_start == 0 {
            // `\s+` needs one space; the next match starts past this marker.
            pos = start;
            continue;
        }
        // A missing `<|sep|>` or `close` fails every later start too.
        let Some(sep) = rest[attrs_start..].find(SEP).map(|i| attrs_start + i) else {
            break;
        };
        let body = &rest[sep + SEP.len()..];
        let Some(end) = body.find(close) else {
            break;
        };
        found.push((&rest[attrs_start..sep], &body[..end]));
        pos = start + sep + SEP.len() + end + close.len();
    }
    found
}

/// `_parse_attrs(attrs).get(name)`: the last `name="..."`, unescaped.
fn attr(attrs: &str, name: &str) -> Option<String> {
    ATTR_RE
        .captures_iter(attrs)
        .filter(|c| &c[1] == name)
        .last()
        .map(|c| c[2].replace("&quot;", "\"").replace("&amp;", "&"))
}

/// Python's `re` `\s` (`str.isspace`), which also matches U+001C..U+001F.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
