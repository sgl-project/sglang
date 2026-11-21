//! Kimi-K3 (`KimiK3Detector` in `function_call/kimik3_detector.py`, markers in
//! `kimik3_format.py`), with `FunctionCallParser`'s wrapping.
//!
//! Calls are numbered in output order and kept whatever their tool name. Typed
//! argument values go through `json.loads` and `json.dumps` as in Python
//! (`python_json_reencode`), except lone surrogates. `_ATTR_RE`'s `\w` is the
//! regex crate's, which differs from Python's only outside ASCII.
//!
//! Streaming converts the XTML to JSON as it arrives (`_stream_calls`): a call's
//! name and `{` once its header is complete, a string value as it streams, a
//! typed value once its close arrives. A call still open is incomplete, which
//! keeps a stop from finishing as `tool_calls`. A typed value `json.loads`
//! fails on with an error other than `JSONDecodeError` (an integer past 4300
//! digits, nesting past its recursion limit) is kept as text here, where
//! Python's stream starts its parse over and its one-shot parse drops every
//! call.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::kimi_k3_format::{
    CALL_CLOSE, CALL_OPEN, MESSAGE_CLOSE, RESPONSE_CLOSE, RESPONSE_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
    partial_suffix_len, strip_partial_marker_suffix, strip_response_wrappers,
};
use crate::py::{python_dict, python_json, python_json_reencode};
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
    /// `_tool_pos`, a byte offset into `buffer`: where streaming resumes in the
    /// tools channel, once one opened.
    tool_pos: Option<usize>,
    /// `current_tool_id + 1`: calls opened so far.
    opened_calls: usize,
    /// `_in_call`: the last call opened has not closed, so it is in
    /// `incomplete_tool_call_indices`.
    in_call: bool,
    /// `_argument_type` of the argument whose value is streaming.
    argument_type: Option<String>,
    /// `_argument_value`: a typed value, held until its close.
    argument_value: String,
    /// `_argument_count` of the open call.
    argument_count: usize,
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

    /// `parse_streaming_increment`.
    fn parse_streaming_increment(&mut self, text: &str) -> Parsed {
        self.buffer.push_str(text);
        let Some(open_idx) = self.buffer.find(TOOLS_OPEN) else {
            return (self.emit_normal_text(None), Vec::new());
        };
        let normal = self.emit_normal_text(Some(open_idx));
        self.tool_pos.get_or_insert(open_idx + TOOLS_OPEN.len());
        (normal, self.stream_calls())
    }

    /// `_stream_calls`: the calls' JSON from `tool_pos` on, as far as it can be
    /// written yet.
    fn stream_calls(&mut self) -> Vec<ToolCallItem> {
        let mut calls = Vec::new();
        let mut pos = self.tool_pos.expect("tools channel open");
        while pos < self.buffer.len() {
            let tool_index = self.opened_calls as i64 - 1;
            // `emit` of an argument fragment of the open call.
            let mut emit = |fragment: String| {
                if !fragment.is_empty() {
                    calls.push(ToolCallItem {
                        tool_index,
                        name: String::new(),
                        parameters: fragment,
                    });
                }
            };
            let remaining = &self.buffer[pos..];
            if let Some(kind) = &self.argument_type {
                let end = remaining.find(ARGUMENT_CLOSE);
                let count = end.unwrap_or_else(|| {
                    remaining.len() - partial_suffix_len(remaining, &[ARGUMENT_CLOSE])
                });
                let value = &remaining[..count];
                pos += count;
                if kind == "string" {
                    let quoted = python_json(&Value::String(value.to_owned()), false);
                    emit(quoted[1..quoted.len() - 1].to_owned());
                } else {
                    self.argument_value.push_str(value);
                }
                if end.is_none() {
                    break;
                }
                if kind == "string" {
                    emit("\"".to_owned());
                } else {
                    let raw = std::mem::take(&mut self.argument_value);
                    let value = python_json_reencode(&raw)
                        .unwrap_or_else(|| python_json(&Value::String(raw), false));
                    emit(value);
                }
                pos += ARGUMENT_CLOSE.len();
                self.argument_type = None;
                continue;
            }

            // Whitespace between XTML elements is not part of any argument.
            let stripped = remaining.trim_start_matches(py_space);
            pos += remaining.len() - stripped.len();
            let remaining = stripped;
            if self.in_call && remaining.starts_with(CALL_CLOSE) {
                emit("}".to_owned());
                self.in_call = false;
                pos += CALL_CLOSE.len();
                continue;
            }

            let (opener, stops): (&str, &[&str]) = match self.in_call {
                true => (ARGUMENT_OPEN, &[CALL_CLOSE, TOOLS_CLOSE]),
                false => (CALL_OPEN, &[TOOLS_CLOSE]),
            };
            let Some((attrs, header_len)) = header(remaining, opener) else {
                // Skip what cannot start the next element, as the one-shot
                // parse does; wait on an incomplete header or the tools close.
                match skippable_len(remaining, opener, stops) {
                    0 => break,
                    skip => {
                        pos += skip;
                        continue;
                    }
                }
            };
            if !self.in_call {
                let name = attr(attrs, "tool").unwrap_or_default();
                if name.is_empty() {
                    // Skip the whole call, as the one-shot parse does.
                    let Some(end) = remaining[header_len..].find(CALL_CLOSE) else {
                        break;
                    };
                    pos += header_len + end + CALL_CLOSE.len();
                    continue;
                }
                self.opened_calls += 1;
                self.argument_count = 0;
                self.in_call = true;
                calls.push(ToolCallItem {
                    tool_index: self.opened_calls as i64 - 1,
                    name,
                    parameters: "{".to_owned(),
                });
            } else {
                let key = attr(attrs, "key").unwrap_or_default();
                let kind = attr(attrs, "type").unwrap_or_else(|| "string".to_owned());
                let mut prefix = match self.argument_count {
                    0 => String::new(),
                    _ => ", ".to_owned(),
                };
                prefix.push_str(&python_json(&Value::String(key), false));
                prefix.push_str(": ");
                if kind == "string" {
                    prefix.push('"');
                }
                emit(prefix);
                self.argument_type = Some(kind);
                self.argument_count += 1;
            }
            pos += header_len;
        }
        self.tool_pos = Some(pos);
        calls
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

    fn has_incomplete_call(&self) -> bool {
        self.in_call
    }
}

/// The header `re.match` finds at the start of `text`: `opener`, whitespace,
/// then attributes up to the first `<|sep|>`. Returns the attributes and the
/// header's length.
fn header<'a>(text: &'a str, opener: &str) -> Option<(&'a str, usize)> {
    let rest = text.strip_prefix(opener)?;
    let attrs = rest.trim_start_matches(py_space);
    if attrs.len() == rest.len() {
        // `\s+` needs one space.
        return None;
    }
    let sep = attrs.find(SEP)?;
    let start = text.len() - attrs.len();
    Some((&attrs[..sep], start + sep + SEP.len()))
}

/// `_skippable_len`: how much of `text` to skip before the next element can
/// start, an `opener` header or one of `stops`, as the one-shot `finditer`
/// skips stray text and a header without whitespace after its tag. 0 when
/// `text` starts with a stop or may still grow into an element.
fn skippable_len(text: &str, opener: &str, stops: &[&str]) -> usize {
    let markers: Vec<&str> = std::iter::once(opener)
        .chain(stops.iter().copied())
        .collect();
    if stops.iter().any(|stop| text.starts_with(stop))
        || markers.iter().any(|marker| marker.starts_with(text))
        || text
            .strip_prefix(opener)
            .and_then(|rest| rest.chars().next())
            .is_some_and(py_space)
    {
        return 0;
    }
    let first = text.chars().next().map_or(0, char::len_utf8);
    match markers
        .iter()
        .filter_map(|marker| text[first..].find(marker))
        .min()
    {
        Some(start) => first + start,
        // Keep a marker the next chunk may complete.
        None => text.len() - partial_suffix_len(text, &markers),
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
    let arguments =
        blocks(body, ARGUMENT_OPEN, ARGUMENT_CLOSE)
            .into_iter()
            .map(|(arg_attrs, raw)| {
                let value = match attr(arg_attrs, "type").as_deref().unwrap_or("string") {
                    "string" => None,
                    _ => python_json_reencode(raw),
                };
                let value = value.unwrap_or_else(|| python_json(&Value::String(raw.into()), false));
                (attr(arg_attrs, "key").unwrap_or_default(), value)
            });
    Some((name, python_dict(arguments)))
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
