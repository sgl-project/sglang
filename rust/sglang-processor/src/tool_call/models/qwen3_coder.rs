//! Qwen3-Coder through Qwen3.8 (`Qwen3CoderDetector` in
//! `function_call/qwen3_coder_detector.py`), with `FunctionCallParser`'s wrapping.
//!
//! Calls are kept whatever their tool name. Argument values are typed by the
//! request's tool schemas, as `_convert_param_value` types them. Values that
//! fall back to `ast.literal_eval` are read for its JSON-like literals
//! (strings, numbers, `True`, `False`, `None`, lists, tuples and dicts); other
//! literals stay strings, including those `json.dumps` cannot serialize (sets,
//! bytes, complex numbers), on which Python fails the whole call.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::py::{
    MAX_INT_DIGITS, py_strip, python_dict, python_float, python_json, python_json_reencode,
};
use crate::tool_call::{Parsed, ToolCallItem, ToolDetector};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_PREFIX: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_PREFIX: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";
const TAGS: [&str; 6] = [
    TOOL_CALL_START,
    TOOL_CALL_END,
    FUNCTION_PREFIX,
    FUNCTION_END,
    PARAMETER_PREFIX,
    PARAMETER_END,
];

/// `tool_call_regex`.
static TOOL_CALL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").expect("valid regex"));
/// `tool_call_function_regex`.
static FUNCTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<function=(.*?)</function>|<function=(.*)$").expect("valid regex")
});

pub(super) struct Qwen3CoderDetector {
    tools: Vec<Value>,
    buffer: String,
    current_tool_id: i64,
    current_tool_param_count: usize,
    json_started: bool,
    is_inside_tool_call: bool,
    current_func_name: Option<String>,
}

impl Qwen3CoderDetector {
    pub(super) fn new(tools: Vec<Value>) -> Self {
        Self {
            tools,
            buffer: String::new(),
            current_tool_id: -1,
            current_tool_param_count: 0,
            json_started: false,
            is_inside_tool_call: false,
            current_func_name: None,
        }
    }

    /// `detect_and_parse`, once the text holds a `<tool_call>`.
    fn detect_and_parse(&self, text: &str) -> Parsed {
        let mut raw_tool_calls: Vec<&str> = TOOL_CALL_RE
            .captures_iter(text)
            .map(|c| c.get(1).map_or("", |m| m.as_str()))
            .collect();
        if raw_tool_calls.is_empty() && text.contains(FUNCTION_PREFIX) {
            raw_tool_calls.push(text);
        }
        let mut calls = Vec::new();
        for tool_content in raw_tool_calls {
            for function in FUNCTION_RE.captures_iter(tool_content) {
                let body = function
                    .get(1)
                    .or(function.get(2))
                    .map_or("", |m| m.as_str());
                let Some((name, params)) = body.split_once('>') else {
                    continue;
                };
                let config = arguments_config(&self.tools, Some(name));
                let pairs = parameter_matches(params)
                    .into_iter()
                    .filter_map(|param| param.split_once('>'))
                    .map(|(param_name, value)| {
                        let value = convert_param_value(strip_newline(value), param_name, &config);
                        (param_name.to_owned(), value)
                    });
                calls.push(ToolCallItem {
                    tool_index: calls.len() as i64,
                    name: name.to_owned(),
                    parameters: python_dict(pairs),
                });
            }
        }
        let start = text.find(TOOL_CALL_START).unwrap_or(0);
        (text[..start].to_owned(), calls)
    }

    /// `parse_streaming_increment`.
    fn parse_streaming_increment(&mut self, new_text: &str) -> Parsed {
        self.buffer.push_str(new_text);
        let mut calls = Vec::new();
        let mut normal = String::new();
        let mut pos = 0;
        let item = |tool_index, name: &str, parameters: String| ToolCallItem {
            tool_index,
            name: name.to_owned(),
            parameters,
        };
        while pos < self.buffer.len() {
            let slice = &self.buffer[pos..];
            if slice.starts_with(TOOL_CALL_START) {
                pos += TOOL_CALL_START.len();
                self.is_inside_tool_call = true;
                continue;
            }
            if slice.starts_with(FUNCTION_PREFIX) {
                let Some(end_angle) = slice.find('>') else {
                    break;
                };
                let name = &slice[FUNCTION_PREFIX.len()..end_angle];
                self.current_tool_id += 1;
                self.current_tool_param_count = 0;
                self.json_started = false;
                self.current_func_name = Some(name.to_owned());
                calls.push(item(self.current_tool_id, name, String::new()));
                pos += end_angle + 1;
                continue;
            }
            if slice.starts_with(PARAMETER_PREFIX) {
                let Some(name_end) = slice.find('>') else {
                    break;
                };
                let rest = &slice[name_end + 1..];
                // The parameter ends at the first of these; only `</parameter>` is consumed.
                let end = [
                    (rest.find(PARAMETER_END), PARAMETER_END.len()),
                    (rest.find(PARAMETER_PREFIX), 0),
                    (rest.find(FUNCTION_END), 0),
                ]
                .into_iter()
                .filter_map(|(found, len)| Some((found?, len)))
                .reduce(|best, next| if next.0 < best.0 { next } else { best });
                let Some((end_pos, end_len)) = end else {
                    break;
                };
                let param_name = &slice[PARAMETER_PREFIX.len()..name_end];
                if !self.json_started {
                    calls.push(item(self.current_tool_id, "", "{".into()));
                    self.json_started = true;
                }
                let config = arguments_config(&self.tools, self.current_func_name.as_deref());
                let value =
                    convert_param_value(strip_newline(&rest[..end_pos]), param_name, &config);
                // `json.dumps` of the name escapes non-ASCII; of the value, it does not.
                let key = python_json(&Value::String(param_name.to_owned()), true);
                let separator = if self.current_tool_param_count > 0 {
                    ", "
                } else {
                    ""
                };
                calls.push(item(
                    self.current_tool_id,
                    "",
                    format!("{separator}{key}: {value}"),
                ));
                self.current_tool_param_count += 1;
                pos += name_end + 1 + end_pos + end_len;
                continue;
            }
            if slice.starts_with(FUNCTION_END) {
                if !self.json_started {
                    calls.push(item(self.current_tool_id, "", "{".into()));
                    self.json_started = true;
                }
                calls.push(item(self.current_tool_id, "", "}".into()));
                pos += FUNCTION_END.len();
                self.current_func_name = None;
                continue;
            }
            if slice.starts_with(TOOL_CALL_END) {
                pos += TOOL_CALL_END.len();
                self.is_inside_tool_call = false;
                continue;
            }
            // Text up to the next `<`, or a `<` that cannot start a tag; inside a
            // tool call it is dropped.
            let consumed = match slice.find('<') {
                Some(0) if TAGS.iter().any(|tag| tag.starts_with(slice)) => break,
                Some(0) => 1,
                Some(next) => next,
                None => slice.len(),
            };
            if !self.is_inside_tool_call {
                normal.push_str(&slice[..consumed]);
            }
            pos += consumed;
        }
        self.buffer.drain(..pos);
        (normal, calls)
    }
}

impl ToolDetector for Qwen3CoderDetector {
    fn has_tool_call(&self, text: &str) -> bool {
        text.contains(TOOL_CALL_START)
    }

    fn parse_non_stream(&self, text: &str) -> Parsed {
        match self.has_tool_call(text) {
            true => self.detect_and_parse(text),
            false => (text.to_owned(), Vec::new()),
        }
    }

    /// The base `finish` flushes nothing.
    fn parse_stream(&mut self, text: &str, _flush: bool) -> Parsed {
        self.parse_streaming_increment(text)
    }
}

/// `tool_call_parameter_regex`'s matches:
/// `<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)`.
fn parameter_matches(text: &str) -> Vec<&str> {
    // Python's `$` also matches before a final newline.
    let dollar = text.strip_suffix('\n').unwrap_or(text).len();
    let mut matches = Vec::new();
    let mut from = 0;
    while let Some(found) = text[from..].find(PARAMETER_PREFIX) {
        let start = from + found + PARAMETER_PREFIX.len();
        let rest = &text[start..];
        let (end, consumed) = [
            (rest.find(PARAMETER_END), PARAMETER_END.len()),
            (rest.find(PARAMETER_PREFIX), 0),
            (rest.find(FUNCTION_END), 0),
            (dollar.checked_sub(start).or(Some(rest.len())), 0),
        ]
        .into_iter()
        .filter_map(|(found, len)| Some((found?, len)))
        .reduce(|best, next| if next.0 < best.0 { next } else { best })
        .expect("`$` always matches");
        matches.push(&rest[..end]);
        from = start + end + consumed;
    }
    matches
}

/// One leading and one trailing newline are dropped from a value.
fn strip_newline(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

/// `_get_arguments_config`: the named tool's parameter schemas, or its whole
/// `parameters` schema when that declares none.
fn arguments_config(tools: &[Value], func_name: Option<&str>) -> Map<String, Value> {
    let tool = tools.iter().find(|tool| {
        tool.get("type").is_none_or(|kind| kind == "function")
            && func_name.is_some_and(|name| tool["function"]["name"] == name)
    });
    let Some(Value::Object(params)) = tool.and_then(|tool| tool["function"].get("parameters"))
    else {
        return Map::new();
    };
    let properties = schema_properties(params);
    match !properties.is_empty() || params.contains_key("properties") {
        true => properties,
        false => params.clone(),
    }
}

/// `get_schema_properties`: top-level `properties`, else those of the
/// `anyOf`/`oneOf`/`allOf` branches, the first branch winning.
fn schema_properties(schema: &Map<String, Value>) -> Map<String, Value> {
    if let Some(Value::Object(properties)) = schema.get("properties") {
        return properties.clone();
    }
    let mut merged = Map::new();
    for keyword in ["anyOf", "oneOf", "allOf"] {
        let branches = schema.get(keyword).and_then(Value::as_array);
        for branch in branches.into_iter().flatten().filter_map(Value::as_object) {
            for (key, value) in schema_properties(branch) {
                merged.entry(key).or_insert(value);
            }
        }
    }
    merged
}

/// `infer_type_from_json_schema`.
fn infer_type(schema: &Value) -> Option<String> {
    let schema = schema.as_object()?;
    let name = |kind: &Value| {
        kind.as_str()
            .map_or_else(|| kind.to_string(), str::to_owned)
    };
    match schema.get("type") {
        Some(Value::String(kind)) => return Some(kind.clone()),
        Some(Value::Array(kinds)) if !kinds.is_empty() => {
            return Some(
                kinds
                    .iter()
                    .find(|k| *k != "null")
                    .map_or("string".into(), name),
            );
        }
        _ => {}
    }
    // `schema.get("anyOf") or schema.get("oneOf")`.
    if schema.contains_key("anyOf") || schema.contains_key("oneOf") {
        let branches = match schema.get("anyOf") {
            Some(any_of) if py_truthy(any_of) => Some(any_of),
            _ => schema.get("oneOf"),
        };
        let types: Vec<String> = branches
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(infer_type)
            .collect();
        if let Some(first) = types.first() {
            let mut distinct = types.clone();
            distinct.sort();
            distinct.dedup();
            let has = |kind: &str| types.iter().any(|t| t == kind);
            return Some(match distinct.len() {
                1 => first.clone(),
                2 if has("null") => distinct
                    .into_iter()
                    .find(|t| t != "null")
                    .expect("two types"),
                _ if has("string") => "string".into(),
                _ => first.clone(),
            });
        }
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        let mut kinds: Vec<&str> = values
            .iter()
            .map(|value| match value {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_f64() => "number",
                Value::Number(_) => "integer",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            })
            .collect();
        kinds.sort_unstable();
        kinds.dedup();
        return Some(match kinds[..] {
            [kind] => kind.into(),
            _ => "string".into(),
        });
    }
    if let Some(Value::Array(branches)) = schema.get("allOf") {
        let inferred = branches
            .iter()
            .filter_map(infer_type)
            .find(|t| t != "string");
        return Some(inferred.unwrap_or_else(|| "string".into()));
    }
    if schema.contains_key("properties") {
        return Some("object".into());
    }
    schema.contains_key("items").then(|| "array".into())
}

fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}

/// `_convert_param_value`, as `json.dumps(value, ensure_ascii=False)`.
fn convert_param_value(value: &str, name: &str, config: &Map<String, Value>) -> String {
    if value.to_lowercase() == "null" {
        return "null".into();
    }
    let string = || python_json(&Value::String(value.to_owned()), false);
    let Some(schema) = config.get(name) else {
        return string();
    };
    let kind = infer_type(schema).map_or("string".into(), |t| py_strip(&t).to_lowercase());
    let starts = |prefixes: &[&str]| prefixes.iter().any(|p| kind.starts_with(p));
    match kind.as_str() {
        "string" | "str" | "text" | "varchar" | "char" | "enum" => string(),
        _ if starts(&["int", "uint", "long", "short", "unsigned"]) => {
            py_int(value).unwrap_or_else(string)
        }
        _ if starts(&["num", "float"]) => {
            let maybe_convert = !value.contains('.') && !value.to_lowercase().contains('e');
            match py_float(value) {
                Some(f) if maybe_convert && f.is_finite() && f.fract() == 0.0 => float_int(f),
                Some(f) => python_float(f),
                None => string(),
            }
        }
        "boolean" | "bool" | "binary" => (value.to_lowercase() == "true").to_string(),
        _ => {
            let structured =
                matches!(kind.as_str(), "object" | "array" | "arr") || starts(&["dict", "list"]);
            structured
                .then(|| python_json_reencode(value))
                .flatten()
                .or_else(|| literal_eval(value))
                .unwrap_or_else(string)
        }
    }
}

/// The first code point of each run of ten Unicode decimal digits (Python 3.12,
/// Unicode 15.0), which `int()` and `float()` read as 0 through 9.
const DECIMAL_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

/// `_PyUnicode_TransformDecimalAndSpaceToASCII`, as `int()` and `float()` read
/// their argument: decimal digits become ASCII, other non-ASCII text is invalid.
fn ascii_decimals(text: &str) -> Option<String> {
    text.chars()
        .map(|c| match c {
            c if c.is_ascii() => Some(c),
            c => DECIMAL_ZEROS
                .iter()
                .find(|&&zero| (zero..zero + 10).contains(&(c as u32)))
                .map(|&zero| char::from(b'0' + (c as u32 - zero) as u8)),
        })
        .collect()
}

/// Digits separated by single underscores, as Python's `int()` and `float()` take them.
fn digit_part(text: &str) -> bool {
    !text.is_empty()
        && !text.starts_with('_')
        && !text.ends_with('_')
        && !text.contains("__")
        && text.bytes().all(|b| b.is_ascii_digit() || b == b'_')
}

/// Python's `int(text)` in base 10, as its decimal digits.
fn py_int(text: &str) -> Option<String> {
    let text = ascii_decimals(py_strip(text))?;
    let text = text.as_str();
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    if !digit_part(digits) {
        return None;
    }
    let digits = digits.replace('_', "");
    if digits.len() > MAX_INT_DIGITS {
        return None;
    }
    let digits = digits.trim_start_matches('0');
    Some(match (digits.is_empty(), negative) {
        (true, _) => "0".into(),
        (false, true) => format!("-{digits}"),
        (false, false) => digits.into(),
    })
}

/// Python's `float(text)`.
fn py_float(text: &str) -> Option<f64> {
    let text = ascii_decimals(py_strip(text))?;
    let text = text.as_str();
    let unsigned = text.strip_prefix(['+', '-']).unwrap_or(text);
    let special = ["inf", "infinity", "nan"].contains(&unsigned.to_lowercase().as_str());
    if !special {
        let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
            Some(i) => (&unsigned[..i], Some(&unsigned[i + 1..])),
            None => (unsigned, None),
        };
        let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let valid = match (int.is_empty(), frac.is_empty()) {
            (true, true) => false,
            (true, false) => digit_part(frac),
            (false, true) => digit_part(int),
            (false, false) => digit_part(int) && digit_part(frac),
        } && exponent
            .is_none_or(|e| digit_part(e.strip_prefix(['+', '-']).unwrap_or(e)));
        if !valid {
            return None;
        }
    }
    text.replace('_', "").parse().ok()
}

/// `int(f)` of an integral float, exactly.
fn float_int(f: f64) -> String {
    if f.abs() < 9.0e15 {
        return format!("{}", f as i64);
    }
    // f = mantissa * 2^exponent, with exponent > 0 here.
    let bits = f.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as usize - 1075;
    let mantissa = (bits & ((1 << 52) - 1)) | (1 << 52);
    let binary = format!("{mantissa:b}{}", "0".repeat(exponent));
    let sign = if f < 0.0 { "-" } else { "" };
    format!("{sign}{}", to_decimal(&binary, 2).expect("binary digits"))
}

/// The decimal digits of the non-negative integer `digits` in `radix`, at any
/// size, or `None` past [`MAX_INT_DIGITS`], which `json.dumps` cannot print.
fn to_decimal(digits: &str, radix: u32) -> Option<String> {
    const BASE: u64 = 1_000_000_000;
    let digits = digits.trim_start_matches('0');
    // 10^4300 is about 2^14284.5: bound the work before it.
    if digits.len().saturating_sub(1) as f64 * f64::from(radix).log2() > 14_300.0 {
        return None;
    }
    // Little-endian base-1e9 limbs.
    let mut limbs = vec![0u64];
    for c in digits.chars() {
        let mut carry = u64::from(c.to_digit(radix)?);
        for limb in &mut limbs {
            let value = *limb * u64::from(radix) + carry;
            *limb = value % BASE;
            carry = value / BASE;
        }
        if carry > 0 {
            limbs.push(carry);
        }
    }
    let mut out = limbs.last().expect("a limb").to_string();
    for limb in limbs.iter().rev().skip(1) {
        out.push_str(&format!("{limb:09}"));
    }
    (out.len() <= MAX_INT_DIGITS).then_some(out)
}

/// A Python literal `json.dumps` can serialize.
enum Literal {
    None,
    Bool(bool),
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<Literal>),
    Dict(Vec<(Literal, Literal)>),
}

/// Python's parser rejects more nested brackets.
const MAX_NESTING: usize = 200;

/// `json.dumps(ast.literal_eval(text), ensure_ascii=False)` for the literals
/// [`Literal`] holds, else `None`.
fn literal_eval(text: &str) -> Option<String> {
    // Python's tokenizer reads `\r\n` and `\r` as `\n`.
    let source = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut parser = LiteralParser {
        text: source.trim_start_matches([' ', '\t']),
        pos: 0,
        depth: 0,
    };
    parser.skip_blank_lines()?;
    let mut items = vec![parser.value()?];
    let mut tuple = false;
    parser.skip_space();
    while parser.eat(',') {
        tuple = true;
        parser.skip_space();
        if parser.at_end_of_line() {
            break;
        }
        items.push(parser.value()?);
        parser.skip_space();
    }
    parser.end()?;
    let value = match tuple {
        true => Literal::List(items),
        false => items.pop().expect("one value"),
    };
    let mut out = String::new();
    dump_literal(&value, &mut out);
    Some(out)
}

struct LiteralParser<'a> {
    text: &'a str,
    pos: usize,
    depth: usize,
}

impl<'a> LiteralParser<'a> {
    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn eat(&mut self, c: char) -> bool {
        let found = self.peek() == Some(c);
        if found {
            self.pos += c.len_utf8();
        }
        found
    }

    /// Spaces and comments, and inside brackets also newlines and line continuations.
    fn skip_space(&mut self) {
        loop {
            let rest = self.rest();
            if rest.starts_with([' ', '\t', '\x0c']) {
                self.pos += 1;
            } else if rest.starts_with('#') {
                self.pos += rest.find('\n').unwrap_or(rest.len());
            } else if self.depth > 0 && rest.starts_with('\n') {
                self.pos += 1;
            } else if self.depth > 0 && rest.starts_with("\\\n") {
                self.pos += 2;
            } else {
                return;
            }
        }
    }

    /// Blank and comment-only lines before the literal, whose line must not be indented.
    fn skip_blank_lines(&mut self) -> Option<()> {
        let mut skipped = false;
        while let Some((line, _)) = self.rest().split_once('\n')
            && blank_line(line)
        {
            self.pos += line.len() + 1;
            skipped = true;
        }
        (!skipped || !self.rest().starts_with([' ', '\t'])).then_some(())
    }

    fn at_end_of_line(&self) -> bool {
        matches!(self.peek(), None | Some('\n' | '#'))
    }

    /// Only blank and comment-only lines may follow the literal's line; a last
    /// line of bare spaces, with no newline, is an `IndentationError`.
    fn end(&mut self) -> Option<()> {
        self.skip_space();
        let Some(rest) = self.rest().strip_prefix('\n') else {
            return self.rest().is_empty().then_some(());
        };
        let mut lines = rest.split('\n');
        let last = lines.next_back().expect("split yields a line");
        let bare_spaces = !last.contains('#') && last.contains([' ', '\t']);
        (lines.all(blank_line) && blank_line(last) && !bare_spaces).then_some(())
    }

    fn value(&mut self) -> Option<Literal> {
        self.skip_space();
        let c = self.peek()?;
        match c {
            '[' | '(' | '{' => {
                self.depth += 1;
                if self.depth > MAX_NESTING {
                    return None;
                }
                self.pos += 1;
                let value = match c {
                    '[' => self.sequence(']').map(|(items, _)| Literal::List(items)),
                    '(' => self.sequence(')').map(|(mut items, comma)| match comma {
                        false if items.len() == 1 => items.pop().expect("one item"),
                        _ => Literal::List(items),
                    }),
                    _ => self.dict(),
                };
                self.depth -= 1;
                value
            }
            '-' | '+' => {
                self.pos += 1;
                match (c, self.grouped_number()?) {
                    ('+', number) => Some(number),
                    (_, Literal::Float(f)) => Some(Literal::Float(-f)),
                    (_, Literal::Int(digits)) if digits == "0" => Some(Literal::Int(digits)),
                    (_, Literal::Int(digits)) => Some(Literal::Int(format!("-{digits}"))),
                    _ => None,
                }
            }
            '0'..='9' | '.' => self.number(),
            '\'' | '"' => self.strings(),
            _ => {
                let word: String = self
                    .rest()
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if matches!(word.as_str(), "r" | "R" | "u" | "U")
                    && self.rest()[word.len()..].starts_with(['\'', '"'])
                {
                    return self.strings();
                }
                self.pos += word.len();
                match word.as_str() {
                    "True" => Some(Literal::Bool(true)),
                    "False" => Some(Literal::Bool(false)),
                    "None" => Some(Literal::None),
                    _ => None,
                }
            }
        }
    }

    /// Comma-separated values up to `close`; also whether a comma was seen.
    fn sequence(&mut self, close: char) -> Option<(Vec<Literal>, bool)> {
        let mut items = Vec::new();
        let mut comma = false;
        loop {
            self.skip_space();
            if self.eat(close) {
                return Some((items, comma));
            }
            items.push(self.value()?);
            self.skip_space();
            if self.eat(',') {
                comma = true;
            } else {
                self.skip_space();
                return self.eat(close).then_some((items, comma));
            }
        }
    }

    /// A dict after its `{`; a set fails.
    fn dict(&mut self) -> Option<Literal> {
        let mut entries: Vec<(Literal, Literal)> = Vec::new();
        loop {
            self.skip_space();
            if self.eat('}') {
                return Some(Literal::Dict(entries));
            }
            let key = self.value()?;
            if matches!(key, Literal::List(_) | Literal::Dict(_)) {
                return None;
            }
            self.skip_space();
            if !self.eat(':') {
                return None;
            }
            let value = self.value()?;
            // `dict(...)` keeps a key's first spelling and last value.
            match entries.iter_mut().find(|(k, _)| same_key(k, &key)) {
                Some(entry) => entry.1 = value,
                None => entries.push((key, value)),
            }
            self.skip_space();
            if !self.eat(',') {
                self.skip_space();
                return self.eat('}').then_some(Literal::Dict(entries));
            }
        }
    }

    /// A number literal, in any parentheses: a sign takes no other operand.
    fn grouped_number(&mut self) -> Option<Literal> {
        self.skip_space();
        if !self.eat('(') {
            return self.number();
        }
        self.depth += 1;
        let number = (self.depth <= MAX_NESTING)
            .then(|| self.grouped_number())
            .flatten();
        self.skip_space();
        let closed = self.eat(')');
        self.depth -= 1;
        number.filter(|_| closed)
    }

    /// An integer or float literal; imaginary ones fail.
    fn number(&mut self) -> Option<Literal> {
        let rest = self.rest();
        let len = rest
            .char_indices()
            .find(|&(i, c)| {
                !(c.is_ascii_alphanumeric()
                    || c == '_'
                    || c == '.'
                    || (matches!(c, '+' | '-') && i > 0 && rest[..i].ends_with(['e', 'E'])))
            })
            .map_or(rest.len(), |(i, _)| i);
        let token = &rest[..len];
        self.pos += len;
        let radix = match token.get(..2).map(str::to_ascii_lowercase).as_deref() {
            Some("0x") => 16,
            Some("0o") => 8,
            Some("0b") => 2,
            _ => 10,
        };
        if radix != 10 {
            let digits = token[2..].strip_prefix('_').unwrap_or(&token[2..]);
            if digits.is_empty() || digits.ends_with('_') || digits.contains("__") {
                return None;
            }
            return to_decimal(&digits.replace('_', ""), radix).map(Literal::Int);
        }
        if token.contains(['.', 'e', 'E']) {
            // Python's float literals and `float()` take the same digit groups.
            return py_float(token).map(Literal::Float);
        }
        let digits = py_int(token).filter(|_| digit_part(token))?;
        // Leading zeros are a syntax error in a non-zero decimal literal.
        if token.starts_with('0') && digits != "0" {
            return None;
        }
        Some(Literal::Int(digits))
    }

    /// One or more adjacent string literals, concatenated.
    fn strings(&mut self) -> Option<Literal> {
        let mut out = String::new();
        loop {
            let raw = self.rest().starts_with(['r', 'R']);
            if self.rest().starts_with(['r', 'R', 'u', 'U']) {
                self.pos += 1;
            }
            self.string(raw, &mut out)?;
            self.skip_space();
            if !self.starts_string() {
                return Some(Literal::Str(out));
            }
        }
    }

    fn starts_string(&self) -> bool {
        let rest = self.rest();
        rest.starts_with(['\'', '"'])
            || (rest.starts_with(['r', 'R', 'u', 'U']) && rest[1..].starts_with(['\'', '"']))
    }

    /// One quoted string at its opening quote, appended to `out`.
    fn string(&mut self, raw: bool, out: &mut String) -> Option<()> {
        let rest = self.rest();
        let quote = &rest[..1];
        let triple = rest.starts_with(&quote.repeat(3));
        let delimiter = if triple {
            quote.repeat(3)
        } else {
            quote.to_owned()
        };
        let mut chars = rest[delimiter.len()..].char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            let body_start = delimiter.len();
            if rest[body_start + i..].starts_with(&delimiter) {
                self.pos += body_start + i + delimiter.len();
                return Some(());
            }
            match c {
                '\n' if !triple => return None,
                '\\' => {
                    let (_, escaped) = chars.next()?;
                    if raw {
                        out.push('\\');
                        out.push(escaped);
                        continue;
                    }
                    match escaped {
                        '\n' => {}
                        '\\' | '\'' | '"' => out.push(escaped),
                        'a' => out.push('\u{7}'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'v' => out.push('\u{b}'),
                        '0'..='7' => {
                            let mut value = escaped.to_digit(8).expect("octal digit");
                            for _ in 0..2 {
                                match chars.peek().and_then(|(_, c)| c.to_digit(8)) {
                                    Some(digit) => {
                                        value = value * 8 + digit;
                                        chars.next();
                                    }
                                    None => break,
                                }
                            }
                            out.push(char::from_u32(value)?);
                        }
                        'x' | 'u' | 'U' => {
                            let width = match escaped {
                                'x' => 2,
                                'u' => 4,
                                _ => 8,
                            };
                            let hex: String = (0..width)
                                .map(|_| {
                                    chars.next().map(|(_, c)| c).filter(char::is_ascii_hexdigit)
                                })
                                .collect::<Option<_>>()?;
                            // Lone surrogates have no `char`.
                            out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                        }
                        // `\N{...}` is not read.
                        'N' => return None,
                        _ => {
                            out.push('\\');
                            out.push(escaped);
                        }
                    }
                }
                _ => out.push(c),
            }
        }
        None
    }
}

/// A line Python's tokenizer skips: whitespace, then maybe a comment.
fn blank_line(line: &str) -> bool {
    line.trim_start_matches([' ', '\t', '\x0c'])
        .chars()
        .next()
        .is_none_or(|c| c == '#')
}

/// Python's `==` between two hashable keys: `1`, `1.0` and `True` are one key.
fn same_key(a: &Literal, b: &Literal) -> bool {
    let number = |key: &Literal| match key {
        Literal::Bool(b) => Some(Ok(if *b { "1".to_owned() } else { "0".to_owned() })),
        Literal::Int(digits) => Some(Ok(digits.clone())),
        Literal::Float(f) if f.is_finite() && f.fract() == 0.0 => Some(Ok(float_int(*f))),
        Literal::Float(f) => Some(Err(*f)),
        _ => None,
    };
    match (a, b) {
        (Literal::None, Literal::None) => true,
        (Literal::Str(a), Literal::Str(b)) => a == b,
        _ => match (number(a), number(b)) {
            (Some(Ok(a)), Some(Ok(b))) => a == b,
            (Some(Err(a)), Some(Err(b))) => a == b,
            _ => false,
        },
    }
}

/// `json.dumps(value, ensure_ascii=False)`.
fn dump_literal(value: &Literal, out: &mut String) {
    let string = |s: &str| python_json(&Value::String(s.to_owned()), false);
    match value {
        Literal::None => out.push_str("null"),
        Literal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Literal::Int(digits) => out.push_str(digits),
        Literal::Float(f) => out.push_str(&python_float(*f)),
        Literal::Str(s) => out.push_str(&string(s)),
        Literal::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_literal(item, out);
            }
            out.push(']');
        }
        Literal::Dict(entries) => {
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                // Keys are written as JSON strings.
                let key = match key {
                    Literal::Str(s) => s.clone(),
                    other => {
                        let mut text = String::new();
                        dump_literal(other, &mut text);
                        text
                    }
                };
                out.push_str(&string(&key));
                out.push_str(": ");
                dump_literal(item, out);
            }
            out.push('}');
        }
    }
}
