//! Response pieces shared by the OpenAI endpoints, shaped as Python serializes them.

use serde_json::{Map, Value, json};

/// A complete HTTP response the host sends instead of a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub status: u16,
    pub body: String,
}

/// `ErrorResponse(...).model_dump()`.
pub(super) fn error_body(message: &str, err_type: &str, code: u16) -> Value {
    json!({"object": "error", "message": message, "type": err_type, "param": null, "code": code})
}

/// `OpenAIServingBase.create_error_response`.
pub(super) fn error_reply(message: &str, err_type: &str, code: u16) -> Reply {
    Reply {
        status: code,
        body: error_body(message, err_type, code).to_string(),
    }
}

/// `OpenAIServingBase.create_streaming_error_response`, as an SSE event.
pub(super) fn stream_error_event(message: &str, err_type: &str, code: u16) -> String {
    let error = json!({"error": error_body(message, err_type, code)});
    format!("data: {}\n\n", python_json(&error))
}

/// The OpenAI reply for a `/generate` error response. `None` passes the
/// engine's reply through: SGLang answers its other errors, such as an
/// `HTTPException`, in the OpenAI error shape on every route.
pub(super) fn engine_error_reply(body: &[u8]) -> Option<Reply> {
    let body: Value = serde_json::from_slice(body).ok()?;
    // A ValueError: `/generate` answers `{"error": {"message"}}`, OpenAI a 400.
    let message = body.pointer("/error/message")?.as_str()?;
    Some(error_reply(message, "BadRequestError", 400))
}

/// Python's answer when the engine output lacks what the OpenAI layer reads.
pub(super) fn malformed_output_reply(missing: &str) -> Reply {
    let message = format!("Internal server error: '{missing}'");
    error_reply(&message, "InternalServerError", 500)
}

/// `HTTPStatus(code).name` for the codes SGLang aborts with.
pub(super) fn http_status_name(code: u64) -> Option<&'static str> {
    Some(match code {
        400 => "BAD_REQUEST",
        401 => "UNAUTHORIZED",
        403 => "FORBIDDEN",
        404 => "NOT_FOUND",
        408 => "REQUEST_TIMEOUT",
        413 => "REQUEST_ENTITY_TOO_LARGE",
        422 => "UNPROCESSABLE_ENTITY",
        429 => "TOO_MANY_REQUESTS",
        500 => "INTERNAL_SERVER_ERROR",
        501 => "NOT_IMPLEMENTED",
        502 => "BAD_GATEWAY",
        503 => "SERVICE_UNAVAILABLE",
        504 => "GATEWAY_TIMEOUT",
        _ => return None,
    })
}

/// `UsageProcessor.calculate_token_usage`, dumped as `UsageInfo`.
pub(super) fn usage(prompt: u64, completion: u64, reasoning: u64, cached: Option<u64>) -> Value {
    json!({
        "prompt_tokens": prompt,
        "total_tokens": prompt + completion,
        "completion_tokens": completion,
        "prompt_tokens_details": cached.filter(|&n| n > 0).map(|n| json!({"cached_tokens": n})),
        "reasoning_tokens": reasoning,
    })
}

/// `model_dump_json(exclude_none=True)`: drop `null` fields, recursively.
pub(super) fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, without_nulls(v)))
                .collect::<Map<_, _>>(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(without_nulls).collect()),
        other => other,
    }
}

pub(super) fn meta_u64(meta: &Value, key: &str) -> u64 {
    meta.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// Python's `json.dumps(value)`: `", "` and `": "` separators, ASCII only.
pub(super) fn python_json(value: &Value) -> String {
    let mut out = String::new();
    write_python_json(value, &mut out);
    out
}

fn write_python_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => out.push_str(&python_float(f)),
            _ => out.push_str(&n.to_string()),
        },
        Value::String(s) => write_python_str(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_str(key, out);
                out.push_str(": ");
                write_python_json(item, out);
            }
            out.push('}');
        }
    }
}

fn write_python_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// Python's `repr(float)`, as `json.dumps` prints it: the shortest digits that
/// round-trip, ties to even, scientific outside 1e-4..1e16.
fn python_float(f: f64) -> String {
    if !f.is_finite() {
        return match f.is_nan() {
            true => "NaN".into(),
            false if f > 0.0 => "Infinity".into(),
            false => "-Infinity".into(),
        };
    }
    let shortest = format!("{:e}", f.abs());
    let len = shortest
        .split('e')
        .next()
        .expect("mantissa")
        .replace('.', "")
        .len();
    // Exact formatting rounds the binary value half to even, as `repr` does.
    let sci = format!("{:.*e}", len - 1, f.abs());
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let digits = mantissa.replace('.', "");
    let exp: i32 = exp.parse().expect("integer exponent");
    let sign = if f.is_sign_negative() { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let (first, rest) = digits.split_at(1);
        let mantissa = match rest.is_empty() {
            true => first.to_owned(),
            false => format!("{first}.{rest}"),
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{mantissa}e{exp_sign}{:02}", exp.abs());
    }
    let (int, frac) = match usize::try_from(exp) {
        Err(_) => (
            "0".to_owned(),
            format!("{}{digits}", "0".repeat((-exp - 1) as usize)),
        ),
        Ok(exp) if exp + 1 < digits.len() => {
            (digits[..=exp].to_owned(), digits[exp + 1..].to_owned())
        }
        Ok(exp) => (
            format!("{digits:0<width$}", width = exp + 1),
            "0".to_owned(),
        ),
    };
    format!("{sign}{int}.{frac}")
}

/// Python's `str.isspace`, which also counts the ASCII separators U+001C..U+001F.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub(super) fn py_strip(text: &str) -> &str {
    text.trim_matches(py_space)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_json_matches_json_dumps() {
        let value = json!({"a": [1, 2.5, 1e-5, 1e16, 100.0, null], "é": "x\"\n\u{1F600}"});
        #[allow(clippy::excessive_precision)]
        let tie = 77947932308658.125;
        assert_eq!(python_float(tie), "77947932308658.12");
        assert_eq!(python_float(0.5), "0.5");
        assert_eq!(python_float(-0.0), "-0.0");
        assert_eq!(python_float(0.0001), "0.0001");
        assert_eq!(python_float(123456789012345680.0), "1.2345678901234568e+17");
        assert_eq!(
            python_json(&value),
            "{\"a\": [1, 2.5, 1e-05, 1e+16, 100.0, null], \"\\u00e9\": \"x\\\"\\n\\ud83d\\ude00\"}"
        );
    }
}
