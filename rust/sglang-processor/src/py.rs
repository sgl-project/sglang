//! Python's `json.dumps` and `str.strip`, which SGLang's output text and
//! tool-call arguments go through.

use serde_json::Value;

/// Python's `json.dumps(value, ensure_ascii=ascii)`: `", "` and `": "` separators.
pub(crate) fn python_json(value: &Value, ascii: bool) -> String {
    let mut out = String::new();
    write_python_json(value, ascii, &mut out);
    out
}

fn write_python_json(value: &Value, ascii: bool, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => out.push_str(&python_float(f)),
            _ => out.push_str(&n.to_string()),
        },
        Value::String(s) => write_python_str(s, ascii, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, ascii, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_str(key, ascii, out);
                out.push_str(": ");
                write_python_json(item, ascii, out);
            }
            out.push('}');
        }
    }
}

fn write_python_str(s: &str, ascii: bool, out: &mut String) {
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
            c if !ascii && c >= ' ' => out.push(c),
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

pub(crate) fn py_strip(text: &str) -> &str {
    text.trim_matches(py_space)
}

pub(crate) fn py_rstrip(text: &str) -> &str {
    text.trim_end_matches(py_space)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
            python_json(&value, true),
            "{\"a\": [1, 2.5, 1e-05, 1e+16, 100.0, null], \"\\u00e9\": \"x\\\"\\n\\ud83d\\ude00\"}"
        );
    }
}
