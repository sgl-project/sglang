//! Python's `json.dumps` and `str.strip`, which SGLang's output text and
//! tool-call arguments go through.

use std::collections::HashMap;

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

/// About as many containers as Python 3.12's `json.loads` opens before it
/// raises `RecursionError` (its C recursion limit, less the caller's frames).
const MAX_DEPTH: usize = 10_000;

/// `json.dumps(json.loads(text), ensure_ascii=False)`, or `None` where
/// `json.loads` raises. Integers keep their digits, as Python's `int` does; a
/// duplicate key keeps its first position and last value. Lone surrogates,
/// which Python keeps, have no Rust `char` and also give `None`. Neither the
/// scan nor the dump recurses, so nesting costs heap, not stack.
pub(crate) fn python_json_reencode(text: &str) -> Option<String> {
    let mut scanner = PyJsonScanner {
        text,
        pos: 0,
        nodes: Vec::new(),
    };
    let root = scanner.document()?;
    Some(scanner.dump(root))
}

/// `json.dumps(dict(pairs), ensure_ascii=False)` over values already dumped: a
/// repeated key keeps its first position and last value.
pub(crate) fn python_dict(pairs: impl IntoIterator<Item = (String, String)>) -> String {
    let mut entries: Vec<(String, String)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for (key, value) in pairs {
        match index.get(&key) {
            Some(&i) => entries[i].1 = value,
            None => {
                index.insert(key.clone(), entries.len());
                entries.push((key, value));
            }
        }
    }
    let mut out = String::from("{");
    for (i, (key, value)) in entries.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_python_str(key, false, &mut out);
        out.push_str(": ");
        out.push_str(value);
    }
    out.push('}');
    out
}

/// A parsed JSON value; children are indices into the scanner's nodes.
enum Node {
    /// A scalar, already dumped.
    Leaf(String),
    Array(Vec<usize>),
    /// Keys in first-seen order, each with its last value, and their positions.
    Object(Vec<(String, usize)>, HashMap<String, usize>),
}

/// The scanner of Python's `json` module (`_json.c`).
struct PyJsonScanner<'a> {
    text: &'a str,
    pos: usize,
    nodes: Vec<Node>,
}

impl PyJsonScanner<'_> {
    fn rest(&self) -> &str {
        &self.text[self.pos..]
    }

    fn skip_ws(&mut self) {
        let rest = self.rest();
        self.pos += rest.len() - rest.trim_start_matches([' ', '\t', '\n', '\r']).len();
    }

    fn eat(&mut self, token: &str) -> bool {
        let found = self.rest().starts_with(token);
        if found {
            self.pos += token.len();
        }
        found
    }

    fn push(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// The whole text as one value, with the containers still open on a stack.
    fn document(&mut self) -> Option<usize> {
        // Each open container, with the key an object's next value goes under.
        let mut open: Vec<(usize, Option<String>)> = Vec::new();
        self.skip_ws();
        loop {
            let mut node = match self.rest().bytes().next()? {
                open_byte @ (b'[' | b'{') => {
                    if open.len() == MAX_DEPTH {
                        return None;
                    }
                    self.pos += 1;
                    self.skip_ws();
                    let is_object = open_byte == b'{';
                    let id = self.push(match is_object {
                        true => Node::Object(Vec::new(), HashMap::new()),
                        false => Node::Array(Vec::new()),
                    });
                    if self.eat(if is_object { "}" } else { "]" }) {
                        id
                    } else {
                        let key = if is_object { Some(self.key()?) } else { None };
                        open.push((id, key));
                        continue;
                    }
                }
                _ => {
                    let leaf = self.scalar()?;
                    self.push(Node::Leaf(leaf))
                }
            };
            // Add the value to its container, then close every container it completes.
            loop {
                let Some(frame) = open.last_mut() else {
                    self.skip_ws();
                    return (self.pos == self.text.len()).then_some(node);
                };
                let (parent, key) = (frame.0, frame.1.take());
                let is_object = match (&mut self.nodes[parent], key) {
                    (Node::Array(items), _) => {
                        items.push(node);
                        false
                    }
                    (Node::Object(entries, index), Some(key)) => {
                        match index.get(&key) {
                            Some(&i) => entries[i].1 = node,
                            None => {
                                index.insert(key.clone(), entries.len());
                                entries.push((key, node));
                            }
                        }
                        true
                    }
                    _ => unreachable!("only containers are open, objects with a key"),
                };
                self.skip_ws();
                if self.eat(",") {
                    self.skip_ws();
                    if is_object {
                        open.last_mut().expect("still open").1 = Some(self.key()?);
                    }
                    break;
                }
                if !self.eat(if is_object { "}" } else { "]" }) {
                    return None;
                }
                node = open.pop().expect("still open").0;
            }
        }
    }

    /// An object key and its `:`.
    fn key(&mut self) -> Option<String> {
        let key = self.rest().starts_with('"').then(|| self.string())??;
        self.skip_ws();
        self.eat(":").then_some(())?;
        self.skip_ws();
        Some(key)
    }

    /// A literal, string or number, dumped.
    fn scalar(&mut self) -> Option<String> {
        for token in ["null", "true", "false", "NaN", "Infinity", "-Infinity"] {
            if self.eat(token) {
                return Some(token.into());
            }
        }
        if self.rest().starts_with('"') {
            let mut out = String::new();
            write_python_str(&self.string()?, false, &mut out);
            return Some(out);
        }
        self.number()
    }

    /// `json.dumps` of the value at `root`, walking the nodes with a stack.
    fn dump(&self, root: usize) -> String {
        let mut out = String::new();
        // Each node being written, with how many children it has written.
        let mut stack = vec![(root, 0)];
        while let Some(&(id, written)) = stack.last() {
            let top = stack.len() - 1;
            let next = match &self.nodes[id] {
                Node::Leaf(text) => {
                    out.push_str(text);
                    None
                }
                Node::Array(items) => {
                    if written == 0 {
                        out.push('[');
                    }
                    let child = items.get(written).copied();
                    if child.is_none() {
                        out.push(']');
                    } else if written > 0 {
                        out.push_str(", ");
                    }
                    child
                }
                Node::Object(entries, _) => {
                    if written == 0 {
                        out.push('{');
                    }
                    match entries.get(written) {
                        Some((key, child)) => {
                            if written > 0 {
                                out.push_str(", ");
                            }
                            write_python_str(key, false, &mut out);
                            out.push_str(": ");
                            Some(*child)
                        }
                        None => {
                            out.push('}');
                            None
                        }
                    }
                }
            };
            match next {
                Some(child) => {
                    stack[top].1 += 1;
                    stack.push((child, 0));
                }
                None => {
                    stack.pop();
                }
            }
        }
        out
    }

    /// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?`: an `int` without a
    /// fraction or exponent, else a `float`.
    fn number(&mut self) -> Option<String> {
        let bytes = self.rest().as_bytes();
        let digits = |from: usize| {
            bytes
                .get(from..)
                .map_or(0, |b| b.iter().take_while(|b| b.is_ascii_digit()).count())
        };
        let mut end = usize::from(bytes.first() == Some(&b'-'));
        end += match bytes.get(end) {
            Some(b'0') => 1,
            Some(b'1'..=b'9') => digits(end),
            _ => return None,
        };
        let int_end = end;
        if bytes.get(end) == Some(&b'.') && digits(end + 1) > 0 {
            end += 1 + digits(end + 1);
        }
        if matches!(bytes.get(end), Some(b'e' | b'E')) {
            let sign = usize::from(matches!(bytes.get(end + 1), Some(b'+' | b'-')));
            let exponent = digits(end + 1 + sign);
            if exponent > 0 {
                end += 1 + sign + exponent;
            }
        }
        let literal = &self.text[self.pos..self.pos + end];
        self.pos += end;
        match end == int_end {
            // `int("-0")` is 0.
            true if literal == "-0" => Some("0".into()),
            true => Some(literal.into()),
            false => Some(python_float(literal.parse().ok()?)),
        }
    }

    /// A string literal, as `scanstring` decodes it: raw control characters
    /// are rejected and `\u` escapes pair surrogates.
    fn string(&mut self) -> Option<String> {
        let mut chars = self.rest()[1..].char_indices();
        let mut out = String::new();
        while let Some((i, c)) = chars.next() {
            match c {
                '"' => {
                    self.pos += i + 2;
                    return Some(out);
                }
                '\\' => out.push(match chars.next()?.1 {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    'u' => {
                        let unit = hex4(&mut chars)?;
                        let code = match unit {
                            0xd800..=0xdbff => {
                                let escape = [chars.next()?.1, chars.next()?.1];
                                let low =
                                    hex4(&mut chars).filter(|u| (0xdc00..=0xdfff).contains(u));
                                match (escape, low) {
                                    (['\\', 'u'], Some(low)) => {
                                        0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00)
                                    }
                                    _ => return None,
                                }
                            }
                            unit => unit,
                        };
                        char::from_u32(code)?
                    }
                    _ => return None,
                }),
                c if c < ' ' => return None,
                c => out.push(c),
            }
        }
        None
    }
}

/// Four hex digits of a `\u` escape.
fn hex4(chars: &mut std::str::CharIndices<'_>) -> Option<u32> {
    let digits: String = (0..4)
        .map(|_| chars.next().map(|(_, c)| c).filter(char::is_ascii_hexdigit))
        .collect::<Option<_>>()?;
    u32::from_str_radix(&digits, 16).ok()
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

    /// Nesting costs heap, not stack: past `MAX_DEPTH`, where Python raises
    /// `RecursionError`, the text is not JSON, even on a small stack.
    #[test]
    fn deep_nesting_is_bounded() {
        let nested = |depth: usize| "[".repeat(depth) + &"]".repeat(depth);
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(move || {
                assert_eq!(
                    python_json_reencode(&nested(MAX_DEPTH)),
                    Some(nested(MAX_DEPTH))
                );
                assert_eq!(python_json_reencode(&nested(MAX_DEPTH + 1)), None);
                assert_eq!(python_json_reencode(&nested(1 << 20)), None);
            })
            .unwrap()
            .join()
            .unwrap();
    }

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
