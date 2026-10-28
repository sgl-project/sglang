//! Order-preserving JSON with CPython-exact `json.dumps`/`json.loads`.
//!
//! Wire JSON appears in the bootstrap `/route` HTTP bodies (default
//! separators `(", ", ": ")`) and in control-plane metadata frames (compact
//! `(",", ":")`). CPython escapes with `ensure_ascii=True`, which serde_json
//! cannot reproduce, so both directions are hand-rolled here.
//!
//! Integers only: no float appears in this wire traffic; a float on decode
//! is a codec error (Python's `json.loads` would hand back a float and
//! downstream code would break — an enum without floats cannot represent
//! one, so the error surfaces here instead).

use std::fmt::Write as _;

use crate::Result;
use crate::error::Error;

/// Order-preserving JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum JVal {
    Null,
    Bool(bool),
    Int(i64),
    Str(String),
    List(Vec<JVal>),
    Obj(Vec<(String, JVal)>),
}

impl JVal {
    pub fn get(&self, key: &str) -> Option<&JVal> {
        match self {
            JVal::Obj(kvs) => kvs.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Python `int(x)` semantics for JSON ints (and bools, which are ints).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            JVal::Int(i) => Some(*i),
            JVal::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JVal::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&[JVal]> {
        match self {
            JVal::List(l) => Some(l),
            _ => None,
        }
    }
    pub fn as_obj(&self) -> Option<&[(String, JVal)]> {
        match self {
            JVal::Obj(o) => Some(o),
            _ => None,
        }
    }
}

/// Escape a string exactly like CPython's `json.dumps` with
/// `ensure_ascii=True` (Lib/json/encoder.py: ESCAPE_ASCII =
/// `([\\"]|[^\ -~])`, ESCAPE_DCT short forms; everything outside 0x20..0x7E
/// — including 0x7F — becomes a lowercase `\uXXXX`; astral chars become a
/// UTF-16 surrogate pair). `/` is NOT escaped.
fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0A}' => out.push_str("\\n"),
            '\u{0C}' => out.push_str("\\f"),
            '\u{0D}' => out.push_str("\\r"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let cp = c as u32;
                if cp <= 0xFFFF {
                    write!(out, "\\u{:04x}", cp).unwrap();
                } else {
                    let v = cp - 0x1_0000;
                    write!(
                        out,
                        "\\u{:04x}\\u{:04x}",
                        0xD800 + (v >> 10),
                        0xDC00 + (v & 0x3FF)
                    )
                    .unwrap();
                }
            }
        }
    }
    out.push('"');
}

fn dumps_sep(v: &JVal, item_sep: &str, key_sep: &str, out: &mut String) {
    match v {
        JVal::Null => out.push_str("null"),
        JVal::Bool(true) => out.push_str("true"),
        JVal::Bool(false) => out.push_str("false"),
        JVal::Int(i) => write!(out, "{}", i).unwrap(),
        JVal::Str(s) => push_json_string(out, s),
        JVal::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dumps_sep(item, item_sep, key_sep, out);
            }
            out.push(']');
        }
        JVal::Obj(kvs) => {
            out.push('{');
            for (i, (k, val)) in kvs.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                push_json_string(out, k);
                out.push_str(key_sep);
                dumps_sep(val, item_sep, key_sep, out);
            }
            out.push('}');
        }
    }
}

/// `json.dumps(v, separators=(",", ":"))` — compact zmq frames.
pub fn dumps_compact(v: &JVal) -> String {
    let mut s = String::new();
    dumps_sep(v, ",", ":", &mut s);
    s
}

/// `json.dumps(v)` — every HTTP body (default separators `(", ", ": ")`).
pub fn dumps_default(v: &JVal) -> String {
    let mut s = String::new();
    dumps_sep(v, ", ", ": ", &mut s);
    s
}

/// `json.loads` subset: order-preserving objects, duplicate keys replace the
/// value in place keeping the first position (Python dict semantics), floats
/// rejected.
pub fn loads(text: &[u8]) -> Result<JVal> {
    let s = std::str::from_utf8(text).map_err(|_| Error::Codec("json: invalid utf-8".into()))?;
    let mut p = JsonParser {
        b: s.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(Error::Codec("json: trailing data".into()));
    }
    Ok(v)
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Result<u8> {
        self.b
            .get(self.i)
            .copied()
            .ok_or_else(|| Error::Codec("json: unexpected end".into()))
    }
    fn value(&mut self) -> Result<JVal> {
        match self.peek()? {
            b'n' => self.lit("null", JVal::Null),
            b't' => self.lit("true", JVal::Bool(true)),
            b'f' => self.lit("false", JVal::Bool(false)),
            b'"' => Ok(JVal::Str(self.string()?)),
            b'[' => Ok(JVal::List(self.list()?)),
            b'{' => Ok(JVal::Obj(self.obj()?)),
            b'-' | b'0'..=b'9' => self.number(),
            c => Err(Error::Codec(format!("json: unexpected byte 0x{:02x}", c))),
        }
    }
    fn lit(&mut self, word: &str, v: JVal) -> Result<JVal> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(Error::Codec(format!("json: expected {}", word)))
        }
    }
    fn string(&mut self) -> Result<String> {
        debug_assert_eq!(self.peek()?, b'"');
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek()?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0C}'),
                        b'n' => out.push('\u{0A}'),
                        b'r' => out.push('\u{0D}'),
                        b't' => out.push('\u{09}'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                // surrogate pair
                                if self.peek()? != b'\\' {
                                    return Err(Error::Codec("json: lone surrogate".into()));
                                }
                                self.i += 1;
                                if self.peek()? != b'u' {
                                    return Err(Error::Codec("json: lone surrogate".into()));
                                }
                                self.i += 1;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(Error::Codec("json: bad low surrogate".into()));
                                }
                                0x1_0000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(Error::Codec("json: lone low surrogate".into()));
                            } else {
                                hi
                            };
                            out.push(
                                char::from_u32(cp)
                                    .ok_or_else(|| Error::Codec("json: bad codepoint".into()))?,
                            );
                        }
                        _ => return Err(Error::Codec("json: bad escape".into())),
                    }
                }
                0x00..=0x1F => return Err(Error::Codec("json: raw control char".into())),
                _ => {
                    // Consume a full UTF-8 char starting at c.
                    let width =
                        utf8_width(c).ok_or_else(|| Error::Codec("json: invalid utf-8".into()))?;
                    let start = self.i - 1;
                    let end = start + width;
                    let chunk = self
                        .b
                        .get(start..end)
                        .ok_or_else(|| Error::Codec("json: truncated utf-8".into()))?;
                    let s = std::str::from_utf8(chunk)
                        .map_err(|_| Error::Codec("json: invalid utf-8".into()))?;
                    out.push_str(s);
                    self.i = end;
                }
            }
        }
    }
    fn hex4(&mut self) -> Result<u32> {
        let mut v: u32 = 0;
        for _ in 0..4 {
            let c = self.peek()?;
            self.i += 1;
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err(Error::Codec("json: bad \\u escape".into())),
            };
            v = v * 16 + d as u32;
        }
        Ok(v)
    }
    fn list(&mut self) -> Result<Vec<JVal>> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.ws();
        if self.peek()? == b']' {
            self.i += 1;
            return Ok(items);
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(items);
                }
                _ => return Err(Error::Codec("json: expected ',' or ']'".into())),
            }
        }
    }
    fn obj(&mut self) -> Result<Vec<(String, JVal)>> {
        self.i += 1; // '{'
        let mut kvs: Vec<(String, JVal)> = Vec::new();
        self.ws();
        if self.peek()? == b'}' {
            self.i += 1;
            return Ok(kvs);
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            if self.peek()? != b':' {
                return Err(Error::Codec("json: expected ':'".into()));
            }
            self.i += 1;
            self.ws();
            let val = self.value()?;
            // Python dict: duplicate key replaces value, keeps first position.
            match kvs.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = val,
                None => kvs.push((key, val)),
            }
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(kvs);
                }
                _ => return Err(Error::Codec("json: expected ',' or '}'".into())),
            }
        }
    }
    fn number(&mut self) -> Result<JVal> {
        let start = self.i;
        if self.peek()? == b'-' {
            self.i += 1;
        }
        while let Some(c) = self.b.get(self.i).copied() {
            match c {
                b'0'..=b'9' => self.i += 1,
                // Python json.loads would accept these and return a float; the
                // wire has no floats, so this is a hard codec error.
                b'.' | b'e' | b'E' | b'+' => {
                    return Err(Error::Codec("json: floats unsupported".into()));
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).unwrap();
        if text == "-" || text.is_empty() {
            return Err(Error::Codec("json: bad number".into()));
        }
        text.parse::<i64>()
            .map(JVal::Int)
            .map_err(|_| Error::Codec(format!("json: integer out of range: {}", text)))
    }
}

fn utf8_width(b: u8) -> Option<usize> {
    match b {
        0x00..=0x7F => Some(1),
        0xC0..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF7 => Some(4),
        _ => None,
    }
}
