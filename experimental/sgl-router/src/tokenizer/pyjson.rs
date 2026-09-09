// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Python-`json.dumps`-compatible serialization for prompt encoders.
//!
//! Engine-side prompt encoders build their prompts in Python, so any JSON they
//! embed (tool schemas, tool-call argument values) carries Python's byte-level
//! formatting. A router encoder that emits serde's defaults instead produces a
//! different prompt, different block hashes, and no cache-aware match.

use serde::Serialize;

/// Serialize `value` the way Python's `json.dumps(v, ensure_ascii=False)` does:
/// `", "` / `": "` separators (serde's compact form omits the spaces), raw
/// UTF-8, key order preserved (the crate's `preserve_order` feature), floats
/// through [`py_float`].
///
/// RESIDUE: an integer literal outside `i64`/`u64` range is already an `f64`
/// at parse time, so Python's exact-integer output cannot be recovered; such
/// values are routing-quality only.
pub(crate) fn py_json(value: &serde_json::Value) -> String {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
    value
        .serialize(&mut serializer)
        .expect("serializing a serde_json::Value into a Vec is infallible");
    String::from_utf8(buf).expect("serde_json emits valid UTF-8")
}

/// CPython's `repr(float)` — what `json.dumps` uses for floats.
///
/// CPython emits the shortest round-tripping digits, then lays them out
/// scientific iff `exp < -4 || exp >= 16` with a signed, two-digit-padded
/// exponent (`1e-06`, `1e+16`), else positional with a trailing `.0` when
/// there is no fraction. Rust's `{}` never switches to scientific and `{:e}`
/// always does, so neither matches alone: take the digits from `{:e}` and
/// re-lay them out. Exact only because serde_json is built with
/// `float_roundtrip`.
fn py_float(v: f64) -> String {
    // Non-finite floats cannot appear: serde_json parses them to `Null`.
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    let sci = format!("{v:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always emits an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    if !(-4..16).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        let positional = format!("{v}");
        if positional.contains('.') {
            positional
        } else {
            format!("{positional}.0")
        }
    }
}

/// serde_json formatter emitting Python `json.dumps` default separators.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn write_f64<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        value: f64,
    ) -> std::io::Result<()> {
        writer.write_all(py_float(value).as_bytes())
    }

    fn write_f32<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        value: f32,
    ) -> std::io::Result<()> {
        writer.write_all(py_float(value as f64).as_bytes())
    }

    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `py_float` reproduces CPython `repr` across the layout boundaries the
    /// two writers disagree on. This text renders into the tools block at the
    /// FRONT of the prompt, so one divergent byte shifts every block hash.
    #[test]
    fn py_float_matches_cpython_repr() {
        for (v, want) in [
            (1e-6, "1e-06"),
            (1e-5, "1e-05"),
            (2.5e-5, "2.5e-05"),
            (9.75e-5, "9.75e-05"),
            (1e-4, "0.0001"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e300, "1e+300"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (1.0, "1.0"),
            (-1.5, "-1.5"),
            (0.1, "0.1"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            // Full-precision doubles: exact only with `float_roundtrip`.
            (-923807.2472198891, "-923807.2472198891"),
            (1.602176634e-19, "1.602176634e-19"),
        ] {
            assert_eq!(py_float(v), want, "py_float({v:?})");
        }
    }

    #[test]
    fn py_json_uses_python_separators_and_preserves_order() {
        let v = json!({"b": 1, "a": [1, 2], "c": {"x": true}});
        assert_eq!(py_json(&v), r#"{"b": 1, "a": [1, 2], "c": {"x": true}}"#);
    }

    /// `ensure_ascii=False`: non-ASCII stays raw, never `\uXXXX`.
    #[test]
    fn non_ascii_is_not_escaped() {
        assert_eq!(py_json(&json!({"k": "café 中文"})), r#"{"k": "café 中文"}"#);
    }
}
