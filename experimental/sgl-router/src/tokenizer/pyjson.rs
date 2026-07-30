// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Python-`json.dumps`-compatible serialization for prompt encoders.
//!
//! Engine-side prompt encoders build their prompts in Python, so any JSON they
//! embed (tool schemas, tool-call argument values, response schemas) carries
//! Python's byte-level formatting choices. A router encoder that emits serde's
//! defaults instead produces a different prompt, different block hashes, and no
//! cache-aware match — so the two separator conventions Python encoders actually
//! use both live here:
//!
//!   * [`py_json`] — `json.dumps(v, ensure_ascii=False)`, i.e. the DEFAULT
//!     separators `", "` / `": "`.
//!   * [`compact_json`] — `json.dumps(v, ensure_ascii=False, separators=(",", ":"))`.
//!
//! `ensure_ascii=False` needs no special handling: serde already emits raw UTF-8.
//! Key order passes through unchanged (the crate's `preserve_order` feature), so
//! a caller that needs Python's `sorted()` ordering must sort first — see
//! [`deep_sort`].

use serde::Serialize;

/// Serialize `value` the way Python's `json.dumps(v, ensure_ascii=False)` does:
/// `", "` between elements/members and `": "` after each key (serde's compact
/// form omits those spaces, changing the bytes the engine's block hashes key
/// on). Key order and non-ASCII pass through unchanged (`ensure_ascii=False` +
/// the crate's `preserve_order` feature).
///
/// Floats go through [`py_float`], which reproduces CPython's `repr` (what
/// `json.dumps` uses for floats) rather than serde's default writer — the two
/// disagree on every value with `0 < |v| < 1e-4` and on exponent padding, and
/// this text lands in the tools block at the FRONT of the prompt, so a
/// divergence there shifts every block hash.
///
/// RESIDUE: an integer literal outside `i64`/`u64` range (e.g. `2**64`) is
/// already an `f64` by the time serde_json hands it over — the int/float
/// distinction is lost at PARSE time, so no formatter can recover Python's
/// exact-integer output. Integers that FIT in `i64`/`u64` are exact on both
/// sides, including past f64's 2^53 precision limit. JSON Schema keywords don't
/// carry out-of-range values, and `input_ids_safe_to_forward_dsv4` withholds
/// them, so the residue is routing-quality only.
pub(crate) fn py_json(value: &serde_json::Value) -> String {
    format_with(value, PyJsonFormatter)
}

/// Serialize `value` the way Python's
/// `json.dumps(v, ensure_ascii=False, separators=(",", ":"))` does — no spaces
/// after `,` or `:`. The separators are serde's own compact default; it exists
/// as a named counterpart to [`py_json`] so a call site states which Python
/// convention it is mirroring instead of leaving it implicit.
///
/// Formats floats through [`py_float`] and carries the same integer residue as
/// [`py_json`].
pub(crate) fn compact_json(value: &serde_json::Value) -> String {
    format_with(value, CompactPyFormatter)
}

fn format_with<F: serde_json::ser::Formatter>(value: &serde_json::Value, formatter: F) -> String {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value
        .serialize(&mut serializer)
        .expect("serializing a serde_json::Value into a Vec is infallible");
    String::from_utf8(buf).expect("serde_json emits valid UTF-8")
}

/// Recursively sort every object's keys, mirroring `encoding_k3.deep_sort_dict`.
///
/// The K3 encoder canonicalizes tool declarations and response schemas through
/// `{k: ... for k, v in sorted(obj.items())}` before rendering, so two clients
/// that send the same schema with different key order still hash to the same
/// prefix. Tool-call ARGUMENTS are deliberately not sorted — the reference
/// renders them in the order the model emitted them, and sorting here "for
/// consistency" would break parity on every tool call. Python's `sorted()` on
/// `str` orders by code point, and Rust's
/// `str: Ord` orders by UTF-8 bytes — the two agree for all inputs, because
/// UTF-8 byte order preserves code-point order.
///
/// Arrays keep their order (only their elements are sorted, per the Python).
pub(crate) fn deep_sort(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k.clone(), deep_sort(v)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(deep_sort).collect())
        }
        other => other.clone(),
    }
}

/// CPython's `repr(float)` — the function `json.dumps` uses for floats.
///
/// CPython emits the shortest round-tripping digit string, then picks a layout
/// from the decimal exponent: scientific iff `exp < -4 || exp >= 16`, with the
/// exponent signed and zero-padded to two digits (`1e-06`, `1e+16`); otherwise
/// positional, with a trailing `.0` when there is no fractional part. Rust's
/// `{}` also emits shortest-round-trip digits but never switches to scientific
/// and never pads, and `{:e}` always switches and never pads — so neither
/// matches on its own. Take the digits from `{:e}` and re-lay them out.
fn py_float(v: f64) -> String {
    // Non-finite floats cannot appear: serde_json parses them to `Null`.
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    let sci = format!("{v:e}"); // e.g. "1e-6", "2.5e-5", "-9.238072472198891e5"
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always emits an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    // CPython uses positional layout for exponents in -4..16 and scientific
    // outside it.
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

/// Emit `write_f64`/`write_f32` overrides routing through [`py_float`]. Both
/// formatters need them; only their separators differ.
macro_rules! py_float_writers {
    () => {
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
            // serde_json only ever hands us f64 for parsed JSON; mirror it anyway.
            writer.write_all(py_float(value as f64).as_bytes())
        }
    };
}

/// serde_json formatter emitting Python `json.dumps` separators
/// (`separators=(",", ":")`) — serde's own defaults — with Python float
/// formatting.
struct CompactPyFormatter;

impl serde_json::ser::Formatter for CompactPyFormatter {
    py_float_writers!();
}

/// serde_json formatter emitting Python `json.dumps` default separators
/// (`", "` / `": "`) instead of serde's compact `,` / `:`.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    py_float_writers!();

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

    /// `py_float` reproduces CPython `repr` — which `json.dumps` uses for
    /// floats — across the layout boundaries the two disagree on by default.
    /// This text renders into the tools block at the FRONT of the prompt, so a
    /// single divergent byte shifts every downstream block hash. Values below
    /// were checked against real CPython (a 4026-value fuzz over uniforms, log
    /// -uniforms, random bit patterns and the subnormal/overflow extremes found
    /// zero mismatches).
    #[test]
    fn py_float_matches_cpython_repr() {
        for (v, want) in [
            // scientific iff exp < -4 || exp >= 16, exponent signed and
            // zero-padded to two digits — serde's writer does neither.
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
            // positional keeps a trailing `.0` when there is no fraction.
            (1.0, "1.0"),
            (-1.5, "-1.5"),
            (0.1, "0.1"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            // full-precision doubles: correct only because the crate is built
            // with `float_roundtrip`; without it the PARSE lands 1 ULP off and
            // the shortest-roundtrip text differs.
            (-923807.2472198891, "-923807.2472198891"),
            (1.602176634e-19, "1.602176634e-19"),
        ] {
            assert_eq!(py_float(v), want, "py_float({v:?})");
        }
    }

    /// The two separator conventions must stay distinguishable — collapsing
    /// them is exactly the silent byte drift that breaks prefix matching.
    #[test]
    fn separators_match_their_python_counterparts() {
        let v = json!({"a": 1, "b": [1, 2]});
        assert_eq!(py_json(&v), r#"{"a": 1, "b": [1, 2]}"#);
        assert_eq!(compact_json(&v), r#"{"a":1,"b":[1,2]}"#);
    }

    /// `ensure_ascii=False` on both paths: non-ASCII stays raw, never `\uXXXX`.
    #[test]
    fn non_ascii_is_not_escaped() {
        let v = json!({"k": "café 中文"});
        assert_eq!(py_json(&v), r#"{"k": "café 中文"}"#);
        assert_eq!(compact_json(&v), r#"{"k":"café 中文"}"#);
    }

    /// Sorting is recursive through objects AND through objects nested inside
    /// arrays, while array ORDER itself is preserved — both halves of
    /// `deep_sort_dict`'s contract.
    #[test]
    fn deep_sort_orders_keys_recursively_and_keeps_array_order() {
        let v = json!({"b": 1, "a": {"z": 0, "y": [{"n": 1, "m": 2}]}});
        assert_eq!(
            compact_json(&deep_sort(&v)),
            r#"{"a":{"y":[{"m":2,"n":1}],"z":0},"b":1}"#
        );
        let arr = json!([3, 1, 2]);
        assert_eq!(compact_json(&deep_sort(&arr)), "[3,1,2]");
    }
}
