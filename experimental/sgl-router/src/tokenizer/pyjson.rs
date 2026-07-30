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
/// CAVEAT: number formatting matches Python for the ints/floats/bools that
/// appear in real tool schemas, but not universally — scientific-notation floats
/// (`1e-5` renders `0.00001` here vs Python `1e-05`) and integers outside
/// `i64`/`u64` range diverge from Python's `repr` (serde parses those as `f64`,
/// so `18446744073709551616` renders `1.8446744073709552e+19`). Integers that
/// FIT in `i64`/`u64` are exact on both sides, including past f64's 2^53
/// precision limit — `9007199254740993` and `18446744073709551615` both
/// round-trip identically. Such a value in a tool schema would miss the exact
/// cache match and degrade to min-load; none appear in observed (OpenCode) tool
/// schemas, so this is left as a known edge rather than reimplementing Python
/// float formatting.
pub(crate) fn py_json(value: &serde_json::Value) -> String {
    format_with(value, PyJsonFormatter)
}

/// Serialize `value` the way Python's
/// `json.dumps(v, ensure_ascii=False, separators=(",", ":"))` does — no spaces
/// after `,` or `:`. This is serde's own compact default, so it just delegates;
/// it exists as a named counterpart to [`py_json`] so a call site states which
/// Python convention it is mirroring instead of leaving it implicit.
///
/// Carries the same number-formatting caveat as [`py_json`].
pub(crate) fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).expect("serializing a serde_json::Value to a String is infallible")
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

/// serde_json formatter emitting Python `json.dumps` default separators
/// (`", "` / `": "`) instead of serde's compact `,` / `:`.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
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
