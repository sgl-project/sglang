// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Fleet-wide sampling contract (`--override-sampling-params` /
//! `--sampling-param-conflict`): the parameters an operator fixes for every
//! request this router serves, and what a request that disagrees gets.
//!
//! WHY this is parsed by hand rather than with `serde(deny_unknown_fields)`:
//! the flag is read once, at startup, on a router that crash-loops if it is
//! wrong, so the message an operator reads out of `kubectl logs` is the whole
//! debugging session. Every rejection here names the offending key, the value
//! it saw, and the domain it violated. The same reasoning is why both the
//! outer object and a band are decoded as ordered ENTRIES instead of a
//! `serde_json::Map`: a map keeps only the last of a repeated key, so
//! `{"temperature": 0, "temperature": 1}` would start cleanly and enforce a
//! value the operator did not write.

use anyhow::{anyhow, Result};
use std::collections::BTreeMap;

/// Sampling parameters fixed fleet-wide, and what to do with a request that
/// disagrees.
///
/// A configured parameter is always injected into the forwarded body when the
/// request OMITS it, so the engine's own defaults cannot drift from what the
/// operator declared. What differs between the two [`ConflictPolicy`] modes is
/// only the request that DOES send the field: `Reject` makes the value an
/// immutability contract (400 before admission — never a silent rewrite),
/// while `Allow` lets the client value through untouched, degrading the
/// configured value to a fleet-wide default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOverrides {
    /// Configured parameters, keyed so enforcement and injection are one loop
    /// over whatever the operator set instead of a per-field ladder repeated
    /// at each site. Iterating a `BTreeMap` keyed by the field enum is what
    /// fixes the order values are injected in, so a forwarded body is
    /// byte-identical across runs.
    pub params: BTreeMap<SamplingField, ParamSpec>,
    /// Applies to every configured parameter: there is deliberately no
    /// per-parameter mode, so an operator reads one knob off one manifest.
    pub conflict: ConflictPolicy,
}

impl SamplingOverrides {
    /// Re-check every invariant [`parse_sampling_overrides`] enforces, on an
    /// already-built value.
    ///
    /// WHY this is separate from the parser: the parser turns a raw JSON
    /// string into this struct and is reachable only from the CLI, but the
    /// struct itself is reachable from anywhere — a test fixture, a future
    /// config file, an admin API. [`crate::config::Config::validate`] calls
    /// this so no such path can hold a spec the flag would have refused to
    /// start with (an out-of-domain exact value, an inverted band whose
    /// `contains` rejects every value, or a band under `allow`, which names
    /// nothing to inject and rejects nothing).
    pub(crate) fn validate(&self) -> Result<()> {
        for (&field, spec) in &self.params {
            validate_spec(field, spec, self.conflict)?;
        }
        Ok(())
    }
}

/// What a request sending a value that differs from the configured one gets
/// (`--sampling-param-conflict`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum ConflictPolicy {
    /// 400 before admission, quoting the configured value. The default: the
    /// point of declaring a fleet-wide sampling contract is usually that it
    /// holds, and silently serving something other than what the client asked
    /// for is the one behavior no client can detect.
    #[default]
    Reject,
    /// Forward the client's value to the engine untouched. The configured
    /// value degrades to a fill-when-absent default.
    Allow,
}

/// One configured parameter's value: a single value, or an inclusive band of
/// accepted ones.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamSpec {
    /// A single value: injected when the request omits the field, and under
    /// [`ConflictPolicy::Reject`] the only value a request may send.
    ///
    /// Held as the parsed JSON number rather than an `f64` so injection
    /// re-emits the operator's literal — `"n": 1` stays `1` and does not
    /// become `1.0` on the wire for the integer-typed fields.
    Exact(serde_json::Number),
    /// An inclusive `[lo, hi]` band of accepted values, for a contract that
    /// fixes most sampling knobs but leaves one tunable inside a range. A band
    /// names no single value, so it never injects; it only rejects
    /// out-of-band values, which is why a band under [`ConflictPolicy::Allow`]
    /// is a startup error rather than a no-op.
    ///
    /// A band therefore constrains only the requests that NAME the parameter.
    /// A request that omits it gets the model's own default (the engine reads
    /// `generation_config`), which the router cannot see and which may itself
    /// lie outside the band. An operator who needs the omitting majority
    /// pinned too wants an exact value, not a band.
    Range { lo: f64, hi: f64 },
}

/// A sampling parameter that can be fixed fleet-wide. The enum is what makes a
/// typo in the `--override-sampling-params` JSON a startup error instead of a
/// key that silently never matches a request field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SamplingField {
    Temperature,
    TopP,
    TopK,
    MinP,
    RepetitionPenalty,
    FrequencyPenalty,
    PresencePenalty,
    N,
}

impl SamplingField {
    /// Every field, in the order they are written into the forwarded body.
    pub const ALL: [Self; 8] = [
        Self::Temperature,
        Self::TopP,
        Self::TopK,
        Self::MinP,
        Self::RepetitionPenalty,
        Self::FrequencyPenalty,
        Self::PresencePenalty,
        Self::N,
    ];

    /// This field's slot in [`Self::ALL`], and in the request probe's
    /// fixed-size array of probed values.
    ///
    /// Declaration order IS the slot order, so this cannot assign a wrong
    /// one. What still needs checking is that [`Self::ALL`] agrees — see the
    /// assertion below; `from_wire_name` and `supported_fields` both iterate
    /// `ALL`, so a field missing from it is rejected at startup as an unknown
    /// key, the failure that is invisible to any test that also iterates
    /// `ALL`.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The request-body key, identical on the wire in and out: the JSON config
    /// names the parameter exactly as a client sends it.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Temperature => "temperature",
            Self::TopP => "top_p",
            Self::TopK => "top_k",
            Self::MinP => "min_p",
            Self::RepetitionPenalty => "repetition_penalty",
            Self::FrequencyPenalty => "frequency_penalty",
            Self::PresencePenalty => "presence_penalty",
            Self::N => "n",
        }
    }

    /// Parse a key from the `--override-sampling-params` JSON object.
    pub fn from_wire_name(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.wire_name() == key)
    }

    /// True for the fields the engine types as `int`, which a configured value
    /// must therefore be integral for.
    pub const fn is_integral(self) -> bool {
        matches!(self, Self::TopK | Self::N)
    }
}

/// [`SamplingField::ALL`] agrees with [`SamplingField::index`] — see that
/// method for why a disagreement is a silent startup rejection.
const _: () = {
    let mut i = 0;
    while i < SamplingField::ALL.len() {
        assert!(SamplingField::ALL[i].index() == i);
        i += 1;
    }
};

/// Parse `--override-sampling-params` into a [`SamplingOverrides`].
pub(crate) fn parse_sampling_overrides(
    raw: &str,
    conflict: ConflictPolicy,
) -> Result<SamplingOverrides> {
    let ObjectEntries(entries) = serde_json::from_str(raw).map_err(|e| {
        anyhow!(
            "--override-sampling-params must be a JSON object like \
             '{{\"temperature\": 1, \"top_p\": 0.95}}': {e}"
        )
    })?;
    if entries.is_empty() {
        return Err(anyhow!(
            "--override-sampling-params is empty: pass at least one of {}, or omit the flag",
            supported_fields()
        ));
    }
    let mut params = BTreeMap::new();
    for (key, value) in entries {
        let field = SamplingField::from_wire_name(&key).ok_or_else(|| {
            anyhow!(
                "--override-sampling-params: unknown parameter \"{key}\" (supported: {})",
                supported_fields()
            )
        })?;
        let spec = match value {
            ParamValue::Number(n) => {
                ParamSpec::Exact(canonical_number(field, checked_value(field, &n)?, n))
            }
            ParamValue::Band(entries) => parse_band(field, entries)?,
            ParamValue::Other(other) => {
                return Err(anyhow!(
                    "--override-sampling-params: {key} must be a number or a \
                     {{\"min\": LO, \"max\": HI}} band, got {other}"
                ))
            }
        };
        if params.insert(field, spec).is_some() {
            return Err(anyhow!(
                "--override-sampling-params: {} is set more than once",
                field.wire_name()
            ));
        }
    }
    let overrides = SamplingOverrides { params, conflict };
    // Re-checks the domains `checked_value` already covered above. That first
    // pass is not redundant: it is what quotes the operator's own literal
    // (`1e2`, not `100`) and what guards `canonical_number`'s saturating
    // `as i64` cast before it runs. This pass is what a hand-built
    // `SamplingOverrides` gets, and is the only check a band's bounds see.
    overrides.validate()?;
    Ok(overrides)
}

/// Check one already-built spec. Shared by [`parse_sampling_overrides`] and
/// [`SamplingOverrides::validate`] so a hand-built `SamplingOverrides` is held
/// to exactly the domain the flag is.
fn validate_spec(field: SamplingField, spec: &ParamSpec, conflict: ConflictPolicy) -> Result<()> {
    let key = field.wire_name();
    match spec {
        ParamSpec::Exact(n) => {
            checked_value(field, n)?;
        }
        &ParamSpec::Range { lo, hi } => {
            check_domain(field, lo, &lo.to_string())?;
            check_domain(field, hi, &hi.to_string())?;
            if lo > hi {
                return Err(anyhow!(
                    "--override-sampling-params: {key} band needs min <= max, got min {lo} > max {hi}"
                ));
            }
            // Bounds are checked one at a time, which is only sufficient for a
            // contiguous domain. `top_k`'s is not ({-1} U [1, inf)): `{"min": -1,
            // "max": 100}` has two individually legal bounds and would admit
            // `top_k: 0`, which is rejected as an exact value. -1 is a sentinel,
            // not a range endpoint.
            if field == SamplingField::TopK && lo < 1.0 {
                return Err(anyhow!(
                    "--override-sampling-params: top_k band bounds must both be >= 1 \
                     (-1 disables top_k entirely and cannot bound a range)"
                ));
            }
            // A band only ever rejects, so under `allow` it would be dead config
            // that silently accepts everything.
            if conflict == ConflictPolicy::Allow {
                return Err(anyhow!(
                    "--override-sampling-params: the {key} band requires \
                     --sampling-param-conflict reject — under `allow` nothing is rejected \
                     and a band names no value to inject"
                ));
            }
        }
    }
    Ok(())
}

/// Turn one parameter's `{"min": LO, "max": HI}` entries into a
/// [`ParamSpec::Range`].
fn parse_band(
    field: SamplingField,
    entries: Vec<(String, serde_json::Value)>,
) -> Result<ParamSpec> {
    let key = field.wire_name();
    let (mut min, mut max) = (None, None);
    for (bound, v) in entries {
        let slot = match bound.as_str() {
            "min" => &mut min,
            "max" => &mut max,
            _ => {
                return Err(anyhow!(
                    "--override-sampling-params: {key} band must be exactly \
                     {{\"min\": LO, \"max\": HI}}, got an unexpected \"{bound}\""
                ))
            }
        };
        if slot.is_some() {
            return Err(anyhow!(
                "--override-sampling-params: {key} band sets \"{bound}\" more than once"
            ));
        }
        let serde_json::Value::Number(n) = v else {
            return Err(anyhow!(
                "--override-sampling-params: {key} band needs numeric bounds, got {bound}: {v}"
            ));
        };
        *slot = Some(checked_value(field, &n)?);
    }
    let (Some(lo), Some(hi)) = (min, max) else {
        return Err(anyhow!(
            "--override-sampling-params: {key} band must be exactly \
             {{\"min\": LO, \"max\": HI}} with numeric bounds"
        ));
    };
    // `lo <= hi`, `top_k`'s discontiguous domain and the band-under-`allow`
    // rule are all properties of the finished spec, so they live in
    // `validate_spec` and hold for a hand-built `SamplingOverrides` too.
    Ok(ParamSpec::Range { lo, hi })
}

/// Check one configured value against its parameter's domain, at startup
/// instead of per request. Written as positive containment so a NaN bound
/// fails too.
///
/// These are the OpenAI API's domains, which are NARROWER than what the engine
/// itself accepts (`SamplingParams.verify` requires only that `temperature` be
/// non-negative and finite, so it would take `temperature: 5`). Narrower is
/// deliberate: the values here are injected into request bodies, and a fleet
/// contract outside the range every OpenAI client library validates against is
/// far more likely a typo than an intent. The one exception is `top_k`, where
/// `-1` is the engine's own "disable / whole vocabulary" spelling and its
/// default — a legitimate thing to fix fleet-wide. Note `top_k: 1` is greedy
/// decoding, NOT "disabled".
fn checked_value(field: SamplingField, n: &serde_json::Number) -> Result<f64> {
    let name = field.wire_name();
    // `as_f64` is infallible for a JSON number unless serde_json's
    // `arbitrary_precision` is on (it is not); kept total rather than
    // `expect`-ing, so enabling that feature can't turn config into a panic.
    let v = n.as_f64().ok_or_else(|| {
        anyhow!("--override-sampling-params: {name} ({n}) is not a finite number")
    })?;
    // The operator's own literal is what the message quotes, not the parsed
    // f64: `1e2` should read back as `1e2`.
    check_domain(field, v, &n.to_string())?;
    Ok(v)
}

/// The domain half of [`checked_value`], over an f64 that may not have come
/// from a literal (a band's bounds are stored as f64). `shown` is what the
/// error quotes back to the operator.
fn check_domain(field: SamplingField, v: f64, shown: &str) -> Result<()> {
    let name = field.wire_name();
    let (ok, domain) = match field {
        SamplingField::Temperature => ((0.0..=2.0).contains(&v), "in [0, 2]"),
        SamplingField::TopP => (v > 0.0 && v <= 1.0, "in (0, 1]"),
        SamplingField::TopK => (v >= 1.0 || v == -1.0, ">= 1, or -1 to disable"),
        // Not an OpenAI parameter: `min_p` is the engine's own nucleus floor,
        // and 0 is its default (disabled), so the whole [0, 1] range is
        // legitimate to fix fleet-wide.
        SamplingField::MinP => ((0.0..=1.0).contains(&v), "in [0, 1]"),
        // Also engine-only. 1.0 is "no penalty"; the engine requires > 0, and
        // values above ~2 degrade output badly enough that a fleet-wide pin
        // there is far more likely a typo than an intent.
        SamplingField::RepetitionPenalty => (v > 0.0 && v <= 2.0, "in (0, 2]"),
        SamplingField::FrequencyPenalty | SamplingField::PresencePenalty => {
            ((-2.0..=2.0).contains(&v), "in [-2, 2]")
        }
        // OpenAI caps `n` at 128. Unbounded here, a typo'd digit would be
        // injected into every request that omits `n` and fan each one out to
        // that many sequences at the engine — the exact per-request failure
        // this startup check exists to convert into a launch failure.
        SamplingField::N => ((1.0..=128.0).contains(&v), "in [1, 128]"),
    };
    if !ok {
        return Err(anyhow!(
            "--override-sampling-params: {name} ({shown}) must be {domain}"
        ));
    }
    if field.is_integral() {
        if v.fract() != 0.0 {
            return Err(anyhow!(
                "--override-sampling-params: {name} ({shown}) must be a whole number"
            ));
        }
        // `canonical_number` casts to `i64`, and a Rust float-to-int cast
        // SATURATES rather than failing, so a literal past the i64 range would
        // silently become `i64::MAX` in every forwarded body. The exactly
        // convertible f64s are [-2^63, 2^63), which is this half-open range
        // written as positive containment — `i64::MAX as f64` rounds UP to
        // 2^63, so an inclusive `<=` against it would admit 2^63 itself and
        // saturate exactly as described.
        if !(i64::MIN as f64..i64::MAX as f64).contains(&v) {
            return Err(anyhow!(
                "--override-sampling-params: {name} ({shown}) is too large to forward"
            ));
        }
    }
    Ok(())
}

/// Normalize an integer-typed parameter's literal so injection writes `1`
/// rather than `1.0` for a config that spelled it `1.0` — the engine types
/// these fields as `int`, and the forwarded body should look like what a
/// client would have sent. Non-integral fields keep the operator's literal.
fn canonical_number(
    field: SamplingField,
    value: f64,
    literal: serde_json::Number,
) -> serde_json::Number {
    if field.is_integral() {
        serde_json::Number::from(value as i64)
    } else {
        literal
    }
}

/// The supported `--override-sampling-params` keys, for error messages.
fn supported_fields() -> String {
    SamplingField::ALL
        .iter()
        .map(|f| f.wire_name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// A JSON object decoded to its entries IN ORDER, keeping a repeated key
/// instead of collapsing it. See the module WHY note.
struct ObjectEntries(Vec<(String, ParamValue)>);

impl<'de> serde::Deserialize<'de> for ObjectEntries {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct EntryVisitor;
        impl<'de> serde::de::Visitor<'de> for EntryVisitor {
            type Value = ObjectEntries;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<ObjectEntries, M::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, ParamValue>()? {
                    entries.push(entry);
                }
                Ok(ObjectEntries(entries))
            }
        }
        // `deserialize_map` rejects a non-object with the type error the
        // caller wraps into the flag's own message.
        d.deserialize_map(EntryVisitor)
    }
}

/// One parameter's raw value: a number, a band's entries, or anything else.
/// `Other` keeps the offending value so the caller can name it, rather than
/// degrading a wrong-type message into a serde type error behind the outer
/// object's context.
enum ParamValue {
    Number(serde_json::Number),
    Band(Vec<(String, serde_json::Value)>),
    Other(serde_json::Value),
}

impl<'de> serde::Deserialize<'de> for ParamValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> serde::de::Visitor<'de> for ValueVisitor {
            type Value = ParamValue;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a number or a {\"min\": LO, \"max\": HI} band")
            }

            fn visit_i64<E>(self, v: i64) -> Result<ParamValue, E> {
                Ok(ParamValue::Number(v.into()))
            }

            fn visit_u64<E>(self, v: u64) -> Result<ParamValue, E> {
                Ok(ParamValue::Number(v.into()))
            }

            fn visit_f64<E>(self, v: f64) -> Result<ParamValue, E> {
                // `from_f64` is None only for NaN/inf, which JSON cannot
                // express; `checked_value` rejects the null either way.
                Ok(match serde_json::Number::from_f64(v) {
                    Some(n) => ParamValue::Number(n),
                    None => ParamValue::Other(serde_json::Value::Null),
                })
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<ParamValue, M::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, serde_json::Value>()? {
                    entries.push(entry);
                }
                Ok(ParamValue::Band(entries))
            }

            fn visit_bool<E>(self, v: bool) -> Result<ParamValue, E> {
                Ok(ParamValue::Other(v.into()))
            }

            fn visit_str<E>(self, v: &str) -> Result<ParamValue, E> {
                Ok(ParamValue::Other(v.into()))
            }

            /// JSON `null`. There is deliberately no `visit_none`: this type
            /// is only ever reached through `deserialize_any`, which routes
            /// null here and never to the `Option` hook.
            fn visit_unit<E>(self) -> Result<ParamValue, E> {
                Ok(ParamValue::Other(serde_json::Value::Null))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<ParamValue, A::Error> {
                let mut items = Vec::new();
                while let Some(v) = seq.next_element::<serde_json::Value>()? {
                    items.push(v);
                }
                Ok(ParamValue::Other(serde_json::Value::Array(items)))
            }
        }
        d.deserialize_any(ValueVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &str) -> Result<SamplingOverrides> {
        parse_sampling_overrides(raw, ConflictPolicy::Reject)
    }

    /// The exact value configured for one parameter, as an f64.
    fn exact_of(o: &SamplingOverrides, field: SamplingField) -> Option<f64> {
        match o.params.get(&field) {
            Some(ParamSpec::Exact(n)) => n.as_f64(),
            _ => None,
        }
    }

    #[test]
    fn parses_every_supported_parameter() {
        let o = parse(
            r#"{"temperature": 1, "top_p": 1.0, "top_k": 20,
                "frequency_penalty": 0, "presence_penalty": -0.5, "n": 1}"#,
        )
        .unwrap();
        assert_eq!(exact_of(&o, SamplingField::Temperature), Some(1.0));
        assert_eq!(exact_of(&o, SamplingField::TopP), Some(1.0));
        assert_eq!(exact_of(&o, SamplingField::TopK), Some(20.0));
        assert_eq!(exact_of(&o, SamplingField::FrequencyPenalty), Some(0.0));
        assert_eq!(exact_of(&o, SamplingField::PresencePenalty), Some(-0.5));
        assert_eq!(exact_of(&o, SamplingField::N), Some(1.0));
        assert_eq!(o.conflict, ConflictPolicy::Reject);
    }

    #[test]
    fn parses_a_band() {
        let o = parse(r#"{"temperature": {"min": 0, "max": 1}}"#).unwrap();
        assert_eq!(
            o.params.get(&SamplingField::Temperature),
            Some(&ParamSpec::Range { lo: 0.0, hi: 1.0 })
        );
    }

    /// A band can only ever reject, so under `allow` it would be dead config.
    #[test]
    fn a_band_under_allow_is_a_startup_error() {
        let err = parse_sampling_overrides(
            r#"{"temperature": {"min": 0, "max": 1}}"#,
            ConflictPolicy::Allow,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("requires --sampling-param-conflict reject"),
            "got: {err}"
        );
    }

    /// Every malformed spelling fails the launch, naming the offending key and
    /// its domain.
    #[test]
    fn rejects_malformed_input() {
        for (json, needle) in [
            ("not json", "must be a JSON object"),
            ("[1, 2]", "must be a JSON object"),
            (r#"[["temperature", 1]]"#, "must be a JSON object"),
            ("{}", "is empty"),
            (r#"{"temp": 1}"#, "unknown parameter"),
            (r#"{"max_tokens": 100}"#, "unknown parameter"),
            (r#"{"temperature": 2.5}"#, "in [0, 2]"),
            (r#"{"temperature": -0.1}"#, "in [0, 2]"),
            (r#"{"top_p": 0}"#, "in (0, 1]"),
            (r#"{"top_p": 1.5}"#, "in (0, 1]"),
            (r#"{"top_k": 0}"#, ">= 1"),
            (r#"{"top_k": -2}"#, ">= 1"),
            (r#"{"top_k": 1.5}"#, "whole number"),
            (r#"{"frequency_penalty": 2.1}"#, "in [-2, 2]"),
            (r#"{"presence_penalty": -2.1}"#, "in [-2, 2]"),
            (r#"{"n": 0}"#, "in [1, 128]"),
            (r#"{"n": 129}"#, "in [1, 128]"),
            (r#"{"n": 1.5}"#, "whole number"),
            (r#"{"n": 1e20}"#, "in [1, 128]"),
            (r#"{"temperature": "1"}"#, "must be a number"),
            (r#"{"temperature": true}"#, "must be a number"),
            (r#"{"temperature": null}"#, "must be a number"),
            (r#"{"temperature": [1]}"#, "must be a number"),
            (
                r#"{"temperature": 0, "temperature": 1}"#,
                "temperature is set more than once",
            ),
            (r#"{"temperature": {"min": 1, "max": 0}}"#, "min <= max"),
            (r#"{"temperature": {"min": 0, "max": 2.5}}"#, "in [0, 2]"),
            (r#"{"temperature": {"min": 0}}"#, "band must be exactly"),
            (
                r#"{"temperature": {"min": 0, "max": 1, "typo": 2}}"#,
                "unexpected \"typo\"",
            ),
            (
                r#"{"temperature": {"min": 0, "max": 1, "max": 0.5}}"#,
                "band sets \"max\" more than once",
            ),
            (
                r#"{"temperature": {"min": "0", "max": "1"}}"#,
                "needs numeric bounds",
            ),
            (
                r#"{"top_k": {"min": -1, "max": 100}}"#,
                "band bounds must both be >= 1",
            ),
        ] {
            let err = parse(json).unwrap_err().to_string();
            assert!(err.contains(needle), "{json}: got {err}");
        }
    }

    /// `top_p`'s domain is (0, 1] — the inclusive upper bound is valid — and
    /// `top_k` has no upper bound, so a wide sample width parses.
    #[test]
    fn top_p_upper_bound_and_wide_top_k_are_accepted() {
        let o = parse(r#"{"top_p": 1, "top_k": 1000}"#).unwrap();
        assert_eq!(exact_of(&o, SamplingField::TopP), Some(1.0));
        assert_eq!(exact_of(&o, SamplingField::TopK), Some(1000.0));
    }

    /// `top_k: -1` is the engine's own "disable / whole vocabulary" spelling
    /// (and its default), so a fleet may legitimately fix `top_k` to it —
    /// unlike every other parameter, whose domain is the OpenAI one.
    #[test]
    fn top_k_accepts_the_engines_disable_sentinel() {
        let o = parse(r#"{"top_k": -1}"#).unwrap();
        assert_eq!(exact_of(&o, SamplingField::TopK), Some(-1.0));
    }

    /// An integer-typed parameter spelled as a float is normalized, so the
    /// forwarded body carries `1` and not `1.0` for a field the engine types
    /// as `int`.
    #[test]
    fn integral_params_are_normalized_to_integers() {
        let o = parse(r#"{"n": 1.0, "top_k": 20.0}"#).unwrap();
        for field in [SamplingField::N, SamplingField::TopK] {
            let Some(ParamSpec::Exact(n)) = o.params.get(&field) else {
                panic!("{field:?} must be an exact value");
            };
            assert!(n.is_i64(), "{field:?} kept a float literal: {n}");
        }
    }

    /// The wire name is the request-body key in both directions, so a
    /// configured key round-trips back to its field.
    #[test]
    fn every_field_round_trips_through_its_wire_name() {
        for field in SamplingField::ALL {
            assert_eq!(
                SamplingField::from_wire_name(field.wire_name()),
                Some(field)
            );
        }
        assert_eq!(SamplingField::from_wire_name("max_tokens"), None);
    }
    /// `canonical_number` casts to `i64` and a Rust float-to-int cast
    /// SATURATES, so a literal past the i64 range must fail the launch rather
    /// than be injected as `i64::MAX`. The boundary case is the trap: `i64::MAX
    /// as f64` rounds UP to 2^63, so a `>` comparison against it admits 2^63
    /// itself.
    #[test]
    fn integral_literals_beyond_i64_fail_the_launch() {
        for raw in [
            // 2^63 exactly — equal to `i64::MAX as f64`, not greater than it.
            r#"{"top_k": 9223372036854775808}"#,
            // i64::MAX, which also rounds to 2^63 as an f64.
            r#"{"top_k": 9223372036854775807}"#,
            r#"{"top_k": 1e30}"#,
        ] {
            let err = parse(raw).unwrap_err().to_string();
            assert!(err.contains("too large to forward"), "{raw}: got {err}");
        }
        // Nothing in range regressed.
        assert_eq!(
            exact_of(
                &parse(r#"{"top_k": 1000000}"#).unwrap(),
                SamplingField::TopK
            ),
            Some(1_000_000.0)
        );
    }

    /// `min_p` and `repetition_penalty` are the only two parameters besides
    /// `temperature`/`top_p`/`top_k` that the engine resolves from the model's
    /// own `generation_config`, so they are exactly the ones a fleet-wide pin
    /// exists to stop drifting when an image is swapped. Rejecting them as
    /// unknown keys would crash-loop the router for the operator who needs the
    /// flag most.
    #[test]
    fn governs_the_engine_defaulted_parameters() {
        let o = parse(r#"{"min_p": 0.05, "repetition_penalty": 1.1}"#).unwrap();
        assert_eq!(exact_of(&o, SamplingField::MinP), Some(0.05));
        assert_eq!(exact_of(&o, SamplingField::RepetitionPenalty), Some(1.1));

        // Both defaults are legitimate fleet-wide pins.
        let o = parse(r#"{"min_p": 0, "repetition_penalty": 1}"#).unwrap();
        assert_eq!(exact_of(&o, SamplingField::MinP), Some(0.0));
        assert_eq!(exact_of(&o, SamplingField::RepetitionPenalty), Some(1.0));

        for (raw, needle) in [
            (r#"{"min_p": -0.1}"#, "in [0, 1]"),
            (r#"{"min_p": 1.1}"#, "in [0, 1]"),
            (r#"{"repetition_penalty": 0}"#, "in (0, 2]"),
            (r#"{"repetition_penalty": 2.5}"#, "in (0, 2]"),
        ] {
            let err = parse(raw).unwrap_err().to_string();
            assert!(err.contains(needle), "{raw}: got {err}");
        }
    }

    /// `ALL` is what `from_wire_name` and `supported_fields` iterate, so a
    /// field missing from it is silently rejected at startup as an unknown
    /// key — invisible to any test that also iterates `ALL`. Pin the length
    /// and the slot mapping against the wire names instead.
    #[test]
    fn all_covers_every_field_exactly_once() {
        // The slot mapping itself is asserted at compile time (see the
        // `const _` block above `parse_sampling_overrides`); what only a test
        // can catch is a field missing from `ALL` entirely, which is why the
        // names below are written out rather than derived from it.
        let names: std::collections::BTreeSet<_> =
            SamplingField::ALL.iter().map(|f| f.wire_name()).collect();
        assert_eq!(names.len(), SamplingField::ALL.len(), "duplicate wire name");
        for name in [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "n",
        ] {
            assert!(names.contains(name), "{name} is not governed");
        }
    }

    /// The struct-level invariants must hold for a `SamplingOverrides` that
    /// never went through the parser — a test fixture, a future config file or
    /// admin API. An inverted band is the nastiest of these: `(lo..=hi)`
    /// contains nothing, so it would 400 every request naming the parameter.
    #[test]
    fn validate_rejects_hand_built_specs_the_parser_would_refuse() {
        let bad = [
            (
                ParamSpec::Exact(serde_json::Number::from_f64(50.0).unwrap()),
                ConflictPolicy::Reject,
                "in [0, 2]",
            ),
            (
                ParamSpec::Range { lo: 1.0, hi: 0.0 },
                ConflictPolicy::Reject,
                "min <= max",
            ),
            (
                ParamSpec::Range { lo: 0.0, hi: 1.0 },
                ConflictPolicy::Allow,
                "requires --sampling-param-conflict reject",
            ),
        ];
        for (spec, conflict, needle) in bad {
            let o = SamplingOverrides {
                params: [(SamplingField::Temperature, spec)].into_iter().collect(),
                conflict,
            };
            let err = o.validate().unwrap_err().to_string();
            assert!(err.contains(needle), "got {err}");
        }

        // A well-formed contract still validates.
        parse(r#"{"temperature": 1, "min_p": 0.05}"#)
            .unwrap()
            .validate()
            .unwrap();
    }
}
