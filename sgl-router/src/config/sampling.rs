// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Fleet sampling defaults and constraints. Custom JSON visitors preserve duplicate
//! keys so validation can reject them and report the offending parameter.
//!
//! The flag is read once, at startup, on a router that crash-loops if it is wrong,
//! so the message an operator reads out of `kubectl logs` is the whole debugging
//! session: every rejection names the offending key, the value it saw, and the
//! domain it violated. (A `serde_json::Map` would keep only the last of a repeated
//! key and silently enforce a value the operator did not write.)

use anyhow::{anyhow, ensure, Result};
use std::collections::BTreeMap;

/// Fleet sampling defaults. Exact values fill absent fields; [`ConflictPolicy`]
/// determines whether differing client values are rejected or forwarded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOverrides {
    /// Parameters in deterministic injection order.
    pub params: BTreeMap<SamplingField, ParamSpec>,
    /// Conflict behavior shared by all configured parameters.
    pub conflict: ConflictPolicy,
}

impl SamplingOverrides {
    /// Validate parsed and programmatically constructed overrides alike.
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
    /// Reject differing client values with 400 before admission.
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
    /// Injected when absent; under [`ConflictPolicy::Reject`], the only accepted value.
    /// JSON numbers preserve integer wire types.
    Exact(serde_json::Number),
    /// Inclusive bounds for supplied values; never injects a default.
    /// Requires [`ConflictPolicy::Reject`]. Omitted fields use the engine default,
    /// which may lie outside the band.
    Range { lo: f64, hi: f64 },
}

/// Supported fleet sampling parameters.
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

    /// Slot in [`Self::ALL`] and in the request probe array.
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
    ensure!(
        !entries.is_empty(),
        "--override-sampling-params is empty: pass at least one of {}, or omit the flag",
        supported_fields()
    );
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
        ensure!(
            params.insert(field, spec).is_none(),
            "--override-sampling-params: {} is set more than once",
            field.wire_name()
        );
    }
    let overrides = SamplingOverrides { params, conflict };
    // Validate complete specs too, including bands and programmatically built overrides.
    // The earlier exact-value check protects the integer cast in `canonical_number`.
    overrides.validate()?;
    Ok(overrides)
}

/// Validate a spec independently of how it was constructed.
fn validate_spec(field: SamplingField, spec: &ParamSpec, conflict: ConflictPolicy) -> Result<()> {
    let key = field.wire_name();
    match spec {
        ParamSpec::Exact(n) => {
            checked_value(field, n)?;
        }
        &ParamSpec::Range { lo, hi } => {
            check_domain(field, lo, &lo.to_string())?;
            check_domain(field, hi, &hi.to_string())?;
            ensure!(
                lo <= hi,
                "--override-sampling-params: {key} band needs min <= max, got min {lo} > max {hi}"
            );
            // `top_k = -1` disables filtering; it cannot bound a band that would admit zero.
            ensure!(
                field != SamplingField::TopK || lo >= 1.0,
                "--override-sampling-params: top_k band bounds must both be >= 1 \
                     (-1 disables top_k entirely and cannot bound a range)"
            );
            // A band only ever rejects, so under `allow` it would be dead config
            // that silently accepts everything.
            ensure!(
                conflict == ConflictPolicy::Reject,
                "--override-sampling-params: the {key} band requires \
                     --sampling-param-conflict reject — under `allow` nothing is rejected \
                     and a band names no value to inject"
            );
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
        ensure!(
            slot.is_none(),
            "--override-sampling-params: {key} band sets \"{bound}\" more than once"
        );
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
    // Finished-spec constraints are checked by `validate_spec` for all construction paths.
    Ok(ParamSpec::Range { lo, hi })
}

/// Validate before normalization so diagnostics retain the configured number.
/// Domains follow the OpenAI contract plus engine-specific parameters — deliberately
/// NARROWER than what the engine accepts: these values are injected into request
/// bodies, and a fleet contract outside the range every OpenAI client library
/// validates against is far more likely a typo than an intent.
fn checked_value(field: SamplingField, n: &serde_json::Number) -> Result<f64> {
    let name = field.wire_name();
    // Handle conversion failure even if serde_json arbitrary precision is enabled later.
    let v = n.as_f64().ok_or_else(|| {
        anyhow!("--override-sampling-params: {name} ({n}) is not a finite number")
    })?;
    check_domain(field, v, &n.to_string())?;
    Ok(v)
}

/// Validate a numeric domain; `shown` is the value quoted in diagnostics.
fn check_domain(field: SamplingField, v: f64, shown: &str) -> Result<()> {
    let name = field.wire_name();
    let (ok, domain) = match field {
        SamplingField::Temperature => ((0.0..=2.0).contains(&v), "in [0, 2]"),
        SamplingField::TopP => (v > 0.0 && v <= 1.0, "in (0, 1]"),
        SamplingField::TopK => (v >= 1.0 || v == -1.0, ">= 1, or -1 to disable"),
        // Engine-specific: zero disables `min_p`.
        SamplingField::MinP => ((0.0..=1.0).contains(&v), "in [0, 1]"),
        // Engine-specific: one disables the penalty; cap fleet defaults at two.
        SamplingField::RepetitionPenalty => (v > 0.0 && v <= 2.0, "in (0, 2]"),
        SamplingField::FrequencyPenalty | SamplingField::PresencePenalty => {
            ((-2.0..=2.0).contains(&v), "in [-2, 2]")
        }
        // Bound sequence fan-out when `n` is injected into requests.
        SamplingField::N => ((1.0..=128.0).contains(&v), "in [1, 128]"),
    };
    ensure!(
        ok,
        "--override-sampling-params: {name} ({shown}) must be {domain}"
    );
    if field.is_integral() {
        ensure!(
            v.fract() == 0.0,
            "--override-sampling-params: {name} ({shown}) must be a whole number"
        );
        // Float-to-i64 casts saturate. Use [-2^63, 2^63): `i64::MAX as f64` rounds up.
        ensure!(
            (i64::MIN as f64..i64::MAX as f64).contains(&v),
            "--override-sampling-params: {name} ({shown}) is too large to forward"
        );
    }
    Ok(())
}

/// Emit integer-typed parameters as integers; preserve other JSON numbers.
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

/// Ordered JSON entries preserve duplicate keys for validation.
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

/// Raw parameter value; `Other` retains invalid values for precise diagnostics.
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

            /// `deserialize_any` routes JSON null to `visit_unit`.
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

    #[test]
    fn top_k_accepts_the_engines_disable_sentinel() {
        let o = parse(r#"{"top_k": -1}"#).unwrap();
        assert_eq!(exact_of(&o, SamplingField::TopK), Some(-1.0));
    }

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
    /// Float-to-i64 casts saturate. Use [-2^63, 2^63): `i64::MAX as f64` rounds up.
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

    #[test]
    fn all_covers_every_field_exactly_once() {
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
