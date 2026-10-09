// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! API profiles: the client-facing request contract (`docs/api-profiles.md`).

mod enforce;
mod load;

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

pub use enforce::Applied;
pub use load::{preset_names, resolve, ENV};

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApiProfile {
    pub name: String,
    /// Where the profile was loaded from, for the startup log.
    #[serde(skip)]
    pub origin: String,
    pub models: Models,
    pub limits: Limits,
    pub output: Output,
    /// Rules keyed by top-level request field (`temperature`, `reasoning_format`, …).
    pub params: BTreeMap<String, ParamRule>,
    pub errors: Errors,
    pub protocols: Protocols,
    pub messages: MessagesOpts,
}

/// `/v1/messages` behaviour.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MessagesOpts {
    pub thinking_blocks: ThinkingBlocks,
}

/// When `thinking` blocks are returned.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingBlocks {
    /// Whenever the model reasons.
    #[default]
    Always,
    /// Only when the request sets `thinking.type` to `enabled` or `adaptive`
    /// (Anthropic's default). The model still reasons; the blocks are dropped.
    OnRequest,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Models {
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    #[serde(with = "byte_size")]
    pub max_body_bytes: Option<usize>,
    pub max_images: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Output {
    pub max_tokens: Option<MaxTokens>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MaxTokens {
    pub cap: u64,
    #[serde(default)]
    pub on_exceed: OnExceed,
    /// Injected when the request sets no budget; `cap` when unset.
    #[serde(default)]
    pub default: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OnExceed {
    #[default]
    Reject,
    Clamp,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParamRule {
    pub default: Option<Value>,
    pub pin: Option<Value>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub exclusive_min: bool,
    pub exclusive_max: bool,
    pub values: Option<Vec<Value>>,
    #[serde(rename = "type")]
    pub ty: Option<ParamType>,
    /// `[from, to]` pairs applied to accepted values before forwarding.
    pub normalize: Vec<[Value; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    Number,
    Int,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Errors {
    /// `type` of every 400 the client sees; the protocol's standard value when unset.
    pub bad_request_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Protocols {
    pub chat: bool,
    pub messages: bool,
    pub responses: bool,
}

impl Default for Protocols {
    fn default() -> Self {
        Self {
            chat: true,
            messages: true,
            responses: true,
        }
    }
}

impl ParamRule {
    /// Why `v` is not accepted, or `None` if it is. Numeric checks only judge
    /// values the engine would read as numbers; anything else is the engine's.
    fn check(&self, name: &str, v: &Value) -> Option<String> {
        if let Some(pin) = &self.pin {
            return (!same(v, pin)).then(|| {
                format!("{name} is immutable for this model: got {v}, expected {pin} (or omit the field)")
            });
        }
        if let Some(values) = &self.values {
            if !values.iter().any(|a| same(v, a)) {
                return Some(format!("{name} must be one of {}, got {v}", list(values)));
            }
        }
        let n = lax_number(v)?;
        if self.ty == Some(ParamType::Int) && n.fract() != 0.0 {
            return Some(format!("{name} must be an integer, got {v}"));
        }
        let below = self
            .min
            .is_some_and(|m| if self.exclusive_min { n <= m } else { n < m });
        let above = self
            .max
            .is_some_and(|m| if self.exclusive_max { n >= m } else { n > m });
        (below || above).then(|| format!("{name} must be {}, got {v}", self.range()))
    }

    fn normalized(&self, v: Value) -> Value {
        self.normalize
            .iter()
            .find(|[from, _]| same(&v, from))
            .map_or(v, |[_, to]| to.clone())
    }

    fn range(&self) -> String {
        if let (Some(lo), Some(hi), false, false) =
            (self.min, self.max, self.exclusive_min, self.exclusive_max)
        {
            return format!("between {lo} and {hi}");
        }
        let lo = self
            .min
            .map(|m| format!("{} {m}", if self.exclusive_min { ">" } else { ">=" }));
        let hi = self
            .max
            .map(|m| format!("{} {m}", if self.exclusive_max { "<" } else { "<=" }));
        [lo, hi]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" and ")
    }

    fn validate(&self, name: &str) -> Result<()> {
        let constrained = self.default.is_some()
            || self.min.is_some()
            || self.max.is_some()
            || self.values.is_some();
        if self.pin.is_some() && constrained {
            bail!("params.{name}: `pin` cannot be combined with default, min, max or values");
        }
        if let (Some(lo), Some(hi)) = (self.min, self.max) {
            if lo > hi {
                bail!("params.{name}: min {lo} > max {hi}");
            }
        }
        for (what, v) in [("default", &self.default), ("pin", &self.pin)] {
            if let Some(v) = v {
                let unpinned = ParamRule {
                    pin: None,
                    ..self.clone()
                };
                if let Some(why) = unpinned.check(name, v) {
                    bail!("params.{name}.{what}: {why}");
                }
            }
        }
        for [from, _] in &self.normalize {
            if let Some(why) = self.check(name, from) {
                bail!("params.{name}.normalize: `{from}` is never accepted: {why}");
            }
        }
        Ok(())
    }
}

impl ApiProfile {
    /// The profile as YAML, without unset fields.
    pub fn to_yaml(&self) -> String {
        fn prune(v: &mut Value) -> bool {
            match v {
                Value::Object(m) => {
                    m.retain(|_, v| !prune(v));
                    m.is_empty()
                }
                Value::Array(a) => a.is_empty(),
                Value::Null | Value::Bool(false) => true,
                _ => false,
            }
        }
        let mut v = serde_json::to_value(self).expect("serialize profile");
        prune(&mut v);
        serde_yaml::to_string(&v).expect("profile to yaml")
    }

    pub fn validate(&self) -> Result<()> {
        for (name, rule) in &self.params {
            rule.validate(name)?;
        }
        if let Some(mt) = &self.output.max_tokens {
            if mt.cap == 0 {
                bail!("output.max_tokens.cap must be > 0");
            }
            if mt.default.is_some_and(|d| d == 0 || d > mt.cap) {
                bail!("output.max_tokens.default must be in 1..={}", mt.cap);
            }
        }
        if self.limits.max_body_bytes == Some(0) || self.limits.max_images == Some(0) {
            bail!("limits values must be > 0");
        }
        Ok(())
    }
}

/// A JSON value as the engine's pydantic lax mode reads a number.
pub(crate) fn lax_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Equality that treats `1`, `1.0` and `"1"` as the same number.
fn same(a: &Value, b: &Value) -> bool {
    match (lax_number(a), lax_number(b)) {
        (Some(x), Some(y)) if a.is_number() || b.is_number() => x == y,
        _ => a == b,
    }
}

fn list(values: &[Value]) -> String {
    values
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn number(n: f64) -> Value {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        Value::from(n as i64)
    } else {
        Number::from_f64(n).map_or(Value::Null, Value::Number)
    }
}

/// `128MiB`, `512KiB`, `1GiB`, `100B` or a plain integer.
mod byte_size {
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<usize>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(n) => s.serialize_u64(*n as u64),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<usize>, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            N(usize),
            S(String),
        }
        let s = match Option::<Raw>::deserialize(d)? {
            None => return Ok(None),
            Some(Raw::N(n)) => return Ok(Some(n)),
            Some(Raw::S(s)) => s,
        };
        let t = s.trim();
        let split = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
        let (num, unit) = t.split_at(split);
        let mult = match unit.trim() {
            "" | "B" => 1,
            "KiB" => 1 << 10,
            "MiB" => 1 << 20,
            "GiB" => 1 << 30,
            u => {
                return Err(D::Error::custom(format!(
                    "unknown size unit `{u}` (use B, KiB, MiB, GiB)"
                )))
            }
        };
        num.parse::<usize>()
            .map(|n| Some(n * mult))
            .map_err(|_| D::Error::custom(format!("invalid size `{s}`")))
    }
}
