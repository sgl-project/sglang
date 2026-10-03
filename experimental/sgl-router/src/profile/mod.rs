// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! API profiles: the client-facing contract a router enforces for its model,
//! as one YAML file. See the README's "API profiles" section.

mod enforce;
mod load;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::sampling::{parse_sampling_overrides, ConflictPolicy, SamplingOverrides};

pub use enforce::{BudgetEdit, BudgetField};
pub use load::{preset_names, resolve, ENV};

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApiProfile {
    pub name: String,
    /// Where the profile came from, for logs.
    #[serde(skip_deserializing)]
    pub origin: String,
    pub models: Models,
    pub limits: Limits,
    pub output: Output,
    pub sampling: Option<Sampling>,
    pub errors: Errors,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Models {
    /// Accepted besides `--model-id`; routed as it.
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    #[serde(with = "byte_size")]
    pub max_body_bytes: Option<usize>,
    /// `image_url` parts per chat request.
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
    /// Injected when a request sets no budget.
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

/// `--override-sampling-params` as a profile section.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    pub conflict: Conflict,
    /// Same shape as the flag: a number, or `{min, max}`.
    pub params: Map<String, Value>,
}

/// Mirrors [`ConflictPolicy`], which is not serde-enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Conflict {
    #[default]
    Reject,
    Allow,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Errors {
    /// Replaces `type` in every 400 body on the chat route.
    pub bad_request_type: Option<String>,
}

impl ApiProfile {
    /// The sampling contract, validated by the flag's own parser.
    pub fn sampling_overrides(&self) -> Result<SamplingOverrides> {
        let Some(s) = self.sampling.as_ref().filter(|s| !s.params.is_empty()) else {
            return Ok(SamplingOverrides::default());
        };
        let conflict = match s.conflict {
            Conflict::Reject => ConflictPolicy::Reject,
            Conflict::Allow => ConflictPolicy::Allow,
        };
        parse_sampling_overrides(&Value::Object(s.params.clone()).to_string(), conflict)
            .context("sampling")
    }

    fn validate(&self) -> Result<()> {
        if let Some(m) = &self.output.max_tokens {
            ensure!(m.cap > 0, "output.max_tokens.cap must be positive");
            if let Some(d) = m.default {
                ensure!(
                    (1..=m.cap).contains(&d),
                    "output.max_tokens.default ({d}) must be in 1..={}",
                    m.cap
                );
            }
        }
        ensure!(
            self.limits.max_body_bytes != Some(0),
            "limits.max_body_bytes must be positive"
        );
        ensure!(
            self.limits.max_images != Some(0),
            "limits.max_images must be positive"
        );
        ensure!(
            self.models.aliases.iter().all(|a| !a.is_empty()),
            "models.aliases must not contain an empty name"
        );
        self.sampling_overrides()?;
        Ok(())
    }
}

/// `128MiB`-style sizes, or a plain byte count.
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
            Int(usize),
            Text(String),
        }
        match Option::<Raw>::deserialize(d)? {
            None => Ok(None),
            Some(Raw::Int(n)) => Ok(Some(n)),
            Some(Raw::Text(s)) => parse(&s).map(Some).map_err(D::Error::custom),
        }
    }

    pub(super) fn parse(s: &str) -> Result<usize, String> {
        let s = s.trim();
        let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        let (num, unit) = s.split_at(split);
        let shift = match unit.trim() {
            "" | "B" => 0,
            "KiB" => 10,
            "MiB" => 20,
            "GiB" => 30,
            other => {
                return Err(format!(
                    "unknown size unit `{other}` (use B, KiB, MiB or GiB)"
                ))
            }
        };
        num.parse::<usize>()
            .ok()
            .and_then(|n| n.checked_mul(1 << shift))
            .ok_or_else(|| format!("invalid size `{s}`"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_sizes() {
        assert_eq!(byte_size::parse("128MiB"), Ok(128 << 20));
        assert_eq!(byte_size::parse("4096"), Ok(4096));
        assert_eq!(byte_size::parse("2 GiB"), Ok(2 << 30));
        assert!(byte_size::parse("1TB").is_err());
        assert!(byte_size::parse("MiB").is_err());
    }

    #[test]
    fn sampling_compiles_through_the_flag_parser() {
        let p: ApiProfile = serde_yaml::from_str(
            "sampling: {params: {top_p: 0.95, temperature: {min: 0, max: 1}}}",
        )
        .unwrap();
        let o = p.sampling_overrides().unwrap();
        assert_eq!(o.params.len(), 2);
        assert_eq!(o.conflict, ConflictPolicy::Reject);

        let bad: ApiProfile = serde_yaml::from_str("sampling: {params: {top_p: 7}}").unwrap();
        assert!(format!("{:#}", bad.validate().unwrap_err()).contains("top_p"));
    }

    #[test]
    fn rejects_inconsistent_output_and_limits() {
        for yaml in [
            "output: {max_tokens: {cap: 0}}",
            "output: {max_tokens: {cap: 10, default: 11}}",
            "limits: {max_images: 0}",
            "models: {aliases: ['']}",
        ] {
            let p: ApiProfile = serde_yaml::from_str(yaml).unwrap();
            assert!(p.validate().is_err(), "{yaml}");
        }
        assert!(serde_yaml::from_str::<ApiProfile>("output: {max_tokns: {}}").is_err());
    }
}
