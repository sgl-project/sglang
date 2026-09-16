//! Scenario assertions run on complete responses before value exceptions.
//!
//! These checks prove the intended metadata condition was exercised. A failed
//! assertion leaves the response intact for repeatability and parity comparison.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sglang_parity::compare::Violation;
use sglang_parity::runner::AssertionResult;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "check", rename_all = "snake_case", deny_unknown_fields)]
pub enum Expectation {
    CachedTokens { positive: bool },
    CacheDetails { positive: bool },
    DpRank { value: Option<u64> },
    ReasoningTokens { positive: bool },
    Retractions { positive: bool },
    Timestamp,
    WeightVersion { value: String },
    InputTopLogprobs { populated: bool },
    OutputLogprobsLength,
}

impl Expectation {
    pub fn name(&self) -> &'static str {
        match self {
            Self::CachedTokens { .. } => "cached_tokens",
            Self::CacheDetails { .. } => "cached_tokens_details",
            Self::DpRank { .. } => "dp_rank",
            Self::ReasoningTokens { .. } => "reasoning_tokens",
            Self::Retractions { .. } => "num_retractions",
            Self::Timestamp => "response_sent_to_client_ts",
            Self::WeightVersion { .. } => "weight_version",
            Self::InputTopLogprobs { .. } => "input_top_logprobs",
            Self::OutputLogprobsLength => "output_token_logprobs_length",
        }
    }

    pub fn validate_all(expectations: &[Self]) -> Result<(), String> {
        let mut names = std::collections::BTreeSet::new();
        for expectation in expectations {
            if !names.insert(expectation.name()) {
                return Err(format!("duplicate expectation {}", expectation.name()));
            }
            if matches!(expectation, Self::WeightVersion { value } if value.is_empty()) {
                return Err("expected weight version must not be empty".into());
            }
        }
        Ok(())
    }

    fn satisfied(&self, result: &Value) -> bool {
        let meta = &result["meta_info"];
        let value = &meta[self.name()];
        // Missing is not equivalent to an explicit null/default value.
        if meta.get(self.name()).is_none() {
            return false;
        }
        match self {
            Self::CachedTokens { positive } | Self::Retractions { positive } => value
                .as_u64()
                .is_some_and(|n| if *positive { n > 0 } else { n == 0 }),
            Self::CacheDetails { positive: false } => value.is_null(),
            Self::CacheDetails { positive: true } => value
                .as_object()
                .is_some_and(|m| m.values().any(|v| v.as_u64().is_some_and(|n| n > 0))),
            Self::DpRank { value: expected } => match expected {
                Some(n) => value.as_u64() == Some(*n),
                None => value.is_null(),
            },
            Self::ReasoningTokens { positive } => value.as_u64().is_some_and(|n| {
                (if *positive { n > 0 } else { n == 0 })
                    && meta["completion_tokens"]
                        .as_u64()
                        .is_some_and(|total| n <= total)
            }),
            Self::Timestamp => value.as_f64().is_some_and(|n| n.is_finite() && n > 0.0),
            Self::WeightVersion { value: version } => {
                value.as_str() == Some(version)
                    && meta["weight_versions"].as_array().is_some_and(|spans| {
                        let mut end = 0;
                        !spans.is_empty()
                            && spans.iter().all(|span| {
                                let valid = span["version"].as_str() == Some(version)
                                    && span["start"].as_u64() == Some(end)
                                    && span["end"].as_u64().is_some_and(|n| n > end);
                                end = span["end"].as_u64().unwrap_or(0);
                                valid
                            })
                            && meta["completion_tokens"].as_u64() == Some(end)
                    })
            }
            Self::InputTopLogprobs { populated } => value.as_array().is_some_and(|tokens| {
                if *populated {
                    tokens.iter().any(|token| {
                        token.as_array().is_some_and(|entries| {
                            entries
                                .iter()
                                .any(|entry| entry.get(0).and_then(Value::as_f64).is_some())
                        })
                    })
                } else {
                    tokens.is_empty()
                }
            }),
            Self::OutputLogprobsLength => value.as_u64().is_some_and(|n| {
                n > 0
                    && meta["output_token_logprobs"]
                        .as_array()
                        .is_some_and(|values| values.len() as u64 == n)
            }),
        }
    }

    pub fn evaluate(&self, value: &Value) -> AssertionResult {
        let results: Vec<&Value> = match value.as_array() {
            Some(values) => values.iter().collect(),
            None => vec![value],
        };
        // Retraction selects a subset of a running batch. Do not require every
        // request to be retracted; zero/default controls still check every item.
        let any = matches!(self, Self::Retractions { positive: true });
        let violations = if any && results.iter().any(|result| self.satisfied(result)) {
            Vec::new()
        } else {
            results
                .iter()
                .enumerate()
                .filter(|(_, result)| !self.satisfied(result))
                .map(|(index, result)| {
                    let field = if matches!(self, Self::WeightVersion { value } if result["meta_info"]["weight_version"].as_str() == Some(value)) {
                        "weight_versions"
                    } else { self.name() };
                    Violation::new(
                        format!(
                            "{}/meta_info/{}",
                            if value.is_array() {
                                format!("/{index}")
                            } else {
                                String::new()
                            },
                            field
                        ),
                        format!(
                            "expected {}{}",
                            serde_json::to_string(self).expect("serializable expectation"),
                            if any {
                                " in at least one batch result"
                            } else {
                                ""
                            }
                        ),
                    )
                })
                .collect()
        };
        AssertionResult {
            name: self.name().into(),
            violations,
        }
    }
}
