//! Scenario checks prove the requested empty/populated state was exercised.
//!
//! Absence and null may both mean a feature is disabled; these assertions never
//! normalize either value. Full parity still distinguishes their wire shapes.

use super::*;
use sglang_parity::runner::AssertionResult;

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "check", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Expectation {
    Logprobs {
        enabled: bool,
        #[serde(default)]
        alternatives: usize,
    },
    TokenIds,
    Reasoning {
        enabled: bool,
    },
    ReasoningTokens {
        positive: bool,
    },
    CachedTokens {
        positive: bool,
    },
    WeightVersion {
        value: String,
    },
    Stop {
        matched: bool,
    },
    Content {
        empty: bool,
    },
    ToolCalls {
        enabled: bool,
        #[serde(default)]
        arguments: Option<Value>,
    },
    NoRefusal,
}

fn absent_or_null(value: Option<&Value>) -> bool {
    value.is_none_or(Value::is_null)
}

impl Expectation {
    pub(super) fn applies_to(&self, check: CheckTarget) -> bool {
        check == CheckTarget::FullResponse
            || matches!(
                self,
                Self::Content { .. }
                    | Self::Reasoning { .. }
                    | Self::ToolCalls { .. }
                    | Self::NoRefusal
            )
    }

    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Logprobs { .. } => "logprobs",
            Self::TokenIds => "token_ids",
            Self::Reasoning { .. } => "reasoning_content",
            Self::ReasoningTokens { .. } => "reasoning_tokens",
            Self::CachedTokens { .. } => "cached_tokens",
            Self::WeightVersion { .. } => "weight_version",
            Self::Stop { .. } => "matched_stop",
            Self::Content { .. } => "content",
            Self::ToolCalls { .. } => "tool_calls",
            Self::NoRefusal => "no_refusal",
        }
    }

    pub(super) fn validate_all(values: &[Self]) -> Result<(), String> {
        let mut names = BTreeSet::new();
        for value in values {
            if !names.insert(value.name()) {
                return Err(format!("duplicate expectation {}", value.name()));
            }
            if matches!(value, Self::ToolCalls { enabled, arguments } if *enabled != arguments.is_some() || arguments.as_ref().is_some_and(|v| !v.is_object()))
                || matches!(value, Self::Logprobs { enabled: false, alternatives } if *alternatives != 0)
                || matches!(value, Self::WeightVersion { value } if value.is_empty())
            {
                return Err(format!("invalid expectation {}", value.name()));
            }
        }
        Ok(())
    }

    pub(super) fn evaluate(
        &self,
        response: &Value,
        case: &HttpCase,
        target: CheckTarget,
    ) -> AssertionResult {
        let mut violations = Vec::new();
        let mut check = |path: String, valid: bool| {
            if !valid {
                violations.push(invalid(
                    &path,
                    &format!(
                        "scenario not exercised: expected {}",
                        serde_json::to_string(self).unwrap()
                    ),
                ));
            }
        };
        match self {
            Self::ReasoningTokens { positive } => {
                check(
                    "/usage/reasoning_tokens".into(),
                    response["usage"]["reasoning_tokens"]
                        .as_u64()
                        .is_some_and(|n| {
                            (if *positive { n > 0 } else { n == 0 })
                                && response["usage"]["completion_tokens"]
                                    .as_u64()
                                    .is_some_and(|total| n <= total)
                        }),
                );
            }
            Self::CachedTokens { positive } => {
                let details = response["usage"].get("prompt_tokens_details");
                check(
                    "/usage/prompt_tokens_details".into(),
                    if *positive {
                        details
                            .and_then(|v| v["cached_tokens"].as_u64())
                            .is_some_and(|n| {
                                n > 0
                                    && response["usage"]["prompt_tokens"]
                                        .as_u64()
                                        .is_some_and(|total| n <= total)
                            })
                    } else {
                        absent_or_null(details)
                            || details.and_then(|v| v["cached_tokens"].as_u64()) == Some(0)
                    },
                );
            }
            Self::WeightVersion { value } => {
                let metadata = &response["metadata"];
                check(
                    "/metadata".into(),
                    metadata["weight_version"] == *value
                        && metadata["weight_versions"].as_array().is_some_and(|spans| {
                            !spans.is_empty() && spans.iter().all(|span| span["version"] == *value)
                        }),
                );
            }
            _ => {
                for (i, choice) in response["choices"].as_array().unwrap().iter().enumerate() {
                    let prefix = format!("/choices/{i}");
                    let key = if case.capture == CaptureMode::Sse {
                        "delta"
                    } else {
                        "message"
                    };
                    let message = &choice[key];
                    let lp = &choice["logprobs"];
                    match self {
                        Self::Logprobs { enabled: false, .. } => check(
                            format!("{prefix}/logprobs"),
                            absent_or_null(choice.get("logprobs")),
                        ),
                        Self::Logprobs {
                            enabled: true,
                            alternatives,
                        } => {
                            let valid_count = |n: usize| {
                                if *alternatives == 0 {
                                    n == 0
                                } else {
                                    n > 0 && n <= *alternatives
                                }
                            };
                            let valid = if is_chat(&case.path) {
                                lp["content"].as_array().is_some_and(|tokens| {
                                    !tokens.is_empty()
                                        && tokens.iter().all(|token| {
                                            token["logprob"].is_number()
                                                && token["top_logprobs"]
                                                    .as_array()
                                                    .is_some_and(|top| valid_count(top.len()))
                                        })
                                })
                            } else {
                                lp["token_logprobs"].as_array().is_some_and(|tokens| {
                                    !tokens.is_empty() && tokens.iter().all(Value::is_number)
                                }) && lp["top_logprobs"].as_array().is_some_and(|tokens| {
                                    (*alternatives == 0 || !tokens.is_empty())
                                        && tokens.iter().all(|top| {
                                            (*alternatives == 0 && top.is_null())
                                                || top
                                                    .as_object()
                                                    .is_some_and(|top| valid_count(top.len()))
                                        })
                                })
                            };
                            check(format!("{prefix}/logprobs"), valid);
                        }
                        Self::TokenIds => check(
                            format!("{prefix}/logprobs/content"),
                            lp["content"].as_array().is_some_and(|tokens| {
                                !tokens.is_empty()
                                    && tokens.iter().all(|token| token["token_id"].is_u64())
                            }),
                        ),
                        Self::Reasoning { enabled } => check(
                            format!("{prefix}/{key}/reasoning_content"),
                            if *enabled {
                                message["reasoning_content"]
                                    .as_str()
                                    .is_some_and(|s| !s.is_empty())
                            } else {
                                absent_or_null(message.get("reasoning_content"))
                                    || message["reasoning_content"] == ""
                            },
                        ),
                        Self::Stop { matched } => check(
                            format!("{prefix}/matched_stop"),
                            if *matched {
                                choice["finish_reason"] == "stop"
                                    && choice.get("matched_stop").is_some_and(|stop| {
                                        !stop.is_null()
                                            && (case.body.get("stop") == Some(stop)
                                                || case.body["stop"]
                                                    .as_array()
                                                    .is_some_and(|stops| stops.contains(stop))
                                                || case.body["stop_token_ids"]
                                                    .as_array()
                                                    .is_some_and(|stops| stops.contains(stop)))
                                    })
                            } else {
                                choice["finish_reason"] == "length"
                                    && choice.get("matched_stop").is_some_and(Value::is_null)
                            },
                        ),
                        Self::Content { empty } => {
                            let value = if is_chat(&case.path) {
                                &message["content"]
                            } else {
                                &choice["text"]
                            };
                            check(
                                format!(
                                    "{prefix}/{}",
                                    if is_chat(&case.path) {
                                        format!("{key}/content")
                                    } else {
                                        "text".into()
                                    }
                                ),
                                value.as_str().is_some_and(|s| s.is_empty() == *empty)
                                    || (*empty && is_chat(&case.path) && value.is_null()),
                            );
                        }
                        Self::ToolCalls { enabled, arguments } => {
                            let calls = message.get("tool_calls");
                            let valid = if *enabled {
                                calls.and_then(Value::as_array).is_some_and(|calls| {
                                    calls.len() == 1 && {
                                        let call = &calls[0];
                                        let expected = &case.body["tools"][0]["function"];
                                        let args = call["function"]["arguments"]
                                            .as_str()
                                            .and_then(|s| serde_json::from_str::<Value>(s).ok());
                                        (target == CheckTarget::GeneratedContent
                                            || choice["finish_reason"] == "tool_calls")
                                            && call["function"]["name"] == expected["name"]
                                            && args.as_ref() == arguments.as_ref()
                                    }
                                })
                            } else {
                                absent_or_null(calls)
                                    || calls.and_then(Value::as_array).is_some_and(Vec::is_empty)
                            };
                            check(format!("{prefix}/{key}/tool_calls"), valid);
                        }
                        Self::NoRefusal => {
                            check(
                                format!("{prefix}/{key}/refusal"),
                                absent_or_null(message.get("refusal")) || message["refusal"] == "",
                            );
                            if target == CheckTarget::FullResponse {
                                check(
                                    format!("{prefix}/logprobs/refusal"),
                                    absent_or_null(lp.get("refusal"))
                                        || lp["refusal"].as_array().is_some_and(Vec::is_empty),
                                );
                            }
                        }
                        _ => unreachable!("response-wide expectation"),
                    }
                }
            }
        }
        AssertionResult {
            name: self.name().into(),
            violations,
        }
    }
}
