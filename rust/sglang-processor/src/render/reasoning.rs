//! The effort and thinking toggle SGLang's `ChatCompletionRequest` derives from
//! `thinking`, `reasoning` and `reasoning_effort` (`protocol.py`), shared with hosts.

use std::collections::HashMap;

use serde_json::Value;

use super::thinking::{ThinkingPolicy, has_explicit_enable_thinking_default_false};

/// Effort conventions that SGLang applies outside the Jinja template.
#[derive(Clone, Copy, Default)]
pub enum JinjaEffort {
    #[default]
    None,
    Hunyuan,
    Nemotron,
}

impl JinjaEffort {
    pub(super) fn detect(template: &str, policy: ThinkingPolicy) -> Self {
        // Earlier rules in parser/template_detection.py take precedence.
        if matches!(
            policy,
            ThinkingPolicy::Always | ThinkingPolicy::ReasoningEffort
        ) || ["<ifm|think>", "<ifm|think_fast>", "reasoning_effort"]
            .iter()
            .all(|marker| template.contains(marker))
        {
            Self::None
        } else if ["reasoning_effort", "reasoning_mode_token", "no_think"]
            .iter()
            .all(|marker| template.contains(marker))
        {
            Self::Hunyuan
        } else if template.contains("low_effort")
            && template.contains("truncate_history_thinking")
            && !has_explicit_enable_thinking_default_false(template)
        {
            Self::Nemotron
        } else {
            Self::None
        }
    }

    pub(super) fn apply(
        self,
        body: &Value,
        defaults: &HashMap<String, Value>,
        kwargs: &mut HashMap<String, Value>,
    ) {
        let own_effort = body["chat_template_kwargs"].get("reasoning_effort");
        let default_effort = || defaults.get("reasoning_effort");
        match self {
            Self::None => {}
            Self::Hunyuan => {
                // normalize_hunyuan_reasoning_effort reads the normalized root
                // effort first, followed by the merged template kwargs.
                let effort = requested_effort(body)
                    .or(own_effort)
                    .or_else(default_effort)
                    .unwrap_or(&Value::Null);
                let normalized = match (effort.is_null(), effort.as_str()) {
                    (true, _) | (_, Some("medium" | "high" | "xhigh" | "max")) => "high",
                    (_, Some("none" | "no_think")) => "no_think",
                    (_, Some("minimal" | "low")) => "low",
                    // The renderer does not know the worker's parser. Keep
                    // the generic kwargs so templates that accept other values
                    // can supply routing tokens. A matching Hunyuan worker
                    // still validates the effort before using forwarded IDs.
                    _ => return,
                };
                kwargs.insert("reasoning_effort".into(), normalized.into());
            }
            Self::Nemotron => {
                // _apply_jinja_template checks the request effort after
                // _convert_to_internal_request has read the nested override.
                let effort = own_effort
                    .filter(|effort| !effort.is_null())
                    .or_else(|| requested_effort(body))
                    .or_else(default_effort);
                if effort.is_some_and(|effort| effort == "low") {
                    kwargs.entry("low_effort".into()).or_insert(true.into());
                }
            }
        }
    }
}

/// `protocol.ReasoningEffortTier`.
pub(crate) const EFFORT_TIERS: [&str; 8] = [
    "none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max",
];

/// The request's effort, in the order used by `normalize_reasoning_inputs`.
pub fn requested_effort(request: &Value) -> Option<&Value> {
    let reasoning = &request["reasoning"];
    [
        &request["thinking"]["effort"],
        &reasoning["effort"],
        &reasoning["reasoning_effort"],
        &request["reasoning_effort"],
    ]
    .into_iter()
    .find(|effort| !effort.is_null())
}

/// The thinking toggle used to fill unset template kwargs in
/// `ChatCompletionRequest.normalize_reasoning_inputs`.
pub fn requested_thinking(request: &Value) -> Option<bool> {
    let thinking = &request["thinking"];
    if !thinking["type"].is_null() {
        return Some(thinking["type"] != "disabled");
    }
    if let Some(effort) = requested_effort(request) {
        return Some(effort != "none");
    }
    if thinking.is_object() {
        return Some(true);
    }
    let reasoning = &request["reasoning"];
    let enabled = match reasoning
        .get("enabled")
        .filter(|v| !v.is_null())
        .or_else(|| reasoning.get("enable"))?
    {
        Value::String(enabled) => {
            ["1", "true", "yes", "y", "on"].contains(&enabled.trim().to_lowercase().as_str())
        }
        enabled => minijinja::Value::from_serialize(enabled).is_true(),
    };
    enabled.then_some(true)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn root_thinking_takes_precedence() {
        let mut body = json!({
            "thinking": {"effort": "none"},
            "reasoning": {"effort": "high", "enabled": true},
        });
        assert_eq!(requested_effort(&body), Some(&json!("none")));
        assert_eq!(requested_thinking(&body), Some(false));
        body["thinking"]["type"] = json!("enabled");
        assert_eq!(requested_thinking(&body), Some(true));
        body["thinking"] = json!({"type": "disabled", "effort": "low"});
        assert_eq!(requested_effort(&body), Some(&json!("low")));
        assert_eq!(requested_thinking(&body), Some(false));
    }
}
