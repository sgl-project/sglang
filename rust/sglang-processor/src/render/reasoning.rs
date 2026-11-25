//! The effort and thinking toggle SGLang's `ChatCompletionRequest` derives from
//! `thinking`, `reasoning` and `reasoning_effort` (`protocol.py`), shared with hosts.

use serde_json::Value;

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
