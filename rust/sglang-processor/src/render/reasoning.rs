//! The effort and thinking toggle SGLang's `ChatCompletionRequest` derives from
//! `reasoning` and `reasoning_effort` (`protocol.py`), shared with hosts.

use serde_json::Value;

/// The request's effort: `reasoning.effort`, then `reasoning.reasoning_effort`,
/// then the top-level `reasoning_effort`.
pub fn requested_effort(request: &Value) -> Option<&Value> {
    let reasoning = &request["reasoning"];
    [
        &reasoning["effort"],
        &reasoning["reasoning_effort"],
        &request["reasoning_effort"],
    ]
    .into_iter()
    .find(|effort| !effort.is_null())
}

/// The thinking toggle the request implies: an effort turns it off only when it
/// is `"none"`; without one, a truthy `reasoning.enabled` turns it on.
pub fn requested_thinking(request: &Value) -> Option<bool> {
    if let Some(effort) = requested_effort(request) {
        return Some(effort != "none");
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
