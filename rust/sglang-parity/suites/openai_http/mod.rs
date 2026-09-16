//! OpenAI HTTP contracts and explicit JSON/SSE semantic equivalence.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sglang_parity::compare::ComparisonRules;
use sglang_parity::{
    CaptureMode, ComparisonScope, EquivalenceValue, ExecutionPlan, HttpCase, HttpObservation,
    HttpRequest, HttpSuite, Isolation, PreparedResponse, ProfilePlan, ResponsePolicy, RunConfig,
    Violation,
};

mod streaming;
#[cfg(test)]
mod tests;

pub const DEFAULT_SPEC: &str = include_str!("suite.json");

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Specification {
    name: String,
    comparison: ComparisonRules,
    streaming: StreamingRules,
    equivalence: Vec<String>,
    cases: Vec<Case>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    path: String,
    profiles: Vec<String>,
    body: Value,
    expect_status: u16,
    #[serde(default)]
    equivalence_group: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StreamingRules {
    envelope: BTreeMap<String, Rule>,
    completion: BTreeMap<String, Rule>,
    chat: BTreeMap<String, Rule>,
    delta: BTreeMap<String, Rule>,
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Rule {
    Constant,
    Text,
    Logprobs,
    Terminal,
    Index,
    Delta,
    Usage,
}

pub struct OpenAiPolicy {
    rules: StreamingRules,
}

const SEMANTICS: [&str; 5] = ["model", "content", "finish_reason", "logprobs", "usage"];

pub fn load_plan(text: &str, config: &RunConfig) -> Result<ExecutionPlan<OpenAiPolicy>, String> {
    let spec: Specification = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if spec.name != "openai_http" || spec.equivalence != SEMANTICS {
        return Err("openai_http requires its declared model/content/finish_reason/logprobs/usage projection".into());
    }
    // The finite rule vocabulary documents the supported contract, not a DSL.
    let defaults: Specification = serde_json::from_str(DEFAULT_SPEC).map_err(|e| e.to_string())?;
    if serde_json::to_value(&spec.streaming).unwrap()
        != serde_json::to_value(&defaults.streaming).unwrap()
    {
        return Err(
            "unsupported OpenAI streaming rules; extend the policy and its tests first".into(),
        );
    }
    spec.comparison.validate()?;
    let profiles = config.resolve_profiles()?;
    let mut names = BTreeSet::new();
    let mut groups: BTreeMap<(&str, &str), Vec<&Case>> = BTreeMap::new();
    for case in &spec.cases {
        if !names.insert(&case.name) || !case.body.is_object() {
            return Err(format!(
                "{}: duplicate name or non-object request",
                case.name
            ));
        }
        if !matches!(
            case.path.as_str(),
            "/v1/completions" | "/v1/chat/completions"
        ) {
            return Err(format!("{}: unsupported endpoint", case.name));
        }
        let selected: BTreeSet<_> = case.profiles.iter().collect();
        if selected.is_empty()
            || selected.len() != case.profiles.len()
            || selected
                .iter()
                .any(|id| !profiles.iter().any(|p| &p.id == *id))
        {
            return Err(format!(
                "{}: unknown, duplicate or empty profiles",
                case.name
            ));
        }
        if case.expect_status != 200 && !(400..500).contains(&case.expect_status) {
            return Err(format!("{}: expected status must be 200 or 4xx", case.name));
        }
        if case.expect_status == 200 {
            choice_count(&case.body, is_chat(&case.path)).map_err(|e| e.message)?;
            if !case.body.get("stream").is_some_and(Value::is_boolean) {
                return Err(format!("{}: stream must be explicit", case.name));
            }
        }
        if let Some(group) = &case.equivalence_group {
            if group.is_empty() || case.expect_status != 200 {
                return Err("equivalence groups require successful generation cases".into());
            }
            for profile in &case.profiles {
                groups.entry((profile, group)).or_default().push(case);
            }
        }
    }
    for ((_, group), members) in groups {
        if members.len() != 2 || members[0].path != members[1].path {
            return Err(format!(
                "{group}: expected one JSON/SSE pair on the same endpoint"
            ));
        }
        let mut bodies = Vec::new();
        let mut modes = BTreeSet::new();
        for case in members {
            let mut body = case.body.clone();
            let object = body.as_object_mut().unwrap();
            let stream = object.remove("stream").unwrap();
            modes.insert(stream.as_bool().unwrap());
            if stream == true
                && case.body.pointer("/stream_options/include_usage") != Some(&json!(true))
            {
                return Err(format!(
                    "{group}: streaming equivalence requires final usage"
                ));
            }
            object.remove("stream_options");
            bodies.push(body);
        }
        if modes.len() != 2 || bodies[0] != bodies[1] {
            return Err(format!(
                "{group}: only stream and stream_options may differ"
            ));
        }
    }
    let mut plans = Vec::new();
    for profile in profiles {
        let mut cases = Vec::new();
        let mut model = profile.server.model.clone();
        let mut args = profile.server.args.iter();
        while let Some(arg) = args.next() {
            if arg == "--served-model-name" {
                model = args
                    .next()
                    .ok_or("--served-model-name requires a value")?
                    .clone();
            } else if let Some(value) = arg.strip_prefix("--served-model-name=") {
                model = value.to_owned();
            }
        }
        for case in spec
            .cases
            .iter()
            .filter(|c| c.profiles.contains(&profile.id))
        {
            let mut body = case.body.clone();
            body.as_object_mut()
                .unwrap()
                .entry("model")
                .or_insert(json!(model));
            let capture = if case.expect_status == 200 && body["stream"] == true {
                CaptureMode::Sse
            } else {
                CaptureMode::Json
            };
            cases.push(HttpCase {
                name: case.name.clone(),
                request: HttpRequest {
                    method: "POST".into(),
                    path: case.path.clone(),
                    body,
                    expect_status: case.expect_status,
                    capture,
                    comparison_scope: ComparisonScope::Root,
                },
                equivalence_group: case.equivalence_group.clone(),
                assertions: Vec::new(),
                before_each: Vec::new(),
                isolation: Isolation::Shared,
                requires: Default::default(),
            });
        }
        if cases.is_empty() {
            continue;
        }
        let incremental = profile.server.incremental_output();
        let suite = HttpSuite {
            name: spec.name.clone(),
            response_implementation: "openai_http".into(),
            output_mode: if incremental {
                "incremental"
            } else {
                "cumulative"
            }
            .into(),
            response_policy: Some(json!({
                "streaming": spec.streaming,
                "wire_content": "OpenAI SSE delta, independent of backend output mode",
                "equivalence": spec.equivalence,
            })),
            comparison: spec.comparison.clone(),
            cases,
        };
        plans.push(ProfilePlan {
            profile,
            suite,
            policy: OpenAiPolicy {
                rules: spec.streaming.clone(),
            },
        });
    }
    let plan = ExecutionPlan { profiles: plans };
    plan.validate()?;
    Ok(plan)
}

fn is_chat(path: &str) -> bool {
    path == "/v1/chat/completions"
}

fn invalid(path: &str, message: &str) -> Violation {
    Violation::new(path, message)
}

fn choice_count(body: &Value, chat: bool) -> Result<usize, Violation> {
    let n = body
        .get("n")
        .map_or(Some(1), Value::as_u64)
        .filter(|n| (1..=4096).contains(n))
        .ok_or_else(|| invalid("/n", "expected positive n <= 4096"))? as usize;
    let prompts = if chat {
        1
    } else {
        match body.get("prompt") {
            Some(Value::String(_)) => 1,
            Some(Value::Array(values)) if !values.is_empty() => {
                if values.iter().all(Value::is_u64) {
                    1
                } else if values.iter().all(|v| {
                    v.is_string()
                        || v.as_array()
                            .is_some_and(|a| !a.is_empty() && a.iter().all(Value::is_u64))
                }) {
                    values.len()
                } else {
                    return Err(invalid("/prompt", "invalid prompt batch"));
                }
            }
            _ => return Err(invalid("/prompt", "expected prompt")),
        }
    };
    n.checked_mul(prompts)
        .filter(|n| *n <= 4096)
        .ok_or_else(|| invalid("/choices", "too many choices"))
}

fn envelope(value: &Value, case: &HttpCase, stream: bool) -> Result<(), Violation> {
    if !value.is_object() {
        return Err(invalid("", "expected response object"));
    }
    for key in ["id", "model"] {
        if value[key].as_str().is_none_or(|s| s.is_empty()) {
            return Err(invalid(&format!("/{key}"), "expected nonempty string"));
        }
    }
    if !value["created"].is_u64() {
        return Err(invalid("/created", "expected nonnegative integer"));
    }
    let object = if is_chat(&case.path) {
        if stream {
            "chat.completion.chunk"
        } else {
            "chat.completion"
        }
    } else {
        "text_completion"
    };
    if value["object"] != object {
        return Err(invalid("/object", "unexpected response object type"));
    }
    if value["model"] != case.body["model"] {
        return Err(invalid("/model", "model differs from request"));
    }
    Ok(())
}

fn usage(value: &Value) -> Result<(), Violation> {
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        if !value[key].is_u64() {
            return Err(invalid(
                &format!("/usage/{key}"),
                "expected nonnegative integer",
            ));
        }
    }
    if value["prompt_tokens"]
        .as_u64()
        .unwrap()
        .checked_add(value["completion_tokens"].as_u64().unwrap())
        != value["total_tokens"].as_u64()
    {
        return Err(invalid(
            "/usage/total_tokens",
            "total must equal prompt + completion",
        ));
    }
    Ok(())
}

fn logprobs(value: &Value, chat: bool) -> Result<(), Violation> {
    if value.is_null() {
        return Ok(());
    }
    let object = value
        .as_object()
        .ok_or_else(|| invalid("/logprobs", "expected object or null"))?;
    if chat {
        for (key, entries) in object {
            if !matches!(key.as_str(), "content" | "refusal") {
                return Err(invalid(
                    &format!("/logprobs/{key}"),
                    "logprob rule not covered",
                ));
            }
            if entries.is_null() {
                continue;
            }
            for entry in entries
                .as_array()
                .ok_or_else(|| invalid("/logprobs", "expected token array"))?
            {
                if !entry["token"].is_string()
                    || !entry["logprob"].is_number()
                    || !entry["top_logprobs"].is_array()
                {
                    return Err(invalid("/logprobs", "invalid token logprob"));
                }
            }
        }
    } else {
        let mut length = None;
        for (key, entries) in object {
            if !matches!(
                key.as_str(),
                "tokens" | "token_logprobs" | "top_logprobs" | "text_offset"
            ) {
                return Err(invalid(
                    &format!("/logprobs/{key}"),
                    "logprob rule not covered",
                ));
            }
            if entries.is_null() {
                continue;
            }
            let array = entries
                .as_array()
                .ok_or_else(|| invalid("/logprobs", "expected array"))?;
            if length
                .replace(array.len())
                .is_some_and(|n| n != array.len())
            {
                return Err(invalid(
                    "/logprobs",
                    "logprob arrays must have equal lengths",
                ));
            }
            for item in array {
                let valid = match key.as_str() {
                    "tokens" => item.is_string(),
                    "token_logprobs" => item.is_null() || item.is_number(),
                    "text_offset" => item.is_i64(),
                    _ => {
                        item.is_null()
                            || item
                                .as_object()
                                .is_some_and(|o| o.values().all(Value::is_number))
                    }
                };
                if !valid {
                    return Err(invalid("/logprobs", "invalid logprob array item"));
                }
            }
        }
    }
    Ok(())
}

impl ResponsePolicy for OpenAiPolicy {
    fn prepare(
        &self,
        case: &HttpCase,
        observation: &HttpObservation,
    ) -> Result<PreparedResponse, Vec<Violation>> {
        self.prepare_response(case, observation)
            .map_err(|e| vec![e])
    }
}

impl OpenAiPolicy {
    fn prepare_response(
        &self,
        case: &HttpCase,
        observation: &HttpObservation,
    ) -> Result<PreparedResponse, Violation> {
        let mut prepared = if case.capture == CaptureMode::Sse {
            streaming::reconstruct(case, observation, &self.rules)?
        } else {
            let value = observation
                .json
                .clone()
                .ok_or_else(|| invalid("", "JSON evidence unavailable"))?;
            if case.expect_status >= 400 {
                if !value["error"].is_object() || !value["error"]["message"].is_string() {
                    return Err(invalid("/error", "expected structured error with message"));
                }
                return Ok(value.into());
            }
            envelope(&value, case, false)?;
            usage(&value["usage"])?;
            let count = choice_count(&case.body, is_chat(&case.path))?;
            let choices = value["choices"]
                .as_array()
                .ok_or_else(|| invalid("/choices", "expected array"))?;
            let mut indices = BTreeSet::new();
            for choice in choices {
                let index = choice["index"]
                    .as_u64()
                    .filter(|i| *i < count as u64)
                    .ok_or_else(|| invalid("/choices/index", "choice index out of range"))?;
                if !indices.insert(index) {
                    return Err(invalid("/choices/index", "duplicate index"));
                }
                if choice["finish_reason"]
                    .as_str()
                    .is_none_or(|s| s.is_empty())
                {
                    return Err(invalid(
                        "/choices/finish_reason",
                        "expected terminal reason",
                    ));
                }
                if is_chat(&case.path) {
                    if choice["message"]["role"] != "assistant"
                        || !choice["message"]["content"].is_string()
                    {
                        return Err(invalid(
                            "/choices/message",
                            "expected assistant text message",
                        ));
                    }
                } else if !choice["text"].is_string() {
                    return Err(invalid("/choices/text", "expected text"));
                }
                if let Some(value) = choice.get("logprobs") {
                    logprobs(value, is_chat(&case.path))?;
                }
                if requested_logprobs(case) && !choice.get("logprobs").is_some_and(Value::is_object)
                {
                    return Err(invalid("/choices/logprobs", "requested logprobs missing"));
                }
            }
            if indices.len() != count {
                return Err(invalid("/choices", "missing choices"));
            }
            value.into()
        };
        if case.equivalence_group.is_some() {
            prepared.equivalence = Some(project(&prepared, is_chat(&case.path))?);
        }
        Ok(prepared)
    }
}

fn requested_logprobs(case: &HttpCase) -> bool {
    if is_chat(&case.path) {
        case.body["logprobs"] == true
    } else {
        case.body.get("logprobs").is_some_and(|v| !v.is_null())
    }
}

/// A deliberately smaller view; never used for cross-implementation parity.
fn project(response: &PreparedResponse, chat: bool) -> Result<EquivalenceValue, Violation> {
    let value = &response.value;
    let mut choices: Vec<_> = value["choices"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .collect();
    choices.sort_by_key(|(_, c)| c["index"].as_u64());
    let mut origins = BTreeMap::new();
    let mut projected = Vec::new();
    for (position, (source, choice)) in choices.into_iter().enumerate() {
        let content_key = if chat {
            if choice.get("message").is_some() {
                "message"
            } else {
                "delta"
            }
        } else {
            "text"
        };
        let content = if chat {
            &choice[content_key]["content"]
        } else {
            &choice[content_key]
        };
        let mut result = json!({"index": choice["index"], "content": content, "finish_reason": choice["finish_reason"]});
        if let Some(logprobs) = choice.get("logprobs") {
            result["logprobs"] = logprobs.clone();
        }
        for (destination, source_key) in [
            ("content", content_key),
            ("finish_reason", "finish_reason"),
            ("logprobs", "logprobs"),
        ] {
            let prefix = format!("/choices/{source}/{source_key}");
            let events: BTreeSet<_> = response
                .origins
                .iter()
                .filter(|(key, _)| *key == &prefix || key.starts_with(&format!("{prefix}/")))
                .flat_map(|(_, indices)| indices.iter().copied())
                .collect();
            if !events.is_empty() {
                origins.insert(
                    format!("/choices/{position}/{destination}"),
                    events.into_iter().collect(),
                );
            }
        }
        projected.push(result);
    }
    if !value.get("usage").is_some_and(Value::is_object) {
        return Err(invalid("/usage", "equivalence requires final usage"));
    }
    for key in ["model", "usage"] {
        if let Some(events) = response.origins.get(&format!("/{key}")) {
            origins.insert(format!("/{key}"), events.clone());
        }
    }
    Ok(EquivalenceValue {
        value: json!({"model":value["model"], "choices":projected, "usage":value["usage"]}),
        origins,
    })
}
