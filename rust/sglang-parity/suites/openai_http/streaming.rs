//! Reconstruct OpenAI deltas by choice identity, not network event boundaries.

use super::*;

#[derive(Default)]
struct Choice {
    value: serde_json::Map<String, Value>,
    finished: bool,
    origins: BTreeMap<String, Vec<usize>>,
    tools: BTreeMap<usize, serde_json::Map<String, Value>>,
}

pub(super) fn reconstruct(
    case: &HttpCase,
    observation: &HttpObservation,
    rules: &StreamingRules,
    check: CheckTarget,
) -> Result<PreparedResponse, Violation> {
    let content_only = check == CheckTarget::GeneratedContent;
    let chat = is_chat(&case.path);
    let count = choice_count(&case.body, chat)?;
    let mut choices: BTreeMap<usize, Choice> = BTreeMap::new();
    let mut result = serde_json::Map::new();
    let mut origins = BTreeMap::new();
    let mut done = false;
    let mut final_usage = false;
    let mut usage_by_choices: BTreeMap<Vec<u64>, Value> = BTreeMap::new();
    let include_usage = case.body.pointer("/stream_options/include_usage") == Some(&json!(true));
    let continuous =
        case.body.pointer("/stream_options/continuous_usage_stats") == Some(&json!(true));
    for (event_index, event) in observation.events.iter().enumerate() {
        let process = (|| {
            if done {
                return Err(invalid("", "event after [DONE]"));
            }
            if !(matches!(event.event.as_str(), "" | "message")
                || content_only && event.event == "sglext_ids")
            {
                return Err(invalid("", "SSE event type rule not covered"));
            }
            if event.data == "[DONE]" {
                done = true;
                return Ok(());
            }
            let value: Value =
                serde_json::from_str(&event.data).map_err(|_| invalid("", "invalid event JSON"))?;
            if !value.is_object() {
                return Err(invalid("", "expected response object"));
            }
            if value.get("error").is_some() || value["object"] == "error" {
                return Err(invalid("/error", "in-band error"));
            }
            if !content_only {
                envelope(&value, case, true)?;
            }
            let entries = value["choices"]
                .as_array()
                .ok_or_else(|| invalid("/choices", "expected choices array"))?;
            for (key, item) in value.as_object().unwrap() {
                if key == "choices" || content_only {
                    continue;
                }
                let rule = rules
                    .envelope
                    .get(key)
                    .ok_or_else(|| invalid(&format!("/{key}"), "streaming rule not covered"))?;
                if *rule == Rule::Usage {
                    if !item.is_null() {
                        usage(item)?;
                        if !entries.is_empty() {
                            let mut scope: Vec<_> =
                                entries.iter().filter_map(|c| c["index"].as_u64()).collect();
                            scope.sort_unstable();
                            if let Some(previous) = usage_by_choices.insert(scope, item.clone())
                                && (previous["prompt_tokens"] != item["prompt_tokens"]
                                    || previous["completion_tokens"].as_u64()
                                        > item["completion_tokens"].as_u64())
                            {
                                return Err(invalid(
                                    "/usage",
                                    "continuous usage changed prompt count or decreased completion count",
                                ));
                            }
                        }
                        if entries.is_empty() {
                            if !include_usage
                                || final_usage
                                || choices.len() != count
                                || choices.values().any(|c| !c.finished)
                            {
                                return Err(invalid(
                                    "/usage",
                                    "unexpected, duplicate or premature final usage",
                                ));
                            }
                            final_usage = true;
                        } else if !continuous {
                            return Err(invalid("/usage", "unsolicited continuous usage"));
                        }
                        result.insert(key.clone(), item.clone());
                        origins.insert(format!("/{key}"), vec![event_index]);
                    } else if !result.contains_key(key) {
                        result.insert(key.clone(), Value::Null);
                        origins.insert(format!("/{key}"), vec![event_index]);
                    }
                } else {
                    if *rule == Rule::ChatConstant && !chat {
                        // Completion IDs may vary across chunks. Retain the first
                        // as evidence; choice indices still identify the results.
                        result.entry(key).or_insert_with(|| item.clone());
                    } else {
                        constant(&mut result, key, item, &format!("/{key}"))?;
                    }
                    origins
                        .entry(format!("/{key}"))
                        .or_insert_with(|| vec![event_index]);
                }
            }
            if !content_only
                && entries.is_empty()
                && !value.get("usage").is_some_and(Value::is_object)
            {
                return Err(invalid("/choices", "empty choices requires final usage"));
            }
            let mut seen = BTreeSet::new();
            for entry in entries {
                let index = entry["index"]
                    .as_u64()
                    .filter(|i| *i < count as u64)
                    .ok_or_else(|| invalid("/choices/index", "choice index out of range"))?
                    as usize;
                if !seen.insert(index) {
                    return Err(invalid("/choices/index", "duplicate choice within event"));
                }
                if content_only {
                    let wrong_containers = if chat {
                        ["message", "text"]
                    } else {
                        ["message", "delta"]
                    };
                    for key in wrong_containers {
                        if entry.get(key).is_some_and(|value| !value.is_null()) {
                            return Err(invalid(
                                &format!("/choices/{index}/{key}"),
                                "unexpected generated content container in stream",
                            ));
                        }
                    }
                }
                let state = choices.entry(index).or_default();
                if state.finished {
                    if content_only
                        && entry["finish_reason"].is_null()
                        && !has_generated_content(entry, chat)?
                    {
                        continue;
                    }
                    return Err(invalid(
                        &format!("/choices/{index}"),
                        "choice output after termination",
                    ));
                }
                let terminal = entry["finish_reason"].is_string();
                if content_only {
                    successful_finish(&entry["finish_reason"])?;
                }
                if let Some(reason) = entry.get("finish_reason")
                    && !reason.is_null()
                    && reason.as_str().is_none_or(|s| s.is_empty())
                {
                    return Err(invalid("/choices/finish_reason", "invalid finish reason"));
                }
                let fields = if chat { &rules.chat } else { &rules.completion };
                let object = entry
                    .as_object()
                    .ok_or_else(|| invalid("/choices", "expected choice object"))?;
                for (key, item) in object {
                    if content_only
                        && !matches!(key.as_str(), "index" | "text" | "delta" | "finish_reason")
                    {
                        continue;
                    }
                    let path = format!("/choices/{index}/{key}");
                    match fields
                        .get(key)
                        .ok_or_else(|| invalid(&path, "streaming rule not covered"))?
                    {
                        Rule::Index | Rule::Constant => {
                            constant(&mut state.value, key, item, &path)?
                        }
                        Rule::Text => append_text(&mut state.value, key, item, &path)?,
                        Rule::Terminal => {
                            if !terminal && !item.is_null() {
                                return Err(invalid(&path, "terminal field before finish"));
                            }
                            state.value.insert(key.clone(), item.clone());
                        }
                        Rule::Logprobs => {
                            logprobs(item, chat)?;
                            merge_logprobs(&mut state.value, key, item);
                        }
                        Rule::Delta => {
                            let delta = item
                                .as_object()
                                .ok_or_else(|| invalid(&path, "expected delta object"))?;
                            let target = state
                                .value
                                .entry(key)
                                .or_insert_with(|| json!({}))
                                .as_object_mut()
                                .unwrap();
                            for (field, value) in delta {
                                if content_only
                                    && !matches!(
                                        field.as_str(),
                                        "content"
                                            | "reasoning_content"
                                            | "refusal"
                                            | "tool_calls"
                                            | "function_call"
                                    )
                                {
                                    continue;
                                }
                                let path = format!("{path}/{field}");
                                match rules
                                    .delta
                                    .get(field)
                                    .ok_or_else(|| invalid(&path, "delta rule not covered"))?
                                {
                                    Rule::Text => append_text(target, field, value, &path)?,
                                    Rule::ToolCalls => {
                                        merge_tools(&mut state.tools, value, &path, content_only)?;
                                        // Preserve explicit null/empty arrays even when no calls arrive.
                                        target.entry(field).or_insert_with(|| value.clone());
                                    }
                                    Rule::Constant => {
                                        if field == "role"
                                            && value.is_null()
                                            && target.contains_key(field)
                                        {
                                            continue;
                                        }
                                        if field == "function_call" && !value.is_null() {
                                            return Err(invalid(
                                                &path,
                                                "tool generation is outside this suite",
                                            ));
                                        }
                                        if field == "role"
                                            && !value.is_null()
                                            && value != "assistant"
                                        {
                                            return Err(invalid(&path, "expected assistant role"));
                                        }
                                        if field == "role"
                                            && target.get(field).is_some_and(Value::is_null)
                                        {
                                            target.remove(field);
                                        }
                                        constant(target, field, value, &path)?;
                                    }
                                    _ => unreachable!("validated delta rule"),
                                }
                                state.origins.entry(path).or_default().push(event_index);
                            }
                        }
                        Rule::Usage | Rule::ChatConstant | Rule::ToolCalls => {
                            unreachable!("validated choice rule")
                        }
                    }
                    state.origins.entry(path).or_default().push(event_index);
                }
                // Role-only and empty terminal deltas are control events.
                let content = if chat {
                    &entry["delta"]["content"]
                } else {
                    &entry["text"]
                };
                let has_content = content.as_str().is_some_and(|s| !s.is_empty())
                    || (chat
                        && (["reasoning_content", "refusal"]
                            .iter()
                            .any(|k| entry["delta"][*k].as_str().is_some_and(|s| !s.is_empty()))
                            || entry["delta"]["tool_calls"]
                                .as_array()
                                .is_some_and(|a| !a.is_empty())));
                if !content_only
                    && continuous
                    && has_content
                    && !value.get("usage").is_some_and(Value::is_object)
                {
                    return Err(invalid("/usage", "requested continuous usage missing"));
                }
                state.finished = terminal;
            }
            Ok(())
        })();
        process.map_err(|mut error| {
            error.event = Some(event_index);
            error
        })?;
    }
    if !done || choices.len() != count || choices.values().any(|c| !c.finished) {
        return Err(invalid("", "stream ended without all choices and [DONE]"));
    }
    if !content_only && include_usage && !final_usage {
        return Err(invalid("/usage", "requested final usage missing"));
    }
    let mut values = Vec::new();
    for (index, mut state) in choices {
        if !state.tools.is_empty() {
            if state.tools.keys().copied().ne(0..state.tools.len()) {
                return Err(invalid(
                    &format!("/choices/{index}/delta/tool_calls"),
                    "non-contiguous tool indices",
                ));
            }
            state.value.get_mut("delta").unwrap()["tool_calls"] =
                Value::Array(state.tools.into_values().map(Value::Object).collect());
        }
        let value = Value::Object(state.value);
        if chat {
            assistant_message(
                &value["delta"],
                &format!("/choices/{index}/delta"),
                content_only,
            )?;
        } else if !value["text"].is_string() {
            return Err(invalid(
                &format!("/choices/{index}/text"),
                "missing completion text",
            ));
        }
        if !content_only
            && requested_logprobs(case)
            && !value.get("logprobs").is_some_and(Value::is_object)
        {
            return Err(invalid(
                &format!("/choices/{index}/logprobs"),
                "requested logprobs missing",
            ));
        }
        origins.extend(state.origins);
        values.push(value);
    }
    result.insert("choices".into(), Value::Array(values));
    Ok(PreparedResponse {
        value: Value::Object(result),
        origins,
        assertions: Vec::new(),
        equivalence: None,
    })
}

fn constant(
    target: &mut serde_json::Map<String, Value>,
    key: &str,
    value: &Value,
    path: &str,
) -> Result<(), Violation> {
    if target.get(key).is_some_and(|previous| previous != value) {
        return Err(invalid(path, "constant field changed"));
    }
    target.insert(key.into(), value.clone());
    Ok(())
}

/// Metadata may follow termination, but malformed or further content may not.
fn has_generated_content(entry: &Value, chat: bool) -> Result<bool, Violation> {
    if !chat {
        return match entry.get("text") {
            None | Some(Value::Null) => Ok(false),
            Some(Value::String(text)) => Ok(!text.is_empty()),
            _ => Err(invalid("/choices/text", "expected text delta")),
        };
    }
    let Some(delta) = entry.get("delta") else {
        return Ok(false);
    };
    let delta = delta
        .as_object()
        .ok_or_else(|| invalid("/choices/delta", "expected delta object"))?;
    let mut content = false;
    for (key, value) in delta {
        let path = format!("/choices/delta/{key}");
        match key.as_str() {
            "content" | "reasoning_content" | "refusal" if !value.is_null() => {
                content |= !value
                    .as_str()
                    .ok_or_else(|| invalid(&path, "expected string or null delta"))?
                    .is_empty();
            }
            "tool_calls" if !value.is_null() => {
                content |= !value
                    .as_array()
                    .ok_or_else(|| invalid(&path, "expected tool call array"))?
                    .is_empty();
            }
            "function_call" if !value.is_null() => {
                return Err(invalid(
                    &path,
                    "legacy function_call content is not supported",
                ));
            }
            _ => {}
        }
    }
    Ok(content)
}

fn append_text(
    target: &mut serde_json::Map<String, Value>,
    key: &str,
    value: &Value,
    path: &str,
) -> Result<(), Violation> {
    if value.is_null() {
        target.entry(key).or_insert(Value::Null);
        return Ok(());
    }
    let text = value
        .as_str()
        .ok_or_else(|| invalid(path, "expected string or null delta"))?;
    let previous = target.entry(key).or_insert(Value::Null);
    if previous.is_null() {
        *previous = json!("");
    }
    let Value::String(combined) = previous else {
        return Err(invalid(path, "invalid accumulated text"));
    };
    combined.push_str(text);
    Ok(())
}

fn merge_logprobs(target: &mut serde_json::Map<String, Value>, key: &str, value: &Value) {
    if value.is_null() {
        target.entry(key).or_insert(Value::Null);
        return;
    }
    let previous = target.entry(key).or_insert(Value::Null);
    if previous.is_null() {
        *previous = json!({});
    }
    let object = previous.as_object_mut().unwrap();
    for (field, values) in value.as_object().unwrap() {
        if values.is_null() {
            object.entry(field).or_insert(Value::Null);
            continue;
        }
        let entries = object.entry(field).or_insert(Value::Null);
        if entries.is_null() {
            *entries = json!([]);
        }
        entries
            .as_array_mut()
            .unwrap()
            .extend(values.as_array().unwrap().iter().cloned());
    }
}

/// The wire index routes argument fragments; it is retained in the full result.
fn merge_tools(
    target: &mut BTreeMap<usize, serde_json::Map<String, Value>>,
    value: &Value,
    path: &str,
    content_only: bool,
) -> Result<(), Violation> {
    if value.is_null() {
        return Ok(());
    }
    for call in value
        .as_array()
        .ok_or_else(|| invalid(path, "expected tool call array"))?
    {
        let index = call["index"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| invalid(path, "missing tool index"))?;
        let call = call
            .as_object()
            .ok_or_else(|| invalid(path, "expected tool call object"))?;
        let state = target.entry(index).or_default();
        for (key, value) in call {
            if content_only && !matches!(key.as_str(), "index" | "type" | "function") {
                continue;
            }
            let path = format!("{path}/{index}/{key}");
            match key.as_str() {
                "index" => constant(state, key, value, &path)?,
                "id" | "type" => {
                    if !value.is_null() {
                        if value.as_str().is_none_or(str::is_empty) {
                            return Err(invalid(&path, "expected nonempty tool identity"));
                        }
                        constant(state, key, value, &path)?;
                    }
                }
                "function" => {
                    let function = value
                        .as_object()
                        .ok_or_else(|| invalid(&path, "expected function object"))?;
                    let output = state
                        .entry(key)
                        .or_insert_with(|| json!({}))
                        .as_object_mut()
                        .unwrap();
                    for (field, value) in function {
                        if !matches!(field.as_str(), "name" | "arguments") {
                            if content_only {
                                continue;
                            }
                            return Err(invalid(
                                &format!("{path}/{field}"),
                                "tool function rule not covered",
                            ));
                        }
                        append_text(output, field, value, &format!("{path}/{field}"))?;
                    }
                }
                _ => return Err(invalid(&path, "tool call rule not covered")),
            }
        }
    }
    Ok(())
}
