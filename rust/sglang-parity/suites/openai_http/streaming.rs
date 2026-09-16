//! Reconstruct OpenAI deltas by choice identity, not network event boundaries.

use super::*;

#[derive(Default)]
struct Choice {
    value: serde_json::Map<String, Value>,
    finished: bool,
    origins: BTreeMap<String, Vec<usize>>,
}

pub(super) fn reconstruct(
    case: &HttpCase,
    observation: &HttpObservation,
    rules: &StreamingRules,
) -> Result<PreparedResponse, Violation> {
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
            if !matches!(event.event.as_str(), "" | "message") {
                return Err(invalid("", "SSE event type rule not covered"));
            }
            if event.data == "[DONE]" {
                done = true;
                return Ok(());
            }
            let value: Value =
                serde_json::from_str(&event.data).map_err(|_| invalid("", "invalid event JSON"))?;
            if value.get("error").is_some() {
                return Err(invalid("/error", "in-band error"));
            }
            envelope(&value, case, true)?;
            let entries = value["choices"]
                .as_array()
                .ok_or_else(|| invalid("/choices", "expected choices array"))?;
            for (key, item) in value.as_object().unwrap() {
                if key == "choices" {
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
                    constant(&mut result, key, item, &format!("/{key}"))?;
                    origins
                        .entry(format!("/{key}"))
                        .or_insert_with(|| vec![event_index]);
                }
            }
            if entries.is_empty() && !value.get("usage").is_some_and(Value::is_object) {
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
                let state = choices.entry(index).or_default();
                if state.finished {
                    return Err(invalid(
                        &format!("/choices/{index}"),
                        "choice output after termination",
                    ));
                }
                let terminal = entry["finish_reason"].is_string();
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
                                let path = format!("{path}/{field}");
                                match rules
                                    .delta
                                    .get(field)
                                    .ok_or_else(|| invalid(&path, "delta rule not covered"))?
                                {
                                    Rule::Text => append_text(target, field, value, &path)?,
                                    Rule::Constant => {
                                        if field == "role"
                                            && value.is_null()
                                            && target.contains_key(field)
                                        {
                                            continue;
                                        }
                                        if matches!(field.as_str(), "tool_calls" | "function_call")
                                            && !value.is_null()
                                        {
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
                        Rule::Usage => unreachable!("validated choice rule"),
                    }
                    state.origins.entry(path).or_default().push(event_index);
                }
                if continuous && !value.get("usage").is_some_and(Value::is_object) {
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
    if include_usage && !final_usage {
        return Err(invalid("/usage", "requested final usage missing"));
    }
    let mut values = Vec::new();
    for (index, state) in choices {
        let value = Value::Object(state.value);
        if chat {
            if value["delta"]["role"] != "assistant" || !value["delta"]["content"].is_string() {
                return Err(invalid(
                    &format!("/choices/{index}/delta"),
                    "missing assistant text",
                ));
            }
        } else if !value["text"].is_string() {
            return Err(invalid(
                &format!("/choices/{index}/text"),
                "missing completion text",
            ));
        }
        if requested_logprobs(case) && !value.get("logprobs").is_some_and(Value::is_object) {
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
