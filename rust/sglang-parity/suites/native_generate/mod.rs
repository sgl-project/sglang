//! Native `/generate` requests and lossless response validation.
//!
//! The specification owns comparison exceptions. This module only interprets
//! generation frames, checks their lifecycle, and reconstructs the complete
//! final response. Unknown terminal fields survive for the core comparator.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::{Map, Value};
use sglang_parity::compare::{ComparisonRules, ComparisonScope, Violation};
use sglang_parity::http::{CaptureMode, HttpCase, HttpObservation};
use sglang_parity::runner::{HttpSuite, ResponsePolicy, RunConfig};

pub const DEFAULT_SPEC: &str = include_str!("suite.json");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteSpec {
    name: String,
    http: HttpSpec,
    comparison: ComparisonRules,
    cases: Vec<CaseSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpSpec {
    method: String,
    path: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseSpec {
    name: String,
    body: Value,
    expect_status: u16,
    #[serde(default)]
    equivalence_group: Option<String>,
}

/// One interpretation of streaming output, shared by both implementations.
pub struct GeneratePolicy {
    incremental: bool,
}

#[derive(Clone, Copy)]
struct Shape {
    count: usize,
    batch: bool,
}

#[derive(Default)]
struct ResultState {
    last: Option<Value>,
    text: Option<String>,
    ids: Option<Vec<Value>>,
    output: BTreeMap<&'static str, Vec<Value>>,
    input: BTreeMap<&'static str, Vec<Value>>,
    finished: bool,
}

// These are protocol accumulation rules, not comparison exceptions. Every
// other terminal field remains untouched, including newly introduced fields.
const OUTPUT_LOGPROBS: [&str; 3] = [
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
];
const INPUT_LOGPROBS: [&str; 3] = [
    "input_token_logprobs",
    "input_top_logprobs",
    "input_token_ids_logprobs",
];

/// Resolve the same specification for describe, execution, and reporting.
///
/// This performs no I/O or model initialization. Invalid configuration and
/// request shapes whose result count cannot be determined are rejected.
pub fn load(spec: &str, config: &RunConfig) -> Result<(HttpSuite, GeneratePolicy), String> {
    let spec: SuiteSpec = serde_json::from_str(spec).map_err(|error| error.to_string())?;
    if spec.name != "native_generate" {
        return Err("the native_generate implementation requires name=native_generate".into());
    }
    if spec.http.method != "POST" || spec.http.path != "/generate" {
        return Err("native_generate requires POST /generate".into());
    }
    spec.comparison.validate()?;
    if spec.cases.is_empty() {
        return Err("a suite must contain at least one case".into());
    }
    let negative = spec.cases[0].expect_status >= 400;
    if negative && !spec.comparison.per_result_value_exceptions.is_empty() {
        return Err("HTTP error suites require an empty value-exception list".into());
    }
    let mut names = BTreeSet::new();
    let mut cases = Vec::with_capacity(spec.cases.len());
    for case in spec.cases {
        if case.name.trim().is_empty() || !names.insert(case.name.clone()) {
            return Err("case names must be non-empty and unique".into());
        }
        if case
            .equivalence_group
            .as_ref()
            .is_some_and(|group| group.trim().is_empty())
        {
            return Err(format!(
                "{}: equivalence_group must not be empty",
                case.name
            ));
        }
        if !(case.expect_status == 200 || (400..=599).contains(&case.expect_status)) {
            return Err(format!(
                "{}: expect_status must be 200 or 4xx/5xx",
                case.name
            ));
        }
        if (case.expect_status >= 400) != negative {
            return Err(
                "success and HTTP error cases require separate suite specifications".into(),
            );
        }
        if !case.body.is_object() {
            return Err(format!("{}: body must be a JSON object", case.name));
        }
        let (capture, comparison_scope) = if negative {
            (CaptureMode::Json, ComparisonScope::Root)
        } else {
            let shape =
                request_shape(&case.body).map_err(|error| format!("{}: {error}", case.name))?;
            let stream = match case.body.get("stream") {
                None => false,
                Some(Value::Bool(value)) => *value,
                _ => return Err(format!("{}: stream must be a boolean", case.name)),
            };
            (
                if stream {
                    CaptureMode::Sse
                } else {
                    CaptureMode::Json
                },
                if shape.batch {
                    ComparisonScope::TopLevelArrayItems
                } else {
                    ComparisonScope::Root
                },
            )
        };
        cases.push(HttpCase {
            name: case.name,
            method: spec.http.method.clone(),
            path: spec.http.path.clone(),
            body: case.body,
            expect_status: case.expect_status,
            capture,
            comparison_scope,
            equivalence_group: case.equivalence_group,
        });
    }
    let incremental = config.server.incremental_output();
    Ok((
        HttpSuite {
            name: spec.name,
            response_implementation: "native_generate::GeneratePolicy".into(),
            output_mode: if incremental {
                "incremental"
            } else {
                "cumulative"
            }
            .into(),
            comparison: spec.comparison,
            cases,
        },
        GeneratePolicy { incremental },
    ))
}

fn request_shape(body: &Value) -> Result<Shape, String> {
    let (count, batch) = match (body.get("text"), body.get("input_ids")) {
        (Some(Value::String(_)), None | Some(Value::Null)) => (1, false),
        (Some(Value::Array(texts)), None | Some(Value::Null))
            if !texts.is_empty() && texts.iter().all(Value::is_string) =>
        {
            (texts.len(), true)
        }
        (None | Some(Value::Null), Some(Value::Array(ids)))
            if !ids.is_empty() && ids.iter().all(|id| id.as_u64().is_some()) =>
        {
            (1, false)
        }
        (None | Some(Value::Null), Some(Value::Array(rows)))
            if !rows.is_empty()
                && rows.iter().all(|row| {
                    row.as_array().is_some_and(|ids| {
                        !ids.is_empty() && ids.iter().all(|id| id.as_u64().is_some())
                    })
                }) =>
        {
            (rows.len(), true)
        }
        _ => {
            return Err(
                "successful cases require one text/input_ids prompt or a non-empty batch".into(),
            );
        }
    };
    let sampling: Vec<&Value> = match body.get("sampling_params") {
        None | Some(Value::Null) => vec![],
        Some(value @ Value::Object(_)) => vec![value],
        Some(Value::Array(values))
            if values.len() == count && values.iter().all(Value::is_object) =>
        {
            values.iter().collect()
        }
        _ => return Err("sampling_params must be an object or one object per prompt".into()),
    };
    let mut n = None;
    for params in sampling {
        let current = match params.get("n") {
            None => 1,
            Some(value) => value
                .as_u64()
                .filter(|n| *n > 0)
                .ok_or("sampling n must be a positive integer")?,
        };
        if n.is_some_and(|n| n != current) {
            return Err("sampling n must be the same for all prompts".into());
        }
        n = Some(current);
    }
    let n = usize::try_from(n.unwrap_or(1)).map_err(|_| "sampling n is too large")?;
    let count = count
        .checked_mul(n)
        .filter(|count| *count <= 65_536)
        .ok_or("result count exceeds 65536")?;
    Ok(Shape {
        count,
        batch: batch || n > 1,
    })
}

impl ResponsePolicy for GeneratePolicy {
    fn prepare(
        &self,
        case: &HttpCase,
        observation: &HttpObservation,
    ) -> Result<Value, Vec<Violation>> {
        let mut violations = observation.violations.clone();
        if let Some(error) = &observation.transport_error {
            violations.push(Violation::new("", format!("transport failed: {error}")));
        }
        if observation.status != Some(case.expect_status) {
            violations.push(Violation::new(
                "",
                format!(
                    "expected HTTP {}, received {:?}",
                    case.expect_status, observation.status
                ),
            ));
        }
        if !violations.is_empty() {
            return Err(violations);
        }
        if case.expect_status >= 400 {
            let Some(value) = observation.json.as_ref() else {
                return Err(vec![Violation::new(
                    "",
                    "expected a JSON HTTP error response",
                )]);
            };
            if !value.is_object() || !value.get("error").is_some_and(Value::is_object) {
                return Err(vec![Violation::new(
                    "/error",
                    "expected a native error object",
                )]);
            }
            return Ok(value.clone());
        }
        let shape = request_shape(&case.body).map_err(|error| vec![Violation::new("", error)])?;
        match case.capture {
            CaptureMode::Json => prepare_json(case, observation, shape),
            CaptureMode::Sse => self.prepare_stream(case, observation, shape),
        }
    }
}

fn prepare_json(
    case: &HttpCase,
    observation: &HttpObservation,
    shape: Shape,
) -> Result<Value, Vec<Violation>> {
    let Some(value) = observation.json.as_ref() else {
        return Err(vec![Violation::new("", "missing JSON response")]);
    };
    let values = if shape.batch {
        let Some(values) = value.as_array() else {
            return Err(vec![Violation::new("", "expected a batch response array")]);
        };
        if values.len() != shape.count {
            return Err(vec![Violation::new(
                "",
                format!(
                    "expected {} results, received {}",
                    shape.count,
                    values.len()
                ),
            )]);
        }
        values.iter().collect::<Vec<_>>()
    } else {
        vec![value]
    };
    let mut violations = Vec::new();
    for (index, result) in values.iter().enumerate() {
        let path = if shape.batch {
            format!("/{index}")
        } else {
            String::new()
        };
        match validate_frame(result, &path, true) {
            Ok(_) => {
                let mut state = ResultState::default();
                if let Err(error) = state.accept(result, false, &path) {
                    violations.push(error);
                }
                if let Err(error) = validate_requested_logprobs(case, index, result, &path) {
                    violations.push(error);
                }
            }
            Err(error) => violations.push(error),
        }
    }
    if violations.is_empty() {
        Ok(value.clone())
    } else {
        Err(violations)
    }
}

impl GeneratePolicy {
    fn prepare_stream(
        &self,
        case: &HttpCase,
        observation: &HttpObservation,
        shape: Shape,
    ) -> Result<Value, Vec<Violation>> {
        let content_type = observation
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("content-type"))
            .map(|(_, value)| value.split(';').next().unwrap_or("").trim());
        if !content_type.is_some_and(|value| value.eq_ignore_ascii_case("text/event-stream")) {
            return Err(vec![Violation::new(
                "",
                "successful streams require text/event-stream",
            )]);
        }
        let mut states: Vec<ResultState> =
            (0..shape.count).map(|_| ResultState::default()).collect();
        let mut done = false;
        let mut violations = Vec::new();
        for (event_index, event) in observation.events.iter().enumerate() {
            let result = (|| {
                if done {
                    return Err(Violation::new("", "data event after [DONE]"));
                }
                if event.event != "message" && !event.event.is_empty() {
                    return Err(Violation::new(
                        "",
                        "native generation requires message events",
                    ));
                }
                if event.data == "[DONE]" {
                    done = true;
                    if states.iter().any(|state| !state.finished) {
                        return Err(Violation::new(
                            "",
                            "[DONE] arrived before every result terminated",
                        ));
                    }
                    return Ok(());
                }
                let value: Value = serde_json::from_str(&event.data)
                    .map_err(|error| Violation::new("", format!("invalid event JSON: {error}")))?;
                let index = if shape.batch {
                    value
                        .get("index")
                        .and_then(Value::as_u64)
                        .and_then(|index| usize::try_from(index).ok())
                        .filter(|index| *index < shape.count)
                        .ok_or_else(|| {
                            Violation::new("/index", "missing or out-of-range batch index")
                        })?
                } else {
                    if value.get("index").is_some() {
                        return Err(Violation::new(
                            "/index",
                            "single-result streams must omit index",
                        ));
                    }
                    0
                };
                let path = if shape.batch {
                    format!("/{index}")
                } else {
                    String::new()
                };
                let state = &mut states[index];
                if state.finished {
                    return Err(Violation::new(
                        path,
                        "result emitted data after termination",
                    ));
                }
                validate_frame(&value, &path, false)?;
                state.accept(&value, self.incremental, &path)?;
                Ok(())
            })();
            if let Err(mut violation) = result {
                violation.event = Some(event_index);
                violations.push(violation);
            }
        }
        if !done {
            violations.push(Violation::new("", "missing [DONE] event"));
        }
        for (index, state) in states.iter().enumerate() {
            if !state.finished {
                violations.push(Violation::new(
                    if shape.batch {
                        format!("/{index}")
                    } else {
                        String::new()
                    },
                    "result did not terminate",
                ));
            }
        }
        if !violations.is_empty() {
            return Err(violations);
        }
        let mut values = Vec::with_capacity(shape.count);
        for (index, state) in states.into_iter().enumerate() {
            let value = state.finish(self.incremental, shape.batch);
            let path = if shape.batch {
                format!("/{index}")
            } else {
                String::new()
            };
            validate_requested_logprobs(case, index, &value, &path).map_err(|error| vec![error])?;
            values.push(value);
        }
        Ok(if shape.batch {
            Value::Array(values)
        } else {
            values.remove(0)
        })
    }
}

fn validate_frame<'a>(
    value: &'a Value,
    path: &str,
    terminal: bool,
) -> Result<&'a Map<String, Value>, Violation> {
    let object = value
        .as_object()
        .ok_or_else(|| Violation::new(path, "result must be an object"))?;
    if object.contains_key("error") {
        return Err(Violation::new(
            format!("{path}/error"),
            "error response in a successful generation",
        ));
    }
    let meta = value
        .get("meta_info")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            Violation::new(format!("{path}/meta_info"), "meta_info must be an object")
        })?;
    for key in ["prompt_tokens", "completion_tokens"] {
        if meta.get(key).and_then(Value::as_u64).is_none() {
            return Err(Violation::new(
                format!("{path}/meta_info/{key}"),
                "token count must be a non-negative integer",
            ));
        }
    }
    if meta
        .get("id")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(Violation::new(
            format!("{path}/meta_info/id"),
            "result id must be a non-empty string",
        ));
    }
    match meta.get("finish_reason") {
        Some(Value::Null) if !terminal => {}
        Some(Value::Object(reason))
            if reason
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| matches!(kind, "stop" | "length")) => {}
        _ => {
            return Err(Violation::new(
                format!("{path}/meta_info/finish_reason"),
                "expected a normal finish reason, or null before termination",
            ));
        }
    }
    if !object.contains_key("text") && !object.contains_key("output_ids") {
        return Err(Violation::new(
            path,
            "result must contain text or output_ids",
        ));
    }
    if object.get("text").is_some_and(|text| !text.is_string()) {
        return Err(Violation::new(
            format!("{path}/text"),
            "text must be a string",
        ));
    }
    if object.get("output_ids").is_some_and(|ids| {
        !ids.as_array()
            .is_some_and(|ids| ids.iter().all(|id| id.as_u64().is_some()))
    }) {
        return Err(Violation::new(
            format!("{path}/output_ids"),
            "output_ids must contain non-negative integers",
        ));
    }
    Ok(meta)
}

impl ResultState {
    fn accept(&mut self, value: &Value, incremental: bool, path: &str) -> Result<(), Violation> {
        let meta = value["meta_info"].as_object().expect("validated metadata");
        // Existing delta positions already agreed on a previous frame. If either
        // optional family is arriving for the first time, validate all overlap.
        let checked_ids = if incremental {
            self.ids
                .as_ref()
                .zip(self.output.get("output_token_logprobs"))
                .map_or(0, |(ids, logprobs)| ids.len().min(logprobs.len()))
        } else {
            0
        };
        if let Some(previous) = &self.last {
            for key in ["id", "prompt_tokens"] {
                if previous["meta_info"][key] != meta[key] {
                    return Err(Violation::new(
                        format!("{path}/meta_info/{key}"),
                        "value changed within one result",
                    ));
                }
            }
            if previous["meta_info"]["completion_tokens"].as_u64()
                > meta["completion_tokens"].as_u64()
            {
                return Err(Violation::new(
                    format!("{path}/meta_info/completion_tokens"),
                    "completion count moved backwards",
                ));
            }
        }
        if let Some(text) = value.get("text").and_then(Value::as_str) {
            if incremental {
                self.text.get_or_insert_default().push_str(text);
            } else {
                if self.text.as_ref().is_some_and(|old| !text.starts_with(old)) {
                    return Err(Violation::new(
                        format!("{path}/text"),
                        "cumulative text is not a prefix extension",
                    ));
                }
                self.text = Some(text.into());
            }
        } else if self.text.is_some() && !incremental {
            return Err(Violation::new(
                format!("{path}/text"),
                "cumulative text disappeared",
            ));
        }
        if let Some(ids) = value.get("output_ids").and_then(Value::as_array) {
            merge_sequence(
                &mut self.ids,
                ids,
                incremental,
                &format!("{path}/output_ids"),
            )?;
        } else if self.ids.is_some() && !incremental {
            return Err(Violation::new(
                format!("{path}/output_ids"),
                "cumulative output_ids disappeared",
            ));
        }
        for key in OUTPUT_LOGPROBS {
            update_logprobs(&mut self.output, meta, key, incremental, path)?;
        }
        for key in INPUT_LOGPROBS {
            // Input logprobs are set once in Rust deltas and repeated by Python;
            // accept an empty placeholder but never concatenate those repeats.
            if let Some(raw) = meta.get(key) {
                let values = validate_logprobs(raw, key, &format!("{path}/meta_info/{key}"))?;
                if !values.is_empty() {
                    if let Some(old) = self.input.get(key)
                        && !old.is_empty()
                        && values != old
                        && (incremental || !values.starts_with(old))
                    {
                        return Err(Violation::new(
                            format!("{path}/meta_info/{key}"),
                            "input logprobs changed after being received",
                        ));
                    }
                    self.input.insert(key, values.clone());
                } else {
                    if !incremental && self.input.get(key).is_some_and(|old| !old.is_empty()) {
                        return Err(Violation::new(
                            format!("{path}/meta_info/{key}"),
                            "cumulative input logprobs moved backwards",
                        ));
                    }
                    self.input.entry(key).or_default();
                }
            } else if self.input.contains_key(key) && !incremental {
                return Err(Violation::new(
                    format!("{path}/meta_info/{key}"),
                    "cumulative input logprobs disappeared",
                ));
            }
        }
        let count = meta["completion_tokens"].as_u64().expect("validated count");
        // A matched stop token is trimmed from output_ids but remains in the
        // generated-token count and its logprobs (detokenizer::decode_chunk).
        if let Some(ids) = &self.ids {
            let trimmed_stop = meta["finish_reason"]["type"] == "stop"
                && meta["finish_reason"]["matched"]
                    .as_i64()
                    .is_some_and(|token| token != 0);
            let missing = count.checked_sub(ids.len() as u64);
            if missing != Some(0) && !(trimmed_stop && missing == Some(1)) {
                return Err(Violation::new(
                    format!("{path}/output_ids"),
                    "output token count must match completion_tokens, except one trimmed terminal stop token",
                ));
            }
        }
        for (key, values) in &self.output {
            if values.len() as u64 != count {
                return Err(Violation::new(
                    format!("{path}/meta_info/{key}"),
                    "output logprob positions do not match cumulative completion_tokens",
                ));
            }
        }
        let prompt_count = meta["prompt_tokens"].as_u64().expect("validated count");
        for (key, values) in &self.input {
            if values.len() as u64 > prompt_count {
                return Err(Violation::new(
                    format!("{path}/meta_info/{key}"),
                    "input logprob positions exceed prompt_tokens",
                ));
            }
        }
        if let (Some(ids), Some(logprobs)) = (&self.ids, self.output.get("output_token_logprobs"))
            && ids[checked_ids..]
                .iter()
                .zip(&logprobs[checked_ids..])
                .any(|(id, lp)| id != &lp[1])
        {
            return Err(Violation::new(
                format!("{path}/meta_info/output_token_logprobs"),
                "logprob token ids disagree with output_ids",
            ));
        }
        self.finished = !meta["finish_reason"].is_null();
        self.last = Some(value.clone());
        Ok(())
    }

    fn finish(self, incremental: bool, batch: bool) -> Value {
        let mut value = self.last.expect("every result terminated");
        if incremental {
            if let Some(text) = self.text {
                value["text"] = Value::String(text);
            }
            if let Some(ids) = self.ids {
                value["output_ids"] = Value::Array(ids);
            }
            for (key, values) in self.output.into_iter().chain(self.input) {
                value["meta_info"][key] = Value::Array(values);
            }
        }
        if batch {
            value
                .as_object_mut()
                .expect("validated object")
                .remove("index");
        }
        value
    }
}

fn merge_sequence(
    current: &mut Option<Vec<Value>>,
    values: &[Value],
    incremental: bool,
    path: &str,
) -> Result<(), Violation> {
    if incremental {
        current.get_or_insert_default().extend_from_slice(values);
    } else {
        if current.as_ref().is_some_and(|old| !values.starts_with(old)) {
            return Err(Violation::new(
                path,
                "cumulative sequence is not a prefix extension",
            ));
        }
        *current = Some(values.to_vec());
    }
    Ok(())
}

fn update_logprobs(
    current: &mut BTreeMap<&'static str, Vec<Value>>,
    meta: &Map<String, Value>,
    key: &'static str,
    incremental: bool,
    path: &str,
) -> Result<(), Violation> {
    let path = format!("{path}/meta_info/{key}");
    if let Some(raw) = meta.get(key) {
        let values = validate_logprobs(raw, key, &path)?;
        let old = current.entry(key).or_default();
        if incremental {
            old.extend_from_slice(values);
        } else {
            if !values.starts_with(old) {
                return Err(Violation::new(
                    path,
                    "cumulative logprobs are not a prefix extension",
                ));
            }
            *old = values.clone();
        }
    } else if current.contains_key(key) && !incremental {
        return Err(Violation::new(
            path,
            "cumulative output logprobs disappeared",
        ));
    }
    Ok(())
}

fn validate_logprobs<'a>(
    raw: &'a Value,
    key: &str,
    path: &str,
) -> Result<&'a Vec<Value>, Violation> {
    let values = raw
        .as_array()
        .ok_or_else(|| Violation::new(path, "logprobs must be an array"))?;
    let flat = key.ends_with("_token_logprobs");
    for (position, value) in values.iter().enumerate() {
        if !flat && value.is_null() {
            continue;
        }
        let tuples = if flat {
            std::slice::from_ref(value)
        } else {
            value
                .as_array()
                .ok_or_else(|| {
                    Violation::new(
                        format!("{path}/{position}"),
                        "top/token-id logprobs must contain arrays or null",
                    )
                })?
                .as_slice()
        };
        for tuple in tuples {
            let valid = tuple.as_array().is_some_and(|tuple| {
                tuple.len() == 3
                    && (tuple[0].is_null() || tuple[0].as_f64().is_some_and(f64::is_finite))
                    && tuple[1].as_u64().is_some()
                    && (tuple[2].is_null() || tuple[2].is_string())
            });
            if !valid {
                return Err(Violation::new(
                    format!("{path}/{position}"),
                    "expected [logprob, token_id, text-or-null] tuple",
                ));
            }
        }
    }
    Ok(values)
}

fn per_result<'a>(body: &'a Value, key: &str, index: usize) -> Option<&'a Value> {
    match body.get(key) {
        Some(Value::Array(values)) => values.get(index % values.len().max(1)),
        value => value,
    }
}

fn validate_requested_logprobs(
    case: &HttpCase,
    index: usize,
    value: &Value,
    path: &str,
) -> Result<(), Violation> {
    if per_result(&case.body, "return_logprob", index).and_then(Value::as_bool) != Some(true) {
        return Ok(());
    }
    let meta = value["meta_info"].as_object().expect("validated metadata");
    for key in ["input_token_logprobs", "output_token_logprobs"] {
        if !meta.contains_key(key) {
            return Err(Violation::new(
                format!("{path}/meta_info/{key}"),
                "requested logprobs are missing",
            ));
        }
    }
    if per_result(&case.body, "top_logprobs_num", index)
        .and_then(Value::as_u64)
        .unwrap_or(0)
        > 0
        && !meta.contains_key("output_top_logprobs")
    {
        return Err(Violation::new(
            format!("{path}/meta_info/output_top_logprobs"),
            "requested top logprobs are missing",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
