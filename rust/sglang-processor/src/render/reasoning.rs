//! The effort and thinking toggle SGLang's `ChatCompletionRequest` derives from
//! `reasoning` and `reasoning_effort` (`protocol.py`), shared with hosts.

use serde_json::{Map, Value};

/// `protocol.ReasoningEffortTier`.
#[cfg(feature = "render")]
pub(crate) const EFFORT_TIERS: [&str; 8] = [
    "none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max",
];

/// Python's request-model normalization and validation before rendering:
/// nested aliases override the top-level effort; explicit template kwargs win.
pub fn prepare_reasoning(raw: &mut Value) -> Result<(), String> {
    if !raw.is_object() {
        return Err("chat request must be an object".into());
    }
    let nested = raw.get("reasoning").and_then(Value::as_object);
    let nested_effort = nested
        .and_then(|reasoning| {
            reasoning
                .get("effort")
                .filter(|value| !value.is_null())
                .or_else(|| reasoning.get("reasoning_effort"))
        })
        .filter(|value| !value.is_null());
    let mut effort = nested_effort
        .map(|value| normalize_effort(value, true, false))
        .transpose()?
        .or_else(|| {
            raw.get("reasoning_effort")
                .filter(|value| !value.is_null())
                .cloned()
        });
    let enabled = nested
        .and_then(|reasoning| {
            reasoning
                .get("enabled")
                .filter(|value| !value.is_null())
                .or_else(|| reasoning.get("enable"))
        })
        .is_some_and(|value| match value {
            Value::String(value) => matches!(
                value
                    .trim_matches(|character: char| {
                        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
                    })
                    .to_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "y" | "on"
            ),
            _ => python_truthy(value),
        });
    let thinking_config = raw.get("thinking").and_then(Value::as_object);
    let thinking_type = thinking_config
        .and_then(|thinking| thinking.get("type"))
        .filter(|value| !value.is_null());
    if let Some(value) = thinking_config
        .and_then(|thinking| thinking.get("effort"))
        .filter(|value| !value.is_null())
    {
        effort = Some(normalize_effort(value, true, false)?);
    }
    let effort = effort
        .map(|value| normalize_effort(&value, false, true))
        .transpose()?;
    let mut thinking = effort
        .as_ref()
        .map(|effort| effort != "none")
        .or(enabled.then_some(true));
    if let Some(kind) = thinking_type {
        thinking = Some(kind != "disabled");
    } else if thinking_config.is_some() && thinking.is_none() {
        thinking = Some(true);
    }
    let mut extra = match raw.get("chat_template_kwargs") {
        Some(Value::Object(extra)) => extra.clone(),
        None | Some(Value::Null) => Map::new(),
        Some(_) if thinking.is_some() => Map::new(),
        Some(_) => return Err("chat_template_kwargs must be an object".into()),
    };
    if let Some(thinking) = thinking {
        extra.entry("thinking").or_insert(Value::Bool(thinking));
        extra
            .entry("enable_thinking")
            .or_insert(Value::Bool(thinking));
    }
    if let Some(effort) = effort {
        raw["reasoning_effort"] = effort;
    }
    // Consumers read the canonical top-level effort after normalization. The
    // aliases are inputs, not a second source of unparsed template values.
    if let Some(reasoning) = raw.get_mut("reasoning").and_then(Value::as_object_mut) {
        reasoning.remove("effort");
        reasoning.remove("reasoning_effort");
    }
    if let Some(thinking) = raw.get_mut("thinking").and_then(Value::as_object_mut) {
        thinking.remove("effort");
    }
    raw["chat_template_kwargs"] = Value::Object(extra);
    Ok(())
}

fn normalize_effort(value: &Value, nested: bool, check_range: bool) -> Result<Value, String> {
    if matches!(
        value.as_str(),
        Some("none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "ultra" | "max")
    ) {
        return Ok(value.clone());
    }
    let number = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(number) => parse_float_string(number, nested),
        _ => None,
    }
    .filter(|number| !check_range || number.is_finite() && (0.0..=0.99).contains(number))
    .ok_or_else(|| "invalid reasoning effort".to_string())?;
    Ok(Value::from(number))
}

fn parse_float_string(number: &str, nested: bool) -> Option<f64> {
    // Pydantic retries underscore removal on the original, untrimmed string.
    if !nested && number.contains('_') && number != number.trim() {
        return None;
    }
    let number = number.trim();
    let number = if nested {
        number
            .chars()
            .map(|character| decimal_to_ascii(character).unwrap_or(character))
            .collect::<String>()
    } else {
        number.to_owned()
    };
    // Pydantic's top-level lax float strips individual interior underscores.
    // The nested alias calls Python float(), which requires digit adjacency.
    let bytes = number.as_bytes();
    if bytes.iter().enumerate().any(|(index, byte)| {
        *byte == b'_'
            && (index == 0
                || index + 1 == bytes.len()
                || bytes[index - 1] == b'_'
                || nested
                    && (!bytes[index - 1].is_ascii_digit() || !bytes[index + 1].is_ascii_digit()))
    }) {
        return None;
    }
    number.replace('_', "").parse().ok()
}

fn decimal_to_ascii(character: char) -> Option<char> {
    // Python 3.12 unicodedata 15.0.0: zero code points of all Nd blocks.
    // Each block contains its decimal digits consecutively from zero to nine.
    const ZEROS: &[u32] = &[
        0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
        0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
        0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0,
        0xff10, 0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0,
        0x11650, 0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60,
        0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0,
        0x1e950, 0x1fbf0,
    ];
    let point = character as u32;
    let index = ZEROS
        .partition_point(|zero| *zero <= point)
        .checked_sub(1)?;
    let digit = point - ZEROS[index];
    (digit < 10).then(|| char::from(b'0' + digit as u8))
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

/// Read the request effort (including aliases on a not-yet-normalized body).
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
        enabled => python_truthy(enabled),
    };
    enabled.then_some(true)
}
