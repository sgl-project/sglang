//! Python's prompt-facing reasoning normalization for the Jinja HTTP path.

use serde_json::{Map, Value};

pub(crate) fn prepare_reasoning(raw: &mut Value) -> Result<(), String> {
    let nested = raw.get("reasoning").and_then(Value::as_object);
    let nested_effort = nested
        .and_then(|reasoning| {
            reasoning
                .get("effort")
                .filter(|value| !value.is_null())
                .or_else(|| reasoning.get("reasoning_effort"))
        })
        .filter(|value| !value.is_null());
    let effort = nested_effort
        .or_else(|| raw.get("reasoning_effort"))
        .filter(|value| !value.is_null())
        .map(|value| normalize_effort(value, nested_effort.is_some()))
        .transpose()?;
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
    let thinking = effort
        .as_ref()
        .map(|effort| effort != "none")
        .or(enabled.then_some(true));
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
    if extra.get("reasoning_effort").is_some_and(Value::is_null) {
        extra.remove("reasoning_effort");
    }
    if let Some(effort) = effort {
        // Dynamo's shared request type only supports tier strings. The Jinja
        // kwargs sidecar carries a validated Python float past that boundary.
        raw["reasoning_effort"] = if effort.is_number() {
            Value::Null
        } else {
            effort.clone()
        };
        extra.entry("reasoning_effort").or_insert(effort);
    }
    raw["chat_template_kwargs"] = Value::Object(extra);
    Ok(())
}

fn normalize_effort(value: &Value, nested: bool) -> Result<Value, String> {
    if matches!(
        value.as_str(),
        Some("none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max")
    ) {
        return Ok(value.clone());
    }
    let number = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(number) => parse_float_string(number, nested),
        _ => None,
    }
    .filter(|number| number.is_finite() && (0.0..=0.99).contains(number))
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

#[cfg(test)]
mod tests {
    use super::prepare_reasoning;
    use serde_json::json;

    #[test]
    fn nested_none_replaces_invalid_top_level_before_validation() {
        let mut raw = json!({
            "reasoning_effort": true,
            "reasoning": {"effort": "none"},
            "chat_template_kwargs": {"reasoning_effort": "high"}
        });
        prepare_reasoning(&mut raw).unwrap();
        assert_eq!(raw["reasoning_effort"], "none");
        assert_eq!(
            raw["chat_template_kwargs"],
            json!({"reasoning_effort":"high","thinking":false,"enable_thinking":false})
        );
    }

    #[test]
    fn nested_enabled_uses_python_truthiness_and_preserves_explicit_flags() {
        for (reasoning, expected) in [
            (json!({"enabled":" YES "}), true),
            (json!({"enable":[0]}), true),
            (json!({"enabled":{"x":0}}), true),
            (json!({"enabled":null,"enable":true}), true),
            (json!({"enabled":false,"enable":true}), false),
            (json!({"enabled":"off","enable":true}), false),
            (json!({"enabled":[]}), false),
        ] {
            let mut raw = json!({"reasoning":reasoning});
            prepare_reasoning(&mut raw).unwrap();
            assert_eq!(
                raw["chat_template_kwargs"].get("thinking"),
                expected.then_some(&json!(true)),
                "{reasoning}"
            );
        }
        let mut raw = json!({"reasoning":{"enabled":true},"chat_template_kwargs":"oops"});
        prepare_reasoning(&mut raw).unwrap();
        assert_eq!(
            raw["chat_template_kwargs"],
            json!({"thinking":true,"enable_thinking":true})
        );

        let mut raw = json!({"reasoning":{"enabled":true,"effort":"none"},"chat_template_kwargs":{"thinking":null}});
        prepare_reasoning(&mut raw).unwrap();
        assert_eq!(
            raw["chat_template_kwargs"],
            json!({"reasoning_effort":"none","thinking":null,"enable_thinking":false})
        );

        let mut raw = json!({"reasoning":{"enabled":false},"chat_template_kwargs":"oops"});
        assert!(prepare_reasoning(&mut raw).is_err());
    }

    #[test]
    fn numeric_effort_is_validated_before_kwargs_override_and_kept_as_a_float() {
        for (input, expected) in [
            (json!(0), json!(0.0)),
            (json!(0.99), json!(0.99)),
            (json!(" +2.5e-1 "), json!(0.25)),
            (json!("0_0.2_5"), json!(0.25)),
        ] {
            for nested in [false, true] {
                let mut raw = if nested {
                    json!({"reasoning":{"effort":input}})
                } else {
                    json!({"reasoning_effort":input})
                };
                prepare_reasoning(&mut raw).unwrap();
                assert_eq!(raw["reasoning_effort"], json!(null));
                assert_eq!(
                    raw["chat_template_kwargs"],
                    json!({"reasoning_effort":expected,"thinking":true,"enable_thinking":true})
                );
            }
        }
        for input in [
            json!(-1),
            json!(1),
            json!(0.99000001),
            json!("NaN"),
            json!("inf"),
            json!(true),
            json!([]),
            json!({}),
        ] {
            let mut raw = json!({"reasoning_effort":input,"chat_template_kwargs":{"reasoning_effort":"high"}});
            assert!(prepare_reasoning(&mut raw).is_err(), "{input}");
        }
        let mut raw =
            json!({"reasoning":{"effort":0.25},"chat_template_kwargs":{"reasoning_effort":true}});
        prepare_reasoning(&mut raw).unwrap();
        assert_eq!(raw["chat_template_kwargs"]["reasoning_effort"], true);
    }

    #[test]
    fn nested_numeric_strings_use_python_unicode_decimal_digits() {
        for input in [
            "\u{660}.\u{662}\u{665}",
            "\u{ff10}.\u{ff12}\u{ff15}",
            "\u{1e4f0}.\u{1e4f2}\u{1e4f5}",
            "0.\u{662}_\u{665}",
        ] {
            let mut raw = json!({"reasoning":{"effort":input}});
            prepare_reasoning(&mut raw).unwrap();
            assert_eq!(raw["chat_template_kwargs"]["reasoning_effort"], json!(0.25));
            let mut raw = json!({"reasoning_effort":input});
            assert!(prepare_reasoning(&mut raw).is_err(), "{input}");
        }
        for input in ["\u{b2}.25", "\u{2160}.25", "\u{2080}.25"] {
            let mut raw = json!({"reasoning":{"effort":input}});
            assert!(prepare_reasoning(&mut raw).is_err(), "{input}");
        }
        for (input, top_accepts) in [
            ("0_.25", true),
            ("0._25", true),
            ("0e-_1", true),
            ("_0.25", false),
            ("0.2__5", false),
            ("0.25_", false),
        ] {
            let mut raw = json!({"reasoning_effort":input});
            assert_eq!(prepare_reasoning(&mut raw).is_ok(), top_accepts, "{input}");
            let mut raw = json!({"reasoning":{"effort":input}});
            assert!(prepare_reasoning(&mut raw).is_err(), "{input}");
        }
    }

    #[test]
    fn enabled_strip_accepts_ascii_separators_while_float_conversion_rejects_them() {
        for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            let mut raw = json!({"reasoning":{"enabled":format!("{separator}true{separator}")}});
            prepare_reasoning(&mut raw).unwrap();
            assert_eq!(raw["chat_template_kwargs"]["thinking"], true);
            let literal = format!("{separator}0.25{separator}");
            for mut raw in [
                json!({"reasoning_effort":literal}),
                json!({"reasoning":{"effort":literal}}),
            ] {
                assert!(prepare_reasoning(&mut raw).is_err());
            }
        }
    }
}
