//! Thinking-mode defaults read from a Jinja chat template.

use std::collections::HashMap;

use minijinja::machinery::{Token, tokenize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum ThinkingPolicy {
    #[default]
    Unknown,
    Always,
    TemplateToggle {
        key: &'static str,
        default_enabled: bool,
    },
    NativeToggle {
        default_enabled: bool,
        named_tool_disables: bool,
    },
    ReasoningEffort,
}

impl ThinkingPolicy {
    pub(super) fn apply(
        self,
        args: &mut Option<HashMap<String, Value>>,
        named_tool_choice: bool,
    ) -> Option<bool> {
        match self {
            Self::Unknown => None,
            Self::Always => Some(true),
            Self::TemplateToggle {
                key,
                default_enabled,
            } => {
                let enabled = args
                    .as_ref()
                    .and_then(|args| args.get(key))
                    .and_then(Value::as_bool)
                    .unwrap_or(default_enabled);
                args.get_or_insert_default()
                    .insert(key.to_owned(), Value::Bool(enabled));
                Some(enabled)
            }
            Self::NativeToggle {
                default_enabled,
                named_tool_disables,
            } => {
                let enabled = if named_tool_disables && named_tool_choice {
                    false
                } else {
                    dynamo_renderer::thinking_bool_from_args(args.as_ref())
                        .unwrap_or(default_enabled)
                };
                args.get_or_insert_default()
                    .insert("thinking".to_owned(), Value::Bool(enabled));
                Some(enabled)
            }
            Self::ReasoningEffort => Some(
                args.as_ref()
                    .and_then(|args| args.get("reasoning_effort"))
                    .is_some_and(|effort| effort.as_str() != Some("none")),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ThinkingTemplates {
    default: ThinkingPolicy,
    tool_use: Option<ThinkingPolicy>,
}

impl ThinkingTemplates {
    pub(crate) fn native(default_enabled: bool, named_tool_disables: bool) -> Self {
        let policy = ThinkingPolicy::NativeToggle {
            default_enabled,
            named_tool_disables,
        };
        Self {
            default: policy,
            tool_use: Some(policy),
        }
    }

    pub(crate) fn always() -> Self {
        Self {
            default: ThinkingPolicy::Always,
            tool_use: Some(ThinkingPolicy::Always),
        }
    }

    pub(super) fn for_request(self, tools_enabled: bool) -> ThinkingPolicy {
        if tools_enabled {
            self.tool_use.unwrap_or(self.default)
        } else {
            self.default
        }
    }

    pub(super) fn from_config(config: &Value) -> Self {
        let Some(template) = config.get("chat_template") else {
            return Self::default();
        };
        if let Some(template) = template.as_str() {
            let policy = detect_thinking_policy(template);
            return Self {
                default: policy,
                tool_use: Some(policy),
            };
        }
        let mut policies = Self::default();
        for templates in template.as_array().into_iter().flatten() {
            let Some(templates) = templates.as_object() else {
                continue;
            };
            for (name, template) in templates {
                let Some(template) = template.as_str() else {
                    continue;
                };
                match name.as_str() {
                    "default" => policies.default = detect_thinking_policy(template),
                    "tool_use" => {
                        policies.tool_use = Some(detect_thinking_policy(template));
                    }
                    _ => {}
                }
            }
        }
        policies
    }
}

fn detect_thinking_policy(template: &str) -> ThinkingPolicy {
    if template.contains("<|channel|>")
        || ((!template.contains("enable_thinking") && !template.contains("thinking"))
            && (template.contains(r"<|im_start|>assistant\n<think>\n")
                || template.contains("<|im_start|>assistant\n<think>\n")))
    {
        return ThinkingPolicy::Always;
    }

    if template.contains("reasoning_effort") && template.contains("[THINK]") {
        return ThinkingPolicy::ReasoningEffort;
    }

    let Some(tokens) = jinja_code_tokens(template) else {
        return ThinkingPolicy::Unknown;
    };
    for key in ["enable_thinking", "thinking"] {
        if let Some(default_enabled) = detect_toggle_default(&tokens, key) {
            return ThinkingPolicy::TemplateToggle {
                key,
                default_enabled,
            };
        }
    }
    ThinkingPolicy::Unknown
}

fn jinja_code_tokens(template: &str) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    for token in tokenize(template, false, Default::default(), Default::default()) {
        let (token, _) = token.ok()?;
        let value = match token {
            Token::Ident(value) => value.to_owned(),
            Token::Pipe => "|".to_owned(),
            Token::Assign => "=".to_owned(),
            Token::Comma => ",".to_owned(),
            Token::ParenOpen => "(".to_owned(),
            Token::ParenClose => ")".to_owned(),
            Token::Dot => ".".to_owned(),
            Token::BlockStart | Token::VariableStart => ";".to_owned(),
            Token::BlockEnd | Token::VariableEnd => ";".to_owned(),
            Token::TemplateData(_) | Token::Str(_) | Token::String(_) => continue,
            _ => continue,
        };
        tokens.push(value);
    }
    Some(tokens)
}

fn detect_toggle_default(tokens: &[String], key: &str) -> Option<bool> {
    if has_default_filter(tokens, key, false) || has_guarded_default(tokens, key, false) {
        return Some(false);
    }
    if has_default_filter(tokens, key, true)
        || has_guarded_default(tokens, key, true)
        || contains_tokens(
            tokens,
            &[
                "set", key, "=", key, "if", key, "is", "defined", "else", "true",
            ],
        )
        || contains_tokens(tokens, &[key, "is", "defined", "and", key, "is", "false"])
        || contains_tokens(tokens, &[key, "is", "defined", "and", "not", key])
        || contains_tokens(tokens, &[key, "is", "not", "defined", "or", key])
        || contains_after(tokens, &["namespace", "("], &[key, "=", "true"], None)
    {
        return Some(true);
    }
    None
}

fn has_default_filter(tokens: &[String], key: &str, expected: bool) -> bool {
    for (index, token) in tokens.iter().enumerate() {
        if token != key || index.checked_sub(1).is_some_and(|i| tokens[i] == ".") {
            continue;
        }
        let Some(filter) = tokens.get(index + 1..index + 5) else {
            continue;
        };
        if filter[0] != "|" || !matches!(filter[1].as_str(), "default" | "d") || filter[2] != "(" {
            continue;
        }
        let Some(default_enabled) = jinja_bool(&filter[3]) else {
            continue;
        };
        let mut cursor = index + 5;
        let boolean_mode = match tokens.get(cursor).map(String::as_str) {
            Some(")") => false,
            Some(",") => {
                cursor += 1;
                match tokens.get(cursor).map(String::as_str) {
                    Some("true") => true,
                    Some("false") => false,
                    Some("boolean") if tokens.get(cursor + 1).is_some_and(|token| token == "=") => {
                        let Some(value) =
                            tokens.get(cursor + 2).and_then(|value| jinja_bool(value))
                        else {
                            continue;
                        };
                        value
                    }
                    _ => continue,
                }
            }
            _ => continue,
        };
        if default_enabled == expected && !(default_enabled && boolean_mode) {
            return true;
        }
    }
    false
}

fn has_guarded_default(tokens: &[String], key: &str, enabled: bool) -> bool {
    let value = if enabled { "true" } else { "false" };
    for guard in [
        ["if", "not", key, "is", "defined"],
        ["if", key, "is", "not", "defined"],
    ] {
        if contains_after(tokens, &guard, &["set", key, "=", value], Some("endif")) {
            return true;
        }
    }
    false
}

fn contains_tokens(tokens: &[String], expected: &[&str]) -> bool {
    tokens.windows(expected.len()).any(|window| {
        window
            .iter()
            .map(String::as_str)
            .eq(expected.iter().copied())
    })
}

fn contains_after(tokens: &[String], prefix: &[&str], suffix: &[&str], stop: Option<&str>) -> bool {
    for start in 0..tokens.len().saturating_sub(prefix.len()).saturating_add(1) {
        if !tokens[start..]
            .iter()
            .take(prefix.len())
            .map(String::as_str)
            .eq(prefix.iter().copied())
        {
            continue;
        }
        let remainder = &tokens[start + prefix.len()..];
        let end = stop
            .and_then(|stop| remainder.iter().position(|token| token == stop))
            .unwrap_or(remainder.len());
        if contains_tokens(&remainder[..end], suffix) {
            return true;
        }
    }
    false
}

fn jinja_bool(value: &str) -> Option<bool> {
    match value {
        "true" | "True" => Some(true),
        "false" | "False" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{ThinkingPolicy, detect_thinking_policy};

    #[test]
    fn thinking_toggle_detection_matches_template_defaults() {
        assert_eq!(
            detect_thinking_policy(
                "{% if enable_thinking is not defined %}{% set enable_thinking = true %}{% endif %}"
            ),
            ThinkingPolicy::TemplateToggle {
                key: "enable_thinking",
                default_enabled: true,
            }
        );
        assert_eq!(
            detect_thinking_policy(
                "{% if not thinking is defined %}{% set thinking = false %}{% endif %}"
            ),
            ThinkingPolicy::TemplateToggle {
                key: "thinking",
                default_enabled: false,
            }
        );
        assert_eq!(
            detect_thinking_policy("{{ messages }}"),
            ThinkingPolicy::Unknown
        );
    }

    #[test]
    fn thinking_default_filters_match_python_semantics() {
        let cases = [
            ("{{ enable_thinking | d(true) }}", true),
            ("{{ enable_thinking|default(false) }}", false),
            ("{{ enable_thinking | default(true, false) }}", true),
            ("{{ enable_thinking | default(false, true) }}", false),
            (
                "{{ enable_thinking | default(false, boolean=true) }}",
                false,
            ),
            (
                "{% if enable_thinking | default(false) %}x{% endif %}",
                false,
            ),
        ];
        for (template, default_enabled) in cases {
            assert_eq!(
                detect_thinking_policy(template),
                ThinkingPolicy::TemplateToggle {
                    key: "enable_thinking",
                    default_enabled,
                },
                "{template}"
            );
        }
    }

    #[test]
    fn thinking_detection_ignores_non_executable_toggle_text() {
        for template in [
            "{# {{ enable_thinking | default(false) }} #}{{ enable_thinking | default(true) }}",
            "{% raw %}{{ enable_thinking | default(false) }}{% endraw %}{{ enable_thinking | default(true) }}",
            "{{ 'enable_thinking | default(false)' }}{{ enable_thinking | default(true) }}",
        ] {
            assert_eq!(
                detect_thinking_policy(template),
                ThinkingPolicy::TemplateToggle {
                    key: "enable_thinking",
                    default_enabled: true,
                },
                "{template}"
            );
        }
        assert_eq!(
            detect_thinking_policy("{{ enable_thinking | default(true, true) }}"),
            ThinkingPolicy::Unknown
        );
        assert_eq!(
            detect_thinking_policy("{% if enable_thinking %}think{% endif %}"),
            ThinkingPolicy::Unknown
        );
    }

    #[test]
    fn channel_templates_are_always_on() {
        assert_eq!(
            detect_thinking_policy("<|start|>assistant<|channel|>analysis<|message|>"),
            ThinkingPolicy::Always
        );
    }
}
