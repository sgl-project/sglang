// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Request-time checks for the parts of a profile the sampling contract does
//! not cover.

use serde::Deserialize;
use serde_json::Value;

use super::{ApiProfile, OnExceed};

/// The request field that carries the output budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetField {
    MaxTokens,
    MaxCompletionTokens,
}

impl BudgetField {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

/// An output-budget value to write into the forwarded body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetEdit {
    pub field: BudgetField,
    pub value: u64,
    /// Overwrites a client value, so the body cannot be spliced.
    pub replaces: bool,
}

impl ApiProfile {
    pub fn is_alias(&self, model: &str) -> bool {
        self.models.aliases.iter().any(|a| a == model)
    }

    /// Applies `output.max_tokens` to the request's budget, read with the
    /// engine's precedence (`max_completion_tokens` first).
    pub fn output_budget(
        &self,
        max_completion_tokens: Option<u64>,
        max_tokens: Option<u64>,
    ) -> Result<Option<BudgetEdit>, String> {
        let Some(rule) = &self.output.max_tokens else {
            return Ok(None);
        };
        let (field, requested) = match (max_completion_tokens, max_tokens) {
            (Some(v), _) => (BudgetField::MaxCompletionTokens, v),
            (None, Some(v)) => (BudgetField::MaxTokens, v),
            (None, None) => {
                let value = rule.default.unwrap_or(rule.cap);
                return Ok(Some(BudgetEdit {
                    field: BudgetField::MaxTokens,
                    value,
                    replaces: false,
                }));
            }
        };
        if requested <= rule.cap {
            return Ok(None);
        }
        match rule.on_exceed {
            OnExceed::Reject => Err(format!(
                "{} must be at most {}, got {requested}",
                field.wire_name(),
                rule.cap
            )),
            OnExceed::Clamp => Ok(Some(BudgetEdit {
                field,
                value: rule.cap,
                replaces: true,
            })),
        }
    }

    /// Enforces `limits.max_images` on a chat body.
    pub fn check_images(&self, body: &[u8]) -> Result<(), String> {
        let Some(max) = self.limits.max_images else {
            return Ok(());
        };
        /// Only `messages` is materialized; other fields are skipped.
        #[derive(Deserialize)]
        struct Messages {
            messages: Option<Value>,
        }
        // An unparsable body is left to the chat handler's own validation.
        let Some(messages) = serde_json::from_slice::<Messages>(body)
            .ok()
            .and_then(|m| m.messages)
        else {
            return Ok(());
        };
        let count = messages
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m.get("content")?.as_array())
            .flatten()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("image_url"))
            .count();
        if count > max {
            return Err(format!("at most {max} images per request, got {count}"));
        }
        Ok(())
    }

    /// Replaces `type` in a 400 body: the OpenAI `{"error": {...}}` envelope or
    /// sglang's flat error. `None` keeps the body.
    pub fn retype_bad_request(&self, body: &[u8]) -> Option<Vec<u8>> {
        let typ = self.errors.bad_request_type.as_deref()?;
        let mut v: Value = serde_json::from_slice(body).ok()?;
        let slot = if v.get("error").is_some_and(Value::is_object) {
            &mut v["error"]
        } else if v.get("message").is_some() {
            &mut v
        } else {
            return None;
        };
        slot["type"] = Value::String(typ.to_owned());
        serde_json::to_vec(&v).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn profile(yaml: &str) -> ApiProfile {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn output_budget() {
        let p = profile("output: {max_tokens: {cap: 100, default: 10}}");
        let edit = |field, value, replaces| {
            Ok(Some(BudgetEdit {
                field,
                value,
                replaces,
            }))
        };
        use BudgetField::*;
        assert_eq!(p.output_budget(None, None), edit(MaxTokens, 10, false));
        assert_eq!(p.output_budget(None, Some(100)), Ok(None));
        assert!(p
            .output_budget(Some(101), Some(5))
            .unwrap_err()
            .contains("max_completion_tokens must be at most 100"));

        let p = profile("output: {max_tokens: {cap: 100, on_exceed: clamp}}");
        assert_eq!(p.output_budget(None, None), edit(MaxTokens, 100, false));
        assert_eq!(p.output_budget(None, Some(500)), edit(MaxTokens, 100, true));
        assert_eq!(
            p.output_budget(Some(500), None),
            edit(MaxCompletionTokens, 100, true)
        );
        assert_eq!(ApiProfile::default().output_budget(None, Some(9)), Ok(None));
    }

    #[test]
    fn image_limit() {
        let p = profile("limits: {max_images: 1}");
        let body = |n: usize| {
            let parts: Vec<Value> = (0..n)
                .map(|_| json!({"type": "image_url", "image_url": {"url": "u"}}))
                .collect();
            json!({"messages": [{"role": "user", "content": parts}]}).to_string()
        };
        assert!(p.check_images(body(1).as_bytes()).is_ok());
        assert!(p
            .check_images(body(2).as_bytes())
            .unwrap_err()
            .contains("got 2"));
        assert!(p.check_images(b"not json").is_ok());
    }

    #[test]
    fn retypes_both_error_shapes() {
        let p = profile("errors: {bad_request_type: request_params_invalid}");
        let retyped = |b: Value| -> Value {
            serde_json::from_slice(&p.retype_bad_request(b.to_string().as_bytes()).unwrap())
                .unwrap()
        };
        assert_eq!(
            retyped(json!({"error": {"type": "invalid_request_error", "message": "m"}}))["error"]
                ["type"],
            "request_params_invalid"
        );
        assert_eq!(
            retyped(json!({"object": "error", "type": "BadRequestError", "message": "m"}))["type"],
            "request_params_invalid"
        );
        assert!(p.retype_bad_request(b"plain text").is_none());
        assert!(ApiProfile::default()
            .retype_bad_request(br#"{"error": {}}"#)
            .is_none());
    }
}
