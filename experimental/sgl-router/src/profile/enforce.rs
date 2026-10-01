// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Applying a profile to a chat request body, before admission.

use std::borrow::Cow;
use std::collections::BTreeMap;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;

use super::{lax_number, number, ApiProfile, OnExceed};
use crate::server::error::ApiError;

/// The (possibly rewritten) body and what the reply conversion must do.
#[derive(Debug)]
pub struct Applied {
    pub body: Bytes,
    /// `reasoning_format: general`: reply with `reasoning` instead of `reasoning_content`.
    pub reasoning_general: bool,
}

impl ApiProfile {
    fn is_passthrough(&self) -> bool {
        self.models.aliases.is_empty()
            && self.limits.max_images.is_none()
            && self.output.max_tokens.is_none()
            && self.params.is_empty()
    }

    /// Validate and rewrite a chat request. Only top-level fields are parsed;
    /// the body is re-serialized only if something changed.
    pub fn apply(&self, body: Bytes, model_id: &str) -> Result<Applied, ApiError> {
        if self.is_passthrough() {
            return Ok(Applied {
                body,
                reasoning_general: false,
            });
        }
        let top: BTreeMap<String, &RawValue> = serde_json::from_slice(&body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".into())
        })?;
        let get = |k: &str| {
            top.get(k)
                .and_then(|r| serde_json::from_str::<Value>(r.get()).ok())
                .filter(|v| !v.is_null())
        };
        let mut set: BTreeMap<String, Value> = BTreeMap::new();

        if let Some(Value::String(m)) = get("model") {
            if self.models.aliases.contains(&m) {
                set.insert("model".into(), model_id.into());
            }
        }

        for (name, rule) in &self.params {
            match get(name) {
                Some(v) => {
                    if let Some(why) = rule.check(name, &v) {
                        return Err(ApiError::BadRequest(why));
                    }
                    let n = rule.normalized(v.clone());
                    if n != v {
                        set.insert(name.clone(), n);
                    }
                }
                None => {
                    if let Some(d) = rule.pin.clone().or_else(|| rule.default.clone()) {
                        set.insert(name.clone(), rule.normalized(d));
                    }
                }
            }
        }

        if let Some(mt) = &self.output.max_tokens {
            // Engine precedence: `max_completion_tokens` unless it is zero.
            let asked = get("max_completion_tokens")
                .filter(|v| lax_number(v) != Some(0.0))
                .map(|v| ("max_completion_tokens", v))
                .or_else(|| get("max_tokens").map(|v| ("max_tokens", v)));
            match asked {
                None => {
                    set.insert("max_tokens".into(), mt.default.unwrap_or(mt.cap).into());
                }
                Some((field, v)) if lax_number(&v).is_some_and(|n| n > mt.cap as f64) => match mt.on_exceed {
                    OnExceed::Reject => {
                        return Err(ApiError::BadRequest(format!(
                            "max_tokens is too large: {v}. This model supports at most {} completion tokens.",
                            mt.cap
                        )))
                    }
                    OnExceed::Clamp => {
                        set.insert(field.into(), number(mt.cap as f64));
                    }
                },
                Some(_) => {}
            }
        }

        if let (Some(max), Some(messages)) = (self.limits.max_images, top.get("messages")) {
            let n = count_images(messages);
            if n > max {
                return Err(ApiError::BadRequest(format!(
                    "too many images: {n}; at most {max} are allowed per request"
                )));
            }
        }

        let reasoning_general = self.params.contains_key("reasoning_format")
            && get("reasoning_format").as_ref().and_then(Value::as_str) == Some("general");

        if set.is_empty() {
            return Ok(Applied {
                body,
                reasoning_general,
            });
        }
        #[derive(Serialize)]
        #[serde(untagged)]
        enum Field<'a> {
            Raw(&'a RawValue),
            New(Value),
        }
        let mut out: BTreeMap<&str, Field> = top
            .iter()
            .map(|(k, v)| (k.as_str(), Field::Raw(v)))
            .collect();
        for (k, v) in &set {
            out.insert(k, Field::New(v.clone()));
        }
        let bytes = serde_json::to_vec(&out)
            .map_err(|e| ApiError::Internal(anyhow::anyhow!("re-serialize request body: {e}")))?;
        Ok(Applied {
            body: Bytes::from(bytes),
            reasoning_general,
        })
    }
}

/// `image_url` parts across all messages; malformed shapes count as 0 (the engine rejects them).
fn count_images(messages: &RawValue) -> usize {
    #[derive(Deserialize)]
    struct Msg<'a> {
        #[serde(borrow)]
        content: Option<&'a RawValue>,
    }
    #[derive(Deserialize)]
    struct Part<'a> {
        #[serde(rename = "type", borrow)]
        ty: Option<Cow<'a, str>>,
    }
    let Ok(msgs) = serde_json::from_str::<Vec<Msg>>(messages.get()) else {
        return 0;
    };
    msgs.iter()
        .filter_map(|m| m.content)
        .filter_map(|c| serde_json::from_str::<Vec<Part>>(c.get()).ok())
        .flatten()
        .filter(|p| p.ty.as_deref() == Some("image_url"))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{MaxTokens, ParamRule, ParamType};
    use serde_json::json;

    fn profile(yaml: &str) -> ApiProfile {
        let p: ApiProfile = serde_yaml::from_str(yaml).unwrap();
        p.validate().unwrap();
        p
    }

    fn run(p: &ApiProfile, body: Value) -> Result<Value, String> {
        p.apply(Bytes::from(body.to_string()), "m")
            .map(|a| serde_json::from_slice(&a.body).unwrap())
            .map_err(|e| e.to_string())
    }

    #[test]
    fn passthrough_keeps_the_exact_bytes() {
        let body = Bytes::from_static(br#"{"model":"m",  "messages":[]}"#);
        let out = ApiProfile::default().apply(body.clone(), "m").unwrap();
        assert_eq!(out.body, body);
        let p = profile("params:\n  temperature: {min: 0, max: 2}\n");
        assert_eq!(
            p.apply(body.clone(), "m").unwrap().body,
            body,
            "nothing changed, nothing re-serialized"
        );
    }

    #[test]
    fn ranges_defaults_normalize() {
        let p = profile(
            "params:\n  temperature: {default: 1.0, min: 0, max: 2}\n  top_p: {min: 0, max: 1, exclusive_min: true}\n  top_k: {min: 0, normalize: [[0, -1]]}\n  n: {type: int, min: 1, max: 8}\n",
        );
        let out = run(&p, json!({"model": "m", "top_k": 0})).unwrap();
        assert_eq!(out["temperature"], 1.0);
        assert_eq!(out["top_k"], -1);
        assert!(
            out.get("top_p").is_none(),
            "a rule without default injects nothing"
        );
        assert_eq!(
            run(&p, json!({"model": "m", "top_k": 250})).unwrap()["top_k"],
            250
        );
        for (bad, needle) in [
            (json!({"temperature": 2.1}), "between 0 and 2"),
            (json!({"temperature": -0.1}), "temperature"),
            (json!({"top_p": 0}), "> 0"),
            (json!({"n": 1.5}), "integer"),
            (json!({"n": 0}), "n must be"),
            (json!({"top_k": "-5"}), "top_k"),
        ] {
            let err = run(&p, bad.clone()).unwrap_err();
            assert!(err.contains(needle), "{bad}: {err}");
        }
        // Not a number: the engine's to judge.
        assert!(run(&p, json!({"temperature": "hot"})).is_ok());
        // Explicit null counts as omitted.
        assert_eq!(
            run(&p, json!({ "temperature": null })).unwrap()["temperature"],
            1.0
        );
    }

    #[test]
    fn pins_match_laxly() {
        let mut p = ApiProfile::default();
        p.params.insert(
            "top_p".into(),
            ParamRule {
                pin: Some(json!(0.95)),
                ..Default::default()
            },
        );
        p.params.insert(
            "n".into(),
            ParamRule {
                pin: Some(json!(1)),
                ty: Some(ParamType::Int),
                ..Default::default()
            },
        );
        assert_eq!(run(&p, json!({})).unwrap(), json!({"top_p": 0.95, "n": 1}));
        for ok in [
            json!({"top_p": 0.95}),
            json!({"top_p": "0.95"}),
            json!({"n": 1.0}),
        ] {
            assert!(run(&p, ok.clone()).is_ok(), "{ok}");
        }
        let err = run(&p, json!({"top_p": 0.8})).unwrap_err();
        assert!(err.contains("immutable"), "{err}");
    }

    #[test]
    fn enum_params() {
        let p = profile("params:\n  reasoning_format: {values: [general, deepseek-style]}\n");
        let a = p
            .apply(Bytes::from(r#"{"reasoning_format":"general"}"#), "m")
            .unwrap();
        assert!(a.reasoning_general);
        assert!(
            !p.apply(Bytes::from(r#"{"reasoning_format":"deepseek-style"}"#), "m")
                .unwrap()
                .reasoning_general
        );
        let err = run(&p, json!({"reasoning_format": "nope"})).unwrap_err();
        assert!(err.contains("must be one of"), "{err}");
    }

    #[test]
    fn output_budget() {
        let mut p = ApiProfile::default();
        p.output.max_tokens = Some(MaxTokens {
            cap: 100,
            on_exceed: OnExceed::Reject,
            default: None,
        });
        assert_eq!(run(&p, json!({})).unwrap()["max_tokens"], 100);
        assert_eq!(
            run(&p, json!({ "max_tokens": null })).unwrap()["max_tokens"],
            100
        );
        assert_eq!(
            run(&p, json!({"max_tokens": 100})).unwrap()["max_tokens"],
            100
        );
        // max_completion_tokens wins; a zero one falls through to max_tokens.
        assert!(run(&p, json!({"max_tokens": 999, "max_completion_tokens": 50})).is_ok());
        assert!(run(&p, json!({"max_tokens": 50, "max_completion_tokens": 999})).is_err());
        assert!(run(&p, json!({"max_completion_tokens": 0, "max_tokens": 999})).is_err());
        // Coercible over-cap values reject; uncoercible ones are the engine's.
        let err = run(&p, json!({"max_tokens": "999"})).unwrap_err();
        assert!(err.contains("999") && err.contains("100"), "{err}");
        assert!(run(&p, json!({"max_tokens": "large"})).is_ok());

        p.output.max_tokens = Some(MaxTokens {
            cap: 100,
            on_exceed: OnExceed::Clamp,
            default: Some(10),
        });
        assert_eq!(run(&p, json!({})).unwrap()["max_tokens"], 10);
        assert_eq!(
            run(&p, json!({"max_tokens": 70000})).unwrap()["max_tokens"],
            100
        );
        let out = run(&p, json!({"max_completion_tokens": 70000})).unwrap();
        assert_eq!(out["max_completion_tokens"], 100);
    }

    #[test]
    fn aliases_and_image_limit() {
        let p = profile("models: {aliases: [step-5-preview]}\nlimits: {max_images: 2}\n");
        assert_eq!(
            run(&p, json!({"model": "step-5-preview"})).unwrap()["model"],
            "m"
        );
        assert_eq!(
            run(&p, json!({"model": "other"})).unwrap()["model"],
            "other"
        );
        let img = json!({"type": "image_url", "image_url": {"url": "data:,"}});
        let msgs = |n: usize| json!([{"role": "user", "content": vec![img.clone(); n]}, {"role": "user", "content": "x"}]);
        assert!(run(&p, json!({"messages": msgs(2)})).is_ok());
        let err = run(&p, json!({"messages": msgs(3)})).unwrap_err();
        assert!(err.contains("too many images: 3"), "{err}");
    }
}
