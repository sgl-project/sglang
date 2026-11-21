// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Applying a profile to a chat request body, before admission.

use std::borrow::Cow;
use std::collections::BTreeMap;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;

use super::{lax_number, number, ApiProfile, OnExceed, OnInvalid};
use crate::server::error::ApiError;

/// The (possibly rewritten) body and what the reply conversion must do.
#[derive(Debug)]
pub struct Applied {
    pub body: Bytes,
    /// `reasoning_format: general`: reply with `reasoning` instead of `reasoning_content`.
    pub reasoning_general: bool,
    pub reasoning_effort: Option<Value>,
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
                reasoning_effort: None,
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

        let mut nested_effort = if self.raw_reasoning_effort() {
            take_nested_effort(&get, &mut set)
        } else {
            None
        };
        let mut reasoning_effort = None;
        for (name, rule) in &self.params {
            let nested = if name == EFFORT {
                nested_effort.take()
            } else {
                None
            };
            let moved = nested.is_some();
            let effective = match nested.or_else(|| get(name)) {
                Some(v) => {
                    let n = match rule.check(name, &v) {
                        None => rule.normalized(v.clone()),
                        Some(why) => match (rule.on_invalid, &rule.default) {
                            (OnInvalid::Default, Some(d)) => rule.normalized(d.clone()),
                            _ => return Err(ApiError::BadRequest(why)),
                        },
                    };
                    if moved || n != v {
                        set.insert(name.clone(), n.clone());
                    }
                    Some(n)
                }
                None => {
                    let d = rule
                        .pin
                        .clone()
                        .or_else(|| rule.default.clone())
                        .map(|d| rule.normalized(d));
                    if let Some(d) = &d {
                        set.insert(name.clone(), d.clone());
                    }
                    d
                }
            };
            if name == EFFORT {
                reasoning_effort = effective;
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
                reasoning_effort,
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
            reasoning_effort,
        })
    }
}

const EFFORT: &str = "reasoning_effort";

fn take_nested_effort(
    get: &impl Fn(&str) -> Option<Value>,
    set: &mut BTreeMap<String, Value>,
) -> Option<Value> {
    let mut found = Vec::new();
    for (field, keys) in [
        ("chat_template_kwargs", &[EFFORT][..]),
        ("reasoning", &["effort", EFFORT][..]),
    ] {
        let Some(Value::Object(mut obj)) = get(field) else {
            continue;
        };
        let mut hit = false;
        for key in keys {
            if let Some(v) = obj.remove(*key) {
                hit = true;
                if !v.is_null() {
                    found.push(v);
                }
            }
        }
        if hit {
            let rest = if obj.is_empty() {
                Value::Null
            } else {
                Value::Object(obj)
            };
            set.insert(field.into(), rest);
        }
    }
    found.into_iter().next()
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

    #[test]
    fn on_invalid_default_replaces_rejected_values() {
        let p = profile(
            "params:\n  temperature: {default: 1.0, min: 0, max: 2, on_invalid: default}\n  top_k: {default: 0, min: 0, normalize: [[0, -1]], on_invalid: default}\n",
        );
        assert_eq!(
            run(&p, json!({"temperature": 5})).unwrap()["temperature"],
            1.0
        );
        assert_eq!(
            run(&p, json!({"temperature": 0.5})).unwrap()["temperature"],
            0.5
        );
        assert_eq!(run(&p, json!({"top_k": -3})).unwrap()["top_k"], -1);
        let strict = profile("params:\n  reasoning_effort: {default: max, values: [low, max]}\n");
        let err = run(&strict, json!({"reasoning_effort": "MAX"})).unwrap_err();
        assert!(err.contains("must be one of"), "{err}");
    }

    fn stepfun_extended() -> ApiProfile {
        let dir =
            std::env::temp_dir().join(format!("api-profile-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("profile.yaml");
        std::fs::write(
            &file,
            "name: stepfun-step5-ext\nextends: stepfun-step5\nlimits:\n  max_images: 120\noutput:\n  max_tokens: { cap: 128000, on_exceed: clamp }\n",
        )
        .unwrap();
        crate::profile::resolve(Some(&file), None, None, &Default::default()).unwrap()
    }

    #[test]
    fn stepfun_reasoning_effort_on_every_protocol() {
        let p = stepfun_extended();
        assert_eq!(p.limits.max_images, Some(120));
        assert_eq!(p.output.max_tokens.as_ref().unwrap().cap, 128_000);
        assert!(p.raw_reasoning_effort());
        let cases = effort_cases();
        let msgs = json!([{"role": "user", "content": "hi"}]);
        for (effort, want) in cases {
            let mut chat = json!({"model": "m", "messages": msgs});
            let mut responses = json!({"model": "m", "input": "hi"});
            let mut messages = json!({"model": "m", "max_tokens": 64, "messages": msgs});
            if let Some(e) = &effort {
                chat["reasoning_effort"] = e.clone();
                responses["reasoning"] = json!({ "effort": e });
                messages["output_config"] = json!({ "effort": e });
            }
            let responses = crate::protocol::responses::to_chat(responses, true)
                .unwrap()
                .chat;
            let messages = crate::protocol::anthropic::to_chat(messages, false, true)
                .unwrap()
                .chat;
            for (api, body) in [
                ("chat", chat),
                ("responses", responses),
                ("messages", messages),
            ] {
                let out = run(&p, body).unwrap_or_else(|e| panic!("{api} {effort:?}: {e}"));
                assert_eq!(out["reasoning_effort"], want, "{api} {effort:?}");
            }
        }
    }

    fn effort_cases() -> Vec<(Option<Value>, &'static str)> {
        let mut cases: Vec<(Option<Value>, &str)> =
            ["none", "low", "medium", "high", "xhigh", "max"]
                .into_iter()
                .map(|l| (Some(json!(l)), l))
                .collect();
        for bad in [
            json!("MAX"),
            json!(""),
            json!(null),
            json!(5),
            json!(0.5),
            json!(true),
            json!("minimal"),
            json!(["max"]),
        ] {
            cases.push((Some(bad), "max"));
        }
        cases.push((None, "max"));
        cases
    }

    fn effort_of(p: &ApiProfile, body: Value) -> (Value, Option<Value>) {
        let a = p.apply(Bytes::from(body.to_string()), "m").unwrap();
        (serde_json::from_slice(&a.body).unwrap(), a.reasoning_effort)
    }

    #[test]
    fn stepfun_reasoning_effort_from_nested_fields() {
        let p = stepfun_extended();
        for (effort, want) in effort_cases() {
            let Some(e) = effort else { continue };
            for body in [
                json!({"chat_template_kwargs": {"reasoning_effort": e, "thinking": true}}),
                json!({"reasoning": {"effort": e}}),
                json!({"reasoning": {"reasoning_effort": e, "enabled": true}}),
            ] {
                let (out, applied) = effort_of(&p, body.clone());
                assert_eq!(out["reasoning_effort"], want, "{body}");
                assert_eq!(applied, Some(json!(want)), "{body}");
                assert!(
                    out["chat_template_kwargs"]
                        .get("reasoning_effort")
                        .is_none(),
                    "{body}: {out}"
                );
                assert!(out["reasoning"].get("effort").is_none(), "{body}: {out}");
                assert!(
                    out["reasoning"].get("reasoning_effort").is_none(),
                    "{body}: {out}"
                );
            }
        }
    }

    #[test]
    fn nested_effort_follows_engine_precedence() {
        let p = stepfun_extended();
        for (body, want) in [
            (
                json!({"reasoning_effort": "low", "reasoning": {"effort": "high"}}),
                "high",
            ),
            (
                json!({"reasoning_effort": "low", "reasoning": {"effort": "high"},
                       "chat_template_kwargs": {"reasoning_effort": "medium"}}),
                "medium",
            ),
            (
                json!({"reasoning": {"effort": null, "reasoning_effort": "xhigh"}}),
                "xhigh",
            ),
            (
                json!({"reasoning": {"effort": "none", "reasoning_effort": "xhigh"}}),
                "none",
            ),
            (
                json!({"reasoning_effort": "low", "reasoning": {"effort": "MAX"}}),
                "max",
            ),
            (
                json!({"reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": null}}),
                "none",
            ),
            (
                json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": ""}}),
                "max",
            ),
        ] {
            let (out, applied) = effort_of(&p, body.clone());
            assert_eq!(out["reasoning_effort"], want, "{body}");
            assert_eq!(applied, Some(json!(want)), "{body}");
        }
        let (out, _) = effort_of(
            &p,
            json!({"reasoning": {"effort": "low", "enabled": true},
                   "chat_template_kwargs": {"reasoning_effort": null, "thinking": true}}),
        );
        assert_eq!(out["reasoning"], json!({"enabled": true}));
        assert_eq!(out["chat_template_kwargs"], json!({"thinking": true}));
        let (out, _) = effort_of(
            &p,
            json!({"reasoning": {"effort": "low"}, "chat_template_kwargs": {"reasoning_effort": "high"}}),
        );
        assert!(
            out["reasoning"].is_null() && out["chat_template_kwargs"].is_null(),
            "{out}"
        );
        let (out, _) = effort_of(
            &p,
            json!({"reasoning": {"summary": "auto"}, "chat_template_kwargs": {}}),
        );
        assert_eq!(out["reasoning"], json!({"summary": "auto"}));
        assert_eq!(out["chat_template_kwargs"], json!({}));
        assert_eq!(out["reasoning_effort"], "max");
    }

    #[test]
    fn other_presets_keep_reasoning_effort_untouched() {
        for name in crate::profile::preset_names().filter(|n| *n != "stepfun-step5") {
            let p = crate::profile::resolve(None, Some(name), None, &Default::default()).unwrap();
            assert!(!p.raw_reasoning_effort(), "{name}");
            let out = run(&p, json!({"reasoning_effort": "MAX"})).unwrap();
            assert_eq!(out["reasoning_effort"], "MAX", "{name}");
            let body = json!({"reasoning_effort": "MAX", "reasoning": {"effort": "huge"},
                              "chat_template_kwargs": {"reasoning_effort": "x"}});
            let a = p.apply(Bytes::from(body.to_string()), "m").unwrap();
            assert!(a.reasoning_effort.is_none(), "{name}");
            let out: Value = serde_json::from_slice(&a.body).unwrap();
            assert_eq!(out["reasoning"], body["reasoning"], "{name}");
            assert_eq!(
                out["chat_template_kwargs"], body["chat_template_kwargs"],
                "{name}"
            );
            assert!(
                run(&p, json!({}))
                    .unwrap()
                    .get("reasoning_effort")
                    .is_none(),
                "{name}"
            );
        }
    }
}
