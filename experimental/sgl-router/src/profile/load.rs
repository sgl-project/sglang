// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Profile resolution: presets, files, `extends` merging, legacy flags.

use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};

use super::{ApiProfile, MaxTokens, OnExceed, ParamRule, ParamType};
use crate::config::{ConflictPolicy, ParamSpec, SamplingOverrides};

pub const ENV: &str = "SGLANG_ROUTER_API_PROFILE";
const IMAGE_PROFILE: &str = "/etc/sgl-router/profile.yaml";
const DEFAULT_PRESET: &str = "openai-compatible";

const PRESETS: &[(&str, &str)] = &[
    (
        "openai-compatible",
        include_str!("../../profiles/openai-compatible.yaml"),
    ),
    (
        "moonshot-kimi",
        include_str!("../../profiles/moonshot-kimi.yaml"),
    ),
    (
        "stepfun-step5",
        include_str!("../../profiles/stepfun-step5.yaml"),
    ),
];

pub fn preset_names() -> impl Iterator<Item = &'static str> {
    PRESETS.iter().map(|(n, _)| *n)
}

enum Spec {
    File(PathBuf),
    Preset(String),
}

impl Spec {
    /// A path if it looks like one, else a preset name.
    fn parse(s: &str, base: Option<&Path>) -> Self {
        if s.contains('/') || s.ends_with(".yaml") || s.ends_with(".yml") {
            Spec::File(base.map_or_else(|| PathBuf::from(s), |b| b.join(s)))
        } else {
            Spec::Preset(s.to_owned())
        }
    }
}

/// First match wins: file flag, preset flag, env var, image file, default preset.
/// Legacy flags are applied on top.
pub fn resolve(
    file: Option<&Path>,
    preset: Option<&str>,
    max_output_tokens: Option<NonZeroU64>,
    overrides: &SamplingOverrides,
) -> Result<ApiProfile> {
    let env = std::env::var(ENV).ok().filter(|e| !e.is_empty());
    let (spec, origin) = select(file, preset, env.as_deref(), Path::new(IMAGE_PROFILE));
    let mut profile: ApiProfile = serde_json::from_value(load(&spec, &mut Vec::new())?)
        .map_err(|e| anyhow!("api profile ({origin}): {e}"))?;
    apply_legacy(&mut profile, max_output_tokens, overrides);
    profile
        .validate()
        .with_context(|| format!("api profile ({origin})"))?;
    profile.origin = origin;
    Ok(profile)
}

fn select(
    file: Option<&Path>,
    preset: Option<&str>,
    env: Option<&str>,
    image: &Path,
) -> (Spec, String) {
    if let Some(f) = file {
        (
            Spec::File(f.into()),
            format!("--api-profile-file {}", f.display()),
        )
    } else if let Some(p) = preset {
        (Spec::Preset(p.into()), format!("--api-profile {p}"))
    } else if let Some(e) = env {
        (Spec::parse(e, None), format!("{ENV}={e}"))
    } else if image.exists() {
        (Spec::File(image.into()), image.display().to_string())
    } else {
        (
            Spec::Preset(DEFAULT_PRESET.into()),
            "built-in default".into(),
        )
    }
}

/// Load one profile document with its `extends` chain merged in.
fn load(spec: &Spec, chain: &mut Vec<String>) -> Result<Value> {
    let (key, text, dir) = match spec {
        Spec::File(p) => (
            p.display().to_string(),
            std::fs::read_to_string(p)
                .with_context(|| format!("read api profile {}", p.display()))?,
            p.parent().map(Path::to_path_buf),
        ),
        Spec::Preset(n) => {
            let text = PRESETS
                .iter()
                .find(|(name, _)| name == n)
                .map(|(_, t)| *t)
                .ok_or_else(|| {
                    anyhow!(
                        "unknown api profile preset `{n}` (known: {})",
                        preset_names().collect::<Vec<_>>().join(", ")
                    )
                })?;
            (format!("preset {n}"), text.to_owned(), None)
        }
    };
    if chain.contains(&key) {
        bail!(
            "api profile `extends` cycle: {} -> {key}",
            chain.join(" -> ")
        );
    }
    let mut doc = match serde_yaml::from_str::<Value>(&text)
        .with_context(|| format!("parse api profile {key}"))?
    {
        Value::Null => Value::Object(Map::new()),
        v @ Value::Object(_) => v,
        _ => bail!("api profile {key}: top level must be a mapping"),
    };
    let parent = match doc.as_object_mut().and_then(|m| m.remove("extends")) {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::String(s)) => {
            chain.push(key);
            let parent = load(&Spec::parse(&s, dir.as_deref()), chain)?;
            chain.pop();
            parent
        }
        Some(other) => bail!("api profile {key}: `extends` must be a string, got {other}"),
    };
    let mut merged = parent;
    merge(&mut merged, doc);
    Ok(merged)
}

/// Maps merge key by key, everything else replaces, `null` deletes.
fn merge(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Object(b), Value::Object(o)) => {
            for (k, v) in o {
                if v.is_null() {
                    b.remove(&k);
                    continue;
                }
                let slot = b.entry(k).or_insert(Value::Null);
                if v.is_object() && !slot.is_object() {
                    *slot = Value::Object(Map::new());
                }
                merge(slot, v);
            }
        }
        (b, o) => *b = o,
    }
}

/// `--max-output-tokens` and `--override-sampling-params` replace the
/// profile's settings for what they name.
fn apply_legacy(p: &mut ApiProfile, cap: Option<NonZeroU64>, o: &SamplingOverrides) {
    if let Some(cap) = cap {
        let prev = p.output.max_tokens.take();
        p.output.max_tokens = Some(MaxTokens {
            cap: cap.get(),
            on_exceed: prev.as_ref().map_or(OnExceed::Reject, |m| m.on_exceed),
            default: prev.and_then(|m| m.default).filter(|d| *d <= cap.get()),
        });
    }
    for (field, spec) in &o.params {
        let mut rule = ParamRule {
            ty: field.is_integral().then_some(ParamType::Int),
            ..Default::default()
        };
        match (spec, o.conflict) {
            (ParamSpec::Exact(n), ConflictPolicy::Reject) => {
                rule.pin = Some(Value::Number(n.clone()))
            }
            (ParamSpec::Exact(n), ConflictPolicy::Allow) => {
                rule.default = Some(Value::Number(n.clone()))
            }
            (ParamSpec::Range { lo, hi }, _) => (rule.min, rule.max) = (Some(*lo), Some(*hi)),
        }
        p.params.insert(field.wire_name().to_owned(), rule);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn from_yaml(files: &[(&str, &str)], start: &str) -> Result<ApiProfile> {
        let dir = tempdir();
        for (name, body) in files {
            std::fs::write(dir.join(name), body).unwrap();
        }
        let v = load(&Spec::File(dir.join(start)), &mut Vec::new())?;
        let p: ApiProfile = serde_json::from_value(v)?;
        p.validate()?;
        Ok(p)
    }

    fn tempdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("api-profile-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn every_preset_loads_and_validates() {
        for name in preset_names() {
            let v = load(&Spec::Preset(name.into()), &mut Vec::new()).unwrap();
            let p: ApiProfile = serde_json::from_value(v).unwrap_or_else(|e| panic!("{name}: {e}"));
            p.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(p.name, name);
        }
    }

    #[test]
    fn default_preset_is_a_no_op() {
        let v = load(&Spec::Preset(DEFAULT_PRESET.into()), &mut Vec::new()).unwrap();
        let p: ApiProfile = serde_json::from_value(v).unwrap();
        assert_eq!(
            p,
            ApiProfile {
                name: DEFAULT_PRESET.into(),
                ..Default::default()
            }
        );
    }

    #[test]
    fn extends_merges_maps_replaces_scalars_and_null_deletes() {
        let p = from_yaml(
            &[
                ("base.yaml", "name: base\nparams:\n  temperature: {default: 1.0, min: 0, max: 2}\n  n: {min: 1}\n"),
                ("child.yaml", "extends: base.yaml\nname: child\nparams:\n  temperature: {max: 1.5}\n  n: null\n"),
            ],
            "child.yaml",
        )
        .unwrap();
        assert_eq!(p.name, "child");
        let t = &p.params["temperature"];
        assert_eq!(
            (t.default.clone(), t.min, t.max),
            (Some(json!(1.0)), Some(0.0), Some(1.5))
        );
        assert!(!p.params.contains_key("n"));
    }

    #[test]
    fn extends_a_preset() {
        let p = from_yaml(
            &[(
                "p.yaml",
                "extends: stepfun-step5\noutput: {max_tokens: {cap: 32000}}\n",
            )],
            "p.yaml",
        )
        .unwrap();
        assert_eq!(p.output.max_tokens.unwrap().cap, 32000);
        assert!(p.params.contains_key("top_k"));
    }

    #[test]
    fn startup_errors() {
        for (body, needle) in [
            ("bogus: 1\n", "unknown field"),
            ("params:\n  t: {min: 2, max: 1}\n", "min 2 > max 1"),
            ("params:\n  t: {pin: 1, max: 2}\n", "cannot be combined"),
            ("params:\n  t: {default: 3, max: 2}\n", "default"),
            (
                "params:\n  t: {min: 1, normalize: [[0, -1]]}\n",
                "never accepted",
            ),
            ("limits: {max_body_bytes: 12XB}\n", "unknown size unit"),
            ("extends: nope\n", "unknown api profile preset"),
        ] {
            let err = format!(
                "{:#}",
                from_yaml(&[("p.yaml", body)], "p.yaml").unwrap_err()
            );
            assert!(err.contains(needle), "{body}: {err}");
        }
        let err = from_yaml(
            &[
                ("a.yaml", "extends: b.yaml\n"),
                ("b.yaml", "extends: a.yaml\n"),
            ],
            "a.yaml",
        )
        .unwrap_err();
        assert!(err.to_string().contains("cycle"), "{err}");
    }

    #[test]
    fn selection_order() {
        let img = tempdir().join("profile.yaml");
        let origin = |f, p, e| select(f, p, e, &img).1;
        assert!(origin(Some(Path::new("/x.yaml")), Some("p"), Some("e"))
            .starts_with("--api-profile-file"));
        assert!(origin(None, Some("p"), Some("e")).starts_with("--api-profile"));
        assert!(origin(None, None, Some("e")).starts_with(ENV));
        assert_eq!(origin(None, None, None), "built-in default");
        std::fs::write(&img, "").unwrap();
        assert_eq!(origin(None, None, None), img.display().to_string());
    }

    #[test]
    fn legacy_flags_replace_rules() {
        let mut p = ApiProfile::default();
        let o = crate::config::cli::parse_sampling_overrides(
            r#"{"top_p": 0.95, "temperature": {"min": 0, "max": 1}}"#,
            ConflictPolicy::Reject,
        )
        .unwrap();
        apply_legacy(&mut p, NonZeroU64::new(64), &o);
        assert_eq!(p.params["top_p"].pin, Some(json!(0.95)));
        assert_eq!(
            (p.params["temperature"].min, p.params["temperature"].max),
            (Some(0.0), Some(1.0))
        );
        assert_eq!(
            p.output.max_tokens.unwrap(),
            MaxTokens {
                cap: 64,
                on_exceed: OnExceed::Reject,
                default: None
            }
        );
    }
}
