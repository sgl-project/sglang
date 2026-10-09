// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Profile resolution: presets, files and `extends` merging.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};

use super::ApiProfile;

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
pub fn resolve(file: Option<&Path>, preset: Option<&str>) -> Result<ApiProfile> {
    let env = std::env::var(ENV).ok().filter(|e| !e.is_empty());
    let (spec, origin) = select(file, preset, env.as_deref(), Path::new(IMAGE_PROFILE));
    let mut profile: ApiProfile = serde_json::from_value(load(&spec, &mut Vec::new())?)
        .map_err(|e| anyhow!("api profile ({origin}): {e}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
                (
                    "base.yaml",
                    "name: base\nlimits: {max_images: 8, max_body_bytes: 1MiB}\n\
                     models: {aliases: [a, b]}\n",
                ),
                (
                    "child.yaml",
                    "extends: base.yaml\nname: child\nlimits: {max_images: null}\n\
                     models: {aliases: [c]}\n",
                ),
            ],
            "child.yaml",
        )
        .unwrap();
        assert_eq!(p.name, "child");
        assert_eq!(p.limits.max_images, None);
        assert_eq!(p.limits.max_body_bytes, Some(1 << 20));
        assert_eq!(p.models.aliases, ["c"]);
    }

    #[test]
    fn extends_a_preset() {
        let p = from_yaml(
            &[(
                "p.yaml",
                "extends: moonshot-kimi\noutput: {max_tokens: {cap: 32000}}\n",
            )],
            "p.yaml",
        )
        .unwrap();
        assert_eq!(p.output.max_tokens.as_ref().unwrap().cap, 32000);
        assert_eq!(p.sampling_overrides().unwrap().params.len(), 6);
    }

    #[test]
    fn startup_errors() {
        for (body, needle) in [
            ("bogus: 1\n", "unknown field"),
            ("sampling: {params: {top_p: 7}}\n", "top_p"),
            ("output: {max_tokens: {cap: 8, default: 9}}\n", "default"),
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
}
