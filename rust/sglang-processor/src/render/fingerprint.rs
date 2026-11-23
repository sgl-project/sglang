//! Shared startup digest for a render host and its tokens-in worker.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io;
use std::path::Path;

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::ChatFormatterOptions;
use super::selection::{ModelIdentity, SelectedFormatter, resolve_formatter};
use crate::model_files::{resolve_model_file, resolve_tokenizer_file};

/// Effective launch values after defaults and model overrides have been applied.
pub struct RenderFingerprintConfig<'a> {
    pub default_chat_template_kwargs: &'a HashMap<String, Value>,
    pub context_length: u64,
    pub reasoning_parser: Option<&'a str>,
    pub incremental_streaming_output: bool,
    /// Resolved multimodal token IDs and processor settings, absent on text workers.
    pub multimodal_token_config: Option<&'a Map<String, Value>>,
}

fn file_digest(path: &str) -> io::Result<String> {
    let mut hash = Sha256::new();
    let mut file = File::open(path)?;
    io::copy(&mut file, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

/// SHA-256 of compact UTF-8 JSON. Keys are recursively sorted except within
/// template kwarg values, whose object order is visible to Jinja. File inputs are
/// represented by their SHA-256 content digests. Built-in conversation
/// templates contribute their resolved definition; native renderers use a
/// stable identifier. Bump the schema for incompatible renderer changes.
pub fn render_fingerprint(
    options: &ChatFormatterOptions,
    config: &RenderFingerprintConfig<'_>,
) -> io::Result<String> {
    let identity = options.identity();
    let selected = resolve_formatter(options, &identity);
    fingerprint_with_formatter(options, config, &identity, selected)
}

fn fingerprint_with_formatter(
    options: &ChatFormatterOptions,
    config: &RenderFingerprintConfig<'_>,
    identity: &ModelIdentity,
    selected: SelectedFormatter,
) -> io::Result<String> {
    let revision = options.revision.as_deref();
    let tokenizer = resolve_tokenizer_file(&options.tokenizer_path, revision).ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, "render tokenizer file not found")
    })?;
    let mut files = Map::new();
    for name in [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ] {
        if let Some(path) = resolve_model_file(&options.tokenizer_path, revision, name) {
            files.insert(name.into(), Value::String(file_digest(&path)?));
        }
    }
    let huggingface = tokenizer.ends_with(".json");
    // Tiktoken selects its BPE pattern from the tokenizer's sibling config.
    // The checkpoint may live in a different directory.
    if !huggingface
        && let Some(path) = resolve_model_file(&options.tokenizer_path, revision, "config.json")
    {
        files.insert("config.json".into(), Value::String(file_digest(&path)?));
    }
    let template = match selected {
        SelectedFormatter::DeepSeekRecipe { v41 } => {
            let defaults = super::models::deepseek_v4_fingerprint_defaults(
                v41,
                config.default_chat_template_kwargs,
            );
            json!({
                "native": if v41 { "deepseek_recipe_v41" } else { "deepseek_recipe_v4" },
                "defaults": defaults,
            })
        }
        SelectedFormatter::Native(id, _) => {
            let id = id.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unrecognized native chat formatter",
                )
            })?;
            json!({"native": id})
        }
        SelectedFormatter::Builtin(formatter) => json!({"builtin": formatter.spec}),
        SelectedFormatter::Template {
            path: Some(path), ..
        } => {
            // The loader interprets Jinja and JSON overrides differently.
            json!({"jinja": Path::new(&path).extension().is_some_and(|e| e == "jinja"), "sha256": file_digest(&path)?})
        }
        SelectedFormatter::Template { path: None, .. } => Value::Null,
    };
    let mut inputs = json!({
        "schema_version": 1,
        "tokenizer_sha256": file_digest(&tokenizer)?,
        "tokenizer_format": if huggingface { "huggingface" } else { "tiktoken" },
        "files": files,
        "model_identity": identity,
        "chat_template": template,
        "default_chat_template_kwargs": null,
        "context_length": config.context_length,
        "reasoning_parser": config.reasoning_parser,
        "incremental_streaming_output": config.incremental_streaming_output,
        "multimodal_token_config": config.multimodal_token_config,
    });
    inputs.sort_all_objects();
    // Sort the HashMap of variable names, preserving each variable's value as Jinja sees it.
    let kwargs: BTreeMap<_, _> = config.default_chat_template_kwargs.iter().collect();
    inputs["default_chat_template_kwargs"] = json!(kwargs);
    Ok(format!(
        "{:x}",
        Sha256::digest(inputs.to_string().as_bytes())
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(kwargs: &HashMap<String, Value>) -> RenderFingerprintConfig<'_> {
        RenderFingerprintConfig {
            default_chat_template_kwargs: kwargs,
            context_length: 2048,
            incremental_streaming_output: false,
            reasoning_parser: None,
            multimodal_token_config: None,
        }
    }

    fn options(dir: &Path) -> ChatFormatterOptions {
        let path = dir.to_str().unwrap();
        ChatFormatterOptions {
            tokenizer_path: path.into(),
            model_path: path.into(),
            ..Default::default()
        }
    }

    #[test]
    fn worker_mm_render_config_matches_the_shared_v1_fingerprint() {
        // Python checks the worker's actual ServerArgs against the same vector.
        #[derive(serde::Deserialize)]
        struct Vector {
            model_files: HashMap<String, String>,
            context_length: u64,
            reasoning_parser: Option<String>,
            incremental_streaming_output: bool,
            default_chat_template_kwargs: HashMap<String, Value>,
            mm_render_config: Map<String, Value>,
            expected_fingerprint: String,
        }
        let vector: Vector = serde_json::from_str(include_str!(
            "../../tests/fixtures/render_fingerprint_mm_v1.json"
        ))
        .unwrap();
        let root = std::env::temp_dir().join(format!("render-mm-v1-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        for (name, contents) in &vector.model_files {
            std::fs::write(root.join(name), contents).unwrap();
        }
        let options = options(&root);
        let digest = |mm: &Map<String, Value>| {
            render_fingerprint(
                &options,
                &RenderFingerprintConfig {
                    default_chat_template_kwargs: &vector.default_chat_template_kwargs,
                    context_length: vector.context_length,
                    reasoning_parser: vector.reasoning_parser.as_deref(),
                    incremental_streaming_output: vector.incremental_streaming_output,
                    multimodal_token_config: Some(mm),
                },
            )
            .unwrap()
        };
        let mut mm = vector.mm_render_config;
        assert_eq!(digest(&mm), vector.expected_fingerprint);
        // Lower may construct its object in a different order.
        mm.sort_keys();
        assert_eq!(digest(&mm), vector.expected_fingerprint);
        // The worker emits explicit null for an unavailable optional token ID.
        let mut missing = mm.clone();
        assert_eq!(missing.remove("video_token_id"), Some(Value::Null));
        assert_ne!(digest(&missing), vector.expected_fingerprint);
        mm.insert("resample".into(), json!("aten_u8"));
        assert_ne!(digest(&mm), vector.expected_fingerprint);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fingerprint_tracks_render_inputs_and_is_independent_of_install_directory() {
        let root = std::env::temp_dir().join(format!("render-fingerprint-{}", std::process::id()));
        let first = root.join("first");
        let second = root.join("second");
        let kwargs = HashMap::from([("chat".into(), json!({"a": 1, "z": 0}))]);
        let config = config(&kwargs);
        for dir in [&first, &second] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
            std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
            std::fs::write(dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
        }
        let digest = render_fingerprint(&options(&first), &config).unwrap();
        // Pin the v1 encoding for future callers, including Lower.
        assert_eq!(
            digest,
            "bf97c5922f99c9ff8563c9963b9aefec5e2a4380210aabccfc921e8b249adc53"
        );
        let options = options(&second);
        assert_eq!(digest, render_fingerprint(&options, &config).unwrap());
        for changed in [
            RenderFingerprintConfig {
                context_length: 4096,
                ..config
            },
            RenderFingerprintConfig {
                reasoning_parser: Some("qwen3"),
                ..config
            },
            RenderFingerprintConfig {
                incremental_streaming_output: true,
                ..config
            },
            RenderFingerprintConfig {
                multimodal_token_config: Some(&Map::new()),
                ..config
            },
            RenderFingerprintConfig {
                default_chat_template_kwargs: &HashMap::new(),
                ..config
            },
        ] {
            assert_ne!(digest, render_fingerprint(&options, &changed).unwrap());
        }
        for file in ["tokenizer.json", "chat_template.jinja"] {
            let path = second.join(file);
            let previous = std::fs::read(&path).unwrap();
            std::fs::write(&path, "changed").unwrap();
            assert_ne!(digest, render_fingerprint(&options, &config).unwrap());
            std::fs::write(path, previous).unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fingerprint_preserves_jinja_iteration_order_in_default_kwargs() {
        let root = std::env::temp_dir().join(format!("render-kwargs-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            r#"{"chat_template":"{% for k in fields %}{{ k }}{% endfor %}|{% for k in rows[0] %}{{ k }}{% endfor %}"}"#,
        )
        .unwrap();
        let options = options(&root);
        let formatter = crate::select_chat_formatter(&options).0.unwrap();
        let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        let request = serde_json::from_value(body.clone()).unwrap();
        let render_and_hash = |kwargs: Value, expected: &str| {
            let kwargs = serde_json::from_value(kwargs).unwrap();
            let args = crate::TemplateArgs::new(&body, &kwargs);
            let (prompt, _) = formatter.render_template(&request, &args).unwrap();
            assert_eq!(prompt.as_str(), expected);
            render_fingerprint(&options, &config(&kwargs)).unwrap()
        };
        let az = json!({"a": 1, "z": 0});
        let za = json!({"z": 0, "a": 1});
        let digest = render_and_hash(json!({"fields": az, "rows": [az]}), "az|az");
        assert_eq!(
            digest,
            render_and_hash(json!({"rows": [az], "fields": az}), "az|az")
        );
        assert_ne!(
            digest,
            render_and_hash(json!({"fields": za, "rows": [az]}), "za|az")
        );
        assert_ne!(
            digest,
            render_and_hash(json!({"fields": az, "rows": [za]}), "az|za")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn native_fingerprint_uses_the_selected_family_across_install_directories() {
        let root = std::env::temp_dir().join(format!("render-native-{}", std::process::id()));
        let first = root.join("first").join("DeepSeek-V3.2");
        let second = root.join("second").join("DeepSeek-V3.2-copy");
        let kwargs = HashMap::new();
        let config = config(&kwargs);
        for dir in [&first, &second] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        }
        let first_options = options(&first);
        assert!(matches!(
            resolve_formatter(&first_options, &first_options.identity()),
            SelectedFormatter::Native(Some("deepseek_v32"), _)
        ));
        let digest = render_fingerprint(&first_options, &config).unwrap();
        assert_eq!(
            digest,
            render_fingerprint(&options(&second), &config).unwrap()
        );
        // A checkpoint template wins over the name-only DeepSeek fallback.
        let generic = root.join("generic");
        std::fs::create_dir_all(&generic).unwrap();
        std::fs::write(generic.join("tokenizer.json"), "{}").unwrap();
        for dir in [&second, &generic] {
            std::fs::write(
                dir.join("tokenizer_config.json"),
                r#"{"chat_template":"hello"}"#,
            )
            .unwrap();
        }
        assert_eq!(
            render_fingerprint(&options(&second), &config).unwrap(),
            render_fingerprint(&options(&generic), &config).unwrap()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn builtin_fingerprint_tracks_changes_to_the_same_template_definition() {
        let root =
            std::env::temp_dir().join(format!("render-builtin-vicuna-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        let kwargs = HashMap::new();
        let config = config(&kwargs);
        let mut options = options(&root);
        let digest = render_fingerprint(&options, &config).unwrap();
        options.chat_template = Some("vicuna_v1.1".into());
        assert_eq!(digest, render_fingerprint(&options, &config).unwrap());
        let identity = options.identity();
        let SelectedFormatter::Builtin(mut formatter) = resolve_formatter(&options, &identity)
        else {
            panic!("expected the registered vicuna template");
        };
        formatter.spec.sep.push('!');
        let changed = fingerprint_with_formatter(
            &options,
            &config,
            &identity,
            SelectedFormatter::Builtin(formatter),
        )
        .unwrap();
        assert_ne!(digest, changed);
        std::fs::remove_dir_all(root).unwrap();
    }
}
