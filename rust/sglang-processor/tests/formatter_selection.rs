//! Selection through the same entry point as a rendering host, with local tokenizer files.

#![cfg(feature = "render")]

use std::{fs, path::PathBuf};

use dynamo_protocols::types::CreateChatCompletionRequest;
use serde_json::{Value, json};
use sglang_processor::{ChatFormatter, ChatFormatterOptions, select_chat_formatter};

fn checkpoint(label: &str) -> (PathBuf, ChatFormatterOptions) {
    let dir = std::env::temp_dir().join(format!("sglang-selection-{label}-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let options = ChatFormatterOptions {
        tokenizer_path: dir.to_str().unwrap().into(),
        ..Default::default()
    };
    (dir, options)
}

fn selected(options: &ChatFormatterOptions) -> ChatFormatter {
    let (formatter, error) = select_chat_formatter(options);
    assert_eq!(error, None, "{options:?}");
    formatter.unwrap()
}

fn request() -> Value {
    json!({"model": "test", "messages": [{"role": "user", "content": "Hi"}]})
}

fn prompt(options: &ChatFormatterOptions) -> String {
    let request: CreateChatCompletionRequest = serde_json::from_value(request()).unwrap();
    selected(options)
        .render_prompt(&request)
        .unwrap()
        .into_text()
}

fn assert_error(options: &ChatFormatterOptions, reason: &str) {
    let (formatter, error) = select_chat_formatter(options);
    assert!(formatter.is_none(), "{options:?}");
    assert_eq!(
        error,
        Some(format!("this model has no usable chat template: {reason}"))
    );
}

#[test]
fn model_type_renderers_precede_checkpoint_template() {
    let (dir, options) = checkpoint("model-type");
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"chat_template":"embedded"}"#,
    )
    .unwrap();
    let set_type = |model_type| {
        fs::write(
            dir.join("config.json"),
            json!({"model_type": model_type}).to_string(),
        )
        .unwrap();
    };
    // Only the recipe supports render_request for these families. High effort
    // also distinguishes its V4 instruction from V4.1's numeric budget.
    let mut body = request();
    body["chat_template_kwargs"] = json!({"thinking": true, "reasoning_effort": "high"});
    for (model_type, instruction) in [
        ("deepseek_v4", "Reasoning Effort: Absolute maximum"),
        ("deepseek_v41", "Reasoning Effort: 75"),
    ] {
        set_type(model_type);
        let (rendered, _) = selected(&options)
            .render_request(body.clone(), &Default::default())
            .unwrap();
        assert!(
            rendered.as_str().contains(instruction),
            "{model_type}: {rendered:?}"
        );
    }
    let kimi_turn = r#"<|open|>message role="user"<|sep|>Hi"#;
    for (model_type, turn) in [
        ("kimi_k3", kimi_turn),
        ("kimi_k31", kimi_turn),
        ("inkling_mm_model", "<|message_user|><|content_text|>Hi"),
    ] {
        set_type(model_type);
        assert!(prompt(&options).contains(turn), "{model_type}");
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn deepseek_recipe_fingerprint_tracks_process_defaults() {
    use sglang_processor::{RenderFingerprintConfig, render_fingerprint};

    const CHECKPOINT: &str = "SGLANG_TEST_RECIPE_CHECKPOINT";
    if let Some(dir) = std::env::var_os(CHECKPOINT).map(PathBuf::from) {
        let options = ChatFormatterOptions {
            tokenizer_path: dir.to_str().unwrap().into(),
            ..Default::default()
        };
        let kwargs = serde_json::from_slice(&fs::read(dir.join("kwargs.json")).unwrap()).unwrap();
        let config = RenderFingerprintConfig {
            default_chat_template_kwargs: &kwargs,
            context_length: 2048,
            reasoning_parser: None,
            incremental_streaming_output: false,
            multimodal_token_config: None,
        };
        // "none" in template kwargs selects the environment effort even when
        // a launch kwarg sets one, while leaving thinking at its default.
        let mut body = request();
        body["chat_template_kwargs"] = json!({"reasoning_effort": "none"});
        let (prompt, _) = selected(&options).render_request(body, &kwargs).unwrap();
        let digest = render_fingerprint(&options, &config).unwrap();
        fs::write(
            dir.join("result.json"),
            json!([prompt.into_text(), digest]).to_string(),
        )
        .unwrap();
        return;
    }

    let (dir, _) = checkpoint("recipe-defaults");
    tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
        .save(dir.join("tokenizer.json"), false)
        .unwrap();
    let run =
        |model_type: &str, thinking: &str, v4: &str, v41: &str, kwargs: Value| -> [String; 2] {
            fs::write(
                dir.join("config.json"),
                json!({"model_type": model_type}).to_string(),
            )
            .unwrap();
            fs::write(dir.join("kwargs.json"), kwargs.to_string()).unwrap();
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "deepseek_recipe_fingerprint_tracks_process_defaults",
                ])
                .env(CHECKPOINT, &dir)
                .env("SGLANG_DEFAULT_THINKING", thinking)
                .env("SGLANG_DSV4_REASONING_EFFORT", v4)
                .env("SGLANG_DSV41_REASONING_EFFORT", v41)
                .output()
                .unwrap();
            assert!(
                child.status.success(),
                "{}{}",
                String::from_utf8_lossy(&child.stdout),
                String::from_utf8_lossy(&child.stderr)
            );
            let path = dir.join("result.json");
            let result = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            fs::remove_file(path).unwrap();
            result
        };
    let v4 = |thinking, effort, other| run("deepseek_v4", thinking, effort, other, json!({}));
    let high = v4("true", "high", "");
    assert_eq!(high, v4("YES", " high ", "30"));
    for changed in [v4("false", "high", ""), v4("true", "low", "")] {
        assert_ne!(high[0], changed[0], "rendered prompt");
        assert_ne!(high[1], changed[1], "fingerprint");
    }

    let defaults = json!({"thinking": true, "reasoning_effort": "high"});
    let v41 =
        |thinking, other, effort| run("deepseek_v41", thinking, other, effort, defaults.clone());
    let budget = v41("false", "low", "30");
    assert_eq!(budget, v41("true", "high", "0.30"));
    let changed = v41("false", "low", "40");
    assert_ne!(budget[0], changed[0], "rendered prompt");
    assert_ne!(budget[1], changed[1], "fingerprint");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn name_only_fallback_respects_native_model_families() {
    let (dir, mut options) = checkpoint("name");
    // An uncached repo ID keeps the display name independent of the temp path.
    options.model_path = "sglang-selection-test/DeepSeek-V3.2".into();
    fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    assert!(prompt(&options).contains("<｜User｜>Hi<｜Assistant｜><think>"));
    let req: CreateChatCompletionRequest = serde_json::from_value(request()).unwrap();
    for (name, family) in [
        ("DeepSeek-V4-Flash", "deepseek_v4"),
        ("DeepSeek-V4.1", "deepseek_v41"),
    ] {
        options.model_path = name.into();
        let dynamo_renderer::PromptFormatter::OAI(native) =
            dynamo_renderer::native_formatter_for(&Some(family.into()), "").unwrap();
        assert_eq!(
            prompt(&options),
            native.render_prompt(&req).unwrap().into_text(),
            "{name}"
        );
    }
    for name in [
        "sglang-selection-test/DeepSeek-V3.2-Exp",
        "sglang-selection-test/Inkling",
    ] {
        options.model_path = name.into();
        assert_error(&options, "tokenizer has no chat template");
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn explicit_template_precedes_model_type_and_checkpoint_sources() {
    let (dir, mut options) = checkpoint("override");
    fs::write(dir.join("config.json"), r#"{"model_type":"deepseek_v4"}"#).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"chat_template":"embedded"}"#,
    )
    .unwrap();
    fs::write(dir.join("chat_template.jinja"), "discovered").unwrap();
    for (name, source, expected) in [
        (
            "override.jinja",
            "JINJA {{ messages[0].content }}",
            "JINJA Hi",
        ),
        (
            "override.json",
            r#"{"chat_template":"JSON {{ messages[0].content }}"}"#,
            "JSON Hi",
        ),
    ] {
        let path = dir.join(name);
        fs::write(&path, source).unwrap();
        options.chat_template = Some(path.to_str().unwrap().into());
        assert_eq!(prompt(&options), expected);
    }
    let invalid = dir.join("invalid.json");
    fs::write(&invalid, "[]").unwrap();
    options.chat_template = Some(invalid.to_str().unwrap().into());
    assert_error(
        &options,
        &format!(
            "legacy chat template `{}` must be a JSON object",
            invalid.display()
        ),
    );
    fs::remove_file(dir.join("tokenizer_config.json")).unwrap();
    assert_error(
        &options,
        "tokenizer_config.json is required for this chat template source but was not found",
    );
    fs::remove_dir_all(dir).unwrap();
}
