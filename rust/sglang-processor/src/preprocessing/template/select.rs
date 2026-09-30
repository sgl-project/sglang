//! Chat formatter selection from a model's template files and identity.

use std::sync::Arc;

use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;
use dynamo_renderer::deepseek::v32::DeepSeekV32Formatter;
use dynamo_renderer::{PromptFormatter, kimi_k3_formatter_for, native_formatter_for};

use super::{ChatFormatter, DeepSeekV4Profile, ThinkingTemplates, load_chat_formatter};
use crate::ChatConfig;
use crate::preprocessing::tokenizer::{resolve_chat_template_file, resolve_model_file};

pub(crate) fn load_chat_support(config: &ChatConfig) -> (Option<ChatFormatter>, Option<String>) {
    if config.tokenizer_path.is_empty() {
        return (None, None);
    }
    let tokenizer_config_file = resolve_model_file(
        &config.tokenizer_path,
        config.revision.as_deref(),
        "tokenizer_config.json",
    );
    let model_source = if config.model_path.is_empty() {
        config.tokenizer_path.as_str()
    } else {
        config.model_path.as_str()
    };
    let model_config_file =
        resolve_model_file(model_source, config.revision.as_deref(), "config.json");
    let identity = match model_config_file.as_deref().map(load_model_identity) {
        Some(Err(error)) => return (None, Some(error)),
        Some(Ok(identity)) => identity,
        None => ModelIdentity::default(),
    };
    let model_type_lower = identity.model_type.as_deref().map(str::to_ascii_lowercase);
    let display_name_lower = model_source.to_ascii_lowercase();
    if config.chat_template.is_none() {
        if identity.is_deepseek_v4() {
            let profile =
                match resolve_dsv4_profile(&identity, model_source, config.revision.as_deref()) {
                    Ok(profile) => profile,
                    Err(error) => return (None, Some(error)),
                };
            return (
                Some(ChatFormatter::DeepSeekV4 {
                    formatter: PromptFormatter::OAI(Arc::new(DeepSeekV4Formatter::new_chat())),
                    profile,
                    environment_effort: std::env::var("SGLANG_DSV4_REASONING_EFFORT").ok(),
                }),
                None,
            );
        }
        if identity.is_deepseek_v32() {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter: PromptFormatter::OAI(Arc::new(DeepSeekV32Formatter::new_chat())),
                    thinking: ThinkingTemplates::native(false, false),
                }),
                None,
            );
        }
        if let Some(formatter) = kimi_k3_formatter_for(&model_type_lower, &display_name_lower, true)
        {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    thinking: ThinkingTemplates::native(true, true),
                }),
                None,
            );
        }
        if model_type_lower.as_deref() == Some("inkling_mm_model")
            && let Some(formatter) = native_formatter_for(&model_type_lower, &display_name_lower)
        {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    thinking: ThinkingTemplates::always(),
                }),
                None,
            );
        }
        if let Some(formatter) = native_formatter_for(&model_type_lower, &display_name_lower) {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    // The remaining display-name fallback formatters are
                    // constructed with their thinking mode enabled.
                    thinking: ThinkingTemplates::native(true, false),
                }),
                None,
            );
        }
    }
    let discovered_template = config
        .chat_template
        .is_none()
        .then(|| resolve_chat_template_file(&config.tokenizer_path, config.revision.as_deref()))
        .flatten();
    let template_source = config
        .chat_template
        .as_deref()
        .or(discovered_template.as_deref());
    match load_chat_formatter(
        tokenizer_config_file.as_deref(),
        (!config.model_path.is_empty()).then_some(config.model_path.as_str()),
        template_source,
    ) {
        Ok(mut formatter) => {
            if identity.is_kimi_k25()
                && let ChatFormatter::HuggingFace {
                    formatter: inner,
                    thinking,
                } = formatter
            {
                formatter = ChatFormatter::KimiK25 {
                    formatter: inner,
                    thinking,
                };
            }
            tracing::info!(
                config = ?tokenizer_config_file.as_deref().unwrap_or("<built-in / inferred>"),
                template = ?template_source,
                "loaded OpenAI chat template"
            );
            (Some(formatter), None)
        }
        Err(error) => {
            tracing::warn!(%error, "OpenAI chat completions disabled");
            (
                None,
                Some(format!("this model has no usable chat template: {error}")),
            )
        }
    }
}

#[derive(Debug, Default)]
struct ModelIdentity {
    model_type: Option<String>,
    architectures: Vec<String>,
    dsv4_reasoning_effort_profile: Option<String>,
}

impl ModelIdentity {
    fn is_deepseek_v4(&self) -> bool {
        self.model_type.as_deref() == Some("deepseek_v4")
            || self
                .architectures
                .iter()
                .any(|architecture| architecture.starts_with("DeepseekV4"))
    }

    fn is_deepseek_v32(&self) -> bool {
        matches!(
            self.model_type.as_deref(),
            Some("deepseek_v32" | "deepseek_v3_2")
        ) || self
            .architectures
            .iter()
            .any(|architecture| architecture == "DeepseekV32ForCausalLM")
    }

    fn is_kimi_k25(&self) -> bool {
        self.model_type.as_deref() == Some("kimi_k25")
            || self
                .architectures
                .iter()
                .any(|architecture| architecture == "KimiK25ForConditionalGeneration")
    }
}

fn load_model_identity(config_file: &str) -> Result<ModelIdentity, String> {
    let Ok(config) = std::fs::read_to_string(config_file) else {
        return Ok(ModelIdentity::default());
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config) else {
        return Ok(ModelIdentity::default());
    };
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let architectures = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .map(|architectures| {
            architectures
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let dsv4_reasoning_effort_profile = match config.get("dsv4_reasoning_effort_profile") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(profile)) => Some(profile.clone()),
        Some(profile) => {
            return Err(format!(
                "invalid dsv4_reasoning_effort_profile: {profile}; expected \"preview\" or \"official\""
            ));
        }
    };
    Ok(ModelIdentity {
        model_type,
        architectures,
        dsv4_reasoning_effort_profile,
    })
}

fn resolve_dsv4_profile(
    identity: &ModelIdentity,
    model_source: &str,
    revision: Option<&str>,
) -> Result<DeepSeekV4Profile, String> {
    if let Some(profile) = identity.dsv4_reasoning_effort_profile.as_deref() {
        return match profile {
            "preview" => Ok(DeepSeekV4Profile::Preview),
            "official" => Ok(DeepSeekV4Profile::Official),
            _ => Err(format!(
                "invalid dsv4_reasoning_effort_profile: {profile:?}; expected \"preview\" or \"official\""
            )),
        };
    }
    let Some(encoder) = resolve_model_file(model_source, revision, "encoding/encoding_dsv4.py")
    else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    let Ok(metadata) = std::fs::metadata(&encoder) else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    if metadata.len() > 1 << 20 {
        return Ok(DeepSeekV4Profile::Preview);
    }
    let Ok(source) = std::fs::read_to_string(encoder) else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    let default = top_level_python_assignment(&source, "DEFAULT_REASONING_EFFORT")
        .and_then(python_string_literal);
    let prompt_keys = top_level_python_assignment(&source, "REASONING_EFFORT_PROMPTS")
        .and_then(python_dict_keys)
        .unwrap_or_default();
    if default.as_deref() == Some("low")
        && ["low", "high", "max"]
            .iter()
            .all(|key| prompt_keys.iter().any(|candidate| candidate == key))
    {
        Ok(DeepSeekV4Profile::Official)
    } else {
        Ok(DeepSeekV4Profile::Preview)
    }
}

fn top_level_python_assignment<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if !trimmed.starts_with(char::is_whitespace)
            && let Some((target, _)) = trimmed.split_once('=')
            && target
                .split(':')
                .next()
                .is_some_and(|target| target.trim() == name)
        {
            let equals = line.find('=')?;
            return Some(&source[offset + equals + 1..]);
        }
        offset += line.len();
    }
    None
}

fn python_string_literal(source: &str) -> Option<String> {
    let source = source.trim_start();
    let quote = source.chars().next()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let mut escaped = false;
    let mut value = String::new();
    for character in source[quote.len_utf8()..].chars() {
        if escaped {
            value.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == quote {
            return Some(value);
        } else {
            value.push(character);
        }
    }
    None
}

fn python_dict_keys(source: &str) -> Option<Vec<String>> {
    let source = source.trim_start();
    if !source.starts_with('{') {
        return None;
    }
    let mut keys = Vec::new();
    let mut depth = 0usize;
    let mut index = 0usize;
    let bytes = source.as_bytes();
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' | b'(' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' | b')' => {
                depth = depth.checked_sub(1)?;
                index += 1;
                if depth == 0 {
                    return Some(keys);
                }
            }
            quote @ (b'\'' | b'"') => {
                let start = index + 1;
                index = start;
                let mut escaped = false;
                while index < bytes.len() {
                    if escaped {
                        escaped = false;
                    } else if bytes[index] == b'\\' {
                        escaped = true;
                    } else if bytes[index] == quote {
                        break;
                    }
                    index += 1;
                }
                if index == bytes.len() {
                    return None;
                }
                let value = std::str::from_utf8(&bytes[start..index]).ok()?;
                index += 1;
                if depth == 1 {
                    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                        index += 1;
                    }
                    if bytes.get(index) == Some(&b':') {
                        keys.push(value.to_owned());
                    }
                }
            }
            b'#' => {
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            _ => index += 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatPreprocessor, ChatRequest};
    use dynamo_protocols::types::CreateChatCompletionRequest;

    fn model_config(model_path: String) -> ChatConfig {
        ChatConfig {
            tokenizer_path: model_path.clone(),
            model_path,
            ..Default::default()
        }
    }

    fn chat_request() -> ChatRequest {
        ChatRequest {
            model: "model".into(),
            messages: serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "hello"}
            ]))
            .unwrap(),
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
            continue_final_message: false,
            chat_template_args: None,
            parallel_tool_calls: true,
        }
    }

    #[test]
    fn dedicated_jinja_template_is_discovered_from_model_directory() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-dedicated-template-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("tokenizer_config.json"), "{}").unwrap();
        std::fs::write(
            directory.join("chat_template.jinja"),
            "{% for message in messages %}{{ message.content }}{% endfor %}",
        )
        .unwrap();

        let (formatter, error) =
            load_chat_support(&model_config(directory.to_string_lossy().into_owned()));

        assert!(formatter.is_some(), "{error:?}");
        assert!(error.is_none());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn kimi_k3_native_formatter_preserves_segments() {
        let directory =
            std::env::temp_dir().join(format!("sglang-renderer-kimi-k3-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("config.json"), r#"{"model_type":"kimi_k3"}"#).unwrap();
        let (formatter, error) =
            load_chat_support(&model_config(directory.to_string_lossy().into_owned()));
        let formatter = formatter.unwrap_or_else(|| panic!("{error:?}"));
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let prompt = formatter.render_prompt(&request).unwrap();

        assert!(prompt.segments().is_some());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn native_formatters_forward_their_effective_thinking_mode() {
        let root = std::env::temp_dir().join(format!(
            "sglang-renderer-native-thinking-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let kimi = root.join("kimi");
        std::fs::create_dir_all(&kimi).unwrap();
        std::fs::write(kimi.join("config.json"), r#"{"model_type":"kimi_k3"}"#).unwrap();
        let mut config = model_config(kimi.to_string_lossy().into_owned());
        config.reasoning_parser = Some("kimi_k3".into());
        config.tool_call_parser = Some("kimi_k3".into());
        let service = ChatPreprocessor::load(&config);
        let enabled = service.preprocess(chat_request()).unwrap();
        assert!(enabled.require_reasoning);

        let mut disabled_request = chat_request();
        disabled_request.chat_template_args = Some(std::collections::HashMap::from([(
            "thinking".into(),
            serde_json::Value::Bool(false),
        )]));
        let disabled = service.preprocess(disabled_request).unwrap();
        assert!(!disabled.require_reasoning);

        let mut named_request = chat_request();
        named_request.tools = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]))
        .unwrap();
        named_request.tool_choice = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        }))
        .unwrap();
        let named = service.preprocess(named_request).unwrap();
        assert!(!named.require_reasoning);

        let deepseek = root.join("deepseek");
        std::fs::create_dir_all(&deepseek).unwrap();
        std::fs::write(
            deepseek.join("config.json"),
            r#"{"model_type":"deepseek_v32"}"#,
        )
        .unwrap();
        let mut config = model_config(deepseek.to_string_lossy().into_owned());
        config.reasoning_parser = Some("deepseek-v3".into());
        let service = ChatPreprocessor::load(&config);
        let default = service.preprocess(chat_request()).unwrap();
        assert!(!default.require_reasoning);

        let mut explicit = chat_request();
        explicit.chat_template_args = Some(std::collections::HashMap::from([(
            "enable_thinking".into(),
            serde_json::Value::Bool(true),
        )]));
        let explicit = service.preprocess(explicit).unwrap();
        assert!(explicit.require_reasoning);

        let mut effort = chat_request();
        effort.reasoning_effort = Some(serde_json::from_value(serde_json::json!("high")).unwrap());
        let effort = service.preprocess(effort).unwrap();
        assert!(effort.require_reasoning);

        let inkling = root.join("inkling");
        std::fs::create_dir_all(&inkling).unwrap();
        std::fs::write(
            inkling.join("config.json"),
            r#"{"model_type":"inkling_mm_model"}"#,
        )
        .unwrap();
        let mut config = model_config(inkling.to_string_lossy().into_owned());
        config.reasoning_parser = Some("inkling".into());
        let service = ChatPreprocessor::load(&config);
        let inkling = service.preprocess(chat_request()).unwrap();
        assert!(inkling.require_reasoning);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicit_chat_template_overrides_native_model_detection() {
        for model_type in ["kimi_k3", "deepseek_v4", "deepseek_v32", "inkling_mm_model"] {
            let directory = std::env::temp_dir().join(format!(
                "sglang-renderer-template-override-{model_type}-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(
                directory.join("config.json"),
                serde_json::json!({"model_type": model_type}).to_string(),
            )
            .unwrap();
            std::fs::write(directory.join("tokenizer_config.json"), "{}").unwrap();
            let mut config = model_config(directory.to_string_lossy().into_owned());
            let template = directory.join("override.jinja");
            std::fs::write(&template, "OVERRIDE {{ messages[0].content }}").unwrap();
            config.chat_template = Some(template.to_string_lossy().into_owned());
            let (formatter, error) = load_chat_support(&config);
            let formatter = formatter.unwrap_or_else(|| panic!("{model_type}: {error:?}"));
            let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap();

            let prompt = formatter.render_prompt(&request).unwrap();

            assert_eq!(prompt.as_str(), "OVERRIDE hello", "{model_type}");
            std::fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn top_level_reasoning_effort_reaches_deepseek_v4_formatter() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-deepseek-v4-effort-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let mut config = model_config(directory.to_string_lossy().into_owned());
        config.reasoning_parser = Some("deepseek-v4".into());
        let messages = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "hello"}
        ]))
        .unwrap();
        let request = ChatRequest {
            model: "model".into(),
            messages,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
            continue_final_message: false,
            chat_template_args: None,
            parallel_tool_calls: true,
        };
        for (profile, max_prefix, high_prefix) in [
            ("preview", "Absolute maximum", None),
            ("official", "Beyond maximum", Some("Absolute maximum")),
        ] {
            std::fs::write(
                directory.join("config.json"),
                serde_json::json!({
                    "model_type": "deepseek_v4",
                    "dsv4_reasoning_effort_profile": profile,
                })
                .to_string(),
            )
            .unwrap();
            let service = ChatPreprocessor::load(&config);
            for (effort, args, thinking, prefix) in [
                (None, serde_json::json!({}), false, None),
                (Some("max"), serde_json::json!({}), true, Some(max_prefix)),
                (Some("high"), serde_json::json!({}), true, high_prefix),
                (Some("none"), serde_json::json!({}), false, None),
                (
                    Some("max"),
                    serde_json::json!({"thinking": false}),
                    false,
                    None,
                ),
                (
                    Some("none"),
                    serde_json::json!({"thinking": true}),
                    true,
                    None,
                ),
                (
                    Some("max"),
                    serde_json::json!({"reasoning_effort": "low"}),
                    true,
                    None,
                ),
            ] {
                let mut request = request.clone();
                request.reasoning_effort =
                    effort.map(|effort| serde_json::from_value(serde_json::json!(effort)).unwrap());
                request.chat_template_args = Some(serde_json::from_value(args).unwrap());
                let chat = service.preprocess(request).unwrap();
                let text_request = &chat;
                assert_eq!(text_request.require_reasoning, thinking);
                let prompt = text_request.prompt.as_str();
                assert!(prompt.ends_with(if thinking { "<think>" } else { "</think>" }));
                assert_eq!(
                    prompt.matches("Reasoning Effort:").count(),
                    usize::from(prefix.is_some()),
                    "{profile}, {effort:?}: {prompt}"
                );
                if let Some(prefix) = prefix {
                    assert!(prompt.starts_with(&format!(
                        "<｜begin▁of▁sentence｜>Reasoning Effort: {prefix}"
                    )));
                }
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn deepseek_v32_metadata_overrides_bundled_jinja_for_exp_checkpoints() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-deepseek-v32-exp-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            r#"{"model_type":"deepseek_v32","architectures":["DeepseekV32ForCausalLM"]}"#,
        )
        .unwrap();
        std::fs::write(
            directory.join("tokenizer_config.json"),
            r#"{"chat_template":"BUNDLED TEMPLATE WITHOUT TOOLS"}"#,
        )
        .unwrap();
        let (formatter, error) =
            load_chat_support(&model_config(directory.to_string_lossy().into_owned()));
        let formatter = formatter.unwrap_or_else(|| panic!("{error:?}"));
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        }))
        .unwrap();

        let prompt = formatter.render_prompt(&request).unwrap();

        assert!(prompt.as_str().contains("get_weather"));
        assert!(prompt.as_str().contains("｜DSML｜"));
        assert!(!prompt.as_str().contains("BUNDLED TEMPLATE"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn kimi_k25_preprocesses_tools_for_checkpoint_jinja() {
        let directory =
            std::env::temp_dir().join(format!("sglang-renderer-kimi-k25-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            r#"{"model_type":"kimi_k25","architectures":["KimiK25ForConditionalGeneration"]}"#,
        )
        .unwrap();
        std::fs::write(
            directory.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{% if tools_ts_str is defined %}{{ tools_ts_str }}{% else %}JSON {{ tools|tojson }}{% endif %}"
            })
            .to_string(),
        )
        .unwrap();
        let (formatter, error) =
            load_chat_support(&model_config(directory.to_string_lossy().into_owned()));
        let formatter = formatter.unwrap_or_else(|| panic!("{error:?}"));
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }]
        }))
        .unwrap();

        let prompt = formatter.render_prompt(&request).unwrap();

        assert!(prompt.as_str().contains("namespace functions"));
        assert!(prompt.as_str().contains("type get_weather"));
        assert!(!prompt.as_str().starts_with("JSON "));

        let unsupported: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "unsupported",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"oneOf": [{"type": "string"}]}
                        }
                    }
                }
            }]
        }))
        .unwrap();
        let fallback = formatter.render_prompt(&unsupported).unwrap();
        assert!(fallback.as_str().starts_with("JSON "));
        assert!(fallback.as_str().contains("unsupported"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn deepseek_v4_profile_resolution_uses_override_then_checkpoint_source() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-deepseek-v4-profile-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(directory.join("encoding")).unwrap();
        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            "DEFAULT_REASONING_EFFORT: str = 'low'\n\
REASONING_EFFORT_PROMPTS = {'low': '', 'high': 'absolute', 'max': 'beyond'}",
        )
        .unwrap();
        let source = directory.to_string_lossy();
        assert_eq!(
            resolve_dsv4_profile(&ModelIdentity::default(), &source, None).unwrap(),
            DeepSeekV4Profile::Official
        );
        let preview = ModelIdentity {
            dsv4_reasoning_effort_profile: Some("preview".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_dsv4_profile(&preview, &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        let invalid = ModelIdentity {
            dsv4_reasoning_effort_profile: Some("future".into()),
            ..Default::default()
        };
        assert!(resolve_dsv4_profile(&invalid, &source, None).is_err());

        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            r#"DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_PROMPTS = {"low": "", "high": "absolute", "max": "Beyond maximum"}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_dsv4_profile(&ModelIdentity::default(), &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn chat_template_argument_precedence_is_request_then_top_level_then_defaults() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-template-defaults-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("tokenizer_config.json"), "{}").unwrap();
        let mut config = model_config(directory.to_string_lossy().into_owned());
        let template = directory.join("arguments.jinja");
        std::fs::write(&template, "{{ marker }}|{{ reasoning_effort }}").unwrap();
        config.chat_template = Some(template.to_string_lossy().into_owned());
        config.default_chat_template_kwargs = std::collections::HashMap::from([
            ("marker".into(), serde_json::json!("default")),
            ("reasoning_effort".into(), serde_json::json!("low")),
        ]);
        let messages = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "hello"}
        ]))
        .unwrap();
        let mut request = ChatRequest {
            model: "model".into(),
            messages,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: Some(serde_json::from_value(serde_json::json!("max")).unwrap()),
            continue_final_message: false,
            chat_template_args: Some(std::collections::HashMap::from([(
                "marker".into(),
                serde_json::json!("request"),
            )])),
            parallel_tool_calls: true,
        };
        let service = ChatPreprocessor::load(&config);

        let chat = service.preprocess(request.clone()).unwrap();
        assert_eq!(chat.prompt.as_str(), "request|max");

        request
            .chat_template_args
            .as_mut()
            .unwrap()
            .insert("reasoning_effort".into(), serde_json::json!("medium"));
        let chat = service.preprocess(request).unwrap();
        assert_eq!(chat.prompt.as_str(), "request|medium");
        std::fs::remove_dir_all(directory).unwrap();
    }
}
