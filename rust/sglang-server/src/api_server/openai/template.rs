//! Resolve chat-template names and files to chat prompt formatters.
//
//! Hugging Face tokenizer configs contain Jinja templates. SGLang also accepts
//! legacy conversation JSON files and the names in Python's template registry.
//! Legacy definitions are rendered by a Rust implementation of Python's
//! `Conversation.get_prompt()` so there is exactly one implementation of the
//! per-style formatting logic (no Jinja translation to drift).

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter, TextInput};

use crate::message::types::OneOrMany;

pub(super) use super::template_loader::load_chat_formatter;
// Legacy rendering and its error type are canonical in the renderer; the
// server keeps only its enum shell and kwargs adapter.
use sglang_renderer::LegacyFormatter;

/// Extra variables for the chat template (`chat_template_kwargs`).
pub type ChatTemplateKwargs = HashMap<String, serde_json::Value>;

/// A chat prompt formatter: either the model's HuggingFace Jinja template (or
/// Dynamo's built-in encoder for models that ship none) or a legacy SGLang
/// conversation template.
#[derive(Clone)]
pub enum ChatFormatter {
    HuggingFace(PromptFormatter),
    Legacy(Box<LegacyFormatter>),
}

impl ChatFormatter {
    /// Render the request's messages to a single prompt string.
    pub(super) fn render(
        &self,
        request: &CreateChatCompletionRequest,
        kwargs: Option<&ChatTemplateKwargs>,
    ) -> Result<String, TemplateError> {
        match self {
            ChatFormatter::HuggingFace(PromptFormatter::OAI(formatter)) => formatter
                .render(&TemplateRequest { request, kwargs })
                .map_err(|error| TemplateError::Renderer {
                    message: error.to_string(),
                }),
            ChatFormatter::Legacy(formatter) => formatter.render(request),
        }
    }

    /// The template's stop strings — Python `Conversation.stop_str`
    /// (`str | list[str] | None`). Legacy/builtin templates define them (e.g.
    /// chatml's `<|im_end|>`); the HuggingFace renderer carries none, matching
    /// Python's jinja path, which keeps only the request's own stops.
    pub(super) fn stop_strs(&self) -> Option<OneOrMany<String>> {
        match self {
            ChatFormatter::HuggingFace(_) => None,
            // The renderer owns its own `OneOrMany`; convert variant-for-variant.
            ChatFormatter::Legacy(formatter) => {
                formatter.spec.stop_str.clone().map(|stops| match stops {
                    sglang_renderer::OneOrMany::One(one) => OneOrMany::One(one),
                    sglang_renderer::OneOrMany::Many(many) => OneOrMany::Many(many),
                })
            }
        }
    }
}

/// The wire request plus its `chat_template_kwargs`, which the protocol type
/// does not carry.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a ChatTemplateKwargs>,
}

impl OAIChatLikeRequest for TemplateRequest<'_> {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.request.messages()
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.request.typed_messages()
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.request.reasoning_effort()
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.request.should_add_generation_prompt()
    }
    fn chat_template_args(&self) -> Option<&ChatTemplateKwargs> {
        self.kwargs
    }
    fn extract_text(&self) -> Option<TextInput> {
        self.request.extract_text()
    }
    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.request.mm_processor_kwargs()
    }
}

/// Single error type for chat templating, owned by the renderer — the server
/// uses it directly so there is exactly one definition to maintain.
pub(super) use sglang_renderer::TemplateError;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_protocols::types::CreateChatCompletionRequest;

    use super::{ChatFormatter, TemplateError, load_chat_formatter};
    use sglang_renderer::{
        LegacyFormatter, builtin_template, infer_legacy_template_from_model_path,
    };

    fn request() -> CreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"}
            ]
        }))
        .unwrap()
    }

    /// Facade wiring: the server enum delegates chatml rendering to the
    /// renderer formatter (style-matrix coverage lives in renderer tests).
    #[test]
    fn built_in_chatml_does_not_require_a_file() {
        let formatter = ChatFormatter::Legacy(Box::new(LegacyFormatter {
            spec: builtin_template("chatml").unwrap(),
        }));
        let rendered = formatter.render(&request(), None).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// stop_str preserves Python's `str | list[str] | None` typing, end to end
    /// through the facade into the renderer table.
    #[test]
    fn stop_str_keeps_python_type_semantics() {
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        // QWEN2_VL_EMBED requires a single string stop (Python raises TypeError
        // on a list / None) — error deliberately.
        use sglang_renderer::OneOrMany as RendererOneOrMany;
        let mut spec = builtin_template("gme-qwen2-vl").unwrap();
        spec.style = "QWEN2_VL_EMBED".into();
        spec.stop_str = Some(RendererOneOrMany::Many(vec!["a".into(), "b".into()]));
        let error = LegacyFormatter { spec }.render(&request).unwrap_err();
        assert!(error.to_string().contains("stop_str"));
        // The built-in gme-qwen2-vl registers a single-string stop.
        let spec = builtin_template("gme-qwen2-vl").unwrap();
        assert!(matches!(spec.stop_str, Some(RendererOneOrMany::One(_))));
        // gemma-it registers a list.
        let spec = builtin_template("gemma-it").unwrap();
        assert!(matches!(spec.stop_str, Some(RendererOneOrMany::Many(_))));
        // An explicit `"stop_str": null` in a legacy JSON file maps to `None`
        // (Python accepts a present null), while a missing key stays an error.
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-stopnull-{}-test.json",
            std::process::id()
        ));
        std::fs::write(
            &base,
            r#"{
                "name": "test",
                "system": "System",
                "user": "USER",
                "assistant": "ASSISTANT",
                "sep_style": "ADD_COLON_SINGLE",
                "stop_str": null
            }"#,
        )
        .unwrap();
        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            None,
            None,
            Some(base.to_str().unwrap()),
        )
        .unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert!(formatter.spec.stop_str.is_none());
        let _ = std::fs::remove_file(base);
    }

    #[test]
    fn json_legacy_template_is_rendered() {
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-base-{}-test.json",
            std::process::id()
        ));
        let legacy = base.with_file_name("sglang-openai-template-legacy.json");
        std::fs::write(&base, r#"{"chat_template":"unused"}"#).unwrap();
        std::fs::write(
            &legacy,
            r#"{
                "name": "test-legacy",
                "system": "System",
                "system_message": "default",
                "user": "USER",
                "assistant": "ASSISTANT",
                "sep_style": "ADD_COLON_SINGLE",
                "sep": "\n",
                "stop_str": "<stop>"
            }"#,
        )
        .unwrap();

        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            None,
            None,
            Some(legacy.to_str().unwrap()),
        )
        .unwrap();
        let rendered = formatter.render(&request(), None).unwrap();
        assert_eq!(rendered, "System\nBe concise.\nUSER: Hello\nASSISTANT:");

        let _ = std::fs::remove_file(base);
        let _ = std::fs::remove_file(legacy);
    }

    /// A built-in `--chat-template` name resolves without any tokenizer config.
    #[test]
    fn builtin_argument_works_without_tokenizer_config() {
        let formatter = load_chat_formatter(None, None, None, Some("chatml")).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "chatml");
    }

    /// Python `load_chat_template`: without `--chat-template`, the model path
    /// infers a legacy template before the HF fallback — so a legacy model
    /// with no `chat_template` in its config still gets one, and even a config
    /// that HAS one loses to the inference.
    #[test]
    fn model_path_inference_precedes_tokenizer_config() {
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-infer-{}-test.json",
            std::process::id()
        ));
        std::fs::write(
            &base,
            r#"{"tokenizer_class":"LlamaTokenizer","chat_template":"{{messages}}"}"#,
        )
        .unwrap();

        // Path matcher: vicuna/llava-v1.5-style paths.
        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            Some("models/vicuna-7b-v1.5"),
            None,
            None,
        )
        .unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "vicuna_v1.1");
        // No config at all + path matcher.
        let formatter = load_chat_formatter(None, Some("deepseek-vl2-7b"), None, None).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "deepseek-vl2");

        // Model-type matcher: reads `<model_path>/config.json`.
        let model_dir = std::env::temp_dir().join(format!(
            "sglang-openai-template-infer-model-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"phi4mm","architectures":["Phi4MMForCausalLM"]}"#,
        )
        .unwrap();
        let formatter =
            load_chat_formatter(None, Some(model_dir.to_str().unwrap()), None, None).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "phi-4-mm");

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_dir_all(model_dir);
    }

    /// Every name the model-path matchers can produce must exist in the
    /// built-in table (parity guard for `MODEL_TYPE_TO_TEMPLATE`).
    #[test]
    fn inferred_template_names_resolve_to_builtins() {
        for model_path in [
            "points-7b-chat",
            "moss-vl",
            "moss2-vl",
            "internvl-2.5",
            "janus-pro",
            "vicuna-7b",
            "llava-v1.5-7b",
            "deepseek-vl2-small",
            "llava-v1.6-34b",
            "minicpm-v-2.6",
            "minicpm-o-4.5",
            "phi-4-multimodal",
            "deepseek-ocr",
            "unlimited-ocr",
            "paddleocr-vl",
            "whisper",
        ] {
            let spec = infer_legacy_template_from_model_path(model_path)
                .unwrap_or_else(|| panic!("no inference for {model_path}"));
            let _ = spec;
        }
    }

    /// MiniCPM 4.6+ must NOT fall back to the legacy template; with no config
    /// and nothing else to try, that surfaces as the missing-config error.
    #[test]
    fn minicpm_4_6_skips_legacy_inference() {
        assert!(infer_legacy_template_from_model_path("minicpm-v-4.6").is_none());
        assert!(matches!(
            load_chat_formatter(None, Some("minicpm-v-4.6"), None, None),
            Err(TemplateError::MissingConfig)
        ));
    }

    fn temp_config(contents: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sglang-template-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("tokenizer_config.json");
        std::fs::write(&config, contents).unwrap();
        config
    }

    #[test]
    fn chat_template_kwargs_reach_the_template() {
        let config = temp_config(r#"{"chat_template": "thinking={{ thinking }}"}"#);
        let formatter =
            load_chat_formatter(Some(config.to_str().unwrap()), None, None, None).unwrap();
        let kwargs = HashMap::from([("thinking".into(), serde_json::json!(true))]);
        assert_eq!(formatter.render(&request(), None).unwrap(), "thinking=");
        assert_eq!(
            formatter.render(&request(), Some(&kwargs)).unwrap(),
            "thinking=True"
        );
    }

    #[test]
    fn missing_template_falls_back_to_native_formatter() {
        let config = temp_config("{}");
        let config = config.to_str().unwrap();
        let load = |config, model_type, arg| {
            load_chat_formatter(config, Some("/models/x"), model_type, arg)
        };

        let formatter = load(Some(config), Some("deepseek_v4"), None).unwrap();
        assert!(matches!(formatter, ChatFormatter::HuggingFace(_)));
        let kwargs = HashMap::from([("thinking".into(), serde_json::json!(false))]);
        assert_eq!(
            formatter.render(&request(), Some(&kwargs)).unwrap(),
            "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Hello<｜Assistant｜></think>"
        );
        assert!(matches!(
            load(None, Some("deepseek_v4"), None),
            Ok(ChatFormatter::HuggingFace(_))
        ));

        // A template, `--chat-template`, or an unknown architecture wins.
        let templated = temp_config(r#"{"chat_template": "Hi"}"#);
        let formatter = load(Some(templated.to_str().unwrap()), Some("deepseek_v4"), None).unwrap();
        assert_eq!(formatter.render(&request(), None).unwrap(), "Hi");
        assert!(matches!(
            load(Some(config), Some("deepseek_v4"), Some("chatml")),
            Ok(ChatFormatter::Legacy(_))
        ));
        assert!(matches!(
            load(Some(config), Some("llama"), None),
            Err(TemplateError::Missing)
        ));
        assert!(matches!(
            load(None, None, None),
            Err(TemplateError::MissingConfig)
        ));
    }
}
