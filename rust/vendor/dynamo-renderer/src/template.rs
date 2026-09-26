// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use anyhow::{Ok, Result};
use minijinja::Environment;

use super::PromptContextMixin;

mod context;
mod formatters;
mod oai;
mod tokcfg;

use super::{OAIPromptFormatter, PromptFormatter};
pub use oai::may_be_fix_tool_schema;
pub use tokcfg::{ChatTemplate, ChatTemplateValue};

/// If the model is a DeepSeek family whose HF repo doesn't ship a Jinja
/// `chat_template`, return the native Rust formatter for it. Returns `None`
/// for everything else (the caller then loads the HF `tokenizer_config.json`
/// template via [`PromptFormatter::from_parts`]).
///
/// `model_type_lower` is the lowercased `config.json` `model_type` (authoritative,
/// survives `--served-model-name` renames); `display_name_lower` is the
/// lowercased served name, used only as a fallback when `model_type` is absent.
pub fn deepseek_formatter_for(
    model_type_lower: &Option<String>,
    display_name_lower: &str,
) -> Option<PromptFormatter> {
    if is_deepseek_v4(model_type_lower, display_name_lower) {
        tracing::info!(
            model_type = ?model_type_lower,
            display_name = %display_name_lower,
            "Detected DeepSeek V4 model, using native Rust formatter",
        );
        return Some(PromptFormatter::OAI(Arc::new(
            super::deepseek::v4::DeepSeekV4Formatter::new_thinking(),
        )));
    }
    if is_deepseek_v3_2_non_exp(model_type_lower, display_name_lower) {
        tracing::info!("Detected DeepSeek V3.2 model (non-Exp), using native Rust formatter");
        return Some(PromptFormatter::OAI(Arc::new(
            super::deepseek::v32::DeepSeekV32Formatter::new_thinking(),
        )));
    }
    None
}

/// If the model is Kimi K3, return its native XTML formatter. K3 ships no
/// Jinja chat template and must preserve special-vs-ordinary segment boundaries
/// until tokenization.
pub fn kimi_k3_formatter_for(
    model_type_lower: &Option<String>,
    display_name_lower: &str,
    exclude_tools_when_tool_choice_none: bool,
) -> Option<PromptFormatter> {
    if !is_kimi_k3(model_type_lower, display_name_lower) {
        return None;
    }

    tracing::info!(
        model_type = ?model_type_lower,
        display_name = %display_name_lower,
        "Detected Kimi K3 model, using native Rust XTML formatter",
    );
    Some(PromptFormatter::OAI(Arc::new(
        super::kimi_k3::KimiK3Formatter::new(exclude_tools_when_tool_choice_none),
    )))
}

fn is_kimi_k3(model_type_lower: &Option<String>, display_name_lower: &str) -> bool {
    match model_type_lower.as_deref() {
        Some("kimi_k3") => true,
        Some(_) => false,
        None => ["kimi-k3", "kimi_k3", "kimik3"]
            .iter()
            .any(|needle| display_name_lower.contains(needle)),
    }
}

/// Select a native formatter for model families that do not ship a usable HF
/// `chat_template`.
///
/// Inkling is selected only from the authoritative `config.json` model type;
/// unlike display-name substring matching, this remains stable under
/// `--served-model-name` aliases. DeepSeek keeps its existing fallback for
/// older model cards that do not publish `model_type`.
pub fn native_formatter_for(
    model_type_lower: &Option<String>,
    display_name_lower: &str,
) -> Option<PromptFormatter> {
    if model_type_lower.as_deref() == Some("inkling_mm_model") {
        tracing::info!(
            model_type = ?model_type_lower,
            "Detected Inkling model, using native Rust formatter",
        );
        return Some(PromptFormatter::OAI(Arc::new(
            super::inkling::InklingFormatter,
        )));
    }

    deepseek_formatter_for(model_type_lower, display_name_lower)
}

impl PromptFormatter {
    pub fn from_parts(
        config: ChatTemplate,
        context: ContextMixins,
        exclude_tools_when_tool_choice_none: bool,
    ) -> Result<PromptFormatter> {
        let formatter = HfTokenizerConfigJsonFormatter::with_options(
            config,
            context,
            exclude_tools_when_tool_choice_none,
        )?;
        Ok(Self::OAI(Arc::new(formatter)))
    }
}

/// Chat Template Jinja Renderer
///
/// Manages a Jinja environment with registered templates for chat formatting.
/// Handles two types of ChatTemplateValue templates:
///
/// 1. String template: Registered as the 'default' template
/// 2. Map template: Contains 'tool_use' and/or 'default' templates
///    - tool_use: Template for tool-based interactions
///    - default: Template for standard chat interactions
///
///   If the map contains both keys, the `tool_use` template is registered as the `tool_use` template
///   and the `default` template is registered as the `default` template.
struct JinjaEnvironment {
    env: Environment<'static>,
}

/// Which message-shape restrictions one chat template enforces, probed once at
/// load. Each rewrite in `normalize_system_messages` is gated on its own flag:
/// the restrictions are independent, so a template that rejects a non-leading
/// `system` (Qwen3.5) but accepts consecutive `user` turns keeps those turns
/// separate instead of being reshaped for a rule it does not have.
#[derive(Debug, Default, Clone, Copy)]
struct SystemNormalization {
    /// Template raises on a `system` turn that is not first.
    demote_nonleading_system: bool,
    /// Template raises on two adjacent `user` turns.
    coalesce_consecutive_users: bool,
}

impl SystemNormalization {
    fn is_required(&self) -> bool {
        self.demote_nonleading_system || self.coalesce_consecutive_users
    }
}

/// Formatter for HuggingFace tokenizer config JSON templates
///
/// Implements chat template rendering based on HuggingFace's tokenizer_config.json format.
/// Supports:
/// - Tool usage templates
/// - Generation prompts
/// - Context mixins for template customization
#[derive(Debug)]
struct HfTokenizerConfigJsonFormatter {
    env: Environment<'static>,
    config: ChatTemplate,
    mixins: Arc<ContextMixins>,
    supports_add_generation_prompt: bool,
    requires_content_arrays: bool,
    /// When true, strip tool definitions from the chat template when tool_choice is "none".
    /// This prevents models from generating raw XML tool calls in the content field.
    exclude_tools_when_tool_choice_none: bool,
    /// True if the `default` template natively references `reasoning_content`.
    /// When true and rendering through `default`, skip injection — the template
    /// handles it. Tracked separately for `default` and `tool_use` because HF
    /// configs may register different sources for each: Gemma4's `tool_use`
    /// template is adapted by `normalize_chat_template_source` to read
    /// `reasoning_content`, while its `default` template is not. A single global
    /// flag would wrongly suppress injection on the untouched `default` path and
    /// silently drop prior assistant reasoning on no-tool renders.
    default_template_handles_reasoning: bool,
    /// True if the `tool_use` template natively references `reasoning_content`.
    /// See `default_template_handles_reasoning` for rationale.
    tool_use_template_handles_reasoning: bool,
    /// Per-family placeholder template for image content parts when flattening
    /// mixed text+image content arrays into a single string (`preserve_arrays`
    /// = false path). `{n}` in the template is substituted with the 1-based
    /// image index. `None` when the model's chat template handles content
    /// arrays natively (Qwen-VL family) or when we have no flatten strategy
    /// for it (no MM-aware routing benefit either way).
    image_placeholder_template: Option<&'static str>,
    /// True if the `default` template branches on `tool_call.arguments is string`
    /// (Qwen3, Hermes, etc.). When true and rendering through `default`, skip
    /// pre-parsing the JSON-string `tool_calls[].function.arguments` into an
    /// object — the template wants the raw string verbatim. Pre-parsing forces
    /// the `tojson`-with-object branch and re-emits with minijinja's compact
    /// separators, which breaks append-only prefix matching across multi-step
    /// tool-use turns. Tracked separately for `default` and `tool_use` because
    /// HF configs may register different sources for each, and because
    /// `arguments is string` is tool_calls-specific — legacy
    /// `function_call.arguments` lives outside that branch and is still
    /// normalized unconditionally.
    default_template_handles_tool_calls_arguments_string: bool,
    /// True if the `tool_use` template branches on `tool_call.arguments is string`.
    /// See `default_template_handles_tool_calls_arguments_string` for rationale.
    tool_use_template_handles_tool_calls_arguments_string: bool,
    /// Message-shape restrictions the `default` template enforces.
    default_system_normalization: SystemNormalization,
    /// Message-shape restrictions the `tool_use` template enforces.
    /// Kept separate because dict-form HF configs may register templates with
    /// different constraints.
    tool_use_system_normalization: SystemNormalization,
}

// /// OpenAI Standard Prompt Formatter
// pub trait StandardPromptFormatter {
//     fn render(&self, context: &impl StandardPromptContext) -> Result<String>;
// }

// pub trait StandardPromptContext {
//     fn messages(&self) -> Value;
//     fn tools(&self) -> Option<Value>;
// }

#[derive(Debug, Clone, Default)]
pub struct ContextMixins {
    context_mixins: HashSet<PromptContextMixin>,
}

/// Decides whether to activate the DeepSeek-V4 native formatter.
///
/// Primary signal: config.json `model_type`. DeepSeek-V4-Pro and V4-Flash both
/// ship `"model_type": "deepseek_v4"`, set by the model author — this survives
/// any `--served-model-name` rename.
///
/// Fallback: `display_name`, tight-matched against
/// `^deepseek(?:[-_.])?v4(?:[-_.]|$)`. Only consulted when config.json is
/// absent (tokenizer-only MDCs) or unreadable; a concrete config.json value
/// that is *not* `deepseek_v4` is authoritative and suppresses the fallback.
fn is_deepseek_v4(model_type_lower: &Option<String>, display_name_lower: &str) -> bool {
    match model_type_lower.as_deref() {
        Some("deepseek_v4") => true,
        Some(_) => false, // config.json says something else — trust it
        None => is_deepseek_v4_name(display_name_lower),
    }
}

/// Decides whether to activate the DeepSeek-V3.2 (non-Exp) native formatter.
/// Same config-primary / name-fallback rule as V4.
fn is_deepseek_v3_2_non_exp(model_type_lower: &Option<String>, display_name_lower: &str) -> bool {
    let name_match = display_name_lower.contains("deepseek")
        && display_name_lower.contains("v3.2")
        && !display_name_lower.contains("exp");
    match model_type_lower.as_deref() {
        // HF ships `deepseek_v32` (no underscore between 3 and 2); Dynamo's
        // internal/tool-parser key is `deepseek_v3_2`. Accept both.
        Some("deepseek_v3_2" | "deepseek_v32") => !display_name_lower.contains("exp"),
        Some(_) => false,
        None => name_match,
    }
}

/// Tight, anchored match for DeepSeek-V4 display names. Equivalent to the
/// regex `^deepseek(?:[-_.])?v4(?:[-_.]|$)` over an already-lowercased string.
/// Written with string ops to avoid pulling in the `regex` crate.
///
/// Rejects composite names that previously short-circuited the V4 branch:
/// - `deepseek-v3.2-v4-foo` (the `v3.2` variant is the real one)
/// - `deepseek-v40` / `deepseek-v4pro` (no separator after `v4`)
/// - `my-deepseek-v4` (prefix must be at the start)
fn is_deepseek_v4_name(name_lower: &str) -> bool {
    let Some(rest) = name_lower.strip_prefix("deepseek") else {
        return false;
    };
    // Optional single separator between "deepseek" and "v4".
    let rest = rest
        .strip_prefix(|c: char| matches!(c, '-' | '_' | '.'))
        .unwrap_or(rest);
    let Some(after_v4) = rest.strip_prefix("v4") else {
        return false;
    };
    // `v4` must end the name or be followed by a separator — anything else
    // (e.g. `v40`, `v4pro`) is a different model family.
    after_v4.is_empty() || after_v4.starts_with(['-', '_', '.'])
}

#[cfg(test)]
mod detection_tests {
    use super::{is_deepseek_v3_2_non_exp, is_deepseek_v4, is_deepseek_v4_name, is_kimi_k3};

    #[test]
    fn kimi_k3_detection_prefers_config_model_type() {
        assert!(is_kimi_k3(&Some("kimi_k3".to_string()), "served-name"));
        assert!(!is_kimi_k3(
            &Some("kimi_k2".to_string()),
            "moonshot-kimi-k3"
        ));
        assert!(is_kimi_k3(&None, "moonshot-kimi-k3"));
        assert!(is_kimi_k3(&None, "kimi_k3-instruct"));
        assert!(!is_kimi_k3(&None, "kimi-k2.5"));
    }

    #[test]
    fn v4_name_matches_canonical_variants() {
        for name in [
            "deepseek-v4",
            "deepseek_v4",
            "deepseek.v4",
            "deepseekv4",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-v4-flash-2507",
            "deepseek-v4.1",
            "deepseek_v4_thinking",
        ] {
            assert!(is_deepseek_v4_name(name), "expected {name} to match V4");
        }
    }

    #[test]
    fn v4_name_rejects_non_v4() {
        // Composite names that previously short-circuited to V4 before the
        // V3.2 branch — now correctly rejected.
        for name in [
            "deepseek-v3.2-v4-foo",
            "my-deepseek-v4",
            "deepseek-v40",
            "deepseek-v4pro",
            "deepseekv40",
            "deepseek-v3",
            "deepseek-v3.2",
            "deepseek-r1",
            "qwen3-v4", // only deepseek-prefixed names qualify
            "dsflash",
            "",
        ] {
            assert!(
                !is_deepseek_v4_name(name),
                "expected {name} to NOT match V4",
            );
        }
    }

    #[test]
    fn v4_detection_prefers_config_model_type() {
        // config.json `model_type = "deepseek_v4"` wins regardless of what
        // the operator calls the model via --served-model-name.
        let v4 = Some("deepseek_v4".to_string());
        for display in ["dsflash", "my-pet-model", "llama-3-8b", ""] {
            assert!(
                is_deepseek_v4(&v4, display),
                "config says deepseek_v4, display {display:?} — expected V4",
            );
        }

        // A concrete non-V4 config.json suppresses the display-name fallback.
        // Even if the operator names the served model "deepseek-v4", a model
        // with `model_type = "llama"` is NOT DeepSeek-V4.
        let llama = Some("llama".to_string());
        for display in ["deepseek-v4", "deepseek-v4-flash", "anything"] {
            assert!(
                !is_deepseek_v4(&llama, display),
                "config says llama, display {display:?} — expected NOT V4",
            );
        }

        // No config.json — fall back to display-name match.
        assert!(is_deepseek_v4(&None, "deepseek-v4-flash"));
        assert!(!is_deepseek_v4(&None, "dsflash"));

        // A config.json with `"model_type": ""` is treated as "no signal" at
        // the call site (normalized to None before is_deepseek_v4 is called),
        // so the display-name fallback still runs — pin that contract.
        let empty: Option<String> = None;
        assert!(is_deepseek_v4(&empty, "deepseek-v4-flash"));
        assert!(!is_deepseek_v4(&empty, "dsflash"));
    }

    #[test]
    fn v3_2_detection_prefers_config_model_type() {
        // config says deepseek_v3_2, any non-"exp" display name triggers.
        let v3_2 = Some("deepseek_v3_2".to_string());
        assert!(is_deepseek_v3_2_non_exp(&v3_2, "whatever"));
        assert!(is_deepseek_v3_2_non_exp(&v3_2, "deepseek-v3.2"));
        // V3.2-Exp is a separate model family; suppress even via config.
        assert!(!is_deepseek_v3_2_non_exp(&v3_2, "deepseek-v3.2-exp"));

        // The actual HF config.json spelling has no underscore between 3 and 2
        // (`deepseek_v32`). It must trigger identically to the internal key.
        let hf_real = Some("deepseek_v32".to_string());
        assert!(is_deepseek_v3_2_non_exp(&hf_real, "whatever"));
        assert!(is_deepseek_v3_2_non_exp(&hf_real, "deepseek-v3.2-nvfp4"));
        assert!(!is_deepseek_v3_2_non_exp(&hf_real, "deepseek-v3.2-exp"));

        // Other config types lose regardless of display name.
        let other = Some("deepseek_v4".to_string());
        assert!(!is_deepseek_v3_2_non_exp(&other, "deepseek-v3.2"));

        // No config — fall back to the original display-name heuristic.
        assert!(is_deepseek_v3_2_non_exp(&None, "deepseek-v3.2-pro"));
        assert!(!is_deepseek_v3_2_non_exp(&None, "deepseek-v3.2-exp"));
        assert!(!is_deepseek_v3_2_non_exp(&None, "deepseek-v4"));
    }
}
