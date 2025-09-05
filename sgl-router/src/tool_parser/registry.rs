use crate::tool_parser::parsers::{
    DeepSeekParser, Glm4MoeParser, GptOssParser, JsonParser, KimiK2Parser, LlamaParser,
    MistralParser, PythonicParser, QwenParser, Step3Parser,
};
use crate::tool_parser::traits::ToolParser;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;

/// Global singleton registry instance - created once and reused
pub static GLOBAL_REGISTRY: Lazy<ParserRegistry> = Lazy::new(ParserRegistry::new_internal);

/// Registry for tool parsers and model mappings
pub struct ParserRegistry {
    /// Map of parser name to parser instance
    parsers: HashMap<String, Arc<dyn ToolParser>>,
    /// Map of model name/pattern to parser name
    model_mapping: HashMap<String, String>,
    /// Default parser to use when no match found
    default_parser: String,
}

impl ParserRegistry {
    /// Get the global singleton instance
    pub fn new() -> &'static Self {
        &GLOBAL_REGISTRY
    }

    /// Create a new instance for testing (not the singleton)
    #[cfg(test)]
    pub fn new_for_testing() -> Self {
        Self::new_internal()
    }

    /// Internal constructor for creating the singleton instance
    fn new_internal() -> Self {
        let mut registry = Self {
            parsers: HashMap::new(),
            model_mapping: HashMap::new(),
            default_parser: "json".to_string(),
        };

        // Register default parsers
        registry.register_default_parsers();

        // Register default model mappings
        registry.register_default_mappings();

        registry
    }

    /// Register a parser
    pub fn register_parser(&mut self, name: impl Into<String>, parser: Arc<dyn ToolParser>) {
        self.parsers.insert(name.into(), parser);
    }

    /// Map a model name/pattern to a parser
    pub fn map_model(&mut self, model: impl Into<String>, parser: impl Into<String>) {
        self.model_mapping.insert(model.into(), parser.into());
    }

    /// Get parser for a specific model
    pub fn get_parser(&self, model: &str) -> Option<Arc<dyn ToolParser>> {
        // Try exact match first
        if let Some(parser_name) = self.model_mapping.get(model) {
            if let Some(parser) = self.parsers.get(parser_name) {
                return Some(parser.clone());
            }
        }

        // Try prefix matching with more specific patterns first
        // Collect all matching patterns and sort by specificity (longer = more specific)
        let mut matches: Vec<(&String, &String)> = self
            .model_mapping
            .iter()
            .filter(|(pattern, _)| {
                if pattern.ends_with('*') {
                    let prefix = &pattern[..pattern.len() - 1];
                    model.starts_with(prefix)
                } else {
                    false
                }
            })
            .collect();

        // Sort by pattern length in descending order (longer patterns are more specific)
        matches.sort_by_key(|(pattern, _)| std::cmp::Reverse(pattern.len()));

        // Return the first matching parser
        for (_, parser_name) in matches {
            if let Some(parser) = self.parsers.get(parser_name) {
                return Some(parser.clone());
            }
        }

        // Fall back to default parser if it exists
        self.parsers.get(&self.default_parser).cloned()
    }

    /// List all registered parsers
    pub fn list_parsers(&self) -> Vec<&str> {
        self.parsers.keys().map(|s| s.as_str()).collect()
    }

    /// List all model mappings
    pub fn list_mappings(&self) -> Vec<(&str, &str)> {
        self.model_mapping
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }

    /// Register default parsers
    fn register_default_parsers(&mut self) {
        // JSON parser - most common format
        self.register_parser("json", Arc::new(JsonParser::new()));

        // Mistral parser - [TOOL_CALLS] [...] format
        self.register_parser("mistral", Arc::new(MistralParser::new()));

        // Qwen parser - <tool_call>...</tool_call> format
        self.register_parser("qwen", Arc::new(QwenParser::new()));

        // Pythonic parser - [func(arg=val)] format
        self.register_parser("pythonic", Arc::new(PythonicParser::new()));

        // Llama parser - <|python_tag|>{...} or plain JSON format
        self.register_parser("llama", Arc::new(LlamaParser::new()));

        // DeepSeek V3 parser - Unicode tokens with JSON blocks
        self.register_parser("deepseek", Arc::new(DeepSeekParser::new()));

        // GLM-4 MoE parser - XML-style key-value format
        self.register_parser("glm4_moe", Arc::new(Glm4MoeParser::new()));

        // Step3 parser - StepTML XML format
        self.register_parser("step3", Arc::new(Step3Parser::new()));

        // Kimi K2 parser - Token-based with indexed functions
        self.register_parser("kimik2", Arc::new(KimiK2Parser::new()));

        // GPT-OSS parser - Channel format
        self.register_parser("gpt_oss", Arc::new(GptOssParser::new()));
    }

    /// Register default model mappings
    fn register_default_mappings(&mut self) {
        // OpenAI models
        self.map_model("gpt-4*", "json");
        self.map_model("gpt-3.5*", "json");
        self.map_model("gpt-4o*", "json");

        // Anthropic models
        self.map_model("claude-*", "json");

        // Mistral models - use Mistral parser
        self.map_model("mistral-*", "mistral");
        self.map_model("mixtral-*", "mistral");

        // Qwen models - use Qwen parser
        self.map_model("qwen*", "qwen");
        self.map_model("Qwen*", "qwen");

        // Llama models
        // Llama 4 uses pythonic format
        self.map_model("llama-4*", "pythonic");
        self.map_model("meta-llama-4*", "pythonic");
        // Llama 3.2 uses python_tag format
        self.map_model("llama-3.2*", "llama");
        self.map_model("meta-llama-3.2*", "llama");
        // Other Llama models use JSON
        self.map_model("llama-*", "json");
        self.map_model("meta-llama-*", "json");

        // DeepSeek models
        // DeepSeek V3 uses custom Unicode token format
        self.map_model("deepseek-v3*", "deepseek");
        self.map_model("deepseek-ai/DeepSeek-V3*", "deepseek");
        // DeepSeek V2 uses pythonic format
        self.map_model("deepseek-*", "pythonic");

        // GLM models
        // GLM-4 MoE uses XML-style format
        self.map_model("glm-4-moe*", "glm4_moe");
        self.map_model("THUDM/glm-4-moe*", "glm4_moe");
        self.map_model("glm-4.5*", "glm4_moe");
        // Other GLM models may use JSON
        self.map_model("glm-*", "json");

        // Step3 models
        self.map_model("step3*", "step3");
        self.map_model("Step-3*", "step3");

        // Kimi models
        self.map_model("kimi-k2*", "kimik2");
        self.map_model("Kimi-K2*", "kimik2");
        self.map_model("moonshot*/Kimi-K2*", "kimik2");

        // GPT-OSS models (T4-style)
        self.map_model("gpt-oss*", "gpt_oss");
        self.map_model("t4-*", "gpt_oss");

        // Other models default to JSON
        self.map_model("gemini-*", "json");
        self.map_model("palm-*", "json");
        self.map_model("gemma-*", "json");
    }

    /// Set the default parser
    pub fn set_default_parser(&mut self, name: impl Into<String>) {
        self.default_parser = name.into();
    }

    /// Check if a parser is registered
    pub fn has_parser(&self, name: &str) -> bool {
        self.parsers.contains_key(name)
    }
}

impl Default for &'static ParserRegistry {
    fn default() -> Self {
        ParserRegistry::new()
    }
}
