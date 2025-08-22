use crate::tool_parser::traits::ToolParser;
use std::collections::HashMap;
use std::sync::Arc;

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
    /// Create a new parser registry with default mappings
    pub fn new() -> Self {
        let mut registry = Self {
            parsers: HashMap::new(),
            model_mapping: HashMap::new(),
            default_parser: "json".to_string(),
        };

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

        // Try prefix matching (e.g., "gpt-4" matches "gpt-*")
        for (pattern, parser_name) in &self.model_mapping {
            if pattern.ends_with('*') {
                let prefix = &pattern[..pattern.len() - 1];
                if model.starts_with(prefix) {
                    if let Some(parser) = self.parsers.get(parser_name) {
                        return Some(parser.clone());
                    }
                }
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

    /// Register default model mappings
    fn register_default_mappings(&mut self) {
        // OpenAI models
        self.map_model("gpt-4*", "json");
        self.map_model("gpt-3.5*", "json");
        self.map_model("gpt-4o*", "json");

        // Anthropic models
        self.map_model("claude-*", "json");

        // Mistral models
        self.map_model("mistral-*", "mistral");
        self.map_model("mixtral-*", "mistral");

        // Qwen models
        self.map_model("qwen*", "qwen");

        // Llama models
        self.map_model("llama-*", "llama");
        self.map_model("meta-llama-*", "llama");

        // Other models default to JSON
        self.map_model("gemini-*", "json");
        self.map_model("palm-*", "json");
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

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}
