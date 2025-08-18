// Factory and registry for creating model-specific reasoning parsers.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::reasoning_parser::parsers::BaseReasoningParser;
use crate::reasoning_parser::traits::{ParseError, ParserConfig, ReasoningParser};

/// Type alias for parser creator functions.
type ParserCreator = Arc<dyn Fn() -> Box<dyn ReasoningParser> + Send + Sync>;

/// Registry for model-specific parsers.
#[derive(Clone)]
pub struct ParserRegistry {
    parsers: Arc<RwLock<HashMap<String, ParserCreator>>>,
    patterns: Arc<RwLock<Vec<(String, String)>>>, // (pattern, parser_name)
}

impl ParserRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            parsers: Arc::new(RwLock::new(HashMap::new())),
            patterns: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a parser creator for a given parser type.
    pub fn register_parser<F>(&self, name: &str, creator: F)
    where
        F: Fn() -> Box<dyn ReasoningParser> + Send + Sync + 'static,
    {
        let mut parsers = self.parsers.write().unwrap();
        parsers.insert(name.to_string(), Arc::new(creator));
    }

    /// Register a model pattern to parser mapping.
    /// Patterns are checked in order, first match wins.
    pub fn register_pattern(&self, pattern: &str, parser_name: &str) {
        let mut patterns = self.patterns.write().unwrap();
        patterns.push((pattern.to_string(), parser_name.to_string()));
    }

    /// Get a parser by exact name.
    pub fn get_parser(&self, name: &str) -> Option<Box<dyn ReasoningParser>> {
        let parsers = self.parsers.read().unwrap();
        parsers.get(name).map(|creator| creator())
    }

    /// Find a parser for a given model ID by pattern matching.
    pub fn find_parser_for_model(&self, model_id: &str) -> Option<Box<dyn ReasoningParser>> {
        let patterns = self.patterns.read().unwrap();
        let model_lower = model_id.to_lowercase();

        for (pattern, parser_name) in patterns.iter() {
            if model_lower.contains(&pattern.to_lowercase()) {
                return self.get_parser(parser_name);
            }
        }
        None
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating reasoning parsers based on model type.
pub struct ParserFactory {
    registry: ParserRegistry,
}

impl ParserFactory {
    /// Create a new factory with default parsers registered.
    pub fn new() -> Self {
        let registry = ParserRegistry::new();

        // Register base parser
        registry.register_parser("base", || {
            Box::new(BaseReasoningParser::new(ParserConfig::default()))
        });

        // Register DeepSeek-R1 parser
        registry.register_parser("deepseek_r1", || {
            let config = ParserConfig {
                think_start_token: "<think>".to_string(),
                think_end_token: "</think>".to_string(),
                force_reasoning: true,
                stream_reasoning: true,
                max_buffer_size: 65536,
            };
            Box::new(BaseReasoningParser::new(config).with_model_type("deepseek_r1".to_string()))
        });

        // Register Qwen3 parser
        registry.register_parser("qwen3", || {
            let config = ParserConfig {
                think_start_token: "<think>".to_string(),
                think_end_token: "</think>".to_string(),
                force_reasoning: false,
                stream_reasoning: true,
                max_buffer_size: 65536,
            };
            Box::new(BaseReasoningParser::new(config).with_model_type("qwen3".to_string()))
        });

        // Register Qwen3-thinking parser (forced reasoning)
        registry.register_parser("qwen3_thinking", || {
            let config = ParserConfig {
                think_start_token: "<think>".to_string(),
                think_end_token: "</think>".to_string(),
                force_reasoning: true,
                stream_reasoning: true,
                max_buffer_size: 65536,
            };
            Box::new(BaseReasoningParser::new(config).with_model_type("qwen3_thinking".to_string()))
        });

        // Register Kimi parser with Unicode tokens
        registry.register_parser("kimi", || {
            let config = ParserConfig {
                think_start_token: "◁think▷".to_string(),
                think_end_token: "◁/think▷".to_string(),
                force_reasoning: false,
                stream_reasoning: true,
                max_buffer_size: 65536,
            };
            Box::new(BaseReasoningParser::new(config).with_model_type("kimi".to_string()))
        });

        // Register model patterns
        registry.register_pattern("deepseek-r1", "deepseek_r1");
        registry.register_pattern("qwen3-thinking", "qwen3_thinking");
        registry.register_pattern("qwen-thinking", "qwen3_thinking");
        registry.register_pattern("qwen3", "qwen3");
        registry.register_pattern("qwen", "qwen3");
        registry.register_pattern("glm45", "qwen3"); // GLM45 uses same format as Qwen3
        registry.register_pattern("kimi", "kimi");
        registry.register_pattern("step3", "deepseek_r1"); // Step3 alias for DeepSeek-R1

        Self { registry }
    }

    /// Create a parser for the given model ID.
    /// Returns a no-op parser if model is not recognized.
    pub fn create(&self, model_id: &str) -> Result<Box<dyn ReasoningParser>, ParseError> {
        // First try to find by pattern
        if let Some(parser) = self.registry.find_parser_for_model(model_id) {
            return Ok(parser);
        }

        // Fall back to no-op parser (base parser without reasoning detection)
        let config = ParserConfig {
            think_start_token: "".to_string(),
            think_end_token: "".to_string(),
            force_reasoning: false,
            stream_reasoning: true,
            max_buffer_size: 65536,
        };
        Ok(Box::new(
            BaseReasoningParser::new(config).with_model_type("passthrough".to_string()),
        ))
    }

    /// Get the internal registry for custom registration.
    pub fn registry(&self) -> &ParserRegistry {
        &self.registry
    }
}

impl Default for ParserFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creates_deepseek_r1() {
        let factory = ParserFactory::new();
        let parser = factory.create("deepseek-r1-distill").unwrap();
        assert_eq!(parser.model_type(), "deepseek_r1");
    }

    #[test]
    fn test_factory_creates_qwen3() {
        let factory = ParserFactory::new();
        let parser = factory.create("qwen3-7b").unwrap();
        assert_eq!(parser.model_type(), "qwen3");
    }

    #[test]
    fn test_factory_creates_kimi() {
        let factory = ParserFactory::new();
        let parser = factory.create("kimi-chat").unwrap();
        assert_eq!(parser.model_type(), "kimi");
    }

    #[test]
    fn test_factory_fallback_to_passthrough() {
        let factory = ParserFactory::new();
        let parser = factory.create("unknown-model").unwrap();
        assert_eq!(parser.model_type(), "passthrough");
    }

    #[test]
    fn test_case_insensitive_matching() {
        let factory = ParserFactory::new();
        let parser1 = factory.create("DeepSeek-R1").unwrap();
        let parser2 = factory.create("QWEN3").unwrap();
        let parser3 = factory.create("Kimi").unwrap();

        assert_eq!(parser1.model_type(), "deepseek_r1");
        assert_eq!(parser2.model_type(), "qwen3");
        assert_eq!(parser3.model_type(), "kimi");
    }

    #[test]
    fn test_alias_models() {
        let factory = ParserFactory::new();
        let step3 = factory.create("step3-model").unwrap();
        let glm45 = factory.create("glm45-v2").unwrap();

        assert_eq!(step3.model_type(), "deepseek_r1");
        assert_eq!(glm45.model_type(), "qwen3");
    }
}
