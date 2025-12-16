// Factory and pool for creating model-specific tool parsers with pooling support.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use tokio::sync::Mutex;

use crate::tool_parser::{
    parsers::{
        DeepSeekParser, Glm4MoeParser, JsonParser, KimiK2Parser, LlamaParser, MinimaxM2Parser,
        MistralParser, PassthroughParser, PythonicParser, QwenParser, Step3Parser,
    },
    traits::ToolParser,
};

/// Type alias for pooled parser instances.
pub type PooledParser = Arc<Mutex<Box<dyn ToolParser>>>;

/// Type alias for parser creator functions.
type ParserCreator = Arc<dyn Fn() -> Box<dyn ToolParser> + Send + Sync>;

/// Registry for model-specific tool parsers with pooling support.
#[derive(Clone)]
pub struct ParserRegistry {
    /// Creator functions for parsers (used when pool is empty)
    creators: Arc<RwLock<HashMap<String, ParserCreator>>>,
    /// Pooled parser instances for reuse
    pool: Arc<RwLock<HashMap<String, PooledParser>>>,
    /// Model pattern to parser name mappings
    model_mapping: Arc<RwLock<HashMap<String, String>>>,
    /// Default parser name
    default_parser: Arc<RwLock<String>>,
}

impl ParserRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            creators: Arc::new(RwLock::new(HashMap::new())),
            pool: Arc::new(RwLock::new(HashMap::new())),
            model_mapping: Arc::new(RwLock::new(HashMap::new())),
            default_parser: Arc::new(RwLock::new("passthrough".to_string())),
        }
    }

    /// Register a parser creator for a given parser type.
    pub fn register_parser<F>(&self, name: &str, creator: F)
    where
        F: Fn() -> Box<dyn ToolParser> + Send + Sync + 'static,
    {
        let mut creators = self.creators.write().unwrap();
        creators.insert(name.to_string(), Arc::new(creator));
    }

    /// Map a model name/pattern to a parser
    pub fn map_model(&self, model: impl Into<String>, parser: impl Into<String>) {
        let mut mapping = self.model_mapping.write().unwrap();
        mapping.insert(model.into(), parser.into());
    }

    /// Get a pooled parser by exact name.
    /// Returns a shared parser instance from the pool, creating one if needed.
    pub fn get_pooled_parser(&self, name: &str) -> Option<PooledParser> {
        // First check if we have a pooled instance
        {
            let pool = self.pool.read().unwrap();
            if let Some(parser) = pool.get(name) {
                return Some(Arc::clone(parser));
            }
        }

        // If not in pool, create one and add to pool
        let creators = self.creators.read().unwrap();
        if let Some(creator) = creators.get(name) {
            let parser = Arc::new(Mutex::new(creator()));

            // Add to pool for future use
            let mut pool = self.pool.write().unwrap();
            pool.insert(name.to_string(), Arc::clone(&parser));

            Some(parser)
        } else {
            None
        }
    }

    /// Check if a parser with the given name is registered.
    pub fn has_parser(&self, name: &str) -> bool {
        let creators = self.creators.read().unwrap();
        creators.contains_key(name)
    }

    /// Create a fresh (non-pooled) parser instance by exact name.
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_parser(&self, name: &str) -> Option<Box<dyn ToolParser>> {
        let creators = self.creators.read().unwrap();
        creators.get(name).map(|creator| creator())
    }

    /// Check if a parser can be created for a specific model without actually creating it.
    /// Returns true if a parser is available (registered) for this model.
    pub fn has_parser_for_model(&self, model: &str) -> bool {
        // Try exact match first
        {
            let mapping = self.model_mapping.read().unwrap();
            if let Some(parser_name) = mapping.get(model) {
                let creators = self.creators.read().unwrap();
                if creators.contains_key(parser_name) {
                    return true;
                }
            }
        }

        // Try prefix matching
        let model_mapping = self.model_mapping.read().unwrap();
        let best_match = model_mapping
            .iter()
            .filter(|(pattern, _)| {
                pattern.ends_with('*') && model.starts_with(&pattern[..pattern.len() - 1])
            })
            .max_by_key(|(pattern, _)| pattern.len());

        if let Some((_, parser_name)) = best_match {
            let creators = self.creators.read().unwrap();
            if creators.contains_key(parser_name) {
                return true;
            }
        }

        // Return false if no specific parser found for this model
        // (get_pooled will still fall back to default parser)
        false
    }

    /// Create a fresh (non-pooled) parser instance for a specific model.
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_for_model(&self, model: &str) -> Option<Box<dyn ToolParser>> {
        // Try exact match first
        {
            let mapping = self.model_mapping.read().unwrap();
            if let Some(parser_name) = mapping.get(model) {
                if let Some(parser) = self.create_parser(parser_name) {
                    return Some(parser);
                }
            }
        }

        // Try prefix matching with more specific patterns first
        let model_mapping = self.model_mapping.read().unwrap();
        let best_match = model_mapping
            .iter()
            .filter(|(pattern, _)| {
                pattern.ends_with('*') && model.starts_with(&pattern[..pattern.len() - 1])
            })
            .max_by_key(|(pattern, _)| pattern.len());

        // Return the best matching parser
        if let Some((_, parser_name)) = best_match {
            if let Some(parser) = self.create_parser(parser_name) {
                return Some(parser);
            }
        }

        // Fall back to default parser
        let default = self.default_parser.read().unwrap().clone();
        self.create_parser(&default)
    }

    /// Get parser for a specific model
    pub fn get_pooled_for_model(&self, model: &str) -> Option<PooledParser> {
        // Try exact match first
        {
            let mapping = self.model_mapping.read().unwrap();
            if let Some(parser_name) = mapping.get(model) {
                if let Some(parser) = self.get_pooled_parser(parser_name) {
                    return Some(parser);
                }
            }
        }

        // Try prefix matching with more specific patterns first
        let model_mapping = self.model_mapping.read().unwrap();
        let best_match = model_mapping
            .iter()
            .filter(|(pattern, _)| {
                pattern.ends_with('*') && model.starts_with(&pattern[..pattern.len() - 1])
            })
            .max_by_key(|(pattern, _)| pattern.len());

        // Return the best matching parser
        if let Some((_, parser_name)) = best_match {
            if let Some(parser) = self.get_pooled_parser(parser_name) {
                return Some(parser);
            }
        }

        // Fall back to default parser
        let default = self.default_parser.read().unwrap().clone();
        self.get_pooled_parser(&default)
    }

    /// Clear the parser pool, forcing new instances to be created.
    pub fn clear_pool(&self) {
        let mut pool = self.pool.write().unwrap();
        pool.clear();
    }

    /// Set the default parser
    pub fn set_default_parser(&self, name: impl Into<String>) {
        let mut default = self.default_parser.write().unwrap();
        *default = name.into();
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating tool parsers based on model type.
#[derive(Clone)]
pub struct ParserFactory {
    registry: ParserRegistry,
}

impl ParserFactory {
    /// Create a new factory with default parsers registered.
    pub fn new() -> Self {
        let registry = ParserRegistry::new();

        // Register default parsers
        registry.register_parser("passthrough", || Box::new(PassthroughParser::new()));
        registry.register_parser("json", || Box::new(JsonParser::new()));
        registry.register_parser("mistral", || Box::new(MistralParser::new()));
        registry.register_parser("qwen", || Box::new(QwenParser::new()));
        registry.register_parser("pythonic", || Box::new(PythonicParser::new()));
        registry.register_parser("llama", || Box::new(LlamaParser::new()));
        registry.register_parser("deepseek", || Box::new(DeepSeekParser::new()));
        registry.register_parser("glm4_moe", || Box::new(Glm4MoeParser::new()));
        registry.register_parser("step3", || Box::new(Step3Parser::new()));
        registry.register_parser("kimik2", || Box::new(KimiK2Parser::new()));
        registry.register_parser("minimax_m2", || Box::new(MinimaxM2Parser::new()));

        // Register default model mappings
        Self::register_default_mappings(&registry);

        Self { registry }
    }

    fn register_default_mappings(registry: &ParserRegistry) {
        // OpenAI models
        registry.map_model("gpt-4*", "json");
        registry.map_model("gpt-3.5*", "json");
        registry.map_model("gpt-4o*", "json");

        // Anthropic models
        registry.map_model("claude-*", "json");

        // Mistral models
        registry.map_model("mistral-*", "mistral");
        registry.map_model("mixtral-*", "mistral");

        // Qwen models
        registry.map_model("qwen*", "qwen");
        registry.map_model("Qwen*", "qwen");

        // Llama models
        registry.map_model("llama-4*", "pythonic");
        registry.map_model("meta-llama-4*", "pythonic");
        registry.map_model("llama-3.2*", "llama");
        registry.map_model("meta-llama-3.2*", "llama");
        registry.map_model("llama-*", "json");
        registry.map_model("meta-llama-*", "json");

        // DeepSeek models
        registry.map_model("deepseek-v3*", "deepseek");
        registry.map_model("deepseek-ai/DeepSeek-V3*", "deepseek");
        registry.map_model("deepseek-*", "pythonic");

        // GLM models
        registry.map_model("glm-4.5*", "glm4_moe");
        registry.map_model("glm-4.6*", "glm4_moe");
        registry.map_model("glm-*", "json");

        // Step3 models
        registry.map_model("step3*", "step3");
        registry.map_model("Step-3*", "step3");

        // Kimi models
        registry.map_model("kimi-k2*", "kimik2");
        registry.map_model("Kimi-K2*", "kimik2");
        registry.map_model("moonshot*/Kimi-K2*", "kimik2");

        // MiniMax models
        registry.map_model("minimax*", "minimax_m2");
        registry.map_model("MiniMax*", "minimax_m2");

        // Other models
        registry.map_model("gemini-*", "json");
        registry.map_model("palm-*", "json");
        registry.map_model("gemma-*", "json");
    }

    /// Get a pooled parser for the given model ID.
    /// Returns a shared instance that can be used concurrently.
    /// Falls back to passthrough parser if model is not recognized.
    pub fn get_pooled(&self, model_id: &str) -> PooledParser {
        self.registry
            .get_pooled_for_model(model_id)
            .unwrap_or_else(|| {
                // Fallback to passthrough parser (no-op, returns text unchanged)
                self.registry
                    .get_pooled_parser("passthrough")
                    .expect("Passthrough parser should always be registered")
            })
    }

    /// Get the internal registry for custom registration.
    pub fn registry(&self) -> &ParserRegistry {
        &self.registry
    }

    /// Clear the parser pool.
    pub fn clear_pool(&self) {
        self.registry.clear_pool();
    }

    /// Get a non-pooled parser for the given model ID (creates a fresh instance each time).
    /// This is useful for benchmarks and testing where you want independent parser instances.
    pub fn get_parser(&self, model_id: &str) -> Option<Arc<dyn ToolParser>> {
        // Determine which parser type to use
        let parser_type = {
            let mapping = self.registry.model_mapping.read().unwrap();

            // Try exact match first
            if let Some(parser_name) = mapping.get(model_id) {
                parser_name.clone()
            } else {
                // Try prefix matching
                let best_match = mapping
                    .iter()
                    .filter(|(pattern, _)| {
                        pattern.ends_with('*')
                            && model_id.starts_with(&pattern[..pattern.len() - 1])
                    })
                    .max_by_key(|(pattern, _)| pattern.len());

                if let Some((_, parser_name)) = best_match {
                    parser_name.clone()
                } else {
                    // Fall back to default
                    self.registry.default_parser.read().unwrap().clone()
                }
            }
        };

        let creators = self.registry.creators.read().unwrap();
        creators.get(&parser_type).map(|creator| {
            // Call the creator to get a Box<dyn ToolParser>, then convert to Arc
            let boxed_parser = creator();
            Arc::from(boxed_parser)
        })
    }

    /// List all registered parsers (for compatibility with old API).
    pub fn list_parsers(&self) -> Vec<String> {
        self.registry
            .creators
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }
}

impl Default for ParserFactory {
    fn default() -> Self {
        Self::new()
    }
}
