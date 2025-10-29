// Factory and registry for creating model-specific reasoning parsers.
// Now with parser pooling support for efficient reuse across requests.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use tokio::sync::Mutex;

use crate::reasoning_parser::{
    parsers::{
        BaseReasoningParser, DeepSeekR1Parser, Glm45Parser, KimiParser, Qwen3Parser,
        QwenThinkingParser, Step3Parser,
    },
    traits::{ParseError, ParserConfig, ReasoningParser},
};

/// Type alias for pooled parser instances.
/// Uses tokio::Mutex to avoid blocking the async executor.
pub type PooledParser = Arc<Mutex<Box<dyn ReasoningParser>>>;

/// Type alias for parser creator functions.
type ParserCreator = Arc<dyn Fn() -> Box<dyn ReasoningParser> + Send + Sync>;

/// Registry for model-specific parsers with pooling support.
#[derive(Clone)]
pub struct ParserRegistry {
    /// Creator functions for parsers (used when pool is empty)
    creators: Arc<RwLock<HashMap<String, ParserCreator>>>,
    /// Pooled parser instances for reuse
    pool: Arc<RwLock<HashMap<String, PooledParser>>>,
    /// Model pattern to parser name mappings
    patterns: Arc<RwLock<Vec<(String, String)>>>, // (pattern, parser_name)
}

impl ParserRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            creators: Arc::new(RwLock::new(HashMap::new())),
            pool: Arc::new(RwLock::new(HashMap::new())),
            patterns: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a parser creator for a given parser type.
    pub fn register_parser<F>(&self, name: &str, creator: F)
    where
        F: Fn() -> Box<dyn ReasoningParser> + Send + Sync + 'static,
    {
        let mut creators = self.creators.write().unwrap();
        creators.insert(name.to_string(), Arc::new(creator));
    }

    /// Register a model pattern to parser mapping.
    /// Patterns are checked in order, first match wins.
    pub fn register_pattern(&self, pattern: &str, parser_name: &str) {
        let mut patterns = self.patterns.write().unwrap();
        patterns.push((pattern.to_string(), parser_name.to_string()));
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

    /// Create a fresh parser instance by exact name (not pooled).
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_parser(&self, name: &str) -> Option<Box<dyn ReasoningParser>> {
        let creators = self.creators.read().unwrap();
        creators.get(name).map(|creator| creator())
    }

    /// Find a pooled parser for a given model ID by pattern matching.
    pub fn find_pooled_parser_for_model(&self, model_id: &str) -> Option<PooledParser> {
        let patterns = self.patterns.read().unwrap();
        let model_lower = model_id.to_lowercase();

        for (pattern, parser_name) in patterns.iter() {
            if model_lower.contains(&pattern.to_lowercase()) {
                return self.get_pooled_parser(parser_name);
            }
        }
        None
    }

    /// Check if a parser can be created for a specific model without actually creating it.
    /// Returns true if a parser is available (registered) for this model.
    pub fn has_parser_for_model(&self, model_id: &str) -> bool {
        let patterns = self.patterns.read().unwrap();
        let model_lower = model_id.to_lowercase();

        for (pattern, parser_name) in patterns.iter() {
            if model_lower.contains(&pattern.to_lowercase()) {
                let creators = self.creators.read().unwrap();
                return creators.contains_key(parser_name);
            }
        }
        false
    }

    /// Create a fresh parser instance for a given model ID by pattern matching (not pooled).
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_for_model(&self, model_id: &str) -> Option<Box<dyn ReasoningParser>> {
        let patterns = self.patterns.read().unwrap();
        let model_lower = model_id.to_lowercase();

        for (pattern, parser_name) in patterns.iter() {
            if model_lower.contains(&pattern.to_lowercase()) {
                return self.create_parser(parser_name);
            }
        }
        None
    }

    /// Clear the parser pool, forcing new instances to be created.
    /// Useful for testing or when parsers need to be reset globally.
    pub fn clear_pool(&self) {
        let mut pool = self.pool.write().unwrap();
        pool.clear();
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating reasoning parsers based on model type.
#[derive(Clone)]
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

        // Register DeepSeek-R1 parser (starts with in_reasoning=true)
        registry.register_parser("deepseek_r1", || Box::new(DeepSeekR1Parser::new()));

        // Register Qwen3 parser (starts with in_reasoning=false)
        registry.register_parser("qwen3", || Box::new(Qwen3Parser::new()));

        // Register Qwen3-thinking parser (starts with in_reasoning=true)
        registry.register_parser("qwen3_thinking", || Box::new(QwenThinkingParser::new()));

        // Register Kimi parser with Unicode tokens (starts with in_reasoning=false)
        registry.register_parser("kimi", || Box::new(KimiParser::new()));

        // Register GLM45 parser (same format as Qwen3 but separate for debugging)
        registry.register_parser("glm45", || Box::new(Glm45Parser::new()));

        // Register Step3 parser (same format as DeepSeek-R1 but separate for debugging)
        registry.register_parser("step3", || Box::new(Step3Parser::new()));

        // Register model patterns
        registry.register_pattern("deepseek-r1", "deepseek_r1");
        registry.register_pattern("qwen3-thinking", "qwen3_thinking");
        registry.register_pattern("qwen-thinking", "qwen3_thinking");
        registry.register_pattern("qwen3", "qwen3");
        registry.register_pattern("qwen", "qwen3");
        registry.register_pattern("glm45", "glm45");
        registry.register_pattern("kimi", "kimi");
        registry.register_pattern("step3", "step3");

        Self { registry }
    }

    /// Get a pooled parser for the given model ID.
    /// Returns a shared instance that can be used concurrently.
    /// Falls back to a passthrough parser if model is not recognized.
    pub fn get_pooled(&self, model_id: &str) -> PooledParser {
        // First try to find by pattern
        if let Some(parser) = self.registry.find_pooled_parser_for_model(model_id) {
            return parser;
        }

        // Fall back to no-op parser (get or create passthrough in pool)
        self.registry
            .get_pooled_parser("passthrough")
            .unwrap_or_else(|| {
                // Register passthrough if not already registered
                self.registry.register_parser("passthrough", || {
                    let config = ParserConfig {
                        think_start_token: "".to_string(),
                        think_end_token: "".to_string(),
                        stream_reasoning: true,
                        max_buffer_size: 65536,
                        initial_in_reasoning: false,
                    };
                    Box::new(
                        BaseReasoningParser::new(config).with_model_type("passthrough".to_string()),
                    )
                });
                self.registry.get_pooled_parser("passthrough").unwrap()
            })
    }

    /// Create a new parser instance for the given model ID.
    /// Returns a fresh instance (not pooled).
    /// Use this when you need an isolated parser instance.
    pub fn create(&self, model_id: &str) -> Result<Box<dyn ReasoningParser>, ParseError> {
        // First try to find by pattern
        if let Some(parser) = self.registry.create_for_model(model_id) {
            return Ok(parser);
        }

        // Fall back to no-op parser (base parser without reasoning detection)
        let config = ParserConfig {
            think_start_token: "".to_string(),
            think_end_token: "".to_string(),
            stream_reasoning: true,
            max_buffer_size: 65536,
            initial_in_reasoning: false,
        };
        Ok(Box::new(
            BaseReasoningParser::new(config).with_model_type("passthrough".to_string()),
        ))
    }

    /// Get the internal registry for custom registration.
    pub fn registry(&self) -> &ParserRegistry {
        &self.registry
    }

    /// Clear the parser pool.
    /// Useful for testing or when parsers need to be reset globally.
    pub fn clear_pool(&self) {
        self.registry.clear_pool();
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
    fn test_step3_model() {
        let factory = ParserFactory::new();
        let step3 = factory.create("step3-model").unwrap();
        assert_eq!(step3.model_type(), "step3");
    }

    #[test]
    fn test_glm45_model() {
        let factory = ParserFactory::new();
        let glm45 = factory.create("glm45-v2").unwrap();
        assert_eq!(glm45.model_type(), "glm45");
    }

    #[tokio::test]
    async fn test_pooled_parser_reuse() {
        let factory = ParserFactory::new();

        // Get the same parser twice - should be the same instance
        let parser1 = factory.get_pooled("deepseek-r1");
        let parser2 = factory.get_pooled("deepseek-r1");

        // Both should point to the same Arc
        assert!(Arc::ptr_eq(&parser1, &parser2));

        // Different models should get different parsers
        let parser3 = factory.get_pooled("qwen3");
        assert!(!Arc::ptr_eq(&parser1, &parser3));
    }

    #[tokio::test]
    async fn test_pooled_parser_concurrent_access() {
        let factory = ParserFactory::new();
        let parser = factory.get_pooled("deepseek-r1");

        // Spawn multiple async tasks that use the same parser
        let mut handles = vec![];

        for i in 0..3 {
            let parser_clone = Arc::clone(&parser);
            let handle = tokio::spawn(async move {
                let mut parser = parser_clone.lock().await;
                let input = format!("thread {} reasoning</think>answer", i);
                let result = parser.detect_and_parse_reasoning(&input).unwrap();
                assert_eq!(result.normal_text, "answer");
                assert!(result.reasoning_text.contains("reasoning"));
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_pool_clearing() {
        let factory = ParserFactory::new();

        // Get a pooled parser
        let parser1 = factory.get_pooled("deepseek-r1");

        // Clear the pool
        factory.clear_pool();

        // Get another parser - should be a new instance
        let parser2 = factory.get_pooled("deepseek-r1");

        // They should be different instances (different Arc pointers)
        assert!(!Arc::ptr_eq(&parser1, &parser2));
    }

    #[tokio::test]
    async fn test_passthrough_parser_pooling() {
        let factory = ParserFactory::new();

        // Unknown models should get passthrough parser
        let parser1 = factory.get_pooled("unknown-model-1");
        let parser2 = factory.get_pooled("unknown-model-2");

        // Both should use the same passthrough parser instance
        assert!(Arc::ptr_eq(&parser1, &parser2));

        let parser = parser1.lock().await;
        assert_eq!(parser.model_type(), "passthrough");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_high_concurrency_parser_access() {
        use std::{
            sync::atomic::{AtomicUsize, Ordering},
            time::Instant,
        };

        let factory = ParserFactory::new();
        let num_tasks = 100;
        let requests_per_task = 50;
        let models = vec!["deepseek-r1", "qwen3", "kimi", "qwen3-thinking"];

        // Track successful operations
        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));

        let start = Instant::now();
        let mut handles = vec![];

        for task_id in 0..num_tasks {
            let factory = factory.clone();
            let models = models.clone();
            let success_count = Arc::clone(&success_count);
            let error_count = Arc::clone(&error_count);

            let handle = tokio::spawn(async move {
                for request_id in 0..requests_per_task {
                    // Rotate through different models
                    let model = &models[(task_id + request_id) % models.len()];
                    let parser = factory.get_pooled(model);

                    // Use async lock - tokio::Mutex doesn't poison
                    let mut p = parser.lock().await;

                    // Simulate realistic parsing work with substantial text
                    // Typical reasoning can be 500-5000 tokens
                    let reasoning_text = format!(
                        "Task {} is processing request {}. Let me think through this step by step. \
                        First, I need to understand the problem. The problem involves analyzing data \
                        and making calculations. Let me break this down: \n\
                        1. Initial analysis shows that we have multiple variables to consider. \
                        2. The data suggests a pattern that needs further investigation. \
                        3. Computing the values: {} * {} = {}. \
                        4. Cross-referencing with previous results indicates consistency. \
                        5. The mathematical proof follows from the axioms... \
                        6. Considering edge cases and boundary conditions... \
                        7. Validating against known constraints... \
                        8. The conclusion follows logically from premises A, B, and C. \
                        This reasoning chain demonstrates the validity of our approach.",
                        task_id, request_id, task_id, request_id, task_id * request_id
                    );

                    let answer_text = format!(
                        "Based on my analysis, the answer for task {} request {} is: \
                        The solution involves multiple steps as outlined in the reasoning. \
                        The final result is {} with confidence level high. \
                        This conclusion is supported by rigorous mathematical analysis \
                        and has been validated against multiple test cases. \
                        The implementation should handle edge cases appropriately.",
                        task_id,
                        request_id,
                        task_id * request_id
                    );

                    let input = format!("<think>{}</think>{}", reasoning_text, answer_text);

                    match p.detect_and_parse_reasoning(&input) {
                        Ok(result) => {
                            // Note: Some parsers with stream_reasoning=true won't accumulate reasoning text
                            assert!(result.normal_text.contains(&format!("task {}", task_id)));

                            // For parsers that accumulate reasoning (stream_reasoning=false)
                            // the reasoning_text should be populated
                            if !result.reasoning_text.is_empty() {
                                assert!(result
                                    .reasoning_text
                                    .contains(&format!("Task {}", task_id)));
                                assert!(result.reasoning_text.len() > 500); // Ensure substantial reasoning
                            }

                            // Normal text should always be present
                            assert!(result.normal_text.len() > 100); // Ensure substantial answer
                            success_count.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            eprintln!("Parse error: {:?}", e);
                            error_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    // Explicitly drop the lock to release it quickly
                    drop(p);
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        let duration = start.elapsed();
        let total_requests = num_tasks * requests_per_task;
        let successes = success_count.load(Ordering::Relaxed);
        let errors = error_count.load(Ordering::Relaxed);

        // Print stats for debugging
        println!(
            "High concurrency test: {} tasks, {} requests each",
            num_tasks, requests_per_task
        );
        println!(
            "Completed in {:?}, {} successes, {} errors",
            duration, successes, errors
        );
        println!(
            "Throughput: {:.0} requests/sec",
            (total_requests as f64) / duration.as_secs_f64()
        );

        // All requests should succeed
        assert_eq!(successes, total_requests);
        assert_eq!(errors, 0);

        // Performance check: should handle at least 1000 req/sec
        let throughput = (total_requests as f64) / duration.as_secs_f64();
        assert!(
            throughput > 1000.0,
            "Throughput too low: {:.0} req/sec",
            throughput
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_pool_modifications() {
        let factory = ParserFactory::new();
        let mut handles = vec![];

        // Task 1: Continuously get parsers
        let factory1 = factory.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..100 {
                let _parser = factory1.get_pooled("deepseek-r1");
            }
        }));

        // Task 2: Continuously clear pool
        let factory2 = factory.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..10 {
                factory2.clear_pool();
                tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
            }
        }));

        // Task 3: Get different parsers
        let factory3 = factory.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..100 {
                let models = ["qwen3", "kimi", "unknown"];
                let _parser = factory3.get_pooled(models[i % 3]);
            }
        }));

        // Wait for all tasks - should not deadlock or panic
        for handle in handles {
            handle.await.unwrap();
        }
    }
}
