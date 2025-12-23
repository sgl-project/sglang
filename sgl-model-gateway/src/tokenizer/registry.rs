//! Tokenizer Registry for dynamic tokenizer loading
//!
//! Provides thread-safe, deduplicated tokenizer loading for IGW mode where
//! multiple routers (HTTP and gRPC) need to share tokenizers across workers.

use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::Mutex;
use tracing::{debug, info};

use super::traits::Tokenizer;

/// Registry for managing tokenizers keyed by served_model_name
///
/// Features:
/// - Thread-safe concurrent access using DashMap
/// - Per-key locking to prevent duplicate loading
/// - Simple key scheme: served_model_name
pub struct TokenizerRegistry {
    /// Storage for loaded tokenizers
    tokenizers: DashMap<String, Arc<dyn Tokenizer>>,
    /// Per-key locks to prevent duplicate loading
    loading_locks: DashMap<String, Arc<Mutex<()>>>,
}

impl TokenizerRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tokenizers: DashMap::new(),
            loading_locks: DashMap::new(),
        }
    }

    /// Load and register a tokenizer by model ID
    ///
    /// If the tokenizer is already loaded, returns true immediately.
    /// Otherwise, uses the provided loader function to load it.
    /// Per-key locking ensures only one load happens per model, preventing race conditions.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier to use as key
    /// * `loader` - Async function that loads the tokenizer
    ///
    /// # Returns
    /// * `Ok(true)` - Successfully loaded and registered (or already registered)
    /// * `Err(message)` - Error message if loading fails
    ///
    /// # Example
    /// ```ignore
    /// registry.load("meta-llama/Llama-2-7b", || async {
    ///     create_tokenizer_async("/path/to/tokenizer").await
    /// }).await?;
    /// ```
    pub async fn load<F, Fut>(&self, model_id: &str, loader: F) -> Result<bool, String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Arc<dyn Tokenizer>, String>>,
    {
        // Fast path: already loaded
        if self.tokenizers.contains_key(model_id) {
            debug!("Tokenizer already registered for model: {}", model_id);
            return Ok(true);
        }

        debug!("Tokenizer cache miss for model: {}", model_id);

        // Acquire per-key lock to prevent duplicate loading
        let lock = self
            .loading_locks
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();

        let _guard = lock.lock().await;

        // Double-check after acquiring lock (another thread may have loaded it)
        if self.tokenizers.contains_key(model_id) {
            debug!("Tokenizer loaded by another thread for model: {}", model_id);
            return Ok(true);
        }

        // Load tokenizer
        info!("Loading tokenizer for model: {}", model_id);
        let tokenizer = loader().await?;

        // Store in registry
        self.tokenizers.insert(model_id.to_string(), tokenizer);

        // Remove the lock since it's no longer needed for this model.
        self.loading_locks.remove(model_id);

        info!(
            "Successfully loaded and registered tokenizer for model: {}",
            model_id
        );

        Ok(true)
    }

    /// Register a pre-loaded tokenizer
    ///
    /// Atomically inserts a tokenizer into the registry only if no tokenizer
    /// with the same model_name exists. Returns true if the tokenizer was inserted,
    /// false if one already existed.
    ///
    /// This method is thread-safe and uses atomic operations to prevent race conditions.
    /// If you need to replace an existing tokenizer, first use `remove()` then `register()`.
    ///
    /// # Arguments
    /// * `model_name` - The served_model_name to use as key
    /// * `tokenizer` - The tokenizer to register
    ///
    /// # Returns
    /// * `true` - If the tokenizer was successfully registered (didn't exist before)
    /// * `false` - If a tokenizer with this model_name already existed
    ///
    /// # Example
    /// ```ignore
    /// let tokenizer = create_tokenizer_blocking("/path/to/tokenizer")?;
    /// if registry.register("meta-llama/Llama-2-7b", tokenizer) {
    ///     info!("Tokenizer registered successfully");
    /// } else {
    ///     info!("Tokenizer already exists");
    /// }
    /// ```
    pub fn register(&self, model_name: &str, tokenizer: Arc<dyn Tokenizer>) -> bool {
        use dashmap::mapref::entry::Entry;
        match self.tokenizers.entry(model_name.to_string()) {
            Entry::Occupied(_) => {
                debug!(
                    "Tokenizer already exists for model: {}, skipping registration",
                    model_name
                );
                false
            }
            Entry::Vacant(entry) => {
                info!("Registering tokenizer for model: {}", model_name);
                entry.insert(tokenizer);
                true
            }
        }
    }

    /// Get a tokenizer if it's already loaded
    ///
    /// Returns None if the tokenizer hasn't been loaded yet.
    pub fn get(&self, model_name: &str) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizers.get(model_name).map(|t| t.clone())
    }

    /// Check if a tokenizer is loaded for the given model
    pub fn contains(&self, model_name: &str) -> bool {
        self.tokenizers.contains_key(model_name)
    }

    /// Get the number of loaded tokenizers
    pub fn len(&self) -> usize {
        self.tokenizers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tokenizers.is_empty()
    }

    /// List all registered tokenizer keys (model names)
    ///
    /// Returns a sorted vector of model names that have registered tokenizers.
    /// Returns an empty vector if no tokenizers are registered.
    pub fn list(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .tokenizers
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        keys.sort();
        keys
    }

    /// Remove a tokenizer from the registry
    ///
    /// Returns the tokenizer if it was present.
    pub fn remove(&self, model_name: &str) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizers.remove(model_name).map(|(_, v)| v)
    }

    /// Clear all tokenizers from the registry
    pub fn clear(&self) {
        self.tokenizers.clear();
        self.loading_locks.clear();
    }
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::sleep;

    use super::*;
    use crate::tokenizer::mock::MockTokenizer;

    #[tokio::test]
    async fn test_basic_operations() {
        let registry = TokenizerRegistry::new();

        // Registry starts empty
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(!registry.contains("model1"));

        // Load and register a tokenizer
        registry
            .load("model1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        // Verify it's loaded
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("model1"));

        // Get returns the tokenizer
        let tokenizer = registry.get("model1").unwrap();
        assert_eq!(
            tokenizer.vocab_size(),
            MockTokenizer::default().vocab_size()
        );

        // Remove works
        let removed = registry.remove("model1");
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_load_prevents_duplicate_loading() {
        let registry = Arc::new(TokenizerRegistry::new());
        let load_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn multiple tasks trying to load the same tokenizer
        let mut handles = vec![];
        for _ in 0..10 {
            let registry = registry.clone();
            let load_count = load_count.clone();
            let handle = tokio::spawn(async move {
                registry
                    .load("model1", || async {
                        // Simulate slow loading
                        sleep(Duration::from_millis(10)).await;
                        load_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                    })
                    .await
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Verify tokenizer was loaded only once
        assert_eq!(
            load_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "Tokenizer should be loaded exactly once despite concurrent requests"
        );
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn test_multiple_models() {
        let registry = TokenizerRegistry::new();

        // Load multiple tokenizers
        for i in 1..=5 {
            let model_name = format!("model{}", i);
            registry
                .load(&model_name, || async {
                    Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                })
                .await
                .unwrap();
        }

        assert_eq!(registry.len(), 5);
        assert!(registry.contains("model1"));
        assert!(registry.contains("model5"));
        assert!(!registry.contains("model6"));

        // Clear all
        registry.clear();
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_load_failure() {
        let registry = TokenizerRegistry::new();

        // Try to load with a failing loader
        let result = registry
            .load("failing_model", || async { Err("Load failed".to_string()) })
            .await;

        assert!(result.is_err());
        assert!(!registry.contains("failing_model"));
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_concurrent_different_models() {
        let registry = Arc::new(TokenizerRegistry::new());
        let mut handles = vec![];

        // Load different models concurrently
        for i in 1..=10 {
            let registry = registry.clone();
            let handle = tokio::spawn(async move {
                let model_name = format!("model{}", i);
                registry
                    .load(&model_name, || async {
                        sleep(Duration::from_millis(5)).await;
                        Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                    })
                    .await
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        assert_eq!(registry.len(), 10);
    }

    #[tokio::test]
    async fn test_register_only_if_absent() {
        let registry = TokenizerRegistry::new();
        let tokenizer1 = Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>;
        let tokenizer2 = Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>;

        // First registration should succeed
        assert!(registry.register("model1", tokenizer1.clone()));
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("model1"));

        // Second registration with same key should fail
        assert!(!registry.register("model1", tokenizer2.clone()));
        assert_eq!(registry.len(), 1);

        // Verify the original tokenizer is still there (not replaced)
        let retrieved = registry.get("model1").unwrap();
        assert_eq!(
            Arc::as_ptr(&retrieved),
            Arc::as_ptr(&tokenizer1),
            "Original tokenizer should not be replaced"
        );

        // Registration with different key should succeed
        assert!(registry.register("model2", tokenizer2));
        assert_eq!(registry.len(), 2);
    }

    #[tokio::test]
    async fn test_concurrent_register_same_model() {
        let registry = Arc::new(TokenizerRegistry::new());
        let success_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn multiple tasks trying to register the same model
        let mut handles = vec![];
        for _ in 0..10 {
            let registry = registry.clone();
            let success_count = success_count.clone();
            let handle = tokio::spawn(async move {
                let tokenizer = Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>;
                if registry.register("model1", tokenizer) {
                    success_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify only one registration succeeded
        assert_eq!(
            success_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "Only one concurrent registration should succeed"
        );
        assert_eq!(registry.len(), 1);
    }
}
