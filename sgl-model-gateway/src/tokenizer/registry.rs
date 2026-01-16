//! Tokenizer Registry for dynamic tokenizer loading
//!
//! Provides thread-safe, deduplicated tokenizer loading for IGW mode where
//! multiple routers (HTTP and gRPC) need to share tokenizers across workers.
//!
//! ## ID vs Name Lookup
//!
//! Tokenizers are stored with two keys:
//! - **ID (UUID)**: Unique identifier generated at registration, immutable
//! - **Name**: User-provided identifier, must be unique
//!
//! Lookup behavior:
//! - `get(key)`: Tries name first, then ID (backward compatible)
//! - `get_by_id(id)`: Exact ID match only
//! - `get_by_name(name)`: Exact name match only
//! - `remove(name)`: Removes by name
//! - `remove_by_id(id)`: Removes by ID

use std::sync::Arc;

use dashmap::DashMap;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, info};
use uuid::Uuid;

use super::traits::Tokenizer;

/// Outcome of a tokenizer load operation
#[derive(Debug, Clone)]
pub enum LoadOutcome {
    /// Tokenizer was newly loaded and registered
    Loaded { id: String },
    /// Tokenizer already existed, returning existing ID
    AlreadyExists { id: String },
}

impl LoadOutcome {
    /// Get the ID regardless of outcome
    pub fn id(&self) -> &str {
        match self {
            LoadOutcome::Loaded { id } => id,
            LoadOutcome::AlreadyExists { id } => id,
        }
    }

    /// Returns true if the tokenizer was newly loaded
    pub fn is_newly_loaded(&self) -> bool {
        matches!(self, LoadOutcome::Loaded { .. })
    }
}

/// Error type for tokenizer loading operations
#[derive(Debug, Error)]
pub enum LoadError {
    /// Name cannot be empty
    #[error("tokenizer name cannot be empty")]
    EmptyName,
    /// Source cannot be empty
    #[error("tokenizer source cannot be empty")]
    EmptySource,
    /// Loading failed
    #[error("{0}")]
    LoadFailed(String),
}

/// Metadata and tokenizer instance for a registered tokenizer
#[derive(Clone)]
pub struct TokenizerEntry {
    /// Unique identifier (UUID)
    pub id: String,
    /// User-provided name
    pub name: String,
    /// Source path or HuggingFace model ID
    pub source: String,
    /// The tokenizer instance
    pub tokenizer: Arc<dyn Tokenizer>,
}

/// Registry for managing tokenizers keyed by UUID
///
/// Features:
/// - Thread-safe concurrent access using DashMap
/// - Per-key locking to prevent duplicate loading
/// - Lookup by UUID (primary) or name (secondary index)
pub struct TokenizerRegistry {
    /// Storage for loaded tokenizers, keyed by UUID
    tokenizers: DashMap<String, TokenizerEntry>,
    /// Secondary index: name -> UUID for lookup
    name_to_id: DashMap<String, String>,
    /// Per-key locks to prevent duplicate loading
    loading_locks: DashMap<String, Arc<Mutex<()>>>,
}

/// RAII guard that removes the loading lock entry on drop.
/// Ensures cleanup happens on normal completion, early return, or panic.
struct LoadingLockGuard<'a> {
    locks: &'a DashMap<String, Arc<Mutex<()>>>,
    key: String,
}

impl Drop for LoadingLockGuard<'_> {
    fn drop(&mut self) {
        self.locks.remove(&self.key);
    }
}

impl TokenizerRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tokenizers: DashMap::new(),
            name_to_id: DashMap::new(),
            loading_locks: DashMap::new(),
        }
    }

    /// Generate a new UUID for a tokenizer
    pub fn generate_id() -> String {
        Uuid::new_v4().to_string()
    }

    /// Load and register a tokenizer
    ///
    /// Validates inputs, handles deduplication, and loads the tokenizer if needed.
    /// Per-key locking ensures only one load happens per name, preventing race conditions.
    ///
    /// # Arguments
    /// * `id` - Pre-generated UUID (use `generate_id()` to create one)
    /// * `name` - User-provided name (used for deduplication, must not be empty)
    /// * `source` - Source path or HuggingFace model ID (must not be empty)
    /// * `loader` - Async function that loads the tokenizer
    ///
    /// # Returns
    /// * `Ok(LoadOutcome::Loaded { id })` - Tokenizer was newly loaded
    /// * `Ok(LoadOutcome::AlreadyExists { id })` - Tokenizer already existed
    /// * `Err(LoadError)` - Validation failed or loading failed
    pub async fn load<F, Fut>(
        &self,
        id: &str,
        name: &str,
        source: &str,
        loader: F,
    ) -> Result<LoadOutcome, LoadError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Arc<dyn Tokenizer>, String>>,
    {
        // Validate inputs
        if name.is_empty() {
            return Err(LoadError::EmptyName);
        }
        if source.is_empty() {
            return Err(LoadError::EmptySource);
        }

        // Fast path: already loaded by name
        if let Some(existing_id) = self.name_to_id.get(name) {
            debug!("Tokenizer already registered for name: {}", name);
            return Ok(LoadOutcome::AlreadyExists {
                id: existing_id.clone(),
            });
        }

        debug!("Tokenizer cache miss for name: {}", name);

        // Acquire per-name lock to prevent duplicate loading
        let lock = self
            .loading_locks
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();

        let _mutex_guard = lock.lock().await;
        let _lock_cleanup = LoadingLockGuard {
            locks: &self.loading_locks,
            key: name.to_string(),
        };

        // Double-check after acquiring lock (another thread may have loaded it)
        if let Some(existing_id) = self.name_to_id.get(name) {
            debug!("Tokenizer loaded by another thread for name: {}", name);
            return Ok(LoadOutcome::AlreadyExists {
                id: existing_id.clone(),
            });
        }

        // Load tokenizer
        info!("Loading tokenizer '{}' from source: {}", name, source);
        let result = loader().await;

        let tokenizer = result.map_err(LoadError::LoadFailed)?;

        // Create entry with provided ID
        let entry = TokenizerEntry {
            id: id.to_string(),
            name: name.to_string(),
            source: source.to_string(),
            tokenizer,
        };

        // Store in registry
        self.tokenizers.insert(id.to_string(), entry);
        self.name_to_id.insert(name.to_string(), id.to_string());

        info!(
            "Successfully registered tokenizer '{}' with id: {}",
            name, id
        );

        Ok(LoadOutcome::Loaded { id: id.to_string() })
    }

    /// Register a preloaded tokenizer with a pre-generated ID
    ///
    /// Atomically inserts a tokenizer into the registry only if no tokenizer
    /// with the same name exists. Returns the ID if successful.
    ///
    /// This is primarily used for testing. Production code should use `load()`.
    ///
    /// # Returns
    /// * `Some(id)` - If the tokenizer was successfully registered
    /// * `None` - If a tokenizer with this name already existed
    #[cfg(test)]
    pub(crate) fn register(
        &self,
        id: &str,
        name: &str,
        source: &str,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Option<String> {
        use dashmap::mapref::entry::Entry;

        // Check if name already exists
        match self.name_to_id.entry(name.to_string()) {
            Entry::Occupied(_) => {
                debug!(
                    "Tokenizer already exists for name: {}, skipping registration",
                    name
                );
                None
            }
            Entry::Vacant(name_entry) => {
                let entry = TokenizerEntry {
                    id: id.to_string(),
                    name: name.to_string(),
                    source: source.to_string(),
                    tokenizer,
                };

                info!("Registering tokenizer '{}' with id: {}", name, id);
                self.tokenizers.insert(id.to_string(), entry);
                name_entry.insert(id.to_string());
                Some(id.to_string())
            }
        }
    }

    /// Get a tokenizer by UUID
    pub fn get_by_id(&self, id: &str) -> Option<TokenizerEntry> {
        self.tokenizers.get(id).map(|e| e.clone())
    }

    /// Get a tokenizer by name
    pub fn get_by_name(&self, name: &str) -> Option<TokenizerEntry> {
        self.name_to_id
            .get(name)
            .and_then(|id| self.tokenizers.get(id.as_str()).map(|e| e.clone()))
    }

    /// Get a tokenizer (for backward compatibility, tries name first then ID)
    pub fn get(&self, name_or_id: &str) -> Option<Arc<dyn Tokenizer>> {
        self.get_by_name(name_or_id)
            .or_else(|| self.get_by_id(name_or_id))
            .map(|e| e.tokenizer)
    }

    /// Check if a tokenizer is registered by name
    pub fn contains(&self, name: &str) -> bool {
        self.name_to_id.contains_key(name)
    }

    /// Check if a tokenizer is registered by ID
    pub fn contains_id(&self, id: &str) -> bool {
        self.tokenizers.contains_key(id)
    }

    /// Get the number of loaded tokenizers
    pub fn len(&self) -> usize {
        self.tokenizers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tokenizers.is_empty()
    }

    /// List all registered tokenizers
    pub fn list(&self) -> Vec<TokenizerEntry> {
        let mut entries: Vec<TokenizerEntry> =
            self.tokenizers.iter().map(|e| e.value().clone()).collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    /// Remove a tokenizer by ID
    ///
    /// Returns the entry if it was present.
    pub fn remove_by_id(&self, id: &str) -> Option<TokenizerEntry> {
        if let Some((_, entry)) = self.tokenizers.remove(id) {
            self.name_to_id.remove(&entry.name);
            Some(entry)
        } else {
            None
        }
    }

    /// Remove a tokenizer by name
    ///
    /// Returns the entry if it was present.
    pub fn remove(&self, name: &str) -> Option<TokenizerEntry> {
        if let Some((_, id)) = self.name_to_id.remove(name) {
            self.tokenizers.remove(&id).map(|(_, e)| e)
        } else {
            None
        }
    }

    /// Clear all tokenizers from the registry
    pub fn clear(&self) {
        self.tokenizers.clear();
        self.name_to_id.clear();
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
        let id = TokenizerRegistry::generate_id();
        let outcome = registry
            .load(&id, "model1", "path/to/model", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        // Verify LoadOutcome::Loaded
        assert!(outcome.is_newly_loaded());
        assert_eq!(outcome.id(), id);

        // Verify it's loaded
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("model1"));
        assert!(registry.contains_id(&id));

        // Get returns the tokenizer
        let entry = registry.get_by_name("model1").unwrap();
        assert_eq!(entry.id, id);
        assert_eq!(entry.name, "model1");
        assert_eq!(entry.source, "path/to/model");

        // Remove works
        let removed = registry.remove_by_id(&id);
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_load_returns_already_exists() {
        let registry = TokenizerRegistry::new();
        let id1 = TokenizerRegistry::generate_id();
        let id2 = TokenizerRegistry::generate_id();

        // First load should return Loaded
        let outcome1 = registry
            .load(&id1, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
        assert!(outcome1.is_newly_loaded());
        assert_eq!(outcome1.id(), id1);

        // Second load with same name should return AlreadyExists with ORIGINAL id
        let outcome2 = registry
            .load(&id2, "model1", "source2", || async {
                panic!("Loader should not be called for duplicate name");
            })
            .await
            .unwrap();
        assert!(!outcome2.is_newly_loaded());
        assert_eq!(outcome2.id(), id1); // Returns the original ID, not id2

        // Registry still has only one entry
        assert_eq!(registry.len(), 1);

        // Original source is preserved
        let entry = registry.get_by_name("model1").unwrap();
        assert_eq!(entry.source, "source1");
    }

    #[tokio::test]
    async fn test_load_validation() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();

        // Empty name should fail
        let result = registry
            .load(&id, "", "source", || async {
                panic!("Loader should not be called for invalid input");
            })
            .await;
        assert!(matches!(result, Err(LoadError::EmptyName)));

        // Empty source should fail
        let result = registry
            .load(&id, "model", "", || async {
                panic!("Loader should not be called for invalid input");
            })
            .await;
        assert!(matches!(result, Err(LoadError::EmptySource)));

        // Registry should be empty (nothing was loaded)
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_load_prevents_duplicate_loading() {
        let registry = Arc::new(TokenizerRegistry::new());
        let load_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn multiple tasks trying to load the same tokenizer
        let mut handles = vec![];
        for i in 0..10 {
            let registry = registry.clone();
            let load_count = load_count.clone();
            let id = format!("id-{}", i);
            let handle = tokio::spawn(async move {
                registry
                    .load(&id, "model1", "source", || async {
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
            let id = TokenizerRegistry::generate_id();
            registry
                .load(&id, &model_name, "source", || async {
                    Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                })
                .await
                .unwrap();
        }

        assert_eq!(registry.len(), 5);
        assert!(registry.contains("model1"));
        assert!(registry.contains("model5"));
        assert!(!registry.contains("model6"));

        // List returns all with metadata
        let entries = registry.list();
        assert_eq!(entries.len(), 5);
        assert!(entries.iter().any(|e| e.name == "model1"));

        // Clear all
        registry.clear();
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_load_failure() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();

        // Try to load with a failing loader
        let result = registry
            .load(&id, "failing_model", "source", || async {
                Err("Load failed".to_string())
            })
            .await;

        assert!(result.is_err());
        assert!(!registry.contains("failing_model"));
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_get_by_name_and_id() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();

        registry
            .load(&id, "my-model", "hf/model", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        // Get by name
        let by_name = registry.get_by_name("my-model");
        assert!(by_name.is_some());
        assert_eq!(by_name.as_ref().unwrap().id, id);

        // Get by ID
        let by_id = registry.get_by_id(&id);
        assert!(by_id.is_some());
        assert_eq!(by_id.as_ref().unwrap().name, "my-model");

        // Generic get works with both
        assert!(registry.get("my-model").is_some());
        assert!(registry.get(&id).is_some());
    }

    #[tokio::test]
    async fn test_register_only_if_absent() {
        let registry = TokenizerRegistry::new();
        let id1 = TokenizerRegistry::generate_id();
        let id2 = TokenizerRegistry::generate_id();
        let tokenizer1 = Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>;
        let tokenizer2 = Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>;

        // First registration should succeed
        let result1 = registry.register(&id1, "model1", "source1", tokenizer1.clone());
        assert!(result1.is_some());
        assert_eq!(registry.len(), 1);

        // Second registration with same name should fail
        let result2 = registry.register(&id2, "model1", "source2", tokenizer2.clone());
        assert!(result2.is_none());
        assert_eq!(registry.len(), 1);

        // Original tokenizer should still be there
        let entry = registry.get_by_name("model1").unwrap();
        assert_eq!(entry.id, id1);
        assert_eq!(entry.source, "source1");

        // Registration with different name should succeed
        let id3 = TokenizerRegistry::generate_id();
        let result3 = registry.register(&id3, "model2", "source2", tokenizer2);
        assert!(result3.is_some());
        assert_eq!(registry.len(), 2);
    }

    #[tokio::test]
    async fn test_loading_lock_cleanup_on_panic() {
        let registry = Arc::new(TokenizerRegistry::new());

        // Spawn a task that will panic during loading
        let registry_clone = registry.clone();
        let handle = tokio::spawn(async move {
            registry_clone
                .load(
                    &TokenizerRegistry::generate_id(),
                    "panic-model",
                    "source",
                    || async {
                        panic!("Simulated panic during tokenizer loading");
                    },
                )
                .await
        });

        // Wait for the task - it should panic
        let result = handle.await;
        assert!(result.is_err(), "Task should have panicked");

        // The RAII guard should have cleaned up the loading lock.
        // Verify by attempting another load with the same name - it should work,
        // not deadlock or fail due to stale lock.
        let id = TokenizerRegistry::generate_id();
        let outcome = registry
            .load(&id, "panic-model", "source", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await;

        // Should succeed - the lock was properly cleaned up
        assert!(outcome.is_ok(), "Load should succeed after panic cleanup");
        assert!(outcome.unwrap().is_newly_loaded());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("panic-model"));
    }

    #[tokio::test]
    async fn test_loading_lock_cleanup_on_early_return() {
        let registry = Arc::new(TokenizerRegistry::new());

        // Load a tokenizer
        let id1 = TokenizerRegistry::generate_id();
        registry
            .load(&id1, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        // Now simulate concurrent load attempts where one thread wins
        // and another thread sees "already exists" in the double-check.
        // The RAII guard should clean up the lock on early return.

        // First, verify loading_locks is empty after successful load
        // by checking that we can load a different model without issues
        let id2 = TokenizerRegistry::generate_id();
        let outcome = registry
            .load(&id2, "model2", "source2", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        assert!(outcome.is_newly_loaded());
        assert_eq!(registry.len(), 2);

        // Try to load model1 again - should return AlreadyExists
        // and the lock should be cleaned up (not leak)
        let id3 = TokenizerRegistry::generate_id();
        let outcome = registry
            .load(&id3, "model1", "source1", || async {
                panic!("Loader should not be called for existing model");
            })
            .await
            .unwrap();

        assert!(!outcome.is_newly_loaded());
        assert_eq!(outcome.id(), id1); // Returns original ID
    }
}
