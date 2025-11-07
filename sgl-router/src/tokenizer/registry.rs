//! Dynamic Tokenizer Registry
//!
//! Manages a pool of tokenizers that are dynamically fetched from workers.
//! Provides thread-safe caching and automatic download of tokenizer bundles.

use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use dashmap::DashMap;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::{
    bundle::BundleExtractor,
    cache::{CacheConfig, CachedTokenizer},
    create_tokenizer_from_file,
};

/// Key for identifying a tokenizer in the registry
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TokenizerKey {
    /// Model identifier (e.g., "meta-llama/Llama-3.1-8B")
    pub model_id: String,
    /// SHA256 fingerprint of the tokenizer bundle
    pub fingerprint: String,
}

impl TokenizerKey {
    /// Create a new tokenizer key
    pub fn new(model_id: String, fingerprint: String) -> Self {
        Self {
            model_id,
            fingerprint,
        }
    }
}

/// Entry in the tokenizer registry
struct TokenizerEntry {
    /// The cached tokenizer instance
    tokenizer: Arc<CachedTokenizer>,
    /// Path to the extracted tokenizer files
    #[allow(dead_code)]
    path: PathBuf,
}

/// Dynamic tokenizer registry that fetches tokenizers from workers on-demand
///
/// # Architecture
/// - Uses `DashMap` for concurrent read access without locks
/// - Uses `tokio::sync::Mutex` for coordinating downloads (async-friendly)
/// - Implements double-checked locking to prevent duplicate downloads
/// - Caches extracted tokenizers on disk for reuse across restarts
///
/// # Example
/// ```ignore
/// let registry = TokenizerRegistry::new(cache_root, cache_config);
///
/// // Fetch or get cached tokenizer
/// let tokenizer = registry.get_or_fetch(
///     "meta-llama/Llama-3.1-8B",
///     "abc123...",
///     |model_id, fingerprint| async {
///         // Download bundle from worker
///         worker_client.get_tokenizer(model_id, fingerprint).await
///     }
/// ).await?;
/// ```
pub struct TokenizerRegistry {
    /// Cache of loaded tokenizers (model_id+fingerprint -> tokenizer)
    tokenizers: DashMap<TokenizerKey, Arc<TokenizerEntry>>,

    /// Mutex to coordinate pending downloads (prevents duplicate downloads)
    /// Maps (model_id, fingerprint) to an in-progress download
    download_locks: DashMap<TokenizerKey, Arc<Mutex<()>>>,

    /// Bundle extractor for unpacking tokenizer bundles
    extractor: BundleExtractor,

    /// Cache configuration for tokenizers
    cache_config: CacheConfig,
}

impl TokenizerRegistry {
    /// Create a new tokenizer registry
    ///
    /// # Arguments
    /// * `cache_root` - Base directory for tokenizer cache (e.g., `.tokenizer_cache`)
    /// * `cache_config` - Cache configuration for tokenizers
    pub fn new(cache_root: PathBuf, cache_config: CacheConfig) -> Self {
        info!("Initializing tokenizer registry at {:?}", cache_root);

        Self {
            tokenizers: DashMap::new(),
            download_locks: DashMap::new(),
            extractor: BundleExtractor::new(cache_root),
            cache_config,
        }
    }

    /// Get or fetch a tokenizer
    ///
    /// This method implements double-checked locking to ensure that:
    /// 1. Multiple requests for the same tokenizer don't trigger duplicate downloads
    /// 2. Cached tokenizers are returned immediately without locking
    /// 3. Only one download happens per unique (model_id, fingerprint) pair
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "meta-llama/Llama-3.1-8B")
    /// * `fingerprint` - SHA256 fingerprint of the tokenizer bundle
    /// * `fetch_fn` - Async function to fetch the tokenizer bundle from a worker
    ///
    /// # Returns
    /// A reference-counted cached tokenizer instance
    ///
    /// # Type Parameters
    /// * `F` - Async function type
    /// * `Fut` - Future type returned by the fetch function
    pub async fn get_or_fetch<F, Fut>(
        &self,
        model_id: &str,
        fingerprint: &str,
        fetch_fn: F,
    ) -> Result<Arc<CachedTokenizer>>
    where
        F: FnOnce(String, String) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<u8>>>,
    {
        let key = TokenizerKey::new(model_id.to_string(), fingerprint.to_string());

        // Fast path: check if tokenizer is already in cache
        if let Some(entry) = self.tokenizers.get(&key) {
            debug!(
                "Tokenizer cache hit for model {} (fingerprint: {})",
                model_id, fingerprint
            );
            // TODO: Add metrics counter for cache hits
            return Ok(entry.tokenizer.clone());
        }

        debug!(
            "Tokenizer cache miss for model {} (fingerprint: {})",
            model_id, fingerprint
        );
        // TODO: Add metrics counter for cache misses

        // Get or create a download lock for this tokenizer
        let lock = self
            .download_locks
            .entry(key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();

        // Acquire the lock to coordinate downloads
        let _guard = lock.lock().await;

        // Double-check: another task might have loaded it while we waited
        if let Some(entry) = self.tokenizers.get(&key) {
            debug!("Tokenizer loaded by another task: {}", model_id);
            return Ok(entry.tokenizer.clone());
        }

        // No cached tokenizer - need to fetch and extract
        info!(
            "Fetching tokenizer bundle for model {} (fingerprint: {})",
            model_id, fingerprint
        );
        // TODO: Add metrics counter for downloads

        let bundle_data = fetch_fn(model_id.to_string(), fingerprint.to_string())
            .await
            .with_context(|| format!("Failed to fetch tokenizer bundle for model {}", model_id))?;

        debug!(
            "Downloaded tokenizer bundle for {} ({} bytes)",
            model_id,
            bundle_data.len()
        );

        // Extract the bundle
        let extract_path = self
            .extractor
            .extract_bundle(&bundle_data, model_id, fingerprint)
            .with_context(|| {
                format!("Failed to extract tokenizer bundle for model {}", model_id)
            })?;

        // Load the tokenizer from the extracted files
        let tokenizer = self
            .load_tokenizer_from_path(&extract_path)
            .await
            .with_context(|| format!("Failed to load tokenizer from path {:?}", extract_path))?;

        // Create entry and insert into cache
        let entry = Arc::new(TokenizerEntry {
            tokenizer: tokenizer.clone(),
            path: extract_path,
        });

        self.tokenizers.insert(key.clone(), entry);

        // Clean up the download lock
        self.download_locks.remove(&key);

        info!("Successfully loaded tokenizer for model {}", model_id);

        Ok(tokenizer)
    }

    /// Load a tokenizer from a directory path
    ///
    /// Tries to find and load the main tokenizer file (tokenizer.json, etc.)
    async fn load_tokenizer_from_path(&self, path: &PathBuf) -> Result<Arc<CachedTokenizer>> {
        use tokio::fs;

        // Try to find tokenizer.json first (most common)
        let tokenizer_json = path.join("tokenizer.json");
        if fs::try_exists(&tokenizer_json).await.unwrap_or(false) {
            debug!("Loading tokenizer from {:?}", tokenizer_json);
            let tokenizer = create_tokenizer_from_file(
                tokenizer_json.to_str().context("Invalid path encoding")?,
            )
            .context("Failed to create tokenizer from file")?;

            return Ok(Arc::new(CachedTokenizer::new(
                tokenizer,
                self.cache_config.clone(),
            )));
        }

        // Fallback: try tokenizer_config.json (some models use this)
        let tokenizer_config = path.join("tokenizer_config.json");
        if fs::try_exists(&tokenizer_config).await.unwrap_or(false) {
            warn!("Using tokenizer_config.json for model at {:?}", path);
            let tokenizer = create_tokenizer_from_file(
                tokenizer_config.to_str().context("Invalid path encoding")?,
            )
            .context("Failed to create tokenizer from config file")?;

            return Ok(Arc::new(CachedTokenizer::new(
                tokenizer,
                self.cache_config.clone(),
            )));
        }

        Err(anyhow::anyhow!(
            "No tokenizer file found in directory {:?}",
            path
        ))
    }

    /// Get the number of cached tokenizers
    pub fn len(&self) -> usize {
        self.tokenizers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tokenizers.is_empty()
    }

    /// Clear all cached tokenizers
    ///
    /// This removes all in-memory tokenizer instances but does not delete
    /// the extracted files on disk.
    pub fn clear(&self) {
        info!("Clearing tokenizer registry ({} entries)", self.len());
        self.tokenizers.clear();
        self.download_locks.clear();
    }

    /// Get statistics about the registry
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            cached_tokenizers: self.tokenizers.len(),
            pending_downloads: self.download_locks.len(),
        }
    }

    /// Pre-warm the cache by scanning the cache directory for existing tokenizers
    ///
    /// This is useful for recovering state after a restart, loading tokenizers
    /// that were previously downloaded.
    ///
    /// # Returns
    /// The number of tokenizers loaded from disk
    pub async fn prewarm_from_disk(&self) -> Result<usize> {
        use tokio::fs;

        let cache_root = self.extractor.cache_root();
        if !fs::try_exists(cache_root).await.unwrap_or(false) {
            debug!("Cache directory does not exist, skipping prewarm");
            return Ok(0);
        }

        let mut loaded_count = 0;

        // Iterate through model directories
        let mut model_dirs = fs::read_dir(cache_root)
            .await
            .with_context(|| format!("Failed to read cache directory {:?}", cache_root))?;

        while let Some(model_entry) = model_dirs.next_entry().await.transpose() {
            let model_entry = match model_entry {
                Ok(e) => e,
                Err(e) => {
                    warn!("Failed to read model directory entry: {}", e);
                    continue;
                }
            };

            let model_path = model_entry.path();
            let metadata = match model_entry.metadata().await {
                Ok(m) => m,
                Err(e) => {
                    warn!("Failed to read metadata for {:?}: {}", model_path, e);
                    continue;
                }
            };

            if !metadata.is_dir() {
                continue;
            }

            let model_name = match model_path.file_name() {
                Some(name) => name.to_string_lossy().to_string(),
                None => continue,
            };

            // Iterate through fingerprint directories
            let mut fingerprint_dirs = match fs::read_dir(&model_path).await {
                Ok(dirs) => dirs,
                Err(e) => {
                    warn!(
                        "Failed to read fingerprint directories for {}: {}",
                        model_name, e
                    );
                    continue;
                }
            };

            while let Some(fp_entry) = fingerprint_dirs.next_entry().await.transpose() {
                let fp_entry = match fp_entry {
                    Ok(e) => e,
                    Err(e) => {
                        warn!("Failed to read fingerprint entry: {}", e);
                        continue;
                    }
                };

                let fp_path = fp_entry.path();
                let fp_metadata = match fp_entry.metadata().await {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to read metadata for {:?}: {}", fp_path, e);
                        continue;
                    }
                };

                if !fp_metadata.is_dir() {
                    continue;
                }

                let fingerprint = match fp_path.file_name() {
                    Some(name) => name.to_string_lossy().to_string(),
                    None => continue,
                };

                // Try to load this tokenizer
                match self.load_tokenizer_from_path(&fp_path).await {
                    Ok(tokenizer) => {
                        let key = TokenizerKey::new(model_name.clone(), fingerprint.clone());
                        let entry = Arc::new(TokenizerEntry {
                            tokenizer: tokenizer.clone(),
                            path: fp_path.clone(),
                        });
                        self.tokenizers.insert(key, entry);
                        loaded_count += 1;
                        debug!("Pre-warmed tokenizer: {} ({})", model_name, fingerprint);
                    }
                    Err(e) => {
                        warn!("Failed to load tokenizer from {:?}: {}", fp_path, e);
                    }
                }
            }
        }

        if loaded_count > 0 {
            info!("Pre-warmed {} tokenizers from disk", loaded_count);
        }

        Ok(loaded_count)
    }
}

/// Statistics about the tokenizer registry
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Number of tokenizers currently cached in memory
    pub cached_tokenizers: usize,
    /// Number of downloads currently in progress
    pub pending_downloads: usize,
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::TempDir;

    use super::*;

    async fn mock_fetch_fn(_model_id: String, _fingerprint: String) -> Result<Vec<u8>> {
        use std::io::Cursor;

        use zip::write::{FileOptions, ZipWriter};

        // Create a mock tokenizer bundle
        let mut cursor = Cursor::new(Vec::new());
        let mut zip = ZipWriter::new(&mut cursor);

        let options =
            FileOptions::<()>::default().compression_method(zip::CompressionMethod::Stored);

        zip.start_file("tokenizer.json", options).unwrap();
        zip.write_all(
            br#"{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": null,
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"hello": 0, "world": 1},
                    "unk_token": "[UNK]"
                }
            }"#,
        )
        .unwrap();

        zip.finish().unwrap();
        Ok(cursor.into_inner())
    }

    fn compute_fingerprint(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    #[tokio::test]
    async fn test_get_or_fetch_success() {
        let temp_dir = TempDir::new().unwrap();
        let registry =
            TokenizerRegistry::new(temp_dir.path().to_path_buf(), CacheConfig::default());

        // First fetch should download
        let bundle = mock_fetch_fn("test-model".to_string(), "abc".to_string())
            .await
            .unwrap();
        let fingerprint = compute_fingerprint(&bundle);

        let result = registry
            .get_or_fetch("test-model", &fingerprint, mock_fetch_fn)
            .await;

        assert!(result.is_ok());
        assert_eq!(registry.len(), 1);

        // Second fetch should hit cache
        let result2 = registry
            .get_or_fetch("test-model", &fingerprint, mock_fetch_fn)
            .await;

        assert!(result2.is_ok());
        assert_eq!(registry.len(), 1); // Still only 1 entry
    }

    #[tokio::test]
    async fn test_concurrent_fetches() {
        let temp_dir = TempDir::new().unwrap();
        let registry = Arc::new(TokenizerRegistry::new(
            temp_dir.path().to_path_buf(),
            CacheConfig::default(),
        ));

        let bundle = mock_fetch_fn("test-model".to_string(), "abc".to_string())
            .await
            .unwrap();
        let fingerprint = compute_fingerprint(&bundle);

        // Spawn multiple concurrent fetches
        let mut handles = vec![];
        for _ in 0..5 {
            let registry_clone = registry.clone();
            let fp_clone = fingerprint.clone();
            let handle = tokio::spawn(async move {
                registry_clone
                    .get_or_fetch("test-model", &fp_clone, mock_fetch_fn)
                    .await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        // Should only have 1 entry (no duplicate downloads)
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn test_clear() {
        let temp_dir = TempDir::new().unwrap();
        let registry =
            TokenizerRegistry::new(temp_dir.path().to_path_buf(), CacheConfig::default());

        let bundle = mock_fetch_fn("test-model".to_string(), "abc".to_string())
            .await
            .unwrap();
        let fingerprint = compute_fingerprint(&bundle);

        registry
            .get_or_fetch("test-model", &fingerprint, mock_fetch_fn)
            .await
            .unwrap();

        assert_eq!(registry.len(), 1);

        registry.clear();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_stats() {
        let temp_dir = TempDir::new().unwrap();
        let registry =
            TokenizerRegistry::new(temp_dir.path().to_path_buf(), CacheConfig::default());

        let stats = registry.stats();
        assert_eq!(stats.cached_tokenizers, 0);
        assert_eq!(stats.pending_downloads, 0);

        let bundle = mock_fetch_fn("test-model".to_string(), "abc".to_string())
            .await
            .unwrap();
        let fingerprint = compute_fingerprint(&bundle);

        registry
            .get_or_fetch("test-model", &fingerprint, mock_fetch_fn)
            .await
            .unwrap();

        let stats = registry.stats();
        assert_eq!(stats.cached_tokenizers, 1);
    }
}
