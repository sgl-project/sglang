use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

trait TokenizerBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String>;
}

struct HuggingFaceTokenizerBackend {
    inner: Tokenizer,
}

impl HuggingFaceTokenizerBackend {
    fn from_file(path: &Path) -> Result<Self, String> {
        Tokenizer::from_file(path)
            .map(|inner| Self { inner })
            .map_err(|e| format!("failed to load HuggingFace tokenizer: {}", e))
    }
}

impl TokenizerBackend for HuggingFaceTokenizerBackend {
    fn name(&self) -> &'static str {
        "huggingface-tokenizers"
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| format!("Detokenization failed: {}", e))
    }
}

/// Rust-native tokenizer wrapper with pluggable backends.
///
/// This mirrors Python's `get_tokenizer` shape: inspect the tokenizer path,
/// choose a backend, and fall back to Python for unsupported tokenizer families.
pub struct RustTokenizer {
    backend: Box<dyn TokenizerBackend>,
}

impl RustTokenizer {
    /// Load a native tokenizer from a tokenizer path or model directory.
    /// Returns `None` if no supported Rust backend is available.
    pub fn from_tokenizer_path(
        tokenizer_path: &str,
        tokenizer_mode: Option<&str>,
        context_len: i32,
    ) -> Option<Self> {
        let path = Path::new(tokenizer_path);
        let Some(tokenizer_json) = resolve_tokenizer_json(path) else {
            tracing::info!(
                "No native tokenizer candidates found at {:?}; Rust tokenizer disabled",
                path
            );
            return None;
        };

        if matches!(tokenizer_mode, Some("slow")) {
            tracing::info!(
                "Rust tokenizer disabled because tokenizer_mode=slow for {:?}",
                path
            );
            return None;
        }

        match load_backend(&tokenizer_json) {
            Ok(backend) => {
                let tokenizer = Self { backend };
                tracing::info!(
                    "Rust tokenizer loaded via {} from {:?} (context_len={})",
                    tokenizer.backend_name(),
                    tokenizer_json,
                    context_len
                );
                Some(tokenizer)
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load Rust tokenizer from {:?}: {}. Falling back to Python.",
                    tokenizer_json,
                    e
                );
                None
            }
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Tokenize text, returning token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        self.backend.encode(text, add_special_tokens)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.backend.decode(ids, skip_special_tokens)
    }
}

fn load_backend(tokenizer_json: &Path) -> Result<Box<dyn TokenizerBackend>, String> {
    // Add new native backend probes here. Unsupported formats should return an
    // error so callers can fall back to Python without changing the public API.
    HuggingFaceTokenizerBackend::from_file(tokenizer_json)
        .map(|backend| Box::new(backend) as Box<dyn TokenizerBackend>)
}

fn resolve_tokenizer_json(path: &Path) -> Option<PathBuf> {
    let candidate = if path.is_file() {
        path.to_path_buf()
    } else {
        path.join("tokenizer.json")
    };
    candidate.exists().then_some(candidate)
}
