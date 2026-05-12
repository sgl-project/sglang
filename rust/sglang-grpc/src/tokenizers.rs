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

enum TokenizerCandidate {
    HuggingFaceJson(PathBuf),
    TiktokenJson(PathBuf),
}

/// Rust-native tokenizer wrapper with pluggable backends.
///
/// This mirrors Python's `get_tokenizer` shape: inspect the tokenizer path,
/// choose a backend, and fall back to Python for unsupported tokenizer families.
pub struct RustTokenizer {
    backend: Box<dyn TokenizerBackend>,
    context_len: i32,
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
        let candidates = tokenizer_candidates(path);
        if candidates.is_empty() {
            tracing::info!(
                "No native tokenizer candidates found at {:?}; Rust tokenizer disabled",
                path
            );
            return None;
        }

        if matches!(tokenizer_mode, Some("slow")) {
            tracing::info!(
                "Rust tokenizer disabled because tokenizer_mode=slow for {:?}",
                path
            );
            return None;
        }

        for candidate in candidates {
            match candidate {
                TokenizerCandidate::HuggingFaceJson(tokenizer_json) => {
                    match HuggingFaceTokenizerBackend::from_file(&tokenizer_json) {
                        Ok(backend) => {
                            let tokenizer = Self {
                                backend: Box::new(backend),
                                context_len,
                            };
                            tracing::info!(
                                "Rust tokenizer loaded via {} from {:?} (context_len={})",
                                tokenizer.backend_name(),
                                tokenizer_json,
                                context_len
                            );
                            return Some(tokenizer);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to load Rust tokenizer from {:?}: {}. Falling back to Python.",
                                tokenizer_json,
                                e
                            );
                        }
                    }
                }
                TokenizerCandidate::TiktokenJson(tokenizer_json) => {
                    tracing::info!(
                        "Tokenizer at {:?} uses SGLang tiktoken JSON format; native Rust backend is not implemented, falling back to Python",
                        tokenizer_json
                    );
                }
            }
        }

        None
    }

    /// Backwards-compatible entry point for callers that only have a model path.
    pub fn from_model_path(model_path: &str, context_len: i32) -> Option<Self> {
        Self::from_tokenizer_path(model_path, None, context_len)
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

    /// Return the model's context length.
    pub fn context_len(&self) -> i32 {
        self.context_len
    }
}

fn tokenizer_candidates(path: &Path) -> Vec<TokenizerCandidate> {
    if path.is_file() {
        return tokenizer_candidate_from_json(path).into_iter().collect();
    }

    let tokenizer_json = path.join("tokenizer.json");
    tokenizer_candidate_from_json(&tokenizer_json)
        .into_iter()
        .collect()
}

fn tokenizer_candidate_from_json(path: &Path) -> Option<TokenizerCandidate> {
    if !path.exists() {
        return None;
    }

    if is_tiktoken_json(path) {
        Some(TokenizerCandidate::TiktokenJson(path.to_path_buf()))
    } else {
        Some(TokenizerCandidate::HuggingFaceJson(path.to_path_buf()))
    }
}

fn is_tiktoken_json(path: &Path) -> bool {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|contents| serde_json::from_str::<serde_json::Value>(&contents).ok())
        .and_then(|value| {
            Some(value.get("regular_tokens")?.is_array() && value.get("special_tokens")?.is_array())
        })
        .unwrap_or(false)
}
