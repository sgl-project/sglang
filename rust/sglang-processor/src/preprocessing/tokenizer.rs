//! Tokenizer loading and encoding over Dynamo tokenizers.

use std::path::{Path, PathBuf};

use crate::ProcessorError as Error;

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error>;

    fn encode_segments(
        &self,
        segments: &[dynamo_tokenizers::EncodeSegment<'_>],
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, Error> {
        let text = segments
            .iter()
            .map(|segment| segment.text)
            .collect::<String>();
        self.encode(&text, add_special_tokens)
    }
}

/// Load the tokenizer shared (Arc-backed) by the encode pool and detok shards.
/// `tokenizer_path` is a tokenizer file, a model dir, or an HF Hub repo id
/// (resolved from the local cache — no network).
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    add_special_tokens: bool,
) -> Result<dynamo_tokenizers::Tokenizer, String> {
    let path =
        tokenizer_path.ok_or_else(|| "no tokenizer configured: set tokenizer_path".to_string())?;
    let file = resolve_tokenizer_file(path, revision).ok_or_else(|| {
        format!(
            "no supported tokenizer file found for '{path}' (expected tokenizer.json, tiktoken.model, or *.tiktoken)"
        )
    })?;
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file_with_options(
        &file,
        dynamo_tokenizers::TokenizerOptions { add_special_tokens },
    )
    .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?;
    tracing::info!(%path, "loaded tokenizer");
    Ok(tokenizer)
}

/// Resolve the tokenizer source used by the renderer.
pub fn resolve_tokenizer_file(path: &str, revision: Option<&str>) -> Option<String> {
    let input = Path::new(path);
    if input.is_file() && is_supported_tokenizer_file(input) {
        return Some(input.to_string_lossy().into_owned());
    }
    let directory = model_directory(path, revision)?;
    discover_tokenizer_in_dir(&directory).map(|path| path.to_string_lossy().into_owned())
}

/// Resolve a dedicated Hugging Face chat-template file when the template is
/// not embedded in `tokenizer_config.json`.
pub fn resolve_chat_template_file(path: &str, revision: Option<&str>) -> Option<String> {
    let directory = model_directory(path, revision)?;
    discover_chat_template_in_dir(&directory).map(|path| path.to_string_lossy().into_owned())
}

fn model_directory(path: &str, revision: Option<&str>) -> Option<PathBuf> {
    let input = Path::new(path);
    if input.is_dir() {
        return Some(input.to_path_buf());
    }
    if input.is_file() {
        return input.parent().map(Path::to_path_buf);
    }
    let repo = cache_repo(path, revision);
    [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tiktoken.model",
    ]
    .into_iter()
    .find_map(|name| repo.get(name))
    .and_then(|file| file.parent().map(Path::to_path_buf))
}

fn discover_tokenizer_in_dir(directory: &Path) -> Option<PathBuf> {
    let tokenizer_config = directory.join("tokenizer_config.json");
    let prefers_tiktoken = std::fs::read_to_string(tokenizer_config)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|config| {
            config
                .get("tokenizer_class")
                .and_then(serde_json::Value::as_str)
                .map(|class| class.to_ascii_lowercase().contains("tiktoken"))
        })
        .unwrap_or(false);
    let hugging_face = directory.join("tokenizer.json");
    let tiktoken = directory.join("tiktoken.model");
    let discovered_tiktoken = || {
        sorted_directory_files(directory).find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".tiktoken"))
        })
    };
    if prefers_tiktoken {
        tiktoken
            .is_file()
            .then_some(tiktoken)
            .or_else(discovered_tiktoken)
            .or_else(|| hugging_face.is_file().then_some(hugging_face))
    } else {
        hugging_face
            .is_file()
            .then_some(hugging_face)
            .or_else(|| tiktoken.is_file().then_some(tiktoken))
            .or_else(discovered_tiktoken)
    }
}

fn discover_chat_template_in_dir(directory: &Path) -> Option<PathBuf> {
    for name in ["chat_template.json", "chat_template.jinja"] {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    sorted_directory_files(directory).find(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".jinja"))
    })
}

fn sorted_directory_files(directory: &Path) -> impl Iterator<Item = PathBuf> {
    let mut files = std::fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    files.sort();
    files.into_iter()
}

fn is_supported_tokenizer_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name == "tokenizer.json" || name == "tiktoken.model" || name.ends_with(".tiktoken")
        })
}

/// Resolve a model file from the tokenizer source: a dir → `dir/<file>`, a file →
/// its sibling, else an HF Hub repo id → the local cache. `None` if not found.
pub fn resolve_model_file(path: &str, revision: Option<&str>, filename: &str) -> Option<String> {
    let p = Path::new(path);
    if p.is_dir() {
        let f = p.join(filename);
        return f.is_file().then(|| f.to_string_lossy().into_owned());
    }
    if p.is_file() {
        // `path` is a file (e.g. `tokenizer.json`); look for the sibling.
        let f = p.parent()?.join(filename);
        return f.is_file().then(|| f.to_string_lossy().into_owned());
    }
    // Not a local path → HF Hub repo id (offline cache lookup).
    resolve_from_hub_cache(path, revision, filename)
}

/// Locate a file for an HF Hub repo id in the local cache. Offline —
/// the scheduler pre-downloads the model. `None` if not cached.
fn resolve_from_hub_cache(repo_id: &str, revision: Option<&str>, filename: &str) -> Option<String> {
    cache_repo(repo_id, revision)
        .get(filename)
        .map(|p| p.to_string_lossy().into_owned())
}

fn cache_repo(repo_id: &str, revision: Option<&str>) -> hf_hub::CacheRepo {
    use hf_hub::{Cache, Repo, RepoType};

    // Python resolves the cache dir as HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE >
    // HF_HOME/hub > ~/.cache/huggingface/hub; the hf-hub crate only knows
    // HF_HOME. Honor the explicit cache-dir overrides first, or the Rust
    // server misses models the Python scheduler already downloaded.
    let cache = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .map(|dir| Cache::new(dir.into()))
        .unwrap_or_else(Cache::from_env);
    cache.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        revision.unwrap_or("main").to_string(),
    ))
}

/// Real tokenizer over two already-loaded dynamo handles. Dynamo fixes
/// `add_special_tokens` when loading, so selecting the mode at request time
/// requires one handle for each setting.
pub struct DynamoTokenizer {
    without_specials: dynamo_tokenizers::Tokenizer,
    with_specials: dynamo_tokenizers::Tokenizer,
}

impl DynamoTokenizer {
    pub fn new(
        without_specials: dynamo_tokenizers::Tokenizer,
        with_specials: dynamo_tokenizers::Tokenizer,
    ) -> Self {
        Self {
            without_specials,
            with_specials,
        }
    }
}

impl TextTokenizer for DynamoTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error> {
        let encoding = if add_special_tokens {
            &self.with_specials
        } else {
            &self.without_specials
        }
        .encode(text)
        .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Vocab ids are non-negative and fit in i32.
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
    }

    fn encode_segments(
        &self,
        segments: &[dynamo_tokenizers::EncodeSegment<'_>],
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, Error> {
        let encoding = if add_special_tokens {
            &self.with_specials
        } else {
            &self.without_specials
        }
        .encode_segments(segments)
        .map_err(|error| Error::Tokenize(error.to_string()))?;
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_model_dir(label: &str) -> PathBuf {
        let sequence = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sglang-renderer-{label}-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn model_discovery_finds_tiktoken_and_dedicated_chat_template() {
        let directory = temp_model_dir("model-files");
        std::fs::write(
            directory.join("tokenizer_config.json"),
            r#"{"tokenizer_class":"KimiTikTokenTokenizer"}"#,
        )
        .unwrap();
        std::fs::write(directory.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(directory.join("tokenizer.tiktoken"), "token").unwrap();
        std::fs::write(directory.join("chat_template.jinja"), "{{ messages }}").unwrap();

        assert_eq!(
            resolve_tokenizer_file(directory.to_str().unwrap(), None),
            Some(
                directory
                    .join("tokenizer.tiktoken")
                    .to_string_lossy()
                    .into_owned()
            )
        );
        assert_eq!(
            resolve_chat_template_file(directory.to_str().unwrap(), None),
            Some(
                directory
                    .join("chat_template.jinja")
                    .to_string_lossy()
                    .into_owned()
            )
        );

        std::fs::remove_dir_all(directory).unwrap();
    }
}
