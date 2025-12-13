use std::{fs::File, io::Read, path::Path, sync::Arc};

use anyhow::{Error, Result};
use tracing::{debug, info};

use super::{huggingface::HuggingFaceTokenizer, tiktoken::TiktokenTokenizer, traits};
use crate::tokenizer::hub::download_tokenizer_from_hf;

/// Represents the type of tokenizer being used
#[derive(Debug, Clone)]
pub enum TokenizerType {
    HuggingFace(String),
    Mock,
    Tiktoken(String),
    // Future: SentencePiece, GGUF
}

/// Create a tokenizer from a file path to a tokenizer file.
/// The file extension is used to determine the tokenizer type.
/// Supported file types are:
/// - json: HuggingFace tokenizer
/// - For testing: can return mock tokenizer
pub fn create_tokenizer_from_file(file_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    create_tokenizer_with_chat_template(file_path, None)
}

/// Create a tokenizer from a file path with an optional chat template
pub fn create_tokenizer_with_chat_template(
    file_path: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn traits::Tokenizer>> {
    // Special case for testing
    if file_path == "mock" || file_path == "test" {
        return Ok(Arc::new(super::mock::MockTokenizer::new()));
    }

    let path = Path::new(file_path);

    // Check if file exists
    if !path.exists() {
        return Err(Error::msg(format!("File not found: {}", file_path)));
    }

    // If path is a directory, search for tokenizer files
    if path.is_dir() {
        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.exists() {
            // Resolve chat template: provided path takes precedence over auto-discovery
            let final_chat_template =
                resolve_and_log_chat_template(chat_template_path, path, file_path);
            let tokenizer_path_str = tokenizer_json.to_str().ok_or_else(|| {
                Error::msg(format!(
                    "Tokenizer path is not valid UTF-8: {:?}",
                    tokenizer_json
                ))
            })?;
            return create_tokenizer_with_chat_template(
                tokenizer_path_str,
                final_chat_template.as_deref(),
            );
        }

        return Err(Error::msg(format!(
            "Directory '{}' does not contain a valid tokenizer file (tokenizer.json, tokenizer_config.json, or vocab.json)",
            file_path
        )));
    }

    // Try to determine tokenizer type from extension
    let extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|s| s.to_lowercase());

    let result = match extension.as_deref() {
        Some("json") => {
            let tokenizer =
                HuggingFaceTokenizer::from_file_with_chat_template(file_path, chat_template_path)?;

            Ok(Arc::new(tokenizer) as Arc<dyn traits::Tokenizer>)
        }
        Some("model") => {
            // SentencePiece model file
            Err(Error::msg("SentencePiece models not yet supported"))
        }
        Some("gguf") => {
            // GGUF format
            Err(Error::msg("GGUF format not yet supported"))
        }
        _ => {
            // Try to auto-detect by reading file content
            auto_detect_tokenizer(file_path)
        }
    };

    result
}

/// Auto-detect tokenizer type by examining file content
fn auto_detect_tokenizer(file_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    let mut file = File::open(file_path)?;
    let mut buffer = vec![0u8; 512]; // Read first 512 bytes for detection
    let bytes_read = file.read(&mut buffer)?;
    buffer.truncate(bytes_read);

    // Check for JSON (HuggingFace format)
    if is_likely_json(&buffer) {
        let tokenizer = HuggingFaceTokenizer::from_file(file_path)?;
        return Ok(Arc::new(tokenizer));
    }

    // Check for GGUF magic number
    if buffer.len() >= 4 && &buffer[0..4] == b"GGUF" {
        return Err(Error::msg("GGUF format detected but not yet supported"));
    }

    // Check for SentencePiece model
    if is_likely_sentencepiece(&buffer) {
        return Err(Error::msg(
            "SentencePiece model detected but not yet supported",
        ));
    }

    Err(Error::msg(format!(
        "Unable to determine tokenizer type for file: {}",
        file_path
    )))
}

/// Check if the buffer likely contains JSON data
fn is_likely_json(buffer: &[u8]) -> bool {
    // Skip UTF-8 BOM if present
    let content = if buffer.len() >= 3 && buffer[0..3] == [0xEF, 0xBB, 0xBF] {
        &buffer[3..]
    } else {
        buffer
    };

    // Find first non-whitespace character without allocation
    if let Some(first_byte) = content.iter().find(|&&b| !b.is_ascii_whitespace()) {
        *first_byte == b'{' || *first_byte == b'['
    } else {
        false
    }
}

/// Check if the buffer likely contains a SentencePiece model
fn is_likely_sentencepiece(buffer: &[u8]) -> bool {
    // SentencePiece models often start with specific patterns
    // This is a simplified check
    if buffer.len() < 12 {
        return false;
    }

    // Check header patterns first (cheap)
    if buffer.starts_with(b"\x0a\x09") || buffer.starts_with(b"\x08\x00") {
        return true;
    }

    // Single-pass scan for special token markers
    // Instead of multiple windows() calls, scan once looking for all patterns
    let patterns: &[&[u8]] = &[b"<unk", b"<s>", b"</s>"];
    for window in buffer.windows(4) {
        for pattern in patterns {
            if window.starts_with(pattern) {
                return true;
            }
        }
    }
    false
}

/// Helper function to discover chat template files in a directory
pub fn discover_chat_template_in_dir(dir: &Path) -> Option<String> {
    use std::fs;

    // Priority 1: Look for chat_template.json (contains Jinja in JSON format)
    let json_template_path = dir.join("chat_template.json");
    if json_template_path.exists() {
        return json_template_path.to_str().map(|s| s.to_string());
    }

    // Priority 2: Look for chat_template.jinja (standard Jinja file)
    let jinja_path = dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return jinja_path.to_str().map(|s| s.to_string());
    }

    // Priority 3: Look for any .jinja file (for models with non-standard naming)
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".jinja") && name != "chat_template.jinja" {
                    return entry.path().to_str().map(|s| s.to_string());
                }
            }
        }
    }

    None
}

/// Helper function to resolve and log chat template selection
///
/// Resolves the final chat template to use by prioritizing provided path over auto-discovery,
/// and logs the source for debugging purposes.
fn resolve_and_log_chat_template(
    provided_path: Option<&str>,
    discovery_dir: &Path,
    model_name: &str,
) -> Option<String> {
    let final_chat_template = provided_path
        .map(|s| s.to_string())
        .or_else(|| discover_chat_template_in_dir(discovery_dir));

    match (&provided_path, &final_chat_template) {
        (Some(provided), _) => {
            info!("Using provided chat template: {}", provided);
        }
        (None, Some(discovered)) => {
            info!(
                "Auto-discovered chat template in '{}': {}",
                discovery_dir.display(),
                discovered
            );
        }
        (None, None) => {
            debug!(
                "No chat template provided or discovered for model: {}",
                model_name
            );
        }
    }

    final_chat_template
}

/// Factory function to create tokenizer from a model name or path (async version)
pub async fn create_tokenizer_async(
    model_name_or_path: &str,
) -> Result<Arc<dyn traits::Tokenizer>> {
    create_tokenizer_async_with_chat_template(model_name_or_path, None).await
}

/// Factory function to create tokenizer with optional chat template (async version)
pub async fn create_tokenizer_async_with_chat_template(
    model_name_or_path: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn traits::Tokenizer>> {
    // Check if it's a file path
    let path = Path::new(model_name_or_path);
    if path.exists() {
        return create_tokenizer_with_chat_template(model_name_or_path, chat_template_path);
    }

    // Check if it's a GPT model name that should use Tiktoken
    if model_name_or_path.contains("gpt-")
        || model_name_or_path.contains("davinci")
        || model_name_or_path.contains("curie")
        || model_name_or_path.contains("babbage")
        || model_name_or_path.contains("ada")
    {
        let tokenizer = TiktokenTokenizer::from_model_name(model_name_or_path)?;
        return Ok(Arc::new(tokenizer));
    }

    // Try to download tokenizer files from HuggingFace
    match download_tokenizer_from_hf(model_name_or_path).await {
        Ok(cache_dir) => {
            // Look for tokenizer.json in the cache directory
            let tokenizer_path = cache_dir.join("tokenizer.json");
            if tokenizer_path.exists() {
                // Resolve chat template: provided path takes precedence over auto-discovery
                let final_chat_template = resolve_and_log_chat_template(
                    chat_template_path,
                    &cache_dir,
                    model_name_or_path,
                );

                let tokenizer_path_str = tokenizer_path.to_str().ok_or_else(|| {
                    Error::msg(format!(
                        "Tokenizer path is not valid UTF-8: {:?}",
                        tokenizer_path
                    ))
                })?;
                create_tokenizer_with_chat_template(
                    tokenizer_path_str,
                    final_chat_template.as_deref(),
                )
            } else {
                // Try other common tokenizer file names
                let possible_files = ["tokenizer_config.json", "vocab.json"];
                for file_name in &possible_files {
                    let file_path = cache_dir.join(file_name);
                    if file_path.exists() {
                        // Resolve chat template: provided path takes precedence over auto-discovery
                        let final_chat_template = resolve_and_log_chat_template(
                            chat_template_path,
                            &cache_dir,
                            model_name_or_path,
                        );

                        let file_path_str = file_path.to_str().ok_or_else(|| {
                            Error::msg(format!("File path is not valid UTF-8: {:?}", file_path))
                        })?;
                        return create_tokenizer_with_chat_template(
                            file_path_str,
                            final_chat_template.as_deref(),
                        );
                    }
                }
                Err(Error::msg(format!(
                    "Downloaded model '{}' but couldn't find a suitable tokenizer file",
                    model_name_or_path
                )))
            }
        }
        Err(e) => Err(Error::msg(format!(
            "Failed to download tokenizer from HuggingFace: {}",
            e
        ))),
    }
}

/// Factory function to create tokenizer from a model name or path (blocking version)
///
/// This delegates to `create_tokenizer_with_chat_template_blocking` with no chat template,
/// which handles both local files and HuggingFace Hub downloads uniformly.
pub fn create_tokenizer(model_name_or_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    create_tokenizer_with_chat_template_blocking(model_name_or_path, None)
}

/// Factory function to create tokenizer with optional chat template (blocking version)
pub fn create_tokenizer_with_chat_template_blocking(
    model_name_or_path: &str,
    chat_template_path: Option<&str>,
) -> Result<Arc<dyn traits::Tokenizer>> {
    // Check if it's a file path
    let path = Path::new(model_name_or_path);
    if path.exists() {
        return create_tokenizer_with_chat_template(model_name_or_path, chat_template_path);
    }

    // Check if it's a GPT model name that should use Tiktoken
    if model_name_or_path.contains("gpt-")
        || model_name_or_path.contains("davinci")
        || model_name_or_path.contains("curie")
        || model_name_or_path.contains("babbage")
        || model_name_or_path.contains("ada")
    {
        let tokenizer = TiktokenTokenizer::from_model_name(model_name_or_path)?;
        return Ok(Arc::new(tokenizer));
    }

    // Only use tokio for HuggingFace downloads
    // Check if we're already in a tokio runtime
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        // We're in a runtime, use block_in_place
        tokio::task::block_in_place(|| {
            handle.block_on(create_tokenizer_async_with_chat_template(
                model_name_or_path,
                chat_template_path,
            ))
        })
    } else {
        // No runtime, create a temporary one
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(create_tokenizer_async_with_chat_template(
            model_name_or_path,
            chat_template_path,
        ))
    }
}

/// Get information about a tokenizer file
pub fn get_tokenizer_info(file_path: &str) -> Result<TokenizerType> {
    let path = Path::new(file_path);

    if !path.exists() {
        return Err(Error::msg(format!("File not found: {}", file_path)));
    }

    let extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|s| s.to_lowercase());

    match extension.as_deref() {
        Some("json") => Ok(TokenizerType::HuggingFace(file_path.to_string())),
        _ => {
            // Try auto-detection
            use std::{fs::File, io::Read};

            let mut file = File::open(file_path)?;
            let mut buffer = vec![0u8; 512];
            let bytes_read = file.read(&mut buffer)?;
            buffer.truncate(bytes_read);

            if is_likely_json(&buffer) {
                Ok(TokenizerType::HuggingFace(file_path.to_string()))
            } else {
                Err(Error::msg("Unknown tokenizer type"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_detection() {
        assert!(is_likely_json(b"{\"test\": \"value\"}"));
        assert!(is_likely_json(b"  \n\t{\"test\": \"value\"}"));
        assert!(is_likely_json(b"[1, 2, 3]"));
        assert!(!is_likely_json(b"not json"));
        assert!(!is_likely_json(b""));
    }

    #[test]
    fn test_mock_tokenizer_creation() {
        let tokenizer = create_tokenizer_from_file("mock").unwrap();
        assert_eq!(tokenizer.vocab_size(), 14); // Mock tokenizer has 14 tokens
    }

    #[test]
    fn test_file_not_found() {
        let result = create_tokenizer_from_file("/nonexistent/file.json");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("File not found"));
        }
    }

    #[test]
    fn test_create_tiktoken_tokenizer() {
        let tokenizer = create_tokenizer("gpt-4").unwrap();
        assert!(tokenizer.vocab_size() > 0);

        let text = "Hello, world!";
        let encoding = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
        assert_eq!(decoded, text);
    }

    #[tokio::test]
    async fn test_download_tokenizer_from_hf() {
        // Skip this test if HF_TOKEN is not set and we're in CI
        if std::env::var("CI").is_ok() && std::env::var("HF_TOKEN").is_err() {
            println!("Skipping HF download test in CI without HF_TOKEN");
            return;
        }

        // Try to create tokenizer for a known small model
        let result = create_tokenizer_async("bert-base-uncased").await;

        // The test might fail due to network issues or rate limiting
        // so we just check that the function executes without panic
        match result {
            Ok(tokenizer) => {
                assert!(tokenizer.vocab_size() > 0);
                println!("Successfully downloaded and created tokenizer");
            }
            Err(e) => {
                println!("Download failed (this might be expected): {}", e);
                // Don't fail the test - network issues shouldn't break CI
            }
        }
    }
}
