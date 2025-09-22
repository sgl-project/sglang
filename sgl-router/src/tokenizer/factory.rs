use super::traits;
use anyhow::{Error, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use super::huggingface::HuggingFaceTokenizer;
use super::tiktoken::TiktokenTokenizer;
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
    buffer.len() >= 12
        && (buffer.starts_with(b"\x0a\x09")
            || buffer.starts_with(b"\x08\x00")
            || buffer.windows(4).any(|w| w == b"<unk")
            || buffer.windows(4).any(|w| w == b"<s>")
            || buffer.windows(4).any(|w| w == b"</s>"))
}

/// Factory function to create tokenizer from a model name or path (async version)
pub async fn create_tokenizer_async(
    model_name_or_path: &str,
) -> Result<Arc<dyn traits::Tokenizer>> {
    // Check if it's a file path
    let path = Path::new(model_name_or_path);
    if path.exists() {
        return create_tokenizer_from_file(model_name_or_path);
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
                create_tokenizer_from_file(tokenizer_path.to_str().unwrap())
            } else {
                // Try other common tokenizer file names
                let possible_files = ["tokenizer_config.json", "vocab.json"];
                for file_name in &possible_files {
                    let file_path = cache_dir.join(file_name);
                    if file_path.exists() {
                        return create_tokenizer_from_file(file_path.to_str().unwrap());
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
pub fn create_tokenizer(model_name_or_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    // Check if it's a file path
    let path = Path::new(model_name_or_path);
    if path.exists() {
        return create_tokenizer_from_file(model_name_or_path);
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
        tokio::task::block_in_place(|| handle.block_on(create_tokenizer_async(model_name_or_path)))
    } else {
        // No runtime, create a temporary one
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(create_tokenizer_async(model_name_or_path))
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
            use std::fs::File;
            use std::io::Read;

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
        assert_eq!(tokenizer.vocab_size(), 8); // Mock tokenizer has 8 tokens
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
        // Test creating tokenizer for GPT models
        let tokenizer = create_tokenizer("gpt-4").unwrap();
        assert!(tokenizer.vocab_size() > 0);

        // Test encoding and decoding
        let text = "Hello, world!";
        let encoding = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
        assert_eq!(decoded, text);
    }

    #[tokio::test]
    async fn test_download_tokenizer_from_hf() {
        // Test with a small model that should have tokenizer files
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
