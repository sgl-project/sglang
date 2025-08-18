use super::{traits, TokenizerTrait};
use crate::metrics::TokenizerMetrics;
use anyhow::{Error, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "huggingface")]
use super::huggingface::HuggingFaceTokenizer;

/// Represents the type of tokenizer being used
#[derive(Debug, Clone)]
pub enum TokenizerType {
    HuggingFace(String),
    Mock,
    // Future: SentencePiece, GGUF, Tiktoken
}

/// Create a tokenizer from a file path to a tokenizer file.
/// The file extension is used to determine the tokenizer type.
/// Supported file types are:
/// - json: HuggingFace tokenizer
/// - For testing: can return mock tokenizer
pub fn create_tokenizer_from_file(file_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    let start_time = Instant::now();

    // Special case for testing
    if file_path == "mock" || file_path == "test" {
        return Ok(Arc::new(super::mock::MockTokenizer::new()));
    }

    let path = Path::new(file_path);

    // Check if file exists
    if !path.exists() {
        TokenizerMetrics::record_factory_error("file_not_found");
        return Err(Error::msg(format!("File not found: {}", file_path)));
    }

    // Try to determine tokenizer type from extension
    let extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|s| s.to_lowercase());

    let result = match extension.as_deref() {
        Some("json") => {
            #[cfg(feature = "huggingface")]
            {
                let tokenizer = HuggingFaceTokenizer::from_file(file_path)?;

                TokenizerMetrics::record_factory_load("json");
                TokenizerMetrics::set_vocab_size("huggingface", tokenizer.vocab_size());

                Ok(Arc::new(tokenizer) as Arc<dyn traits::Tokenizer>)
            }
            #[cfg(not(feature = "huggingface"))]
            {
                TokenizerMetrics::record_factory_error("huggingface_disabled");
                Err(Error::msg(
                    "HuggingFace support not enabled. Enable the 'huggingface' feature.",
                ))
            }
        }
        Some("model") => {
            // SentencePiece model file
            TokenizerMetrics::record_factory_error("unsupported_sentencepiece");
            Err(Error::msg("SentencePiece models not yet supported"))
        }
        Some("gguf") => {
            // GGUF format
            TokenizerMetrics::record_factory_error("unsupported_gguf");
            Err(Error::msg("GGUF format not yet supported"))
        }
        _ => {
            // Try to auto-detect by reading file content
            auto_detect_tokenizer(file_path).inspect(|tokenizer| {
                TokenizerMetrics::record_factory_load("auto_detected");
                TokenizerMetrics::set_vocab_size("auto_detected", tokenizer.vocab_size());
            })
        }
    };

    if result.is_ok() {
        TokenizerMetrics::record_factory_load_duration(start_time.elapsed());
    }
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
        #[cfg(feature = "huggingface")]
        {
            let tokenizer = HuggingFaceTokenizer::from_file(file_path)?;
            return Ok(Arc::new(tokenizer));
        }
        #[cfg(not(feature = "huggingface"))]
        {
            return Err(Error::msg(
                "File appears to be JSON (HuggingFace) format, but HuggingFace support is not enabled",
            ));
        }
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

/// Factory function to create tokenizer from a model name or path
pub fn create_tokenizer(model_name_or_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    // Check if it's a file path
    let path = Path::new(model_name_or_path);
    if path.exists() {
        return create_tokenizer_from_file(model_name_or_path);
    }

    // Otherwise, try to load from HuggingFace Hub
    #[cfg(feature = "huggingface")]
    {
        // This would download from HF Hub - not implemented yet
        Err(Error::msg(
            "Loading from HuggingFace Hub not yet implemented",
        ))
    }

    #[cfg(not(feature = "huggingface"))]
    {
        Err(Error::msg(format!(
            "Model '{}' not found locally and HuggingFace support is not enabled",
            model_name_or_path
        )))
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
}
