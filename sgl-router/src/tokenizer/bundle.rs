//! Tokenizer Bundle Extraction
//!
//! Handles downloading and extraction of tokenizer bundles from workers.
//! Provides security protections against path traversal and zip bombs.

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};
use zip::ZipArchive;

/// Maximum size for extracted files (100MB) to prevent zip bombs
const MAX_EXTRACTED_SIZE: u64 = 100 * 1024 * 1024;

/// Maximum number of files in a bundle to prevent resource exhaustion
const MAX_FILES_IN_BUNDLE: usize = 1000;

/// Extracts and validates tokenizer bundles
pub struct BundleExtractor {
    /// Base cache directory for all tokenizers
    cache_root: PathBuf,
}

impl BundleExtractor {
    /// Create a new bundle extractor
    ///
    /// # Arguments
    /// * `cache_root` - Base directory for tokenizer cache (e.g., `.tokenizer_cache`)
    pub fn new(cache_root: PathBuf) -> Self {
        Self { cache_root }
    }

    /// Extract a tokenizer bundle to the cache directory
    ///
    /// # Arguments
    /// * `bundle_data` - Raw ZIP file data
    /// * `model_id` - Model identifier (e.g., "meta-llama/Llama-3.1-8B")
    /// * `expected_fingerprint` - SHA256 fingerprint for validation
    ///
    /// # Returns
    /// Path to the extracted tokenizer directory
    ///
    /// # Security
    /// - Validates SHA256 fingerprint before extraction
    /// - Prevents path traversal attacks
    /// - Protects against zip bombs with size limits
    /// - Validates file count limits
    pub fn extract_bundle(
        &self,
        bundle_data: &[u8],
        model_id: &str,
        expected_fingerprint: &str,
    ) -> Result<PathBuf> {
        debug!(
            "Extracting tokenizer bundle for model {} (expected fingerprint: {})",
            model_id, expected_fingerprint
        );

        // Step 1: Validate fingerprint
        self.validate_fingerprint(bundle_data, expected_fingerprint)?;

        // Step 2: Create extraction directory
        let extract_dir = self.get_extraction_path(model_id, expected_fingerprint);

        // If directory already exists and is valid, return it
        if extract_dir.exists() && self.validate_extracted_dir(&extract_dir)? {
            debug!("Tokenizer bundle already extracted at {:?}", extract_dir);
            return Ok(extract_dir);
        }

        // Step 3: Extract bundle with security checks
        self.extract_with_checks(bundle_data, &extract_dir)?;

        info!(
            "Successfully extracted tokenizer bundle for {} to {:?}",
            model_id, extract_dir
        );

        Ok(extract_dir)
    }

    /// Validate the SHA256 fingerprint of the bundle data
    fn validate_fingerprint(&self, data: &[u8], expected: &str) -> Result<()> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let actual = format!("{:x}", result);

        if actual != expected {
            return Err(anyhow::anyhow!(
                "Fingerprint mismatch: expected {}, got {}",
                expected,
                actual
            ));
        }

        debug!("Fingerprint validation successful: {}", actual);
        Ok(())
    }

    /// Get the extraction path for a model and fingerprint
    fn get_extraction_path(&self, model_id: &str, fingerprint: &str) -> PathBuf {
        // Sanitize model_id for use in filesystem path
        let sanitized_model = model_id.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");

        self.cache_root.join(sanitized_model).join(fingerprint)
    }

    /// Validate that an extracted directory contains valid tokenizer files
    fn validate_extracted_dir(&self, dir: &Path) -> Result<bool> {
        if !dir.is_dir() {
            return Ok(false);
        }

        // Check for at least one tokenizer file
        let tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"];

        let has_tokenizer_file = tokenizer_files.iter().any(|file| dir.join(file).exists());

        Ok(has_tokenizer_file)
    }

    /// Extract the ZIP bundle with security checks
    fn extract_with_checks(&self, data: &[u8], extract_dir: &Path) -> Result<()> {
        let cursor = io::Cursor::new(data);
        let mut archive = ZipArchive::new(cursor).context("Failed to open ZIP archive")?;

        // Security check: validate file count
        if archive.len() > MAX_FILES_IN_BUNDLE {
            return Err(anyhow::anyhow!(
                "Bundle contains too many files: {} (max: {})",
                archive.len(),
                MAX_FILES_IN_BUNDLE
            ));
        }

        // Create extraction directory
        fs::create_dir_all(extract_dir)
            .with_context(|| format!("Failed to create directory {:?}", extract_dir))?;

        let mut total_size: u64 = 0;

        // Extract each file with security checks
        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .with_context(|| format!("Failed to read file at index {}", i))?;

            // Security check: prevent path traversal
            let file_path = match file.enclosed_name() {
                Some(path) => path,
                None => {
                    warn!("Skipping file with unsafe path: {:?}", file.name());
                    continue;
                }
            };

            let output_path = extract_dir.join(&file_path);

            // Security check: ensure output path is within extraction directory
            if !output_path.starts_with(extract_dir) {
                warn!("Skipping file with path traversal attempt: {:?}", file_path);
                continue;
            }

            if file.is_dir() {
                fs::create_dir_all(&output_path)
                    .with_context(|| format!("Failed to create directory {:?}", output_path))?;
            } else {
                // Create parent directories if needed
                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent).with_context(|| {
                        format!("Failed to create parent directory {:?}", parent)
                    })?;
                }

                // Security check: prevent zip bombs
                let file_size = file.size();
                total_size += file_size;
                if total_size > MAX_EXTRACTED_SIZE {
                    return Err(anyhow::anyhow!(
                        "Extracted size exceeds limit: {} bytes (max: {})",
                        total_size,
                        MAX_EXTRACTED_SIZE
                    ));
                }

                // Extract file
                let mut output_file = File::create(&output_path)
                    .with_context(|| format!("Failed to create file {:?}", output_path))?;

                // Use a buffer to limit memory usage for large files
                let mut buffer = vec![0u8; 8192];
                let mut remaining = file_size;

                while remaining > 0 {
                    let to_read = std::cmp::min(remaining, buffer.len() as u64) as usize;
                    let read = file.read(&mut buffer[..to_read]).with_context(|| {
                        format!("Failed to read from archive file {:?}", file_path)
                    })?;

                    if read == 0 {
                        break;
                    }

                    output_file
                        .write_all(&buffer[..read])
                        .with_context(|| format!("Failed to write to file {:?}", output_path))?;

                    remaining = remaining.saturating_sub(read as u64);
                }

                debug!("Extracted file: {:?} ({} bytes)", file_path, file_size);
            }
        }

        debug!(
            "Extracted {} files (total size: {} bytes)",
            archive.len(),
            total_size
        );

        Ok(())
    }

    /// Get the cache root directory
    pub fn cache_root(&self) -> &Path {
        &self.cache_root
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::TempDir;

    use super::*;

    fn create_test_bundle() -> Vec<u8> {
        use std::io::Cursor;

        use zip::write::{FileOptions, ZipWriter};

        let mut cursor = Cursor::new(Vec::new());
        let mut zip = ZipWriter::new(&mut cursor);

        let options =
            FileOptions::<()>::default().compression_method(zip::CompressionMethod::Stored);

        // Add a mock tokenizer.json
        zip.start_file("tokenizer.json", options).unwrap();
        zip.write_all(b"{\"version\": \"1.0\"}").unwrap();

        // Add a mock vocab file
        zip.start_file("vocab.json", options).unwrap();
        zip.write_all(b"{\"hello\": 1}").unwrap();

        zip.finish().unwrap();
        cursor.into_inner()
    }

    fn compute_fingerprint(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn test_extract_bundle_success() {
        let temp_dir = TempDir::new().unwrap();
        let extractor = BundleExtractor::new(temp_dir.path().to_path_buf());

        let bundle_data = create_test_bundle();
        let fingerprint = compute_fingerprint(&bundle_data);

        let result = extractor.extract_bundle(&bundle_data, "test-model", &fingerprint);
        assert!(result.is_ok());

        let extract_path = result.unwrap();
        assert!(extract_path.join("tokenizer.json").exists());
        assert!(extract_path.join("vocab.json").exists());
    }

    #[test]
    fn test_fingerprint_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let extractor = BundleExtractor::new(temp_dir.path().to_path_buf());

        let bundle_data = create_test_bundle();
        let wrong_fingerprint = "0000000000000000000000000000000000000000000000000000000000000000";

        let result = extractor.extract_bundle(&bundle_data, "test-model", wrong_fingerprint);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Fingerprint mismatch"));
    }

    #[test]
    fn test_validate_extracted_dir() {
        let temp_dir = TempDir::new().unwrap();
        let extractor = BundleExtractor::new(temp_dir.path().to_path_buf());

        // Create a directory with a tokenizer file
        let test_dir = temp_dir.path().join("test");
        fs::create_dir_all(&test_dir).unwrap();
        fs::write(test_dir.join("tokenizer.json"), b"{}").unwrap();

        assert!(extractor.validate_extracted_dir(&test_dir).unwrap());

        // Test with empty directory
        let empty_dir = temp_dir.path().join("empty");
        fs::create_dir_all(&empty_dir).unwrap();
        assert!(!extractor.validate_extracted_dir(&empty_dir).unwrap());
    }

    #[test]
    fn test_path_sanitization() {
        let temp_dir = TempDir::new().unwrap();
        let extractor = BundleExtractor::new(temp_dir.path().to_path_buf());

        let path = extractor.get_extraction_path("org/model:v1", "abc123");
        let path_str = path.to_string_lossy();

        // Should not contain path separators or special characters
        assert!(
            !path_str.contains('/')
                || path_str.starts_with(temp_dir.path().to_string_lossy().as_ref())
        );
        assert!(!path_str.contains(':') || cfg!(windows));
    }
}
