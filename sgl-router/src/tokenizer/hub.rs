use std::{
    env,
    path::{Path, PathBuf},
};

use hf_hub::api::tokio::ApiBuilder;

const IGNORED: [&str; 5] = [
    ".gitattributes",
    "LICENSE",
    "LICENSE.txt",
    "README.md",
    "USE_POLICY.md",
];

const HF_TOKEN_ENV_VAR: &str = "HF_TOKEN";

/// Checks if a file is a model weight file
fn is_weight_file(filename: &str) -> bool {
    filename.ends_with(".bin")
        || filename.ends_with(".safetensors")
        || filename.ends_with(".h5")
        || filename.ends_with(".msgpack")
        || filename.ends_with(".ckpt.index")
}

/// Checks if a file is an image file
fn is_image(filename: &str) -> bool {
    filename.ends_with(".png")
        || filename.ends_with("PNG")
        || filename.ends_with(".jpg")
        || filename.ends_with("JPG")
        || filename.ends_with(".jpeg")
        || filename.ends_with("JPEG")
}

/// Checks if a file is a tokenizer file
fn is_tokenizer_file(filename: &str) -> bool {
    filename.ends_with("tokenizer.json")
        || filename.ends_with("tokenizer_config.json")
        || filename.ends_with("special_tokens_map.json")
        || filename.ends_with("vocab.json")
        || filename.ends_with("merges.txt")
        || filename.ends_with(".model")  // SentencePiece models
        || filename.ends_with(".tiktoken")
        || is_chat_template_file(filename) // Include chat template files
}

/// Checks if a file is a chat template file
fn is_chat_template_file(filename: &str) -> bool {
    filename.ends_with(".jinja")  // Direct Jinja files
        || filename == "chat_template.json" // JSON file containing Jinja template
}

/// Attempt to download tokenizer files from Hugging Face
/// Returns the directory containing the downloaded tokenizer files
pub async fn download_tokenizer_from_hf(model_id: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
    let model_id = model_id.as_ref();
    let token = env::var(HF_TOKEN_ENV_VAR).ok();
    let api = ApiBuilder::new()
        .with_progress(true)
        .with_token(token)
        .build()?;
    let model_name = model_id.display().to_string();

    let repo = api.model(model_name.clone());

    let info = match repo.info().await {
        Ok(info) => info,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to fetch model '{}' from HuggingFace: {}. Is this a valid HuggingFace ID?",
                model_name,
                e
            ));
        }
    };

    if info.siblings.is_empty() {
        return Err(anyhow::anyhow!(
            "Model '{}' exists but contains no downloadable files.",
            model_name
        ));
    }

    let mut cache_dir = None;
    let mut tokenizer_files_found = false;

    // First, identify all tokenizer files to download
    let tokenizer_files: Vec<_> = info
        .siblings
        .iter()
        .filter(|sib| {
            !IGNORED.contains(&sib.rfilename.as_str())
                && !is_image(&sib.rfilename)
                && !is_weight_file(&sib.rfilename)
                && is_tokenizer_file(&sib.rfilename)
        })
        .collect();

    if tokenizer_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No tokenizer files found for model '{}'.",
            model_name
        ));
    }

    // Download all tokenizer files
    for sib in tokenizer_files {
        match repo.get(&sib.rfilename).await {
            Ok(path) => {
                if cache_dir.is_none() {
                    cache_dir = path.parent().map(|p| p.to_path_buf());
                }
                tokenizer_files_found = true;
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to download tokenizer file '{}' from model '{}': {}",
                    sib.rfilename,
                    model_name,
                    e
                ));
            }
        }
    }

    if !tokenizer_files_found {
        return Err(anyhow::anyhow!(
            "No tokenizer files could be downloaded for model '{}'.",
            model_name
        ));
    }

    match cache_dir {
        Some(dir) => {
            // Ensure we return the correct model directory, not a subfolder
            // Some models have an "original" subfolder for PyTorch weights
            // We want the main model directory that contains tokenizer files
            let final_dir = resolve_model_cache_dir(&dir, &model_name);
            Ok(final_dir)
        }
        None => Err(anyhow::anyhow!(
            "Invalid HF cache path for model '{}'",
            model_name
        )),
    }
}

/// Attempt to download a model from Hugging Face (including weights)
/// Returns the directory it is in
/// If ignore_weights is true, model weight files will be skipped
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let token = env::var(HF_TOKEN_ENV_VAR).ok();
    let api = ApiBuilder::new()
        .with_progress(true)
        .with_token(token)
        .build()?;
    let model_name = name.display().to_string();

    let repo = api.model(model_name.clone());

    let info = match repo.info().await {
        Ok(info) => info,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to fetch model '{}' from HuggingFace: {}. Is this a valid HuggingFace ID?",
                model_name,
                e
            ));
        }
    };

    if info.siblings.is_empty() {
        return Err(anyhow::anyhow!(
            "Model '{}' exists but contains no downloadable files.",
            model_name
        ));
    }

    let mut p = PathBuf::new();
    let mut files_downloaded = false;

    for sib in info.siblings {
        if IGNORED.contains(&sib.rfilename.as_str()) || is_image(&sib.rfilename) {
            continue;
        }

        // If ignore_weights is true, skip weight files
        if ignore_weights && is_weight_file(&sib.rfilename) {
            continue;
        }

        match repo.get(&sib.rfilename).await {
            Ok(path) => {
                p = path;
                files_downloaded = true;
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to download file '{}' from model '{}': {}",
                    sib.rfilename,
                    model_name,
                    e
                ));
            }
        }
    }

    if !files_downloaded {
        let file_type = if ignore_weights {
            "non-weight"
        } else {
            "valid"
        };
        return Err(anyhow::anyhow!(
            "No {} files found for model '{}'.",
            file_type,
            model_name
        ));
    }

    match p.parent() {
        Some(p) => {
            let final_dir = resolve_model_cache_dir(p, &model_name);
            Ok(final_dir)
        }
        None => Err(anyhow::anyhow!("Invalid HF cache path: {}", p.display())),
    }
}

/// Resolve the correct model cache directory
/// Handles cases where files might be in subfolders (e.g., "original" folder)
fn resolve_model_cache_dir(path: &Path, model_name: &str) -> PathBuf {
    // Check if we're in a subfolder like "original"
    if let Some(parent) = path.parent() {
        if let Some(folder_name) = path.file_name() {
            if folder_name == "original" {
                // We're in the "original" subfolder, go up one level
                return parent.to_path_buf();
            }
        }
    }

    // Check if the current path contains the model name components
    // This helps ensure we're at the right directory level
    let model_parts: Vec<&str> = model_name.split('/').collect();
    if model_parts.len() >= 2 {
        let expected_pattern = format!(
            "models--{}--{}",
            model_parts[0].replace("-", "--"),
            model_parts[1].replace("-", "--")
        );

        if path.to_string_lossy().contains(&expected_pattern) {
            // We're already at the correct level
            return path.to_path_buf();
        }

        let mut current = path.to_path_buf();

        // First check if current path already contains tokenizer files
        if current.join("tokenizer.json").exists() || current.join("tokenizer_config.json").exists()
        {
            return current;
        }

        // If not, traverse up to find the model root, then look in snapshots
        while let Some(parent) = current.parent() {
            if parent.to_string_lossy().contains(&expected_pattern) {
                let snapshots_dir = parent.join("snapshots");
                if snapshots_dir.exists() && snapshots_dir.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let snapshot_path = entry.path();
                            if snapshot_path.is_dir()
                                && (snapshot_path.join("tokenizer.json").exists()
                                    || snapshot_path.join("tokenizer_config.json").exists())
                            {
                                return snapshot_path;
                            }
                        }
                    }
                }
                return parent.to_path_buf();
            }
            current = parent.to_path_buf();
        }
    }

    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_tokenizer_file() {
        assert!(is_tokenizer_file("tokenizer.json"));
        assert!(is_tokenizer_file("tokenizer_config.json"));
        assert!(is_tokenizer_file("special_tokens_map.json"));
        assert!(is_tokenizer_file("vocab.json"));
        assert!(is_tokenizer_file("merges.txt"));
        assert!(is_tokenizer_file("spiece.model"));
        assert!(is_tokenizer_file("chat_template.jinja"));
        assert!(is_tokenizer_file("template.jinja"));
        assert!(!is_tokenizer_file("model.bin"));
        assert!(!is_tokenizer_file("README.md"));
    }

    #[test]
    fn test_is_chat_template_file() {
        assert!(is_chat_template_file("chat_template.jinja"));
        assert!(is_chat_template_file("template.jinja"));
        assert!(is_chat_template_file("any_file.jinja"));
        assert!(is_chat_template_file("chat_template.json"));
        assert!(!is_chat_template_file("tokenizer.json"));
        assert!(!is_chat_template_file("other_file.json"));
        assert!(!is_chat_template_file("chat_template"));
        assert!(!is_chat_template_file("README.md"));
    }

    #[test]
    fn test_is_weight_file() {
        assert!(is_weight_file("model.bin"));
        assert!(is_weight_file("model.safetensors"));
        assert!(is_weight_file("pytorch_model.bin"));
        assert!(!is_weight_file("tokenizer.json"));
        assert!(!is_weight_file("config.json"));
    }
}
