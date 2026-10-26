//! Tokenizer and corpus files: the local HF cache first, then the Hub.
//!
//! The cache lookup mirrors `rust/sglang-server`'s `resolve_model_file`, so a
//! model the Python scheduler already downloaded is found without network
//! access. Only a cache miss reaches the Hub, and the download is cached for
//! the next run.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

pub const SHAREGPT_REPO: &str = "anon8231489123/ShareGPT_Vicuna_unfiltered";
pub const SHAREGPT_FILE: &str = "ShareGPT_V3_unfiltered_cleaned_split.json";

/// Locate a `tokenizer.json`, given a file, a directory holding one, or an HF
/// repo id.
pub async fn resolve_tokenizer(http: &reqwest::Client, source: &str) -> Result<PathBuf> {
    resolve_file(http, source, "tokenizer.json", RepoKind::Model)
        .await
        .with_context(|| format!("cannot find a tokenizer for {source}"))
}

/// Load a tokenizer from an already-resolved path.
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))
        .context("tokenizer.json failed to load")
}

/// The ShareGPT conversations file, from `--dataset-path` or the Hub.
pub async fn resolve_sharegpt(http: &reqwest::Client, dataset_path: &str) -> Result<PathBuf> {
    if !dataset_path.is_empty() {
        let path = PathBuf::from(dataset_path);
        anyhow::ensure!(
            path.is_file(),
            "--dataset-path {dataset_path} is not a file"
        );
        return Ok(path);
    }
    resolve_file(http, SHAREGPT_REPO, SHAREGPT_FILE, RepoKind::Dataset)
        .await
        .context("cannot obtain the ShareGPT dataset; pass --dataset-path")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RepoKind {
    Model,
    Dataset,
}

impl RepoKind {
    /// The Hub URL segment; a dataset repo is namespaced, a model repo is not.
    fn url_prefix(self) -> &'static str {
        match self {
            Self::Model => "",
            Self::Dataset => "datasets/",
        }
    }

    fn hub_type(self) -> hf_hub::RepoType {
        match self {
            Self::Model => hf_hub::RepoType::Model,
            Self::Dataset => hf_hub::RepoType::Dataset,
        }
    }
}

/// A local path as given, a file inside a given directory, or `filename` from
/// the repo `source` names.
async fn resolve_file(
    http: &reqwest::Client,
    source: &str,
    filename: &str,
    kind: RepoKind,
) -> Result<PathBuf> {
    let path = Path::new(source);
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        let candidate = path.join(filename);
        anyhow::ensure!(candidate.is_file(), "{} has no {filename}", path.display());
        return Ok(candidate);
    }
    if let Some(cached) = from_hub_cache(source, filename, kind) {
        return Ok(cached);
    }
    download(http, source, filename, kind).await
}

/// Look `filename` up in the local HF hub cache. Python resolves the cache
/// directory as `HF_HUB_CACHE` > `HUGGINGFACE_HUB_CACHE` > `HF_HOME/hub`,
/// while the hf-hub crate only knows `HF_HOME`; honour the overrides first or
/// a model the scheduler downloaded is missed.
fn from_hub_cache(repo_id: &str, filename: &str, kind: RepoKind) -> Option<PathBuf> {
    use hf_hub::{Cache, Repo};

    let cache = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .map(|dir| Cache::new(dir.into()))
        .unwrap_or_else(Cache::from_env);
    cache
        .repo(Repo::with_revision(
            repo_id.to_owned(),
            kind.hub_type(),
            "main".to_owned(),
        ))
        .get(filename)
}

/// Fetch `filename` from the Hub into this crate's own cache directory.
/// Downloads to a temporary file and renames, so an interrupted run does not
/// leave a truncated file that the next one would read as complete.
async fn download(
    http: &reqwest::Client,
    repo_id: &str,
    filename: &str,
    kind: RepoKind,
) -> Result<PathBuf> {
    let destination = download_cache_dir()?.join(format!(
        "{}--{filename}",
        repo_id.replace(['/', '\\'], "--")
    ));
    if destination.is_file() {
        return Ok(destination);
    }

    let url = format!(
        "https://huggingface.co/{}{repo_id}/resolve/main/{filename}",
        kind.url_prefix()
    );
    eprintln!("downloading {url}");
    let mut request = http.get(&url);
    if let Some(token) = hf_token() {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.with_context(|| format!("GET {url}"))?;
    anyhow::ensure!(
        response.status().is_success(),
        "GET {url} returned {}",
        response.status()
    );
    let body = response.bytes().await?;

    let partial = destination.with_extension("partial");
    std::fs::write(&partial, &body)
        .with_context(|| format!("cannot write {}", partial.display()))?;
    std::fs::rename(&partial, &destination)?;
    eprintln!("cached at {}", destination.display());
    Ok(destination)
}

fn hf_token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .filter(|token| !token.is_empty())
}

/// Where downloads land: under the HF home when one is set, else a per-user
/// directory in the temp dir.
fn download_cache_dir() -> Result<PathBuf> {
    let root = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("sglang-bench");
    std::fs::create_dir_all(&root).with_context(|| format!("cannot create {}", root.display()))?;
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path that exists is used as given; a directory is searched for the
    /// file; a missing file inside a real directory is an error rather than a
    /// silent fall through to a Hub download of the directory's name.
    #[tokio::test]
    async fn local_paths_resolve_before_the_hub() {
        let dir = std::env::temp_dir().join(format!("sglang-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("tokenizer.json");
        std::fs::write(&file, b"{}").unwrap();

        let http = reqwest::Client::new();
        let from_file = resolve_file(
            &http,
            file.to_str().unwrap(),
            "tokenizer.json",
            RepoKind::Model,
        )
        .await;
        assert_eq!(from_file.unwrap(), file);

        let from_dir = resolve_file(
            &http,
            dir.to_str().unwrap(),
            "tokenizer.json",
            RepoKind::Model,
        )
        .await;
        assert_eq!(from_dir.unwrap(), file);

        let missing =
            resolve_file(&http, dir.to_str().unwrap(), "absent.json", RepoKind::Model).await;
        assert!(missing.is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A dataset repo lives under a different Hub path than a model repo;
    /// swapping them yields a 404 on every download.
    #[test]
    fn dataset_and_model_repos_use_different_url_prefixes() {
        assert_eq!(RepoKind::Model.url_prefix(), "");
        assert_eq!(RepoKind::Dataset.url_prefix(), "datasets/");
    }

    #[tokio::test]
    async fn an_explicit_dataset_path_must_exist() {
        let http = reqwest::Client::new();
        assert!(
            resolve_sharegpt(&http, "/nonexistent/sharegpt.json")
                .await
                .is_err()
        );
    }
}
