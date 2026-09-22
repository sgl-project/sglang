//! Tokenizer primitives shared by renderer hosts.

use crate::{
    RendererError as Error, RendererLimits, SamplingParams, TextRequest, TokenIds, TokenIdsRequest,
};
use futures::channel::oneshot;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

enum PoolJob {
    Tokenize {
        request: Box<TextRequest>,
        reply: oneshot::Sender<Result<TokenIdsRequest, Error>>,
    },
    Stop,
}

struct TokenizerPoolInner {
    jobs: flume::Sender<PoolJob>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Drop for TokenizerPoolInner {
    fn drop(&mut self) {
        let workers = self.workers.get_mut().expect("tokenizer workers mutex");
        for _ in 0..workers.len() {
            let _ = self.jobs.send(PoolJob::Stop);
        }
        for worker in workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Bounded CPU tokenizer pool owned by renderer state.
#[derive(Clone)]
pub(crate) struct PooledTokenizer {
    inner: Arc<TokenizerPoolInner>,
}

impl PooledTokenizer {
    pub fn new(
        tokenizer: Arc<dyn TextTokenizer>,
        worker_count: usize,
        queue_capacity: usize,
    ) -> Self {
        let worker_count = worker_count.max(1);
        let (jobs, rx) = flume::bounded(queue_capacity.max(1));
        let mut workers = Vec::with_capacity(worker_count);
        for index in 0..worker_count {
            let rx = rx.clone();
            let tokenizer = tokenizer.clone();
            workers.push(
                std::thread::Builder::new()
                    .name(format!("renderer-tokenizer-{index}"))
                    .spawn(move || {
                        while let Ok(job) = rx.recv() {
                            match job {
                                PoolJob::Tokenize { request, reply } => {
                                    let result =
                                        tokenize_text_request(*request, tokenizer.as_ref());
                                    let _ = reply.send(result);
                                }
                                PoolJob::Stop => break,
                            }
                        }
                    })
                    .expect("spawn renderer tokenizer worker"),
            );
        }
        Self {
            inner: Arc::new(TokenizerPoolInner {
                jobs,
                workers: Mutex::new(workers),
            }),
        }
    }
}

impl PooledTokenizer {
    pub(crate) async fn tokenize(&self, request: TextRequest) -> Result<TokenIdsRequest, Error> {
        let jobs = self.inner.jobs.clone();
        let (reply, result) = oneshot::channel();
        jobs.send_async(PoolJob::Tokenize {
            request: Box::new(request),
            reply,
        })
        .await
        .map_err(|_| Error::Unavailable)?;
        result.await.map_err(|_| Error::WorkerDropped)?
    }
}

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<TokenIds, Error>;

    fn encode_segments(
        &self,
        segments: &[dynamo_tokenizers::EncodeSegment<'_>],
        add_special_tokens: bool,
    ) -> Result<TokenIds, Error> {
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
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<TokenIds, Error> {
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
    ) -> Result<TokenIds, Error> {
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

fn resolve_stop_token_window(sampling_params: &mut SamplingParams, tokenizer: &dyn TextTokenizer) {
    // Size the scheduler's stop-match window in TOKENS, as Python's
    // `normalize(tokenizer)` does.
    if let Some(stop_tokens) = sampling_params
        .stop_strs
        .iter()
        // A stop that won't encode falls back to its byte length rather
        // than failing the request: still an over-estimate, never an
        // under-estimate, so the scheduler cannot miss that stop.
        .map(|stop| {
            tokenizer
                .encode(stop, false)
                .map_or(stop.len(), |ids| ids.len())
        })
        .max()
    {
        sampling_params.stop_str_max_len = stop_tokens;
    }
}

/// Convert a text input into the token-ID request consumed by shared
/// post-tokenization preparation.
pub fn tokenize_text_request(
    request: TextRequest,
    tokenizer: &dyn TextTokenizer,
) -> Result<TokenIdsRequest, Error> {
    let TextRequest {
        rid,
        prompt,
        add_special_tokens,
        mut options,
        metadata,
    } = request;
    resolve_stop_token_window(&mut options.sampling_params, tokenizer);
    let input_ids = match prompt.encode_segments() {
        Some(segments) => tokenizer.encode_segments(&segments, add_special_tokens)?,
        None => tokenizer.encode(prompt.as_str(), add_special_tokens)?,
    };
    Ok(TokenIdsRequest {
        rid,
        input_ids,
        options,
        metadata,
    })
}

/// Validate fields that must be safe before tokenization or engine submission.
pub fn validate_text_request(request: &TextRequest, limits: &RendererLimits) -> Result<(), Error> {
    validate_request_id(&request.rid)?;
    if request.prompt.as_str().is_empty() {
        return Err(Error::Validation("prompt cannot be empty".into()));
    }
    let options = &request.options;
    validate_completion_fields(
        None,
        options.token_ids_logprob.as_deref(),
        options.return_hidden_states,
        limits,
    )
}

/// Validate an already-tokenized request without passing it through the text
/// tokenizer path.
pub fn validate_token_ids_request(
    request: &TokenIdsRequest,
    limits: &RendererLimits,
) -> Result<(), Error> {
    validate_request_id(&request.rid)?;
    if request.input_ids.is_empty() {
        return Err(Error::Validation("input_ids cannot be empty".into()));
    }
    let options = &request.options;
    validate_completion_fields(
        Some(&request.input_ids),
        options.token_ids_logprob.as_deref(),
        options.return_hidden_states,
        limits,
    )
}

pub(crate) fn validate_request_id(rid: &str) -> Result<(), Error> {
    if rid.len() > 128 {
        return Err(Error::Validation(format!(
            "rid is {} bytes, over the 128-byte limit",
            rid.len()
        )));
    }
    Ok(())
}

/// Validate the common completion fields before tokenization or engine
/// submission. Request identity remains an enclosing host concern.
pub fn validate_completion_fields(
    input_ids: Option<&[i32]>,
    token_ids_logprob: Option<&[i32]>,
    return_hidden_states: bool,
    limits: &RendererLimits,
) -> Result<(), Error> {
    for &id in input_ids.iter().flat_map(|ids| ids.iter()) {
        if id < 0 || id as u64 >= limits.vocab_size {
            return Err(Error::Validation(format!(
                "input_ids contains out-of-vocabulary token id {id}; valid range is [0, {})",
                limits.vocab_size
            )));
        }
    }
    for &id in token_ids_logprob.iter().flat_map(|ids| ids.iter()) {
        if id < 0 || id as u64 >= limits.vocab_size {
            return Err(Error::Validation(format!(
                "token_ids_logprob contains out-of-vocabulary token id {id}; valid range is [0, {})",
                limits.vocab_size
            )));
        }
    }
    if return_hidden_states && !limits.enable_return_hidden_states {
        return Err(Error::Validation(
            "The server is not configured to return the hidden states. Please set `--enable-return-hidden-states` to enable this feature."
                .into(),
        ));
    }
    Ok(())
}

/// Enforce the model context limit after tokenization.
pub fn check_total_tokens(
    request: &mut TokenIdsRequest,
    limits: &RendererLimits,
) -> Result<(), Error> {
    let mut input_ids = Some(std::mem::take(&mut request.input_ids));
    let result =
        check_completion_token_budget(&mut input_ids, &mut request.options.sampling_params, limits);
    request.input_ids = input_ids.expect("validated token-ID request retains input_ids");
    result
}

/// Enforce the context limit over the common token-only completion fields.
pub fn check_completion_token_budget(
    input_ids: &mut Option<TokenIds>,
    sampling_params: &mut SamplingParams,
    limits: &RendererLimits,
) -> Result<(), Error> {
    let max_req_len = limits.context_len;
    let input_len = input_ids.as_ref().map_or(0, Vec::len) as u64 + limits.num_reserved_tokens;
    if input_len >= max_req_len {
        if !limits.allow_auto_truncate {
            return Err(Error::Validation(format!(
                "The input ({input_len} tokens) is longer than the model's context length ({max_req_len} tokens)."
            )));
        }
        if let Some(ids) = input_ids {
            ids.truncate(max_req_len as usize);
        }
    }
    let input_len = input_ids.as_ref().map_or(0, Vec::len) as u64 + limits.num_reserved_tokens;
    let Some(max_new_tokens) = sampling_params.max_new_tokens else {
        return Ok(());
    };
    let total = input_len.saturating_add(max_new_tokens.max(0) as u64);
    if total <= max_req_len {
        return Ok(());
    }
    if !limits.allow_auto_truncate {
        return Err(Error::Validation(format!(
            "Requested token count exceeds the model's maximum context length of {max_req_len} tokens. You requested a total of {total} tokens: {input_len} tokens from the input messages and {max_new_tokens} tokens for the completion. Please reduce the number of tokens in the input messages or the completion to fit within the limit."
        )));
    }
    let clamped = max_req_len.saturating_sub(input_len) as i64;
    if sampling_params.min_new_tokens > clamped {
        return Err(Error::Validation(format!(
            "min_new_tokens must be in [0, max_new_tokens({clamped})], got {}",
            sampling_params.min_new_tokens
        )));
    }
    sampling_params.max_new_tokens = Some(clamped);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use dynamo_renderer::{RenderedPrompt, RenderedSegment};

    use super::*;
    use crate::GenerationOptions;

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

    struct SegmentTokenizer;

    impl TextTokenizer for SegmentTokenizer {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<TokenIds, Error> {
            Ok(vec![9])
        }

        fn encode_segments(
            &self,
            segments: &[dynamo_tokenizers::EncodeSegment<'_>],
            _add_special_tokens: bool,
        ) -> Result<TokenIds, Error> {
            Ok(segments
                .iter()
                .map(|segment| if segment.allow_special { 1 } else { 2 })
                .collect())
        }
    }

    #[test]
    fn rendered_prompt_preserves_segment_boundaries_until_tokenization() {
        let prompt = RenderedPrompt::segmented(vec![
            RenderedSegment::new("<control>", true),
            RenderedSegment::new("user text", false),
        ]);
        let tokenized = tokenize_text_request(
            TextRequest::rendered("request", prompt, false, GenerationOptions::default()),
            &SegmentTokenizer,
        )
        .unwrap();

        assert_eq!(tokenized.input_ids, [1, 2]);
    }
}
