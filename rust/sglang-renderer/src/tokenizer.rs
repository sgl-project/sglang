//! Tokenizer primitives shared by renderer hosts.

use crate::{RendererError as Error, RendererLimits, TextCompletionRequest, TokenIds};
use std::path::Path;

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<TokenIds, Error>;

    /// The special tokens this tokenizer auto-prepends on every `encode` —
    /// Python's `encode("")` probe (`serving_chat._tokenizer_auto_adds_specials`).
    /// Empty when it adds none (tiktoken backends, no BOS/EOS post-processor).
    fn auto_specials(&self) -> Vec<i32> {
        Vec::new()
    }
}

/// Load the tokenizer shared (Arc-backed) by the encode pool and detok shards.
/// `None` under `skip_tokenizer_init`, else required (missing/failed load → `Err`).
/// `tokenizer_path` is a tokenizer file, a model dir, or an HF Hub repo id
/// (resolved from the local cache — no network).
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    skip_tokenizer_init: bool,
) -> Result<Option<dynamo_tokenizers::Tokenizer>, String> {
    if skip_tokenizer_init {
        tracing::info!("skip_tokenizer_init: token ids in and out; no tokenizer/detokenizer");
        return Ok(None);
    }
    let path = tokenizer_path.ok_or_else(|| {
        "no tokenizer configured: set tokenizer_path or enable skip_tokenizer_init".to_string()
    })?;
    let file = resolve_model_file(path, revision, "tokenizer.json")
        .ok_or_else(|| format!("tokenizer.json not found for '{path}'"))?;
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file_with_options(
        &file,
        dynamo_tokenizers::TokenizerOptions {
            add_special_tokens: true,
        },
    )
    .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?;
    tracing::info!(%path, "loaded tokenizer");
    Ok(Some(tokenizer))
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

    let rev = revision.unwrap_or("main");
    cache
        .repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        ))
        .get(filename)
        .map(|p| p.to_string_lossy().into_owned())
}

/// Real tokenizer over an already-loaded dynamo `Tokenizer` (Arc inside).
pub struct DynamoTokenizer {
    inner: dynamo_tokenizers::Tokenizer,
}

impl DynamoTokenizer {
    pub fn new(inner: dynamo_tokenizers::Tokenizer) -> Self {
        Self { inner }
    }
}

impl TextTokenizer for DynamoTokenizer {
    fn encode(&self, text: &str) -> Result<TokenIds, Error> {
        if text.is_empty() {
            // Match Python sglang: reject an empty prompt as a 400 (`Validation`),
            // not the misleading 500 a tokenize error would give.
            return Err(Error::Validation("prompt cannot be empty".into()));
        }
        let encoding = self
            .inner
            .encode(text)
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Vocab ids are non-negative and fit in i32.
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
    }

    /// The post-processor prepends exactly what `encode("")` returns, so the
    /// probe is the same prefix [`strip_auto_specials`] removes.
    fn auto_specials(&self) -> Vec<i32> {
        self.inner
            .encode("")
            .map(|encoding| encoding.token_ids().iter().map(|&id| id as i32).collect())
            .unwrap_or_default()
    }
}

/// Remove one leading run of auto-added specials — exactly what an
/// `add_special_tokens=false` encode would have produced, without a second
/// tokenizer instance (the post-processor always prepends the same prefix, so
/// a template-rendered copy of those tokens is preserved).
fn strip_auto_specials(mut ids: Vec<i32>, auto_specials: &[i32]) -> Vec<i32> {
    if ids.starts_with(auto_specials) {
        ids.drain(..auto_specials.len());
    }
    ids
}

/// Apply the text-tokenization stage to one generate request. Both the normal
/// tokenizer worker and the standalone renderer call this function so stop
/// sizing and special-token handling cannot drift between the two paths.
pub fn tokenize_generate_request(
    request: &mut TextCompletionRequest,
    tokenizer: &dyn TextTokenizer,
    auto_specials: &[i32],
) -> Result<(), Error> {
    // Size the scheduler's stop-match window in TOKENS, as Python's
    // `normalize(tokenizer)` does.
    if let Some(stop_tokens) = request
        .sampling_params
        .stop_strs
        .iter()
        // A stop that won't encode falls back to its byte length rather
        // than failing the request: still an over-estimate, never an
        // under-estimate, so the scheduler cannot miss that stop.
        .map(|stop| tokenizer.encode(stop).map_or(stop.len(), |ids| ids.len()))
        .max()
    {
        request.sampling_params.stop_str_max_len = stop_tokens;
    }
    let ids = tokenizer.encode(request.text.as_deref().unwrap_or(""))?;
    request.input_ids = Some(if request.skip_special_tokens {
        strip_auto_specials(ids, auto_specials)
    } else {
        ids
    });
    Ok(())
}

/// Run the engine-free validation, normalization, tokenization and context checks.
pub fn prepare_direct_request(
    mut request: TextCompletionRequest,
    tokenizer: &dyn TextTokenizer,
    auto_specials: &[i32],
    limits: &RendererLimits,
) -> Result<TextCompletionRequest, Error> {
    validate_request(&request, limits)?;
    request
        .sampling_params
        .normalize(limits.skip_tokenizer_init, limits.vocab_size)?;
    if !request.already_tokenized() {
        tokenize_generate_request(&mut request, tokenizer, auto_specials)?;
    }
    check_total_tokens(&mut request, limits)?;
    Ok(request)
}

/// Validate fields that must be safe before tokenization or engine submission.
pub fn validate_request(
    request: &TextCompletionRequest,
    limits: &RendererLimits,
) -> Result<(), Error> {
    if request.rid.len() > 128 {
        return Err(Error::Validation(format!(
            "rid is {} bytes, over the 128-byte limit",
            request.rid.len()
        )));
    }
    if limits.skip_tokenizer_init && !request.already_tokenized() {
        return Err(Error::Validation(
            "skip_tokenizer_init is set: request must provide input_ids".into(),
        ));
    }
    for &id in request.input_ids.iter().flatten() {
        if id < 0 || id as u64 >= limits.vocab_size {
            return Err(Error::Validation(format!(
                "input_ids contains out-of-vocabulary token id {id}; valid range is [0, {})",
                limits.vocab_size
            )));
        }
    }
    for &id in request.token_ids_logprob.iter().flatten() {
        if id < 0 || id as u64 >= limits.vocab_size {
            return Err(Error::Validation(format!(
                "token_ids_logprob contains out-of-vocabulary token id {id}; valid range is [0, {})",
                limits.vocab_size
            )));
        }
    }
    if request.return_hidden_states && !limits.enable_return_hidden_states {
        return Err(Error::Validation(
            "The server is not configured to return the hidden states. Please set `--enable-return-hidden-states` to enable this feature."
                .into(),
        ));
    }
    Ok(())
}

/// Enforce the model context limit after tokenization.
pub fn check_total_tokens(
    request: &mut TextCompletionRequest,
    limits: &RendererLimits,
) -> Result<(), Error> {
    let max_req_len = limits.context_len;
    let input_len =
        request.input_ids.as_ref().map_or(0, Vec::len) as u64 + limits.num_reserved_tokens;
    if input_len >= max_req_len {
        if !limits.allow_auto_truncate {
            return Err(Error::Validation(format!(
                "The input ({input_len} tokens) is longer than the model's context length ({max_req_len} tokens)."
            )));
        }
        if let Some(ids) = &mut request.input_ids {
            ids.truncate(max_req_len as usize);
        }
    }
    let input_len =
        request.input_ids.as_ref().map_or(0, Vec::len) as u64 + limits.num_reserved_tokens;
    let Some(max_new_tokens) = request.sampling_params.max_new_tokens else {
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
    if request.sampling_params.min_new_tokens > clamped {
        return Err(Error::Validation(format!(
            "min_new_tokens must be in [0, max_new_tokens({clamped})], got {}",
            request.sampling_params.min_new_tokens
        )));
    }
    request.sampling_params.max_new_tokens = Some(clamped);
    Ok(())
}
