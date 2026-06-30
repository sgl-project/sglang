//! Tokenizer pool — CPU-bound, runs on pinned OS threads (off the async
//! executor). Each worker pulls a `Request` from the shared `flume` receiver,
//! fills `input_ids`, and moves the request back to the TokenizerManager inbox.
//!
//! The text→ids step is behind [`TextTokenizer`], implemented by
//! [`DynamoTokenizer`] (dynamo-tokenizers: HuggingFace / tiktoken / fastokens).
//! A non-skip server requires a real tokenizer (enforced at startup); under
//! `skip_tokenizer_init` the pool isn't spawned at all.
//!
//! Mirrors the Python `_tokenize_one_request` text path: when the request
//! already carries `input_ids` it skips tokenization (handled upstream in the
//! TokenizerManager `classify`); otherwise the prompt text is encoded here.

use std::path::Path;
use std::sync::Arc;

use crate::error::Error;
use crate::fsm::Event;
use crate::message::{EgressItem, Request, RequestKind};
use crate::runtime::Runnable;
use crate::runtime::channels::TmEvent;

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<i32>, Error>;
}

/// Load the tokenizer shared by the tokenizer pool (encode) and the detok shards
/// (decode). `None` under `skip_tokenizer_init`; otherwise a real tokenizer is
/// required, so a missing path or a failed load is an `Err` (the detok backend
/// defaults to `Dynamo` — `Skip` is reserved for skip mode). Loaded once and
/// shared (it is `Clone`/Arc-backed) by both pools.
///
/// `tokenizer_path` may be:
///   * a tokenizer file (`tokenizer.json` for HF, `.model`/`.tiktoken`),
///   * a model directory containing `tokenizer.json`, or
///   * an HF Hub repo id (e.g. `Qwen/Qwen3-0.6B-FP8`) — resolved to its
///     already-downloaded local `tokenizer.json` via the HF cache (no network).
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
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file(&file)
        .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?;
    tracing::info!(%path, "loaded tokenizer");
    Ok(Some(tokenizer))
}

/// Resolve a model file (`tokenizer.json`, `tokenizer_config.json`, …) given the
/// configured tokenizer source: a directory → `dir/<file>`; a tokenizer file →
/// its parent dir; otherwise an HF Hub repo id → the local HF cache. `None` when
/// the file can't be located.
pub fn resolve_model_file(path: &str, revision: Option<&str>, filename: &str) -> Option<String> {
    let p = Path::new(path);
    if p.is_dir() {
        let f = p.join(filename);
        return f.is_file().then(|| f.to_string_lossy().into_owned());
    }
    if p.is_file() {
        // `path` is e.g. a `tokenizer.json`; look for the sibling next to it.
        let f = p.parent()?.join(filename);
        return f.is_file().then(|| f.to_string_lossy().into_owned());
    }
    // Not a local path → treat as an HF Hub repo id (offline cache lookup).
    resolve_from_hub_cache(path, revision, filename)
}

/// Locate a file for an HF Hub repo id in the local cache (shared with
/// `huggingface_hub` via `HF_HOME`). The scheduler downloads the model before
/// the server starts, so the file is already present — this does no network I/O
/// and pulls no TLS/openssl deps. `None` if the file isn't cached.
fn resolve_from_hub_cache(repo_id: &str, revision: Option<&str>, filename: &str) -> Option<String> {
    use hf_hub::{Cache, Repo, RepoType};

    let rev = revision.unwrap_or("main");
    Cache::from_env()
        .repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        ))
        .get(filename)
        .map(|p| p.to_string_lossy().into_owned())
}

/// Real tokenizer backed by dynamo-tokenizers, wrapping an already-loaded
/// `Tokenizer` (Arc inside, cheap to clone into every worker).
pub struct DynamoTokenizer {
    inner: dynamo_tokenizers::Tokenizer,
}

impl DynamoTokenizer {
    pub fn new(inner: dynamo_tokenizers::Tokenizer) -> Self {
        Self { inner }
    }
}

impl TextTokenizer for DynamoTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
        if text.is_empty() {
            // Match Python sglang: an empty prompt is a client error on both
            // `/generate` (`_tokenize_texts`: "texts cannot be empty") and
            // `/v1/completions` ("Prompt cannot be empty"). It does NOT adopt
            // OpenAI's empty→`<|endoftext|>` convention, so reject — but as a 400
            // (`Validation`) rather than the misleading 500 a tokenize error gives.
            return Err(Error::Validation("prompt cannot be empty".into()));
        }
        let encoding = self
            .inner
            .encode(text)
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Vocab ids are non-negative and fit in i32 for the msgpack payload.
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
    }
}

/// One tokenizer worker: pulls a `Request` off the shared MPMC inbox, fills its
/// `input_ids`, and returns it to the TokenizerManager. Spawned (pinned) per
/// worker as a [`Runnable`]; the `tokenizer` backend is shared read-only across
/// all workers.
pub struct TokenizerWorker {
    rx: flume::Receiver<Request>,
    tm: flume::Sender<TmEvent>,
    tokenizer: Arc<dyn TextTokenizer>,
}

impl TokenizerWorker {
    pub fn new(
        rx: flume::Receiver<Request>,
        tm: flume::Sender<TmEvent>,
        tokenizer: Arc<dyn TextTokenizer>,
    ) -> Self {
        Self { rx, tm, tokenizer }
    }
}

impl Runnable for TokenizerWorker {
    fn run(self) {
        while let Ok(mut req) = self.rx.recv() {
            // The tokenizer pool only ever receives generate requests (control
            // requests skip tokenization in the ingress `classify`).
            let RequestKind::Generate(g) = &mut req.kind else {
                tracing::error!("tokenizer pool received a non-generate request");
                continue;
            };
            match self
                .tokenizer
                .encode(g.payload.text.as_deref().unwrap_or(""))
            {
                Ok(ids) => {
                    g.input_ids = Some(ids);
                    if self.tm.send(TmEvent::Tokenized(req)).is_err() {
                        tracing::error!("tm inbox closed; dropping tokenized request");
                        break;
                    }
                }
                Err(e) => {
                    let _ = req.state.apply(Event::Error(e.clone()));
                    let _ = req.sink.try_send(EgressItem::Error(e));
                }
            }
        }
    }
}
