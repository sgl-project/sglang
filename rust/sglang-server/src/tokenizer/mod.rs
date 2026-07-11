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
use crate::message::{Request, RequestKind};
use crate::runtime::Runnable;
use crate::runtime::channels::TmEvent;

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<i32>, Error>;
}

/// Load the dynamo tokenizer used by the detok shards for the incremental decode
/// stream. `None` under `skip_tokenizer_init`, else required (missing/failed load →
/// `Err`). `tokenizer_path` is a tokenizer file, a model dir, or an HF Hub repo id
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
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file(&file)
        .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?;
    tracing::info!(%path, "loaded tokenizer");
    Ok(Some(tokenizer))
}

/// Load the HuggingFace `tokenizers` tokenizer used by the encode pool. Separate
/// from [`load_tokenizer`] (dynamo) because the two libraries differ on special
/// tokens: dynamo hard-codes `add_special_tokens=false`, so it drops the BOS the
/// prompt's post-processor would add, whereas [`HfEncoder`] passes `true` to match
/// the Python server's `tokenizer.encode(text)`. Loads the same `tokenizer.json`;
/// `None` under `skip_tokenizer_init`.
pub fn load_encoder(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    skip_tokenizer_init: bool,
) -> Result<Option<tokenizers::Tokenizer>, String> {
    if skip_tokenizer_init {
        return Ok(None);
    }
    let path = tokenizer_path.ok_or_else(|| {
        "no tokenizer configured: set tokenizer_path or enable skip_tokenizer_init".to_string()
    })?;

    let file = resolve_model_file(path, revision, "tokenizer.json")
        .ok_or_else(|| format!("tokenizer.json not found for '{path}'"))?;
    let tokenizer = tokenizers::Tokenizer::from_file(&file)
        .map_err(|e| format!("encoder load failed ({file}): {e}"))?;
    tracing::info!(%path, "loaded encoder (add_special_tokens=true)");
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

/// Locate a file for an HF Hub repo id in the local cache (`HF_HOME`). Offline —
/// the scheduler pre-downloads the model. `None` if not cached.
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

/// Real encoder over the HuggingFace `tokenizers` library. Encodes with
/// `add_special_tokens=true` so the prompt gets its BOS / post-processor special
/// tokens — matching the Python server's `tokenizer.encode(text)`. dynamo's encoder
/// passes `false`, dropping BOS; on models that prepend one (Llama, gemma, …) that
/// silently changed the prompt and measurably cut accuracy (GSM8K gemma-2: 0.44 →
/// 0.29). Decode still runs through dynamo's `DecodeStream` (see the detok shard).
pub struct HfEncoder {
    inner: tokenizers::Tokenizer,
}

impl HfEncoder {
    pub fn new(inner: tokenizers::Tokenizer) -> Self {
        Self { inner }
    }
}

impl TextTokenizer for HfEncoder {
    fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
        if text.is_empty() {
            // Match Python sglang: reject an empty prompt as a 400 (`Validation`),
            // not the misleading 500 a tokenize error would give.
            return Err(Error::Validation("prompt cannot be empty".into()));
        }
        let encoding = self
            .inner
            .encode(text, true) // add_special_tokens=true → BOS etc., as Python does
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Vocab ids are non-negative and fit in i32.
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }
}

/// One tokenizer worker: pulls a `Request` off the shared inbox, fills
/// `input_ids`, returns it to the TokenizerManager. Pinned; backend shared.
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
            // The tokenizer pool only ever receives generate requests. Encode,
            // then advance the FSM: `TokenizeDone` on success.
            let event = {
                let RequestKind::Generate(g) = &mut req.kind else {
                    tracing::error!("tokenizer pool received a non-generate request");
                    continue;
                };
                match self.tokenizer.encode(g.text.as_deref().unwrap_or("")) {
                    Ok(ids) => {
                        g.input_ids = Some(ids);
                        Event::TokenizeDone
                    }
                    Err(err) => Event::Error(err),
                }
            };
            let _ = req.state.apply(event);
            if self.tm.send(TmEvent::Tokenized(req)).is_err() {
                tracing::error!("tm inbox closed; dropping request");
                break;
            }
        }
    }
}
