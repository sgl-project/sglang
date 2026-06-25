//! Tokenizer pool — CPU-bound, runs on pinned OS threads (off the async
//! executor). Each worker pulls a `Request` from the shared `flume` receiver,
//! fills `input_ids`, and moves the request back to the TokenizerManager inbox.
//!
//! The text→ids step is behind [`TextTokenizer`]. The real backend is
//! [`DynamoTokenizer`] (dynamo-tokenizers: HuggingFace / tiktoken / fastokens);
//! [`StubTokenizer`] is the byte fallback used when no tokenizer path is
//! configured or it fails to load, so the pipeline still runs end-to-end.
//!
//! Mirrors the Python `_tokenize_one_request` text path: when the request
//! already carries `input_ids` it skips tokenization (handled upstream in the
//! TokenizerManager `classify`); otherwise the prompt text is encoded here.

use std::path::Path;
use std::sync::Arc;

use crate::error::Error;
use crate::fsm::Event;
use crate::message::{EgressItem, Request, RequestKind};
use crate::runtime::channels::TmEvent;

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<i32>, Error>;
}

/// Load a dynamo-tokenizers `Tokenizer`. `path` may be:
///   * a tokenizer file (`tokenizer.json` for HF, `.model`/`.tiktoken`),
///   * a model directory containing `tokenizer.json`, or
///   * an HF Hub repo id (e.g. `Qwen/Qwen3-0.6B-FP8`) — resolved to its
///     already-downloaded local `tokenizer.json` via the HF cache (no network).
///
/// Loaded once and shared (it is `Clone`/Arc-backed) by both the tokenizer pool
/// and the detokenizer shards.
pub fn load_tokenizer(
    path: &str,
    revision: Option<&str>,
) -> Result<dynamo_tokenizers::Tokenizer, Error> {
    let p = Path::new(path);
    let file = if p.is_dir() {
        p.join("tokenizer.json").to_string_lossy().into_owned()
    } else if p.is_file() {
        path.to_string()
    } else {
        // Not a local path → treat as an HF Hub repo id.
        resolve_from_hub_cache(path, revision)?
    };
    dynamo_tokenizers::Tokenizer::from_file(&file)
        .map_err(|e| Error::Tokenize(format!("load tokenizer {file}: {e}")))
}

/// Locate `tokenizer.json` for an HF Hub repo id in the local cache (shared with
/// `huggingface_hub` via `HF_HOME`). The scheduler downloads the model before
/// the server starts, so the file is already present — this does no network I/O
/// and pulls no TLS/openssl deps.
fn resolve_from_hub_cache(repo_id: &str, revision: Option<&str>) -> Result<String, Error> {
    use hf_hub::{Cache, Repo, RepoType};

    let rev = revision.unwrap_or("main");
    Cache::from_env()
        .repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        ))
        .get("tokenizer.json")
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| {
            Error::Tokenize(format!(
                "tokenizer.json not in HF cache for repo '{repo_id}' (rev {rev}); \
                 pass a local path or pre-download the tokenizer"
            ))
        })
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
            return Err(Error::Tokenize("empty input text".into()));
        }
        let encoding = self
            .inner
            .encode(text)
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Vocab ids are non-negative and fit in i32 for the msgpack payload.
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
    }
}

/// Byte fallback: maps each UTF-8 byte to an id. Lets the pipeline run before a
/// real tokenizer is configured. Not a valid tokenization for any model.
pub struct StubTokenizer;

impl TextTokenizer for StubTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
        if text.is_empty() {
            return Err(Error::Tokenize("empty input text".into()));
        }
        Ok(text.bytes().map(|b| b as i32).collect())
    }
}

/// One worker iteration loop. `rx` is cloned per worker (MPMC), `tm` is the
/// return path to the TokenizerManager inbox, `tokenizer` is the shared backend.
pub fn run_worker(
    rx: flume::Receiver<Request>,
    tm: flume::Sender<TmEvent>,
    tokenizer: Arc<dyn TextTokenizer>,
) {
    while let Ok(mut req) = rx.recv() {
        // The tokenizer pool only ever receives generate requests (control
        // requests skip tokenization in the ingress `classify`).
        let RequestKind::Generate(g) = &mut req.kind else {
            tracing::error!("tokenizer pool received a non-generate request");
            continue;
        };
        match tokenizer.encode(g.payload.text.as_deref().unwrap_or("")) {
            Ok(ids) => {
                g.input_ids = Some(ids);
                if tm.send(TmEvent::Tokenized(req)).is_err() {
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
