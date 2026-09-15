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

use crate::message::request::{Request, RequestKind};
use crate::message::response::ResponseItem;
use crate::message::types::TokenIds;
use crate::runtime::Runnable;
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::{error::Error, fsm::Event};

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<TokenIds, Error> {
        if text.is_empty() {
            return Err(Error::Validation("prompt cannot be empty".into()));
        }
        self.encode_with_special_tokens(text, true)
    }

    fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenIds, Error>;
}

/// Load the tokenizer shared (Arc-backed) by the encode pool and detok shards.
/// `None` under `skip_tokenizer_init`, else required (missing/failed load → `Err`).
/// `tokenizer_path` is a tokenizer file, a model dir, or an HF Hub repo id
/// (resolved from the local cache — no network).
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    skip_tokenizer_init: bool,
) -> Result<Option<DynamoTokenizer>, String> {
    if skip_tokenizer_init {
        tracing::info!("skip_tokenizer_init: token ids in and out; no tokenizer/detokenizer");
        return Ok(None);
    }
    let path = tokenizer_path.ok_or_else(|| {
        "no tokenizer configured: set tokenizer_path or enable skip_tokenizer_init".to_string()
    })?;
    let file = resolve_model_file(path, revision, "tokenizer.json")
        .ok_or_else(|| format!("tokenizer.json not found for '{path}'"))?;
    let load = |add_special_tokens| {
        dynamo_tokenizers::Tokenizer::from_file_with_options(
            &file,
            dynamo_tokenizers::TokenizerOptions { add_special_tokens },
        )
        .map_err(|e| format!("tokenizer load failed ({file}): {e}"))
    };
    let tokenizer = DynamoTokenizer {
        with_special_tokens: load(true)?,
        without_special_tokens: load(false)?,
    };
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

/// Dynamo fixes post-processing options at construction. Keep both views so
/// disabling special tokens also handles suffix and paired post-processors.
#[derive(Clone)]
pub struct DynamoTokenizer {
    with_special_tokens: dynamo_tokenizers::Tokenizer,
    without_special_tokens: dynamo_tokenizers::Tokenizer,
}

impl DynamoTokenizer {
    pub fn decoder(&self) -> dynamo_tokenizers::Tokenizer {
        self.with_special_tokens.clone()
    }
}

impl TextTokenizer for DynamoTokenizer {
    fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenIds, Error> {
        let tokenizer = if add_special_tokens {
            &self.with_special_tokens
        } else {
            &self.without_special_tokens
        };
        let encoding = tokenizer
            .encode(text)
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Vocab ids are non-negative and fit in i32.
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
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
            if let RequestKind::Tokenize {
                text,
                add_special_tokens,
            } = &req.kind
            {
                let item = match self
                    .tokenizer
                    .encode_with_special_tokens(text, *add_special_tokens)
                {
                    Ok(ids) => ResponseItem::Tokenized(ids),
                    Err(error) => ResponseItem::Error(error),
                };
                let _ = req.sink.try_send(item);
                continue;
            }
            let event = {
                let RequestKind::Generate(g) = &mut req.kind else {
                    tracing::error!("tokenizer pool received a non-generate request");
                    continue;
                };
                // Size the scheduler's stop-match window in TOKENS, as Python's
                // `normalize(tokenizer)` does.
                let stop_tokens = g
                    .sampling_params
                    .stop_strs
                    .iter()
                    // A stop that won't encode falls back to its byte length rather
                    // than failing the request: still an over-estimate, never an
                    // under-estimate, so the scheduler cannot miss that stop.
                    .map(|s| self.tokenizer.encode(s).map_or(s.len(), |ids| ids.len()))
                    .max();
                if let Some(n) = stop_tokens {
                    g.sampling_params.stop_str_max_len = n;
                }
                let text = g.text.as_deref().unwrap_or("");
                let encoded = if text.is_empty() {
                    Err(Error::Validation("prompt cannot be empty".into()))
                } else {
                    self.tokenizer
                        .encode_with_special_tokens(text, !g.skip_special_tokens)
                };
                match encoded {
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

#[cfg(test)]
pub(crate) fn test_tokenizer() -> DynamoTokenizer {
    let dir = std::env::temp_dir().join(format!("sglang-tokenizer-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&dir).unwrap();
    let config = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}
            }
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"<s>": 0, "</s>": 1, "[UNK]": 2, "hello": 3, "world": 4, "世界": 5, "hi": 6}, "unk_token": "[UNK]"}
    });
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let tokenizer = load_tokenizer(dir.to_str(), None, false).unwrap().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    tokenizer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::request::{GenerateRequest, RequestKind};
    use crate::message::response::ResponseSink;
    use crate::message::sampling::SamplingParams;
    use crate::utils::fsm::RequestState;
    use tokio::sync::mpsc;

    /// One token per whitespace-separated word, so a stop's token count differs
    /// from its byte count and the two units cannot be confused.
    struct WordTokenizer;
    impl TextTokenizer for WordTokenizer {
        fn encode_with_special_tokens(&self, text: &str, _: bool) -> Result<TokenIds, Error> {
            Ok(text.split_whitespace().map(|_| 1i32).collect())
        }
    }

    /// The scheduler's stop-match window must reach the wire as a TOKEN count, as
    /// Python's `normalize(tokenizer)` produces.
    ///
    /// `Normalizing` leaves a UTF-8 BYTE count there — a safe over-estimate, but it
    /// makes the scheduler decode a longer tail on EVERY decode step of EVERY
    /// request (14 tokens vs 6 for a typical stop set). This stage owns the
    /// tokenizer, so it is where the exact count is resolved.
    #[test]
    fn tokenizing_replaces_the_byte_window_with_a_token_count() {
        let (req_tx, req_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();

        // 8 bytes vs 3 "tokens" under WordTokenizer — units are distinguishable.
        let sp = SamplingParams {
            stop_strs: vec!["a bb ccc".to_string(), "dd".to_string()],
            stop_str_max_len: 8, // what `normalize_stops` left: max BYTE length
            ..Default::default()
        };
        let (sink_tx, _sink_rx) = mpsc::channel(4);
        req_tx
            .send(Request {
                rid: "1".into(),
                state: RequestState::Tokenizing,
                sink: ResponseSink::Local(sink_tx),
                kind: RequestKind::Generate(Box::new(GenerateRequest {
                    rid: "1".into(),
                    text: Some("hello world".into()),
                    sampling_params: sp,
                    ..Default::default()
                })),
            })
            .expect("send");
        drop(req_tx); // closes the loop after one request

        TokenizerWorker::new(req_rx, tm_tx, Arc::new(WordTokenizer)).run();

        let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
            panic!("expected Tokenized");
        };
        let RequestKind::Generate(g) = &req.kind else {
            panic!("expected generate");
        };
        assert_eq!(
            g.sampling_params.stop_str_max_len, 3,
            "must be the max TOKEN count (3), not the byte count (8)"
        );
    }

    #[test]
    fn special_token_options_preserve_literal_tokens_and_handle_suffixes() {
        let tokenizer = test_tokenizer();
        assert_eq!(
            tokenizer
                .encode_with_special_tokens("<s> hello 世界", true)
                .unwrap(),
            vec![0, 0, 3, 5, 1]
        );
        assert_eq!(
            tokenizer
                .encode_with_special_tokens("<s> hello 世界", false)
                .unwrap(),
            vec![0, 3, 5]
        );
        assert_eq!(
            tokenizer.encode_with_special_tokens("", true).unwrap(),
            vec![0, 1]
        );
        assert!(
            tokenizer
                .encode_with_special_tokens("", false)
                .unwrap()
                .is_empty()
        );
        assert!(matches!(tokenizer.encode(""), Err(Error::Validation(_))));
    }

    #[test]
    fn generation_tokenization_obeys_the_template_special_token_flag() {
        let run = |skip_special_tokens: bool| {
            let (req_tx, req_rx) = flume::unbounded::<Request>();
            let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
            req_tx
                .send(Request {
                    rid: "1".into(),
                    state: RequestState::Tokenizing,
                    sink: ResponseSink::Local(tokio::sync::mpsc::channel(4).0),
                    kind: RequestKind::Generate(Box::new(GenerateRequest {
                        rid: "1".into(),
                        text: Some("hi".into()),
                        skip_special_tokens,
                        ..Default::default()
                    })),
                })
                .expect("send");
            drop(req_tx);
            TokenizerWorker::new(req_rx, tm_tx, Arc::new(test_tokenizer())).run();
            let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
                panic!("expected Tokenized");
            };
            let RequestKind::Generate(g) = &req.kind else {
                panic!("expected generate");
            };
            g.input_ids.clone().expect("tokenized")
        };
        assert_eq!(
            run(false),
            vec![0, 6, 1],
            "plain text prompts keep specials"
        );
        assert_eq!(
            run(true),
            vec![6],
            "templates own both prefix and suffix specials"
        );
    }
}
