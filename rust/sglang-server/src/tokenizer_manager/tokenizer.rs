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
use crate::message::types::TokenIds;
use crate::runtime::Runnable;
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::{error::Error, fsm::Event};

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<TokenIds, Error>;

    /// The configured BOS, which need not be auto-added by `encode`.
    /// Only continuation encoding needs it; tokenizers without a BOS return None.
    fn bos_token_id(&self) -> Result<Option<i32>, Error> {
        Ok(None)
    }

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
    bos_token_id: Result<Option<i32>, Error>,
}

impl DynamoTokenizer {
    pub fn new(inner: dynamo_tokenizers::Tokenizer, config_file: Option<&str>) -> Self {
        // Keep a resolution error local to continuations: ordinary text
        // requests do not need BOS metadata and retain their existing behavior.
        let bos_token_id = resolve_bos_token_id(&inner, config_file);
        Self {
            inner,
            bos_token_id,
        }
    }
}

fn resolve_bos_token_id(
    tokenizer: &dynamo_tokenizers::Tokenizer,
    config_file: Option<&str>,
) -> Result<Option<i32>, Error> {
    let Some(config_file) = config_file else {
        return Ok(None);
    };
    let error = |message| Error::Tokenize(format!("cannot resolve continuation BOS: {message}"));
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(config_file).map_err(|e| error(e.to_string()))?)
            .map_err(|e| error(e.to_string()))?;
    let Some(value) = config.get("bos_token").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let bos = value
        .as_str()
        .or_else(|| value.get("content").and_then(serde_json::Value::as_str))
        .ok_or_else(|| error("bos_token must be a string or token object".into()))?;
    // Dynamo exposes encoding but not token_to_id. A configured special token
    // contributes exactly one ID between the same post-processor specials as
    // encode(""). Verify that insertion and its decoded content; never infer BOS
    // from the first auto-added token, which may instead be EOS.
    let encoded = tokenizer.encode(bos).map_err(|e| error(e.to_string()))?;
    let empty = tokenizer.encode("").map_err(|e| error(e.to_string()))?;
    let ids = encoded.token_ids();
    let specials = empty.token_ids();
    let prefix_len = ids.iter().zip(specials).take_while(|(a, b)| a == b).count();
    if ids.len() != specials.len() + 1 || ids[prefix_len + 1..] != specials[prefix_len..] {
        return Err(error(
            "configured bos_token does not encode as one special token".into(),
        ));
    }
    let id = ids[prefix_len];
    let decoded = tokenizer
        .decode(&[id], false)
        .map_err(|e| error(e.to_string()))?;
    if decoded.as_str() != bos {
        return Err(error(
            "configured bos_token does not resolve to its declared content".into(),
        ));
    }
    Ok(Some(id as i32))
}

impl TextTokenizer for DynamoTokenizer {
    fn bos_token_id(&self) -> Result<Option<i32>, Error> {
        self.bos_token_id.clone()
    }

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

/// Python appends a continuation prefix from a fresh tokenizer encode and
/// removes only its configured leading BOS. An automatically appended EOS and
/// explicit later BOS tokens remain observable.
fn strip_bos(mut ids: Vec<i32>, bos_token_id: Option<i32>) -> Vec<i32> {
    if bos_token_id.is_some_and(|bos| ids.first() == Some(&bos)) {
        ids.remove(0);
    }
    ids
}

/// One tokenizer worker: pulls a `Request` off the shared inbox, fills
/// `input_ids`, returns it to the TokenizerManager. Pinned; backend shared.
///
/// The `auto_specials` prefix (probed once at construction, Python's
/// `encode("")` probe) is stripped from template-rendered prompts —
/// [`GenerateRequest`]'s `skip_special_tokens` — so chat prompts gain no
/// extra BOS/EOS while plain text keeps the post-processor specials.
pub struct TokenizerWorker {
    rx: flume::Receiver<Request>,
    tm: flume::Sender<TmEvent>,
    tokenizer: Arc<dyn TextTokenizer>,
    auto_specials: Vec<i32>,
}

impl TokenizerWorker {
    pub fn new(
        rx: flume::Receiver<Request>,
        tm: flume::Sender<TmEvent>,
        tokenizer: Arc<dyn TextTokenizer>,
    ) -> Self {
        let auto_specials = tokenizer.auto_specials();
        Self {
            rx,
            tm,
            tokenizer,
            auto_specials,
        }
    }
}

impl Runnable for TokenizerWorker {
    fn run(self) {
        while let Ok(mut req) = self.rx.recv() {
            // The tokenizer pool only ever receives generate requests. Encode,
            // then advance the FSM: `TokenizeDone` on success (→ PreSendValidating).
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
                match self
                    .tokenizer
                    .encode(g.text.as_deref().unwrap_or(""))
                    .and_then(|mut ids| {
                        if g.skip_special_tokens {
                            ids = strip_auto_specials(ids, &self.auto_specials);
                        }
                        if let Some(prefix) = g.append_text.as_deref() {
                            ids.extend(strip_bos(
                                self.tokenizer.encode(prefix)?,
                                self.tokenizer.bos_token_id()?,
                            ));
                        }
                        Ok(ids)
                    }) {
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
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
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

    /// The strip reproduces `add_special_tokens=false`: one leading run of
    /// auto-added specials is removed, a template-rendered copy is kept, and
    /// tokenizers with no auto specials (empty probe) are untouched.
    #[test]
    fn strip_auto_specials_matches_add_special_tokens_false() {
        assert_eq!(strip_auto_specials(vec![0, 0, 1, 2], &[0]), vec![0, 1, 2]);
        assert_eq!(strip_auto_specials(vec![1, 2], &[0]), vec![1, 2]);
        assert_eq!(strip_auto_specials(vec![1, 2], &[]), vec![1, 2]);
        assert_eq!(strip_auto_specials(vec![0], &[0, 9]), vec![0]);
    }

    #[test]
    fn strip_bos_keeps_other_specials() {
        assert_eq!(strip_bos(vec![0, 9, 1], Some(0)), vec![9, 1]);
        assert_eq!(strip_bos(vec![0, 0, 1, 9], Some(0)), vec![0, 1, 9]);
        assert_eq!(strip_bos(vec![1], Some(0)), vec![1]);
        assert_eq!(strip_bos(vec![9, 1], None), vec![9, 1]);
    }

    /// Word tokens plus a prepended BOS marker (id 0) — like an HF tokenizer
    /// whose post-processor adds specials.
    struct MarkedTokenizer;
    impl TextTokenizer for MarkedTokenizer {
        fn bos_token_id(&self) -> Result<Option<i32>, Error> {
            Ok(Some(0))
        }

        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            Ok(vec![0, text.len() as i32])
        }
        fn auto_specials(&self) -> Vec<i32> {
            vec![0]
        }
    }

    /// `skip_special_tokens` strips the probed prefix: template-rendered
    /// prompts (chat) must not gain a BOS the template didn't render — Python's
    /// `add_special_tokens=False` at the chat-template encode site.
    #[test]
    fn skip_special_tokens_strips_the_auto_added_specials() {
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
            TokenizerWorker::new(req_rx, tm_tx, Arc::new(MarkedTokenizer)).run();
            let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
                panic!("expected Tokenized");
            };
            let RequestKind::Generate(g) = &req.kind else {
                panic!("expected generate");
            };
            g.input_ids.clone().expect("tokenized")
        };
        assert_eq!(run(false), vec![0, 2], "plain text prompts keep specials");
        assert_eq!(run(true), vec![2], "rendered prompts lose the auto BOS");
    }

    #[test]
    fn tokenizing_appends_a_continuation_prefix_once_without_bos() {
        let (req_tx, req_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
        req_tx
            .send(Request {
                rid: "1".into(),
                state: RequestState::Tokenizing,
                sink: ResponseSink::Local(tokio::sync::mpsc::channel(4).0),
                kind: RequestKind::Generate(Box::new(GenerateRequest {
                    rid: "1".into(),
                    text: Some("prompt".into()),
                    append_text: Some("prefix".into()),
                    skip_special_tokens: true,
                    ..Default::default()
                })),
            })
            .expect("send");
        drop(req_tx);
        TokenizerWorker::new(req_rx, tm_tx, Arc::new(MarkedTokenizer)).run();
        let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
            panic!("expected Tokenized");
        };
        let RequestKind::Generate(g) = &req.kind else {
            panic!("expected generate");
        };
        assert_eq!(g.input_ids, Some(vec![6, 6]));
    }

    fn tokenize_continuation(tokenizer: Arc<dyn TextTokenizer>) -> Request {
        let (req_tx, req_rx) = flume::unbounded();
        let (tm_tx, tm_rx) = flume::unbounded();
        req_tx
            .send(Request {
                rid: "continuation".into(),
                state: RequestState::Tokenizing,
                sink: ResponseSink::Local(mpsc::channel(4).0),
                kind: RequestKind::Generate(Box::new(GenerateRequest {
                    text: Some("prompt".into()),
                    append_text: Some("prefix".into()),
                    skip_special_tokens: true,
                    ..Default::default()
                })),
            })
            .unwrap();
        drop(req_tx);
        TokenizerWorker::new(req_rx, tm_tx, tokenizer).run();
        let TmEvent::Tokenized(request) = tm_rx.try_recv().unwrap() else {
            panic!("expected Tokenized");
        };
        assert!(tm_rx.try_recv().is_err(), "one submission returns once");
        request
    }

    struct EosOnlyTokenizer;

    impl TextTokenizer for EosOnlyTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            // The prefix starts with an explicit EOS, and every encode adds
            // a trailing EOS. An empty encode therefore yields EOS, not BOS.
            Ok(if text == "prefix" {
                vec![9, 2, 9]
            } else {
                vec![1, 9]
            })
        }

        fn auto_specials(&self) -> Vec<i32> {
            vec![9]
        }
    }

    #[test]
    fn continuation_preserves_eos_when_tokenizer_has_no_bos() {
        let request = tokenize_continuation(Arc::new(EosOnlyTokenizer));
        assert!(matches!(request.state, RequestState::PreSendValidating));
        let RequestKind::Generate(g) = request.kind else {
            panic!("expected generate");
        };
        assert_eq!(g.input_ids, Some(vec![1, 9, 9, 2, 9]));
    }

    struct ExplicitBosTokenizer;

    impl TextTokenizer for ExplicitBosTokenizer {
        fn bos_token_id(&self) -> Result<Option<i32>, Error> {
            Ok(Some(0))
        }

        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            Ok(if text == "prefix" {
                vec![0, 2, 0, 9]
            } else {
                vec![1]
            })
        }
    }

    #[test]
    fn continuation_strips_only_the_leading_bos_even_when_not_auto_added() {
        let request = tokenize_continuation(Arc::new(ExplicitBosTokenizer));
        assert!(matches!(request.state, RequestState::PreSendValidating));
        let RequestKind::Generate(g) = request.kind else {
            panic!("expected generate");
        };
        assert_eq!(g.input_ids, Some(vec![1, 2, 0, 9]));
    }

    struct FailingPrefixTokenizer;

    impl TextTokenizer for FailingPrefixTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            if text == "prefix" {
                Err(Error::Tokenize("invalid prefix".into()))
            } else {
                Ok(vec![1])
            }
        }
    }

    #[test]
    fn continuation_encode_error_does_not_return_partial_prompt_ids() {
        let request = tokenize_continuation(Arc::new(FailingPrefixTokenizer));
        assert!(matches!(
            request.state,
            RequestState::Failed(Error::Tokenize(ref message)) if message == "invalid prefix"
        ));
        let RequestKind::Generate(g) = request.kind else {
            panic!("expected generate");
        };
        assert_eq!(g.input_ids, None);
    }

    fn real_tokenizer(auto_bos: bool, auto_eos: bool, bos: serde_json::Value) -> DynamoTokenizer {
        use serde_json::json;

        let mut single = Vec::new();
        if auto_bos {
            single.push(json!({"SpecialToken": {"id": "<bos>", "type_id": 0}}));
        }
        single.push(json!({"Sequence": {"id": "A", "type_id": 0}}));
        let mut pair = single.clone();
        pair.push(json!({"Sequence": {"id": "B", "type_id": 0}}));
        if auto_eos {
            let eos = json!({"SpecialToken": {"id": "<eos>", "type_id": 0}});
            single.push(eos.clone());
            pair.push(eos);
        }
        let tokenizer_json = json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 3, "content": "<bos>", "special": true, "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 4, "content": "<eos>", "special": true, "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false}
            ],
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": {
                "type": "TemplateProcessing", "single": single, "pair": pair,
                "special_tokens": {
                    "<bos>": {"id": "<bos>", "ids": [3], "tokens": ["<bos>"]},
                    "<eos>": {"id": "<eos>", "ids": [4], "tokens": ["<eos>"]}
                }
            },
            "model": {"type": "WordLevel", "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "prompt": 1, "prefix": 2, "<bos>": 3, "<eos>": 4}}
        });
        let dir =
            std::env::temp_dir().join(format!("sglang-continuation-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
        let config = dir.join("tokenizer_config.json");
        std::fs::write(&config, json!({"bos_token": bos}).to_string()).unwrap();
        let inner = load_tokenizer(dir.to_str(), None, false).unwrap().unwrap();
        let tokenizer = DynamoTokenizer::new(inner, config.to_str());
        std::fs::remove_dir_all(dir).unwrap();
        tokenizer
    }

    #[test]
    fn continuation_bos_resolution_uses_configured_token_with_any_postprocessor() {
        for (auto_bos, auto_eos, bos) in [
            (true, true, serde_json::json!("<bos>")),
            (false, true, serde_json::json!({"content": "<bos>"})),
            (false, false, serde_json::json!("<bos>")),
        ] {
            let tokenizer = real_tokenizer(auto_bos, auto_eos, bos);
            let bos_id = tokenizer.bos_token_id().unwrap();
            assert_eq!(bos_id, Some(3));
            let mut expected = vec![2];
            if auto_eos {
                expected.push(4);
            }
            assert_eq!(
                strip_bos(tokenizer.encode("prefix").unwrap(), bos_id),
                expected
            );
            if auto_bos {
                expected.insert(0, 3);
            }
            assert_eq!(
                strip_bos(tokenizer.encode("<bos>prefix").unwrap(), bos_id),
                expected
            );
        }
    }

    #[test]
    fn continuation_real_tokenizer_without_bos_preserves_leading_eos() {
        let tokenizer = real_tokenizer(false, true, serde_json::Value::Null);
        assert_eq!(tokenizer.bos_token_id().unwrap(), None);
        assert_eq!(
            strip_bos(
                tokenizer.encode("<eos>prefix").unwrap(),
                tokenizer.bos_token_id().unwrap()
            ),
            vec![4, 2, 4]
        );
    }

    #[test]
    fn continuation_unresolvable_bos_errors_without_disabling_ordinary_encoding() {
        let tokenizer = real_tokenizer(false, true, serde_json::json!("<missing>"));
        assert!(tokenizer.bos_token_id().is_err());
        assert_eq!(tokenizer.encode("prompt").unwrap(), vec![1, 4]);
        let request = tokenize_continuation(Arc::new(tokenizer));
        assert!(matches!(
            request.state,
            RequestState::Failed(Error::Tokenize(_))
        ));
        let RequestKind::Generate(g) = request.kind else {
            panic!("expected generate");
        };
        assert_eq!(g.input_ids, None);
    }
}
