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

    /// Encode a rendered prompt without adding post-processor special tokens.
    /// The default preserves the existing prefix-only behavior for custom
    /// implementations. `auto_specials` is cached by the worker, so this fallback
    /// does not encode an empty string again for every request.
    fn encode_without_special_tokens(
        &self,
        text: &str,
        auto_specials: &[i64],
    ) -> Result<TokenIds, Error> {
        Ok(strip_auto_specials(self.encode(text)?, auto_specials))
    }

    /// Empty-input probe used by the default prefix-only fallback above.
    /// Backends with suffix or other post-processing must override
    /// `encode_without_special_tokens` instead of assuming these form a prefix.
    fn auto_specials(&self) -> Vec<i64> {
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
    load_tokenizer_with_special_tokens(path, revision, true).map(Some)
}

/// Load the additional encode handle used for already-rendered chat prompts.
/// Dynamo fixes this option at construction time, so it cannot be changed by
/// stripping IDs after encoding (padding and truncation depend on the option).
pub fn load_tokenizer_without_special_tokens(
    tokenizer_path: &str,
    revision: Option<&str>,
) -> Result<dynamo_tokenizers::Tokenizer, String> {
    load_tokenizer_with_special_tokens(tokenizer_path, revision, false)
}

fn load_tokenizer_with_special_tokens(
    path: &str,
    revision: Option<&str>,
    add_special_tokens: bool,
) -> Result<dynamo_tokenizers::Tokenizer, String> {
    let file = resolve_model_file(path, revision, "tokenizer.json")
        .ok_or_else(|| format!("tokenizer.json not found for '{path}'"))?;
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file_with_options(
        &file,
        dynamo_tokenizers::TokenizerOptions { add_special_tokens },
    )
    .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?;
    tracing::info!(%path, add_special_tokens, "loaded tokenizer");
    Ok(tokenizer)
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
    without_specials: Option<dynamo_tokenizers::Tokenizer>,
}

impl DynamoTokenizer {
    pub fn new(inner: dynamo_tokenizers::Tokenizer) -> Self {
        Self {
            inner,
            without_specials: None,
        }
    }

    /// Supply both construction-time modes, shared by every tokenizer worker.
    /// The original handle remains available for ordinary prompts and decoding.
    pub fn with_special_token_modes(
        with_specials: dynamo_tokenizers::Tokenizer,
        without_specials: dynamo_tokenizers::Tokenizer,
    ) -> Self {
        Self {
            without_specials: Some(without_specials),
            ..Self::new(with_specials)
        }
    }

    fn encode_with(
        tokenizer: &dynamo_tokenizers::Tokenizer,
        text: &str,
    ) -> Result<TokenIds, Error> {
        if text.is_empty() {
            // Match Python sglang: reject an empty prompt as a 400 (`Validation`),
            // not the misleading 500 a tokenize error would give.
            return Err(Error::Validation("prompt cannot be empty".into()));
        }
        let encoding = tokenizer
            .encode(text)
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Widened once here, in the parallel pool: the scheduler's `array("q")`
        // is int64, and the ids move into their ring buffer untouched from here.
        Ok(encoding
            .token_ids()
            .iter()
            .map(|&id| i64::from(id))
            .collect())
    }
}

impl TextTokenizer for DynamoTokenizer {
    fn encode(&self, text: &str) -> Result<TokenIds, Error> {
        Self::encode_with(&self.inner, text)
    }

    fn encode_without_special_tokens(
        &self,
        text: &str,
        auto_specials: &[i64],
    ) -> Result<TokenIds, Error> {
        match &self.without_specials {
            Some(tokenizer) => Self::encode_with(tokenizer, text),
            // Preserve the original behavior for callers supplying one handle.
            None => Ok(strip_auto_specials(self.encode(text)?, auto_specials)),
        }
    }

    fn auto_specials(&self) -> Vec<i64> {
        self.inner
            .encode("")
            .map(|encoding| {
                encoding
                    .token_ids()
                    .iter()
                    .map(|&id| i64::from(id))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Legacy fallback for custom tokenizers that only prepend special tokens.
/// Concrete backends can override `encode_without_special_tokens` when their
/// post-processor can also append tokens or perform other transformations.
fn strip_auto_specials(mut ids: Vec<i64>, auto_specials: &[i64]) -> Vec<i64> {
    if ids.starts_with(auto_specials) {
        ids.drain(..auto_specials.len());
    }
    ids
}

/// One tokenizer worker: pulls a `Request` off the shared inbox, fills
/// `input_ids`, returns it to the TokenizerManager. Pinned; backend shared.
///
/// Template-rendered prompts (`GenerateRequest::skip_special_tokens`) use the
/// backend's encode-without-specials path. The once-probed `auto_specials` is
/// retained for custom tokenizers using the legacy prefix-only default.
pub struct TokenizerWorker {
    rx: flume::Receiver<Request>,
    tm: flume::Sender<TmEvent>,
    tokenizer: Arc<dyn TextTokenizer>,
    auto_specials: Vec<i64>,
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
            // then advance the FSM: `TokenizeDone` on success (-> PreSendValidating,
            // or -> Encoding for a multimodal prompt; the state's `then` says which).
            // A request is never dropped here: a kind this pool cannot serve
            // goes back as `Failed`, so intake rejects it and releases its
            // tracking entry instead of leaving the client hung.
            let event = if let RequestKind::Generate(g) = &mut req.kind {
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
                let encoded = if g.skip_special_tokens {
                    self.tokenizer
                        .encode_without_special_tokens(text, &self.auto_specials)
                } else {
                    self.tokenizer.encode(text)
                };
                match encoded {
                    Ok(ids) => {
                        g.input_ids = Some(ids);
                        Event::TokenizeDone
                    }
                    Err(err) => Event::Error(err),
                }
            } else {
                tracing::error!(rid = %req.rid, "tokenizer pool received a non-generate request");
                Event::Error(Error::Internal(
                    "non-generate request in the tokenizer pool".into(),
                ))
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
    use crate::utils::fsm::{AfterTokenize, RequestState};
    use tokio::sync::mpsc;

    /// One token per whitespace-separated word, so a stop's token count differs
    /// from its byte count and the two units cannot be confused.
    struct WordTokenizer;
    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            Ok(text.split_whitespace().map(|_| 1i64).collect())
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
                state: RequestState::Tokenizing {
                    then: AfterTokenize::PreSend,
                },
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

    /// A kind the pool cannot serve is returned `Failed`, never dropped:
    /// intake still holds its sink registration and pool-tracking entry.
    #[test]
    fn non_generate_request_is_returned_failed_not_dropped() {
        let (req_tx, req_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
        let (sink_tx, _sink_rx) = mpsc::channel(4);
        req_tx
            .send(Request {
                rid: "2".into(),
                state: RequestState::Tokenizing {
                    then: AfterTokenize::PreSend,
                },
                sink: ResponseSink::Local(sink_tx),
                kind: RequestKind::Detokenize {
                    token_ids: vec![1, 2],
                },
            })
            .expect("send");
        drop(req_tx);

        TokenizerWorker::new(req_rx, tm_tx, Arc::new(WordTokenizer)).run();

        let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned, not dropped") else {
            panic!("expected Tokenized");
        };
        assert_eq!(req.rid.as_str(), "2");
        assert!(
            matches!(req.state, RequestState::Failed(Error::Internal(_))),
            "{:?}",
            req.state
        );
    }

    /// The legacy fallback removes one probed prefix, preserving an explicit
    /// template-rendered copy and leaving tokenizers with no probe untouched.
    #[test]
    fn strip_auto_specials_matches_add_special_tokens_false() {
        assert_eq!(strip_auto_specials(vec![0, 0, 1, 2], &[0]), vec![0, 1, 2]);
        assert_eq!(strip_auto_specials(vec![1, 2], &[0]), vec![1, 2]);
        assert_eq!(strip_auto_specials(vec![1, 2], &[]), vec![1, 2]);
        assert_eq!(strip_auto_specials(vec![0], &[0, 9]), vec![0]);
    }

    /// Word tokens plus a prepended BOS marker (id 0) — like an HF tokenizer
    /// whose post-processor adds specials.
    struct MarkedTokenizer;
    impl TextTokenizer for MarkedTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            Ok(vec![0, text.len() as i64])
        }
        fn auto_specials(&self) -> Vec<i64> {
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
                    state: RequestState::Tokenizing {
                        then: AfterTokenize::PreSend,
                    },
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
    fn tokenize_prompt(tokenizer: DynamoTokenizer, text: &str, skip: bool) -> TokenIds {
        let (req_tx, req_rx) = flume::unbounded();
        let (tm_tx, tm_rx) = flume::unbounded();
        let (sink_tx, _sink_rx) = mpsc::channel(4);
        req_tx
            .send(Request {
                rid: "specials".into(),
                state: RequestState::Tokenizing {
                    then: AfterTokenize::PreSend,
                },
                sink: ResponseSink::Local(sink_tx),
                kind: RequestKind::Generate(Box::new(GenerateRequest {
                    text: Some(text.into()),
                    skip_special_tokens: skip,
                    ..Default::default()
                })),
            })
            .unwrap();
        drop(req_tx);
        TokenizerWorker::new(req_rx, tm_tx, Arc::new(tokenizer)).run();
        let TmEvent::Tokenized(req) = tm_rx.try_recv().unwrap() else {
            panic!("expected tokenized request")
        };
        assert!(matches!(req.state, RequestState::PreSendValidating));
        let RequestKind::Generate(g) = req.kind else {
            panic!("expected generate")
        };
        g.input_ids.unwrap()
    }

    fn check_postprocessor_specials(prefix: bool, suffix: bool) {
        check_postprocessor_config(
            prefix,
            suffix,
            serde_json::Value::Null,
            serde_json::Value::Null,
        );
    }

    fn check_postprocessor_config(
        prefix: bool,
        suffix: bool,
        padding: serde_json::Value,
        truncation: serde_json::Value,
    ) {
        let dir = std::env::temp_dir().join(format!(
            "sglang-tokenizer-specials-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        let mut single = Vec::new();
        if prefix {
            single.push(serde_json::json!({"SpecialToken": {"id": "<s>", "type_id": 0}}));
        }
        single.push(serde_json::json!({"Sequence": {"id": "A", "type_id": 0}}));
        if suffix {
            single.push(serde_json::json!({"SpecialToken": {"id": "</s>", "type_id": 0}}));
        }
        let json = serde_json::json!({
            "version": "1.0", "truncation": truncation, "padding": padding,
            "added_tokens": [
                {"id":2,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":3,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
            ],
            "normalizer": null, "pre_tokenizer": {"type":"Whitespace"},
            "post_processor": {"type":"TemplateProcessing", "single":single,
                "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
                "special_tokens": {
                    "<s>":{"id":"<s>","ids":[2],"tokens":["<s>"]},
                    "</s>":{"id":"</s>","ids":[3],"tokens":["</s>"]}
                }},
            "decoder": null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"<s>":2,"</s>":3},"unk_token":"[UNK]"}
        });
        std::fs::write(&path, json.to_string()).unwrap();
        let load = |add_special_tokens| {
            dynamo_tokenizers::Tokenizer::from_file_with_options(
                path.to_str().unwrap(),
                dynamo_tokenizers::TokenizerOptions { add_special_tokens },
            )
            .unwrap()
        };
        let with_specials = load(true);
        let without_specials =
            load_tokenizer_without_special_tokens(dir.to_str().unwrap(), None).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
        for text in [
            "hello",
            "hello hello hello hello",
            "<s>hello</s>",
            "hello</s></s>",
        ] {
            for skip in [false, true] {
                let reference = if skip {
                    &without_specials
                } else {
                    &with_specials
                };
                let expected: Vec<_> = reference
                    .encode(text)
                    .unwrap()
                    .token_ids()
                    .iter()
                    .map(|&id| i64::from(id))
                    .collect();
                assert_eq!(
                    tokenize_prompt(
                        DynamoTokenizer::with_special_token_modes(
                            with_specials.clone(),
                            without_specials.clone()
                        ),
                        text,
                        skip
                    ),
                    expected,
                    "prefix={prefix}, suffix={suffix}, skip={skip}, text={text:?}"
                );
            }
        }
        if padding.is_null() && truncation.is_null() {
            assert_eq!(
                tokenize_prompt(
                    DynamoTokenizer::with_special_token_modes(with_specials, without_specials),
                    "<s>hello</s>",
                    true
                ),
                [2, 1, 3]
            );
        }
    }

    #[test]
    fn chat_tokenization_omits_added_suffix_eos() {
        check_postprocessor_specials(false, true);
    }
    #[test]
    fn chat_tokenization_omits_added_bos_and_eos() {
        check_postprocessor_specials(true, true);
    }
    #[test]
    fn chat_tokenization_preserves_prefix_only_behavior() {
        check_postprocessor_specials(true, false);
    }
    #[test]
    fn chat_tokenization_preserves_no_postprocessor_specials() {
        check_postprocessor_specials(false, false);
    }
    #[test]
    fn chat_tokenization_preserves_configured_padding() {
        check_postprocessor_config(
            true,
            true,
            serde_json::json!({
                "strategy": {"Fixed": 6}, "direction": "Right", "pad_to_multiple_of": null,
                "pad_id": 0, "pad_type_id": 0, "pad_token": "[UNK]"
            }),
            serde_json::Value::Null,
        );
    }

    #[test]
    fn chat_tokenization_does_not_reserve_truncation_slots_for_specials() {
        check_postprocessor_config(
            true,
            true,
            serde_json::Value::Null,
            serde_json::json!({
                "direction": "Right", "max_length": 3, "strategy": "LongestFirst", "stride": 0
            }),
        );
    }
}
