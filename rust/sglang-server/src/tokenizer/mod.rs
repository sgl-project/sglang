//! Tokenizer pool — CPU-bound, runs on pinned OS threads (off the async
//! executor). Each worker pulls a `Request` from the shared `flume` receiver,
//! fills `input_ids`, and moves the request back to the TokenizerManager inbox.
//!
//! The text→ids step is behind [`TextTokenizer`], implemented by [`HfEncoder`]
//! over dynamo-tokenizers' HuggingFace backend.
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

/// Load the plain HuggingFace tokenizer used by the encode pool, configured once
/// with `add_special_tokens=true` to match the Python server's
/// `tokenizer.encode(text)`. This construction path intentionally does not select
/// fastokens or add a `CachedTokenizer` wrapper. Loads the same `tokenizer.json`;
/// `None` under `skip_tokenizer_init`.
pub fn load_encoder(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    skip_tokenizer_init: bool,
) -> Result<Option<dynamo_tokenizers::Tokenizer>, String> {
    if skip_tokenizer_init {
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

/// Real encoder over dynamo-tokenizers' plain HuggingFace backend. The inner
/// tokenizer is constructed with `add_special_tokens=true`, so the prompt gets
/// its BOS / post-processor special tokens — matching the Python server's
/// `tokenizer.encode(text)`. On models that prepend one (Llama, Gemma, …), dropping
/// BOS silently changed the prompt and measurably cut accuracy (GSM8K Gemma-2:
/// 0.44 → 0.29). Decode still runs through the default-configured tokenizer used
/// by the detok shard.
pub struct HfEncoder {
    inner: dynamo_tokenizers::Tokenizer,
}

impl HfEncoder {
    pub fn new(inner: dynamo_tokenizers::Tokenizer) -> Self {
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

#[cfg(test)]
mod tests {
    use dynamo_tokenizers::Encoding;
    use tokenizers::{
        AddedToken, Tokenizer as DirectHfTokenizer, models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::WhitespaceSplit, processors::template::TemplateProcessing,
    };

    use super::*;

    struct TokenizerFixture {
        _dir: tempfile::TempDir,
        path: String,
    }

    fn tokenizer_fixture() -> TokenizerFixture {
        let vocab = [
            ("[UNK]", 0),
            ("<s>", 1),
            ("</s>", 2),
            ("<|im_start|>", 3),
            ("<|im_end|>", 4),
            ("system", 5),
            ("user", 6),
            ("hello", 7),
            ("world", 8),
            ("again", 9),
        ]
        .into_iter()
        .map(|(token, id)| (token.to_string(), id))
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("build word-level model");
        let mut tokenizer = DirectHfTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.add_special_tokens(&[
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
            AddedToken::from("<|im_start|>", true),
            AddedToken::from("<|im_end|>", true),
        ]);
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<s> $A </s>")
                .expect("single-sequence template")
                .special_tokens(vec![("<s>", 1), ("</s>", 2)])
                .build()
                .expect("build template processor"),
        ));

        let dir = tempfile::tempdir().expect("create fixture directory");
        let path = dir.path().join("tokenizer.json");
        tokenizer
            .save(&path, false)
            .expect("save tokenizer fixture");
        TokenizerFixture {
            _dir: dir,
            path: path.to_string_lossy().into_owned(),
        }
    }

    #[test]
    fn add_special_tokens_encoder_matches_direct_hf() {
        let fixture = tokenizer_fixture();
        let selected = load_encoder(Some(&fixture.path), None, false)
            .expect("load encoder")
            .expect("encoder enabled");
        let direct = DirectHfTokenizer::from_file(&fixture.path).expect("load direct HF tokenizer");

        let prompts = [
            "<|im_start|> system <|im_end|> <|im_start|> user hello world <|im_end|>",
            "<|im_start|> user hello again <|im_end|>",
        ];

        for prompt in prompts {
            let expected = direct.encode(prompt, true).expect("direct HF encode");
            for _ in 0..3 {
                let actual = selected.encode(prompt).expect("selected encode");
                assert!(
                    matches!(actual, Encoding::Hf(_)),
                    "add_special_tokens=true must select the plain HF backend"
                );
                assert_eq!(actual.token_ids(), expected.get_ids());
            }
        }

        let expected_batch = direct
            .encode_batch(prompts.to_vec(), true)
            .expect("direct HF batch encode");
        let actual_batch = selected
            .encode_batch(&prompts)
            .expect("selected batch encode");
        assert_eq!(actual_batch.len(), expected_batch.len());
        for (actual, expected) in actual_batch.iter().zip(&expected_batch) {
            assert!(matches!(actual, Encoding::Hf(_)));
            assert_eq!(actual.token_ids(), expected.get_ids());
        }

        let ids = actual_batch[0].token_ids();
        let bos = direct.token_to_id("<s>").expect("BOS id");
        let eos = direct.token_to_id("</s>").expect("EOS id");
        assert_eq!(ids.first(), Some(&bos));
        assert_eq!(ids.last(), Some(&eos));
        assert_eq!(ids.iter().filter(|&&id| id == bos).count(), 1);
        assert_eq!(ids.iter().filter(|&&id| id == eos).count(), 1);

        let worker_encoder = HfEncoder::new(selected);
        let worker_ids = worker_encoder.encode(prompts[0]).expect("worker encode");
        assert_eq!(
            worker_ids,
            expected_batch[0]
                .get_ids()
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>()
        );
    }
}
