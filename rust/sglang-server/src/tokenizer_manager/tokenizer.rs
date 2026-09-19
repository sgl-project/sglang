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

use std::path::{Path, PathBuf};
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
    use hf_hub::Cache;

    // Python resolves the cache dir as HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE >
    // HF_HOME/hub > ~/.cache/huggingface/hub; the hf-hub crate only knows
    // HF_HOME. Honor the explicit cache-dir overrides first, or the Rust
    // server misses models the Python scheduler already downloaded.
    let cache = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .map(|dir| Cache::new(dir.into()))
        .unwrap_or_else(Cache::from_env);

    cached_model_file(&cache, repo_id, revision, filename).map(|p| p.to_string_lossy().into_owned())
}

fn cached_model_file(
    cache: &hf_hub::Cache,
    repo_id: &str,
    revision: Option<&str>,
    filename: &str,
) -> Option<PathBuf> {
    use hf_hub::{Repo, RepoType};

    let rev = revision.unwrap_or("main");
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, rev.to_string());
    // hf-hub 0.4 requires refs/<revision>, even for commit-pinned snapshots.
    // Resolve full commits directly, as 1.0 does; keep other lookups on 0.4.
    // https://github.com/huggingface/hf-hub/blob/dcf865adacf7cbb750038d6521deb16c670ffd42/hf-hub/src/repository/download.rs#L348-L379
    if rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()) {
        let file = cache
            .path()
            .join(repo.folder_name())
            .join("snapshots")
            .join(rev)
            .join(filename);
        return file.exists().then_some(file);
    }

    cache.repo(repo).get(filename)
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
                match self.tokenizer.encode(g.text.as_deref().unwrap_or("")) {
                    Ok(ids) => {
                        g.input_ids = Some(if g.skip_special_tokens {
                            strip_auto_specials(ids, &self.auto_specials)
                        } else {
                            ids
                        });
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

    #[test]
    fn cached_model_files_resolve_exact_commits_and_named_refs() {
        const TEST_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
        let dir = std::env::temp_dir().join(format!("sglang-hub-{}", uuid::Uuid::new_v4()));
        let cache = hf_hub::Cache::new(dir.clone());
        let repo_dir = dir.join("models--org--model");
        let snapshot = repo_dir.join("snapshots").join(TEST_COMMIT);
        std::fs::create_dir_all(&snapshot).unwrap();
        let tokenizer = snapshot.join("tokenizer.json");
        std::fs::write(&tokenizer, "{}").unwrap();
        let config = snapshot.join("tokenizer_config.json");
        std::fs::write(&config, "{}").unwrap();
        let directory = snapshot.join("directory.json");
        std::fs::create_dir(&directory).unwrap();
        #[cfg(unix)]
        {
            std::fs::create_dir(repo_dir.join("blobs")).unwrap();
            std::fs::rename(&config, repo_dir.join("blobs/config")).unwrap();
            std::os::unix::fs::symlink("../../blobs/config", &config).unwrap();
            std::os::unix::fs::symlink("../../blobs/absent", snapshot.join("broken.json")).unwrap();
        }
        assert_eq!(
            cached_model_file(&cache, "org/model", Some(TEST_COMMIT), "tokenizer.json"),
            Some(tokenizer.clone())
        );
        assert!(!repo_dir.join("refs").exists());

        let other = "ffffffffffffffffffffffffffffffffffffffff";
        let other_snapshot = repo_dir.join("snapshots").join(other);
        std::fs::create_dir(&other_snapshot).unwrap();
        let other_tokenizer = other_snapshot.join("tokenizer.json");
        std::fs::write(&other_tokenizer, "{}").unwrap();
        std::fs::create_dir_all(repo_dir.join("refs/feature")).unwrap();
        for (name, commit) in [
            ("main", TEST_COMMIT),
            ("v1", TEST_COMMIT),
            ("feature/nested", TEST_COMMIT),
            ("whitespace", &format!(" \n{TEST_COMMIT}\t")),
            (TEST_COMMIT, other),
            ("0123", other),
        ] {
            std::fs::write(repo_dir.join("refs").join(name), commit).unwrap();
        }
        // Preserve 0.4's untrimmed refs and existence checks, including directories.
        for (revision, filename, expected) in [
            (None, "tokenizer.json", Some(&tokenizer)),
            (Some("main"), "tokenizer_config.json", Some(&config)),
            (Some("v1"), "tokenizer.json", Some(&tokenizer)),
            (Some("feature/nested"), "tokenizer.json", Some(&tokenizer)),
            (Some("whitespace"), "tokenizer.json", None),
            (Some(TEST_COMMIT), "tokenizer.json", Some(&tokenizer)),
            (Some(TEST_COMMIT), "tokenizer_config.json", Some(&config)),
            (Some(other), "tokenizer.json", Some(&other_tokenizer)),
            (Some(other), "tokenizer_config.json", None),
            (Some("0123"), "tokenizer.json", Some(&other_tokenizer)),
            (Some("0123456789ab"), "tokenizer.json", None),
            (Some("missing"), "tokenizer.json", None),
            (
                Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                "tokenizer.json",
                None,
            ),
            (Some(TEST_COMMIT), "missing.json", None),
            (Some("main"), "uncached.json", None),
            (Some(TEST_COMMIT), "broken.json", None),
            (Some("main"), "broken.json", None),
            (Some(TEST_COMMIT), "directory.json", Some(&directory)),
            (Some("main"), "directory.json", Some(&directory)),
        ] {
            assert_eq!(
                cached_model_file(&cache, "org/model", revision, filename).as_ref(),
                expected,
                "{revision:?}/{filename}"
            );
        }
        // Local-file sibling lookup keeps snapshot paths, including symlinks.
        for source in [snapshot.as_path(), tokenizer.as_path()] {
            assert_eq!(
                resolve_model_file(source.to_str().unwrap(), None, "tokenizer_config.json"),
                Some(config.to_string_lossy().into_owned())
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

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

    /// Word tokens plus a prepended BOS marker (id 0) — like an HF tokenizer
    /// whose post-processor adds specials.
    struct MarkedTokenizer;
    impl TextTokenizer for MarkedTokenizer {
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
}
