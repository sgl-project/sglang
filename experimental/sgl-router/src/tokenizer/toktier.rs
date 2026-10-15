// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--tokenizer-backend toktier`: session-delta encoding through the
//! [`toktier`] crate. Decoding, vocabulary lookups and every fallback go to
//! the Hugging Face backend loaded from the same `tokenizer.json`.
//!
//! Every id is the one the reference `tokenizers` engine returns for the same
//! text; toktier only changes how much of the text is tokenized again. An
//! encode takes the first route that applies:
//!
//! 1. A session id is bound on the calling thread ([`SessionScope`]): when the
//!    stored transcript is a byte-exact prefix of the prompt, only the appended
//!    tail is tokenized (`session_repair`); otherwise the prompt is tokenized
//!    whole and replaces the transcript (`session_seed`).
//! 2. No session id: a stored transcript that is a verified prefix of the
//!    prompt is completed (`content_lookup`). The lookup walks sealed
//!    4096-character blocks, so it only finds transcripts at least that long.
//! 3. Otherwise toktier's admitted whole-prompt route (`full`).
//!
//! Anything toktier declines encodes on Hugging Face (`fallback_hf`): a
//! toktier error can cost latency, never an id.
//!
//! Routes 1 and 2 are cheaper than `hf` only when toktier admits its fast CPU
//! engine on this build (`plan().backends` holds `FastCpu`), which is what
//! seals the prefix a repair starts from. With only the reference engine, a
//! session turn re-encodes the whole transcript: exact, but no faster than
//! `hf`. The backend warns at startup in that state.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{Context, Result};
use dynamo_tokenizers::traits::{DecodeResult, Decoder, Encoder, Tokenizer};
use dynamo_tokenizers::{Encoding, Error, TokenIdType};
use sha2::{Digest, Sha256};

use super::stats::ToktierCounters;

/// Idle sessions kept per process; beyond this the least recently used one is
/// deleted. Below toktier's own store capacity (1024), so a session the router
/// holds is never evicted underneath it.
const MAX_IDLE_SESSIONS: usize = 512;

thread_local! {
    static SESSION_KEY: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Binds a session id on the current thread for a synchronous tokenization;
/// dropping it restores the previous binding. Not for use across an await.
pub struct SessionScope {
    previous: Option<String>,
}

impl SessionScope {
    pub fn bind(key: Option<&str>) -> Self {
        let previous = SESSION_KEY.with(|k| k.replace(key.map(str::to_owned)));
        Self { previous }
    }

    fn current() -> Option<String> {
        SESSION_KEY.with(|k| k.borrow().clone())
    }
}

impl Drop for SessionScope {
    fn drop(&mut self) {
        SESSION_KEY.with(|k| *k.borrow_mut() = self.previous.take());
    }
}

/// Idle sessions by id with the tick of their last use, for LRU deletion. A
/// session being encoded is taken out of the map, so a key has at most one
/// writer; a second request on the same key meanwhile encodes without it.
#[derive(Default)]
struct Sessions {
    idle: HashMap<String, (toktier::Session, u64)>,
    tick: u64,
}

pub(crate) struct ToktierTokenizer {
    /// The Hugging Face backend: decoding, vocabulary lookups and fallback.
    inner: Arc<dyn Tokenizer>,
    tok: toktier::Tokenizer,
    sessions: Mutex<Sessions>,
    counters: Arc<ToktierCounters>,
}

impl ToktierTokenizer {
    /// Open `tokenizer_json` through toktier. Errs when its SHA-256 is not one
    /// of the artifacts toktier ships certification data for, or toktier cannot
    /// load it; the caller then stays on `inner`.
    pub(crate) fn open(
        tokenizer_json: &Path,
        inner: Arc<dyn Tokenizer>,
        counters: &Arc<ToktierCounters>,
    ) -> Result<Self> {
        let bytes = std::fs::read(tokenizer_json)
            .with_context(|| format!("read {}", tokenizer_json.display()))?;
        let digest = format!("{:x}", Sha256::digest(bytes));
        let runtime = toktier::Runtime::builder()
            .device(toktier::Device::Cpu)
            .build()
            .context("toktier runtime")?;
        let family = runtime
            .family_for_sha256(&digest)
            .with_context(|| format!("toktier ships no certified artifact with sha256 {digest}"))?;
        let tok = runtime
            .load_local(&family, tokenizer_json)
            .with_context(|| format!("toktier load_local {family}"))?;
        let plan = tok.plan();
        tracing::info!(
            family,
            certification = ?plan.certification,
            backends = ?plan.backends,
            reasons = ?plan.reasons,
            "toktier tokenizer loaded"
        );
        if !plan.backends.contains(&toktier::Backend::FastCpu) {
            tracing::warn!(
                family,
                reasons = ?plan.reasons,
                "toktier admitted no accelerated engine on this build: every route runs the \
                 reference engine, exact but no faster than `hf`; `toktier-rust doctor` has \
                 the build facts"
            );
        }
        Ok(Self {
            inner,
            tok,
            sessions: Mutex::new(Sessions::default()),
            counters: Arc::clone(counters),
        })
    }

    /// Route `text` as the module docs describe; `key` is the bound session id.
    fn encode_routed(&self, text: &str, key: Option<&str>) -> Result<Encoding> {
        if text.is_empty() {
            return self.inner.encode(text);
        }
        match self.toktier_ids(text, key) {
            Ok(ids) => Ok(Encoding::Sp(ids)),
            Err(e) => {
                tracing::debug!(error = %e, "toktier declined the encode; encoding on hf");
                bump(&self.counters.fallback_hf);
                self.inner.encode(text)
            }
        }
    }

    /// toktier's routes in order: the session, a content lookup, the whole prompt.
    fn toktier_ids(&self, text: &str, key: Option<&str>) -> toktier::Result<Vec<u32>> {
        if let Some(key) = key {
            match self.encode_session(key, text) {
                Ok(ids) => return Ok(ids),
                Err(e) => tracing::debug!(error = %e,
                    "toktier declined the session route; encoding without the session"),
            }
        }
        match self.tok.lookup(text) {
            Ok(Some(hit)) => {
                bump(&self.counters.content_lookup);
                return Ok(hit.ids().to_vec());
            }
            Ok(None) => {}
            Err(e) => tracing::debug!(error = %e, "toktier content lookup declined"),
        }
        let encoding = self.tok.encode(text)?;
        bump(&self.counters.full);
        Ok(encoding.ids().to_vec())
    }

    /// The ids of `text` as the transcript of session `key`.
    fn encode_session(&self, key: &str, text: &str) -> toktier::Result<Vec<u32>> {
        let mut session = match self.lock().idle.remove(key) {
            Some((session, _)) => session,
            // Also fails while another thread holds this key's session.
            None => self.tok.open_session(key)?,
        };
        let ids = self.encode_in(&mut session, text);
        self.park(key, session);
        ids
    }

    fn encode_in(&self, session: &mut toktier::Session, text: &str) -> toktier::Result<Vec<u32>> {
        let (encoding, route) = if session.revision().is_none() {
            (session.seed(text)?, &self.counters.session_seed)
        } else {
            match session.encode_transcript(text) {
                Ok(encoding) => (encoding, &self.counters.session_repair),
                // The prompt does not extend the stored transcript (edited
                // history, a new conversation under the same id): it replaces it.
                Err(e) if e.code() == toktier::ErrorCode::SessionStateMismatch => {
                    (session.overwrite(text)?, &self.counters.session_seed)
                }
                Err(e) => return Err(e),
            }
        };
        bump(route);
        Ok(encoding.ids().to_vec())
    }

    /// Return a session to the idle map, deleting the least recently used one
    /// beyond [`MAX_IDLE_SESSIONS`] (outside the map lock: delete takes toktier's).
    fn park(&self, key: &str, session: toktier::Session) {
        let evicted = {
            let mut sessions = self.lock();
            sessions.tick += 1;
            let tick = sessions.tick;
            sessions.idle.insert(key.to_owned(), (session, tick));
            if sessions.idle.len() <= MAX_IDLE_SESSIONS {
                return;
            }
            let oldest = sessions
                .idle
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(key, _)| key.clone());
            oldest.and_then(|key| sessions.idle.remove(&key))
        };
        if let Some((session, _)) = evicted {
            if let Err(e) = session.delete() {
                tracing::debug!(error = %e, "toktier session delete failed on eviction");
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, Sessions> {
        self.sessions.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(test)]
    fn idle_sessions(&self) -> usize {
        self.lock().idle.len()
    }
}

fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

impl Encoder for ToktierTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let key = SessionScope::current();
        self.encode_routed(input, key.as_deref())
    }

    /// Batches carry no session; each item is looked up by content.
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode_routed(input, None))
            .collect()
    }
}

impl Decoder for ToktierTokenizer {
    fn has_unstable_suffix(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> bool {
        self.inner
            .has_unstable_suffix(token_ids, skip_special_tokens)
    }

    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for ToktierTokenizer {
    /// toktier keeps its own transcripts; L1 would hold the same prefixes twice.
    fn validate_prefix_cache(&self) -> Result<()> {
        Err(Error::msg(
            "toktier keeps its own session transcripts; the L1 prefix cache does not stack on them",
        ))
    }

    fn vocab_size(&self) -> Option<usize> {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        self.inner.token_to_id(token)
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        self.inner.special_token_ids()
    }

    fn num_special_tokens_added(&self) -> Result<usize> {
        self.inner.num_special_tokens_added()
    }
}

/// Parity against the inner `tokenizers` crate on real `tokenizer.json` files.
///
/// The files are not in the repository: `SGL_ROUTER_TOKTIER_FIXTURES` names a
/// directory of `<family>*/tokenizer.json` snapshots (`qwen3_8b-*`,
/// `llama_3_1_8b-*`, `bert_cased-*`); a cell whose file is absent is skipped
/// with a note. The tiny-fixture cells always run.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{TokenizerBackend, TokenizerConfig};
    use crate::tokenizer::adapter::{encode, load_with};
    use crate::tokenizer::stats::{EncodeBackend, L1State, TokenizerStats};
    use dynamo_tokenizers::create_tokenizer_from_file;
    use std::path::PathBuf;

    /// `<prefix>*/tokenizer.json` under `SGL_ROUTER_TOKTIER_FIXTURES`, or
    /// `None` with a skip note.
    fn fixture(prefix: &str) -> Option<PathBuf> {
        let path = std::env::var_os("SGL_ROUTER_TOKTIER_FIXTURES")
            .and_then(|root| std::fs::read_dir(root).ok())
            .and_then(|entries| {
                entries.flatten().map(|entry| entry.path()).find(|dir| {
                    dir.file_name()
                        .is_some_and(|name| name.to_string_lossy().starts_with(prefix))
                })
            })
            .map(|dir| dir.join("tokenizer.json"))
            .filter(|path| path.is_file());
        if path.is_none() {
            eprintln!("skipped: no {prefix}*/tokenizer.json under SGL_ROUTER_TOKTIER_FIXTURES");
        }
        path
    }

    /// The backend under test and the HF reference, both from `path`.
    fn open(path: &Path) -> (ToktierTokenizer, Arc<dyn Tokenizer>) {
        let hf = create_tokenizer_from_file(path.to_str().unwrap()).unwrap();
        let counters = Arc::new(ToktierCounters::default());
        let backend = ToktierTokenizer::open(path, hf.clone(), &counters)
            .expect("the fixture is a toktier family");
        (backend, hf)
    }

    /// The toktier backend as the router builds it from `path`.
    fn load(
        path: &str,
        l1_cache_mb: usize,
    ) -> (Arc<dynamo_tokenizers::Tokenizer>, Arc<TokenizerStats>) {
        let config = TokenizerConfig {
            backend: TokenizerBackend::Toktier,
            l1_cache_mb,
        };
        load_with(path, config).unwrap()
    }

    fn ids(t: &dyn Tokenizer, text: &str) -> Vec<u32> {
        t.encode(text).unwrap().token_ids().to_vec()
    }

    fn sample_texts() -> Vec<String> {
        let paragraph =
            "The quick brown fox jumps over the lazy dog. 🦊 Ünïcödé, 日本語のテキスト, \
                         and some code: fn main() { println!(\"{}\", 42); }\n\n";
        vec![
            "hello world".into(),
            "   leading and trailing spaces   \n\n\t tabs".into(),
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
                .into(),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>".into(),
            "numbers 1234567890 and 3.14159, mixed CamelCase andsnake_case".into(),
            "日本語と中文と한국어が混ざった文章。".into(),
            paragraph.repeat(1500),
        ]
    }

    /// Turn `n` of a growing chat, each turn extending the previous rendered prompt.
    fn transcript(turns: usize) -> String {
        let mut text = String::from(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\
             Hello there!<|im_end|>\n<|im_start|>assistant\n",
        );
        for turn in 1..turns {
            text.push_str(&format!(
                "Reply {turn}: ünïcödé {} and code `let x = {turn};`<|im_end|>\n\
                 <|im_start|>user\nQuestion {turn}: tell me more about 東京 and tokens.<|im_end|>\n\
                 <|im_start|>assistant\n",
                "é".repeat(turn)
            ));
        }
        text
    }

    #[test]
    fn full_encodes_match_hf() {
        for prefix in ["qwen3_8b-", "llama_3_1_8b-"] {
            let Some(path) = fixture(prefix) else {
                continue;
            };
            let (backend, hf) = open(&path);
            for text in sample_texts() {
                assert_eq!(
                    ids(&backend, &text),
                    ids(hf.as_ref(), &text),
                    "{prefix}: {text:.40}"
                );
            }
            let batch = ["a b c", "日本語", "<|im_start|>x"];
            let got: Vec<Vec<u32>> = backend
                .encode_batch(&batch)
                .unwrap()
                .iter()
                .map(|e| e.token_ids().to_vec())
                .collect();
            let want: Vec<Vec<u32>> = batch.iter().map(|t| ids(hf.as_ref(), t)).collect();
            assert_eq!(got, want);
            let (repair, seed, lookup, full, fallback) = backend.counters.routes();
            assert_eq!((repair, seed, lookup, fallback), (0, 0, 0, 0));
            assert!(full > 0, "unkeyed encodes take the full route");
        }
    }

    #[test]
    fn session_append_matches_hf_from_scratch() {
        let Some(path) = fixture("qwen3_8b-") else {
            return;
        };
        let (backend, hf) = open(&path);
        let turns = 20;
        for turn in 1..=turns {
            let text = transcript(turn);
            let _scope = SessionScope::bind(Some("session-a"));
            assert_eq!(ids(&backend, &text), ids(hf.as_ref(), &text), "turn {turn}");
        }
        assert_eq!(backend.counters.routes(), (turns as u64 - 1, 1, 0, 0, 0));

        // A prompt that does not extend the transcript replaces it, still exactly.
        let other =
            "<|im_start|>user\nA different conversation.<|im_end|>\n<|im_start|>assistant\n";
        let shorter = transcript(3);
        for text in [other, shorter.as_str()] {
            let _scope = SessionScope::bind(Some("session-a"));
            assert_eq!(ids(&backend, text), ids(hf.as_ref(), text));
        }
        assert_eq!(
            backend.counters.routes().1,
            3,
            "two overwrites after the seed"
        );
        assert_eq!(backend.idle_sessions(), 1);
    }

    #[test]
    fn unkeyed_content_lookup_matches_hf() {
        let Some(path) = fixture("qwen3_8b-") else {
            return;
        };
        let (backend, hf) = open(&path);
        // Long enough for toktier to seal content blocks (4096 characters each).
        let stored = transcript(60);
        assert!(stored.chars().count() > 2 * 4096);
        {
            let _scope = SessionScope::bind(Some("session-b"));
            assert_eq!(ids(&backend, &stored), ids(hf.as_ref(), &stored));
        }
        let extended = format!("{stored}Sure, here is more.<|im_end|>\n");
        assert_eq!(ids(&backend, &extended), ids(hf.as_ref(), &extended));
        assert_eq!(ids(&backend, &stored), ids(hf.as_ref(), &stored));
        let routes = backend.counters.routes();
        assert!(
            routes.2 >= 1,
            "the stored transcript is found by content: {routes:?}"
        );
    }

    #[test]
    fn declined_session_route_keeps_ids_exact() {
        let Some(path) = fixture("qwen3_8b-") else {
            return;
        };
        let (backend, hf) = open(&path);
        // toktier limits session names to 1024 bytes; a longer id is declined
        // and the prompt encodes without a session.
        let long_key = "k".repeat(2048);
        let text = transcript(2);
        let _scope = SessionScope::bind(Some(&long_key));
        assert_eq!(ids(&backend, &text), ids(hf.as_ref(), &text));
        let (repair, seed, _, full, _) = backend.counters.routes();
        assert_eq!((repair, seed, full), (0, 0, 1));
        assert_eq!(backend.idle_sessions(), 0);
    }

    #[test]
    fn concurrent_sessions_stay_exact() {
        let Some(path) = fixture("qwen3_8b-") else {
            return;
        };
        let (backend, hf) = open(&path);
        let turns: Vec<String> = (1..=6).map(transcript).collect();
        let expected: Vec<Vec<u32>> = turns.iter().map(|t| ids(hf.as_ref(), t)).collect();
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let (backend, turns, expected) = (&backend, &turns, &expected);
                scope.spawn(move || {
                    // Six workers share two keys, so some encodes find the
                    // session taken and go without it; two run unkeyed.
                    let key = match worker % 4 {
                        0..=2 => Some(format!("shared-{}", worker % 2)),
                        _ => None,
                    };
                    for (text, want) in turns.iter().zip(expected) {
                        let _scope = SessionScope::bind(key.as_deref());
                        assert_eq!(&ids(backend, text), want, "worker {worker}");
                    }
                });
            }
        });
        let (repair, seed, lookup, full, fallback) = backend.counters.routes();
        assert_eq!(repair + seed + lookup + full + fallback, 48);
        assert_eq!(fallback, 0);
    }

    #[test]
    fn foreign_tokenizer_passes_through_to_hf() {
        let Some(path) = fixture("bert_cased-") else {
            return;
        };
        let path = path.to_str().unwrap();
        let (tokenizer, stats) = load(path, 0);
        assert_eq!(stats.backend(), EncodeBackend::ToktierFallbackHf);
        let hf = create_tokenizer_from_file(path).unwrap();
        for text in sample_texts() {
            assert_eq!(encode(&tokenizer, &text).unwrap(), ids(hf.as_ref(), &text));
        }
    }

    #[test]
    fn tiny_fixture_passes_through_to_hf() {
        let path = "tests/fixtures/tiny_tokenizer.json";
        let (tokenizer, stats) = load(path, 0);
        assert_eq!(stats.backend(), EncodeBackend::ToktierFallbackHf);
        let hf = create_tokenizer_from_file(path).unwrap();
        assert_eq!(
            encode(&tokenizer, "hello 😀").unwrap(),
            ids(hf.as_ref(), "hello 😀")
        );
        assert_eq!(stats.toktier_routes(), (0, 0, 0, 0, 0));
    }

    #[test]
    fn l1_cache_is_inert_on_toktier() {
        let Some(path) = fixture("qwen3_8b-") else {
            return;
        };
        let path = path.to_str().unwrap();
        let (tokenizer, stats) = load(path, 64);
        assert_eq!(stats.backend(), EncodeBackend::Toktier);
        assert_eq!(stats.l1_state(), L1State::DisabledByBackend);
        let hf = create_tokenizer_from_file(path).unwrap();
        let text = transcript(3);
        assert_eq!(encode(&tokenizer, &text).unwrap(), ids(hf.as_ref(), &text));
        assert_eq!(stats.l1_tokens(), (0, 0));
        assert_eq!(stats.toktier_routes().3, 1);
    }

    #[test]
    fn session_scope_nests_and_restores() {
        assert_eq!(SessionScope::current(), None);
        {
            let _outer = SessionScope::bind(Some("outer"));
            assert_eq!(SessionScope::current().as_deref(), Some("outer"));
            {
                let _inner = SessionScope::bind(None);
                assert_eq!(SessionScope::current(), None);
            }
            assert_eq!(SessionScope::current().as_deref(), Some("outer"));
        }
        assert_eq!(SessionScope::current(), None);
    }
}
