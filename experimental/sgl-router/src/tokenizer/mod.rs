// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod adapter;
pub mod chat_template;
pub mod dsv4;

use anyhow::Result;
use chat_template::ChatTemplate;
use dashmap::DashMap;
use dynamo_tokenizers::Tokenizer;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// How to turn a chat request's `messages` into the prompt the engine tokenizes
/// and caches. Cache-aware routing renders this before hashing so its query
/// tokens match the engine's stored blocks.
pub enum ChatEncoder {
    /// HuggingFace Jinja chat template from `tokenizer_config.json` (most
    /// models). Boxed: it holds a minijinja `Environment`, far larger than the
    /// other variants.
    Jinja(Box<ChatTemplate>),
    /// DeepSeek-V4 ships no template; the engine encodes in code. See [`dsv4`].
    DeepSeekV4,
}

impl ChatEncoder {
    /// Render `messages` into the engine-equivalent prompt text.
    fn render(&self, messages: &serde_json::Value) -> Result<String> {
        match self {
            ChatEncoder::Jinja(t) => t.render(messages),
            ChatEncoder::DeepSeekV4 => Ok(dsv4::render_messages(messages)),
        }
    }
}

/// A model's chat encoder plus its fallback-logging state.
struct ChatEncoderEntry {
    encoder: ChatEncoder,
    fallback_warned: AtomicBool,
}

impl ChatEncoderEntry {
    fn new(encoder: ChatEncoder) -> Self {
        Self {
            encoder,
            fallback_warned: AtomicBool::new(false),
        }
    }

    /// Log a per-request fallback to raw prompt-text hashing. "Enabled but
    /// failing every request" must be distinguishable from "healthy" at the
    /// default (info) log level — otherwise cache-aware overlap silently
    /// degrades to 0 with no signal — so the first failure for a model logs at
    /// warn; subsequent ones at debug to avoid a per-request log flood.
    fn log_fallback(&self, model_id: &str, cause: &str) {
        if !self.fallback_warned.swap(true, Ordering::Relaxed) {
            tracing::warn!(model = %model_id, %cause,
                "chat-encoder failed; falling back to raw prompt-text hashing \
                 (cache-aware overlap degrades for this model; further failures log at debug)");
        } else {
            tracing::debug!(model = %model_id, %cause,
                "chat-encoder failed; falling back to raw prompt-text hashing");
        }
    }
}

/// N independent `Tokenizer` instances for one model, selected round-robin.
///
/// WHY: `dynamo_tokenizers::Tokenizer` is `Arc<dyn traits::Tokenizer>`
/// internally; cloning it only clones the pointer, so every caller sharing
/// one instance also shares its underlying `BPE` model's word-merge cache —
/// a single `std::sync::RwLock<AHashMap<..>>` behind
/// `tokenizers::utils::cache::Cache`. Under concurrent encode() calls from
/// many tokio worker threads, that one lock becomes the bottleneck.
///
/// This is a second, complementary fix to the `dynamo-tokenizers` version
/// bump documented on that dependency in Cargo.toml (which cut this same
/// `Cache::get` frame's share of total process CPU from a measured 87% to a
/// measured 20%, on identical code/traffic — see that comment for the only
/// numbers in this repo that are independently checkable against a
/// committed artifact). A later live capture during the same investigation,
/// taken after that bump had already landed, broke the REMAINING `Cache::get`
/// time down further: ~25 of its points were specifically `RwLock::try_read`'s
/// CAS-retry loop plus `read_unlock`, not actual cache lookups — i.e. lock
/// overhead, not useful work, on that one shared lock. That capture wasn't
/// saved as a repo artifact, so treat "~25" as this investigation's own
/// finding, not a number a future reader can re-derive from anything
/// committed. Loading N instances from the same file gives each its own
/// independent cache/lock, so concurrent callers spread across N locks
/// instead of contending one. Every instance is loaded from the same source
/// via [`adapter::load`], so which shard a call lands on can never change
/// the tokenization output — only which lock it contends.
struct TokenizerShards {
    /// Always non-empty — the type's only constructors (`load`, and the
    /// `#[cfg(test)]`-only `single`) both guarantee at least one element, so
    /// `next()`'s `% self.shards.len()` can never divide by zero. Never
    /// resized after construction: no method here takes `&mut self`.
    shards: Vec<Arc<Tokenizer>>,
    /// Round-robin cursor, incremented per selection. Wraps via `%
    /// shards.len()`; overflow of the counter itself is harmless (wrapping
    /// add) since only its value modulo `shards.len()` is ever read.
    /// `&self` suffices in `next()` despite mutating this — interior
    /// mutability via `AtomicUsize`, no `&mut` needed (and `&mut self` would
    /// defeat the point: `TokenizerRegistry::get` is called concurrently
    /// from every tokio worker thread, which is exactly the concurrency this
    /// type exists to spread out, not serialize on).
    next: AtomicUsize,
}

impl TokenizerShards {
    /// Load `n` independent tokenizer instances from `source`. `n` is
    /// clamped to at least 1 so a misconfigured 0 can't build an empty,
    /// unusable registry entry.
    ///
    /// Sequential by design — do not parallelize this loop. For an HF
    /// repo-id `source`, `adapter::load`'s first call downloads and
    /// populates `hf-hub`'s on-disk cache; every subsequent call in this
    /// same sequential loop then hits that cache with no network at all
    /// (`hf_hub::api::sync::ApiRepo::get` checks the cache before ever
    /// calling out). Running these N calls concurrently instead would
    /// reintroduce a real race: `ApiRepo::get`'s cache check happens before
    /// its own download lock is taken, so N concurrent first-callers could
    /// each decide "not cached" and each kick off their own download
    /// attempt, serialized only by `hf-hub`'s file lock rather than
    /// deduplicated — turning "1 download + (N-1) free cache reads" back
    /// into up to N downloads racing each other.
    fn load(source: &str, n: usize) -> Result<Self> {
        let n = n.max(1);
        let shards = (0..n)
            .map(|_| adapter::load(source))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shards,
            next: AtomicUsize::new(0),
        })
    }

    /// Pick the next shard round-robin. `Relaxed` is sufficient for this
    /// counter specifically because `shards` itself needs no ordering to
    /// piggyback on: it's fully built in `load` before a `TokenizerShards`
    /// is ever shared (`load_from_config` only inserts it into the registry
    /// `DashMap` afterward), so the `DashMap` insert/lookup that publishes
    /// this struct to other threads already provides the only cross-thread
    /// visibility this type needs — `shards.len()` and its backing
    /// allocation are immutable for the struct's entire life, never
    /// requiring synchronization of their own. `Relaxed` on `fetch_add` only
    /// has to pick a valid index into that fixed, already-visible `Vec`, and
    /// every shard is behaviorally identical (see the type-level doc
    /// comment), so any interleaving of concurrent `fetch_add`s yields a
    /// valid, if not perfectly even, distribution.
    fn next(&self) -> Arc<Tokenizer> {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.shards.len();
        Arc::clone(&self.shards[i])
    }

    /// Wrap a single already-loaded instance as a one-shard set. Lets tests
    /// outside this module that build a `TokenizerRegistry` by hand (via
    /// `attach_chat_encoder_for_test`'s siblings) keep constructing it from a
    /// single `adapter::load` call.
    #[cfg(test)]
    fn single(t: Arc<Tokenizer>) -> Self {
        Self {
            shards: vec![t],
            next: AtomicUsize::new(0),
        }
    }
}

#[derive(Default)]
pub struct TokenizerRegistry {
    inner: DashMap<String, TokenizerShards>,
    /// Per-model chat encoder, present only when the model's prompt format is
    /// known (a `tokenizer_config.json` chat template, or a built-in encoder
    /// like DeepSeek-V4's). Cache-aware routing uses it to tokenize chat
    /// requests the way the engine does; models without one fall back to raw
    /// prompt-text tokenization.
    encoders: DashMap<String, Arc<ChatEncoderEntry>>,
}

impl std::fmt::Debug for TokenizerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerRegistry")
            .field("models", &self.ids())
            .finish()
    }
}

impl TokenizerRegistry {
    pub fn load_from_config(cfg: &crate::config::Config) -> Result<Self> {
        let me = TokenizerRegistry::default();
        let m = &cfg.model;
        let shards = TokenizerShards::load(&m.tokenizer_path, m.tokenizer_shards)?;
        me.inner.insert(m.id.clone(), shards);
        // Resolve the chat encoder, best-effort: a Jinja template from
        // tokenizer_config.json, else a built-in encoder for a recognized model
        // (DeepSeek-V4), else none (chat traffic routes via raw text). Every
        // path logs its outcome — whether chat-aware routing is live for this
        // model is the single most useful signal for diagnosing "cache-aware
        // routing degraded to overlap=0 on chat traffic", so it must never be
        // silent.
        if let Some(encoder) = me.resolve_chat_encoder(&m.id, &m.tokenizer_path) {
            me.encoders
                .insert(m.id.clone(), Arc::new(ChatEncoderEntry::new(encoder)));
        }
        Ok(me)
    }

    /// Pick the chat encoder for a model, logging the outcome on every branch.
    fn resolve_chat_encoder(&self, model_id: &str, tokenizer_path: &str) -> Option<ChatEncoder> {
        match adapter::load_tokenizer_config(tokenizer_path) {
            Ok(Some(cfg_json)) => match ChatTemplate::from_tokenizer_config(&cfg_json) {
                Ok(Some(tmpl)) => {
                    tracing::info!(model = %model_id,
                        "chat-template routing enabled; chat requests route by templated tokens");
                    return Some(ChatEncoder::Jinja(Box::new(tmpl)));
                }
                Ok(None) => {} // no template — fall through to built-in detection
                Err(e) => tracing::warn!(model = %model_id, error = %e,
                    "failed to compile chat template; falling back to built-in detection"),
            },
            Ok(None) => {}
            Err(e) => tracing::warn!(model = %model_id, error = %e,
                "failed to load tokenizer_config.json; falling back to built-in detection"),
        }
        if is_deepseek_v4(model_id) {
            tracing::info!(model = %model_id,
                "DeepSeek-V4 routing enabled; chat requests route via the built-in V4 encoder");
            return Some(ChatEncoder::DeepSeekV4);
        }
        tracing::info!(model = %model_id,
            "no chat template or built-in encoder; chat traffic routes via raw prompt text");
        None
    }

    /// Return one of this model's tokenizer shards, round-robin. Every call
    /// may return a *different* `Arc<Tokenizer>` instance than the previous
    /// one for the same model — callers must not rely on pointer identity
    /// across calls (see [`TokenizerShards`] for why: spreading callers
    /// across N independently-locked instances is the whole point).
    pub fn get(&self, model_id: &str) -> Option<Arc<Tokenizer>> {
        self.inner.get(model_id).map(|r| r.next())
    }

    /// Whether this model has a chat encoder (and thus the chat-aware
    /// tokenization path is available for it).
    pub fn has_chat_encoder(&self, model_id: &str) -> bool {
        self.encoders.contains_key(model_id)
    }

    /// Render `messages` through the model's chat encoder, then tokenize the
    /// result the same way the engine does (`add_special_tokens = false`, so the
    /// encoder's literal `bos_token`/role markers carry the specials). Returns
    /// `None` — caller falls back to raw routing — when the model has no
    /// encoder, no tokenizer, or rendering/encoding fails or yields no tokens.
    pub fn encode_chat(&self, model_id: &str, messages: &serde_json::Value) -> Option<Vec<u32>> {
        // Clone the Arc and drop the DashMap guard before the CPU-bound
        // render+encode (mirrors `get`), so no shard read-lock is held across it.
        let entry = Arc::clone(&*self.encoders.get(model_id)?);
        let tokenizer = self.get(model_id)?;
        let rendered = entry
            .encoder
            .render(messages)
            .inspect_err(|e| {
                // `{e:#}` prints the full anyhow chain, so the underlying
                // minijinja cause (e.g. a `raise_exception` message) is
                // visible, not just the "render chat template" context.
                entry.log_fallback(model_id, &format!("render failed: {e:#}"))
            })
            .ok()?;
        match adapter::encode(&tokenizer, &rendered) {
            Ok(ids) if !ids.is_empty() => Some(ids),
            Ok(_) => {
                entry.log_fallback(model_id, "rendered prompt tokenized to zero tokens");
                None
            }
            Err(e) => {
                entry.log_fallback(model_id, &format!("tokenize failed: {e:#}"));
                None
            }
        }
    }

    pub fn ids(&self) -> Vec<String> {
        self.inner.iter().map(|kv| kv.key().clone()).collect()
    }

    /// Attach a chat encoder to an already-loaded model. Lets policy tests in
    /// other modules exercise the chat-aware routing path without a co-located
    /// fixture.
    #[cfg(test)]
    pub(crate) fn attach_chat_encoder_for_test(&self, model_id: &str, encoder: ChatEncoder) {
        self.encoders.insert(
            model_id.to_string(),
            Arc::new(ChatEncoderEntry::new(encoder)),
        );
    }

    /// Convenience: attach a Jinja chat encoder built from an inline
    /// `tokenizer_config.json` value.
    #[cfg(test)]
    pub(crate) fn attach_chat_template_for_test(
        &self,
        model_id: &str,
        tokenizer_config: &serde_json::Value,
    ) {
        let template = ChatTemplate::from_tokenizer_config(tokenizer_config)
            .expect("valid test chat template")
            .expect("test tokenizer_config has a chat_template");
        self.attach_chat_encoder_for_test(model_id, ChatEncoder::Jinja(Box::new(template)));
    }
}

/// Whether `model_id` denotes a DeepSeek-V4 model, which the engine encodes via
/// the built-in [`dsv4`] encoder rather than a Jinja template. Heuristic on the
/// served model id (the router has no model architecture from `/server_info`);
/// scoped to "deepseek" + "v4" so it doesn't claim V3-family models, whose
/// encoding differs.
fn is_deepseek_v4(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    id.contains("deepseek") && id.contains("v4")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::config::PolicyKind;

    fn cfg() -> crate::config::Config {
        crate::config::Config {
            server: crate::config::ServerConfig {
                host: "0".into(),
                port: 0,
                ..Default::default()
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                tokenizer_shards: 1,
                policy: PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
            admission: crate::config::AdmissionConfig::default(),
            retry: crate::config::RetryConfig::default(),
        }
    }

    #[test]
    fn loads_from_config() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        assert!(r.get("tiny").is_some());
        assert!(r.get("missing").is_none());
    }

    /// With `tokenizer_shards = 1` (the default `cfg()` fixture), round-robin
    /// selection over a single-element shard vec always returns the same
    /// instance, so this preserves the pre-sharding "one shared Arc per
    /// model" behavior for models that don't opt into multiple shards.
    #[test]
    fn shared_arc_per_model_with_one_shard() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let a = r.get("tiny").unwrap();
        let b = r.get("tiny").unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "with a single shard, registry should return the same Arc every call"
        );
    }

    /// With `tokenizer_shards = N > 1`, `get` round-robins across N distinct
    /// `Arc<Tokenizer>` instances rather than always returning the same one.
    #[test]
    fn get_round_robins_across_shards() {
        let mut c = cfg();
        c.model.tokenizer_shards = 4;
        let r = TokenizerRegistry::load_from_config(&c).unwrap();

        let picks: Vec<Arc<Tokenizer>> = (0..8).map(|_| r.get("tiny").unwrap()).collect();

        // Exactly 4 distinct underlying instances, cycling with period 4.
        let distinct: std::collections::HashSet<usize> =
            picks.iter().map(|a| Arc::as_ptr(a) as usize).collect();
        assert_eq!(distinct.len(), 4, "expected exactly 4 distinct shards");
        for i in 0..4 {
            assert!(
                Arc::ptr_eq(&picks[i], &picks[i + 4]),
                "selection should cycle with period == shard count"
            );
        }
    }

    #[test]
    fn decode_complete_preserves_round_trip() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let t = r.get("tiny").unwrap();
        let ids = adapter::encode(&t, "hello world").unwrap();
        assert!(!ids.is_empty());
        let text = adapter::decode_complete(&t, &ids, true).unwrap();
        // tiny BPE fixture is byte-level and lossless for ASCII.
        assert_eq!(text, "hello world");
    }

    /// Forces `decode_complete` through its `DecodeResult::Partial` branch.
    ///
    /// The fixture is a no-merge byte-level BPE. The 4-byte UTF-8 emoji
    /// `😀` (`\xF0\x9F\x98\x80`) encodes to its raw byte token ids:
    /// `[240, 159, 152, 128]`. Decoding only a prefix yields leading bytes
    /// that the HF adapter passes through `String::from_utf8_lossy`,
    /// producing a trailing U+FFFD. dynamo's `DecodeResult::from_decoded`
    /// then classifies that as `Partial`.
    ///
    /// Pinning the literal token ids keeps the test deterministic: if the
    /// fixture shape or upstream byte-level handling ever shifts, this fails
    /// loudly rather than silently dropping back into `Complete` and losing
    /// coverage.
    #[test]
    fn decode_complete_returns_string_on_partial_utf8() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let t = r.get("tiny").unwrap();

        // Sanity-check that the fixture still tokenises `😀` the way we
        // expect; if upstream changes this we want a loud failure here.
        let full = adapter::encode(&t, "😀").unwrap();
        assert_eq!(
            full,
            vec![240, 159, 152, 128],
            "fixture tokenisation drift: '😀' no longer encodes to [240, 159, 152, 128]"
        );

        // Feed only the first three bytes of a 4-byte UTF-8 codepoint,
        // which is incomplete.
        let s = adapter::decode_complete(&t, &full[..3], false).unwrap();

        // We pin the exact output: the lossy decoder folds the 3 leading
        // bytes into a single U+FFFD. Anything else (empty string, Err, or
        // the original bytes) would be a regression.
        assert_eq!(s, "\u{FFFD}");
    }

    /// Concurrent encode against one shared `Arc<Tokenizer>`. Pins that the
    /// registry's `Arc<Tokenizer>` is `Send + Sync` and that
    /// `dynamo_tokenizers::Tokenizer::encode` can be called concurrently
    /// without interior mutability hazards. A regression that wraps
    /// `Tokenizer` in `RefCell` / `!Sync` data would fail to compile;
    /// a regression that introduces non-thread-safe internal caches
    /// would surface as one of the tasks returning wrong ids (caught by
    /// the per-task assertion against the sequentially-computed
    /// reference).
    ///
    /// Uses a multi-thread runtime + `JoinSet` so the 10 tasks really do
    /// run in parallel on distinct worker threads — a single-thread
    /// runtime wouldn't exercise the `Sync` contract.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tokenizer_supports_concurrent_encode() {
        use tokio::task::JoinSet;

        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let t = r.get("tiny").unwrap();

        // Build the reference sequentially — what each task should return.
        let inputs: Vec<String> = (0..10).map(|i| format!("hello {i}")).collect();
        let expected: Vec<Vec<u32>> = inputs
            .iter()
            .map(|s| adapter::encode(&t, s).unwrap())
            .collect();

        let mut set = JoinSet::new();
        for (i, text) in inputs.into_iter().enumerate() {
            let shared = Arc::clone(&t);
            set.spawn(async move {
                let ids = adapter::encode(&shared, &text).expect("concurrent encode must not fail");
                (i, ids)
            });
        }

        let mut got: Vec<Option<Vec<u32>>> = vec![None; expected.len()];
        while let Some(joined) = set.join_next().await {
            let (i, ids) = joined.expect("task panicked");
            got[i] = Some(ids);
        }

        for (i, ids) in got.into_iter().enumerate() {
            let ids = ids.unwrap_or_else(|| panic!("task {i} did not record a result"));
            assert_eq!(
                ids, expected[i],
                "concurrent encode produced wrong tokens for task {i}; \
                 sign of a non-thread-safe internal cache regression"
            );
        }
    }

    #[test]
    fn missing_file_errors() {
        let mut c = cfg();
        c.model.tokenizer_path = "/nonexistent.json".into();
        let err = TokenizerRegistry::load_from_config(&c).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("tokenizer"));
    }

    #[test]
    fn load_tokenizer_config_reads_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let tok = dir.path().join("tokenizer.json");
        std::fs::write(&tok, "{}").unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template":"X","bos_token":"<s>"}"#,
        )
        .unwrap();
        let cfg = adapter::load_tokenizer_config(tok.to_str().unwrap())
            .unwrap()
            .expect("sibling tokenizer_config.json is loaded");
        assert_eq!(cfg["chat_template"], "X");
    }

    #[test]
    fn load_tokenizer_config_absent_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let tok = dir.path().join("tokenizer.json");
        std::fs::write(&tok, "{}").unwrap();
        assert!(adapter::load_tokenizer_config(tok.to_str().unwrap())
            .unwrap()
            .is_none());
    }

    /// `encode_chat` renders the template then tokenizes the result — and that
    /// token sequence differs from tokenizing the raw message content (the very
    /// reason raw-content hashing missed the engine's chat-templated blocks).
    #[test]
    fn encode_chat_renders_then_tokenizes() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::single(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        let cfg = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
            "bos_token": "<s>",
        });
        reg.attach_chat_template_for_test("tiny", &cfg);
        assert!(reg.has_chat_encoder("tiny"));

        let messages = serde_json::json!([{"role":"user","content":"hi"}]);
        let chat_ids = reg.encode_chat("tiny", &messages).expect("encode_chat");
        assert!(!chat_ids.is_empty());

        let tok = reg.get("tiny").unwrap();
        let raw_ids = adapter::encode(&tok, "hi").unwrap();
        assert_ne!(
            chat_ids, raw_ids,
            "chat-templated tokens must differ from raw-content tokens"
        );

        // encode_chat is exactly tokenize(render(messages)).
        let rendered = reg
            .encoders
            .get("tiny")
            .unwrap()
            .encoder
            .render(&messages)
            .unwrap();
        assert_eq!(chat_ids, adapter::encode(&tok, &rendered).unwrap());
    }

    #[test]
    fn encode_chat_none_without_template() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::single(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        assert!(!reg.has_chat_encoder("tiny"));
        let messages = serde_json::json!([{"role":"user","content":"hi"}]);
        assert!(reg.encode_chat("tiny", &messages).is_none());
    }

    /// A template that fails to render (here, one that calls `raise_exception`)
    /// makes `encode_chat` return `None`, so the policy falls back to the raw
    /// prompt-text path rather than failing the request.
    #[test]
    fn encode_chat_none_on_render_failure() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::single(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        reg.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ raise_exception('nope') }}",
                "bos_token": "<s>",
            }),
        );
        assert!(reg.has_chat_encoder("tiny"));
        let messages = serde_json::json!([{"role":"user","content":"hi"}]);
        assert!(
            reg.encode_chat("tiny", &messages).is_none(),
            "a failing render must yield None so routing falls back to raw text"
        );
    }

    #[test]
    fn is_deepseek_v4_matches_v4_only() {
        assert!(is_deepseek_v4("deepseek-ai/DeepSeek-V4-Flash"));
        assert!(is_deepseek_v4("DeepSeek-V4-Pro"));
        // Not V4-family models.
        assert!(!is_deepseek_v4("deepseek-ai/DeepSeek-V3.2"));
        assert!(!is_deepseek_v4("Qwen/Qwen3-0.6B"));
        assert!(!is_deepseek_v4("tiny"));
    }

    /// Find a real, non-trivial `tokenizer.json` in the local HuggingFace
    /// cache, if any is present. Returns `None` (the test that calls this
    /// skips itself) rather than failing on machines/CI runners with no HF
    /// cache populated — this test's value is in exercising a large,
    /// real-world BPE vocab/merge table, not in requiring network access or a
    /// specific model to be cached.
    fn find_cached_real_tokenizer_json() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let hub = std::path::Path::new(&home).join(".cache/huggingface/hub");
        for entry in std::fs::read_dir(&hub).ok()?.flatten() {
            let snapshots = entry.path().join("snapshots");
            for snap in std::fs::read_dir(&snapshots).ok()?.flatten() {
                let candidate = snap.path().join("tokenizer.json");
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Sharding must never change tokenization output: N independently
    /// loaded instances of the same `tokenizer.json` are (by construction)
    /// identical apart from their private, output-invisible merge caches, but
    /// this pins that empirically rather than by argument alone — a
    /// regression here (e.g. a shared mutable default inside the BPE model,
    /// or a loader that isn't actually deterministic) would silently corrupt
    /// downstream cache-affinity hashing, which requires byte-for-byte
    /// identical token ids across shards.
    ///
    /// Prefers a real, large tokenizer.json from the local HF cache when one
    /// is present (extra real-world coverage in local dev), but the fallback
    /// is `tests/fixtures/tiny_bpe_tokenizer.json`, NOT the plain
    /// `tiny_tokenizer.json` fixture other tests in this file use —
    /// `tiny_tokenizer.json` has an empty `merges` array (pure byte-level
    /// vocab, no BPE merging at all), which would make this test check
    /// nothing about merge-cache divergence in exactly the CI environment
    /// (`ubuntu-latest`, no HF cache — see `pr-test-sgl-router.yml`) this
    /// test exists to protect: every run there would silently fall back and
    /// silently pass regardless of whether sharding actually preserves
    /// output. `tiny_bpe_tokenizer.json` is `tiny_tokenizer.json` plus four
    /// real merges (t+h, th+e, i+n, in+g), so `assert_merge_actually_fired`
    /// below can confirm real BPE merging ran, not just byte-level passthrough,
    /// on ANY machine.
    #[test]
    fn sharded_instances_produce_identical_output() {
        let source = find_cached_real_tokenizer_json()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "tests/fixtures/tiny_bpe_tokenizer.json".into());
        eprintln!("sharded_instances_produce_identical_output: using {source}");

        const N: usize = 8;
        let shards = TokenizerShards::load(&source, N).expect("load N independent instances");
        assert_eq!(shards.shards.len(), N);

        let inputs: &[&str] = &[
            "Hello, how are you today?",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
            "The quick brown fox jumps over the lazy dog near the riverbank while the sun sets slowly behind the distant mountains, casting long shadows across the valley below.",
            "你好，世界！这是一个测试。今天天气怎么样？",
            "SGLang is a fast serving framework for large language models and vision language models.",
            "aaaaaaaaaa bbbbbbbbbb aaaaaaaaaa bbbbbbbbbb aaaaaaaaaa",
            "",
            "🚀🔥💯 emoji stress test with mixed ASCII and 日本語 text!!!",
        ];

        let mut any_merge_fired = false;
        for text in inputs {
            let outputs: Vec<Vec<u32>> = shards
                .shards
                .iter()
                .map(|t| adapter::encode(t, text).expect("encode must succeed"))
                .collect();
            // A merge fired if the token count is less than the raw byte
            // count — every unmerged byte is its own token in this
            // byte-level vocab, so any reduction below `text.len()` (bytes,
            // not chars) proves at least one BPE merge actually ran.
            if !outputs[0].is_empty() && outputs[0].len() < text.len() {
                any_merge_fired = true;
            }
            for (i, ids_i) in outputs.iter().enumerate() {
                for (j, ids_j) in outputs.iter().enumerate() {
                    assert_eq!(
                        ids_i, ids_j,
                        "shard {i} and shard {j} produced different token ids for {text:?}: \
                         sharding must never change tokenization output"
                    );
                }
            }
        }
        assert!(
            any_merge_fired,
            "none of the test inputs triggered a single BPE merge against {source} — this test \
             would then be checking N identical byte-level passthroughs, not real shard-to-shard \
             merge-cache divergence, silently validating nothing. Use a tokenizer.json with a \
             non-empty merges table (see tests/fixtures/tiny_bpe_tokenizer.json)."
        );
    }
}
