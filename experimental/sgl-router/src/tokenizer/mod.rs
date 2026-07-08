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
/// with the same opts via [`adapter::load_with_opts`], so which shard a
/// call lands on can never change the tokenization output — only which lock
/// it contends.
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
    /// repo-id `source`, the first `adapter::load_with_opts` call downloads
    /// and populates `hf-hub`'s on-disk cache; every subsequent call in this
    /// same sequential loop then hits that cache with no network at all
    /// (`hf_hub::api::sync::ApiRepo::get` checks the cache before ever
    /// calling out). Running these N calls concurrently instead would
    /// reintroduce a real race: `ApiRepo::get`'s cache check happens before
    /// its own download lock is taken, so N concurrent first-callers could
    /// each decide "not cached" and each kick off their own download
    /// attempt, serialized only by `hf-hub`'s file lock rather than
    /// deduplicated — turning "1 download + (N-1) free cache reads" back
    /// into up to N downloads racing each other.
    fn load(source: &str, n: usize, opts: adapter::TokenizerLoadOpts) -> Result<Self> {
        // Settle the L1 half of the opts FIRST: a requested-but-inert cache
        // (tokenizer declares no safely-splittable specials) is downgraded
        // to budget 0 there, so it must not cost the shard spread below.
        let opts = adapter::finalize_load_opts(source, opts)?;
        // The L1 prefix cache lives INSIDE each CachedTokenizer instance, so
        // N shards would mean N independent caches: N× the byte budget and a
        // 1/N hit rate (a conversation's turn-2 request only hits if it lands
        // on the same shard as turn 1 — round-robin guarantees it usually
        // won't). One shared instance wins for the hit-dominated multi-turn
        // traffic the cache targets: a hit touches the BPE merge cache only
        // for the (short) fresh suffix, and with the fast backend misses
        // avoid that lock too. KNOWN TRADE-OFF: with the HF backend,
        // miss-heavy traffic (all-fresh conversations, boundary-less
        // prompts) funnels full encodes through the single instance's
        // merge-cache RwLock that sharding existed to spread — prefer
        // --tokenizer-backend fast alongside the cache.
        let n = if opts.l1_cache_bytes > 0 {
            if n > 1 {
                tracing::info!(
                    requested_shards = n,
                    "tokenizer L1 cache enabled; using a single shared tokenizer instance \
                     (per-shard caches would split the hit rate and multiply the byte budget)"
                );
            }
            1
        } else {
            n.max(1)
        };
        let shards = (0..n)
            .map(|_| adapter::load_with_opts(source, opts))
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
        let opts = adapter::TokenizerLoadOpts {
            backend: m.tokenizer_backend,
            l1_cache_bytes: m.tokenizer_l1_cache_mb.saturating_mul(1024 * 1024),
        };
        let shards = TokenizerShards::load(&m.tokenizer_path, m.tokenizer_shards, opts)?;
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
                tokenizer_backend: Default::default(),
                tokenizer_l1_cache_mb: 0,
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

    /// The wire from `--tokenizer-l1-cache-mb` to the feature: a nonzero
    /// MiB budget flowing through `load_from_config` must actually enable
    /// the cache — observable as the shard collapse (4 configured shards →
    /// one shared instance). If the MiB→bytes conversion regressed to 0,
    /// four distinct Arcs would come back and this fails.
    #[test]
    fn load_from_config_wires_l1_cache_budget() {
        let mut c = cfg();
        c.model.tokenizer_shards = 4;
        c.model.tokenizer_l1_cache_mb = 1;
        // The BPE fixture's <|endoftext|> special keeps the cache genuinely
        // active (finalize_load_opts would zero the budget on a
        // specials-less tokenizer and the shard collapse wouldn't happen).
        c.model.tokenizer_path = "tests/fixtures/tiny_bpe_tokenizer.json".into();
        let r = TokenizerRegistry::load_from_config(&c).unwrap();
        let a = r.get("tiny").unwrap();
        let b = r.get("tiny").unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "an active L1 cache must collapse to one shared tokenizer instance"
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
        let shards = TokenizerShards::load(&source, N, adapter::TokenizerLoadOpts::default())
            .expect("load N independent instances");
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

    /// A growing multi-turn-shaped conversation over the tiny BPE fixture,
    /// using its one special token (`<|endoftext|>`, id 256) as the turn
    /// boundary — the shape the L1 prefix cache exists for: element `k` is
    /// element `k-1` plus one boundary and one new "turn".
    fn growing_conversation(turns: usize) -> Vec<String> {
        let mut out = Vec::with_capacity(turns);
        let mut text = String::new();
        for t in 0..turns {
            text.push_str(&format!("the thing in turn {t} is interesting "));
            text.push_str("<|endoftext|>");
            out.push(text.clone());
        }
        out
    }

    /// The L1-cached tokenizer must be OUTPUT-INVISIBLE: byte-identical ids
    /// vs a plain (uncached) load, on cold encodes, warm re-encodes, AND the
    /// growing-conversation pattern where extend-on-hit builds ever-deeper
    /// cached prefixes. These ids feed cache-affinity hashing and are
    /// forwarded to the engine as `input_ids`, so any divergence is silent
    /// output corruption — equality here is the gate for enabling the cache.
    #[test]
    fn l1_cached_encode_matches_plain_encode() {
        let plain = adapter::load("tests/fixtures/tiny_bpe_tokenizer.json").unwrap();
        let cached = adapter::load_with_opts(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Hf,
                l1_cache_bytes: 8 * 1024 * 1024,
            },
        )
        .unwrap();

        for text in growing_conversation(8) {
            let want = adapter::encode(&plain, &text).unwrap();
            let cold = adapter::encode(&cached, &text).unwrap();
            assert_eq!(cold, want, "cold cached encode diverged for {text:?}");
            // Second encode of the same text re-runs the lookup; from the
            // 2-turn element on it hits at the deepest INTERIOR boundary (a
            // trailing special's end-of-text boundary is never cached, so
            // the 1-turn element stays a miss and hits are partial).
            let warm = adapter::encode(&cached, &text).unwrap();
            assert_eq!(warm, want, "warm cached encode diverged for {text:?}");
        }
        // Also texts with NO special-token boundary (cache can't help) and
        // suffixes AFTER a cached boundary that differ from what was cached.
        for text in [
            "no boundaries here at all",
            "the thing in turn 0 is interesting <|endoftext|>but a different suffix",
            "",
        ] {
            let want = adapter::encode(&plain, text).unwrap();
            let got = adapter::encode(&cached, text).unwrap();
            assert_eq!(got, want, "cached encode diverged for {text:?}");
        }
    }

    /// The fastokens backend must be encode-equivalent to the HF backend.
    /// The first assertion pins that fastokens genuinely LOADS the fixture:
    /// `load_with_opts` silently falls back to HF on a fastokens load
    /// failure, which would turn the equivalence assertions below into
    /// HF-vs-HF theater (this happened — the fixture originally lacked the
    /// `"type": "BPE"` model tag fastokens requires, and the test passed
    /// while exercising zero fastokens code).
    #[test]
    fn fast_backend_encode_matches_hf() {
        assert!(
            dynamo_tokenizers::FastTokenizer::from_file("tests/fixtures/tiny_bpe_tokenizer.json")
                .is_ok(),
            "fixture no longer loads under fastokens — the equivalence assertions below are \
             vacuous (load_with_opts silently falls back to HF). Fix the fixture or fastokens \
             pin before trusting this test."
        );
        let hf = adapter::load("tests/fixtures/tiny_bpe_tokenizer.json").unwrap();
        let fast = adapter::load_with_opts(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 0,
            },
        )
        .unwrap();
        let corpus = [
            "hello world",
            "the thing <|endoftext|> another thing",
            "你好，世界！ mixed 🚀 content",
            "",
            "aaaaaaaaaa the the the ing ing ing",
        ];
        for text in corpus {
            let want = adapter::encode(&hf, text).unwrap();
            let got = adapter::encode(&fast, text).unwrap();
            assert_eq!(
                got, want,
                "fast-backend encode diverged from HF for {text:?} — fastokens is NOT \
                 engine-equivalent for this tokenizer; do not deploy --tokenizer-backend fast"
            );
        }
    }

    /// The PRODUCTION target config — fast backend + L1 cache STACKED — must
    /// match the plain HF oracle. The two single-lever tests don't compose
    /// automatically (disjoint corpora, and the cache wraps a different
    /// inner), so the deploy config gets its own always-run equivalence.
    #[test]
    fn fast_l1_stacked_encode_matches_plain_encode() {
        let plain = adapter::load("tests/fixtures/tiny_bpe_tokenizer.json").unwrap();
        let stacked = adapter::load_with_opts(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 8 * 1024 * 1024,
            },
        )
        .unwrap();
        for text in growing_conversation(8) {
            let want = adapter::encode(&plain, &text).unwrap();
            assert_eq!(
                adapter::encode(&stacked, &text).unwrap(),
                want,
                "cold fast+L1 encode diverged for {text:?}"
            );
            assert_eq!(
                adapter::encode(&stacked, &text).unwrap(),
                want,
                "warm fast+L1 encode diverged for {text:?}"
            );
        }
    }

    /// Concurrent encodes through ONE shared L1-cached tokenizer — the
    /// production topology (the cache collapses shards to a single
    /// instance). Unlike the plain-path sibling test, every call here also
    /// MUTATES cache state (insert / extend-on-hit / eviction bookkeeping),
    /// so this is the only place a logic race in the cache's concurrent
    /// bookkeeping would surface as wrong ids. Growing-conversation inputs
    /// make extend-on-hit actually race.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cached_tokenizer_supports_concurrent_encode() {
        use tokio::task::JoinSet;

        let plain = adapter::load("tests/fixtures/tiny_bpe_tokenizer.json").unwrap();
        let cached = adapter::load_with_opts(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 8 * 1024 * 1024,
            },
        )
        .unwrap();

        // Sequential reference from the UNCACHED oracle; 4 interleaved
        // copies of each growing-conversation element so concurrent tasks
        // contend on the same prefixes.
        let mut inputs: Vec<String> = Vec::new();
        for text in growing_conversation(6) {
            for _ in 0..4 {
                inputs.push(text.clone());
            }
        }
        let expected: Vec<Vec<u32>> = inputs
            .iter()
            .map(|s| adapter::encode(&plain, s).unwrap())
            .collect();

        let mut set = JoinSet::new();
        for (i, text) in inputs.into_iter().enumerate() {
            let shared = Arc::clone(&cached);
            set.spawn(async move {
                let ids = adapter::encode(&shared, &text)
                    .expect("concurrent cached encode must not fail");
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
                "concurrent cached encode produced wrong tokens for task {i} — \
                 sign of a race in the shared L1 cache's insert/extend path"
            );
        }
    }

    /// An enabled L1 cache forces a single tokenizer instance — per-shard
    /// caches would split the hit rate across shards and multiply the byte
    /// budget (see `TokenizerShards::load`).
    #[test]
    fn l1_cache_forces_single_shard() {
        let shards = TokenizerShards::load(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            8,
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Hf,
                l1_cache_bytes: 1024 * 1024,
            },
        )
        .unwrap();
        assert_eq!(
            shards.shards.len(),
            1,
            "L1 cache enabled must collapse tokenizer shards to one shared instance"
        );
    }

    /// Real-model equivalence + timing for the DSV4 workload this prototype
    /// targets. Ignored by default: needs a real DeepSeek-V4 tokenizer.json,
    /// pointed at via env. Run with:
    ///
    ///   DSV4_TOKENIZER_JSON=/path/to/tokenizer.json \
    ///     cargo test --release -p sgl-router dsv4_real -- --ignored --nocapture
    ///
    /// Asserts (the deploy gates):
    ///   1. fastokens encode ids == HF encode ids on multi-turn DSV4-rendered
    ///      prompts (ids are forwarded to the engine — divergence is silent
    ///      wrong output);
    ///   2. L1-cached encode ids == plain ids at every turn of a growing
    ///      conversation;
    ///   3. the pinned engine-`/tokenize` vector from `dsv4.rs` holds through
    ///      every backend/wrapper combination.
    ///
    /// Prints cold/warm timings for HF, fast, and fast+L1 so the run doubles
    /// as the prototype benchmark.
    #[test]
    #[ignore = "needs DSV4_TOKENIZER_JSON pointing at a real DeepSeek-V4 tokenizer.json"]
    fn dsv4_real_tokenizer_equivalence_and_timing() {
        let Ok(path) = std::env::var("DSV4_TOKENIZER_JSON") else {
            panic!("set DSV4_TOKENIZER_JSON to a DeepSeek-V4 tokenizer.json path");
        };
        let hf = adapter::load(&path).unwrap();
        let fast = adapter::load_with_opts(
            &path,
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 0,
            },
        )
        .unwrap();
        let fast_l1 = adapter::load_with_opts(
            &path,
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 512 * 1024 * 1024,
            },
        )
        .unwrap();

        // Gate 3: the pinned engine-/tokenize vector (dsv4.rs module doc).
        let pinned = serde_json::json!([{"role": "user", "content": "ABCD"}]);
        let rendered = dsv4::render_messages(&pinned);
        for (name, tok) in [("hf", &hf), ("fast", &fast), ("fast+l1", &fast_l1)] {
            assert_eq!(
                adapter::encode(tok, &rendered).unwrap(),
                vec![0, 128803, 51453, 128804, 128822],
                "pinned DSV4 vector diverged through {name}"
            );
        }

        // Build a deterministic multi-turn conversation totalling ~70k tokens
        // when rendered — the shape and scale of the live workload.
        let mut messages: Vec<serde_json::Value> = Vec::new();
        let words = [
            "alpha", "beta", "gamma", "delta", "system", "router", "tensor", "kernel", "deploy",
            "metric", "latency", "bucket", "engine", "token", "prefix", "cache", "the", "of",
            "and", "request",
        ];
        let mut w = 0usize;
        for turn in 0..40 {
            let mut content = format!("turn {turn}: ");
            for _ in 0..1500 {
                content.push_str(words[w % words.len()]);
                content.push(' ');
                w = w.wrapping_mul(31).wrapping_add(17);
            }
            let role = if turn % 2 == 0 { "user" } else { "assistant" };
            messages.push(serde_json::json!({"role": role, "content": content}));
        }

        let time_encode = |tok: &Arc<Tokenizer>, text: &str| -> (Vec<u32>, f64) {
            let t0 = std::time::Instant::now();
            let ids = adapter::encode(tok, text).unwrap();
            (ids, t0.elapsed().as_secs_f64() * 1000.0)
        };

        // Gates 1 + 2 across a GROWING conversation (each iteration appends
        // one turn — the production shape), timing every layer per turn.
        eprintln!("turn | tokens | hf_ms | fast_ms | fast_l1_ms");
        for upto in (2..=messages.len()).step_by(8) {
            let msgs = serde_json::Value::Array(messages[..upto].to_vec());
            let text = dsv4::render_messages(&msgs);
            let (want, hf_ms) = time_encode(&hf, &text);
            let (got_fast, fast_ms) = time_encode(&fast, &text);
            let (got_l1, l1_ms) = time_encode(&fast_l1, &text);
            assert_eq!(got_fast, want, "fastokens diverged at {upto} turns");
            assert_eq!(got_l1, want, "fast+L1 diverged at {upto} turns");
            eprintln!(
                "{upto:4} | {:6} | {hf_ms:7.1} | {fast_ms:7.1} | {l1_ms:7.1}",
                want.len()
            );
        }
        let (hits, misses, cached_tok, encoded_tok) = adapter::l1_cache_counters();
        eprintln!(
            "l1 counters: hits={hits} misses={misses} cached_tokens={cached_tok} encoded_tokens={encoded_tok}"
        );
    }
}
