// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod adapter;
pub mod chat_template;
pub mod dsv4;

use anyhow::Result;
use chat_template::ChatTemplate;
use dashmap::DashMap;
use dynamo_tokenizers::Tokenizer;
use std::sync::atomic::{AtomicBool, Ordering};
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

#[derive(Default)]
pub struct TokenizerRegistry {
    inner: DashMap<String, Arc<Tokenizer>>,
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
        let t = adapter::load(&m.tokenizer_path)?;
        me.inner.insert(m.id.clone(), t);
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

    pub fn get(&self, model_id: &str) -> Option<Arc<Tokenizer>> {
        self.inner.get(model_id).map(|r| Arc::clone(&*r))
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
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
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
        }
    }

    #[test]
    fn loads_from_config() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        assert!(r.get("tiny").is_some());
        assert!(r.get("missing").is_none());
    }

    #[test]
    fn shared_arc_per_model() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let a = r.get("tiny").unwrap();
        let b = r.get("tiny").unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "registry should return shared Arc, not clones"
        );
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
            adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap(),
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
            adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap(),
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
            adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap(),
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
}
