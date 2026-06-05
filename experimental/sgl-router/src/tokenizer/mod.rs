// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod adapter;
pub mod chat_template;

use anyhow::Result;
use chat_template::ChatTemplate;
use dashmap::DashMap;
use dynamo_tokenizers::Tokenizer;
use std::sync::Arc;

#[derive(Default)]
pub struct TokenizerRegistry {
    inner: DashMap<String, Arc<Tokenizer>>,
    /// Per-model chat template, present only when the model's
    /// `tokenizer_config.json` ships a `chat_template`. Cache-aware routing
    /// uses it to tokenize chat requests the way the engine does; models
    /// without one fall back to raw prompt-text tokenization.
    templates: DashMap<String, Arc<ChatTemplate>>,
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
        // Best-effort: a model without a chat template (or without a readable
        // tokenizer_config.json) simply routes chat traffic via raw text.
        match adapter::load_tokenizer_config(&m.tokenizer_path) {
            Ok(Some(cfg_json)) => match ChatTemplate::from_tokenizer_config(&cfg_json) {
                Ok(Some(tmpl)) => {
                    me.templates.insert(m.id.clone(), Arc::new(tmpl));
                }
                Ok(None) => {}
                Err(e) => tracing::warn!(model = %m.id, error = %e,
                    "failed to compile chat template; chat traffic for this model routes via raw prompt text"),
            },
            Ok(None) => {}
            Err(e) => tracing::warn!(model = %m.id, error = %e,
                "failed to load tokenizer_config.json; chat traffic for this model routes via raw prompt text"),
        }
        Ok(me)
    }

    pub fn get(&self, model_id: &str) -> Option<Arc<Tokenizer>> {
        self.inner.get(model_id).map(|r| Arc::clone(&*r))
    }

    /// Whether this model has a chat template (and thus the chat-aware
    /// tokenization path is available for it).
    pub fn has_chat_template(&self, model_id: &str) -> bool {
        self.templates.contains_key(model_id)
    }

    /// Render `messages` through the model's chat template, then tokenize the
    /// result the same way the engine does (`add_special_tokens = false`, so the
    /// template's literal `bos_token`/role markers carry the specials). Returns
    /// `None` — caller falls back to raw routing — when the model has no
    /// template, no tokenizer, or rendering/encoding produced no tokens.
    pub fn encode_chat(&self, model_id: &str, messages: &serde_json::Value) -> Option<Vec<u32>> {
        let template = self.templates.get(model_id)?;
        let tokenizer = self.get(model_id)?;
        let rendered = template
            .render(messages)
            .inspect_err(|e| {
                // `?e` prints the full anyhow chain, so the underlying minijinja
                // cause (e.g. a `raise_exception` message) is visible, not just
                // the "render chat template" context.
                tracing::debug!(model = %model_id, error = ?e,
                    "chat-template render failed; falling back to raw prompt text")
            })
            .ok()?;
        match adapter::encode(&tokenizer, &rendered) {
            Ok(ids) if !ids.is_empty() => Some(ids),
            Ok(_) => None,
            Err(e) => {
                tracing::debug!(model = %model_id, error = %e,
                    "chat-template tokenize failed; falling back to raw prompt text");
                None
            }
        }
    }

    pub fn ids(&self) -> Vec<String> {
        self.inner.iter().map(|kv| kv.key().clone()).collect()
    }

    /// Attach a chat template (built from an inline `tokenizer_config.json`
    /// value) to an already-loaded model. Lets policy tests in other modules
    /// exercise the chat-template routing path without a co-located fixture.
    #[cfg(test)]
    pub(crate) fn attach_chat_template_for_test(
        &self,
        model_id: &str,
        tokenizer_config: &serde_json::Value,
    ) {
        let template = ChatTemplate::from_tokenizer_config(tokenizer_config)
            .expect("valid test chat template")
            .expect("test tokenizer_config has a chat_template");
        self.templates
            .insert(model_id.to_string(), Arc::new(template));
    }
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
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
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
    /// Strategy A: the fixture is a GPT-2 byte-level BPE. The 4-byte UTF-8
    /// emoji `😀` (`\xF0\x9F\x98\x80`) encodes into 2 byte-level BPE tokens
    /// with this fixture: `[47249, 222]`. Decoding just the first token
    /// yields a leading-bytes-only prefix that the HF adapter passes through
    /// `String::from_utf8_lossy`, producing a trailing U+FFFD. dynamo's
    /// `DecodeResult::from_decoded` then classifies that as `Partial`.
    /// Pinning the literal token id keeps the test deterministic — if the
    /// fixture or upstream BPE merges ever shift, this fails loudly rather
    /// than silently dropping back into `Complete` and losing coverage.
    #[test]
    fn decode_complete_returns_string_on_partial_utf8() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let t = r.get("tiny").unwrap();

        // Sanity-check that the fixture still tokenises `😀` the way we
        // expect; if upstream changes this we want a loud failure here.
        let full = adapter::encode(&t, "😀").unwrap();
        assert_eq!(
            full,
            vec![47249, 222],
            "fixture tokenisation drift: '😀' no longer encodes to [47249, 222]"
        );

        // Feed only the first token — its bytes are the leading 3 of a
        // 4-byte UTF-8 codepoint, which is incomplete.
        let s = adapter::decode_complete(&t, &full[..1], false).unwrap();

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
        assert!(reg.has_chat_template("tiny"));

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
            .templates
            .get("tiny")
            .unwrap()
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
        assert!(!reg.has_chat_template("tiny"));
        let messages = serde_json::json!([{"role":"user","content":"hi"}]);
        assert!(reg.encode_chat("tiny", &messages).is_none());
    }
}
