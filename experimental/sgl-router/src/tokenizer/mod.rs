// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod adapter;

use anyhow::Result;
use dashmap::DashMap;
use dynamo_tokenizers::Tokenizer;
use std::sync::Arc;

#[derive(Default)]
pub struct TokenizerRegistry {
    inner: DashMap<String, Arc<Tokenizer>>,
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
        for m in &cfg.models {
            let t = adapter::load(&m.tokenizer_path)?;
            me.inner.insert(m.id.clone(), t);
        }
        Ok(me)
    }

    pub fn get(&self, model_id: &str) -> Option<Arc<Tokenizer>> {
        self.inner.get(model_id).map(|r| Arc::clone(&*r))
    }

    pub fn ids(&self) -> Vec<String> {
        self.inner.iter().map(|kv| kv.key().clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn cfg() -> crate::config::Config {
        crate::config::Config {
            server: crate::config::ServerConfig {
                host: "0".into(),
                port: 0,
            },
            observability: Default::default(),
            models: vec![crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            }],
            workers: vec![crate::config::WorkerConfig {
                url: reqwest::Url::parse("http://x:30000").unwrap(),
                request_timeout: None,
            }],
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

    #[test]
    fn missing_file_errors() {
        let mut c = cfg();
        c.models[0].tokenizer_path = "/nonexistent.json".into();
        let err = TokenizerRegistry::load_from_config(&c).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("tokenizer"));
    }
}
