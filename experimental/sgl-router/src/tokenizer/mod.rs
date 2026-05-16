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
                url: "http://x".into(),
                request_timeout_ms: None,
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
        // gpt2 reproduces "hello world" exactly
        assert_eq!(text, "hello world");
    }

    #[test]
    fn missing_file_errors() {
        let mut c = cfg();
        c.models[0].tokenizer_path = "/nonexistent.json".into();
        let err = TokenizerRegistry::load_from_config(&c).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("tokenizer"));
    }
}
