// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;
use crate::proxy::Proxy;
use crate::tokenizer::TokenizerRegistry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct AppContext {
    pub config: Config,
    pub tokenizers: Arc<TokenizerRegistry>,
    pub proxy: Arc<Proxy>,
    ready: AtomicBool,
}

// `dynamo_tokenizers::Tokenizer` is not Debug, so we implement it manually.
impl std::fmt::Debug for AppContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppContext")
            .field("ready", &self.ready.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl AppContext {
    pub fn new(config: Config, tokenizers: Arc<TokenizerRegistry>, proxy: Arc<Proxy>) -> Self {
        Self {
            config,
            tokenizers,
            proxy,
            ready: AtomicBool::new(false),
        }
    }

    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::SeqCst);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub fn stub() -> Self {
        Self {
            config: Config {
                server: crate::config::ServerConfig {
                    host: "x".into(),
                    port: 0,
                },
                observability: Default::default(),
                models: vec![],
                worker: crate::config::WorkerConfig {
                    url: "http://x".into(),
                },
            },
            tokenizers: Arc::new(TokenizerRegistry::default()),
            proxy: Arc::new(Proxy::new("http://x".into())),
            ready: AtomicBool::new(false),
        }
    }
}
