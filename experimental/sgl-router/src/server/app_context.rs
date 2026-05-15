// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;
use crate::proxy::Proxy;
use crate::tokenizer::TokenizerRegistry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub struct AppContext {
    pub config: Config,
    pub tokenizers: Arc<TokenizerRegistry>,
    pub proxy: Arc<Proxy>,
    ready: AtomicBool,
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
        // Relaxed: this flag does not synchronize other state; readers only
        // care about eventual visibility, not happens-before with surrounding ops.
        self.ready.store(true, Ordering::Relaxed);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn stub() -> Self {
        let url = reqwest::Url::parse("http://x:30000").expect("stub url parses");
        Self {
            config: Config {
                server: crate::config::ServerConfig {
                    host: "x".into(),
                    port: 0,
                },
                observability: Default::default(),
                models: vec![],
                discovery: crate::config::DiscoveryConfig {
                    backend: crate::config::DiscoveryBackend::StaticFile(
                        crate::config::StaticFileDiscoveryConfig {
                            path: "/tmp/test-workers.toml".into(),
                            poll_interval_ms: 200,
                        },
                    ),
                },
            },
            tokenizers: Arc::new(TokenizerRegistry::default()),
            proxy: Arc::new(
                Proxy::new(url, std::time::Duration::from_secs(60)).expect("stub proxy"),
            ),
            ready: AtomicBool::new(false),
        }
    }
}
