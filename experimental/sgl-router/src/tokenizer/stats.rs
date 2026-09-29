// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Resolved settings for startup logging and L1 token counters for `/metrics`.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Encode backend actually in use, which can differ from the requested `--tokenizer-backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodeBackend {
    #[default]
    Hf,
    Fast,
    /// `fast` was requested but fastokens could not load the tokenizer.
    FastFallbackHf,
    /// A tiktoken `.model` tokenizer, which has no fast backend.
    Tiktoken,
}

impl EncodeBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hf => "hf",
            Self::Fast => "fast",
            Self::FastFallbackHf => "fast_fallback_hf",
            Self::Tiktoken => "tiktoken",
        }
    }
}

/// L1 prefix-tokenization cache state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum L1State {
    #[default]
    Off,
    Active,
    /// The cache was requested but the tokenizer declares no splittable special tokens.
    DisabledNoSpecials,
}

impl L1State {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Active => "active",
            Self::DisabledNoSpecials => "disabled_no_specials",
        }
    }
}

/// Counters fed by the L1 cache observers.
#[derive(Debug, Default)]
pub(crate) struct L1Counters {
    pub cached_tokens: AtomicU64,
    pub encoded_tokens: AtomicU64,
}

#[derive(Debug, Default)]
pub struct TokenizerStats {
    pub(crate) backend: EncodeBackend,
    pub(crate) l1_state: L1State,
    pub(crate) l1: Arc<L1Counters>,
}

impl TokenizerStats {
    pub fn backend(&self) -> EncodeBackend {
        self.backend
    }

    pub fn l1_state(&self) -> L1State {
        self.l1_state
    }

    /// Tokens served by L1 as `(cached, freshly encoded)`.
    pub fn l1_tokens(&self) -> (u64, u64) {
        (
            self.l1.cached_tokens.load(Ordering::Relaxed),
            self.l1.encoded_tokens.load(Ordering::Relaxed),
        )
    }

    /// Prometheus exposition for the tokenizer series.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "# HELP sgl_router_tokenizer_l1_tokens_total Tokens produced by encodes through the L1 cache, split into prefix tokens served from cache and freshly encoded tokens.\n",
        );
        out.push_str("# TYPE sgl_router_tokenizer_l1_tokens_total counter\n");
        for (source, v) in [
            ("cached", &self.l1.cached_tokens),
            ("encoded", &self.l1.encoded_tokens),
        ] {
            let _ = writeln!(
                out,
                "sgl_router_tokenizer_l1_tokens_total{{source=\"{source}\"}} {}",
                v.load(Ordering::Relaxed)
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_token_counts() {
        let stats = TokenizerStats::default();
        stats.l1.cached_tokens.fetch_add(70, Ordering::Relaxed);
        stats.l1.encoded_tokens.fetch_add(30, Ordering::Relaxed);
        assert_eq!(stats.l1_tokens(), (70, 30));
        let body = stats.render();
        let samples: Vec<_> = body.lines().filter(|line| !line.starts_with('#')).collect();
        assert_eq!(
            samples,
            [
                "sgl_router_tokenizer_l1_tokens_total{source=\"cached\"} 70",
                "sgl_router_tokenizer_l1_tokens_total{source=\"encoded\"} 30",
            ]
        );
    }
}
