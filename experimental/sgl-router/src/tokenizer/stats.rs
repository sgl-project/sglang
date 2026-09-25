// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Resolved tokenizer encode settings and L1 cache counters, rendered on `/metrics`.

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
    const ALL: [Self; 4] = [Self::Hf, Self::Fast, Self::FastFallbackHf, Self::Tiktoken];

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
    const ALL: [Self; 3] = [Self::Off, Self::Active, Self::DisabledNoSpecials];

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
    pub hits: AtomicU64,
    pub misses: AtomicU64,
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

    /// L1 lookups as `(hits, misses)`.
    pub fn l1_lookups(&self) -> (u64, u64) {
        (
            self.l1.hits.load(Ordering::Relaxed),
            self.l1.misses.load(Ordering::Relaxed),
        )
    }

    /// Prometheus exposition for the tokenizer series.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "# HELP sgl_router_tokenizer_backend Resolved tokenizer encode backend; fast_fallback_hf means --tokenizer-backend fast was requested but fastokens could not load the tokenizer, so encode runs on hf.\n",
        );
        out.push_str("# TYPE sgl_router_tokenizer_backend gauge\n");
        for b in EncodeBackend::ALL {
            let _ = writeln!(
                out,
                "sgl_router_tokenizer_backend{{backend=\"{}\"}} {}",
                b.as_str(),
                u8::from(b == self.backend)
            );
        }
        out.push_str(
            "# HELP sgl_router_tokenizer_l1_state Resolved L1 prefix-tokenization cache state; disabled_no_specials means the cache was requested but the tokenizer declares no safely splittable special tokens, so it is inert.\n",
        );
        out.push_str("# TYPE sgl_router_tokenizer_l1_state gauge\n");
        for s in L1State::ALL {
            let _ = writeln!(
                out,
                "sgl_router_tokenizer_l1_state{{state=\"{}\"}} {}",
                s.as_str(),
                u8::from(s == self.l1_state)
            );
        }
        out.push_str(
            "# HELP sgl_router_tokenizer_l1_lookups_total L1 prefix-tokenization cache lookups by outcome, one per encode while the cache is active; hit means some prefix was served from cache.\n",
        );
        out.push_str("# TYPE sgl_router_tokenizer_l1_lookups_total counter\n");
        for (outcome, v) in [("hit", &self.l1.hits), ("miss", &self.l1.misses)] {
            let _ = writeln!(
                out,
                "sgl_router_tokenizer_l1_lookups_total{{outcome=\"{outcome}\"}} {}",
                v.load(Ordering::Relaxed)
            );
        }
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
    fn render_marks_resolved_state_and_counts() {
        let stats = TokenizerStats {
            backend: EncodeBackend::Fast,
            l1_state: L1State::Active,
            l1: Arc::default(),
        };
        stats.l1.hits.fetch_add(3, Ordering::Relaxed);
        stats.l1.cached_tokens.fetch_add(70, Ordering::Relaxed);
        let body = stats.render();
        assert!(body.contains("sgl_router_tokenizer_backend{backend=\"fast\"} 1\n"));
        assert!(body.contains("sgl_router_tokenizer_backend{backend=\"hf\"} 0\n"));
        assert!(body.contains("sgl_router_tokenizer_l1_state{state=\"active\"} 1\n"));
        assert!(body.contains("sgl_router_tokenizer_l1_state{state=\"off\"} 0\n"));
        assert!(body.contains("sgl_router_tokenizer_l1_lookups_total{outcome=\"hit\"} 3\n"));
        assert!(body.contains("sgl_router_tokenizer_l1_lookups_total{outcome=\"miss\"} 0\n"));
        assert!(body.contains("sgl_router_tokenizer_l1_tokens_total{source=\"cached\"} 70\n"));
    }
}
