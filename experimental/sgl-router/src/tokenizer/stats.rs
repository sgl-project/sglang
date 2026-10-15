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
    /// toktier session-delta encoding; the ids are the reference engine's.
    Toktier,
    /// `toktier` was requested but the tokenizer is not one it ships a certified
    /// artifact for, or it could not load; every encode runs on HF.
    ToktierFallbackHf,
}

impl EncodeBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hf => "hf",
            Self::Fast => "fast",
            Self::FastFallbackHf => "fast_fallback_hf",
            Self::Tiktoken => "tiktoken",
            Self::Toktier => "toktier",
            Self::ToktierFallbackHf => "toktier_fallback_hf",
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
    /// The cache was requested but the encode backend keeps its own prefix
    /// state (toktier sessions), so the two would not stack.
    DisabledByBackend,
}

impl L1State {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Active => "active",
            Self::DisabledNoSpecials => "disabled_no_specials",
            Self::DisabledByBackend => "disabled_by_backend",
        }
    }
}

/// Counters fed by the L1 cache observers.
#[derive(Debug, Default)]
pub(crate) struct L1Counters {
    pub cached_tokens: AtomicU64,
    pub encoded_tokens: AtomicU64,
}

/// Which route each toktier encode took, fed by the toktier backend.
#[derive(Debug, Default)]
pub(crate) struct ToktierCounters {
    /// A bound session id whose stored transcript was a prefix of the prompt:
    /// only the appended tail was tokenized.
    pub session_repair: AtomicU64,
    /// A bound session id that was new, or whose stored transcript the prompt
    /// did not extend: the prompt was tokenized whole and becomes the transcript.
    pub session_seed: AtomicU64,
    /// No session id, but a stored transcript was a verified prefix of the prompt.
    pub content_lookup: AtomicU64,
    /// Tokenized whole on toktier's admitted route.
    pub full: AtomicU64,
    /// toktier declined the call; the Hugging Face encoder answered.
    pub fallback_hf: AtomicU64,
}

impl ToktierCounters {
    /// Counts as `(session_repair, session_seed, content_lookup, full, fallback_hf)`.
    pub fn routes(&self) -> (u64, u64, u64, u64, u64) {
        let load = |v: &AtomicU64| v.load(Ordering::Relaxed);
        (
            load(&self.session_repair),
            load(&self.session_seed),
            load(&self.content_lookup),
            load(&self.full),
            load(&self.fallback_hf),
        )
    }
}

#[derive(Debug, Default)]
pub struct TokenizerStats {
    pub(crate) backend: EncodeBackend,
    pub(crate) l1_state: L1State,
    pub(crate) l1: Arc<L1Counters>,
    pub(crate) toktier: Arc<ToktierCounters>,
}

impl TokenizerStats {
    pub fn backend(&self) -> EncodeBackend {
        self.backend
    }

    pub fn l1_state(&self) -> L1State {
        self.l1_state
    }

    /// toktier encodes by route as `(session_repair, session_seed, content_lookup, full, fallback_hf)`.
    pub fn toktier_routes(&self) -> (u64, u64, u64, u64, u64) {
        self.toktier.routes()
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
        counter_series(
            &mut out,
            "sgl_router_tokenizer_l1_tokens_total",
            "Tokens produced by encodes through the L1 cache, split into prefix tokens served from cache and freshly encoded tokens.",
            "source",
            &[
                ("cached", &self.l1.cached_tokens),
                ("encoded", &self.l1.encoded_tokens),
            ],
        );
        if self.backend == EncodeBackend::Toktier {
            counter_series(
                &mut out,
                "sgl_router_tokenizer_toktier_encodes_total",
                "Encodes through the toktier backend by route: session_repair (appended tail only), session_seed (whole prompt, stored as the session transcript), content_lookup (stored transcript found by content), full (toktier's admitted route), fallback_hf (toktier declined; HF answered).",
                "route",
                &[
                    ("session_repair", &self.toktier.session_repair),
                    ("session_seed", &self.toktier.session_seed),
                    ("content_lookup", &self.toktier.content_lookup),
                    ("full", &self.toktier.full),
                    ("fallback_hf", &self.toktier.fallback_hf),
                ],
            );
        }
        out
    }
}

/// One counter family in Prometheus exposition, a sample per `label` value.
fn counter_series(
    out: &mut String,
    name: &str,
    help: &str,
    label: &str,
    samples: &[(&str, &AtomicU64)],
) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    for (value, counter) in samples {
        let _ = writeln!(
            out,
            "{name}{{{label}=\"{value}\"}} {}",
            counter.load(Ordering::Relaxed)
        );
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

    #[test]
    fn render_toktier_routes_only_for_that_backend() {
        let mut stats = TokenizerStats::default();
        stats.toktier.session_repair.fetch_add(5, Ordering::Relaxed);
        assert!(!stats.render().contains("toktier_encodes_total"));
        stats.backend = EncodeBackend::Toktier;
        assert_eq!(stats.toktier_routes(), (5, 0, 0, 0, 0));
        let body = stats.render();
        let samples: Vec<_> = body
            .lines()
            .filter(|line| line.starts_with("sgl_router_tokenizer_toktier"))
            .collect();
        assert_eq!(
            samples,
            [
                "sgl_router_tokenizer_toktier_encodes_total{route=\"session_repair\"} 5",
                "sgl_router_tokenizer_toktier_encodes_total{route=\"session_seed\"} 0",
                "sgl_router_tokenizer_toktier_encodes_total{route=\"content_lookup\"} 0",
                "sgl_router_tokenizer_toktier_encodes_total{route=\"full\"} 0",
                "sgl_router_tokenizer_toktier_encodes_total{route=\"fallback_hf\"} 0",
            ]
        );
    }
}
