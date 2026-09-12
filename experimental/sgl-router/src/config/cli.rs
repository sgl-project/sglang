// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Command-line interface. The router is configured entirely through
//! flags — there is no config file. [`Cli::into_config`] resolves the
//! flags into a validated [`Config`].

use anyhow::{anyhow, Result};
use clap::Parser;
use std::num::NonZeroU32;

use crate::config::{
    default_cb_cool_down, default_proxy_request_timeout_secs, default_stale_request_timeout_secs,
    resolve_mode, ActiveLoadConfig, AffinityConfig, AffinityMode, CacheAwareConfig,
    CircuitBreakerConfig, Config, DiscoveryBackend, EligibilityConfig, FilterKind, FusedTerm,
    K8sDiscoveryConfig, KvIndexerEndpointConfig, LogFormat, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, SessionAffinityMode, StaticUrlsDiscoveryConfig,
    StickyConfig, StickyFallbackKind, DEFAULT_FUSE,
};

const DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS: u64 = 100;
const DEFAULT_KV_INDEXER_QUERY_MAX_INFLIGHT: usize = sgl_kv_indexer::DEFAULT_QUERY_MAX_INFLIGHT;

/// `sgl-router` — slim KV-aware OpenAI-compatible router for SGLang workers.
///
/// Discovery is mutually exclusive: pass `--worker-urls` for a static
/// worker list, or `--service-discovery` for Kubernetes EndpointSlice
/// discovery — exactly one is required.
#[derive(Parser, Debug)]
#[command(
    name = "sgl-router",
    version,
    about = "Slim KV-aware OpenAI-compatible router for SGLang workers"
)]
pub struct Cli {
    // ---- server ----
    /// Address to bind the HTTP server to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Port to bind the HTTP server to.
    #[arg(long, default_value_t = 30000)]
    pub port: u16,

    // ---- model (exactly one) ----
    /// Model id this router serves (the OpenAI `model` field).
    #[arg(long)]
    pub model_id: String,
    /// Tokenizer source: a local `tokenizer.json` path, or a HuggingFace
    /// repo id to download from. When omitted, falls back to `--model-id`
    /// as the repo id (download honors `HF_TOKEN` / `HF_HOME`).
    #[arg(long)]
    pub tokenizer_path: Option<String>,
    /// Routing policy.
    #[arg(long, value_enum, default_value = "round_robin")]
    pub policy: PolicyKind,

    // ---- circuit breaker (opt-in via --cb-threshold) ----
    /// Consecutive upstream failures before the circuit breaker opens.
    /// Setting this enables the circuit breaker; `0` is rejected.
    #[arg(long)]
    pub cb_threshold: Option<NonZeroU32>,
    /// Circuit-breaker cool-down in seconds. Only meaningful with
    /// `--cb-threshold`; defaults to 30 when the breaker is enabled.
    #[arg(long)]
    pub cb_cool_down_secs: Option<u64>,

    // ---- legacy cache-aware-zmq tuning ----
    /// Min `matched_blocks / total_blocks` for a cache match to win.
    #[arg(long)]
    pub cache_threshold: Option<f32>,
    /// Absolute load spread above which the cache check is skipped.
    #[arg(long)]
    pub balance_abs_threshold: Option<usize>,
    /// Multiplicative load spread gating the absolute balance check.
    #[arg(long)]
    pub balance_rel_threshold: Option<f32>,
    /// External KV indexer gRPC endpoint used as the authoritative cache signal.
    /// Needs an explicit scheme, e.g. `http://10.0.0.1:50051`.
    #[arg(long)]
    pub kv_indexer_endpoint: Option<String>,
    /// KV Indexer query timeout in milliseconds. Requires
    /// `--kv-indexer-endpoint`; defaults to 100.
    #[arg(long)]
    pub kv_indexer_query_timeout_ms: Option<u64>,
    /// Maximum concurrent KV Indexer queries issued by this Router. Requires
    /// `--kv-indexer-endpoint`; defaults to 32.
    #[arg(long)]
    pub kv_indexer_query_max_inflight: Option<usize>,

    // ---- session-affinity tuning ----
    /// Header carrying the session ID for `--policy session_aware`.
    #[arg(long)]
    pub session_id_header: Option<String>,
    /// Idle timeout for a session assignment, in seconds.
    #[arg(long)]
    pub session_idle_secs: Option<u64>,
    /// Session-assignment eviction cadence, in seconds.
    #[arg(long)]
    pub session_eviction_interval_secs: Option<u64>,
    /// Use a deterministic backup for the affinity key and candidate range.
    #[arg(long)]
    pub stable_pair: bool,
    /// Session-affinity admission mode.
    #[arg(long, value_enum)]
    pub affinity_mode: Option<AffinityMode>,
    /// Session-affinity primary lookup and fallback behavior.
    #[arg(long, value_enum)]
    pub session_affinity_mode: Option<SessionAffinityMode>,
    /// Minimum cache-hit tokens for a cache-aware candidate.
    #[arg(long)]
    pub cache_affinity_min_matched_tokens: Option<u64>,
    /// Minimum cache-hit ratio for a cache-aware candidate.
    #[arg(long)]
    pub cache_affinity_min_match_ratio: Option<f64>,
    /// Minimum number of cache-aware candidates to try.
    #[arg(long)]
    pub cache_candidate_min_workers: Option<usize>,
    /// Fraction of healthy prefill workers considered as cache-aware candidates.
    #[arg(long)]
    pub cache_candidate_ratio: Option<f64>,
    /// Maximum number of cache-aware candidates to try.
    #[arg(long)]
    pub cache_candidate_max_workers: Option<usize>,
    /// Maximum uncached-work difference that pressure may override.
    #[arg(long)]
    pub cache_switch_margin_tokens: Option<u64>,

    // ---- score composition ----
    /// Policies to sum, spelled exactly as `--policy` spells them and each
    /// optionally weighted: `--fuse prefix_cache=2.0,load_based=0.3`. An
    /// omitted weight keeps that policy's own default. Requires `--policy
    /// score_policy` or `fused_score`; when either policy is set and this flag
    /// is omitted, the terms default to `prefix_cache,load_based`.
    #[arg(long, value_delimiter = ',')]
    pub fuse: Vec<FusedTerm>,

    /// Ordered hard constraints applied before policy selection.
    #[arg(long, value_delimiter = ',')]
    pub filter: Vec<FilterKind>,
    /// Router-local in-flight limit for `--filter overloaded`.
    #[arg(long)]
    pub max_in_flight: Option<usize>,
    /// Minimum cached prompt share for `--filter prefix_cache`.
    #[arg(long)]
    pub prefix_cache_min_share: Option<f32>,

    // ---- sticky-session policy (only used by `--policy sticky`) ----
    /// Request header carrying the routing key for sticky-session routing.
    /// Defaults to `x-sgl-routing-key` when `--policy sticky` is set.
    #[arg(long)]
    pub routing_key_header: Option<String>,
    /// Policy used to select a worker for requests with no routing key, and
    /// to pick the initial worker when a new key is first seen. One of
    /// `round_robin` / `random` / `power_of_two` / `load_based`. Defaults
    /// to `round_robin`.
    #[arg(long, value_enum)]
    pub sticky_fallback_policy: Option<StickyFallbackKind>,
    /// Evict a sticky assignment after it has been idle (unreferenced) this
    /// many seconds. Defaults to 600.
    #[arg(long)]
    pub sticky_idle_secs: Option<u64>,
    /// Wall-clock cadence of the sticky idle-eviction sweep, in seconds.
    /// Defaults to 60.
    #[arg(long)]
    pub sticky_eviction_interval_secs: Option<u64>,

    // ---- discovery: static ----
    /// Static worker URLs (space-separated or repeated). Mutually
    /// exclusive with `--service-discovery`.
    #[arg(long, num_args = 1..)]
    pub worker_urls: Vec<String>,

    // ---- discovery: kubernetes ----
    /// Enable Kubernetes EndpointSlice discovery.
    #[arg(long)]
    pub service_discovery: bool,
    /// Namespace to watch. Unset/empty watches all namespaces (requires
    /// cluster-wide RBAC).
    #[arg(long)]
    pub service_discovery_namespace: Option<String>,
    /// Plain-mode label selector terms, e.g. `app=engines-qwen3`
    /// (space-separated or repeated `key=value`, AND-joined). Mutually
    /// exclusive with the prefill/decode selectors.
    #[arg(long, num_args = 1..)]
    pub selector: Vec<String>,
    /// PD-mode prefill label selector terms. Requires `--decode-selector`.
    #[arg(long, num_args = 1..)]
    pub prefill_selector: Vec<String>,
    /// PD-mode decode label selector terms. Requires `--prefill-selector`.
    #[arg(long, num_args = 1..)]
    pub decode_selector: Vec<String>,

    // ---- proxy / active-load ----
    /// Per-request upstream timeout in seconds.
    #[arg(long, default_value_t = default_proxy_request_timeout_secs())]
    pub request_timeout_secs: u64,
    /// Max lifetime of an in-flight request entry before the janitor
    /// reaps it (returns 504 `stale_request_expired`).
    #[arg(long, default_value_t = default_stale_request_timeout_secs())]
    pub stale_request_timeout_secs: u64,

    // ---- observability ----
    /// Default tracing level (overridden by `RUST_LOG`).
    #[arg(long, default_value = "info")]
    pub log_level: String,
    /// Log output format.
    #[arg(long, value_enum, default_value = "text")]
    pub log_format: LogFormat,
}

impl Cli {
    /// Resolve parsed flags into a validated [`Config`].
    ///
    /// Builds the [`DiscoveryBackend`] (enforcing static-vs-k8s mutual
    /// exclusivity and resolving the k8s selector grammar via
    /// [`resolve_mode`]), assembles the single [`ModelConfig`], then runs
    /// [`Config::validate`] for the remaining value-level invariants
    /// (model id, static worker URLs).
    pub fn into_config(self) -> Result<Config> {
        let discovery = self.build_discovery()?;

        // Reject knobs that only take effect alongside another flag, rather
        // than silently dropping them — mirrors the discovery mutual-exclusion
        // checks. Otherwise an operator believes they tuned something that has
        // no effect.
        if self.cb_cool_down_secs.is_some() && self.cb_threshold.is_none() {
            return Err(anyhow!(
                "--cb-cool-down-secs requires --cb-threshold (the circuit breaker is \
                 enabled by --cb-threshold)"
            ));
        }
        let tuned_legacy_cache_aware = self.cache_threshold.is_some()
            || self.balance_abs_threshold.is_some()
            || self.balance_rel_threshold.is_some();
        if tuned_legacy_cache_aware && self.policy != PolicyKind::CacheAwareZmq {
            return Err(anyhow!(
                "cache-aware tuning flags require --policy cache_aware_zmq"
            ));
        }
        if self.kv_indexer_query_timeout_ms == Some(0) {
            return Err(anyhow!(
                "--kv-indexer-query-timeout-ms must be greater than zero"
            ));
        }
        if self.kv_indexer_query_timeout_ms.is_some() && self.kv_indexer_endpoint.is_none() {
            return Err(anyhow!(
                "--kv-indexer-query-timeout-ms requires --kv-indexer-endpoint"
            ));
        }
        if self.kv_indexer_query_max_inflight == Some(0) {
            return Err(anyhow!(
                "--kv-indexer-query-max-inflight must be greater than zero"
            ));
        }
        if self.kv_indexer_query_max_inflight.is_some() && self.kv_indexer_endpoint.is_none() {
            return Err(anyhow!(
                "--kv-indexer-query-max-inflight requires --kv-indexer-endpoint"
            ));
        }
        if self.kv_indexer_endpoint.is_some()
            && !matches!(
                self.policy,
                PolicyKind::CacheAware | PolicyKind::CacheAwareZmq
            )
        {
            return Err(anyhow!(
                "--kv-indexer-endpoint requires --policy cache_aware or cache_aware_zmq"
            ));
        }
        if self.policy == PolicyKind::CacheAware && self.kv_indexer_endpoint.is_none() {
            return Err(anyhow!(
                "--policy cache_aware requires --kv-indexer-endpoint"
            ));
        }
        let tuned_cache_aware = tuned_legacy_cache_aware || self.kv_indexer_endpoint.is_some();
        let affinity_policy = matches!(
            self.policy,
            PolicyKind::SessionAware | PolicyKind::CacheAware
        );
        let tuned_session_affinity = self.session_id_header.is_some()
            || self.session_idle_secs.is_some()
            || self.session_eviction_interval_secs.is_some()
            || self.stable_pair
            || self.affinity_mode.is_some()
            || self.session_affinity_mode.is_some();
        if tuned_session_affinity && self.policy != PolicyKind::SessionAware {
            return Err(anyhow!(
                "--session-id-header, --session-*-secs, --stable-pair, --affinity-mode, and \
                 --session-affinity-mode require --policy session_aware"
            ));
        }
        let tuned_cache_candidates = self.cache_affinity_min_matched_tokens.is_some()
            || self.cache_affinity_min_match_ratio.is_some()
            || self.cache_candidate_min_workers.is_some()
            || self.cache_candidate_ratio.is_some()
            || self.cache_candidate_max_workers.is_some()
            || self.cache_switch_margin_tokens.is_some();
        if tuned_cache_candidates && self.policy != PolicyKind::CacheAware {
            return Err(anyhow!(
                "cache candidate tuning flags require --policy cache_aware"
            ));
        }
        let is_score_composition = matches!(
            self.policy,
            PolicyKind::FusedScore | PolicyKind::ScorePolicy
        );
        if !self.fuse.is_empty() && !is_score_composition {
            return Err(anyhow!(
                "--fuse requires --policy score_policy or fused_score"
            ));
        }
        let fused = if is_score_composition {
            let terms = if self.fuse.is_empty() {
                DEFAULT_FUSE
                    .iter()
                    .map(|&kind| FusedTerm { kind, weight: None })
                    .collect()
            } else {
                self.fuse.clone()
            };
            for (i, t) in terms.iter().enumerate() {
                if terms[..i].iter().any(|p| p.kind == t.kind) {
                    return Err(anyhow!("--fuse: `{}` is listed more than once", t.kind));
                }
            }
            Some(terms)
        } else {
            None
        };

        for (i, kind) in self.filter.iter().enumerate() {
            if self.filter[..i].contains(kind) {
                return Err(anyhow!("--filter: `{kind}` is listed more than once"));
            }
        }
        let has = |k: FilterKind| self.filter.contains(&k);
        if self.max_in_flight.is_some() != has(FilterKind::Overloaded) {
            return Err(anyhow!(
                "--max-in-flight and `--filter overloaded` require each other"
            ));
        }
        if self.max_in_flight == Some(0) {
            return Err(anyhow!("--max-in-flight must be greater than 0"));
        }
        if self.prefix_cache_min_share.is_some() != has(FilterKind::PrefixCache) {
            return Err(anyhow!(
                "--prefix-cache-min-share and `--filter prefix_cache` require each other"
            ));
        }
        if self
            .prefix_cache_min_share
            .is_some_and(|s| !(s > 0.0 && s <= 1.0))
        {
            return Err(anyhow!("--prefix-cache-min-share must be in (0, 1]"));
        }
        if self.policy == PolicyKind::Sticky && !self.filter.is_empty() {
            return Err(anyhow!("--filter cannot be combined with --policy sticky"));
        }
        let eligibility = (!self.filter.is_empty()).then(|| EligibilityConfig {
            filters: self.filter.clone(),
            max_in_flight: self.max_in_flight,
            min_prefix_share: self.prefix_cache_min_share,
        });

        let tuned_sticky = self.routing_key_header.is_some()
            || self.sticky_fallback_policy.is_some()
            || self.sticky_idle_secs.is_some()
            || self.sticky_eviction_interval_secs.is_some();
        if tuned_sticky && self.policy != PolicyKind::Sticky {
            return Err(anyhow!(
                "--routing-key-header / --sticky-fallback-policy / --sticky-idle-secs / \
                 --sticky-eviction-interval-secs require --policy sticky"
            ));
        }

        // Build and validate the sticky config exactly when the sticky
        // policy is selected. The header name must parse as an HTTP header
        // name so a typo fails at startup rather than silently never
        // matching any request header.
        let sticky = if self.policy == PolicyKind::Sticky {
            let d = StickyConfig::default();
            let header_name = self.routing_key_header.unwrap_or(d.header_name);
            axum::http::HeaderName::try_from(header_name.as_str()).map_err(|e| {
                anyhow!("--routing-key-header {header_name:?} is not a valid HTTP header name: {e}")
            })?;
            let fallback_policy = self.sticky_fallback_policy.unwrap_or(d.fallback_policy);
            let idle_secs = self.sticky_idle_secs.unwrap_or(d.idle_secs);
            let eviction_interval_secs = self
                .sticky_eviction_interval_secs
                .unwrap_or(d.eviction_interval_secs);
            // Reject zero durations: `--sticky-eviction-interval-secs 0` would
            // panic `tokio::time::interval` at startup, and `--sticky-idle-secs
            // 0` would evict every assignment on the next sweep (defeating
            // stickiness entirely). Fail fast with a clear message instead.
            if eviction_interval_secs == 0 {
                return Err(anyhow!(
                    "--sticky-eviction-interval-secs must be greater than 0"
                ));
            }
            if idle_secs == 0 {
                return Err(anyhow!(
                    "--sticky-idle-secs must be greater than 0 (0 would evict every \
                     assignment immediately, defeating sticky routing)"
                ));
            }
            Some(StickyConfig {
                header_name,
                fallback_policy,
                idle_secs,
                eviction_interval_secs,
            })
        } else {
            None
        };

        let affinity = if affinity_policy {
            let d = AffinityConfig::default();
            let session_id_header = self.session_id_header.unwrap_or(d.session_id_header);
            axum::http::HeaderName::try_from(session_id_header.as_str()).map_err(|e| {
                anyhow!("--session-id-header {session_id_header:?} is not a valid HTTP header name: {e}")
            })?;
            let cache_affinity_min_match_ratio = self
                .cache_affinity_min_match_ratio
                .or(d.cache_affinity_min_match_ratio);
            if cache_affinity_min_match_ratio
                .is_some_and(|ratio| !ratio.is_finite() || !(0.0..=1.0).contains(&ratio))
            {
                return Err(anyhow!(
                    "--cache-affinity-min-match-ratio must be finite and in [0, 1]"
                ));
            }
            let cache_candidate_ratio = self
                .cache_candidate_ratio
                .unwrap_or(d.cache_candidate_ratio);
            if !cache_candidate_ratio.is_finite() || !(0.0..=1.0).contains(&cache_candidate_ratio) {
                return Err(anyhow!(
                    "--cache-candidate-ratio must be finite and in [0, 1]"
                ));
            }
            let cache_candidate_min_workers = self
                .cache_candidate_min_workers
                .unwrap_or(d.cache_candidate_min_workers);
            let cache_candidate_max_workers = self
                .cache_candidate_max_workers
                .unwrap_or(d.cache_candidate_max_workers);
            if cache_candidate_min_workers == 0
                || cache_candidate_max_workers == 0
                || cache_candidate_min_workers > cache_candidate_max_workers
            {
                return Err(anyhow!(
                    "--cache-candidate-min-workers and --cache-candidate-max-workers must be \
                     positive and min must not exceed max"
                ));
            }
            let session_idle_secs = self.session_idle_secs.unwrap_or(d.session_idle_secs);
            let session_eviction_interval_secs = self
                .session_eviction_interval_secs
                .unwrap_or(d.session_eviction_interval_secs);
            if session_idle_secs == 0 {
                return Err(anyhow!("--session-idle-secs must be greater than 0"));
            }
            if session_eviction_interval_secs == 0 {
                return Err(anyhow!(
                    "--session-eviction-interval-secs must be greater than 0"
                ));
            }
            Some(AffinityConfig {
                session_id_header,
                session_idle_secs,
                session_eviction_interval_secs,
                stable_pair: self.stable_pair,
                mode: self.affinity_mode.unwrap_or(d.mode),
                session_affinity_mode: self
                    .session_affinity_mode
                    .unwrap_or(d.session_affinity_mode),
                cache_affinity_min_matched_tokens: self
                    .cache_affinity_min_matched_tokens
                    .or(d.cache_affinity_min_matched_tokens),
                cache_affinity_min_match_ratio,
                cache_candidate_min_workers,
                cache_candidate_ratio,
                cache_candidate_max_workers,
                cache_switch_margin_tokens: self
                    .cache_switch_margin_tokens
                    .unwrap_or(d.cache_switch_margin_tokens),
            })
        } else {
            None
        };

        let circuit_breaker = self.cb_threshold.map(|threshold| CircuitBreakerConfig {
            threshold,
            cool_down_secs: self.cb_cool_down_secs.unwrap_or_else(default_cb_cool_down),
        });

        // Only build a CacheAwareConfig when the operator tuned at least
        // one knob; otherwise leave it None so the policy uses its own
        // defaults. Unset knobs fall back to the per-field defaults.
        let cache_aware = if tuned_cache_aware {
            let d = CacheAwareConfig::default();
            let kv_indexer_endpoint = self.kv_indexer_endpoint.map(|url| KvIndexerEndpointConfig {
                url,
                query_timeout_ms: self
                    .kv_indexer_query_timeout_ms
                    .unwrap_or(DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS),
                query_max_inflight: self
                    .kv_indexer_query_max_inflight
                    .unwrap_or(DEFAULT_KV_INDEXER_QUERY_MAX_INFLIGHT),
            });
            Some(CacheAwareConfig {
                cache_threshold: self.cache_threshold.unwrap_or(d.cache_threshold),
                balance_abs_threshold: self
                    .balance_abs_threshold
                    .unwrap_or(d.balance_abs_threshold),
                balance_rel_threshold: self
                    .balance_rel_threshold
                    .unwrap_or(d.balance_rel_threshold),
                kv_indexer_endpoint,
            })
        } else {
            None
        };

        let config = Config {
            server: ServerConfig {
                host: self.host,
                port: self.port,
            },
            observability: ObservabilityConfig {
                log_level: self.log_level,
                log_format: self.log_format,
            },
            model: ModelConfig {
                // Default the tokenizer source to the model id (treated as a
                // HuggingFace repo id) when --tokenizer-path is omitted.
                tokenizer_path: self.tokenizer_path.unwrap_or_else(|| self.model_id.clone()),
                id: self.model_id,
                policy: self.policy,
                circuit_breaker,
                cache_aware,
                sticky,
                affinity,
                fused,
                eligibility,
            },
            discovery,
            proxy: ProxyConfig {
                request_timeout_secs: self.request_timeout_secs,
            },
            active_load: ActiveLoadConfig {
                stale_request_timeout_secs: self.stale_request_timeout_secs,
            },
        };
        config.validate()?;
        Ok(config)
    }

    /// Resolve the discovery flags into a [`DiscoveryBackend`].
    ///
    /// `--worker-urls` (static) and `--service-discovery` (k8s) are
    /// mutually exclusive and exactly one is required. K8s-only flags
    /// passed without `--service-discovery` are rejected so a typo can't
    /// silently fall back to the static (empty) path. The k8s selector
    /// grammar (plain vs PD) is validated eagerly here by [`resolve_mode`]
    /// before the `K8sDiscoveryConfig` is constructed, so an invalid
    /// combination is never stored.
    fn build_discovery(&self) -> Result<DiscoveryBackend> {
        let has_static = !self.worker_urls.is_empty();
        let backend = match (has_static, self.service_discovery) {
            (true, true) => {
                return Err(anyhow!(
                    "--worker-urls and --service-discovery are mutually exclusive; pass exactly one"
                ))
            }
            (false, false) => {
                return Err(anyhow!(
                    "no discovery backend selected; pass --worker-urls <URL...> (static) \
                     or --service-discovery (kubernetes)"
                ))
            }
            (true, false) => {
                if self.service_discovery_namespace.is_some()
                    || !self.selector.is_empty()
                    || !self.prefill_selector.is_empty()
                    || !self.decode_selector.is_empty()
                {
                    return Err(anyhow!(
                        "--service-discovery-namespace / --selector / --prefill-selector / \
                         --decode-selector require --service-discovery"
                    ));
                }
                DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                    urls: self.worker_urls.clone(),
                })
            }
            (false, true) => {
                // Resolve (and validate) the selector flags into a
                // K8sDiscoveryMode here, so an invalid combination can't be
                // stored. Surfaces ConfigError as anyhow for the CLI.
                let mode = resolve_mode(
                    join_selector(&self.selector).as_deref(),
                    join_selector(&self.prefill_selector).as_deref(),
                    join_selector(&self.decode_selector).as_deref(),
                )
                .map_err(|e| anyhow!("{e}"))?;
                DiscoveryBackend::K8s(K8sDiscoveryConfig {
                    namespace: self.service_discovery_namespace.clone().unwrap_or_default(),
                    mode,
                })
            }
        };
        Ok(backend)
    }
}

/// Join space/repeated `key=value` selector terms into the single
/// comma-joined string the k8s backend's `labels_match_selector`
/// expects. `None` for an empty term list so [`resolve_mode`] can apply
/// its plain-vs-PD rules (and surface `NoSelector`).
fn join_selector(terms: &[String]) -> Option<String> {
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DiscoveryBackend, K8sDiscoveryMode, ScoreTermKind};

    /// Parse argv (without the leading binary name) into a `Config`.
    fn into_config(args: &[&str]) -> Result<Config> {
        let argv = std::iter::once("sgl-router").chain(args.iter().copied());
        let cli = Cli::try_parse_from(argv).map_err(|e| anyhow!("{e}"))?;
        cli.into_config()
    }

    const MODEL_ARGS: &[&str] = &[
        "--model-id",
        "qwen3-0.6b",
        "--tokenizer-path",
        "/tmp/qwen.json",
    ];

    fn with_model(extra: &[&str]) -> Vec<String> {
        MODEL_ARGS
            .iter()
            .chain(extra.iter())
            .map(|s| s.to_string())
            .collect()
    }

    fn into_config_owned(args: Vec<String>) -> Result<Config> {
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        into_config(&refs)
    }

    #[test]
    fn defaults_host_port_and_policy() {
        let c = into_config_owned(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
        assert_eq!(c.server.host, "127.0.0.1");
        assert_eq!(c.server.port, 30000);
        assert_eq!(c.model.policy, PolicyKind::RoundRobin);
        assert_eq!(c.model.id, "qwen3-0.6b");
        assert_eq!(c.proxy.request_timeout_secs, 300);
        assert_eq!(c.active_load.stale_request_timeout_secs, 600);
    }

    /// With `--tokenizer-path` omitted, the tokenizer source defaults to the
    /// model id (treated as an HF repo id at load time).
    #[test]
    fn tokenizer_path_defaults_to_model_id_when_omitted() {
        let c = into_config(&[
            "--model-id",
            "Qwen/Qwen3-0.6B",
            "--worker-urls",
            "http://x:30000",
        ])
        .unwrap();
        assert_eq!(c.model.id, "Qwen/Qwen3-0.6B");
        assert_eq!(c.model.tokenizer_path, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn explicit_tokenizer_path_is_used() {
        let c = into_config(&[
            "--model-id",
            "qwen3",
            "--tokenizer-path",
            "/models/qwen3/tokenizer.json",
            "--worker-urls",
            "http://x:30000",
        ])
        .unwrap();
        assert_eq!(c.model.tokenizer_path, "/models/qwen3/tokenizer.json");
    }

    #[test]
    fn static_urls_backend() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "http://10.0.0.2:30000",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::StaticUrls(s) => assert_eq!(
                s.urls,
                vec![
                    "http://10.0.0.1:30000".to_string(),
                    "http://10.0.0.2:30000".to_string()
                ]
            ),
            _ => panic!("expected static_urls backend"),
        }
    }

    #[test]
    fn rejects_no_discovery_backend() {
        let err = into_config_owned(with_model(&[])).unwrap_err().to_string();
        assert!(err.contains("no discovery backend"), "got: {err}");
    }

    #[test]
    fn rejects_both_discovery_backends() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--service-discovery",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    #[test]
    fn rejects_k8s_flags_without_service_discovery() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--selector",
            "app=sglang",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --service-discovery"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_duplicate() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "http://x:30000",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("duplicate"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_schemeless() {
        let err = into_config_owned(with_model(&["--worker-urls", "10.0.0.1:30000"]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not a valid URL") || err.contains("unsupported scheme"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_static_urls_non_http_scheme() {
        let err = into_config_owned(with_model(&["--worker-urls", "ws://x:30000"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheme"), "got: {err}");
    }

    #[test]
    fn k8s_plain_backend() {
        let c = into_config_owned(with_model(&[
            "--service-discovery",
            "--service-discovery-namespace",
            "prod",
            "--selector",
            "app=engines-qwen3",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::K8s(k) => {
                assert_eq!(k.namespace, "prod");
                assert_eq!(
                    k.mode,
                    K8sDiscoveryMode::Plain {
                        label_selector: "app=engines-qwen3".to_string()
                    }
                );
            }
            _ => panic!("expected k8s backend"),
        }
    }

    /// Multiple `--selector` terms AND-join into one comma-separated
    /// label selector (matches the Python router's space-separated form).
    #[test]
    fn k8s_plain_selector_joins_multiple_terms() {
        let c = into_config_owned(with_model(&[
            "--service-discovery",
            "--selector",
            "app=sglang",
            "zone=us-east",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::K8s(k) => assert_eq!(
                k.mode,
                K8sDiscoveryMode::Plain {
                    label_selector: "app=sglang,zone=us-east".to_string()
                }
            ),
            _ => panic!("expected k8s backend"),
        }
    }

    /// Empty namespace is intentional — it triggers a cluster-wide watch.
    #[test]
    fn k8s_empty_namespace_watches_all() {
        let c = into_config_owned(with_model(&[
            "--service-discovery",
            "--selector",
            "app=sglang",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::K8s(k) => assert_eq!(k.namespace, ""),
            _ => panic!("expected k8s backend"),
        }
    }

    #[test]
    fn k8s_pd_backend() {
        let c = into_config_owned(with_model(&[
            "--service-discovery",
            "--service-discovery-namespace",
            "default",
            "--prefill-selector",
            "app=sglang,role=prefill",
            "--decode-selector",
            "app=sglang,role=decode",
        ]))
        .unwrap();
        match &c.discovery {
            DiscoveryBackend::K8s(k) => assert_eq!(
                k.mode,
                K8sDiscoveryMode::PdDisaggregation {
                    prefill_selector: "app=sglang,role=prefill".to_string(),
                    decode_selector: "app=sglang,role=decode".to_string(),
                }
            ),
            _ => panic!("expected k8s backend"),
        }
    }

    /// `--service-discovery` with no selector at all fails `resolve_mode`
    /// validation with the `NoSelector` wording.
    #[test]
    fn rejects_k8s_without_selector() {
        let err = into_config_owned(with_model(&["--service-discovery"]))
            .unwrap_err()
            .to_string()
            .to_lowercase();
        assert!(err.contains("none were set"), "got: {err}");
    }

    /// `--prefill-selector` without `--decode-selector` is rejected through
    /// the full CLI path — pins that `build_discovery` feeds the right
    /// selectors into `resolve_mode` (a positional mix-up would surface a
    /// different error or none).
    #[test]
    fn rejects_k8s_partial_pd_selectors() {
        let err = into_config_owned(with_model(&[
            "--service-discovery",
            "--prefill-selector",
            "app=sglang,role=prefill",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("PD mode requires BOTH"),
            "expected PartialPdSelectors wording, got: {err}"
        );
    }

    /// Identical prefill/decode selectors are rejected through the full CLI
    /// path (would silently leave the decode pool empty at runtime).
    #[test]
    fn rejects_k8s_identical_pd_selectors() {
        let err = into_config_owned(with_model(&[
            "--service-discovery",
            "--prefill-selector",
            "app=sglang",
            "--decode-selector",
            "app=sglang",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("must differ"),
            "expected IdenticalPdSelectors wording, got: {err}"
        );
    }

    /// clap rejects an unknown `--policy` value at parse time.
    #[test]
    fn rejects_unknown_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "bogus_policy",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("bogus_policy") || err.contains("policy"),
            "got: {err}"
        );
    }

    #[test]
    fn policy_accepts_only_routing_strategies() {
        for value in ["prefix_cache", "overloaded"] {
            let err = into_config_owned(with_model(&[
                "--worker-urls",
                "http://x:30000",
                "--policy",
                value,
            ]))
            .expect_err("score terms and filters are not top-level policies")
            .to_string();
            assert!(err.contains(value), "{value}: {err}");
        }
    }

    #[test]
    fn filters_and_fuse_terms_reject_non_members() {
        let cases = [
            (vec!["--filter", "load_based"], "load_based"),
            (
                vec!["--policy", "fused_score", "--fuse", "sticky"],
                "sticky",
            ),
        ];
        for (args, value) in cases {
            let err = into_config_owned(with_model(
                &[&["--worker-urls", "http://x:30000"], &args[..]].concat(),
            ))
            .expect_err("the option must reject a kind from another layer")
            .to_string();
            assert!(err.contains(value), "{value}: {err}");
        }
    }

    /// `--policy load_based` parses to the load-based selector.
    #[test]
    fn parses_load_based_policy() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--policy",
            "load_based",
        ]))
        .unwrap();
        assert_eq!(c.model.policy, PolicyKind::LoadBased);
    }

    /// clap rejects `--cb-threshold 0` because the field is `NonZeroU32`.
    #[test]
    fn rejects_zero_cb_threshold() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cb-threshold",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("cb-threshold"), "got: {err}");
    }

    #[test]
    fn cb_threshold_enables_circuit_breaker_with_default_cool_down() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cb-threshold",
            "5",
        ]))
        .unwrap();
        let cb = c.model.circuit_breaker.expect("cb enabled");
        assert_eq!(cb.threshold.get(), 5);
        assert_eq!(cb.cool_down_secs, 30);
    }

    #[test]
    fn cb_cool_down_honors_explicit_override() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cb-threshold",
            "3",
            "--cb-cool-down-secs",
            "10",
        ]))
        .unwrap();
        let cb = c.model.circuit_breaker.expect("cb enabled");
        assert_eq!(cb.cool_down_secs, 10);
    }

    #[test]
    fn rejects_cb_cool_down_without_threshold() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cb-cool-down-secs",
            "10",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--cb-cool-down-secs requires --cb-threshold"),
            "got: {err}"
        );
    }

    #[test]
    fn cache_aware_knob_builds_partial_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--cache-threshold",
            "0.7",
        ]))
        .unwrap();
        let ca = c.model.cache_aware.expect("cache_aware set");
        assert_eq!(ca.cache_threshold, 0.7);
        // Untouched knobs fall back to defaults.
        assert_eq!(ca.balance_abs_threshold, 32);
    }

    #[test]
    fn kv_indexer_reuses_cache_aware_policy_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
            "--kv-indexer-query-timeout-ms",
            "75",
            "--kv-indexer-query-max-inflight",
            "17",
        ]))
        .unwrap();
        let cache = c.model.cache_aware.expect("cache-aware config");
        let indexer = cache.kv_indexer_endpoint.expect("Indexer config");
        assert_eq!(indexer.url, "http://indexer:50051");
        assert_eq!(indexer.query_timeout_ms, 75);
        assert_eq!(indexer.query_max_inflight, 17);
    }

    #[test]
    fn kv_indexer_uses_safe_query_defaults() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
        ]))
        .unwrap();
        let indexer = c
            .model
            .cache_aware
            .expect("cache-aware config")
            .kv_indexer_endpoint
            .expect("Indexer config");
        assert_eq!(
            indexer.query_timeout_ms,
            DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS
        );
        assert_eq!(indexer.query_max_inflight, 32);
    }

    #[test]
    fn kv_indexer_is_accepted_by_cache_aware_zmq() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
        ]))
        .unwrap();
        let indexer = c
            .model
            .cache_aware
            .expect("cache-aware config")
            .kv_indexer_endpoint
            .expect("Indexer config");
        assert_eq!(indexer.url, "http://indexer:50051");
    }

    #[test]
    fn kv_indexer_requires_cache_aware_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires --policy cache_aware"), "got: {err}");
    }

    #[test]
    fn kv_indexer_timeout_requires_endpoint() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware",
            "--kv-indexer-query-timeout-ms",
            "75",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires --kv-indexer-endpoint"));
    }

    #[test]
    fn kv_indexer_max_inflight_requires_endpoint() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware",
            "--kv-indexer-query-max-inflight",
            "17",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires --kv-indexer-endpoint"));
    }

    #[test]
    fn kv_indexer_max_inflight_must_be_positive() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware",
            "--kv-indexer-endpoint",
            "http://indexer:50051",
            "--kv-indexer-query-max-inflight",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("must be greater than zero"));
    }

    #[test]
    fn no_cache_aware_flags_leaves_none() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
        ]))
        .unwrap();
        assert!(c.model.cache_aware.is_none());
    }

    #[test]
    fn rejects_cache_aware_knob_without_cache_aware_policy() {
        // Default policy is round_robin, so a cache knob has no effect —
        // reject rather than silently ignore it.
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--cache-threshold",
            "0.7",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("require --policy cache_aware_zmq"),
            "got: {err}"
        );
    }

    #[test]
    fn log_format_parses_json() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--log-format",
            "json",
        ]))
        .unwrap();
        assert_eq!(c.observability.log_format, LogFormat::Json);
    }

    /// Pins that the two timeout overrides land in the right fields — they
    /// are adjacent `u64`s with similar names, so a copy-paste swap would
    /// otherwise go unnoticed (and `stale` must sit above `proxy`).
    #[test]
    fn timeout_overrides_land_in_distinct_fields() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--request-timeout-secs",
            "120",
            "--stale-request-timeout-secs",
            "240",
        ]))
        .unwrap();
        assert_eq!(c.proxy.request_timeout_secs, 120);
        assert_eq!(c.active_load.stale_request_timeout_secs, 240);
    }

    #[test]
    fn sticky_policy_defaults_header_and_tuning() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
        ]))
        .unwrap();
        assert_eq!(c.model.policy, PolicyKind::Sticky);
        let s = c.model.sticky.expect("sticky config built");
        assert_eq!(s.header_name, "x-sgl-routing-key");
        assert_eq!(s.fallback_policy, StickyFallbackKind::RoundRobin);
        assert_eq!(s.idle_secs, 600);
        assert_eq!(s.eviction_interval_secs, 60);
    }

    #[test]
    fn sticky_fallback_help_lists_only_dependency_free_policies() {
        use clap::CommandFactory;

        let mut command = Cli::command();
        let mut help = Vec::new();
        command.write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();
        let (_, after) = help
            .split_once("--sticky-fallback-policy <STICKY_FALLBACK_POLICY>")
            .expect("sticky fallback option is documented");
        let choices = after
            .split_once("--sticky-idle-secs")
            .expect("sticky fallback precedes its tuning")
            .0;

        for value in ["round_robin", "random", "power_of_two", "load_based"] {
            assert!(choices.contains(value), "missing {value}: {choices}");
        }
        for value in ["fused_score", "cache_aware_zmq", "sticky"] {
            assert!(!choices.contains(value), "unexpected {value}: {choices}");
        }
    }

    #[test]
    fn sticky_flags_override_defaults() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--routing-key-header",
            "x-session-id",
            "--sticky-fallback-policy",
            "load_based",
            "--sticky-idle-secs",
            "120",
            "--sticky-eviction-interval-secs",
            "15",
        ]))
        .unwrap();
        let s = c.model.sticky.expect("sticky config built");
        assert_eq!(s.header_name, "x-session-id");
        assert_eq!(s.fallback_policy, StickyFallbackKind::LoadBased);
        assert_eq!(s.idle_secs, 120);
        assert_eq!(s.eviction_interval_secs, 15);
    }

    #[test]
    fn filter_builds_the_eligibility_config_in_order_and_is_off_by_default() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "round_robin",
            "--filter",
            "overloaded,prefix_cache",
            "--max-in-flight",
            "64",
            "--prefix-cache-min-share",
            "0.6",
        ]))
        .unwrap();
        let e = c.model.eligibility.expect("--filter must build the config");
        assert_eq!(
            e.filters,
            vec![FilterKind::Overloaded, FilterKind::PrefixCache],
            "order is priority, so it must survive parsing",
        );
        assert_eq!((e.max_in_flight, e.min_prefix_share), (Some(64), Some(0.6)));
        assert_eq!(
            c.model.policy,
            PolicyKind::RoundRobin,
            "not gated on --policy"
        );

        let bare = into_config_owned(with_model(&["--worker-urls", "http://x:30000"])).unwrap();
        assert!(bare.model.eligibility.is_none(), "no --filter, no layer");
    }

    #[test]
    fn filter_misconfigurations_fail_at_startup() {
        let cases: [(&[&str], &str); 8] = [
            (&["--filter", "overloaded"], "require each other"),
            (&["--max-in-flight", "64"], "require each other"),
            (&["--filter", "prefix_cache"], "require each other"),
            (&["--prefix-cache-min-share", "0.6"], "require each other"),
            (
                &["--filter", "overloaded,overloaded", "--max-in-flight", "64"],
                "listed more than once",
            ),
            (
                &[
                    "--filter",
                    "prefix_cache",
                    "--prefix-cache-min-share",
                    "0.0",
                ],
                "must be in (0, 1]",
            ),
            (
                &["--filter", "overloaded", "--max-in-flight", "0"],
                "must be greater than 0",
            ),
            (
                &[
                    "--policy",
                    "sticky",
                    "--filter",
                    "overloaded",
                    "--max-in-flight",
                    "1",
                ],
                "cannot be combined with --policy sticky",
            ),
        ];
        for (extra, want) in cases {
            let mut args = vec!["--worker-urls", "http://x:30000"];
            args.extend_from_slice(extra);
            let err = into_config_owned(with_model(&args))
                .unwrap_err()
                .to_string();
            assert!(err.contains(want), "for {extra:?} got: {err}");
        }
    }

    #[test]
    fn non_sticky_policy_leaves_sticky_none() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "round_robin",
        ]))
        .unwrap();
        assert!(c.model.sticky.is_none());
    }

    #[test]
    fn rejects_sticky_flags_without_sticky_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--routing-key-header",
            "x-session-id",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("require --policy sticky"), "got: {err}");
    }

    #[test]
    fn rejects_invalid_routing_key_header() {
        // A space is not a legal HTTP header-name character.
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--routing-key-header",
            "bad header",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("not a valid HTTP header name"), "got: {err}");
    }

    #[test]
    fn rejects_cache_aware_zmq_as_sticky_fallback() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-fallback-policy",
            "cache_aware_zmq",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("invalid value"), "got: {err}");
    }

    #[test]
    fn rejects_sticky_as_sticky_fallback() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-fallback-policy",
            "sticky",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("invalid value"), "got: {err}");
    }

    /// A zero eviction interval would panic `tokio::time::interval` at
    /// startup — reject it at config-build time with a clear message.
    #[test]
    fn rejects_zero_sticky_eviction_interval() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-eviction-interval-secs",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--sticky-eviction-interval-secs must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_zero_sticky_idle() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-idle-secs",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--sticky-idle-secs must be greater than 0"),
            "got: {err}"
        );
    }

    /// `argv` is space-split, so a case reads as the command line an operator
    /// would type. Model + worker URL are supplied.
    fn cfg_of(argv: &str) -> Result<Config> {
        let extra: Vec<&str> = argv.split_whitespace().collect();
        into_config_owned(with_model(
            &[&["--worker-urls", "http://10.0.0.1:30000"], &extra[..]].concat(),
        ))
    }

    fn fuse_err(argv: &str) -> String {
        cfg_of(argv).unwrap_err().to_string()
    }

    /// Resolved terms as `(kind, weight)` pairs; `None` when the policy is
    /// not `fused_score` and so builds no term list at all.
    fn fused_of(argv: &str) -> Option<Vec<(ScoreTermKind, Option<f32>)>> {
        let ts = cfg_of(argv).unwrap().model.fused?;
        Some(ts.iter().map(|t| (t.kind, t.weight)).collect())
    }

    fn fuse_ok(argv: &str) -> Vec<(ScoreTermKind, Option<f32>)> {
        fused_of(argv).expect("fused_score builds a term list")
    }

    /// `score_policy` is an independent top-level policy.
    #[test]
    fn score_policy_is_a_top_level_policy_with_its_own_cli_spelling() {
        use PolicyKind::ScorePolicy;
        use ScoreTermKind::{LoadBased, PrefixCache};
        let pair = [(PrefixCache, None), (LoadBased, None)];
        let config = cfg_of("--policy score_policy").unwrap();
        assert_eq!(config.model.policy, ScorePolicy);
        assert_eq!(
            config
                .model
                .fused
                .expect("score_policy must resolve its score terms")
                .iter()
                .map(|t| (t.kind, t.weight))
                .collect::<Vec<_>>(),
            pair,
        );
        assert_eq!(
            fuse_ok("--policy score_policy --fuse prefix_cache=2.0,load_based=0.3"),
            [(PrefixCache, Some(2.0)), (LoadBased, Some(0.3))],
        );
    }

    /// `fused_score` keeps the compatibility entry point.
    #[test]
    fn fuse_defaults_to_the_useful_pair_and_parses_weights() {
        use ScoreTermKind::{LoadBased, PrefixCache, Random};
        let pair = [(PrefixCache, None), (LoadBased, None)];
        assert_eq!(fuse_ok("--policy fused_score"), pair);
        // Comma-separated, order preserved, weight optional per term.
        assert_eq!(
            fuse_ok("--policy fused_score --fuse load_based=0.3,random"),
            [(LoadBased, Some(0.3)), (Random, None)],
        );
        assert!(fused_of("").is_none(), "round_robin builds no term list");
    }

    /// Non-finite and negative weights are refused, naming the term.
    ///
    /// `nan`/`inf` matter more than they look: `str::parse::<f32>` accepts
    /// both, and a NaN weight makes every worker's fused total NaN, so argmax
    /// discards them all and the router silently degrades to least-load.
    #[test]
    fn fuse_rejects_non_finite_and_negative_weights() {
        for bad in ["nan", "NaN", "inf", "-inf", "-0.5", "banana"] {
            let err = fuse_err(&format!("--policy fused_score --fuse load_based={bad}"));
            assert!(err.contains("load_based"), "{bad}: names the term: {err}");
            assert!(
                err.contains("must be finite and >= 0") || err.contains("is not a number"),
                "{bad}: {err}",
            );
        }
        for good in ["0", "0.3", "2", "1e3"] {
            let got = fuse_ok(&format!("--policy fused_score --fuse load_based={good}"))[0].1;
            assert_eq!(got, Some(good.parse::<f32>().unwrap()));
        }
    }

    #[test]
    fn fuse_rejects_malformed_compositions() {
        let cases: [(&str, &[&str]); 6] = [
            ("--fuse load_based", &["--fuse requires", "fused_score"]),
            (
                "--policy fused_score --fuse fused_score,load_based",
                &["fused_score", "not a score term"],
            ),
            (
                "--policy fused_score --fuse load_based,load_based",
                &["load_based", "listed more than once"],
            ),
            (
                "--policy score_policy --fuse score_policy,load_based",
                &["score_policy", "not a score term"],
            ),
            (
                "--policy fused_score --fuse not_a_policy",
                &["not_a_policy", "is not a score term"],
            ),
            (
                "--policy sticky --sticky-fallback-policy prefix_cache",
                &["prefix_cache", "invalid value"],
            ),
        ];
        for (argv, wants) in cases {
            let err = fuse_err(argv);
            for want in wants {
                assert!(err.contains(want), "{argv}: want {want:?}, got: {err}");
            }
        }
    }

    #[test]
    fn session_aware_builds_affinity_config_from_its_cli_knobs() {
        let config = cfg_of(
            "--policy session_aware --session-id-header x-agent-session --stable-pair \
             --affinity-mode strict --session-affinity-mode global-rebind",
        )
        .unwrap();
        let affinity = config
            .model
            .affinity
            .expect("session policy needs affinity config");

        assert_eq!(config.model.policy, PolicyKind::SessionAware);
        assert_eq!(affinity.session_id_header, "x-agent-session");
        assert!(affinity.stable_pair);
        assert_eq!(affinity.mode, AffinityMode::Strict);
        assert_eq!(
            affinity.session_affinity_mode,
            SessionAffinityMode::GlobalRebind
        );
    }

    #[test]
    fn rejects_removed_token_pressure_flags() {
        for flag in [
            "--disable-pressure-guard",
            "--pressure-abs-threshold-tokens 2048",
            "--pressure-rel-threshold 2.0",
        ] {
            let error = cfg_of(&format!("--policy session_aware {flag}"))
                .unwrap_err()
                .to_string();
            assert!(error.contains("unexpected argument"), "{flag}: {error}");
        }
    }

    #[test]
    fn rejects_removed_affinity_aware_range_flag() {
        let error = cfg_of("--policy session_aware --affinity-aware-range global-first")
            .unwrap_err()
            .to_string();
        assert!(error.contains("unexpected argument '--affinity-aware-range'"));
    }

    #[test]
    fn session_affinity_mode_accepts_all_new_values() {
        for (value, expected) in [
            ("bucket", SessionAffinityMode::Bucket),
            ("global-rebind", SessionAffinityMode::GlobalRebind),
            ("global-preserve", SessionAffinityMode::GlobalPreserve),
        ] {
            let config = cfg_of(&format!(
                "--policy session_aware --session-affinity-mode {value}"
            ))
            .unwrap();
            assert_eq!(
                config.model.affinity.unwrap().session_affinity_mode,
                expected
            );
        }
    }

    #[test]
    fn session_aware_configures_bounded_assignment_lifetime() {
        let config = cfg_of(
            "--policy session_aware --session-idle-secs 120 \
             --session-eviction-interval-secs 15",
        )
        .unwrap();
        let affinity = config.model.affinity.expect("session affinity config");
        assert_eq!(affinity.session_idle_secs, 120);
        assert_eq!(affinity.session_eviction_interval_secs, 15);
    }

    #[test]
    fn cache_aware_accepts_indexer_endpoint_and_rejects_affinity_knobs_elsewhere() {
        let config = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --kv-indexer-query-timeout-ms 40 \
             --cache-affinity-min-matched-tokens 512 --cache-affinity-min-match-ratio 0.25 \
             --cache-candidate-min-workers 4 --cache-candidate-ratio 0.1 \
             --cache-candidate-max-workers 16 --cache-switch-margin-tokens 128",
        )
        .unwrap();
        assert_eq!(config.model.policy, PolicyKind::CacheAware);
        assert_eq!(
            config
                .model
                .cache_aware
                .as_ref()
                .expect("cache-aware needs indexer config")
                .kv_indexer_endpoint
                .as_ref()
                .map(|indexer| indexer.url.as_str()),
            Some("http://indexer:50051"),
        );
        let indexer_timeout_ms = config
            .model
            .cache_aware
            .as_ref()
            .and_then(|cache| cache.kv_indexer_endpoint.as_ref())
            .expect("cache-aware needs indexer config")
            .query_timeout_ms;
        let affinity = config
            .model
            .affinity
            .expect("cache-aware needs candidate config");
        assert_eq!(affinity.cache_affinity_min_matched_tokens, Some(512));
        assert_eq!(affinity.cache_affinity_min_match_ratio, Some(0.25));
        assert_eq!(affinity.cache_candidate_min_workers, 4);
        assert_eq!(affinity.cache_candidate_ratio, 0.1);
        assert_eq!(affinity.cache_candidate_max_workers, 16);
        assert_eq!(affinity.cache_switch_margin_tokens, 128);
        assert_eq!(indexer_timeout_ms, 40);

        let defaults =
            cfg_of("--policy cache_aware --kv-indexer-endpoint http://indexer:50051").unwrap();
        let defaults_indexer_timeout_ms = defaults
            .model
            .cache_aware
            .as_ref()
            .and_then(|cache| cache.kv_indexer_endpoint.as_ref())
            .expect("default indexer config")
            .query_timeout_ms;
        let defaults_affinity = defaults
            .model
            .affinity
            .expect("default cache candidate config");
        assert_eq!(
            defaults_affinity.cache_affinity_min_matched_tokens,
            Some(1_024)
        );
        assert_eq!(defaults_affinity.cache_affinity_min_match_ratio, None);
        assert_eq!(
            defaults_indexer_timeout_ms,
            DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS
        );

        let err = cfg_of("--policy power_of_two --stable-pair")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("--stable-pair") && err.contains("session_aware"),
            "got: {err}"
        );

        let err =
            cfg_of("--policy cache_aware --kv-indexer-endpoint http://indexer:50051 --stable-pair")
                .expect_err("Cache-Aware has no stable backup")
                .to_string();
        assert!(err.contains("--stable-pair"), "got: {err}");
    }

    #[test]
    fn cache_candidate_cli_rejects_invalid_bounds() {
        for (args, expected) in [
            (
                "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
                 --cache-affinity-min-match-ratio 1.1",
                "--cache-affinity-min-match-ratio",
            ),
            (
                "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
                 --cache-candidate-min-workers 9 --cache-candidate-max-workers 8",
                "--cache-candidate-min-workers",
            ),
            (
                "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
                 --cache-candidate-ratio=-0.1",
                "--cache-candidate-ratio",
            ),
            (
                "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
                 --kv-indexer-query-timeout-ms 0",
                "--kv-indexer-query-timeout-ms",
            ),
        ] {
            let err = cfg_of(args)
                .expect_err("invalid candidate bound")
                .to_string();
            assert!(err.contains(expected), "got: {err}");
        }
    }

    #[test]
    fn rejects_affinity_options_that_cannot_affect_the_selected_policy() {
        let missing_indexer = cfg_of("--policy cache_aware")
            .expect_err("cache_aware without an indexer can only behave like P2")
            .to_string();
        assert!(
            missing_indexer.contains("--kv-indexer-endpoint"),
            "got: {missing_indexer}"
        );
    }
}
