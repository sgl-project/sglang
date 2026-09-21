// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Grouped CLI options and conversion into a validated [`Config`].

use anyhow::{anyhow, ensure, Result};
use clap::Parser;
use std::num::NonZeroU32;

use crate::config::sampling::{parse_sampling_overrides, ConflictPolicy};
use crate::config::{
    default_cb_cool_down, default_host, default_port, default_proxy_request_timeout_secs,
    default_shutdown_drain_secs, default_stale_request_timeout_secs, resolve_mode, AffinityConfig,
    AffinityMode, CacheAwareConfig, CachePrefixProvider, CircuitBreakerConfig, Config,
    DecodePolicyKind, DiscoveryBackend, EligibilityConfig, FilterKind, FusedTerm,
    InflightLoadConfig, K8sDiscoveryConfig, KvIndexerEndpointConfig, LogFormat, ModelConfig,
    ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig, SessionAffinityMode,
    StaticUrlsDiscoveryConfig, StickyConfig, StickyFallbackKind, DEFAULT_FUSE,
};

const DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS: u64 = 100;
const DEFAULT_KV_INDEXER_QUERY_MAX_INFLIGHT: usize = sgl_kv_indexer::DEFAULT_QUERY_MAX_INFLIGHT;

#[derive(Parser, Debug)]
#[command(
    name = "sgl-router",
    version,
    about = "Slim KV-aware OpenAI-compatible router for SGLang workers",
    after_help = "Examples:\n  sgl-router --model-id Qwen/Qwen3-0.6B --worker-urls http://localhost:30001\n  sgl-router --model-id Qwen/Qwen3-0.6B --service-discovery --selector app=sglang\n\nChoose exactly one discovery backend. Policy-specific options name their required policy in the descriptions."
)]
pub struct Cli {
    #[command(flatten, next_help_heading = "Model, tokenizer and sampling")]
    pub model: ModelArgs,
    #[command(flatten, next_help_heading = "Server, timeouts and logging")]
    pub server: ServerArgs,
    #[command(
        flatten,
        next_help_heading = "Worker discovery (static URLs or Kubernetes)"
    )]
    pub discovery: DiscoveryArgs,
    #[command(
        flatten,
        next_help_heading = "Routing policies, admission and circuit breaker"
    )]
    pub routing: RoutingArgs,
    #[command(
        flatten,
        next_help_heading = "Cache-aware routing (--policy cache_aware)"
    )]
    pub cache: CacheArgs,
    #[command(
        flatten,
        next_help_heading = "Session affinity, sticky routing and pressure guards"
    )]
    pub affinity: AffinityArgs,
}

#[derive(clap::Args, Debug)]
pub struct ModelArgs {
    /// Model id this router serves (the OpenAI `model` field).
    #[arg(long)]
    pub model_id: String,

    /// Local tokenizer.json or HuggingFace repo id. Defaults to --model-id; honors HF_TOKEN / HF_HOME.
    #[arg(long)]
    pub tokenizer_path: Option<String>,

    /// Disable generated input_ids; workers tokenize messages, while routing still renders locally.
    /// Use for worker-only thinking defaults, parser/template overrides, or template stop strings.
    #[arg(long)]
    pub disable_input_ids_forwarding: bool,

    /// Fleet sampling defaults as JSON, e.g. {"temperature": 1, "top_p": 0.95}.
    /// Accepts temperature, top_p, top_k, min_p, repetition_penalty,
    /// frequency_penalty, presence_penalty, and n. Numeric values fill absent
    /// fields; {"min": LO, "max": HI} bands only constrain supplied values
    /// and require reject mode. See README for domains and null handling.
    #[arg(long, value_name = "JSON")]
    pub override_sampling_params: Option<String>,

    /// How to handle sampling values that differ from configured defaults:
    /// reject returns 400 before admission; allow forwards the client value.
    /// Defaults to reject. Requires --override-sampling-params.
    #[arg(
        long,
        value_enum,
        value_name = "MODE",
        requires = "override_sampling_params"
    )]
    pub sampling_param_conflict: Option<ConflictPolicy>,
}

#[derive(clap::Args, Debug)]
pub struct ServerArgs {
    /// Address to bind the HTTP server to.
    #[arg(long, default_value_t = default_host())]
    pub host: String,

    /// Port to bind the HTTP server to.
    #[arg(long, default_value_t = default_port())]
    pub port: u16,

    /// Keep serving after SIGTERM with /readyz returning 503 before stopping accepts.
    /// Leave time in the pod grace period for in-flight requests; cover readiness
    /// probe failureThreshold * periodSeconds when probe-driven. 0 disables the pause.
    #[arg(long, default_value_t = default_shutdown_drain_secs())]
    pub shutdown_drain_secs: u64,

    /// Pod terminationGracePeriodSeconds for the startup drain-budget check.
    /// Omitting this assumes 30 seconds; this flag does not change the pod spec.
    #[arg(long)]
    pub termination_grace_secs: Option<u64>,

    /// Per-request upstream timeout in seconds.
    #[arg(long, default_value_t = default_proxy_request_timeout_secs())]
    pub request_timeout_secs: u64,

    /// Max lifetime of an in-flight request entry before the janitor
    /// reaps it (returns 504 `stale_request_expired`).
    #[arg(long, default_value_t = default_stale_request_timeout_secs())]
    pub stale_request_timeout_secs: u64,

    /// Default tracing level (overridden by `RUST_LOG`).
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Log output format.
    #[arg(long, value_enum, default_value = "text")]
    pub log_format: LogFormat,
}

#[derive(clap::Args, Debug)]
pub struct DiscoveryArgs {
    /// Static worker URLs, space-separated or repeated. Conflicts with --service-discovery.
    #[arg(long, num_args = 1..)]
    pub worker_urls: Vec<String>,

    /// Enable Kubernetes EndpointSlice discovery.
    #[arg(long)]
    pub service_discovery: bool,

    /// Namespace to watch. Unset/empty watches all namespaces (requires
    /// cluster-wide RBAC).
    #[arg(long)]
    pub service_discovery_namespace: Option<String>,

    /// Plain-mode label selector terms, AND-joined; Kubernetes selector grammar.
    /// Mutually exclusive with --prefill-selector and --decode-selector.
    #[arg(long, num_args = 1..)]
    pub selector: Vec<String>,

    /// Prefill equality selector terms (key=value or key==value). Requires --decode-selector.
    #[arg(long, num_args = 1..)]
    pub prefill_selector: Vec<String>,

    /// Decode equality selector terms (key=value or key==value). Requires --prefill-selector.
    #[arg(long, num_args = 1..)]
    pub decode_selector: Vec<String>,
}

#[derive(clap::Args, Debug)]
pub struct RoutingArgs {
    /// Routing policy.
    #[arg(long, value_enum, default_value = "round_robin")]
    pub policy: PolicyKind,

    /// Policy used to select decode workers for PD requests.
    #[arg(long, value_enum, default_value = "power_of_two")]
    pub decode_policy: DecodePolicyKind,

    /// Static P/D bucket configuration. Omit to use the global candidate domain.
    #[arg(long)]
    pub bucket_config: Option<String>,

    /// Weighted scoring terms, e.g. prefix_cache=2.0,load_based=0.3.
    /// Defaults to prefix_cache,load_based for score_policy or fused_score.
    /// Requires --policy score_policy or fused_score. Omitted weights use each term's default.
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

    /// Consecutive upstream failures before opening the breaker. Must be positive; enables the breaker.
    #[arg(long)]
    pub cb_threshold: Option<NonZeroU32>,

    /// Circuit-breaker cool-down in seconds. Only meaningful with
    /// `--cb-threshold`; defaults to 30 when the breaker is enabled.
    #[arg(long)]
    pub cb_cool_down_secs: Option<u64>,
}

#[derive(clap::Args, Debug)]
pub struct CacheArgs {
    /// Prefix-match source: indexer when --kv-indexer-endpoint is set, otherwise radix_tree.
    #[arg(long, value_enum)]
    pub cache_prefix_provider: Option<CachePrefixProvider>,

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

    /// Minimum cache-hit tokens for a candidate. Defaults to 1024.
    #[arg(long)]
    pub cache_affinity_min_matched_tokens: Option<u64>,

    /// Minimum cache-hit ratio for a candidate. Unset by default.
    #[arg(long)]
    pub cache_affinity_min_match_ratio: Option<f64>,

    /// Minimum number of cache candidates to try. Defaults to 8.
    #[arg(long)]
    pub cache_candidate_min_workers: Option<usize>,

    /// Fraction of healthy prefill workers considered as cache candidates. Defaults to 0.05.
    #[arg(long)]
    pub cache_candidate_ratio: Option<f64>,

    /// Maximum number of cache candidates to try. Defaults to 32.
    #[arg(long)]
    pub cache_candidate_max_workers: Option<usize>,

    /// Maximum uncached-work difference that pressure may override. Defaults to 1024 tokens.
    #[arg(long)]
    pub cache_switch_margin_tokens: Option<u64>,

    /// Divert cache-affine requests when an engine queue reaches this limit.
    /// Prefer another prefix owner, then the least-loaded worker. Missing fresh
    /// queue data leaves affinity intact. Unset disables; scale with engine --dp-size.
    #[arg(long)]
    pub worker_queue_limit: Option<u64>,

    /// Keep the least-pressured prefix owner when the queue gate rejects all
    /// admitted cache candidates and no worker has a fresh queue below this floor.
    /// Requires --worker-queue-limit; must be positive and at most that limit.
    /// Unset disables; scale with engine --dp-size.
    #[arg(long)]
    pub saturation_queue_floor: Option<u64>,

    /// Min-load fallback sample size. Defaults to 2; requires --policy cache_aware.
    /// Values at least the pool size choose the exact minimum with random ties.
    /// A value of 1 draws uniformly with no backup for admission or pressure guards.
    /// Unlike --cache-candidate-* (prefix owners), this bounds the fallback sample.
    #[arg(long)]
    pub min_load_choices: Option<usize>,
}

#[derive(clap::Args, Debug)]
pub struct AffinityArgs {
    /// Header carrying the session ID for `--policy session_aware`.
    #[arg(long)]
    pub session_id_header: Option<String>,

    /// Session idle timeout in seconds (--policy session_aware). Defaults to 600.
    #[arg(long)]
    pub session_idle_secs: Option<u64>,

    /// Session eviction interval in seconds (--policy session_aware). Defaults to 60.
    #[arg(long)]
    pub session_eviction_interval_secs: Option<u64>,

    /// Use a deterministic session backup (--policy session_aware).
    #[arg(long)]
    pub stable_pair: bool,

    /// Session admission mode (--policy session_aware). Defaults to soft (allow backup selection).
    #[arg(long, value_enum)]
    pub affinity_mode: Option<AffinityMode>,

    /// Session lookup mode (--policy session_aware). Defaults to bucket (search the target bucket).
    #[arg(long, value_enum)]
    pub session_affinity_mode: Option<SessionAffinityMode>,

    /// Routing-key header (--policy sticky). Defaults to x-sgl-routing-key.
    #[arg(long)]
    pub routing_key_header: Option<String>,

    /// Policy for new or missing routing keys (--policy sticky). Defaults to round_robin.
    #[arg(long, value_enum)]
    pub sticky_fallback_policy: Option<StickyFallbackKind>,

    /// Idle timeout in seconds (--policy sticky). Defaults to 600.
    #[arg(long)]
    pub sticky_idle_secs: Option<u64>,

    /// Eviction sweep interval in seconds (--policy sticky).
    /// Defaults to 60.
    #[arg(long)]
    pub sticky_eviction_interval_secs: Option<u64>,

    /// Disable the pressure guard (--policy session_aware or cache_aware).
    #[arg(long)]
    pub disable_pressure_guard: bool,

    /// Pressure-guard token gap (session_aware or cache_aware). Defaults to 1024.
    #[arg(long)]
    pub pressure_abs_threshold_tokens: Option<u64>,

    /// Pressure-guard gap in ms (session_aware or cache_aware), when a queue estimate exists. Unset by default.
    #[arg(long)]
    pub pressure_abs_threshold_ms: Option<f64>,

    /// Pressure-guard token multiplier (session_aware or cache_aware). Defaults to 1.5.
    #[arg(long)]
    pub pressure_rel_threshold: Option<f64>,
}

impl Cli {
    /// Resolve CLI options and validate the resulting configuration.
    pub fn into_config(self) -> Result<Config> {
        let affinity = self
            .affinity
            .build_config(&self.cache, self.routing.policy)?;
        let discovery = self.discovery.into_config()?;
        let bucket_config = self
            .routing
            .bucket_config
            .as_deref()
            .map(load_bucket_config)
            .transpose()?;
        let circuit_breaker = self.routing.build_circuit_breaker()?;
        let cache_aware = self.cache.into_config(self.routing.policy)?;
        let fused = self.routing.build_fused()?;
        let eligibility = self.routing.build_eligibility()?;
        let sticky = self.affinity.into_sticky_config(self.routing.policy)?;
        let sampling_overrides = self
            .model
            .override_sampling_params
            .as_deref()
            .map(|raw| {
                parse_sampling_overrides(
                    raw,
                    self.model.sampling_param_conflict.unwrap_or_default(),
                )
            })
            .transpose()?
            .unwrap_or_default();

        let config = Config {
            server: ServerConfig {
                host: self.server.host,
                port: self.server.port,
                shutdown_drain_secs: self.server.shutdown_drain_secs,
                termination_grace_secs: self.server.termination_grace_secs,
            },
            observability: ObservabilityConfig {
                log_level: self.server.log_level,
                log_format: self.server.log_format,
            },
            model: ModelConfig {
                tokenizer_path: self
                    .model
                    .tokenizer_path
                    .unwrap_or_else(|| self.model.model_id.clone()),
                id: self.model.model_id,
                disable_input_ids_forwarding: self.model.disable_input_ids_forwarding,
                policy: self.routing.policy,
                decode_policy: self.routing.decode_policy,
                bucket_config,
                circuit_breaker,
                cache_aware,
                sticky,
                affinity,
                fused,
                eligibility,
                sampling_overrides,
            },
            discovery,
            proxy: ProxyConfig {
                request_timeout_secs: self.server.request_timeout_secs,
            },
            router_inflight_load: InflightLoadConfig {
                stale_request_timeout_secs: self.server.stale_request_timeout_secs,
            },
        };
        config.validate()?;
        Ok(config)
    }
}

impl DiscoveryArgs {
    fn into_config(self) -> Result<DiscoveryBackend> {
        let has_static = !self.worker_urls.is_empty();
        let backend = match (has_static, self.service_discovery) {
            (true, true) => {
                return Err(anyhow!(
                    "--worker-urls and --service-discovery are mutually exclusive; pass exactly one"
                ));
            }
            (false, false) => {
                return Err(anyhow!(
                    "no discovery backend selected; pass --worker-urls <URL...> (static) \
                     or --service-discovery (kubernetes)"
                ));
            }
            (true, false) => {
                ensure!(
                    self.service_discovery_namespace.is_none()
                        && self.selector.is_empty()
                        && self.prefill_selector.is_empty()
                        && self.decode_selector.is_empty(),
                    "--service-discovery-namespace / --selector / --prefill-selector / \
                         --decode-selector require --service-discovery"
                );
                DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                    urls: self.worker_urls,
                })
            }
            (false, true) => {
                let mode = resolve_mode(
                    join_selector(&self.selector).as_deref(),
                    join_selector(&self.prefill_selector).as_deref(),
                    join_selector(&self.decode_selector).as_deref(),
                )?;
                DiscoveryBackend::K8s(K8sDiscoveryConfig {
                    namespace: self.service_discovery_namespace.unwrap_or_default(),
                    mode,
                })
            }
        };
        Ok(backend)
    }
}

impl RoutingArgs {
    fn build_circuit_breaker(&self) -> Result<Option<CircuitBreakerConfig>> {
        ensure!(
            self.cb_cool_down_secs.is_none() || self.cb_threshold.is_some(),
            "--cb-cool-down-secs requires --cb-threshold (the circuit breaker is \
                 enabled by --cb-threshold)"
        );
        let circuit_breaker = self.cb_threshold.map(|threshold| CircuitBreakerConfig {
            threshold,
            cool_down_secs: self.cb_cool_down_secs.unwrap_or_else(default_cb_cool_down),
        });

        Ok(circuit_breaker)
    }

    fn build_fused(&self) -> Result<Option<Vec<FusedTerm>>> {
        let is_score_composition = matches!(
            self.policy,
            PolicyKind::FusedScore | PolicyKind::ScorePolicy
        );
        ensure!(
            self.fuse.is_empty() || is_score_composition,
            "--fuse requires --policy score_policy or fused_score"
        );
        if !is_score_composition {
            return Ok(None);
        }
        let terms = if self.fuse.is_empty() {
            DEFAULT_FUSE
                .iter()
                .map(|&kind| FusedTerm { kind, weight: None })
                .collect()
        } else {
            self.fuse.clone()
        };
        for (i, t) in terms.iter().enumerate() {
            ensure!(
                !terms[..i].iter().any(|p| p.kind == t.kind),
                "--fuse: `{}` is listed more than once",
                t.kind
            );
        }
        Ok(Some(terms))
    }

    fn build_eligibility(&self) -> Result<Option<EligibilityConfig>> {
        for (i, kind) in self.filter.iter().enumerate() {
            ensure!(
                !self.filter[..i].contains(kind),
                "--filter: `{kind}` is listed more than once"
            );
        }
        let has = |k: FilterKind| self.filter.contains(&k);
        ensure!(
            (self.max_in_flight.is_some() == has(FilterKind::Overloaded)),
            "--max-in-flight and `--filter overloaded` require each other"
        );
        ensure!(
            self.max_in_flight != Some(0),
            "--max-in-flight must be greater than 0"
        );
        ensure!(
            (self.prefix_cache_min_share.is_some() == has(FilterKind::PrefixCache)),
            "--prefix-cache-min-share and `--filter prefix_cache` require each other"
        );
        ensure!(
            self.prefix_cache_min_share
                .is_none_or(|s| s > 0.0 && s <= 1.0),
            "--prefix-cache-min-share must be in (0, 1]"
        );
        ensure!(
            self.policy != PolicyKind::Sticky || self.filter.is_empty(),
            "--filter cannot be combined with --policy sticky"
        );
        let eligibility = (!self.filter.is_empty()).then_some(EligibilityConfig {
            filters: self.filter.clone(),
            max_in_flight: self.max_in_flight,
            min_prefix_share: self.prefix_cache_min_share,
        });

        Ok(eligibility)
    }
}

impl CacheArgs {
    fn into_config(self, policy: PolicyKind) -> Result<Option<CacheAwareConfig>> {
        let cache_prefix_provider = self.cache_prefix_provider.unwrap_or_else(|| {
            if self.kv_indexer_endpoint.is_some() {
                CachePrefixProvider::Indexer
            } else {
                CachePrefixProvider::RadixTree
            }
        });
        ensure!(
            self.cache_prefix_provider.is_none() || policy == PolicyKind::CacheAware,
            "--cache-prefix-provider requires --policy cache_aware"
        );
        ensure!(
            self.kv_indexer_query_timeout_ms != Some(0),
            "--kv-indexer-query-timeout-ms must be greater than zero"
        );
        ensure!(
            self.kv_indexer_query_timeout_ms.is_none() || self.kv_indexer_endpoint.is_some(),
            "--kv-indexer-query-timeout-ms requires --kv-indexer-endpoint"
        );
        ensure!(
            self.kv_indexer_query_max_inflight != Some(0),
            "--kv-indexer-query-max-inflight must be greater than zero"
        );
        ensure!(
            self.kv_indexer_query_max_inflight.is_none() || self.kv_indexer_endpoint.is_some(),
            "--kv-indexer-query-max-inflight requires --kv-indexer-endpoint"
        );
        let cache_aware_uses_indexer = policy == PolicyKind::CacheAware
            && cache_prefix_provider == CachePrefixProvider::Indexer;
        if self.kv_indexer_endpoint.is_some() && !cache_aware_uses_indexer {
            return Err(if policy == PolicyKind::CacheAware {
                anyhow!("--kv-indexer-endpoint requires --cache-prefix-provider indexer")
            } else {
                anyhow!("--kv-indexer-endpoint requires --policy cache_aware")
            });
        }
        ensure!(
            !cache_aware_uses_indexer || self.kv_indexer_endpoint.is_some(),
            "--cache-prefix-provider indexer requires --kv-indexer-endpoint"
        );
        let kv_indexer_query_timeout_ms = self
            .kv_indexer_query_timeout_ms
            .unwrap_or(DEFAULT_KV_INDEXER_QUERY_TIMEOUT_MS);
        let kv_indexer_query_max_inflight = self
            .kv_indexer_query_max_inflight
            .unwrap_or(DEFAULT_KV_INDEXER_QUERY_MAX_INFLIGHT);
        if policy != PolicyKind::CacheAware {
            return Ok(None);
        }
        let kv_indexer_endpoint = self.kv_indexer_endpoint.map(|url| KvIndexerEndpointConfig {
            url,
            query_timeout_ms: kv_indexer_query_timeout_ms,
            query_max_inflight: kv_indexer_query_max_inflight,
        });
        Ok(Some(CacheAwareConfig {
            prefix_provider: cache_prefix_provider,
            kv_indexer_endpoint,
        }))
    }
}

impl AffinityArgs {
    fn build_config(
        &self,
        cache: &CacheArgs,
        policy: PolicyKind,
    ) -> Result<Option<AffinityConfig>> {
        let affinity_policy = matches!(policy, PolicyKind::SessionAware | PolicyKind::CacheAware);
        let tuned_session_affinity = self.session_id_header.is_some()
            || self.session_idle_secs.is_some()
            || self.session_eviction_interval_secs.is_some()
            || self.stable_pair
            || self.affinity_mode.is_some()
            || self.session_affinity_mode.is_some();
        ensure!(
            !tuned_session_affinity || policy == PolicyKind::SessionAware,
            "--session-id-header, --session-*-secs, --stable-pair, --affinity-mode, and \
                 --session-affinity-mode require --policy session_aware"
        );
        ensure!(
            !self.disable_pressure_guard || affinity_policy,
            "--disable-pressure-guard requires --policy session_aware or cache_aware"
        );
        let tuned_cache_candidates = cache.cache_affinity_min_matched_tokens.is_some()
            || cache.cache_affinity_min_match_ratio.is_some()
            || cache.cache_candidate_min_workers.is_some()
            || cache.cache_candidate_ratio.is_some()
            || cache.cache_candidate_max_workers.is_some()
            || cache.cache_switch_margin_tokens.is_some()
            || cache.worker_queue_limit.is_some()
            || cache.saturation_queue_floor.is_some()
            || cache.min_load_choices.is_some();
        // Value checks before the policy check: a value that is wrong under
        // every policy should say so, rather than pointing at --policy.
        ensure!(
            cache.worker_queue_limit != Some(0),
            "--worker-queue-limit must be at least 1"
        );
        if let Some(floor) = cache.saturation_queue_floor {
            // The floor modifies the gate's diversion; without the gate
            // there is no diversion to cancel and the knob would sit dead.
            let Some(limit) = cache.worker_queue_limit else {
                return Err(anyhow!(
                    "--saturation-queue-floor requires --worker-queue-limit (there is no \
                     diversion to cancel without it)"
                ));
            };
            ensure!(floor != 0, "--saturation-queue-floor must be at least 1");
            ensure!(
                floor <= limit,
                "--saturation-queue-floor ({floor}) must be at most --worker-queue-limit \
                     ({limit})"
            );
        }
        ensure!(
            cache.min_load_choices != Some(0),
            "--min-load-choices must be at least 1"
        );
        ensure!(
            !tuned_cache_candidates || policy == PolicyKind::CacheAware,
            "cache candidate tuning flags require --policy cache_aware"
        );
        ensure!(
            affinity_policy
                || (self.pressure_abs_threshold_tokens.is_none()
                    && self.pressure_abs_threshold_ms.is_none()
                    && self.pressure_rel_threshold.is_none()),
            "pressure guard tuning requires --policy session_aware or cache_aware"
        );
        if !affinity_policy {
            return Ok(None);
        }
        let defaults = AffinityConfig::default();
        let session_id_header = self
            .session_id_header
            .clone()
            .unwrap_or(defaults.session_id_header);
        axum::http::HeaderName::try_from(session_id_header.as_str()).map_err(|e| {
            anyhow!(
                "--session-id-header {session_id_header:?} is not a valid HTTP header name: {e}"
            )
        })?;
        let pressure_rel_threshold = self
            .pressure_rel_threshold
            .unwrap_or(defaults.pressure_rel_threshold);
        ensure!(
            pressure_rel_threshold.is_finite() && pressure_rel_threshold > 1.0,
            "--pressure-rel-threshold must be finite and greater than 1"
        );
        ensure!(
            self.pressure_abs_threshold_ms
                .is_none_or(|threshold| threshold.is_finite() && threshold >= 0.0),
            "--pressure-abs-threshold-ms must be finite and non-negative"
        );
        let cache_affinity_min_match_ratio = cache
            .cache_affinity_min_match_ratio
            .or(defaults.cache_affinity_min_match_ratio);
        ensure!(
            cache_affinity_min_match_ratio
                .is_none_or(|ratio| ratio.is_finite() && (0.0..=1.0).contains(&ratio)),
            "--cache-affinity-min-match-ratio must be finite and in [0, 1]"
        );
        let cache_candidate_ratio = cache
            .cache_candidate_ratio
            .unwrap_or(defaults.cache_candidate_ratio);
        ensure!(
            cache_candidate_ratio.is_finite() && (0.0..=1.0).contains(&cache_candidate_ratio),
            "--cache-candidate-ratio must be finite and in [0, 1]"
        );
        let cache_candidate_min_workers = cache
            .cache_candidate_min_workers
            .unwrap_or(defaults.cache_candidate_min_workers);
        let cache_candidate_max_workers = cache
            .cache_candidate_max_workers
            .unwrap_or(defaults.cache_candidate_max_workers);
        ensure!(
            cache_candidate_min_workers > 0
                && cache_candidate_max_workers > 0
                && cache_candidate_min_workers <= cache_candidate_max_workers,
            "--cache-candidate-min-workers and --cache-candidate-max-workers must be \
                 positive and min must not exceed max"
        );
        let session_idle_secs = self.session_idle_secs.unwrap_or(defaults.session_idle_secs);
        let session_eviction_interval_secs = self
            .session_eviction_interval_secs
            .unwrap_or(defaults.session_eviction_interval_secs);
        ensure!(
            session_idle_secs != 0,
            "--session-idle-secs must be greater than 0"
        );
        ensure!(
            session_eviction_interval_secs != 0,
            "--session-eviction-interval-secs must be greater than 0"
        );
        Ok(Some(AffinityConfig {
            session_id_header,
            session_idle_secs,
            session_eviction_interval_secs,
            stable_pair: self.stable_pair,
            mode: self.affinity_mode.unwrap_or(defaults.mode),
            session_affinity_mode: self
                .session_affinity_mode
                .unwrap_or(defaults.session_affinity_mode),
            pressure_guard: !self.disable_pressure_guard && defaults.pressure_guard,
            pressure_abs_threshold_tokens: self
                .pressure_abs_threshold_tokens
                .unwrap_or(defaults.pressure_abs_threshold_tokens),
            pressure_abs_threshold_ms: self
                .pressure_abs_threshold_ms
                .or(defaults.pressure_abs_threshold_ms),
            pressure_rel_threshold,
            cache_affinity_min_matched_tokens: cache
                .cache_affinity_min_matched_tokens
                .or(defaults.cache_affinity_min_matched_tokens),
            cache_affinity_min_match_ratio,
            cache_candidate_min_workers,
            cache_candidate_ratio,
            cache_candidate_max_workers,
            cache_switch_margin_tokens: cache
                .cache_switch_margin_tokens
                .unwrap_or(defaults.cache_switch_margin_tokens),
            worker_queue_limit: cache.worker_queue_limit.or(defaults.worker_queue_limit),
            saturation_queue_floor: cache
                .saturation_queue_floor
                .or(defaults.saturation_queue_floor),
            min_load_choices: cache.min_load_choices.unwrap_or(defaults.min_load_choices),
        }))
    }

    fn into_sticky_config(self, policy: PolicyKind) -> Result<Option<StickyConfig>> {
        let tuned_sticky = self.routing_key_header.is_some()
            || self.sticky_fallback_policy.is_some()
            || self.sticky_idle_secs.is_some()
            || self.sticky_eviction_interval_secs.is_some();
        ensure!(
            !tuned_sticky || policy == PolicyKind::Sticky,
            "--routing-key-header / --sticky-fallback-policy / --sticky-idle-secs / \
                 --sticky-eviction-interval-secs require --policy sticky"
        );

        if policy != PolicyKind::Sticky {
            return Ok(None);
        }
        let defaults = StickyConfig::default();
        let header_name = self.routing_key_header.unwrap_or(defaults.header_name);
        axum::http::HeaderName::try_from(header_name.as_str()).map_err(|e| {
            anyhow!("--routing-key-header {header_name:?} is not a valid HTTP header name: {e}")
        })?;
        let fallback_policy = self
            .sticky_fallback_policy
            .unwrap_or(defaults.fallback_policy);
        let idle_secs = self.sticky_idle_secs.unwrap_or(defaults.idle_secs);
        let eviction_interval_secs = self
            .sticky_eviction_interval_secs
            .unwrap_or(defaults.eviction_interval_secs);
        ensure!(
            eviction_interval_secs != 0,
            "--sticky-eviction-interval-secs must be greater than 0"
        );
        ensure!(
            idle_secs != 0,
            "--sticky-idle-secs must be greater than 0 (0 would evict every \
                 assignment immediately, defeating sticky routing)"
        );
        Ok(Some(StickyConfig {
            header_name,
            fallback_policy,
            idle_secs,
            eviction_interval_secs,
        }))
    }
}

fn load_bucket_config(path: &str) -> Result<crate::config::BucketConfig> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| anyhow!("--bucket-config cannot read {path:?}: {error}"))?;
    serde_json::from_str(&raw)
        .map_err(|error| anyhow!("--bucket-config {path:?} is not valid JSON: {error}"))
}

fn join_selector(terms: &[String]) -> Option<String> {
    (!terms.is_empty()).then(|| terms.join(","))
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
    fn help_groups_options_by_purpose() {
        use clap::CommandFactory;

        Cli::command().debug_assert();
        for long in [false, true] {
            let mut command = Cli::command();
            let help = if long {
                command.render_long_help()
            } else {
                command.render_help()
            }
            .to_string();
            let headings: std::collections::HashSet<_> = command
                .get_arguments()
                .filter_map(|arg| arg.get_help_heading())
                .collect();
            assert_eq!(
                headings.len(),
                6,
                "keep related options in six broad groups"
            );
            for arg in command
                .get_arguments()
                .filter(|arg| !matches!(arg.get_id().as_str(), "help" | "version"))
            {
                let heading = arg.get_help_heading().expect("every option has a group");
                assert!(
                    arg.get_help().is_some(),
                    "missing help for {}",
                    arg.get_id()
                );
                let section = help
                    .split_once(&format!("{heading}:\n"))
                    .unwrap_or_else(|| panic!("missing help section: {heading}"))
                    .1
                    .split("\n\n")
                    .next()
                    .unwrap();
                // Long help separates individual options with blank lines.
                if !long {
                    assert!(
                        section.contains(&format!("--{}", arg.get_long().unwrap())),
                        "{} is outside {heading}",
                        arg.get_id()
                    );
                }
            }
            assert!(help.contains("Examples:"));
            assert!(help.contains("Choose exactly one discovery backend"));
        }
    }

    #[test]
    fn defaults_host_port_and_policy() {
        let c = into_config_owned(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
        assert_eq!(c.server.host, "127.0.0.1");
        assert_eq!(c.server.port, 30000);
        assert_eq!(c.model.policy, PolicyKind::RoundRobin);
        assert_eq!(c.model.id, "qwen3-0.6b");
        assert_eq!(c.proxy.request_timeout_secs, 300);
        assert_eq!(c.router_inflight_load.stale_request_timeout_secs, 600);
        assert_eq!(c.server.shutdown_drain_secs, 30);
    }

    /// Several values, not just the default: a clamp or a rescale in the
    /// mapping satisfies any single-value assertion.
    #[test]
    fn shutdown_drain_secs_maps_into_config() {
        for secs in ["0", "17", "1800"] {
            let c = into_config_owned(with_model(&[
                "--worker-urls",
                "http://10.0.0.1:30000",
                "--shutdown-drain-secs",
                secs,
            ]))
            .unwrap();
            let expected: u64 = secs.parse().unwrap();
            assert_eq!(
                c.server.shutdown_drain_secs, expected,
                "--shutdown-drain-secs {secs} must map through unchanged",
            );
            assert_eq!(
                c.server.shutdown_drain(),
                std::time::Duration::from_secs(expected),
                "the Duration accessor must agree with the configured seconds",
            );
        }
    }

    #[test]
    fn shutdown_drain_secs_past_the_ceiling_is_rejected() {
        let error = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--shutdown-drain-secs",
            "18000",
        ]))
        .expect_err("a drain past the ceiling must not start")
        .to_string();
        assert!(
            error.contains("shutdown_drain_secs"),
            "the error must name the flag to fix: {error}"
        );
    }

    #[test]
    fn termination_grace_secs_maps_into_config_and_defaults_to_none() {
        let c = into_config_owned(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
        assert_eq!(c.server.termination_grace_secs, None);

        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--termination-grace-secs",
            "120",
        ]))
        .unwrap();
        assert_eq!(c.server.termination_grace_secs, Some(120));
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
    fn rejects_removed_cache_aware_zmq_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("cache_aware_zmq"), "got: {err}");
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
        assert_eq!(c.router_inflight_load.stale_request_timeout_secs, 240);
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
            .split_once("[possible values:")
            .expect("sticky fallback lists its choices")
            .1
            .split_once(']')
            .unwrap()
            .0;

        for value in ["round_robin", "random", "power_of_two", "load_based"] {
            assert!(choices.contains(value), "missing {value}: {choices}");
        }
        for value in [
            "fused_score",
            "score_policy",
            "session_aware",
            "cache_aware",
            "sticky",
        ] {
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
    fn rejects_cache_aware_as_sticky_fallback() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "sticky",
            "--sticky-fallback-policy",
            "cache_aware",
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
    fn native_cache_pressure_flags_build_the_guard_contract() {
        let config = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --disable-pressure-guard --pressure-abs-threshold-tokens 2048 \
             --pressure-abs-threshold-ms 3.5 --pressure-rel-threshold 2.0",
        )
        .unwrap();
        let affinity = config
            .model
            .affinity
            .expect("cache-aware needs affinity config");
        assert!(!affinity.pressure_guard);
        assert_eq!(affinity.pressure_abs_threshold_tokens, 2_048);
        assert_eq!(affinity.pressure_abs_threshold_ms, Some(3.5));
        assert_eq!(affinity.pressure_rel_threshold, 2.0);
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
    fn worker_queue_limit_requires_cache_aware_and_a_positive_value() {
        let config = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --worker-queue-limit 4",
        )
        .unwrap();
        assert_eq!(
            config
                .model
                .affinity
                .expect("cache-aware needs affinity config")
                .worker_queue_limit,
            Some(4)
        );

        // Unset, the gate is disabled.
        let defaults =
            cfg_of("--policy cache_aware --kv-indexer-endpoint http://indexer:50051").unwrap();
        assert_eq!(
            defaults
                .model
                .affinity
                .expect("default affinity config")
                .worker_queue_limit,
            None
        );

        let err = cfg_of("--policy power_of_two --worker-queue-limit 4")
            .expect_err("the gate only governs cache-affinity selection")
            .to_string();
        assert!(
            err.contains("cache candidate tuning flags require --policy cache_aware"),
            "got: {err}"
        );

        let err = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --worker-queue-limit 0",
        )
        .expect_err("a zero limit would reject every queue reading")
        .to_string();
        assert!(err.contains("--worker-queue-limit"), "got: {err}");

        // A zero limit is wrong under every policy, so the value error must
        // win over the policy error rather than being masked by it.
        let err = cfg_of("--policy power_of_two --worker-queue-limit 0")
            .expect_err("a zero limit is rejected regardless of policy")
            .to_string();
        assert!(err.contains("--worker-queue-limit"), "got: {err}");
    }

    #[test]
    fn saturation_queue_floor_requires_the_queue_gate_and_stays_below_it() {
        let config = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --worker-queue-limit 4 --saturation-queue-floor 2",
        )
        .unwrap();
        assert_eq!(
            config
                .model
                .affinity
                .expect("cache-aware needs affinity config")
                .saturation_queue_floor,
            Some(2)
        );

        // Unset, the pin is disabled and the gate behaves as before.
        let defaults = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --worker-queue-limit 4",
        )
        .unwrap();
        assert_eq!(
            defaults
                .model
                .affinity
                .expect("default affinity config")
                .saturation_queue_floor,
            None
        );

        // Without the gate there is no diversion to cancel.
        let err = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --saturation-queue-floor 2",
        )
        .expect_err("the floor modifies the gate's diversion")
        .to_string();
        assert!(err.contains("--saturation-queue-floor"), "got: {err}");
        assert!(err.contains("--worker-queue-limit"), "got: {err}");

        // A floor above the limit would declare saturation while workers
        // the gate still admits exist.
        let err = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --worker-queue-limit 4 --saturation-queue-floor 5",
        )
        .expect_err("floor must not exceed the limit")
        .to_string();
        assert!(err.contains("at most"), "got: {err}");

        let err = cfg_of(
            "--policy cache_aware --kv-indexer-endpoint http://indexer:50051 \
             --worker-queue-limit 4 --saturation-queue-floor 0",
        )
        .expect_err("a zero floor would reject every queue reading")
        .to_string();
        assert!(err.contains("--saturation-queue-floor"), "got: {err}");

        let err = cfg_of("--policy power_of_two --worker-queue-limit 4 --saturation-queue-floor 2")
            .expect_err("the pin only governs cache-affinity selection")
            .to_string();
        assert!(
            err.contains("cache candidate tuning flags require --policy cache_aware"),
            "got: {err}"
        );
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
    fn cache_aware_defaults_to_router_radix_tree() {
        let config = cfg_of("--policy cache_aware")
            .expect("native cache-aware should not require an Indexer endpoint");
        let cache = config
            .model
            .cache_aware
            .expect("native cache-aware needs its default configuration");
        assert_eq!(cache.prefix_provider, CachePrefixProvider::RadixTree);
        assert!(cache.kv_indexer_endpoint.is_none());
    }

    #[test]
    fn decode_policy_defaults_to_p2_and_accepts_legacy_compatibility_mode() {
        let default_config = cfg_of("--policy power_of_two").unwrap();
        assert_eq!(
            default_config.model.decode_policy,
            DecodePolicyKind::PowerOfTwo
        );

        let legacy_config = cfg_of("--decode-policy legacy_host_affinity").unwrap();
        assert_eq!(
            legacy_config.model.decode_policy,
            DecodePolicyKind::LegacyHostAffinity
        );
    }

    #[test]
    fn bucket_config_json_is_loaded_and_validated_at_startup() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            r#"{
                "ttft_slo_policy": "slo_first",
                "tps_slo_policy": "best_effort",
                "buckets": [
                    {
                        "id": "p-fast",
                        "stage": "prefill",
                        "rank": 10,
                        "worker_ids": ["http://worker:30000"],
                        "max_extend_tokens": 4096,
                        "max_context_tokens": 8192,
                        "ttft_p95_at_capacity_ms": 120
                    }
                ]
            }"#,
        )
        .unwrap();
        let path = file.path().to_str().unwrap().to_string();
        let config = into_config_owned(with_model(&[
            "--worker-urls",
            "http://worker:30000",
            "--bucket-config",
            &path,
        ]))
        .unwrap();

        let buckets = config
            .model
            .bucket_config
            .expect("Bucket config must be retained");
        assert_eq!(buckets.buckets.len(), 1);
        assert_eq!(buckets.buckets[0].id, "p-fast");
        assert_eq!(
            buckets.ttft_slo_policy,
            crate::config::SloBucketPolicy::SloFirst
        );
        assert_eq!(
            buckets.tps_slo_policy,
            crate::config::SloBucketPolicy::BestEffort
        );
    }

    /// The flag reaches `ModelConfig`, and is opt-in: unset leaves the model
    /// with an empty sampling contract, so no request is ever checked.
    #[test]
    fn override_sampling_params_reaches_the_model_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--override-sampling-params",
            r#"{"temperature": 1, "top_p": 0.95}"#,
        ]))
        .unwrap();
        assert_eq!(c.model.sampling_overrides.params.len(), 2);
        // `reject` is the default mode: declaring a contract is the usual
        // reason to declare one.
        assert_eq!(c.model.sampling_overrides.conflict, ConflictPolicy::Reject);

        let c = into_config_owned(with_model(&["--worker-urls", "http://x:30000"])).unwrap();
        assert!(c.model.sampling_overrides.params.is_empty());
    }

    #[test]
    fn sampling_param_conflict_selects_the_mode() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--override-sampling-params",
            r#"{"temperature": 1}"#,
            "--sampling-param-conflict",
            "allow",
        ]))
        .unwrap();
        assert_eq!(c.model.sampling_overrides.conflict, ConflictPolicy::Allow);

        // The mode alone governs nothing, so clap rejects it (`requires`).
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--sampling-param-conflict",
            "reject",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("--override-sampling-params"), "got: {err}");
    }

    /// A malformed contract fails the launch with the parser's own message,
    /// rather than starting a router that 400s every request at the engine.
    #[test]
    fn malformed_override_sampling_params_fails_the_launch() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--override-sampling-params",
            r#"{"temp": 1}"#,
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("unknown parameter"), "got: {err}");
    }
    #[test]
    fn input_ids_forwarding_can_be_disabled_for_the_model() {
        let defaults = into_config_owned(with_model(&["--worker-urls", "http://x:30000"])).unwrap();
        assert!(!defaults.model.disable_input_ids_forwarding);
        let disabled = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--disable-input-ids-forwarding",
        ]))
        .unwrap();
        assert!(disabled.model.disable_input_ids_forwarding);
    }

    #[test]
    fn min_load_choices_is_plumbed_and_validated() {
        let config = cfg_of("--policy cache_aware --min-load-choices 5").unwrap();
        assert_eq!(
            config
                .model
                .affinity
                .expect("cache-aware needs affinity config")
                .min_load_choices,
            5
        );

        // Unset keeps the pre-existing power-of-2 behavior.
        let defaults = cfg_of("--policy cache_aware").unwrap();
        assert_eq!(
            defaults
                .model
                .affinity
                .expect("default affinity config")
                .min_load_choices,
            2
        );

        let err = cfg_of("--policy cache_aware --min-load-choices 0")
            .expect_err("a zero sample size would select nothing")
            .to_string();
        assert!(err.contains("--min-load-choices"), "got: {err}");

        let err = cfg_of("--policy power_of_two --min-load-choices 3")
            .expect_err("the knob only tunes the cache-aware fallback")
            .to_string();
        assert!(
            err.contains("cache candidate tuning flags require --policy cache_aware"),
            "got: {err}"
        );
    }
}
