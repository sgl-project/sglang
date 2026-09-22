use crate::config::sampling::SamplingOverrides;
use serde::Deserialize;
use std::num::NonZeroU32;

/// Single-model configuration built and validated by [`crate::config::Cli::into_config`].
#[derive(Debug, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub observability: ObservabilityConfig,
    pub model: ModelConfig,
    /// Discovery mode resolved from CLI options; static URLs are checked by [`Config::validate`].
    pub discovery: DiscoveryBackend,
    pub proxy: ProxyConfig,
    pub router_inflight_load: InflightLoadConfig,
}

/// Outbound request timeout settings.
#[derive(Debug, Clone, Copy)]
pub struct ProxyConfig {
    /// Timeout for upstream response headers and body. Counts as a circuit-breaker failure.
    pub request_timeout_secs: u64,
    /// Maximum silence between streamed upstream chunks before the stream fails.
    pub stream_idle_timeout_secs: u64,
}

pub fn default_proxy_request_timeout_secs() -> u64 {
    300
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: default_proxy_request_timeout_secs(),
            stream_idle_timeout_secs: 180,
        }
    }
}

/// Request-tracking timeout; defaults above the proxy timeout.
#[derive(Debug, Clone, Copy)]
pub struct InflightLoadConfig {
    /// Maximum request-entry lifetime before cancellation with 504 `stale_request_expired`.
    pub stale_request_timeout_secs: u64,
}

pub fn default_stale_request_timeout_secs() -> u64 {
    600
}

impl Default for InflightLoadConfig {
    fn default() -> Self {
        Self {
            stale_request_timeout_secs: default_stale_request_timeout_secs(),
        }
    }
}

/// Routing strategies accepted by `--policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum PolicyKind {
    #[default]
    #[value(name = "round_robin")]
    RoundRobin,
    #[value(name = "random")]
    Random,
    #[value(name = "power_of_two")]
    PowerOfTwo,
    /// Selects the currently least-loaded worker.
    #[value(name = "load_based")]
    LoadBased,
    /// Weighted sum of `--fuse` terms.
    #[value(name = "fused_score")]
    FusedScore,
    /// Composes compatible scoring terms into a single routing policy.
    #[value(name = "score_policy")]
    ScorePolicy,
    /// Selects a worker from session affinity.
    #[value(name = "session_aware")]
    SessionAware,
    /// Selects cache-affine prefill candidates from the configured prefix provider.
    #[value(name = "cache_aware")]
    CacheAware,
    /// Pin a request-header routing key to a worker.
    #[value(name = "sticky")]
    Sticky,
}

/// Policy used to select decode workers for PD requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum DecodePolicyKind {
    #[default]
    #[value(name = "power_of_two")]
    PowerOfTwo,
    #[value(name = "legacy_host_affinity")]
    LegacyHostAffinity,
}

/// Role served by a static bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BucketStage {
    Prefill,
    Decode,
}

/// SLO matching rules for a bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloBucketPolicy {
    #[default]
    Disabled,
    BestEffort,
    SloFirst,
}

/// Static bucket configuration loaded at Router startup.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucketConfig {
    pub buckets: Vec<BucketSpec>,
    #[serde(default)]
    pub ttft_slo_policy: SloBucketPolicy,
    #[serde(default)]
    pub tps_slo_policy: SloBucketPolicy,
}

/// Runtime capacity assigned to one role. Lower ranks have higher priority.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucketSpec {
    pub id: String,
    pub stage: BucketStage,
    pub rank: u32,
    pub worker_ids: Vec<String>,
    #[serde(default)]
    pub min_extend_tokens: Option<u64>,
    #[serde(default)]
    pub max_extend_tokens: Option<u64>,
    #[serde(default)]
    pub min_sequence_tokens: Option<u64>,
    #[serde(default)]
    pub max_sequence_tokens: Option<u64>,
    #[serde(default)]
    pub max_context_tokens: Option<u64>,
    #[serde(default)]
    pub ttft_p95_at_capacity_ms: Option<u64>,
    #[serde(default)]
    pub tps_p05_at_capacity: Option<f64>,
    #[serde(default)]
    pub max_pending_prefill_tokens: Option<u64>,
}

impl std::fmt::Display for PolicyKind {
    /// The CLI spelling for this policy kind.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = <Self as clap::ValueEnum>::to_possible_value(self)
            .expect("PolicyKind skips no variants");
        f.write_str(v.get_name())
    }
}

/// A hard admission constraint accepted by `--filter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FilterKind {
    /// Router-local in-flight capacity limit.
    #[value(name = "overloaded")]
    Overloaded,
    /// Requires a minimum share of cached prompt blocks.
    #[value(name = "prefix_cache")]
    PrefixCache,
}

impl std::fmt::Display for FilterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = <Self as clap::ValueEnum>::to_possible_value(self)
            .expect("FilterKind skips no variants");
        f.write_str(v.get_name())
    }
}

/// A soft scoring term accepted by `--fuse`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ScoreTermKind {
    /// Independent uniform-random preference.
    #[value(name = "random")]
    Random,
    /// Prefers the least router-local active load.
    #[value(name = "load_based")]
    LoadBased,
    /// Prefers the largest local prefix-cache overlap.
    #[value(name = "prefix_cache")]
    PrefixCache,
}

impl std::fmt::Display for ScoreTermKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = <Self as clap::ValueEnum>::to_possible_value(self)
            .expect("ScoreTermKind skips no variants");
        f.write_str(v.get_name())
    }
}

/// Policy choices that can initialize or handle a keyless sticky request.
/// These policies have no request-scoped cache or sticky-state dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum StickyFallbackKind {
    #[value(name = "round_robin")]
    RoundRobin,
    #[value(name = "random")]
    Random,
    #[value(name = "power_of_two")]
    PowerOfTwo,
    #[value(name = "load_based")]
    LoadBased,
}

impl std::fmt::Display for StickyFallbackKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = <Self as clap::ValueEnum>::to_possible_value(self)
            .expect("StickyFallbackKind skips no variants");
        f.write_str(v.get_name())
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Pause after SIGTERM with `/readyz` returning 503 before stopping accepts.
    /// Allows endpoint removal or readiness-probe failures to reach load balancers.
    /// Leave time in the pod grace period for in-flight draining; 0 disables the pause.
    pub shutdown_drain_secs: u64,
    /// Declared pod termination grace period; `None` uses the Kubernetes default for advisories.
    pub termination_grace_secs: Option<u64>,
}

impl ServerConfig {
    /// Shutdown pause as a duration.
    pub fn shutdown_drain(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.shutdown_drain_secs)
    }
}

pub fn default_host() -> String {
    "127.0.0.1".into()
}

pub fn default_port() -> u16 {
    30000
}

pub fn default_shutdown_drain_secs() -> u64 {
    30
}

/// Defaults for config construction; the CLI mapping remains exhaustive.
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            shutdown_drain_secs: default_shutdown_drain_secs(),
            termination_grace_secs: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub log_level: String,
    /// Tracing output format.
    pub log_format: LogFormat,
}

/// `text` for human-readable dev output, `json` for one-line-per-record
/// JSON suitable for k8s log aggregators (fluent-bit / vector / Loki).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum LogFormat {
    #[default]
    #[value(name = "text")]
    Text,
    #[value(name = "json")]
    Json,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            log_format: LogFormat::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub id: String,
    /// Local tokenizer.json or HuggingFace repo id; defaults to `id`.
    /// Resolved by [`crate::tokenizer::adapter::load`].
    pub tokenizer_path: String,
    /// Disable router-generated input IDs for this model; keep routing tokenization.
    /// Use when workers have rendering defaults or template stops the router cannot see.
    pub disable_input_ids_forwarding: bool,
    pub policy: PolicyKind,
    /// Selection policy for the decode pool.
    pub decode_policy: DecodePolicyKind,
    /// Optional static bucket configuration. `None` uses the global domain.
    pub bucket_config: Option<BucketConfig>,
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    /// Cache-Aware prefix configuration.
    pub cache_aware: Option<CacheAwareConfig>,
    /// Present only for the sticky policy; the header supplies the request routing key.
    pub sticky: Option<StickyConfig>,
    /// Session and cache-affinity tuning.
    pub affinity: Option<AffinityConfig>,
    /// Terms for `fused_score` or `score_policy`; defaults to [`DEFAULT_FUSE`].
    pub fused: Option<Vec<FusedTerm>>,
    /// Hard constraints applied before policy selection.
    pub eligibility: Option<EligibilityConfig>,
    /// Fleet sampling defaults and conflict behavior. See [`SamplingOverrides`].
    pub sampling_overrides: SamplingOverrides,
}

/// External KV Indexer client settings.
#[derive(Debug, Clone)]
pub struct KvIndexerEndpointConfig {
    pub url: String,
    pub query_timeout_ms: u64,
    pub query_max_inflight: usize,
}

/// Eligibility filter configuration.
#[derive(Debug, Clone, Default)]
pub struct EligibilityConfig {
    /// Filters in priority order.
    pub filters: Vec<FilterKind>,
    /// `overloaded`: in-flight count at which a worker stops being eligible.
    pub max_in_flight: Option<usize>,
    /// `prefix_cache` minimum cached prompt share.
    pub min_prefix_share: Option<f32>,
}

/// Default `--policy fused_score` terms.
pub const DEFAULT_FUSE: [ScoreTermKind; 2] = [ScoreTermKind::PrefixCache, ScoreTermKind::LoadBased];

/// One `--fuse` policy and optional weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedTerm {
    pub kind: ScoreTermKind,
    /// Weight override; `None` keeps the term's own `Criterion::weight()`.
    pub weight: Option<f32>,
}

impl std::str::FromStr for FusedTerm {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let (name, weight) = match s.split_once('=') {
            Some((n, w)) => (n, Some(parse_fuse_weight(n, w)?)),
            None => (s, None),
        };
        let kind = <ScoreTermKind as clap::ValueEnum>::from_str(name, false)
            .map_err(|_| format!("--fuse: `{name}` is not a score term"))?;
        Ok(FusedTerm { kind, weight })
    }
}

/// Parses a finite, non-negative term weight.
fn parse_fuse_weight(name: &str, raw: &str) -> Result<f32, String> {
    let w: f32 = raw
        .parse()
        .map_err(|_| format!("--fuse: `{name}` weight `{raw}` is not a number"))?;
    if !w.is_finite() || w < 0.0 {
        return Err(format!(
            "--fuse: `{name}` weight `{raw}` must be finite and >= 0"
        ));
    }
    Ok(w)
}

/// Cache-Aware prefix-match source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum CachePrefixProvider {
    #[default]
    #[value(name = "radix_tree")]
    RadixTree,
    #[value(name = "indexer")]
    Indexer,
}

/// Per-model Cache-Aware configuration.
#[derive(Debug, Clone, Default)]
pub struct CacheAwareConfig {
    /// Prefix-match source for native Cache-Aware.
    pub prefix_provider: CachePrefixProvider,
    /// External Indexer configuration when `prefix_provider = indexer`.
    pub kv_indexer_endpoint: Option<KvIndexerEndpointConfig>,
}

/// Default request header for sticky routing.
pub const DEFAULT_STICKY_HEADER: &str = "x-sgl-routing-key";

/// Default request header for session-aware routing.
pub const DEFAULT_SESSION_ID_HEADER: &str = "x-session-id";

/// Default external-indexer request limits.
pub const DEFAULT_KV_INDEXER_QUERY_MAX_INFLIGHT: usize = 32;

/// Default min-load sample size: the pre-existing power-of-2 behavior.
/// Every code path that has no `AffinityConfig` to read must fall back to
/// this, so the no-affinity path never drifts from the configured default.
pub const DEFAULT_MIN_LOAD_CHOICES: usize = 2;

/// Controls whether admission may select a session-affinity backup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum AffinityMode {
    /// Keep the primary after it passes admission.
    #[value(name = "strict")]
    Strict,
    /// Allow the admitted backup to relieve pressure.
    #[default]
    #[value(name = "soft")]
    Soft,
}

/// Controls the session-affinity lookup and fallback behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum SessionAffinityMode {
    /// Search only within the target bucket.
    #[default]
    #[value(name = "bucket")]
    Bucket,
    /// Rebind to a target-bucket fallback when the global primary is unavailable.
    #[value(name = "global-rebind")]
    GlobalRebind,
    /// Keep a valid global assignment when a bucket fallback is used.
    #[value(name = "global-preserve")]
    GlobalPreserve,
}

/// Shared session-aware and cache-aware settings.
#[derive(Debug, Clone)]
pub struct AffinityConfig {
    pub session_id_header: String,
    pub session_idle_secs: u64,
    pub session_eviction_interval_secs: u64,
    pub stable_pair: bool,
    pub mode: AffinityMode,
    pub session_affinity_mode: SessionAffinityMode,
    pub pressure_guard: bool,
    pub pressure_abs_threshold_tokens: u64,
    pub pressure_abs_threshold_ms: Option<f64>,
    pub pressure_rel_threshold: f64,
    pub cache_affinity_min_matched_tokens: Option<u64>,
    pub cache_affinity_min_match_ratio: Option<f64>,
    pub cache_candidate_min_workers: usize,
    pub cache_candidate_ratio: f64,
    pub cache_candidate_max_workers: usize,
    pub cache_switch_margin_tokens: u64,
    /// Waiting-request limit for cache affinity; `None` disables.
    /// Gates on the engine-published *waiting* count rather than total depth because
    /// waiting is the question the request cares about — will it sit behind other work —
    /// while depth proxies it badly (an engine can queue far below its running cap on
    /// long-prompt traffic). Fails open without a fresh sample: the router-side
    /// in-flight counter cannot separate running from waiting requests.
    /// Counts sum across DP ranks, so scale the limit with `dp_size`.
    pub worker_queue_limit: Option<u64>,
    /// Keep the least-pressured prefix owner when the queue gate rejects all admitted
    /// cache candidates and no fresh fleet queue is below this floor. Unknown queues
    /// do not count as idle — the opposite of the gate's fail-open, deliberately: the
    /// pin asks whether a provably better destination exists, and an unknown queue is
    /// not proof. Requires `floor <= worker_queue_limit`; scale with `dp_size`.
    pub saturation_queue_floor: Option<u64>,
    /// Min-load fallback sample size; defaults to power-of-two. At least the pool
    /// size chooses the exact minimum with random ties; 1 draws uniformly without
    /// a backup for admission or pressure guards. Separate from cache-owner limits.
    pub min_load_choices: usize,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            session_id_header: DEFAULT_SESSION_ID_HEADER.to_string(),
            session_idle_secs: default_sticky_idle_secs(),
            session_eviction_interval_secs: default_sticky_eviction_interval_secs(),
            stable_pair: false,
            mode: AffinityMode::Soft,
            session_affinity_mode: SessionAffinityMode::Bucket,
            pressure_guard: true,
            pressure_abs_threshold_tokens: 1_024,
            pressure_abs_threshold_ms: None,
            pressure_rel_threshold: 1.5,
            // Indexer prefix scans are truncated, so use an absolute token floor.
            cache_affinity_min_matched_tokens: Some(1_024),
            cache_affinity_min_match_ratio: None,
            cache_candidate_min_workers: 8,
            cache_candidate_ratio: 0.05,
            cache_candidate_max_workers: 32,
            cache_switch_margin_tokens: 1_024,
            worker_queue_limit: None,
            saturation_queue_floor: None,
            min_load_choices: DEFAULT_MIN_LOAD_CHOICES,
        }
    }
}

/// Sticky routing settings; the CLI validates the header name and positive durations.
#[derive(Debug, Clone)]
pub struct StickyConfig {
    /// Request header carrying the routing key. Validated to parse as a
    /// `http::HeaderName` at config-build time.
    pub header_name: String,
    /// Fallback for new or missing routing keys.
    pub fallback_policy: StickyFallbackKind,
    /// Evict an assignment after it has been idle (unreferenced) this many
    /// seconds. Bounds the map against unbounded routing-key cardinality.
    pub idle_secs: u64,
    /// Wall-clock cadence of the background eviction sweep.
    pub eviction_interval_secs: u64,
}

pub fn default_sticky_idle_secs() -> u64 {
    600
}
pub fn default_sticky_eviction_interval_secs() -> u64 {
    60
}

impl Default for StickyConfig {
    fn default() -> Self {
        Self {
            header_name: DEFAULT_STICKY_HEADER.to_string(),
            fallback_policy: StickyFallbackKind::RoundRobin,
            idle_secs: default_sticky_idle_secs(),
            eviction_interval_secs: default_sticky_eviction_interval_secs(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Consecutive failures before opening the breaker; zero is invalid.
    pub threshold: NonZeroU32,
    pub cool_down_secs: u64,
}

/// Default circuit-breaker cool-down, applied when `--cb-threshold` is
/// set without an explicit `--cb-cool-down-secs`.
pub fn default_cb_cool_down() -> u64 {
    30
}

#[derive(Debug, Clone)]
pub enum DiscoveryBackend {
    StaticUrls(StaticUrlsDiscoveryConfig),
    K8s(K8sDiscoveryConfig),
}

/// Workers registered at startup. Roles, models, and bootstrap ports come from
/// `/server_info`; topology changes require a restart.
#[derive(Debug, Clone)]
pub struct StaticUrlsDiscoveryConfig {
    pub urls: Vec<String>,
}

/// Kubernetes EndpointSlice discovery. Selectors classify slices; worker roles
/// and bootstrap ports come from `/server_info` introspection.
#[derive(Debug, Clone)]
pub struct K8sDiscoveryConfig {
    pub namespace: String,
    /// Resolved + validated selector mode (plain vs PD).
    pub mode: K8sDiscoveryMode,
}

/// Validated selector mode. Plain selectors run server-side; PD selectors
/// classify EndpointSlices client-side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum K8sDiscoveryMode {
    /// One global label selector; every matched EndpointSlice becomes a
    /// `WorkerMode::Plain` worker.
    Plain { label_selector: String },
    /// Two label selectors; an EndpointSlice's labels are matched against
    /// each to classify it as `WorkerMode::Prefill` or `WorkerMode::Decode`.
    PdDisaggregation {
        prefill_selector: String,
        decode_selector: String,
    },
}

/// Error returned by [`resolve_mode`] when the selector combination is
/// invalid.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error(
        "discovery.k8s requires either `label_selector` (plain) or both `prefill_selector` and `decode_selector` (PD); none were set"
    )]
    NoSelector,
    #[error(
        "discovery.k8s: `label_selector` (plain) and `prefill_selector`/`decode_selector` (PD) are mutually exclusive — set one or the other, not both"
    )]
    MixedModes,
    #[error("discovery.k8s: PD mode requires BOTH `prefill_selector` and `decode_selector`")]
    PartialPdSelectors,
    #[error(
        "discovery.k8s: {selector}_selector `{value}` uses unsupported syntax — \
         only equality terms (`key=value` or `key==value`) joined by `,` are accepted. \
         Set-based operators (`in`, `notin`), presence tests, and `!=` silently match \
         zero endpoints at runtime and are rejected at startup."
    )]
    UnsupportedSelectorGrammar {
        selector: &'static str,
        value: String,
    },
    #[error(
        "discovery.k8s: PD `{selector}_selector` is empty (or only whitespace/commas) — \
         it would match every EndpointSlice, and since classify_mode checks prefill before \
         decode, the opposite role's pool would stay empty. Set non-empty equality terms \
         distinguishing the two roles."
    )]
    EmptyPdSelector { selector: &'static str },
    #[error(
        "discovery.k8s: `prefill_selector` and `decode_selector` resolve to the same set \
         of equality terms — classify_mode would tag every matching slice as Prefill and \
         leave the decode pool empty. The two selectors must differ."
    )]
    IdenticalPdSelectors,
}

/// An empty selector matches every slice, which is invalid for a PD role.
fn is_selector_empty(selector: &str) -> bool {
    selector.split(',').all(|t| t.trim().is_empty())
}

/// Normalize validated equality terms for comparison, matching runtime whitespace
/// and `=`/`==` handling. Term order does not affect matching.
fn canonical_selector(selector: &str) -> Vec<(&str, &str)> {
    let mut terms: Vec<_> = selector
        .split(',')
        .filter_map(|raw| {
            let term = raw.trim();
            if term.is_empty() {
                return None;
            }
            // Prefer `==` so its second equals sign does not become part of the value.
            let (k, v) = term.split_once("==").or_else(|| term.split_once('='))?;
            Some((k.trim(), v.trim()))
        })
        .collect();
    terms.sort_unstable();
    terms
}

/// Check the equality-only grammar supported by client-side PD matching.
fn is_equality_selector(selector: &str) -> bool {
    for term in selector.split(',') {
        let term = term.trim();
        if term.is_empty() {
            // Treat lone trailing commas / whitespace as fine; the runtime
            // splitter ignores empty terms.
            continue;
        }
        if let Some((k, _)) = term.split_once("==") {
            if k.trim().is_empty() {
                return false;
            }
            continue;
        }
        if let Some((k, _value)) = term.split_once('=') {
            // Reject `!=`; empty label values are valid.
            if k.trim().is_empty() || k.trim().ends_with('!') {
                return false;
            }
            continue;
        }
        // No `=` at all → set-based operator, presence test, or garbage.
        return false;
    }
    true
}

/// Resolve and validate the plain or prefill/decode selector combination.
pub fn resolve_mode(
    label_selector: Option<&str>,
    prefill_selector: Option<&str>,
    decode_selector: Option<&str>,
) -> Result<K8sDiscoveryMode, ConfigError> {
    match (label_selector, prefill_selector, decode_selector) {
        (Some(label), None, None) => {
            // Plain selectors run on the Kubernetes API, which supports the full grammar.
            // PD selectors are checked client-side and only support equality.
            Ok(K8sDiscoveryMode::Plain {
                label_selector: label.to_string(),
            })
        }
        (None, Some(prefill), Some(decode)) => {
            // Validate both grammars before checking for empty or identical selectors.
            let selectors = [("prefill", prefill), ("decode", decode)];
            for (selector, value) in selectors {
                if !is_equality_selector(value) {
                    return Err(ConfigError::UnsupportedSelectorGrammar {
                        selector,
                        value: value.to_string(),
                    });
                }
            }
            // Empty PD selectors match everything and starve the opposite role.
            for (selector, value) in selectors {
                if is_selector_empty(value) {
                    return Err(ConfigError::EmptyPdSelector { selector });
                }
            }
            // Prefill wins when both selectors match, leaving decode empty.
            if canonical_selector(prefill) == canonical_selector(decode) {
                return Err(ConfigError::IdenticalPdSelectors);
            }
            Ok(K8sDiscoveryMode::PdDisaggregation {
                prefill_selector: prefill.to_string(),
                decode_selector: decode.to_string(),
            })
        }
        (None, None, None) => Err(ConfigError::NoSelector),
        (None, Some(_), None) | (None, None, Some(_)) => Err(ConfigError::PartialPdSelectors),
        (Some(_), _, _) => Err(ConfigError::MixedModes),
    }
}

#[cfg(test)]
mod k8s_discovery_config_tests {
    use super::*;

    #[test]
    fn mode_constructs_pd_disaggregation_from_prefill_and_decode_selectors() {
        let m = resolve_mode(None, Some("app=sglang,role=p"), Some("app=sglang,role=d"))
            .expect("PD mode is now valid");
        assert_eq!(
            m,
            K8sDiscoveryMode::PdDisaggregation {
                prefill_selector: "app=sglang,role=p".to_string(),
                decode_selector: "app=sglang,role=d".to_string(),
            }
        );
    }

    #[test]
    fn mode_pd_rejects_set_based_prefill_selector() {
        let err =
            resolve_mode(None, Some("app in (sglang, vllm)"), Some("app=sglang")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::UnsupportedSelectorGrammar {
                    selector: "prefill",
                    ..
                },
            ),
            "expected UnsupportedSelectorGrammar(prefill), got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_set_based_decode_selector() {
        let err =
            resolve_mode(None, Some("app=sglang"), Some("app in (sglang, vllm)")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::UnsupportedSelectorGrammar {
                    selector: "decode",
                    ..
                },
            ),
            "expected UnsupportedSelectorGrammar(decode), got {err:?}",
        );
    }

    #[test]
    fn mode_accepts_plain_with_equality_selector() {
        let m = resolve_mode(Some("app=sglang"), None, None).unwrap();
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: "app=sglang".to_string()
            }
        );
    }

    /// Plain selectors run on the Kubernetes API, which supports the full grammar.
    /// PD selectors are checked client-side and only support equality.
    #[test]
    fn mode_accepts_set_based_selector_in_plain_mode() {
        let m = resolve_mode(Some("app in (sglang,sglang-small)"), None, None)
            .expect("plain mode must accept set-based selectors");
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: "app in (sglang,sglang-small)".to_string(),
            }
        );
    }

    #[test]
    fn mode_accepts_other_set_based_forms_in_plain_mode() {
        for raw in [
            "app notin (vllm,trtllm)",
            "tier",
            "!deprecated",
            "tier!=canary",
        ] {
            let m = resolve_mode(Some(raw), None, None)
                .unwrap_or_else(|e| panic!("plain mode must accept `{raw}`, got {e:?}"));
            assert_eq!(
                m,
                K8sDiscoveryMode::Plain {
                    label_selector: raw.to_string(),
                },
                "selector roundtrip mismatch for `{raw}`",
            );
        }
    }

    #[test]
    fn mode_pd_rejects_notin_prefill_selector() {
        let err =
            resolve_mode(None, Some("app notin (vllm, trtllm)"), Some("app=sglang")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::UnsupportedSelectorGrammar {
                    selector: "prefill",
                    ..
                },
            ),
            "expected UnsupportedSelectorGrammar(prefill), got {err:?}",
        );
    }

    #[test]
    fn mode_accepts_comma_separated_equality_terms() {
        // The canonical Plain-mode selector form: `key1=v1,key2=v2`.
        let m = resolve_mode(Some("app=sglang,zone=us-east"), None, None).unwrap();
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: "app=sglang,zone=us-east".to_string()
            }
        );
    }

    #[test]
    fn mode_rejects_when_no_selector_is_set() {
        let err = resolve_mode(None, None, None).unwrap_err();
        assert!(matches!(err, ConfigError::NoSelector), "got {err:?}");
    }

    #[test]
    fn mode_rejects_mixed_plain_and_pd_selectors() {
        let err = resolve_mode(
            Some("app=sglang"),
            Some("role=prefill"),
            Some("role=decode"),
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::MixedModes), "got {err:?}");
    }

    #[test]
    fn mode_rejects_partial_pd_selectors() {
        let err = resolve_mode(None, Some("role=prefill"), None).unwrap_err();
        assert!(
            matches!(err, ConfigError::PartialPdSelectors),
            "got {err:?}"
        );
        let err = resolve_mode(None, None, Some("role=decode")).unwrap_err();
        assert!(
            matches!(err, ConfigError::PartialPdSelectors),
            "got {err:?}"
        );
    }

    #[test]
    fn mode_accepts_empty_plain_label_selector() {
        let m = resolve_mode(Some(""), None, None).unwrap();
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: String::new()
            }
        );
    }

    #[test]
    fn mode_pd_rejects_empty_prefill_selector() {
        let err = resolve_mode(None, Some(""), Some("role=decode")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::EmptyPdSelector {
                    selector: "prefill"
                },
            ),
            "expected EmptyPdSelector(prefill), got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_empty_decode_selector() {
        let err = resolve_mode(None, Some("role=prefill"), Some("")).unwrap_err();
        assert!(
            matches!(err, ConfigError::EmptyPdSelector { selector: "decode" },),
            "expected EmptyPdSelector(decode), got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_whitespace_only_prefill_selector() {
        let err = resolve_mode(None, Some("  ,  "), Some("role=decode")).unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::EmptyPdSelector {
                    selector: "prefill"
                },
            ),
            "expected EmptyPdSelector(prefill), got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_identical_prefill_and_decode_selectors() {
        let err = resolve_mode(None, Some("app=sglang"), Some("app=sglang")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// Trailing whitespace must not be a loophole that bypasses the
    /// identical-selector check.
    #[test]
    fn mode_pd_rejects_identical_selectors_under_whitespace_normalization() {
        let err = resolve_mode(None, Some("app=sglang"), Some("  app=sglang  ")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_identical_selectors_under_eq_alias() {
        let err = resolve_mode(None, Some("app=sglang"), Some("app==sglang")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_identical_selectors_under_inner_whitespace() {
        let err = resolve_mode(None, Some("app=sglang"), Some("app =  sglang")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_identical_selectors_under_term_order_permutation() {
        let err =
            resolve_mode(None, Some("role=p,app=sglang"), Some("app=sglang,role=p")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    #[test]
    fn mode_pd_accepts_truly_distinct_selectors() {
        let m = resolve_mode(
            None,
            Some("app=sglang,role=prefill"),
            Some("app=sglang,role=decode"),
        )
        .expect("distinct selectors must validate");
        assert!(matches!(m, K8sDiscoveryMode::PdDisaggregation { .. }));
    }
}
