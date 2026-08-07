use std::num::NonZeroU32;
use std::num::NonZeroUsize;

/// In-memory router configuration, built from CLI flags by
/// [`crate::config::cli::Cli::into_config`] and validated by
/// [`Config::validate`]. The router serves exactly one model.
#[derive(Debug, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub observability: ObservabilityConfig,
    pub model: ModelConfig,
    /// Selected discovery backend. Built from CLI flags by
    /// [`crate::config::cli::Cli::into_config`]: the static-vs-k8s choice
    /// and the k8s selector grammar are resolved there (the latter via
    /// [`resolve_mode`]); static worker-URL validity is checked by
    /// [`Config::validate`].
    pub discovery: DiscoveryBackend,
    pub proxy: ProxyConfig,
    pub active_load: ActiveLoadConfig,
}

/// Outbound proxy tuning. Default mirrors SGLang's typical prefill /
/// decode latency budget; e2e tests lower it so per-request failures
/// trip the circuit breaker within the test's wall-time.
#[derive(Debug, Clone, Copy)]
pub struct ProxyConfig {
    /// Maximum time to wait for a single upstream HTTP request to
    /// return headers + body. Default 300 s. The circuit breaker
    /// records a failure when this fires.
    pub request_timeout_secs: u64,
}

pub fn default_proxy_request_timeout_secs() -> u64 {
    300
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: default_proxy_request_timeout_secs(),
        }
    }
}

/// Active-load (per-request) tracking. Production default (10 min)
/// sits above `proxy.request_timeout_secs` so the proxy timeout is the
/// one users hit first for normal slow upstreams; tests lower it to
/// let the janitor fire within their wall-time budget.
#[derive(Debug, Clone, Copy)]
pub struct ActiveLoadConfig {
    /// How long a request entry can live in the registry before the
    /// janitor fires its `cancel_token` and the chat handler returns
    /// 504 `stale_request_expired`. Default 600 s.
    pub stale_request_timeout_secs: u64,
}

pub fn default_stale_request_timeout_secs() -> u64 {
    600
}

impl Default for ActiveLoadConfig {
    fn default() -> Self {
        Self {
            stale_request_timeout_secs: default_stale_request_timeout_secs(),
        }
    }
}

/// Routing policy selector — the enum form lets `clap` reject unknown
/// values at parse time and removes the runtime string match in the
/// policy factory.
///
/// Accepted on the CLI (`--policy`) as `round_robin` / `random` /
/// `power_of_two` / `load_based` / `cache_aware_zmq` / `sticky`.
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
    /// Cache-aware routing fed by SGLang's ZMQ KV-cache event publisher.
    /// Requires the model to have a tokenizer loaded; cache_aware tuning
    /// lives on `ModelConfig::cache_aware`.
    #[value(name = "cache_aware_zmq")]
    CacheAwareZmq,
    /// Sticky-session routing: pins a routing key (read from a
    /// configurable request header) to a worker via an in-memory map, so
    /// stateful sessions land on the same backend. Tuning — header name,
    /// keyless-fallback policy, and TTL eviction — lives on
    /// `ModelConfig::sticky`.
    #[value(name = "sticky")]
    Sticky,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub log_level: String,
    /// Selects the tracing-subscriber output format. `clap` rejects
    /// unrecognized values at parse time (`--log-format jsonl` and
    /// similar typos surface as an error instead of silently degrading
    /// to text).
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
    /// Tokenizer source: a local `tokenizer.json` path or a HuggingFace repo
    /// id (downloaded on demand). Defaults to `id` when `--tokenizer-path`
    /// is omitted. Resolved by [`crate::tokenizer::adapter::load`].
    pub tokenizer_path: String,
    pub policy: PolicyKind,
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    /// Tuning for the cache-aware ZMQ policy. Ignored unless
    /// `policy = "cache_aware_zmq"`. `None` falls back to defaults at
    /// policy construction time.
    pub cache_aware: Option<CacheAwareConfig>,
    /// Tuning for the sticky-session policy. `Some` exactly when
    /// `policy = "sticky"` (built by [`crate::config::cli::Cli::into_config`]).
    /// The chat handler reads `sticky.header_name` to populate
    /// [`crate::policies::SelectionContext::routing_key`].
    pub sticky: Option<StickyConfig>,
}

/// How the cache-aware policy decides that load should override cache
/// affinity. The two strategies are alternatives, not layers, so this is an
/// enum: a struct carrying both sets of knobs can represent a configuration
/// where half of them are silently unread, which is the state the CLI has to
/// reject by hand and which a config dump would still print as if live.
///
/// The distinction that matters is *what is measured*. `FleetSpread` measures
/// the candidate set; `PerWorkerQueue` measures whether the worker this request
/// would actually land on is already making requests wait.
#[derive(Debug, Clone, Copy)]
pub enum LoadGate {
    /// Skip the cache lookup entirely when `max_load - min_load >
    /// abs_threshold` AND `max_load > min_load * rel_threshold`, and route to
    /// the least-loaded of a `min_load_choices` sample instead.
    ///
    /// `max - min` over the candidate set is an order statistic of the fleet,
    /// so in a large fleet it is dominated by the tail and says little about
    /// any single request. It therefore both over-fires (one busy worker
    /// diverts every request, including the vast majority whose cache home is
    /// idle) and under-fires (a worker one slot from shedding is unprotected
    /// whenever the fleet minimum is high enough to close the spread).
    FleetSpread {
        /// Default 32 — picked to dominate over typical batch-of-8 effect.
        abs_threshold: usize,
        /// Default 1.1 — 10 % relative difference triggers re-balancing.
        rel_threshold: f32,
    },
    /// Number of already-queued requests at or above which a worker stops being
    /// eligible to win a selection on cache affinity — the request goes to
    /// another worker holding the same prefix, or failing that to the
    /// least-loaded of a `min_load_choices` sample.
    ///
    /// The cache lookup itself is unchanged, but every min-load fallback in the
    /// policy additionally *prefers* a non-queueing worker, so this reorders
    /// more than the matched-set pick — including the ordinary below-threshold
    /// tree-miss path, which is the highest-volume one. The fallback is never
    /// queue-*bounded*, though: when no unqueued worker exists it samples the
    /// whole fleet and takes the least-loaded of the sample regardless, so a
    /// fleet where everything is queueing still routes.
    ///
    /// Gating on the queue rather than on total depth is what makes this
    /// targeted. `num_waiting_reqs` IS the question the request cares about —
    /// will I sit behind other work before my prefill starts — whereas depth
    /// only proxies it, and proxies it badly: on long-prompt traffic engines
    /// have been observed queueing at 7-8 running, far below
    /// `max_running_requests`, so a depth threshold high enough to avoid firing
    /// on healthy busy workers misses most of the workers that are actually
    /// making requests wait.
    ///
    /// Gating on the queue makes the healthy *baseline* topology-independent,
    /// which a depth threshold is not: depth summed across a worker's dp ranks
    /// carries a `dp_size × max_running_requests` term, while a healthy worker
    /// reads ~0 waiting at any `dp_size`. The *firing point* still scales,
    /// though — `fresh_worker_state` sums `waiting` across ranks while a request
    /// is dispatched to one of them, so on a `dp_size=8` worker a queue of one
    /// on four ranks sums to 4 and trips a limit of 4, where the request would
    /// actually sit behind at most one. Scale the limit with `dp_size`.
    ///
    /// Read from `cache_aware_zmq::WorkerLoads::waiting_of`, which is `None`
    /// when no fresh all-ranks snapshot exists. There is no router-side
    /// substitute — the in-flight counter cannot distinguish a running request
    /// from a waiting one — so an unknown queue leaves the worker eligible.
    /// The gate fails open and says so, rather than comparing the threshold
    /// against a different quantity.
    PerWorkerQueue(NonZeroUsize),
}

impl Default for LoadGate {
    fn default() -> Self {
        Self::FleetSpread {
            abs_threshold: default_balance_abs(),
            rel_threshold: default_balance_rel(),
        }
    }
}

impl LoadGate {
    pub const DEFAULT_ABS_THRESHOLD: usize = 32;
    pub const DEFAULT_REL_THRESHOLD: f32 = 1.1;

    /// Whether a worker with this queue reading may still win a selection on
    /// cache affinity, and be preferred by the min-load fallback.
    ///
    /// The single place the gate's rule is written. Both decisions live here:
    /// the boundary is `<` (at the limit disqualifies), and an unknown queue
    /// (`None`) admits — there is no per-worker router-side substitute for the
    /// engine's queue, so the gate fails open rather than comparing the limit
    /// against a different quantity. Callers that re-derived this from
    /// [`Self::queue_limit`] could drift apart; the matched-set filter and the
    /// fallback's preference tier must agree or the fallback hands requests
    /// back to workers the gate just rejected.
    pub fn admits_affinity(&self, waiting: Option<usize>) -> bool {
        match (self, waiting) {
            (Self::PerWorkerQueue(limit), Some(waiting)) => waiting < limit.get(),
            _ => true,
        }
    }

    /// The configured queue limit, or `None` under [`Self::FleetSpread`].
    /// For logging and diagnostics — the routing decision goes through
    /// [`Self::admits_affinity`].
    pub fn queue_limit(&self) -> Option<usize> {
        match self {
            Self::PerWorkerQueue(limit) => Some(limit.get()),
            Self::FleetSpread { .. } => None,
        }
    }
}

/// Per-model cache-aware-ZMQ tuning.
#[derive(Debug, Clone, Copy)]
pub struct CacheAwareConfig {
    /// Lower bound on `matched_blocks / total_blocks` for the tree match
    /// to win the selection. Below this, the policy falls back to
    /// min-load. Default 0.5 — a half-cached prompt is still a strong
    /// signal but not so weak that random hash collisions could trigger
    /// affinity to an arbitrary worker.
    pub cache_threshold: f32,
    /// Which load signal is allowed to override cache affinity.
    pub load_gate: LoadGate,
    /// Candidates sampled for each min-load fallback pick: the pick is the
    /// least-loaded of this many uniformly random eligible workers, not the
    /// fleet-wide minimum. Default 2 (power-of-two choices), which damps the
    /// cross-replica herd on the shared current minimum while losing almost
    /// no load quality. Applies within each gating tier (unqueued first),
    /// so a busy fleet never becomes a selection error. A value at or above
    /// the fleet size recovers the exact-minimum behavior.
    pub min_load_choices: usize,
    /// Fleet-saturation floor for the matched-set fallback, only meaningful
    /// under [`LoadGate::PerWorkerQueue`] (the CLI rejects it otherwise, so
    /// it is never silently set-but-unread). `None` — the default — keeps the
    /// historical behavior: when every prefix owner is over the queue limit,
    /// the request diverts to a sampled min-load worker regardless of how
    /// busy the rest of the fleet is.
    ///
    /// With a floor set, that diversion is first checked for a payoff: if no
    /// worker in the fleet has a fresh queue reading strictly below the
    /// floor, every destination is already making requests wait, so
    /// diverting cannot dodge the queue — it only forfeits the matched
    /// prefix (and on long-prompt traffic converts a wait into a full cold
    /// prefill, which is how a saturated fleet's hit rate collapses and
    /// stays collapsed). The request then routes to the least-loaded prefix
    /// owner instead, recorded as `cache_hit_all_queued`.
    ///
    /// A worker with no fresh snapshot does NOT count as idle here — the
    /// opposite polarity from [`LoadGate::admits_affinity`], and deliberately
    /// so: that gate fails open because keeping affinity is the safe default
    /// action, while this check asks whether a *provably better* destination
    /// exists, and an unknown queue is not proof. Both polarities keep the
    /// request with its prefix owner when the signal is missing.
    ///
    /// The CLI enforces `floor <= worker-queue-limit`, which yields the
    /// invariant that makes the metric readable: every selection booked
    /// `cache_hit_all_queued` actually kept its affinity (the old
    /// sampled-draw path for that label becomes unreachable — all workers
    /// fresh-and-over-limit implies all are at-or-over the floor, so the pin
    /// fires first). Like the queue limit, scale it with `dp_size`: the
    /// queue reading is summed across a worker's dp ranks.
    pub saturation_queue_floor: Option<NonZeroUsize>,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: default_cache_threshold(),
            load_gate: LoadGate::default(),
            min_load_choices: default_min_load_choices(),
            saturation_queue_floor: None,
        }
    }
}

fn default_cache_threshold() -> f32 {
    0.5
}
fn default_balance_abs() -> usize {
    LoadGate::DEFAULT_ABS_THRESHOLD
}
fn default_balance_rel() -> f32 {
    LoadGate::DEFAULT_REL_THRESHOLD
}

fn default_min_load_choices() -> usize {
    2
}

/// Default routing-key header for the sticky policy. The `x-sgl-` prefix
/// matches the router's other emitted/consumed metadata headers
/// (`x-sgl-decode-url`, `x-sgl-router-error-code`).
pub const DEFAULT_STICKY_HEADER: &str = "x-sgl-routing-key";

/// Per-model sticky-session tuning. Built from the `--routing-key-header`
/// / `--sticky-*` flags by [`crate::config::cli::Cli::into_config`], which
/// also validates that `header_name` parses as an HTTP header name and
/// that `fallback_policy` is one of the dependency-free policies.
#[derive(Debug, Clone)]
pub struct StickyConfig {
    /// Request header carrying the routing key. Validated to parse as a
    /// `http::HeaderName` at config-build time.
    pub header_name: String,
    /// Policy used to pick a worker when a request has no routing key, and
    /// to pick the initial worker when a new key is first seen. One of
    /// `round_robin` / `random` / `power_of_two` / `load_based` — the
    /// dependency-free policies the factory can build standalone (no
    /// `HashTree` / tokenizer / ZMQ feed). `cache_aware_zmq` and `sticky`
    /// are rejected at config-build time.
    pub fallback_policy: PolicyKind,
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
            fallback_policy: PolicyKind::RoundRobin,
            idle_secs: default_sticky_idle_secs(),
            eviction_interval_secs: default_sticky_eviction_interval_secs(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Consecutive failures required before the breaker opens. Encoded
    /// as `NonZeroU32` so `--cb-threshold 0` (which would open the
    /// breaker before any failure) is rejected at CLI-parse time rather
    /// than silently behaving as "always open".
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

/// Fixed list of worker URLs. Each URL is registered once at startup;
/// `mode`, `model_ids`, and `bootstrap_port` are resolved per-worker
/// from `/server_info` (see [`crate::workers::introspect`]).
///
/// No file watcher, no hot-reload: topology change requires a restart.
#[derive(Debug, Clone)]
pub struct StaticUrlsDiscoveryConfig {
    pub urls: Vec<String>,
}

/// Configuration for the Kubernetes `EndpointSlice` discovery backend.
/// Built from the `--service-discovery*` / `--selector` / `--prefill-selector`
/// / `--decode-selector` flags by [`crate::config::cli::Cli::build_discovery`].
///
/// Two operating modes, distinguished by which selector flags are set:
///
/// 1. **Plain** — all matched workers share the same role:
///    `--service-discovery-namespace default --selector app=sglang`
///
/// 2. **PD disaggregation** — prefill and decode workers are separated by
///    different selectors:
///    `--service-discovery-namespace default
///    --prefill-selector app=sglang,role=prefill
///    --decode-selector app=sglang,role=decode`
///
/// In PD mode, the selectors drive **slice-classification** (which
/// EndpointSlices feed the prefill pool vs the decode pool). The actual
/// `WorkerMode` and `bootstrap_port` for each worker are filled in by
/// the worker manager from each worker's `/server_info` introspection,
/// so PD works without any pod-level annotations — see
/// [`crate::workers::introspect`] for the `disaggregation_mode` and
/// `disaggregation_bootstrap_port` extraction.
///
/// [`resolve_mode`] validates the selector flags and produces the
/// resolved [`K8sDiscoveryMode`] once, at construction in
/// [`crate::config::cli::Cli::build_discovery`] — so an invalid selector
/// combination is unrepresentable here.
#[derive(Debug, Clone)]
pub struct K8sDiscoveryConfig {
    pub namespace: String,
    /// Resolved + validated selector mode (plain vs PD).
    pub mode: K8sDiscoveryMode,
}

/// Resolved discovery mode, produced by [`resolve_mode`] from the CLI
/// selector flags and stored on [`K8sDiscoveryConfig`].
///
/// The discovery backend uses this to:
/// * pick the server-side `LIST` label selector (Plain: the single selector;
///   PD: empty, with classification done client-side per slice), and
/// * assign each `EndpointSlice` a [`crate::discovery::WorkerMode`] in
///   `extract_workers`.
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
    #[error("discovery.k8s requires either `label_selector` (plain) or both `prefill_selector` and `decode_selector` (PD); none were set")]
    NoSelector,
    #[error("discovery.k8s: `label_selector` (plain) and `prefill_selector`/`decode_selector` (PD) are mutually exclusive — set one or the other, not both")]
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

/// Returns `true` when `selector` has zero non-empty terms after
/// trimming and splitting on `,`. `labels_match_selector` then returns
/// `true` for every label set, which is the "matches everything"
/// degenerate case PD mode must reject.
fn is_selector_empty(selector: &str) -> bool {
    selector.split(',').all(|t| t.trim().is_empty())
}

/// Canonicalize a comma-separated equality selector to a sorted list of
/// parsed `(key, value)` tuples. Comparison happens at the parsed-term
/// level — *not* the raw string level — because `labels_match_selector`
/// already strips whitespace and treats `key=value` and `key==value` as
/// the same equality test. Comparing raw strings would let
/// `"app=sglang"` vs `"app==sglang"` (and `"app = sglang"` vs
/// `"app=sglang"`) past the identical-selector check, even though
/// `classify_mode` would treat them identically at runtime — exactly
/// the silent decode-pool-empty failure mode this check exists to
/// prevent.
///
/// Returns an empty `Vec` for selectors with no parseable terms
/// (whitespace-only, comma-only, or any term that doesn't match the
/// `key=value` / `key==value` grammar). Callers must run
/// [`is_equality_selector`] before this to surface malformed
/// selectors as `UnsupportedSelectorGrammar`.
fn canonical_selector(selector: &str) -> Vec<(String, String)> {
    let mut terms: Vec<(String, String)> = selector
        .split(',')
        .filter_map(|raw| {
            let term = raw.trim();
            if term.is_empty() {
                return None;
            }
            // Mirror `labels_match_selector`: prefer the `==` alias so a
            // term like `key==value` parses to `(key, value)` instead of
            // `(key, =value)`.
            let (k, v) = term.split_once("==").or_else(|| term.split_once('='))?;
            Some((k.trim().to_string(), v.trim().to_string()))
        })
        .collect();
    terms.sort();
    terms
}

/// Returns `true` when `selector` parses as a comma-separated equality
/// selector — every term has the shape `key=value` or `key==value`.
/// See [`ConfigError::UnsupportedSelectorGrammar`] for rationale.
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
            // Reject `!=` (rendered as `key!` + `=value` by split_once).
            // Empty value is legal in K8s — `label_selector = "tier="`
            // matches pods with `tier=""` — so we don't constrain it.
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

/// Validate the selector combination and return the resolved
/// [`K8sDiscoveryMode`]. Called once at construction by
/// [`crate::config::cli::Cli::build_discovery`], so an invalid
/// combination can never be stored on a [`K8sDiscoveryConfig`].
pub fn resolve_mode(
    label_selector: Option<&str>,
    prefill_selector: Option<&str>,
    decode_selector: Option<&str>,
) -> Result<K8sDiscoveryMode, ConfigError> {
    match (label_selector, prefill_selector, decode_selector) {
        (Some(label), None, None) => {
            // Plain mode pushes `label` to the K8s API as the
            // server-side `labelSelector` of the EndpointSlice
            // watcher (`watcher::Config::default().labels(&label)`
            // in `discovery::k8s::spawn`). K8s itself parses the
            // full label-selector grammar — equality, set-based
            // (`in` / `notin`), presence (`key` / `!key`), and
            // `!=` — and rejects malformed selectors at
            // watch-start time. So we don't grammar-check `label`
            // here and let the K8s API be the syntax authority. PD
            // mode, in contrast, evaluates selectors client-side via
            // `labels_match_selector` which only understands
            // equality — so PD selectors are still grammar-checked
            // below.
            Ok(K8sDiscoveryMode::Plain {
                label_selector: label.to_string(),
            })
        }
        (None, Some(prefill), Some(decode)) => {
            // Both selectors validated individually so the operator
            // sees which one is malformed. WorkerMode + bootstrap_port
            // for each prefill pod are filled in by the worker
            // manager from each worker's `/server_info` — these
            // selectors only drive client-side classification per
            // EndpointSlice (see `classify_mode` in discovery/k8s.rs).
            if !is_equality_selector(prefill) {
                return Err(ConfigError::UnsupportedSelectorGrammar {
                    selector: "prefill",
                    value: prefill.to_string(),
                });
            }
            if !is_equality_selector(decode) {
                return Err(ConfigError::UnsupportedSelectorGrammar {
                    selector: "decode",
                    value: decode.to_string(),
                });
            }
            // Empty PD selector matches every EndpointSlice at
            // runtime; combined with classify_mode's prefill-first
            // ordering, an empty selector would silently funnel all
            // workers into one role. Reject up front.
            if is_selector_empty(prefill) {
                return Err(ConfigError::EmptyPdSelector {
                    selector: "prefill",
                });
            }
            if is_selector_empty(decode) {
                return Err(ConfigError::EmptyPdSelector { selector: "decode" });
            }
            // Identical selectors degrade the same way as an empty
            // one: every slice matches both, prefill wins, decode
            // stays empty.
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
        // K8s PD now works without per-pod annotations: each worker's
        // `/server_info` carries `disaggregation_bootstrap_port`, and the
        // worker manager applies it post-discovery. The K8s config layer's
        // job is just to validate the selector combination.
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
        // Both PD selectors get the same equality-only grammar check as
        // the plain label_selector. A set-based prefill selector would
        // silently match zero pods at runtime → fail-fast at load.
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

    /// Plain mode pushes its selector to the K8s API server-side
    /// (`watcher::Config::default().labels(&selector)` in
    /// `discovery::k8s::spawn`), so the full K8s label-selector grammar
    /// — including set-based operators like `app in (a,b)` — is
    /// supported and must not be grammar-checked at startup. PD mode
    /// (checked client-side) is the opposite; see the PD tests below.
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

    /// `notin`, presence (`key`), absence (`!key`), and inequality (`!=`)
    /// are all valid K8s server-side selector grammar — plain mode must
    /// pass them through.
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

    /// PD mode evaluates selectors *client-side* via
    /// `labels_match_selector`, which only handles equality. A set-based
    /// PD selector would silently match zero pods → fail-fast at load.
    /// Pins the plain-server-side / PD-client-side asymmetry: relaxing
    /// the grammar check for plain (see `mode_accepts_set_based_*`
    /// above) must not accidentally relax it for PD selectors. Uses
    /// `notin` so this test covers a different set-based form than
    /// `mode_pd_rejects_set_based_prefill_selector` (which uses `in`)
    /// — both must keep failing.
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

    /// Empty plain `label_selector` is valid — matches every
    /// EndpointSlice in the namespace (documented K8s behavior; the
    /// operator opts in by setting plain mode at all).
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

    /// PD mode is the *opposite* of plain: an empty selector would match
    /// every EndpointSlice, and since `classify_mode` checks prefill
    /// before decode, an empty `prefill_selector` would classify
    /// everything as Prefill — decode pool stays empty and the resolver
    /// surfaces the wrong `no_decode_workers_available` error. Fail-fast
    /// at config load.
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

    /// Whitespace-only / comma-only PD selector parses to zero terms in
    /// `labels_match_selector` and matches every slice at runtime — same
    /// failure mode as a literal empty string.
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

    /// Identical prefill and decode selectors degrade silently: every
    /// slice matches both, but `classify_mode` returns `Prefill` first,
    /// so the decode pool stays empty.
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

    /// `labels_match_selector` accepts both `key=value` and `key==value`
    /// for equality and parses them to the same `(key, value)` tuple.
    /// Two selectors that differ only in this alias choice are runtime-
    /// equivalent — they'd match the same EndpointSlices, then
    /// `classify_mode`'s prefill-first ordering would funnel every slice
    /// into Prefill, leaving decode empty. The check must canonicalize
    /// at the term level (parsed `(key, value)` tuples), not the raw
    /// string level.
    #[test]
    fn mode_pd_rejects_identical_selectors_under_eq_alias() {
        let err = resolve_mode(None, Some("app=sglang"), Some("app==sglang")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// Inner whitespace inside a term (`"app = sglang"`) is the same
    /// label as no whitespace (`"app=sglang"`) — the runtime
    /// `labels_match_selector` trims key and value independently
    /// (see `key.trim()` / `expected.trim()` in `k8s.rs`). Canonical
    /// form must agree.
    #[test]
    fn mode_pd_rejects_identical_selectors_under_inner_whitespace() {
        let err = resolve_mode(None, Some("app=sglang"), Some("app =  sglang")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// Term order doesn't matter for label matching, so `"a=1,b=2"` and
    /// `"b=2,a=1"` must be treated as identical. (Implied by the sort
    /// in `canonical_selector`, but pinned explicitly so a future
    /// "preserve user order for diagnostics" refactor can't silently
    /// reintroduce the silent-failure bug.)
    #[test]
    fn mode_pd_rejects_identical_selectors_under_term_order_permutation() {
        let err =
            resolve_mode(None, Some("role=p,app=sglang"), Some("app=sglang,role=p")).unwrap_err();
        assert!(
            matches!(err, ConfigError::IdenticalPdSelectors),
            "expected IdenticalPdSelectors, got {err:?}",
        );
    }

    /// Sanity: two selectors that genuinely differ at the term level
    /// must still pass validation — the canonicalizer must not be so
    /// aggressive that it false-positives on legitimate PD configs.
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

#[cfg(test)]
mod load_gate_tests {
    use super::*;

    /// The gate's whole rule, pinned directly rather than through a policy
    /// fixture: the boundary is `<` (at the limit disqualifies) and an unknown
    /// queue admits. Both the matched-set filter and the fallback's preference
    /// tier read this one method, so if they ever disagree it is because this
    /// changed — and then this test says so, instead of one routing fixture's
    /// numbers happening to still work out.
    #[test]
    fn load_gate_admits_affinity_encodes_the_boundary_and_fails_open() {
        let gate = LoadGate::PerWorkerQueue(NonZeroUsize::new(4).unwrap());
        assert!(gate.admits_affinity(Some(0)));
        assert!(gate.admits_affinity(Some(3)), "3 < 4 is still eligible");
        assert!(!gate.admits_affinity(Some(4)), "at the limit disqualifies");
        assert!(!gate.admits_affinity(Some(99)));
        assert!(
            gate.admits_affinity(None),
            "an unknown queue must admit: there is no per-worker router-side \
             substitute, so the gate fails open rather than comparing the limit \
             against a different quantity"
        );

        // The fleet-spread strategy has no per-worker queue opinion at all.
        let spread = LoadGate::default();
        for waiting in [None, Some(0), Some(1_000)] {
            assert!(spread.admits_affinity(waiting));
        }
        assert_eq!(spread.queue_limit(), None);
    }

    /// A limit of 1 gates any queue at all — the most aggressive representable
    /// setting, and the reason `NonZeroUsize` matters: 0 would gate everything
    /// unconditionally and silently degrade the policy to pure min-load.
    #[test]
    fn load_gate_limit_of_one_gates_any_queue() {
        let gate = LoadGate::PerWorkerQueue(NonZeroUsize::new(1).unwrap());
        assert!(gate.admits_affinity(Some(0)));
        assert!(!gate.admits_affinity(Some(1)));
    }
}
