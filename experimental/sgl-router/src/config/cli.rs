// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Command-line interface. The router is configured entirely through
//! flags — there is no config file. [`Cli::into_config`] resolves the
//! flags into a validated [`Config`].

use anyhow::{anyhow, Result};
use clap::Parser;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

use crate::config::{
    default_cache_sim_max_concurrent_captures, default_cb_cool_down,
    default_proxy_request_timeout_secs, default_shutdown_drain_secs,
    default_stale_request_timeout_secs, default_stream_idle_timeout_secs,
    default_stream_send_stall_secs, default_tokenizer_shards, resolve_mode, ActiveLoadConfig,
    AdmissionConfig, CacheAwareConfig, CircuitBreakerConfig, Config, DiscoveryBackend,
    K8sDiscoveryConfig, LoadGate, LogFormat, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, RetryConfig, ServerConfig, StaticUrlsDiscoveryConfig, StickyConfig,
};

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
    /// Number of independent tokenizer instances to load for this model.
    /// Every tokio worker thread otherwise shares one `Arc<Tokenizer>`,
    /// serializing on the BPE word-merge cache's internal `RwLock` under
    /// concurrent load; loading several independent instances and
    /// round-robining across them (see
    /// `crate::tokenizer::TokenizerRegistry::get`) spreads that contention
    /// across N locks with no change to tokenization output. Must be >= 1.
    #[arg(long, default_value_t = default_tokenizer_shards())]
    pub tokenizer_shards: usize,
    /// Tokenizer encode backend: "hf" (HuggingFace `tokenizers`, the
    /// pre-existing default) or "fast" (`fastokens` hybrid — fast encode,
    /// HF decode, automatic HF fallback if fastokens can't load the
    /// tokenizer file; the fallback is visible on
    /// `sgl_router_tokenizer_backend`).
    #[arg(long, value_enum, default_value = "hf")]
    pub tokenizer_backend: crate::tokenizer::adapter::TokenizerBackend,
    /// MiB budget for the L1 special-token-boundary prefix tokenization
    /// cache. 0 (default) disables it. When the cache is genuinely active
    /// (the tokenizer declares safe special tokens) --tokenizer-shards is
    /// forced to 1; pair with `--tokenizer-backend fast` so miss-heavy
    /// traffic doesn't funnel through one HF merge-cache lock.
    #[arg(long, default_value_t = 0)]
    pub tokenizer_l1_cache_mb: usize,
    /// Per-request output-token cap for this model (e.g. `131072`). A
    /// request whose `max_completion_tokens` / `max_tokens` exceeds this is
    /// rejected with 400 before admission; a request that sets neither gets
    /// `max_tokens = <cap>` injected into the forwarded body. Unset (the
    /// default) preserves today's behavior: the engine's context length is
    /// the only output bound.
    #[arg(long)]
    pub max_output_tokens: Option<NonZeroU64>,
    /// Central kill switch for the ingress tokenize offload: when set, the
    /// router NEVER injects its ingress-computed `input_ids` into the
    /// forwarded body, so the engine always tokenizes from `messages` itself.
    /// Ingress tokenization still runs — routing and the cache-sim tees keep
    /// consuming the ids; only the engine-facing forward is gated. Unset (the
    /// default) forwards `input_ids` when they are engine-equivalent (chat-
    /// encoder model) and the request passes `input_ids_safe_to_forward`,
    /// per `select_forward_input_ids`. Also read from
    /// `SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD` so the platform can flip it
    /// via env — an env var an older router (without this flag) simply
    /// ignores, rather than crash-looping on an unknown CLI flag. Both the
    /// flag and the env value accept boolish spellings (e.g.
    /// `true/false/yes/no/on/off/1/0`, case-insensitive — see clap's
    /// `BoolishValueParser`) — the strict `true`/`false`-only parser would
    /// crash-loop the rollout on the common `value: "1"` convention this
    /// env wiring exists to support. An empty or unrecognized env value is
    /// a startup parse error — unset the var instead of setting it empty.
    #[arg(
        long,
        env = "SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        value_parser = clap::builder::BoolishValueParser::new()
    )]
    pub disable_input_ids_offload: bool,
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

    // ---- cache-aware-zmq tuning (only used by that policy) ----
    /// Min `matched_blocks / total_blocks` for a cache match to win.
    #[arg(long)]
    pub cache_threshold: Option<f32>,
    /// How long `/readyz` may stay 503 while this replica bootstraps its
    /// cache-aware tree from a warm sibling. Defaults to 5000.
    #[arg(long)]
    pub kv_bootstrap_timeout_ms: Option<u64>,
    /// Upper bound on one peer-snapshot fetch during bootstrap, in
    /// milliseconds. The per-fetch timeout is derived as a quarter of
    /// `--kv-bootstrap-timeout-ms` (raised toward a 5s floor for short
    /// budgets); this caps that derivation (default 30000). Raise it when
    /// the fleet's tree is large enough that one transfer + decode no
    /// longer fits under the derived value.
    #[arg(long)]
    pub kv_bootstrap_fetch_timeout_cap_ms: Option<u64>,
    /// Label selector matching this router's own pods, so a booting replica can
    /// find siblings to pull a tree snapshot from. Unset disables peer
    /// bootstrap and every replica starts cold.
    #[arg(long)]
    pub kv_peer_selector: Option<String>,
    /// Absolute load spread above which the cache check is skipped.
    #[arg(long)]
    pub balance_abs_threshold: Option<usize>,
    /// Multiplicative load spread gating the absolute balance check.
    #[arg(long)]
    pub balance_rel_threshold: Option<f32>,
    /// Queued-request count at or above which a worker stops winning
    /// selections on cache affinity: the request is sent to another worker
    /// holding the same prefix, or to the least-loaded of a
    /// `--min-load-choices` sample if there is none. Counts
    /// `num_waiting_reqs` only, so it measures whether the
    /// request would wait rather than how busy the worker is.
    ///
    /// Start at 4 on a single-rank worker — low enough to catch a real
    /// backlog, high enough not to chase a queue of 1 that drains
    /// immediately — and scale it with `dp_size`: the count is summed across
    /// a worker's dp ranks while a request lands on one of them, so on a
    /// dp-8 worker a queue of one on four ranks already sums to 4.
    ///
    /// Requires engines that publish load (same enablement as KV events).
    /// With no fresh load snapshot the gate has nothing to read, fails open,
    /// and never fires — and because it REPLACES the fleet-spread check
    /// rather than layering on it, the router then has no load override at
    /// all. Mutually exclusive with `--balance-abs-threshold` /
    /// `--balance-rel-threshold`.
    #[arg(long)]
    pub worker_queue_limit: Option<NonZeroUsize>,
    /// Candidates sampled for each min-load fallback pick: the fallback
    /// ranks this many uniformly random workers and takes the least-loaded
    /// of them. Default 2 (power-of-two choices).
    ///
    /// A fleet-wide exact minimum converges across router replicas: every
    /// replica reads the same engine load snapshots, none sees the others'
    /// dispatches made since the snapshot, so they all hand the request to
    /// the same current minimum and overshoot it. Ranking a small random
    /// sample keeps the pick near-loaded-minimal while making concurrent
    /// replicas diverge.
    ///
    /// Sampling applies within each gating tier, so the tier order is
    /// unchanged: with `--worker-queue-limit` set, the sample is drawn from
    /// unqueued workers first and only from queueing ones when nothing is
    /// unqueued. A value of 1 is uniform-random routing among the eligible
    /// tier, not the exact minimum — set this at or above the fleet size for
    /// the deterministic exact fleet-wide minimum.
    #[arg(long)]
    pub min_load_choices: Option<NonZeroUsize>,
    /// Fleet-saturation floor below which `--worker-queue-limit` diversions
    /// are cancelled and the request stays with its prefix owner. When every
    /// owner is over the queue limit, the router normally diverts to a
    /// min-load sample; with this set, it first checks whether any worker in
    /// the fleet has a fresh queue reading strictly below the floor. If none
    /// does, the fleet is saturated: diverting cannot dodge a wait — every
    /// destination is already queueing — but it does forfeit the matched
    /// prefix, and on long-prompt traffic that converts one wait into a full
    /// cold prefill that evicts other prefixes in turn (the self-sustaining
    /// hit-rate-collapse mode under overload). The request then routes to
    /// the least-loaded prefix owner instead, recorded as
    /// `cache_hit_all_queued`.
    ///
    /// Must be at most `--worker-queue-limit` (and requires it). Start at 2-4
    /// on a single-rank worker: high enough that a queue of one draining
    /// immediately doesn't read as saturation, low enough that a worker with
    /// real spare capacity still attracts diversions. Scale with `dp_size`
    /// like the limit — the queue reading is summed across dp ranks. Workers
    /// with no fresh load snapshot do not count as idle: an unknown queue is
    /// not proof that diverting pays.
    #[arg(long)]
    pub saturation_queue_floor: Option<NonZeroUsize>,

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
    pub sticky_fallback_policy: Option<PolicyKind>,
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
    /// Between-bytes idle timeout (seconds) for the upstream→router streaming
    /// leg. If the engine delivers no bytes for this long mid-stream, the pump
    /// aborts and releases the request's guards. A stall cap, NOT a total-time
    /// cap: a slow-but-progressing generation is unaffected. Also settable via
    /// the `SGLANG_ROUTER_STREAM_IDLE_TIMEOUT_SECS` env var.
    #[arg(
        long,
        env = "SGLANG_ROUTER_STREAM_IDLE_TIMEOUT_SECS",
        default_value_t = default_stream_idle_timeout_secs()
    )]
    pub stream_idle_timeout_secs: u64,
    /// Backpressure stall timeout (seconds) for the router→client streaming
    /// leg. If the client accepts no bytes for this long while the pump is
    /// blocked on read-ahead permits, the pump gives up and releases the
    /// per-worker in-flight slot. Also settable via the
    /// `SGLANG_ROUTER_STREAM_SEND_STALL_SECS` env var.
    #[arg(
        long,
        env = "SGLANG_ROUTER_STREAM_SEND_STALL_SECS",
        default_value_t = default_stream_send_stall_secs()
    )]
    pub stream_send_stall_secs: u64,
    /// Max lifetime of an in-flight request entry before the janitor
    /// reaps it (returns 504 `stale_request_expired`).
    #[arg(long, default_value_t = default_stale_request_timeout_secs())]
    pub stale_request_timeout_secs: u64,
    /// Seconds to keep serving after SIGTERM, with `/readyz` returning 503,
    /// before the server stops accepting — so k8s deregisters this pod first.
    /// Must be <= the pod's terminationGracePeriodSeconds. 0 disables the pause.
    #[arg(long, default_value_t = default_shutdown_drain_secs())]
    pub shutdown_drain_secs: u64,

    // ---- admission control ----
    /// Maximum in-flight requests dispatched to a single worker. When set,
    /// the router parks a request once every candidate worker is at this cap
    /// and dispatches it when a slot frees. Must be > 0. Unset (default)
    /// disables admission control: requests dispatch immediately as before.
    #[arg(long)]
    pub max_concurrent_requests_per_worker: Option<NonZeroUsize>,
    /// Maximum requests parked in the admission wait queue before further
    /// arrivals are shed with 503. Requires `--max-concurrent-requests-per-worker`.
    /// Unset leaves the wait queue unbounded (park, never shed).
    #[arg(long)]
    pub max_queued_requests: Option<usize>,

    // ---- retry (plain-mode failover on transient dispatch failures) ----
    /// Retry a plain-mode request ONCE, on a *different* worker, when it hits a
    /// transient upstream failure (connection refused, request-headers timeout,
    /// breaker-open, malformed worker URL) before any bytes reach the client.
    /// When an admission cap (`--max-concurrent-requests-per-worker`) is
    /// configured, the retry only goes to a worker whose in-flight load is
    /// below it — never onto a full one, and it never waits for a slot;
    /// without a cap the retry is ungated (a startup advisory warns about
    /// this combination). Off by default; PD-disaggregated requests are
    /// always single-attempt.
    #[arg(long)]
    pub enable_retry: bool,
    /// ITL load gate for retries: a retry target's router-observed inter-token
    /// latency must be <= this many ms. Unset disables the ITL gate (retry falls
    /// over to any below-cap worker). Requires `--enable-retry`.
    #[arg(long)]
    pub retry_max_target_itl_ms: Option<u64>,
    /// A retry target's ITL must also be <= the failed worker's ITL times this
    /// factor (default 1.0 — no worse than the worker we just left; applied only
    /// when both ITLs are known). Requires `--enable-retry`, and only engages
    /// when `--retry-max-target-itl-ms` is also set — it refines that ceiling and
    /// is a no-op without it.
    #[arg(long)]
    pub retry_itl_rel_factor: Option<f32>,
    /// Retry TTFT gate (ms): does NOT time out or interrupt an attempt — every
    /// attempt runs to its natural end. It only gates the RETRY: when an attempt
    /// fails with a retryable error, if it already ran at least this long before
    /// failing, the retry is skipped and the original error surfaces (a retry
    /// would burden a healthy worker for a request that already blew its budget).
    /// Unset disables the gate. Applied together with the ITL / KV-util load
    /// gates and the admission cap — a retry needs ALL of them to pass. Requires
    /// `--enable-retry`.
    #[arg(long)]
    pub retry_attempt_deadline_ms: Option<u64>,

    // ---- observability ----
    /// Default tracing level (overridden by `RUST_LOG`).
    #[arg(long, default_value = "info")]
    pub log_level: String,
    /// Log output format.
    #[arg(long, value_enum, default_value = "text")]
    pub log_format: LogFormat,
    /// Base URL of the theoretical cache-sim service (e.g.
    /// `http://radixark-cache-sim:9095`). When set, each request's
    /// ingress-computed `input_ids` are teed to `<url>/ingest_ids`
    /// (best-effort, fire-and-forget — never blocks or fails a request).
    /// Unset disables the tee. Also read from `RADIXARK_CACHE_SIM_URL` so the
    /// platform can wire it via env — an env var an older router (without this
    /// flag) simply ignores, rather than crash-looping on an unknown CLI flag.
    #[arg(long, env = "RADIXARK_CACHE_SIM_URL")]
    pub cache_sim_url: Option<String>,
    /// Max concurrent streaming captures the cache-sim extend tee holds. Each
    /// buffers up to 16 MiB, so this hard-caps aggregate capture memory at
    /// `N × 16 MiB` (default 256 ⇒ ~4 GiB ceiling); excess streams skip the
    /// capture. Only meaningful with `--cache-sim-url`.
    #[arg(long, default_value_t = default_cache_sim_max_concurrent_captures())]
    pub cache_sim_max_concurrent_captures: usize,
    /// `s3://bucket/prefix/` 目标；设置即开启 token 数据集导出，写入每个
    /// 请求的 ingest/extend token 序列（NDJSON+gzip）。凭证/region 走标准
    /// AWS 环境变量。不设则关闭。
    #[arg(long, env = "RADIXARK_TOKEN_EXPORT_S3_URI")]
    pub token_export_s3_uri: Option<String>,
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
        let tuned_cache_aware = self.cache_threshold.is_some()
            || self.balance_abs_threshold.is_some()
            || self.balance_rel_threshold.is_some()
            || self.kv_bootstrap_timeout_ms.is_some()
            || self.kv_bootstrap_fetch_timeout_cap_ms.is_some()
            || self.kv_peer_selector.is_some()
            || self.worker_queue_limit.is_some()
            || self.min_load_choices.is_some()
            || self.saturation_queue_floor.is_some();
        if tuned_cache_aware && self.policy != PolicyKind::CacheAwareZmq {
            return Err(anyhow!(
                "cache-aware tuning (--cache-threshold / --balance-abs-threshold / \
                 --balance-rel-threshold / --kv-bootstrap-timeout-ms / \
                 --kv-bootstrap-fetch-timeout-cap-ms / --kv-peer-selector / \
                 --worker-queue-limit / --min-load-choices / --saturation-queue-floor) \
                 requires --policy cache_aware_zmq"
            ));
        }
        // The queue gate replaces the fleet-spread check rather than layering on
        // it, so accepting both would leave the operator believing a knob is
        // live when the policy never reads it.
        if self.worker_queue_limit.is_some()
            && (self.balance_abs_threshold.is_some() || self.balance_rel_threshold.is_some())
        {
            return Err(anyhow!(
                "--worker-queue-limit replaces the fleet-spread check, so it cannot be \
                 combined with --balance-abs-threshold / --balance-rel-threshold; pass only one"
            ));
        }
        // The floor modifies the queue gate's diversion; without the gate there
        // is no diversion to cancel, and the policy would never read it.
        if let Some(floor) = self.saturation_queue_floor {
            let Some(limit) = self.worker_queue_limit else {
                return Err(anyhow!(
                    "--saturation-queue-floor gates the diversions made by \
                     --worker-queue-limit, so it requires that flag"
                ));
            };
            // floor <= limit is what makes `cache_hit_all_queued` mean
            // "affinity kept": all workers fresh-and-over-limit then implies
            // all are at-or-over the floor, so the owner pin fires before the
            // sampled fallback can book that label for an off-owner draw.
            if floor > limit {
                return Err(anyhow!(
                    "--saturation-queue-floor ({floor}) must be at most \
                     --worker-queue-limit ({limit}): a floor above the limit would \
                     declare the fleet saturated while workers the gate still \
                     admits exist"
                ));
            }
        }
        // `peer_selector` is only carried on the k8s discovery backend (it needs a
        // namespace to watch), so with any other backend it would be accepted and
        // then silently ignored — the operator would see cold boots with no
        // explanation.
        if self.kv_peer_selector.is_some() && !self.service_discovery {
            return Err(anyhow!(
                "--kv-peer-selector requires --service-discovery (peer replicas are \
                 found via Kubernetes EndpointSlices)"
            ));
        }
        // `Instant::now() + Duration::from_millis(timeout)` panics on overflow, so
        // an absurd value would abort the process at startup rather than being
        // rejected here. The ceiling is also well past any sane readinessProbe
        // budget.
        const MAX_KV_BOOTSTRAP_TIMEOUT_MS: u64 = 600_000;
        if let Some(ms) = self.kv_bootstrap_timeout_ms {
            if ms > MAX_KV_BOOTSTRAP_TIMEOUT_MS {
                return Err(anyhow!(
                    "--kv-bootstrap-timeout-ms {ms} exceeds the {MAX_KV_BOOTSTRAP_TIMEOUT_MS}ms \
                     ceiling; /readyz stays 503 for this long, so a larger value would \
                     outlast any reasonable readinessProbe"
                ));
            }
        }

        // Below the internal fetch floor (5s) the cap would cut every fetch
        // short of a body transfer, and past the deadline ceiling it could
        // never bind — both read as typos, so reject them at startup.
        // The 5s below mirrors SNAPSHOT_FETCH_TIMEOUT_FLOOR in
        // policies::kv_events::index; keep them in agreement if it moves.
        if let Some(ms) = self.kv_bootstrap_fetch_timeout_cap_ms {
            const MIN_KV_BOOTSTRAP_FETCH_TIMEOUT_CAP_MS: u64 = 5_000;
            if !(MIN_KV_BOOTSTRAP_FETCH_TIMEOUT_CAP_MS..=MAX_KV_BOOTSTRAP_TIMEOUT_MS).contains(&ms)
            {
                return Err(anyhow!(
                    "--kv-bootstrap-fetch-timeout-cap-ms {ms} is out of range \
                     ({MIN_KV_BOOTSTRAP_FETCH_TIMEOUT_CAP_MS}..={MAX_KV_BOOTSTRAP_TIMEOUT_MS}ms)"
                ));
            }
        }

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

        // Build (and validate) the sticky config exactly when the sticky
        // policy is selected. The header name must parse as an HTTP header
        // name so a typo fails at startup rather than silently never
        // matching any request header; the fallback must be a
        // dependency-free policy the factory can build standalone.
        let sticky = if self.policy == PolicyKind::Sticky {
            let d = StickyConfig::default();
            let header_name = self.routing_key_header.unwrap_or(d.header_name);
            axum::http::HeaderName::try_from(header_name.as_str()).map_err(|e| {
                anyhow!("--routing-key-header {header_name:?} is not a valid HTTP header name: {e}")
            })?;
            let fallback_policy = self.sticky_fallback_policy.unwrap_or(d.fallback_policy);
            if matches!(
                fallback_policy,
                PolicyKind::Sticky | PolicyKind::CacheAwareZmq
            ) {
                return Err(anyhow!(
                    "--sticky-fallback-policy must be one of round_robin / random / \
                     power_of_two / load_based; cache_aware_zmq and sticky are not allowed"
                ));
            }
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

        // A wait-queue depth is meaningless without a per-worker cap (nothing
        // ever parks), so reject it rather than silently ignore it.
        if self.max_queued_requests.is_some() && self.max_concurrent_requests_per_worker.is_none() {
            return Err(anyhow!(
                "--max-queued-requests requires --max-concurrent-requests-per-worker \
                 (the wait queue only fills once workers hit their in-flight cap)"
            ));
        }

        if self.tokenizer_shards == 0 {
            return Err(anyhow!("--tokenizer-shards must be at least 1"));
        }

        let circuit_breaker = self.cb_threshold.map(|threshold| CircuitBreakerConfig {
            threshold,
            cool_down_secs: self.cb_cool_down_secs.unwrap_or_else(default_cb_cool_down),
        });

        // Only build a CacheAwareConfig when the operator tuned at least
        // one knob; otherwise leave it None so the policy uses its own
        // defaults. Unset knobs fall back to the per-field defaults.
        let cache_aware = if tuned_cache_aware {
            let d = CacheAwareConfig::default();
            // `validate` has already rejected the combination, so the queue
            // limit alone decides which gate is built.
            let load_gate = match self.worker_queue_limit {
                Some(limit) => LoadGate::PerWorkerQueue(limit),
                None => LoadGate::FleetSpread {
                    abs_threshold: self
                        .balance_abs_threshold
                        .unwrap_or(LoadGate::DEFAULT_ABS_THRESHOLD),
                    rel_threshold: self
                        .balance_rel_threshold
                        .unwrap_or(LoadGate::DEFAULT_REL_THRESHOLD),
                },
            };
            Some(CacheAwareConfig {
                cache_threshold: self.cache_threshold.unwrap_or(d.cache_threshold),
                load_gate,
                bootstrap_timeout_ms: self
                    .kv_bootstrap_timeout_ms
                    .unwrap_or(d.bootstrap_timeout_ms),
                bootstrap_fetch_timeout_cap_ms: self
                    .kv_bootstrap_fetch_timeout_cap_ms
                    .unwrap_or(d.bootstrap_fetch_timeout_cap_ms),
                min_load_choices: self
                    .min_load_choices
                    .map(NonZeroUsize::get)
                    .unwrap_or(d.min_load_choices),
                saturation_queue_floor: self.saturation_queue_floor,
            })
        } else {
            None
        };

        let config = Config {
            server: ServerConfig {
                host: self.host,
                port: self.port,
                shutdown_drain_secs: self.shutdown_drain_secs,
            },
            observability: ObservabilityConfig {
                log_level: self.log_level,
                log_format: self.log_format,
                cache_sim_url: self.cache_sim_url,
                cache_sim_max_concurrent_captures: self.cache_sim_max_concurrent_captures,
                token_export_s3_uri: self.token_export_s3_uri,
            },
            model: ModelConfig {
                // Default the tokenizer source to the model id (treated as a
                // HuggingFace repo id) when --tokenizer-path is omitted.
                tokenizer_path: self.tokenizer_path.unwrap_or_else(|| self.model_id.clone()),
                tokenizer_shards: self.tokenizer_shards,
                tokenizer_backend: self.tokenizer_backend,
                tokenizer_l1_cache_mb: self.tokenizer_l1_cache_mb,
                id: self.model_id,
                policy: self.policy,
                circuit_breaker,
                cache_aware,
                sticky,
                max_output_tokens: self.max_output_tokens,
                forward_input_ids: !self.disable_input_ids_offload,
            },
            discovery,
            proxy: ProxyConfig {
                request_timeout_secs: self.request_timeout_secs,
                stream_idle_timeout_secs: self.stream_idle_timeout_secs,
                stream_send_stall_secs: self.stream_send_stall_secs,
            },
            active_load: ActiveLoadConfig {
                stale_request_timeout_secs: self.stale_request_timeout_secs,
            },
            admission: match self.max_concurrent_requests_per_worker {
                Some(max_concurrent_per_worker) => AdmissionConfig::Enabled {
                    max_concurrent_per_worker,
                    max_queued_requests: self.max_queued_requests,
                },
                None => AdmissionConfig::Disabled,
            },
            retry: RetryConfig {
                enabled: self.enable_retry,
                max_target_itl_ms: self.retry_max_target_itl_ms,
                itl_rel_factor: self.retry_itl_rel_factor,
                attempt_deadline_ms: self.retry_attempt_deadline_ms,
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
                    peer_selector: self.kv_peer_selector.clone(),
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
    use crate::config::{DiscoveryBackend, K8sDiscoveryMode};

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

    /// Parse under a SHARED env lock, so no sibling test can be mutating
    /// `SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD` mid-parse. Tests that mutate
    /// the env hold the lock exclusively and must call [`parse_env_locked`]
    /// instead — `RwLock` is not reentrant, so re-locking here would deadlock.
    fn into_config_owned(args: Vec<String>) -> Result<Config> {
        let _shared = ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        parse_env_locked(args)
    }

    /// Parse WITHOUT taking the env lock — for callers already holding it
    /// exclusively via [`lock_env`].
    fn parse_env_locked(args: Vec<String>) -> Result<Config> {
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
        assert_eq!(c.server.shutdown_drain_secs, 5);
        // Streaming stall budgets default to 180 s (both legs).
        assert_eq!(c.proxy.stream_idle_timeout_secs, 180);
        assert_eq!(c.proxy.stream_send_stall_secs, 180);
    }

    #[test]
    fn retry_disabled_by_default_in_cli() {
        let c = into_config_owned(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
        assert!(!c.retry.enabled, "retry must be opt-in");
    }

    /// Serializes env mutation against config parsing. `Cli` declares
    /// `env = "SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD"`, so EVERY parse reads
    /// that var — not just the tests that set it. A plain mutex held only by
    /// the setters is therefore not enough: while one test parks the var at an
    /// intentionally-invalid value (`"banana"`, to prove startup fails), any
    /// sibling test parsing in parallel sees it too and fails with a wholly
    /// unrelated error. Hence a RwLock — mutators take it exclusively, every
    /// parse takes it shared, so parses still run concurrently with each other.
    /// Poison-tolerant: a panicking test must neither leak the var nor
    /// cascade-fail its siblings with a misleading PoisonError.
    static ENV_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

    /// Restores SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD on drop, even
    /// through a panic mid-test.
    struct EnvReset;
    impl Drop for EnvReset {
        fn drop(&mut self) {
            std::env::remove_var("SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD");
        }
    }

    /// Take the env lock (recovering from poison) and clean slate the var,
    /// returning a guard pair whose drop removes it again.
    fn lock_env() -> (std::sync::RwLockWriteGuard<'static, ()>, EnvReset) {
        let guard = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD");
        (guard, EnvReset)
    }

    #[test]
    fn input_ids_offload_on_by_default_and_gated_by_flag() {
        let (_guard, _reset) = lock_env();
        let on = parse_env_locked(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
        assert!(
            on.model.forward_input_ids,
            "the input_ids offload must default to enabled"
        );
        let off = parse_env_locked(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--disable-input-ids-offload",
        ]))
        .unwrap();
        assert!(
            !off.model.forward_input_ids,
            "--disable-input-ids-offload must gate the forward"
        );
    }

    /// The env var is the platform's activation channel: it must gate the
    /// offload accepting the common Kubernetes `value: "1"` spelling (and
    /// other boolish forms), and an empty/unrecognized value must be a
    /// startup parse error — never a silent default.
    #[test]
    fn input_ids_offload_gated_by_boolish_env_var() {
        let (_guard, _reset) = lock_env();
        for value in ["1", "true", "TRUE", "yes"] {
            std::env::set_var("SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD", value);
            let c =
                parse_env_locked(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
            assert!(
                !c.model.forward_input_ids,
                "env value {value:?} must gate the offload"
            );
        }
        for value in ["0", "false", "no"] {
            std::env::set_var("SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD", value);
            let c =
                parse_env_locked(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
            assert!(
                c.model.forward_input_ids,
                "env value {value:?} must leave the offload enabled"
            );
        }
        for value in ["", "banana", " 1"] {
            std::env::set_var("SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD", value);
            assert!(
                parse_env_locked(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).is_err(),
                "env value {value:?} must fail startup parsing, not silently default"
            );
        }
    }

    /// An explicit CLI value is the operator's per-replica escape hatch: it
    /// overrides the env var (clap's occurrence-beats-env precedence), so a
    /// single replica can be taken off a platform-wide engaged switch.
    #[test]
    fn input_ids_offload_cli_value_overrides_env_var() {
        let (_guard, _reset) = lock_env();
        std::env::set_var("SGLANG_ROUTER_DISABLE_INPUT_IDS_OFFLOAD", "1");
        let c = parse_env_locked(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--disable-input-ids-offload=false",
        ]))
        .unwrap();
        assert!(
            c.model.forward_input_ids,
            "--disable-input-ids-offload=false must override the env var"
        );
    }

    #[test]
    fn enable_retry_flag_maps_into_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--enable-retry",
        ]))
        .unwrap();
        assert!(c.retry.enabled);
    }

    #[test]
    fn shutdown_drain_secs_maps_into_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--shutdown-drain-secs",
            "0",
        ]))
        .unwrap();
        assert_eq!(
            c.server.shutdown_drain_secs, 0,
            "--shutdown-drain-secs 0 must disable the drain pause",
        );
    }

    #[test]
    fn admission_control_disabled_by_default() {
        let c = into_config_owned(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
        assert!(matches!(c.admission, AdmissionConfig::Disabled));
    }

    #[test]
    fn admission_flags_map_into_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--max-concurrent-requests-per-worker",
            "32",
            "--max-queued-requests",
            "8",
        ]))
        .unwrap();
        match c.admission {
            AdmissionConfig::Enabled {
                max_concurrent_per_worker,
                max_queued_requests,
            } => {
                assert_eq!(max_concurrent_per_worker.get(), 32);
                assert_eq!(max_queued_requests, Some(8));
            }
            AdmissionConfig::Disabled => panic!("expected Enabled, got Disabled"),
        }
    }

    #[test]
    fn zero_per_worker_cap_is_rejected() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--max-concurrent-requests-per-worker",
            "0",
        ]))
        .expect_err("a zero per-worker cap must be rejected");
        // clap rejects `0` for a NonZeroUsize-typed flag at parse time.
        assert!(
            err.to_string()
                .contains("max-concurrent-requests-per-worker")
                || err.to_string().to_lowercase().contains("0"),
            "got: {err}",
        );
    }

    /// `--tokenizer-backend` is a clap `ValueEnum`, so an unknown value is
    /// rejected at parse time with the flag named in the error.
    #[test]
    fn bogus_tokenizer_backend_is_rejected() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--tokenizer-backend",
            "bogus",
        ]))
        .expect_err("--tokenizer-backend bogus must be rejected");
        assert!(err.to_string().contains("tokenizer-backend"), "got: {err}");
    }

    /// The two tokenizer flags map into `ModelConfig` — and default to the
    /// pre-existing behavior (HF backend, cache off) when omitted.
    #[test]
    fn tokenizer_backend_and_l1_cache_map_into_config() {
        use crate::tokenizer::adapter::TokenizerBackend;

        let c = into_config_owned(with_model(&["--worker-urls", "http://10.0.0.1:30000"])).unwrap();
        assert_eq!(c.model.tokenizer_backend, TokenizerBackend::Hf);
        assert_eq!(c.model.tokenizer_l1_cache_mb, 0);

        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--tokenizer-backend",
            "fast",
            "--tokenizer-l1-cache-mb",
            "64",
        ]))
        .unwrap();
        assert_eq!(c.model.tokenizer_backend, TokenizerBackend::Fast);
        assert_eq!(c.model.tokenizer_l1_cache_mb, 64);
    }

    #[test]
    fn zero_tokenizer_shards_is_rejected() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--tokenizer-shards",
            "0",
        ]))
        .expect_err("--tokenizer-shards 0 must be rejected");
        // Unlike --max-concurrent-requests-per-worker (NonZeroUsize, rejected
        // by clap at parse time), tokenizer_shards is a plain `usize` — the
        // manual `if == 0` check in `into_config` (below) is the only guard,
        // so this test is what actually pins that check firing, not clap.
        assert!(err.to_string().contains("tokenizer-shards"), "got: {err}",);
    }

    #[test]
    fn max_queued_requests_requires_per_worker_cap() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://10.0.0.1:30000",
            "--max-queued-requests",
            "8",
        ]))
        .expect_err("--max-queued-requests without a per-worker cap must be rejected");
        assert!(
            err.to_string().contains("--max-queued-requests requires"),
            "got: {err}",
        );
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
        assert!(matches!(
            ca.load_gate,
            LoadGate::FleetSpread {
                abs_threshold: 32,
                ..
            }
        ));
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
    fn rejects_peer_selector_without_service_discovery() {
        let err = Cli::parse_from([
            "sgl-router",
            "--model-id",
            "m",
            "--tokenizer-path",
            "t",
            "--policy",
            "cache_aware_zmq",
            "--worker-urls",
            "http://w:1",
            "--kv-peer-selector",
            "app=router",
        ])
        .into_config()
        .expect_err("peer bootstrap needs k8s discovery to find siblings");
        assert!(
            err.to_string().contains("requires --service-discovery"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn rejects_absurd_kv_bootstrap_timeout() {
        let err = Cli::parse_from([
            "sgl-router",
            "--model-id",
            "m",
            "--tokenizer-path",
            "t",
            "--policy",
            "cache_aware_zmq",
            "--worker-urls",
            "http://w:1",
            "--kv-bootstrap-timeout-ms",
            &u64::MAX.to_string(),
        ])
        .into_config()
        .expect_err("an unbounded timeout would overflow Instant at startup");
        assert!(
            err.to_string().contains("ceiling"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn kv_bootstrap_fetch_timeout_cap_validates_its_range_and_plumbs() {
        let base_args = with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
        ]);
        for bad in ["4999", "600001"] {
            let mut args = base_args.clone();
            args.push("--kv-bootstrap-fetch-timeout-cap-ms".to_string());
            args.push(bad.to_string());
            let err = into_config_owned(args).unwrap_err().to_string();
            assert!(
                err.contains("kv-bootstrap-fetch-timeout-cap-ms"),
                "cap {bad} must be rejected; got: {err}",
            );
        }
        for good in ["5000", "30000", "600000"] {
            let mut args = base_args.clone();
            args.push("--kv-bootstrap-fetch-timeout-cap-ms".to_string());
            args.push(good.to_string());
            let c = into_config_owned(args)
                .unwrap_or_else(|e| panic!("cap {good} must be accepted; got: {e}"));
            assert_eq!(
                c.model.cache_aware.unwrap().bootstrap_fetch_timeout_cap_ms,
                good.parse::<u64>().unwrap(),
                "the flag must reach CacheAwareConfig",
            );
        }
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
            err.contains("requires --policy cache_aware_zmq"),
            "got: {err}"
        );
    }

    #[test]
    fn worker_queue_limit_reaches_the_policy_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--worker-queue-limit",
            "4",
        ]))
        .unwrap();
        let ca = c.model.cache_aware.expect("cache_aware set");
        assert_eq!(ca.load_gate.queue_limit(), Some(4));
        // The gate REPLACES the spread strategy — the spread knobs must not
        // survive alongside it in the built config.
        assert!(matches!(ca.load_gate, LoadGate::PerWorkerQueue(_)));
    }

    /// A limit of 0 would make every worker ineligible, silently degrading the
    /// policy to pure min-load. `NonZeroUsize` makes that unrepresentable.
    #[test]
    fn rejects_zero_worker_queue_limit() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--worker-queue-limit",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("worker-queue-limit"), "got: {err}");
    }

    #[test]
    fn rejects_worker_queue_limit_without_cache_aware_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--worker-queue-limit",
            "4",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("requires --policy cache_aware_zmq"),
            "got: {err}"
        );
    }

    #[test]
    fn min_load_choices_reaches_the_policy_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--min-load-choices",
            "3",
        ]))
        .unwrap();
        let ca = c.model.cache_aware.expect("cache_aware set");
        assert_eq!(ca.min_load_choices, 3);
    }

    #[test]
    fn min_load_choices_defaults_to_two() {
        // The default IS the herd damping this knob exists for: a silent
        // edit from 2 to 1 turns every fallback into uniform-random routing
        // and a large value disables sampling — neither fails any other test.
        assert_eq!(CacheAwareConfig::default().min_load_choices, 2);
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
        // Tuning any other knob must not disturb the shipped default.
        assert_eq!(ca.min_load_choices, 2);
    }

    #[test]
    fn rejects_min_load_choices_without_cache_aware_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--min-load-choices",
            "2",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("requires --policy cache_aware_zmq"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_zero_min_load_choices() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--min-load-choices",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("min-load-choices"), "got: {err}");
    }

    #[test]
    fn saturation_queue_floor_reaches_the_policy_config() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--worker-queue-limit",
            "14",
            "--saturation-queue-floor",
            "4",
        ]))
        .unwrap();
        let ca = c.model.cache_aware.expect("cache_aware set");
        assert_eq!(ca.saturation_queue_floor.map(NonZeroUsize::get), Some(4));
        // Tuning any other knob must not set a floor: `None` keeps the
        // historical divert-on-queued behavior.
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--worker-queue-limit",
            "14",
        ]))
        .unwrap();
        let ca = c.model.cache_aware.expect("cache_aware set");
        assert_eq!(ca.saturation_queue_floor, None);
    }

    /// Without the queue gate there is no diversion for the floor to cancel;
    /// accepting it would leave the knob silently dead.
    #[test]
    fn rejects_saturation_queue_floor_without_worker_queue_limit() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--saturation-queue-floor",
            "4",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires"), "got: {err}");
        assert!(err.contains("worker-queue-limit"), "got: {err}");
    }

    /// `floor <= limit` is the invariant that makes `cache_hit_all_queued`
    /// mean "affinity kept": a floor above the limit would declare the fleet
    /// saturated while workers the gate still admits exist.
    #[test]
    fn rejects_saturation_queue_floor_above_worker_queue_limit() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--worker-queue-limit",
            "4",
            "--saturation-queue-floor",
            "5",
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("at most"), "got: {err}");
    }

    /// The boundary is inclusive: floor == limit is the strictest useful
    /// setting (divert only when some worker is under the gate's own bar).
    #[test]
    fn accepts_saturation_queue_floor_equal_to_worker_queue_limit() {
        let c = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--policy",
            "cache_aware_zmq",
            "--worker-queue-limit",
            "4",
            "--saturation-queue-floor",
            "4",
        ]))
        .unwrap();
        let ca = c.model.cache_aware.expect("cache_aware set");
        assert_eq!(ca.saturation_queue_floor.map(NonZeroUsize::get), Some(4));
    }

    #[test]
    fn rejects_saturation_queue_floor_without_cache_aware_policy() {
        let err = into_config_owned(with_model(&[
            "--worker-urls",
            "http://x:30000",
            "--saturation-queue-floor",
            "4",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("requires --policy cache_aware_zmq"),
            "got: {err}"
        );
    }

    /// The queue gate replaces the fleet-spread check rather than layering
    /// on it, so accepting both would leave the spread knob silently dead.
    #[test]
    fn rejects_worker_queue_limit_combined_with_balance_threshold() {
        for spread in [
            ["--balance-abs-threshold", "32"],
            // The relative knob alone is just as dead, and is the arm a
            // `&&`-instead-of-`||` slip would let through.
            ["--balance-rel-threshold", "1.5"],
        ] {
            let err = into_config_owned(with_model(&[
                "--worker-urls",
                "http://x:30000",
                "--policy",
                "cache_aware_zmq",
                "--worker-queue-limit",
                "4",
                spread[0],
                spread[1],
            ]))
            .unwrap_err()
            .to_string();
            assert!(err.contains("cannot be combined"), "{spread:?} got: {err}");
        }
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
            "--stream-idle-timeout-secs",
            "90",
            "--stream-send-stall-secs",
            "45",
        ]))
        .unwrap();
        assert_eq!(c.proxy.request_timeout_secs, 120);
        assert_eq!(c.active_load.stale_request_timeout_secs, 240);
        assert_eq!(c.proxy.stream_idle_timeout_secs, 90);
        assert_eq!(c.proxy.stream_send_stall_secs, 45);
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
        assert_eq!(s.fallback_policy, PolicyKind::RoundRobin);
        assert_eq!(s.idle_secs, 600);
        assert_eq!(s.eviction_interval_secs, 60);
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
        assert_eq!(s.fallback_policy, PolicyKind::LoadBased);
        assert_eq!(s.idle_secs, 120);
        assert_eq!(s.eviction_interval_secs, 15);
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
        assert!(
            err.contains("--sticky-fallback-policy must be one of"),
            "got: {err}"
        );
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
        assert!(
            err.contains("--sticky-fallback-policy must be one of"),
            "got: {err}"
        );
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
}
