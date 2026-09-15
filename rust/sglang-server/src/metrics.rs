//! Frontend-owned observations. Collection never runs on a scheduler thread.

mod http;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};
use serde::Deserialize;

use crate::message::config::{DisaggregationMode, ServerArgs};
use crate::message::request::GenerateRequest;
use crate::message::response::ChunkEvent;

pub(crate) use http::HttpMetrics;
pub(crate) use http::apply as apply_http;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub(crate) struct MetricsConfig {
    pub labels: BTreeMap<String, String>,
    pub http_labels: BTreeMap<String, String>,
    pub allowed_custom_labels: Vec<String>,
    pub custom_labels_header: Option<String>,
    pub priority_enabled: bool,
    pub bucket_time_to_first_token: Option<Vec<f64>>,
    pub bucket_inter_token_latency: Option<Vec<f64>>,
    pub bucket_e2e_request_latency: Option<Vec<f64>>,
    pub prompt_tokens_buckets: Option<Vec<f64>>,
    pub generation_tokens_buckets: Option<Vec<f64>>,
}

// These are TokenizerMetricsCollector's defaults; the Python/Rust schema test
// checks the complete families, labels, descriptions and bucket boundaries.
const TOKEN_BUCKETS: &[f64] = &[
    100., 300., 500., 700., 1000., 1500., 2000., 3000., 4000., 5000., 6000., 7000., 8000., 9000.,
    10000., 12500., 15000., 17500., 20000., 22500., 25000., 27500., 30000., 35000., 40000., 60000.,
    80000., 100000., 200000., 300000., 400000., 600000., 800000., 1000000., 1100000.,
];
const TTFT_BUCKETS: &[f64] = &[
    0.1, 0.2, 0.4, 0.6, 0.8, 1., 2., 4., 6., 8., 10., 20., 40., 60., 80., 100., 200., 400.,
];
const E2E_BUCKETS: &[f64] = &[
    0.1, 0.2, 0.4, 0.6, 0.8, 1., 2., 4., 6., 8., 10., 20., 40., 60., 80., 100., 200., 400., 600.,
    1200., 1800., 2400.,
];
const ITL_BUCKETS: &[f64] = &[
    0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060, 0.080,
    0.100, 0.200, 0.400, 0.600, 0.800, 1., 2., 4., 6., 8.,
];

#[derive(Debug)]
pub struct FrontendMetrics {
    pub(crate) registry: Registry,
    pub(crate) config: MetricsConfig,
    clock: (Instant, f64),
    prefill: bool,
    counters: BTreeMap<&'static str, IntCounterVec>,
    histograms: BTreeMap<&'static str, HistogramVec>,
    pub(crate) http: Arc<HttpMetrics>,
}

impl FrontendMetrics {
    pub fn new(args: &ServerArgs) -> Result<Arc<Self>, String> {
        let mut config: MetricsConfig = args
            .metrics_config
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(|error| format!("metrics configuration: {error}"))?
            .unwrap_or_default();
        config
            .labels
            .entry("model_name".into())
            .or_insert_with(|| args.served_model_name.clone());
        config
            .labels
            .entry("engine_type".into())
            .or_insert_with(|| {
                match args.disaggregation_mode {
                    DisaggregationMode::Prefill => "prefill",
                    DisaggregationMode::Decode => "decode",
                    DisaggregationMode::Null => "unified",
                }
                .into()
            });
        for name in &config.allowed_custom_labels {
            config.labels.entry(name.clone()).or_default();
        }
        if config.priority_enabled {
            config.labels.entry("priority".into()).or_default();
        }
        let registry = Registry::new();
        let labels: Vec<_> = config.labels.keys().map(String::as_str).collect();
        let mut counters = BTreeMap::new();
        for (name, help, extra) in [
            (
                "prompt_tokens_total",
                "Number of prefill tokens processed.",
                Some("is_streaming"),
            ),
            (
                "generation_tokens_total",
                "Number of generation tokens processed.",
                Some("is_streaming"),
            ),
            (
                "spec_verify_calls_total",
                "Number of speculative decoding verification calls.",
                None,
            ),
            (
                "cached_tokens_total",
                "Number of cached prompt tokens by source (device/host/storage).",
                Some("cache_source"),
            ),
            (
                "num_requests_total",
                "Number of requests processed.",
                Some("is_streaming"),
            ),
            (
                "num_so_requests_total",
                "Number of structured output requests processed.",
                None,
            ),
            (
                "num_aborted_requests_total",
                "Number of requests aborted.",
                None,
            ),
        ] {
            let mut names = labels.clone();
            names.extend(extra);
            let counter = IntCounterVec::new(Opts::new(format!("sglang:{name}"), help), &names)
                .map_err(|error| error.to_string())?;
            registry
                .register(Box::new(counter.clone()))
                .map_err(|error| error.to_string())?;
            counters.insert(name, counter);
        }
        let mut histograms = BTreeMap::new();
        for (name, help, buckets, stream) in [
            (
                "prompt_tokens_histogram",
                "Histogram of prompt token length.",
                config
                    .prompt_tokens_buckets
                    .as_deref()
                    .unwrap_or(TOKEN_BUCKETS),
                false,
            ),
            (
                "uncached_prompt_tokens_histogram",
                "Histogram of uncached (compute) prompt token length.",
                config
                    .prompt_tokens_buckets
                    .as_deref()
                    .unwrap_or(TOKEN_BUCKETS),
                false,
            ),
            (
                "generation_tokens_histogram",
                "Histogram of generation token length.",
                config
                    .generation_tokens_buckets
                    .as_deref()
                    .unwrap_or(TOKEN_BUCKETS),
                false,
            ),
            (
                "get_loads_duration_seconds",
                "Time spent serving /v1/loads requests (seconds).",
                &[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0][..],
                false,
            ),
            (
                "time_to_first_token_seconds",
                "Histogram of time to first token in seconds.",
                config
                    .bucket_time_to_first_token
                    .as_deref()
                    .unwrap_or(TTFT_BUCKETS),
                true,
            ),
            (
                "inter_token_latency_seconds",
                "Histogram of inter-token latency in seconds.",
                config
                    .bucket_inter_token_latency
                    .as_deref()
                    .unwrap_or(ITL_BUCKETS),
                false,
            ),
            (
                "e2e_request_latency_seconds",
                "Histogram of End-to-end request latency in seconds",
                config
                    .bucket_e2e_request_latency
                    .as_deref()
                    .unwrap_or(E2E_BUCKETS),
                true,
            ),
        ] {
            let mut names = labels.clone();
            if stream {
                names.push("is_streaming");
            }
            let histogram = HistogramVec::new(
                HistogramOpts::new(format!("sglang:{name}"), help).buckets(buckets.to_vec()),
                &names,
            )
            .map_err(|error| error.to_string())?;
            registry
                .register(Box::new(histogram.clone()))
                .map_err(|error| error.to_string())?;
            histograms.insert(name, histogram);
        }
        let http = Arc::new(HttpMetrics::new(&registry, &config.http_labels)?);
        Ok(Arc::new(Self {
            registry,
            config,
            counters,
            histograms,
            http,
            clock: (Instant::now(), monotonic_seconds()?),
            prefill: args.disaggregation_mode == DisaggregationMode::Prefill,
        }))
    }

    pub fn render(&self) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();
        TextEncoder::new()
            .encode(&self.registry.gather(), &mut output)
            .map_err(|error| error.to_string())?;
        Ok(output)
    }

    fn counter(
        &self,
        name: &str,
        labels: &BTreeMap<String, String>,
        extra: Option<(&str, &str)>,
        value: u64,
    ) {
        let mut labels: HashMap<_, _> = labels
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        labels.extend(extra);
        if let Some(counter) = self.counters.get(name) {
            match counter.get_metric_with(&labels) {
                Ok(counter) => counter.inc_by(value),
                Err(error) => tracing::error!(%error, name, "frontend counter label mismatch"),
            }
        } else {
            tracing::error!(name, "unregistered frontend counter");
        }
    }

    fn histogram(
        &self,
        name: &str,
        labels: &BTreeMap<String, String>,
        stream: Option<bool>,
    ) -> Option<prometheus::Histogram> {
        let mut labels: HashMap<_, _> = labels
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        if let Some(stream) = stream {
            labels.insert("is_streaming", if stream { "true" } else { "false" });
        }
        let Some(histogram) = self.histograms.get(name) else {
            tracing::error!(name, "unregistered frontend histogram");
            return None;
        };
        match histogram.get_metric_with(&labels) {
            Ok(histogram) => Some(histogram),
            Err(error) => {
                tracing::error!(%error, name, "frontend histogram label mismatch");
                None
            }
        }
    }

    pub fn aborted(&self) {
        self.counter("num_aborted_requests_total", &self.config.labels, None, 1);
    }

    pub fn load_duration(&self, duration: f64) {
        if let Some(histogram) =
            self.histogram("get_loads_duration_seconds", &self.config.labels, None)
        {
            histogram.observe(duration);
        }
    }

    pub(crate) fn custom_labels(
        &self,
        headers: &axum::http::HeaderMap,
    ) -> Option<BTreeMap<String, String>> {
        if self.config.allowed_custom_labels.is_empty() {
            return None;
        }
        let header = self.config.custom_labels_header.as_ref()?;
        let raw = headers.get(header)?.to_str().ok()?;
        let labels: BTreeMap<String, String> = match serde_json::from_str(raw) {
            Ok(labels) => labels,
            Err(error) => {
                tracing::warn!(%error, "ignoring malformed custom metric labels header");
                return None;
            }
        };
        Some(
            labels
                .into_iter()
                .filter(|(name, _)| self.config.allowed_custom_labels.contains(name))
                .collect(),
        )
    }
}

/// A scheduler request owns one observation state on its detokenizer shard.
/// Dropping it without a terminal output does not fabricate a completion.
#[derive(Clone, Debug)]
pub struct RequestMetrics {
    metrics: Arc<FrontendMetrics>,
    labels: BTreeMap<String, String>,
    created: Instant,
    received_age: f64,
    last: Option<Instant>,
    tokens: u64,
    streaming: bool,
    grammar: bool,
    finished: bool,
    itl: Option<prometheus::Histogram>,
}

impl RequestMetrics {
    pub fn new(metrics: Arc<FrontendMetrics>, request: &GenerateRequest) -> Option<Self> {
        if request.log_metrics == Some(false) {
            return None;
        }
        let mut labels = metrics.config.labels.clone();
        for name in &metrics.config.allowed_custom_labels {
            if let Some(value) = request
                .custom_labels
                .as_ref()
                .and_then(|labels| labels.get(name))
            {
                labels.insert(name.clone(), value.clone());
            }
        }
        if metrics.config.priority_enabled
            && let Some(priority) = request.priority
        {
            labels.insert("priority".into(), priority.to_string());
        }
        let params = &request.sampling_params;
        let created = request.started.unwrap_or_else(Instant::now);
        // received_time is Python perf_counter/CLOCK_MONOTONIC, not Unix time.
        let received_age = request
            .received_time
            .filter(|value| *value != 0.)
            .map(|received| {
                metrics.clock.1 + created.duration_since(metrics.clock.0).as_secs_f64() - received
            })
            .unwrap_or(0.);
        Some(Self {
            metrics,
            labels,
            created,
            received_age,
            last: None,
            tokens: 0,
            streaming: request.stream,
            grammar: params.json_schema.is_some()
                || params.regex.is_some()
                || params.ebnf.is_some()
                || params.structural_tag.is_some(),
            finished: false,
            itl: None,
        })
    }

    pub fn observe(&mut self, output: &ChunkEvent) {
        self.observe_at(output, Instant::now());
    }

    fn observe_at(&mut self, output: &ChunkEvent, now: Instant) {
        if self.finished {
            return;
        }
        let tokens = output
            .counts
            .generation_tokens
            .unwrap_or(self.tokens.saturating_add(output.completion_tokens));
        let first = self.last.is_none();
        let previous_time = *self.last.get_or_insert(now);
        if first && !self.metrics.prefill {
            if let Some(histogram) = self.metrics.histogram(
                "time_to_first_token_seconds",
                &self.labels,
                Some(self.streaming),
            ) {
                histogram
                    .observe(now.duration_since(self.created).as_secs_f64() + self.received_age);
            }
        } else if let Some(new_tokens) = tokens.checked_sub(self.tokens).filter(|&n| n > 0) {
            let interval = now.duration_since(previous_time).as_secs_f64();
            if self.itl.is_none() {
                self.itl =
                    self.metrics
                        .histogram("inter_token_latency_seconds", &self.labels, None);
            }
            if let Some(histogram) = &self.itl {
                // Use the registry's local accumulator so a speculative chunk
                // adds k observations with one shared-counter flush.
                let local = histogram.local();
                for _ in 0..new_tokens {
                    local.observe(interval / new_tokens as f64);
                }
                local.flush();
            }
            self.last = Some(now);
        }
        self.tokens = tokens;
        if output.finish_reason.is_none() {
            return;
        }
        self.finished = true;
        let labels = &self.labels;
        let stream = Some((
            "is_streaming",
            if self.streaming { "true" } else { "false" },
        ));
        self.metrics.counter(
            "prompt_tokens_total",
            labels,
            stream,
            output.prompt_tokens.into(),
        );
        self.metrics
            .counter("generation_tokens_total", labels, stream, tokens);
        self.metrics
            .counter("num_requests_total", labels, stream, 1);
        if self.grammar {
            self.metrics
                .counter("num_so_requests_total", labels, None, 1);
        }
        if let Some(calls) = output.counts.spec_verify_ct.filter(|&value| value > 0) {
            self.metrics
                .counter("spec_verify_calls_total", labels, None, calls);
        }
        let cached = output.counts.cached_tokens;
        if cached > 0 {
            if let Some(Some(details)) = &output.counts.cached_tokens_details {
                for (source, value) in [
                    ("device", details.device),
                    ("host", details.host),
                    ("storage", details.storage.unwrap_or(0)),
                ] {
                    if value > 0 {
                        self.metrics.counter(
                            "cached_tokens_total",
                            labels,
                            Some(("cache_source", source)),
                            value,
                        );
                    }
                }
            } else {
                self.metrics.counter(
                    "cached_tokens_total",
                    labels,
                    Some(("cache_source", "total")),
                    cached,
                );
            }
        }
        for (name, value, stream) in [
            ("prompt_tokens_histogram", output.prompt_tokens as f64, None),
            (
                "uncached_prompt_tokens_histogram",
                output.prompt_tokens as f64 - cached as f64,
                None,
            ),
            ("generation_tokens_histogram", tokens as f64, None),
            (
                "e2e_request_latency_seconds",
                now.duration_since(self.created).as_secs_f64() + self.received_age,
                Some(self.streaming),
            ),
        ] {
            if let Some(histogram) = self.metrics.histogram(name, labels, stream) {
                histogram.observe(value);
            }
        }
    }
}

pub(crate) fn realtime_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("request clock is after the Unix epoch")
        .as_secs_f64()
}

pub(crate) fn monotonic_seconds() -> Result<f64, String> {
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: timestamp points to writable timespec storage; CLOCK_MONOTONIC
    // is the clock used by Python perf_counter on the supported Linux hosts.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timestamp) };
    if result != 0 {
        return Err(format!(
            "reading request clock: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(timestamp.tv_sec as f64 + timestamp.tv_nsec as f64 / 1e9)
}
