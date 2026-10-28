//! The frontend's request metrics: the families Python's `TokenizerManager`
//! records through `TokenizerMetricsCollector`, under the same names, help,
//! labels and default buckets (`testdata/tokenizer_metrics.json`, generated
//! from Python, pins them). Each request observes them through its
//! [`RequestMetrics`] as `TokenizerManager.collect_metrics` does; values are
//! atomics, so the per-output cost is a few atomic adds.
//!
//! [`FrontendMetrics::flush`] writes them as this process's prometheus_client
//! multiprocess files, so every reader of the directory merges them like any
//! other process's samples: this listener's `/metrics`, another DP rank's
//! listener (which sums every rank's requests, as Python's single
//! `TokenizerManager` counts them), and Python's own collector.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::message::config::{DisaggregationMode, MetricsConfig};
use crate::message::finish_reason::FinishReason;
use crate::message::request::GenerateRequest;
use crate::message::response::{CachedTokensSplit, ChunkEvent, ResponseItem};
use crate::utils::prometheus::{self, Entry};

/// How often the frontend's files are rewritten for other readers of the
/// directory. A scrape of this listener writes them first, so this bounds only
/// how stale another listener's view of this one can be.
pub(crate) const FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// The least time between two logged flush failures.
const FLUSH_WARNING_INTERVAL: Duration = Duration::from_secs(60);

const LATENCY_BUCKETS: &[f64] = &[
    0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0,
    400.0,
];
const E2E_LATENCY_BUCKETS: &[f64] = &[
    0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0,
    400.0, 600.0, 1200.0, 1800.0, 2400.0,
];
const INTER_TOKEN_LATENCY_BUCKETS: &[f64] = &[
    0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060, 0.080,
    0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 2.000, 4.000, 6.000, 8.000,
];
const TOKEN_COUNT_BUCKETS: &[f64] = &[
    100.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0,
    8000.0, 9000.0, 10000.0, 12500.0, 15000.0, 17500.0, 20000.0, 22500.0, 25000.0, 27500.0,
    30000.0, 35000.0, 40000.0, 60000.0, 80000.0, 100000.0, 200000.0, 300000.0, 400000.0, 600000.0,
    800000.0, 1000000.0, 1100000.0,
];
const GET_LOADS_BUCKETS: &[f64] = &[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0];
/// The `cache_source` label of `sglang:cached_tokens_total`, by cache tier.
const CACHE_SOURCES: [&str; 3] = ["device", "host", "storage"];

type Labels = Vec<(String, String)>;
/// What tells one request's label set from another's: the priority (when it
/// is a label) and whether it streams.
type LabelSetKey = (Option<i64>, bool);

/// A prometheus_client counter series. Every increment here is a whole count.
#[derive(Default)]
struct Counter(AtomicU64);

impl Counter {
    fn inc(&self, amount: u64) {
        self.0.fetch_add(amount, Ordering::Relaxed);
    }
}

/// A prometheus_client histogram series: per-bucket (not cumulative) counts,
/// the last bucket `+Inf`, and the sum as `f64` bits.
struct Histogram {
    bounds: Arc<[f64]>,
    buckets: Box<[AtomicU64]>,
    sum: AtomicU64,
}

impl Histogram {
    fn new(bounds: Arc<[f64]>) -> Self {
        Self {
            buckets: bounds.iter().map(|_| AtomicU64::new(0)).collect(),
            bounds,
            sum: AtomicU64::new(0f64.to_bits()),
        }
    }

    fn observe(&self, value: f64) {
        self.observe_many(value, value, 1);
    }

    /// `count` observations of `value` that add `total` to the sum: Python's
    /// fast path in `observe_inter_token_latency`.
    fn observe_many(&self, total: f64, value: f64, count: u64) {
        let _ = self
            .sum
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                Some((f64::from_bits(bits) + total).to_bits())
            });
        if let Some(i) = self.bounds.iter().position(|&bound| value <= bound) {
            self.buckets[i].fetch_add(count, Ordering::Relaxed);
        }
    }
}

/// One counter family and the label sets observed so far.
struct CounterFamily {
    name: &'static str,
    help: &'static str,
    series: Mutex<Vec<(Labels, Arc<Counter>)>>,
}

/// One histogram family, its bucket bounds and the label sets observed so far.
struct HistogramFamily {
    name: &'static str,
    help: &'static str,
    bounds: Arc<[f64]>,
    series: Mutex<Vec<(Labels, Arc<Histogram>)>>,
}

impl CounterFamily {
    fn new(name: &'static str, help: &'static str) -> Self {
        Self {
            name,
            help,
            series: Mutex::new(Vec::new()),
        }
    }

    fn series(&self, labels: &Labels) -> Arc<Counter> {
        let mut series = self.series.lock().unwrap();
        if let Some((_, counter)) = series.iter().find(|(l, _)| l == labels) {
            return counter.clone();
        }
        let counter = Arc::new(Counter::default());
        series.push((labels.clone(), counter.clone()));
        counter
    }

    /// prometheus_client's entries: the family name loses its `_total`, the
    /// sample keeps it.
    fn entries(&self, out: &mut Vec<Entry>) {
        let metric_name = self.name.strip_suffix("_total").unwrap_or(self.name);
        for (labels, counter) in self.series.lock().unwrap().iter() {
            out.push(Entry {
                key: prometheus::mmap_key(metric_name, self.name, labels, self.help),
                value: counter.0.load(Ordering::Relaxed) as f64,
                timestamp: 0.0,
            });
        }
    }
}

impl HistogramFamily {
    /// `buckets` as prometheus_client's `Histogram` takes them: sorted, and
    /// `+Inf` appended unless already last.
    fn new(name: &'static str, help: &'static str, buckets: &[f64]) -> Result<Self, String> {
        if buckets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(format!("{name}: buckets not in sorted order: {buckets:?}"));
        }
        let mut bounds = buckets.to_vec();
        if bounds.last() != Some(&f64::INFINITY) {
            bounds.push(f64::INFINITY);
        }
        if bounds.len() < 2 {
            return Err(format!("{name}: must have at least two buckets"));
        }
        Ok(Self {
            name,
            help,
            bounds: bounds.into(),
            series: Mutex::new(Vec::new()),
        })
    }

    fn series(&self, labels: &Labels) -> Arc<Histogram> {
        let mut series = self.series.lock().unwrap();
        if let Some((_, histogram)) = series.iter().find(|(l, _)| l == labels) {
            return histogram.clone();
        }
        let histogram = Arc::new(Histogram::new(self.bounds.clone()));
        series.push((labels.clone(), histogram.clone()));
        histogram
    }

    /// prometheus_client's entries, in its creation order: `_sum`, then one
    /// `_bucket` per bound.
    fn entries(&self, out: &mut Vec<Entry>) {
        for (labels, histogram) in self.series.lock().unwrap().iter() {
            let sum = f64::from_bits(histogram.sum.load(Ordering::Relaxed));
            out.push(Entry {
                key: prometheus::mmap_key(
                    self.name,
                    &format!("{}_sum", self.name),
                    labels,
                    self.help,
                ),
                value: sum,
                timestamp: 0.0,
            });
            let bucket_name = format!("{}_bucket", self.name);
            for (bound, count) in histogram.bounds.iter().zip(histogram.buckets.iter()) {
                let mut labels = labels.clone();
                labels.push(("le".into(), prometheus::float_to_go_string(*bound)));
                out.push(Entry {
                    key: prometheus::mmap_key(self.name, &bucket_name, &labels, self.help),
                    value: count.load(Ordering::Relaxed) as f64,
                    timestamp: 0.0,
                });
            }
        }
    }
}

/// The series one request observes, resolved once per label set. A latency
/// this request cannot record has no series, so it stays absent rather than
/// reading as zero.
struct RequestSeries {
    num_requests: Arc<Counter>,
    prompt_tokens: Arc<Counter>,
    generation_tokens: Arc<Counter>,
    num_so_requests: Arc<Counter>,
    /// Cached tokens by [`CACHE_SOURCES`], each series registered at its
    /// source's first hit, as Python labels a source only once it counts one.
    cached_tokens: [OnceLock<Arc<Counter>>; 3],
    prompt_tokens_histogram: Arc<Histogram>,
    uncached_prompt_tokens_histogram: Arc<Histogram>,
    generation_tokens_histogram: Arc<Histogram>,
    e2e_request_latency: Arc<Histogram>,
    /// None of these on a PD prefill server.
    time_to_first_token: Option<Arc<Histogram>>,
    time_per_output_token: Option<Arc<Histogram>>,
    inter_token_latency: Option<Arc<Histogram>>,
    /// The label set, which `cached_tokens` extends with its `cache_source`.
    labels: Labels,
}

/// The frontend's metric families and where they are written.
pub(crate) struct FrontendMetrics {
    /// `tokenizer_metric_labels()`: the labels on every family.
    labels: Labels,
    /// `--enable-priority-scheduling`: the `priority` label takes each
    /// request's priority.
    priority_label: bool,
    /// A PD prefill server never observes TTFT, inter-token latency or TPOT.
    prefill: bool,
    num_requests: CounterFamily,
    prompt_tokens: CounterFamily,
    generation_tokens: CounterFamily,
    num_so_requests: CounterFamily,
    num_aborted_requests: CounterFamily,
    cached_tokens: CounterFamily,
    prompt_tokens_histogram: HistogramFamily,
    uncached_prompt_tokens_histogram: HistogramFamily,
    generation_tokens_histogram: HistogramFamily,
    time_to_first_token: HistogramFamily,
    inter_token_latency: HistogramFamily,
    time_per_output_token: HistogramFamily,
    e2e_request_latency: HistogramFamily,
    get_loads_duration: HistogramFamily,
    by_label_set: RwLock<HashMap<LabelSetKey, Arc<RequestSeries>>>,
    dir: PathBuf,
    /// This process's files, named apart from the ones prometheus_client
    /// writes for the scheduler running in the same process.
    counter_file: String,
    histogram_file: String,
    /// The bytes last written per file; an unchanged file is not rewritten.
    written: Mutex<BTreeMap<String, Vec<u8>>>,
    /// When [`Self::flush_or_warn`] last logged a failure.
    last_flush_warning: Mutex<Option<Instant>>,
}

impl FrontendMetrics {
    pub(crate) fn new(
        config: &MetricsConfig,
        disaggregation_mode: DisaggregationMode,
    ) -> Result<Self, String> {
        let buckets = |configured: &Option<Vec<f64>>, default: &[f64]| {
            configured.clone().unwrap_or_else(|| default.to_vec())
        };
        let pid = std::process::id();
        Ok(Self {
            labels: config
                .labels
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            priority_label: config.priority_label,
            prefill: disaggregation_mode == DisaggregationMode::Prefill,
            num_requests: CounterFamily::new(
                "sglang:num_requests_total",
                "Number of requests processed.",
            ),
            prompt_tokens: CounterFamily::new(
                "sglang:prompt_tokens_total",
                "Number of prefill tokens processed.",
            ),
            generation_tokens: CounterFamily::new(
                "sglang:generation_tokens_total",
                "Number of generation tokens processed.",
            ),
            num_so_requests: CounterFamily::new(
                "sglang:num_so_requests_total",
                "Number of structured output requests processed.",
            ),
            num_aborted_requests: CounterFamily::new(
                "sglang:num_aborted_requests_total",
                "Number of requests aborted.",
            ),
            cached_tokens: CounterFamily::new(
                "sglang:cached_tokens_total",
                "Number of cached prompt tokens by source (device/host/storage).",
            ),
            prompt_tokens_histogram: HistogramFamily::new(
                "sglang:prompt_tokens_histogram",
                "Histogram of prompt token length.",
                &buckets(&config.bucket_prompt_tokens, TOKEN_COUNT_BUCKETS),
            )?,
            uncached_prompt_tokens_histogram: HistogramFamily::new(
                "sglang:uncached_prompt_tokens_histogram",
                "Histogram of uncached (compute) prompt token length.",
                &buckets(&config.bucket_prompt_tokens, TOKEN_COUNT_BUCKETS),
            )?,
            generation_tokens_histogram: HistogramFamily::new(
                "sglang:generation_tokens_histogram",
                "Histogram of generation token length.",
                &buckets(&config.bucket_generation_tokens, TOKEN_COUNT_BUCKETS),
            )?,
            time_to_first_token: HistogramFamily::new(
                "sglang:time_to_first_token_seconds",
                "Histogram of time to first token in seconds.",
                &buckets(&config.bucket_time_to_first_token, LATENCY_BUCKETS),
            )?,
            inter_token_latency: HistogramFamily::new(
                "sglang:inter_token_latency_seconds",
                "Histogram of inter-token latency in seconds.",
                &buckets(
                    &config.bucket_inter_token_latency,
                    INTER_TOKEN_LATENCY_BUCKETS,
                ),
            )?,
            time_per_output_token: HistogramFamily::new(
                "sglang:request_time_per_output_token_seconds",
                "Per-request TPOT in seconds: (finished_time - first_token_time) / (completion_tokens - 1).",
                &buckets(
                    &config.bucket_inter_token_latency,
                    INTER_TOKEN_LATENCY_BUCKETS,
                ),
            )?,
            e2e_request_latency: HistogramFamily::new(
                "sglang:e2e_request_latency_seconds",
                "Histogram of End-to-end request latency in seconds",
                &buckets(&config.bucket_e2e_request_latency, E2E_LATENCY_BUCKETS),
            )?,
            get_loads_duration: HistogramFamily::new(
                "sglang:get_loads_duration_seconds",
                "Time spent serving /v1/loads requests (seconds).",
                GET_LOADS_BUCKETS,
            )?,
            by_label_set: RwLock::new(HashMap::new()),
            dir: PathBuf::from(&config.multiproc_dir),
            counter_file: format!("counter_rust-{pid}.db"),
            histogram_file: format!("histogram_rust-{pid}.db"),
            written: Mutex::new(BTreeMap::new()),
            last_flush_warning: Mutex::new(None),
        })
    }

    /// Start observing one generation request, created at `created_at`.
    pub(crate) fn start_request(
        self: &Arc<Self>,
        request: &GenerateRequest,
        created_at: Instant,
    ) -> RequestMetrics {
        let sampling = &request.sampling_params;
        let has_grammar = [
            &sampling.json_schema,
            &sampling.regex,
            &sampling.ebnf,
            &sampling.structural_tag,
        ]
        .into_iter()
        .any(|grammar| grammar.as_deref().is_some_and(|g| !g.is_empty()));
        RequestMetrics {
            metrics: self.clone(),
            series: self.request_series(request.priority, request.stream),
            stream: request.stream,
            has_grammar,
            created_at,
            first_output_at: None,
            last_at: created_at,
            ttft_observed: false,
            ttft_pending: false,
            completion_tokens: 0,
            last_completion_tokens: 0,
            prompt_tokens: 0,
            cached_tokens: 0,
            cached_tokens_split: CachedTokensSplit::default(),
            terminal: false,
        }
    }

    /// `/v1/loads` time spent reading the snapshots.
    pub(crate) fn observe_get_loads(&self, elapsed: Duration) {
        self.get_loads_duration
            .series(&self.labels)
            .observe(elapsed.as_secs_f64());
    }

    fn request_series(&self, priority: Option<i64>, stream: bool) -> Arc<RequestSeries> {
        let priority = priority.filter(|_| self.priority_label);
        if let Some(series) = self.by_label_set.read().unwrap().get(&(priority, stream)) {
            return series.clone();
        }
        // Python's `collect_metrics` labels, `is_streaming` added where the
        // family takes it.
        let mut labels = self.labels.clone();
        if let Some(priority) = priority
            && let Some((_, value)) = labels.iter_mut().find(|(k, _)| k == "priority")
        {
            *value = priority.to_string();
        }
        let mut stream_labels = labels.clone();
        stream_labels.push(("is_streaming".into(), stream.to_string()));
        let decodes = !self.prefill;
        let series = Arc::new(RequestSeries {
            num_requests: self.num_requests.series(&stream_labels),
            prompt_tokens: self.prompt_tokens.series(&stream_labels),
            generation_tokens: self.generation_tokens.series(&stream_labels),
            num_so_requests: self.num_so_requests.series(&labels),
            cached_tokens: Default::default(),
            prompt_tokens_histogram: self.prompt_tokens_histogram.series(&labels),
            uncached_prompt_tokens_histogram: self.uncached_prompt_tokens_histogram.series(&labels),
            generation_tokens_histogram: self.generation_tokens_histogram.series(&labels),
            e2e_request_latency: self.e2e_request_latency.series(&stream_labels),
            time_to_first_token: decodes.then(|| self.time_to_first_token.series(&stream_labels)),
            time_per_output_token: decodes
                .then(|| self.time_per_output_token.series(&stream_labels)),
            inter_token_latency: decodes.then(|| self.inter_token_latency.series(&labels)),
            labels,
        });
        self.by_label_set
            .write()
            .unwrap()
            .entry((priority, stream))
            .or_insert(series)
            .clone()
    }

    fn counters(&self) -> [&CounterFamily; 6] {
        [
            &self.num_requests,
            &self.prompt_tokens,
            &self.generation_tokens,
            &self.num_so_requests,
            &self.num_aborted_requests,
            &self.cached_tokens,
        ]
    }

    fn histograms(&self) -> [&HistogramFamily; 8] {
        [
            &self.prompt_tokens_histogram,
            &self.uncached_prompt_tokens_histogram,
            &self.generation_tokens_histogram,
            &self.time_to_first_token,
            &self.inter_token_latency,
            &self.time_per_output_token,
            &self.e2e_request_latency,
            &self.get_loads_duration,
        ]
    }

    /// Write the current values as this process's multiprocess files, each
    /// only if it changed since the last write.
    pub(crate) fn flush(&self) -> io::Result<()> {
        let mut counters = Vec::new();
        for family in self.counters() {
            family.entries(&mut counters);
        }
        let mut histograms = Vec::new();
        for family in self.histograms() {
            family.entries(&mut histograms);
        }
        let mut written = self.written.lock().unwrap();
        for (file, entries) in [
            (&self.counter_file, counters),
            (&self.histogram_file, histograms),
        ] {
            if entries.is_empty() {
                continue;
            }
            let data = prometheus::encode_entries(&entries);
            if written.get(file) != Some(&data) {
                prometheus::write_file(&self.dir, file, &data)?;
                written.insert(file.clone(), data);
            }
        }
        Ok(())
    }

    /// [`Self::flush`], best effort: a full or read-only directory must not
    /// fail a scrape of what is already on disk, nor log every second.
    pub(crate) fn flush_or_warn(&self) {
        let Err(error) = self.flush() else {
            return;
        };
        let mut last = self.last_flush_warning.lock().unwrap();
        if last.is_none_or(|at| at.elapsed() >= FLUSH_WARNING_INTERVAL) {
            *last = Some(Instant::now());
            tracing::warn!(%error, dir = %self.dir.display(), "writing frontend metrics failed");
        }
    }
}

/// One request's progress through `TokenizerManager.collect_metrics`.
pub(crate) struct RequestMetrics {
    metrics: Arc<FrontendMetrics>,
    series: Arc<RequestSeries>,
    stream: bool,
    has_grammar: bool,
    created_at: Instant,
    first_output_at: Option<Instant>,
    /// The inter-token latency baseline.
    last_at: Instant,
    ttft_observed: bool,
    /// A non-streaming request passed the first-token checks; its first-token
    /// time comes with its terminal row (see [`Self::observe_output`]).
    ttft_pending: bool,
    completion_tokens: u64,
    last_completion_tokens: u64,
    prompt_tokens: u32,
    cached_tokens: u32,
    /// `cached_tokens` by cache tier.
    cached_tokens_split: CachedTokensSplit,
    /// A terminal item (or the end of the stream) was seen; dropping the call
    /// before that aborts the request (see [`Self::call_dropped`]).
    terminal: bool,
}

impl RequestMetrics {
    /// Observe one runtime item; `None` is the stream ending.
    pub(crate) fn observe(&mut self, item: Option<&ResponseItem>) {
        match item {
            Some(ResponseItem::Frame(chunk)) => self.observe_output(chunk, false),
            Some(ResponseItem::Done(chunk)) => self.observe_output(chunk, true),
            Some(_) | None => self.terminal = true,
        }
    }

    fn observe_output(&mut self, chunk: &ChunkEvent, finished: bool) {
        let extras = chunk.extras.as_deref();
        if extras.is_some_and(|extras| extras.abort_output) {
            // Ended outside the output stream: Python's `_handle_abort_req`
            // records nothing for it.
            self.terminal = true;
            return;
        }
        let now = Instant::now();
        self.completion_tokens += chunk.completion_tokens;
        self.prompt_tokens = chunk.prompt_tokens;
        self.cached_tokens = chunk.cached_tokens;
        self.cached_tokens_split = extras
            .and_then(|extras| extras.cached_tokens_details.as_ref())
            .map_or(
                CachedTokensSplit {
                    device: chunk.cached_tokens,
                    ..Default::default()
                },
                |details| CachedTokensSplit {
                    device: details.device,
                    host: details.host,
                    storage: details.storage.unwrap_or(0),
                },
            );
        let first_output_at = *self.first_output_at.get_or_insert(now);
        let finish_type = chunk
            .finish_reason
            .as_ref()
            .and_then(FinishReason::kind_name);
        if !self.ttft_observed {
            if finish_type != Some("abort") && !self.metrics.prefill {
                self.ttft_observed = true;
                self.last_at = now;
                self.last_completion_tokens = self.completion_tokens;
                if self.stream {
                    self.observe_first_token(first_output_at);
                } else {
                    // Python times a non-streaming request's first token by
                    // the scheduler's prefill stamp; this frontend learns it
                    // from the terminal row's decode latency.
                    self.ttft_pending = true;
                }
            }
        } else if self.completion_tokens > self.last_completion_tokens {
            let new_tokens = self.completion_tokens - self.last_completion_tokens;
            let interval = (now - self.last_at).as_secs_f64();
            if let Some(histogram) = &self.series.inter_token_latency {
                histogram.observe_many(interval, interval / new_tokens as f64, new_tokens);
            }
            self.last_at = now;
            self.last_completion_tokens = self.completion_tokens;
        }
        if finished {
            self.terminal = true;
            // The terminal output less the scheduler's decode latency is the
            // prefill stamp. Without one (a streaming request, or no stamp),
            // Python falls back to the first output, as here.
            let first_token_at = extras
                .and_then(|extras| extras.decode_latency)
                .filter(|_| !self.stream)
                .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                .and_then(|latency| now.checked_sub(latency))
                .unwrap_or(first_output_at);
            if std::mem::take(&mut self.ttft_pending) {
                self.observe_first_token(first_token_at);
            }
            self.observe_finished(finish_type, first_token_at, now);
        }
    }

    fn observe_first_token(&self, first_token_at: Instant) {
        if let Some(histogram) = &self.series.time_to_first_token {
            histogram.observe(
                first_token_at
                    .saturating_duration_since(self.created_at)
                    .as_secs_f64(),
            );
        }
    }

    /// `observe_one_finished_request`.
    fn observe_finished(&self, finish_type: Option<&str>, first_token_at: Instant, now: Instant) {
        let series = &self.series;
        series.prompt_tokens.inc(u64::from(self.prompt_tokens));
        series.generation_tokens.inc(self.completion_tokens);
        if self.cached_tokens > 0 {
            let CachedTokensSplit {
                device,
                host,
                storage,
            } = self.cached_tokens_split;
            for ((counter, source), count) in series
                .cached_tokens
                .iter()
                .zip(CACHE_SOURCES)
                .zip([device, host, storage])
            {
                if count > 0 {
                    counter
                        .get_or_init(|| {
                            let mut labels = series.labels.clone();
                            labels.push(("cache_source".into(), source.into()));
                            self.metrics.cached_tokens.series(&labels)
                        })
                        .inc(u64::from(count));
                }
            }
        }
        series.num_requests.inc(1);
        if self.has_grammar {
            series.num_so_requests.inc(1);
        }
        series
            .e2e_request_latency
            .observe((now - self.created_at).as_secs_f64());
        if let Some(histogram) = &series.time_per_output_token
            && matches!(finish_type, Some("stop" | "length"))
            && self.completion_tokens > 1
        {
            let decode = now.saturating_duration_since(first_token_at).as_secs_f64();
            if decode > 0.0 {
                histogram.observe(decode / (self.completion_tokens - 1) as f64);
            }
        }
        series
            .prompt_tokens_histogram
            .observe(f64::from(self.prompt_tokens));
        series
            .uncached_prompt_tokens_histogram
            .observe(f64::from(self.prompt_tokens) - f64::from(self.cached_tokens));
        series
            .generation_tokens_histogram
            .observe(self.completion_tokens as f64);
    }

    /// The request's call is being dropped. Before its terminal item, that
    /// aborts the request (the client went away): Python counts the abort with
    /// the base labels, then records the request as finished when the
    /// scheduler's abort output arrives. Here the finish has the tokens seen so
    /// far, and needs one output: before it, the request may still have been
    /// queued, which Python's scheduler ends outside the stream, unrecorded.
    pub(crate) fn call_dropped(&self) {
        if self.terminal {
            return;
        }
        let metrics = &self.metrics;
        metrics.num_aborted_requests.series(&metrics.labels).inc(1);
        if let Some(first_output_at) = self.first_output_at {
            self.observe_finished(Some("abort"), first_output_at, Instant::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::finish_reason::{AbortReason, FinishKind};
    use crate::message::ids::Rid;
    use crate::message::response::ChunkExtras;

    /// One family as Python's `TokenizerMetricsCollector` registers it.
    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Spec {
        name: String,
        kind: String,
        documentation: String,
        labelnames: Vec<String>,
        buckets: Option<Vec<f64>>,
    }

    #[derive(serde::Deserialize)]
    struct Fixture {
        labels: BTreeMap<String, String>,
        families: Vec<Spec>,
    }

    /// Families Python registers that this frontend does not record yet, each
    /// for want of data the scheduler does not send it.
    const NOT_RECORDED: &[&str] = &[
        // Startup phases are reported to the main process, not the frontend.
        "sglang:startup_time_seconds",
        "sglang:startup_cuda_graph_time_seconds",
        // Speculative verify counts are not on the output frames.
        "sglang:spec_verify_calls_total",
    ];

    fn fixture() -> Fixture {
        serde_json::from_str(include_str!("../../../testdata/tokenizer_metrics.json")).unwrap()
    }

    fn config(dir: &std::path::Path, labels: BTreeMap<String, String>) -> MetricsConfig {
        MetricsConfig {
            labels,
            ..MetricsConfig::for_dir(dir)
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("frontend-metrics-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Every family carries Python's name, type, help, label names and default
    /// buckets; every other family Python registers is listed as not recorded.
    #[test]
    fn families_match_python() {
        let fixture = fixture();
        let metrics = FrontendMetrics::new(
            &config(&temp_dir("spec"), fixture.labels.clone()),
            DisaggregationMode::Null,
        )
        .unwrap();
        let base: Vec<String> = fixture.labels.keys().cloned().collect();
        let label_names = |name: &str| {
            let extra = match name {
                "sglang:num_requests_total"
                | "sglang:prompt_tokens_total"
                | "sglang:generation_tokens_total"
                | "sglang:time_to_first_token_seconds"
                | "sglang:request_time_per_output_token_seconds"
                | "sglang:e2e_request_latency_seconds" => Some("is_streaming"),
                "sglang:cached_tokens_total" => Some("cache_source"),
                _ => None,
            };
            let mut names = base.clone();
            names.extend(extra.map(String::from));
            names
        };
        let mut ours: Vec<Spec> = metrics
            .counters()
            .into_iter()
            .map(|family| Spec {
                name: family.name.into(),
                kind: "counter".into(),
                documentation: family.help.into(),
                labelnames: label_names(family.name),
                buckets: None,
            })
            .chain(metrics.histograms().into_iter().map(|family| Spec {
                name: family.name.into(),
                kind: "histogram".into(),
                documentation: family.help.into(),
                labelnames: label_names(family.name),
                buckets: Some(family.bounds[..family.bounds.len() - 1].to_vec()),
            }))
            .collect();
        ours.sort_by(|a, b| a.name.cmp(&b.name));
        let (mut python, not_recorded): (Vec<Spec>, Vec<Spec>) = fixture
            .families
            .into_iter()
            .partition(|spec| !NOT_RECORDED.contains(&spec.name.as_str()));
        python.sort_by(|a, b| a.name.cmp(&b.name));
        for spec in &mut python {
            spec.labelnames.sort();
        }
        for spec in &mut ours {
            spec.labelnames.sort();
        }
        assert_eq!(ours, python);
        assert_eq!(not_recorded.len(), NOT_RECORDED.len());
    }

    fn request(stream: bool, priority: Option<i64>) -> GenerateRequest {
        GenerateRequest {
            rid: Rid::new(),
            stream,
            priority,
            ..Default::default()
        }
    }

    fn chunk(tokens: u64, finish_reason: Option<FinishReason>) -> ChunkEvent {
        ChunkEvent {
            prompt_tokens: 7,
            completion_tokens: tokens,
            finish_reason,
            ..Default::default()
        }
    }

    fn stop() -> Option<FinishReason> {
        Some(FinishReason::Known(FinishKind::Stop { matched: None }))
    }

    /// A terminal chunk with the given extras metadata.
    fn terminal(
        tokens: u64,
        finish_reason: Option<FinishReason>,
        extras: ChunkExtras,
    ) -> ChunkEvent {
        ChunkEvent {
            extras: Some(Box::new(extras)),
            ..chunk(tokens, finish_reason)
        }
    }

    fn rendered(metrics: &FrontendMetrics, dir: &std::path::Path) -> String {
        metrics.flush().unwrap();
        let files = prometheus::read_dir_files(dir).unwrap();
        prometheus::render(&prometheus::merge(
            files
                .iter()
                .map(|(name, data)| (name.as_str(), data.as_slice())),
        ))
    }

    fn sample(text: &str, prefix: &str) -> Option<f64> {
        text.lines()
            .find(|line| line.starts_with(prefix))
            .map(|line| line.rsplit(' ').next().unwrap().parse().unwrap())
    }

    /// The `collect_metrics` sequence: streaming TTFT on the first output,
    /// inter-token latency per new token after it, and the finished-request
    /// families at the end; a non-streaming request's first token is the
    /// terminal output less the scheduler's decode latency; cached tokens count
    /// as device hits; a dropped call counts an abort, and a finished request
    /// once it had an output.
    #[test]
    fn records_request_lifecycles() {
        let dir = temp_dir("lifecycle");
        let labels = BTreeMap::from([
            ("engine_type".to_string(), "unified".to_string()),
            ("model_name".to_string(), "m".to_string()),
            ("priority".to_string(), String::new()),
        ]);
        let mut config = config(&dir, labels);
        config.priority_label = true;
        let metrics = Arc::new(FrontendMetrics::new(&config, DisaggregationMode::Null).unwrap());
        let created = Instant::now() - Duration::from_millis(50);

        let mut streamed = metrics.start_request(&request(true, Some(3)), created);
        streamed.observe(Some(&ResponseItem::Frame(chunk(1, None))));
        streamed.observe(Some(&ResponseItem::Frame(chunk(2, None))));
        streamed.observe(Some(&ResponseItem::Done(chunk(1, stop()))));
        streamed.call_dropped();

        let mut unary = metrics.start_request(&request(false, None), created);
        let decoded = ChunkExtras {
            decode_latency: Some(0.02),
            ..Default::default()
        };
        unary.observe(Some(&ResponseItem::Done(ChunkEvent {
            cached_tokens: 3,
            ..terminal(5, stop(), decoded)
        })));
        unary.call_dropped();

        let mut gone = metrics.start_request(&request(true, None), created);
        gone.observe(Some(&ResponseItem::Frame(chunk(1, None))));
        gone.call_dropped();
        let queued = metrics.start_request(&request(true, None), created);
        queued.call_dropped();

        let text = rendered(&metrics, &dir);
        std::fs::remove_dir_all(&dir).unwrap();
        // Requests and tokens by label set (labels render sorted).
        assert_eq!(
            sample(
                &text,
                r#"sglang:num_requests_total{engine_type="unified",is_streaming="true",model_name="m",priority="3"}"#
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &text,
                r#"sglang:generation_tokens_total{engine_type="unified",is_streaming="true",model_name="m",priority="3"}"#
            ),
            Some(4.0)
        );
        assert_eq!(
            sample(
                &text,
                r#"sglang:prompt_tokens_total{engine_type="unified",is_streaming="false",model_name="m",priority=""}"#
            ),
            Some(7.0)
        );
        assert_eq!(
            sample(
                &text,
                r#"sglang:num_aborted_requests_total{engine_type="unified",model_name="m",priority=""}"#
            ),
            Some(2.0)
        );
        // The disconnect after one output finishes as an abort with what it
        // got (no TPOT); the one before any output only counts the abort.
        let gone_labels =
            r#"{engine_type="unified",is_streaming="true",model_name="m",priority=""}"#;
        for (family, expected) in [
            ("sglang:num_requests_total", 1.0),
            ("sglang:generation_tokens_total", 1.0),
            ("sglang:e2e_request_latency_seconds_count", 1.0),
            ("sglang:request_time_per_output_token_seconds_count", 0.0),
        ] {
            assert_eq!(
                sample(&text, &format!("{family}{gone_labels}")),
                Some(expected),
                "{family}"
            );
        }
        assert_eq!(
            sample(
                &text,
                r#"sglang:cached_tokens_total{cache_source="device",engine_type="unified",model_name="m",priority=""}"#
            ),
            Some(3.0)
        );
        // The unary request's 7 - 3, and the dropped one's 7.
        assert_eq!(
            sample(
                &text,
                r#"sglang:uncached_prompt_tokens_histogram_sum{engine_type="unified",model_name="m",priority=""}"#
            ),
            Some(11.0)
        );
        // One TTFT per label set. The unary one ends the scheduler's 20 ms
        // decode before its terminal output, its TPOT spreads them over 4 tokens.
        for labels in [
            r#"is_streaming="true",model_name="m",priority="3""#,
            r#"is_streaming="true",model_name="m",priority="""#,
            r#"is_streaming="false",model_name="m",priority="""#,
        ] {
            let prefix = format!(
                r#"sglang:time_to_first_token_seconds_count{{engine_type="unified",{labels}}}"#
            );
            assert_eq!(sample(&text, &prefix), Some(1.0), "{labels}");
        }
        let unary_labels =
            r#"{engine_type="unified",is_streaming="false",model_name="m",priority=""}"#;
        let ttft = sample(
            &text,
            &format!("sglang:time_to_first_token_seconds_sum{unary_labels}"),
        )
        .unwrap();
        let e2e = sample(
            &text,
            &format!("sglang:e2e_request_latency_seconds_sum{unary_labels}"),
        )
        .unwrap();
        assert!((e2e - ttft - 0.02).abs() < 1e-3, "ttft {ttft} e2e {e2e}");
        let tpot = sample(
            &text,
            &format!("sglang:request_time_per_output_token_seconds_sum{unary_labels}"),
        )
        .unwrap();
        assert!((tpot - 0.005).abs() < 1e-6, "{tpot}");
        // Three new tokens after the first: two frames, three observations.
        assert_eq!(
            sample(
                &text,
                r#"sglang:inter_token_latency_seconds_count{engine_type="unified",model_name="m",priority="3"}"#
            ),
            Some(3.0)
        );
        assert_eq!(
            sample(
                &text,
                r#"sglang:e2e_request_latency_seconds_count{engine_type="unified",is_streaming="false",model_name="m",priority=""}"#
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &text,
                r#"sglang:request_time_per_output_token_seconds_count{engine_type="unified",is_streaming="true",model_name="m",priority="3"}"#
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &text,
                r#"sglang:generation_tokens_histogram_bucket{engine_type="unified",le="100.0",model_name="m",priority=""}"#
            ),
            Some(2.0)
        );
    }

    /// Each request of the fixture's cache-tier frames
    /// (`testdata/frame_metadata.json`) adds its cached tokens under the
    /// `cache_source` labels Python's collector increments for it, and only
    /// those: a source is labelled once it has a hit.
    #[test]
    fn cached_tokens_split_by_source_matches_python() {
        use crate::message::response::{for_each_chunk, frame_decode_batch_cols};

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../testdata/frame_metadata.json")).unwrap();
        let bytes = |hex: &serde_json::Value| -> Vec<u8> {
            let hex = hex.as_str().unwrap();
            (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect()
        };
        for name in ["host_tier", "storage_tier"] {
            let batch = &fixture[name];
            let framed =
                frame_decode_batch_cols(&bytes(&batch["header"]), &[&bytes(&batch["data"])]);
            let mut events = Vec::new();
            assert!(for_each_chunk(&framed[1..], |event| events.push(event)).ok);
            for event in events {
                let rid = event.rid.as_str().to_owned();
                // Never flushed, so the directory is not written.
                let config = config(&std::env::temp_dir(), BTreeMap::new());
                let metrics =
                    Arc::new(FrontendMetrics::new(&config, DisaggregationMode::Null).unwrap());
                let mut observed = metrics.start_request(&request(false, None), Instant::now());
                observed.observe(Some(&ResponseItem::Done(event)));
                let counted: serde_json::Map<String, serde_json::Value> = metrics
                    .cached_tokens
                    .series
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|(labels, counter)| {
                        let (_, source) = labels.iter().find(|(k, _)| k == "cache_source").unwrap();
                        (source.clone(), counter.0.load(Ordering::Relaxed).into())
                    })
                    .collect();
                assert_eq!(
                    serde_json::Value::Object(counted),
                    batch["cached_tokens_total"][&rid],
                    "{name} {rid}"
                );
            }
        }
    }

    /// An abort finish counts the request but no first-token time; a request
    /// the scheduler ended outside the output stream counts nothing, as
    /// Python's `_handle_abort_req`; a PD prefill server never records
    /// first-token or inter-token latency.
    #[test]
    fn aborts_and_prefill_skip_latencies() {
        let dir = temp_dir("prefill");
        let labels = BTreeMap::from([("model_name".to_string(), "m".to_string())]);
        let metrics = Arc::new(
            FrontendMetrics::new(&config(&dir, labels.clone()), DisaggregationMode::Null).unwrap(),
        );
        let abort = Some(FinishReason::Known(FinishKind::Abort(Box::new(
            AbortReason {
                message: None,
                status_code: None,
                err_type: None,
            },
        ))));
        let mut aborted = metrics.start_request(&request(true, None), Instant::now());
        aborted.observe(Some(&ResponseItem::Done(chunk(0, abort.clone()))));
        aborted.call_dropped();
        let mut rejected = metrics.start_request(&request(false, None), Instant::now());
        let outside_stream = ChunkExtras {
            abort_output: true,
            ..Default::default()
        };
        rejected.observe(Some(&ResponseItem::Done(terminal(
            0,
            abort,
            outside_stream,
        ))));
        rejected.call_dropped();
        let text = rendered(&metrics, &dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(sample(&text, "sglang:num_requests_total{"), Some(1.0));
        assert_eq!(
            sample(&text, "sglang:time_to_first_token_seconds_count{"),
            Some(0.0)
        );
        assert_eq!(sample(&text, "sglang:num_aborted_requests_total"), None);
        for family in [
            "sglang:num_requests_total",
            "sglang:e2e_request_latency_seconds_count",
            "sglang:time_to_first_token_seconds_count",
        ] {
            let prefix = format!(r#"{family}{{is_streaming="false""#);
            assert_eq!(sample(&text, &prefix), Some(0.0), "{text}");
        }

        let dir = temp_dir("prefill2");
        let metrics = Arc::new(
            FrontendMetrics::new(&config(&dir, labels), DisaggregationMode::Prefill).unwrap(),
        );
        let mut prefill = metrics.start_request(&request(true, None), Instant::now());
        prefill.observe(Some(&ResponseItem::Frame(chunk(1, None))));
        prefill.observe(Some(&ResponseItem::Done(chunk(1, stop()))));
        prefill.call_dropped();
        let text = rendered(&metrics, &dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(sample(&text, "sglang:num_requests_total{"), Some(1.0));
        for family in [
            "sglang:time_to_first_token_seconds",
            "sglang:inter_token_latency_seconds",
            "sglang:request_time_per_output_token_seconds",
        ] {
            assert!(!text.contains(family), "{text}");
        }
    }

    /// Configured buckets replace the defaults; unsorted ones fail at startup,
    /// as prometheus_client's `Histogram` does.
    #[test]
    fn configured_buckets() {
        let dir = temp_dir("buckets");
        let mut config = config(&dir, BTreeMap::new());
        config.bucket_inter_token_latency = Some(vec![0.001, 0.5]);
        let metrics = FrontendMetrics::new(&config, DisaggregationMode::Null).unwrap();
        assert_eq!(
            &metrics.inter_token_latency.bounds[..],
            [0.001, 0.5, f64::INFINITY]
        );
        assert_eq!(
            &metrics.time_per_output_token.bounds[..],
            [0.001, 0.5, f64::INFINITY]
        );
        config.bucket_time_to_first_token = Some(vec![2.0, 1.0]);
        assert!(FrontendMetrics::new(&config, DisaggregationMode::Null).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
