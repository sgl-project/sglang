//! Metric aggregation and the result report.
//!
//! The statistics follow NumPy, because the Python script's numbers are
//! NumPy's: `percentile` interpolates linearly between the two neighbouring
//! order statistics, and `std` is the population standard deviation (ddof=0).
//! An empty sample reports 0 where the Python script writes `x or 0`, and NaN
//! for the end-to-end latencies, which it does not guard.

use crate::args::{Args, Backend};
use crate::request::RequestOutput;

/// NumPy's `percentile` with the default linear interpolation. `q` is in
/// `[0, 100]`; `values` need not be sorted.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = q / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower as f64)
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation, NumPy's default.
pub fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = mean(values);
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

pub fn max(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0, f64::max)
}

/// `np.mean` on an unguarded empty sample: NaN, as the Python script reports
/// for the end-to-end latencies when every request failed.
fn mean_or_nan(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        mean(values)
    }
}

fn std_or_nan(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        std_dev(values)
    }
}

fn percentile_or_nan(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        percentile(values, q)
    }
}

/// One statistic family, reported as mean / median / std / p90 / p95 / p99.
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Summary {
    /// Milliseconds from a sample of seconds, with 0 for an empty sample.
    fn from_seconds(values: &[f64]) -> Self {
        Self {
            mean: mean(values) * 1e3,
            median: percentile(values, 50.0) * 1e3,
            std: std_dev(values) * 1e3,
            p90: percentile(values, 90.0) * 1e3,
            p95: percentile(values, 95.0) * 1e3,
            p99: percentile(values, 99.0) * 1e3,
        }
    }

    /// The same, for the unguarded end-to-end family.
    fn from_seconds_or_nan(values: &[f64]) -> Self {
        Self {
            mean: mean_or_nan(values) * 1e3,
            median: percentile_or_nan(values, 50.0) * 1e3,
            std: std_or_nan(values) * 1e3,
            p90: percentile_or_nan(values, 90.0) * 1e3,
            p95: percentile_or_nan(values, 95.0) * 1e3,
            p99: percentile_or_nan(values, 99.0) * 1e3,
        }
    }
}

#[derive(Debug)]
pub struct BenchmarkMetrics {
    pub completed: usize,
    pub total_input: usize,
    pub total_input_text: usize,
    pub total_input_vision: usize,
    pub total_output: usize,
    pub total_output_retokenized: usize,

    pub request_throughput: f64,
    pub input_throughput: f64,
    pub output_throughput: f64,
    pub total_throughput: f64,

    pub ttft: Summary,
    pub tpot: Summary,
    pub itl: Summary,
    pub max_itl_ms: f64,
    pub e2e: Summary,

    pub concurrency: f64,
    pub max_output_tokens_per_s: f64,
    pub max_concurrent_requests: usize,
    /// Per-request output lengths in request order, 0 for a failure.
    pub output_lens: Vec<usize>,
}

/// Fold the per-request outputs into the reported metrics. `retokenized_lens`
/// is parallel to `outputs` and already 0 for failures; pass all zeros when
/// re-tokenization is disabled.
pub fn calculate(
    outputs: &[RequestOutput],
    retokenized_lens: &[usize],
    duration_s: f64,
) -> BenchmarkMetrics {
    let mut output_lens = Vec::with_capacity(outputs.len());
    let (mut total_input, mut total_input_text, mut total_input_vision) = (0, 0, 0);
    let mut completed = 0usize;
    let mut itls: Vec<f64> = Vec::new();
    let mut tpots: Vec<f64> = Vec::new();
    let mut ttfts: Vec<f64> = Vec::new();
    let mut e2e_latencies: Vec<f64> = Vec::new();

    for output in outputs {
        if !output.success {
            output_lens.push(0);
            continue;
        }
        output_lens.push(output.output_len);
        total_input += output.dataset_prompt_len;
        total_input_text += output.dataset_text_prompt_len;
        total_input_vision += output.dataset_vision_prompt_len;
        if output.output_len > 1 {
            tpots.push((output.latency - output.ttft) / (output.output_len - 1) as f64);
        }
        itls.extend_from_slice(&output.itl);
        ttfts.push(output.ttft);
        e2e_latencies.push(output.latency);
        completed += 1;
    }

    let total_output: usize = output_lens.iter().sum();
    let total_output_retokenized: usize = retokenized_lens.iter().sum();
    let (max_output_tokens_per_s, max_concurrent_requests) = peak_windows(outputs);

    BenchmarkMetrics {
        completed,
        total_input,
        total_input_text,
        total_input_vision,
        total_output,
        total_output_retokenized,
        request_throughput: completed as f64 / duration_s,
        input_throughput: total_input as f64 / duration_s,
        output_throughput: total_output as f64 / duration_s,
        total_throughput: (total_input + total_output) as f64 / duration_s,
        ttft: Summary::from_seconds(&ttfts),
        tpot: Summary::from_seconds(&tpots),
        itl: Summary::from_seconds(&itls),
        max_itl_ms: max(&itls) * 1e3,
        e2e: Summary::from_seconds_or_nan(&e2e_latencies),
        concurrency: e2e_latencies.iter().sum::<f64>() / duration_s,
        max_output_tokens_per_s,
        max_concurrent_requests,
        output_lens,
    }
}

/// Peak output tokens per second and peak concurrent requests, bucketed into
/// one-second windows from the first request's start. Token times are
/// reconstructed from each request's TTFT plus its inter-token latencies,
/// which is how the Python script derives them.
fn peak_windows(outputs: &[RequestOutput]) -> (f64, usize) {
    let successful = || outputs.iter().filter(|o| o.success);
    let Some(min_start) = successful().map(|o| o.start_time).reduce(f64::min) else {
        return (0.0, 0);
    };
    let max_end = successful()
        .map(|o| o.start_time + o.latency)
        .fold(f64::NEG_INFINITY, f64::max);

    let buckets = (max_end - min_start).ceil() as usize + 1;
    let mut tokens_per_second = vec![0u64; buckets];
    let mut concurrent_per_second = vec![0u64; buckets];

    for output in successful() {
        let mut token_time = output.start_time + output.ttft;
        let mut record = |time: f64| {
            let bucket = (time - min_start) as isize;
            if bucket >= 0 && (bucket as usize) < buckets {
                tokens_per_second[bucket as usize] += 1;
            }
        };
        record(token_time);
        for itl in &output.itl {
            token_time += itl;
            record(token_time);
        }

        let start_second = (output.start_time - min_start) as usize;
        let end_second = ((output.start_time + output.latency) - min_start) as usize;
        let last = (end_second + 1).min(buckets);
        for bucket in &mut concurrent_per_second[start_second.min(last)..last] {
            *bucket += 1;
        }
    }

    (
        tokens_per_second.iter().copied().max().unwrap_or(0) as f64,
        concurrent_per_second.iter().copied().max().unwrap_or(0) as usize,
    )
}

/// The `=== Serving Benchmark Result ===` block, line for line as the Python
/// script prints it: a 40-column label, then the value in 10 columns.
pub fn print_report(
    metrics: &BenchmarkMetrics,
    args: &Args,
    request_rate: f64,
    duration_s: f64,
    accept_length: Option<f64>,
) {
    println!("\n{:=^50}", " Serving Benchmark Result ");
    println!("{:<40} {:<10}", "Backend:", args.backend.as_str());
    let rate = if request_rate.is_infinite() {
        "inf".to_owned()
    } else {
        format!("{request_rate}")
    };
    println!("{:<40} {:<10}", "Traffic request rate:", rate);
    println!(
        "{:<40} {:<10}",
        "Max request concurrency:",
        args.max_concurrency
            .map_or_else(|| "not set".to_owned(), |limit| limit.to_string())
    );
    println!("{:<40} {:<10}", "Successful requests:", metrics.completed);
    println!("{:<40} {:<10.2}", "Benchmark duration (s):", duration_s);
    println!("{:<40} {:<10}", "Total input tokens:", metrics.total_input);
    println!(
        "{:<40} {:<10}",
        "Total input text tokens:", metrics.total_input_text
    );
    println!(
        "{:<40} {:<10}",
        "Total generated tokens:", metrics.total_output
    );
    println!(
        "{:<40} {:<10}",
        "Total generated tokens (retokenized):", metrics.total_output_retokenized
    );
    println!(
        "{:<40} {:<10.2}",
        "Request throughput (req/s):", metrics.request_throughput
    );
    println!(
        "{:<40} {:<10.2}",
        "Input token throughput (tok/s):", metrics.input_throughput
    );
    println!(
        "{:<40} {:<10.2}",
        "Output token throughput (tok/s):", metrics.output_throughput
    );
    println!(
        "{:<40} {:<10.2}",
        "Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s
    );
    println!(
        "{:<40} {:<10}",
        "Peak concurrent requests:", metrics.max_concurrent_requests
    );
    println!(
        "{:<40} {:<10.2}",
        "Total token throughput (tok/s):", metrics.total_throughput
    );
    println!("{:<40} {:<10.2}", "Concurrency:", metrics.concurrency);
    if let Some(accept_length) = accept_length {
        println!("{:<40} {:<10.2}", "Accept length:", accept_length);
    }

    print_family("End-to-End Latency", "E2E Latency", &metrics.e2e);
    print_family("Time to First Token", "TTFT", &metrics.ttft);
    print_family(
        "Time per Output Token (excl. 1st token)",
        "TPOT",
        &metrics.tpot,
    );
    println!("{:-^50}", "Inter-Token Latency");
    println!("{:<40} {:<10.2}", "Mean ITL (ms):", metrics.itl.mean);
    println!("{:<40} {:<10.2}", "Median ITL (ms):", metrics.itl.median);
    println!("{:<40} {:<10.2}", "P90 ITL (ms):", metrics.itl.p90);
    println!("{:<40} {:<10.2}", "P95 ITL (ms):", metrics.itl.p95);
    println!("{:<40} {:<10.2}", "P99 ITL (ms):", metrics.itl.p99);
    println!("{:<40} {:<10.2}", "Max ITL (ms):", metrics.max_itl_ms);
    println!("{}", "=".repeat(50));
}

fn print_family(heading: &str, label: &str, summary: &Summary) {
    println!("{heading:-^50}");
    println!(
        "{:<40} {:<10.2}",
        format!("Mean {label} (ms):"),
        summary.mean
    );
    println!(
        "{:<40} {:<10.2}",
        format!("Median {label} (ms):"),
        summary.median
    );
    println!("{:<40} {:<10.2}", format!("P90 {label} (ms):"), summary.p90);
    println!("{:<40} {:<10.2}", format!("P95 {label} (ms):"), summary.p95);
    println!("{:<40} {:<10.2}", format!("P99 {label} (ms):"), summary.p99);
}

/// The default result filename, as the Python script derives it:
/// `<backend>_<MMDD>_<num_prompts>_<in>_<out>.jsonl` for a random dataset,
/// `<backend>_<MMDD>_<num_prompts>_<dataset>.jsonl` otherwise.
pub fn default_output_file(args: &Args, now: &str) -> String {
    let backend = args.backend.as_str();
    if args.dataset_name.is_random() {
        format!(
            "{backend}_{now}_{}_{}_{}.jsonl",
            args.num_prompts, args.random_input_len, args.random_output_len
        )
    } else {
        format!(
            "{backend}_{now}_{}_{}.jsonl",
            args.num_prompts,
            args.dataset_name.as_str()
        )
    }
}

/// The JSON result line, with the Python script's keys so existing readers of
/// the `.jsonl` keep working.
pub fn result_json(
    metrics: &BenchmarkMetrics,
    args: &Args,
    request_rate: f64,
    duration_s: f64,
    accept_length: Option<f64>,
    server_info: Option<serde_json::Value>,
    outputs: &[RequestOutput],
) -> serde_json::Value {
    let rate = if request_rate.is_infinite() {
        serde_json::Value::String("inf".into())
    } else {
        serde_json::json!(request_rate)
    };
    // Built by insertion rather than one `json!` literal: fifty keys in one
    // macro invocation exceeds the expander's recursion limit.
    let mut object = serde_json::Map::new();
    // Scoped so the borrow ends before the map is moved into the result.
    {
        let mut put = |key: &str, value: serde_json::Value| {
            object.insert(key.to_owned(), value);
        };
        // Arguments.
        put("tag", serde_json::json!(args.tag));
        put("backend", serde_json::json!(args.backend.as_str()));
        put(
            "dataset_name",
            serde_json::json!(args.dataset_name.as_str()),
        );
        put("request_rate", rate);
        put("max_concurrency", serde_json::json!(args.max_concurrency));
        put(
            "sharegpt_output_len",
            serde_json::json!(args.sharegpt_output_len),
        );
        put("random_input_len", serde_json::json!(args.random_input_len));
        put(
            "random_output_len",
            serde_json::json!(args.random_output_len),
        );
        put(
            "random_range_ratio",
            serde_json::json!(args.random_range_ratio),
        );
        // Information.
        put("server_info", serde_json::json!(server_info));
        // Results.
        put("duration", serde_json::json!(duration_s));
        put("completed", serde_json::json!(metrics.completed));
        put("total_input_tokens", serde_json::json!(metrics.total_input));
        put(
            "total_input_text_tokens",
            serde_json::json!(metrics.total_input_text),
        );
        put(
            "total_input_vision_tokens",
            serde_json::json!(metrics.total_input_vision),
        );
        put(
            "total_output_tokens",
            serde_json::json!(metrics.total_output),
        );
        put(
            "total_output_tokens_retokenized",
            serde_json::json!(metrics.total_output_retokenized),
        );
        put(
            "request_throughput",
            serde_json::json!(metrics.request_throughput),
        );
        put(
            "input_throughput",
            serde_json::json!(metrics.input_throughput),
        );
        put(
            "output_throughput",
            serde_json::json!(metrics.output_throughput),
        );
        put(
            "total_throughput",
            serde_json::json!(metrics.total_throughput),
        );
        for (prefix, summary) in [
            ("e2e_latency", &metrics.e2e),
            ("ttft", &metrics.ttft),
            ("tpot", &metrics.tpot),
            ("itl", &metrics.itl),
        ] {
            for (stat, value) in [
                ("mean", summary.mean),
                ("median", summary.median),
                ("std", summary.std),
                ("p90", summary.p90),
                ("p95", summary.p95),
                ("p99", summary.p99),
            ] {
                // NaN has no JSON form; the Python script writes it as a bare
                // `NaN` token, which strict readers reject, so it becomes null.
                let value = if value.is_finite() {
                    serde_json::json!(value)
                } else {
                    serde_json::Value::Null
                };
                put(&format!("{stat}_{prefix}_ms"), value);
            }
        }
        put("concurrency", serde_json::json!(metrics.concurrency));
        put("accept_length", serde_json::json!(accept_length));
        put(
            "max_output_tokens_per_s",
            serde_json::json!(metrics.max_output_tokens_per_s),
        );
        put(
            "max_concurrent_requests",
            serde_json::json!(metrics.max_concurrent_requests),
        );
    }
    let mut result = serde_json::Value::Object(object);

    if args.output_details {
        let details = serde_json::json!({
            "input_lens": outputs.iter().map(|o| o.prompt_len).collect::<Vec<_>>(),
            "output_lens": metrics.output_lens,
            "ttfts": outputs.iter().map(|o| o.ttft).collect::<Vec<_>>(),
            "itls": outputs.iter().map(|o| o.itl.clone()).collect::<Vec<_>>(),
            "generated_texts": outputs.iter().map(|o| o.generated_text.clone()).collect::<Vec<_>>(),
            "errors": outputs.iter().map(|o| o.error.clone()).collect::<Vec<_>>(),
        });
        let (Some(result), Some(details)) = (result.as_object_mut(), details.as_object()) else {
            unreachable!("both are JSON objects");
        };
        for (key, value) in details {
            result.insert(key.clone(), value.clone());
        }
    }
    result
}

/// `avg_spec_accept_length` out of a `/server_info` body, following the PD
/// shape where the decode node's info is nested under `decode[0]`.
pub fn accept_length_from_server_info(info: &serde_json::Value) -> Option<f64> {
    let node = info
        .get("decode")
        .and_then(|decode| decode.get(0))
        .unwrap_or(info);
    node.get("internal_states")?
        .get(0)?
        .get("avg_spec_accept_length")?
        .as_f64()
}

/// Whether this backend reports spec-decoding and cache stats on
/// `/server_info` at all.
pub fn reports_server_info(backend: Backend) -> bool {
    backend.is_sglang()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(ttft: f64, itl: Vec<f64>, latency: f64, output_len: usize) -> RequestOutput {
        RequestOutput {
            success: true,
            ttft,
            itl,
            latency,
            output_len,
            prompt_len: 10,
            dataset_prompt_len: 10,
            dataset_text_prompt_len: 10,
            start_time: 0.0,
            ..Default::default()
        }
    }

    /// NumPy's linear interpolation, pinned against values computed with
    /// `np.percentile`: the Python script's reported percentiles are these.
    #[test]
    fn percentile_matches_numpy_linear_interpolation() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 50.0), 2.5);
        assert_eq!(percentile(&values, 100.0), 4.0);
        // rank = 0.9 * 3 = 2.7 -> 3 + (4-3)*0.7
        assert!((percentile(&values, 90.0) - 3.7).abs() < 1e-12);
        // Unsorted input must give the same answer.
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 50.0), 2.5);
        // A single sample is every percentile.
        assert_eq!(percentile(&[7.0], 99.0), 7.0);
        // `np.percentile(x or 0, q)` on an empty sample is 0.
        assert_eq!(percentile(&[], 99.0), 0.0);
    }

    /// `np.std` defaults to the population deviation (ddof=0), not the sample
    /// one; reporting the sample deviation would inflate every std by
    /// `sqrt(n/(n-1))`.
    #[test]
    fn std_dev_is_the_population_deviation() {
        assert!((std_dev(&[1.0, 2.0, 3.0, 4.0]) - 1.118_033_988_749_895).abs() < 1e-12);
        assert_eq!(std_dev(&[5.0]), 0.0);
        assert_eq!(std_dev(&[]), 0.0);
    }

    /// A request contributes one TPOT sample only when it produced more than
    /// one token, and ITLs pool across requests.
    #[test]
    fn metrics_fold_per_request_samples() {
        let outputs = vec![
            output(0.1, vec![0.01, 0.01], 0.12, 3),
            output(0.2, vec![0.02], 0.22, 2),
            // One token: no TPOT sample, since there is no decode interval.
            output(0.3, vec![], 0.3, 1),
            RequestOutput {
                success: false,
                output_len: 99,
                ..Default::default()
            },
        ];
        let metrics = calculate(&outputs, &[3, 2, 1, 0], 2.0);

        assert_eq!(metrics.completed, 3);
        // A failed request contributes 0, not its requested length.
        assert_eq!(metrics.output_lens, vec![3, 2, 1, 0]);
        assert_eq!(metrics.total_output, 6);
        assert_eq!(metrics.total_input, 30);
        assert_eq!(metrics.request_throughput, 1.5);
        assert_eq!(metrics.output_throughput, 3.0);
        // TPOT from the two multi-token requests only.
        assert!((metrics.tpot.mean - 15.0).abs() < 1e-9);
        // ITLs pooled: 10, 10, 20 ms.
        assert!((metrics.itl.mean - 40.0 / 3.0).abs() < 1e-9);
        assert!((metrics.max_itl_ms - 20.0).abs() < 1e-9);
        assert!((metrics.concurrency - 0.32) < 1e-9);
    }

    /// Every request failing leaves the end-to-end family NaN (as NumPy's
    /// unguarded `mean([])` does) while the guarded families report 0, so the
    /// report distinguishes "no data" from "zero latency".
    #[test]
    fn all_failures_report_nan_e2e_and_zero_elsewhere() {
        let outputs = vec![RequestOutput {
            success: false,
            ..Default::default()
        }];
        let metrics = calculate(&outputs, &[0], 1.0);
        assert_eq!(metrics.completed, 0);
        assert!(metrics.e2e.mean.is_nan());
        assert!(metrics.e2e.p99.is_nan());
        assert_eq!(metrics.ttft.mean, 0.0);
        assert_eq!(metrics.max_itl_ms, 0.0);
        assert_eq!(metrics.max_concurrent_requests, 0);
    }

    /// The peak windows bucket token arrivals by whole second from the first
    /// start, so two requests overlapping in second 0 peak at 2.
    #[test]
    fn peak_windows_bucket_by_second() {
        let outputs = vec![
            RequestOutput {
                success: true,
                start_time: 10.0,
                ttft: 0.1,
                itl: vec![0.1, 0.1],
                latency: 0.3,
                output_len: 3,
                ..Default::default()
            },
            RequestOutput {
                success: true,
                start_time: 10.5,
                ttft: 0.1,
                itl: vec![1.0],
                latency: 1.1,
                output_len: 2,
                ..Default::default()
            },
        ];
        let (peak_tokens, peak_requests) = peak_windows(&outputs);
        // Second 0 holds 3 tokens from the first request plus 1 from the second.
        assert_eq!(peak_tokens, 4.0);
        assert_eq!(peak_requests, 2);
    }

    #[test]
    fn accept_length_reads_plain_and_pd_shapes() {
        let plain = serde_json::json!({"internal_states": [{"avg_spec_accept_length": 2.5}]});
        assert_eq!(accept_length_from_server_info(&plain), Some(2.5));
        let pd = serde_json::json!({
            "decode": [{"internal_states": [{"avg_spec_accept_length": 3.5}]}]
        });
        assert_eq!(accept_length_from_server_info(&pd), Some(3.5));
        assert_eq!(
            accept_length_from_server_info(&serde_json::json!({"internal_states": []})),
            None
        );
        assert_eq!(accept_length_from_server_info(&serde_json::json!({})), None);
    }
}
