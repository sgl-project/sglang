//! The benchmark run: set up, dispatch, measure, report.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand::rngs::StdRng;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use tokio::sync::Semaphore;

use crate::args::Args;
use crate::dataset::DatasetRow;
use crate::request::{RequestOutput, RequestTemplate};
use crate::{dataset, hf, metrics, request};

/// aiohttp's benchmark session allows 6 hours for one request; a long
/// generation under heavy load legitimately takes minutes, and a shorter
/// timeout would report the client's impatience as a server failure.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);

/// Run one benchmark to completion and return its result document, the same
/// object the Python client writes to its `.jsonl`.
///
/// Blocking: it owns the Tokio runtime for the run. Callers holding the GIL
/// must release it first, since this neither needs nor touches Python.
pub fn run_blocking(args: Args) -> Result<serde_json::Value> {
    args.validate()?;
    let worker_threads = args.worker_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4)
    });
    anyhow::ensure!(worker_threads > 0, "--worker-threads must be at least 1");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
        .context("cannot start the Tokio runtime")?;
    runtime.block_on(run(args, worker_threads))
}

async fn run(args: Args, worker_threads: usize) -> Result<serde_json::Value> {
    let base_url = args.base_url();
    let request_rate = args.parse_request_rate()?;
    let headers = build_headers(&args)?;
    let http = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        // One connection per in-flight request, so a request never waits on
        // the pool instead of the server. Unbounded concurrency falls back to
        // a generous ceiling.
        .pool_max_idle_per_host(args.max_concurrency.unwrap_or(4096))
        .build()
        .context("cannot build the HTTP client")?;

    if args.ready_check_timeout_sec > 0 {
        wait_for_endpoint(
            &http,
            &format!("{base_url}/v1/models"),
            &headers,
            args.ready_check_timeout_sec,
        )
        .await?;
    }

    let model = resolve_model(&http, &base_url, &headers, &args).await?;
    let model_id = args
        .served_model_name
        .clone()
        .unwrap_or_else(|| model.clone());
    let tokenizer_source = match &args.tokenizer {
        Some(source) => source.clone(),
        None => resolve_tokenizer_source(&http, &base_url, &headers)
            .await
            .unwrap_or_else(|| model.clone()),
    };

    println!("Loading tokenizer from {tokenizer_source}...");
    let tokenizer_path = hf::resolve_tokenizer(&http, &tokenizer_source).await?;
    let tokenizer = hf::load_tokenizer(&tokenizer_path)?;
    let corpus = if dataset::needs_corpus(args.dataset_name) {
        Some(hf::resolve_sharegpt(&http, &args.dataset_path).await?)
    } else {
        None
    };
    let rows = dataset::load(&args, &tokenizer, corpus.as_deref())?;

    let template = RequestTemplate {
        protocol: args.backend.protocol(),
        api_url: args.api_url(),
        model: model_id,
        stream: !args.disable_stream,
        temperature: args.temperature,
        top_p: args.top_p,
        ignore_eos: !args.disable_ignore_eos,
        return_logprob: args.return_logprob,
        top_logprobs_num: args.top_logprobs_num,
        logprob_start_len: args.logprob_start_len,
        extra_body: args.parse_extra_body()?,
        headers,
    };

    // One time base for every request, so the peak-window buckets in `metrics`
    // place all of them on a single timeline.
    let base = Instant::now();
    warmup(&http, &template, &rows, &args, base).await?;
    if should_flush_cache(&args) {
        flush_cache(
            &http,
            &base_url,
            &template.headers,
            args.flush_cache_timeout,
        )
        .await;
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    if args.profile {
        profile(&http, &base_url, &template.headers, "start").await;
    }

    println!(
        "Running {} requests on {worker_threads} worker threads (concurrency: {})...",
        rows.len(),
        args.max_concurrency
            .map_or_else(|| "unbounded".to_owned(), |limit| limit.to_string())
    );
    let started = Instant::now();
    let outputs = dispatch(&http, &template, &rows, &args, request_rate, base).await;
    let duration_s = started.elapsed().as_secs_f64();

    if args.profile {
        profile(&http, &base_url, &template.headers, "stop").await;
    }

    let server_info = if metrics::reports_server_info(args.backend) {
        fetch_json(&http, &format!("{base_url}/server_info"), &template.headers).await
    } else {
        None
    };
    let accept_length = server_info
        .as_ref()
        .and_then(metrics::accept_length_from_server_info);

    let retokenized = retokenize(&tokenizer, &outputs, &args)?;
    let metrics = metrics::calculate(&outputs, &retokenized, duration_s);
    metrics::print_report(&metrics, &args, request_rate, duration_s, accept_length);
    report_failures(&outputs);

    let result = metrics::result_json(
        &metrics,
        &args,
        request_rate,
        duration_s,
        accept_length,
        server_info,
        &outputs,
    );
    write_result(&args, &result)?;
    Ok(result)
}

/// Submit every request, pacing arrivals and capping concurrency.
///
/// Arrivals are paced on this task while the requests themselves run as
/// independent tasks, so a slow response never delays the next arrival. The
/// outputs come back in request order, which the per-request result columns
/// rely on.
async fn dispatch(
    http: &reqwest::Client,
    template: &RequestTemplate,
    rows: &[DatasetRow],
    args: &Args,
    request_rate: f64,
    base: Instant,
) -> Vec<RequestOutput> {
    let permits = args
        .max_concurrency
        .map(|limit| Arc::new(Semaphore::new(limit)));
    let completed = Arc::new(AtomicU64::new(0));
    let progress = progress_bar(args, rows.len(), completed.clone());

    let mut rng = StdRng::seed_from_u64(args.seed ^ 0x5eed);
    let mut handles = Vec::with_capacity(rows.len());
    for row in rows {
        let (http, template, row) = (http.clone(), template.clone(), row.clone());
        let permits = permits.clone();
        let completed = completed.clone();
        handles.push(tokio::spawn(async move {
            // Held for the request's lifetime, so the cap counts requests in
            // flight rather than requests started.
            let _permit = match &permits {
                Some(permits) => Some(permits.acquire().await.expect("semaphore is never closed")),
                None => None,
            };
            let output = request::send(&http, &template, &row, base).await;
            completed.fetch_add(1, Ordering::Relaxed);
            output
        }));

        if request_rate.is_finite() {
            // Poisson arrivals: exponential gaps with mean 1/rate. Sampled
            // from the uniform inverse CDF, since the gap distribution is the
            // only random draw this loop makes.
            let uniform: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
            let interval = -uniform.ln() / request_rate;
            tokio::time::sleep(Duration::from_secs_f64(interval)).await;
        }
    }

    let mut outputs = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(output) => outputs.push(output),
            // A panicked task is this client's bug, not a server failure, but
            // dropping the run would discard every other measurement.
            Err(error) => outputs.push(RequestOutput {
                error: format!("benchmark task failed: {error}"),
                ..Default::default()
            }),
        }
    }
    if let Some(progress) = progress {
        progress.finish_and_clear();
    }
    outputs
}

/// Send a few requests before measuring, so the first measured request does
/// not pay for the server's lazy initialization. Run at the measured
/// concurrency: an unbounded warmup burst is its own, different load.
async fn warmup(
    http: &reqwest::Client,
    template: &RequestTemplate,
    rows: &[DatasetRow],
    args: &Args,
    base: Instant,
) -> Result<()> {
    if args.warmup_requests == 0 {
        return Ok(());
    }
    println!("Starting warmup with {} sequences...", args.warmup_requests);
    let mut row = rows[0].clone();
    row.output_len = row.output_len.min(32);

    let permits = args
        .max_concurrency
        .map(|limit| Arc::new(Semaphore::new(limit)));
    let mut handles = Vec::with_capacity(args.warmup_requests);
    for _ in 0..args.warmup_requests {
        let (http, template, row) = (http.clone(), template.clone(), row.clone());
        let permits = permits.clone();
        handles.push(tokio::spawn(async move {
            let _permit = match &permits {
                Some(permits) => Some(permits.acquire().await.expect("semaphore is never closed")),
                None => None,
            };
            request::send(&http, &template, &row, base).await
        }));
    }

    let mut first_error = String::new();
    let mut any_succeeded = false;
    for handle in handles {
        match handle.await {
            Ok(output) if output.success => any_succeeded = true,
            Ok(output) if first_error.is_empty() => first_error = output.error,
            Ok(_) => {}
            Err(error) if first_error.is_empty() => first_error = error.to_string(),
            Err(_) => {}
        }
    }
    anyhow::ensure!(
        any_succeeded,
        "warmup failed; check the benchmark arguments. Error: {first_error}"
    );
    println!(
        "Warmup completed with {} sequences. Starting main benchmark run...",
        args.warmup_requests
    );
    Ok(())
}

/// Flush the server's prefix cache so the measured run does not reuse the
/// warmup's prefixes. On by default in CI, as in the Python script.
fn should_flush_cache(args: &Args) -> bool {
    if args.flush_cache {
        return true;
    }
    args.backend.is_sglang()
        && std::env::var("SGLANG_IS_IN_CI")
            .map(|value| matches!(value.to_lowercase().as_str(), "true" | "1"))
            .unwrap_or(false)
}

async fn flush_cache(http: &reqwest::Client, base_url: &str, headers: &HeaderMap, timeout_s: f64) {
    let url = format!("{base_url}/flush_cache");
    let result = http
        .post(&url)
        .headers(headers.clone())
        .timeout(Duration::from_secs_f64(timeout_s))
        .send()
        .await;
    match result {
        Ok(response) if response.status().is_success() => println!("Cache flushed."),
        Ok(response) => eprintln!("flush_cache returned {}", response.status()),
        Err(error) => eprintln!("flush_cache failed: {error}"),
    }
}

async fn profile(http: &reqwest::Client, base_url: &str, headers: &HeaderMap, action: &str) {
    println!(
        "{}ing profiler...",
        &action[..action.len() - 1].to_uppercase()
    );
    let url = format!("{base_url}/{action}_profile");
    match http.post(&url).headers(headers.clone()).send().await {
        Ok(response) if response.status().is_success() => println!("Profiler {action}ed"),
        Ok(response) => eprintln!("{url} returned {}", response.status()),
        Err(error) => eprintln!("{url} failed: {error}"),
    }
}

/// Re-tokenize the generated texts, batched so the tokenizer's own
/// parallelism applies. `--disable-retokenize` reports zeros instead.
fn retokenize(
    tokenizer: &tokenizers::Tokenizer,
    outputs: &[RequestOutput],
    args: &Args,
) -> Result<Vec<usize>> {
    if args.disable_retokenize {
        return Ok(vec![0; outputs.len()]);
    }
    let texts: Vec<&str> = outputs
        .iter()
        .map(|output| {
            if output.success {
                output.generated_text.as_str()
            } else {
                ""
            }
        })
        .collect();
    dataset::retokenized_lens(tokenizer, &texts)
}

fn progress_bar(args: &Args, total: usize, completed: Arc<AtomicU64>) -> Option<ProgressBar> {
    if args.disable_tqdm {
        return None;
    }
    let bar = ProgressBar::new(total as u64);
    bar.set_style(
        ProgressStyle::with_template("{wide_bar} {pos}/{len} [{elapsed_precise}<{eta_precise}]")
            .expect("the template is valid"),
    );
    // The bar is driven from one place rather than from every request: at tens
    // of thousands of requests, each task taking the bar's lock to report its
    // own completion is itself measurable.
    let ticker = bar.clone();
    tokio::spawn(async move {
        loop {
            let done = completed.load(Ordering::Relaxed);
            ticker.set_position(done);
            if done >= total as u64 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });
    Some(bar)
}

/// Summarize the failures, so a run that measured far fewer requests than it
/// sent says why on the terminal rather than only in the result file.
fn report_failures(outputs: &[RequestOutput]) {
    let failed: Vec<&RequestOutput> = outputs.iter().filter(|output| !output.success).collect();
    if failed.is_empty() {
        return;
    }
    eprintln!("\n{} of {} requests failed.", failed.len(), outputs.len());
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for output in &failed {
        *counts.entry(output.error.as_str()).or_default() += 1;
    }
    for (error, count) in counts.iter().take(5) {
        let truncated: String = error.chars().take(200).collect();
        eprintln!("  {count} x {truncated}");
    }
}

fn write_result(args: &Args, result: &serde_json::Value) -> Result<()> {
    use std::io::Write;

    let path = match &args.output_file {
        Some(path) => path.clone(),
        None => metrics::default_output_file(args, &month_day()),
    };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("cannot open {path}"))?;
    writeln!(file, "{}", serde_json::to_string(result)?)
        .with_context(|| format!("cannot write {path}"))?;
    println!("Result appended to {path}");
    Ok(())
}

/// `MMDD` in local time, the stamp the Python script puts in the default
/// result filename. Derived from the Unix epoch so the crate needs no date
/// library; the civil-date arithmetic is Howard Hinnant's algorithm.
fn month_day() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let (_, month, day) = civil_from_days(days);
    format!("{month:02}{day:02}")
}

/// Days since 1970-01-01 to a `(year, month, day)` civil date.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (year + i64::from(month <= 2), month, day)
}

/// `OPENAI_API_KEY` as a bearer token, else `API_KEY` verbatim, plus any
/// `--header` values. Same precedence as the Python `get_auth_headers`.
fn build_headers(args: &Args) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    match (std::env::var("OPENAI_API_KEY"), std::env::var("API_KEY")) {
        (Ok(key), _) if !key.is_empty() => {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {key}"))?,
            );
        }
        (_, Ok(key)) if !key.is_empty() => {
            headers.insert(reqwest::header::AUTHORIZATION, HeaderValue::from_str(&key)?);
        }
        _ => {}
    }
    for raw in &args.headers {
        let (name, value) = raw
            .split_once(':')
            .with_context(|| format!("--header must be `Key: Value`, got {raw}"))?;
        headers.insert(
            HeaderName::from_bytes(name.trim().as_bytes())
                .with_context(|| format!("invalid header name in {raw}"))?,
            HeaderValue::from_str(value.trim())
                .with_context(|| format!("invalid header value in {raw}"))?,
        );
    }
    Ok(headers)
}

/// Poll until the server answers, so a benchmark started alongside a server
/// waits for it instead of reporting a wall of connection errors.
async fn wait_for_endpoint(
    http: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    timeout_sec: u64,
) -> Result<()> {
    println!("Waiting up to {timeout_sec}s for {url} to become ready...");
    let started = Instant::now();
    loop {
        let ready = http
            .get(url)
            .headers(headers.clone())
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success());
        if ready {
            println!("Server ready in {:.1}s.", started.elapsed().as_secs_f64());
            return Ok(());
        }
        anyhow::ensure!(
            started.elapsed() < Duration::from_secs(timeout_sec),
            "server at {url} did not become ready within {timeout_sec}s"
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// `--model`, else the first entry the server lists on `/v1/models`.
async fn resolve_model(
    http: &reqwest::Client,
    base_url: &str,
    headers: &HeaderMap,
    args: &Args,
) -> Result<String> {
    if let Some(model) = &args.model {
        return Ok(model.clone());
    }
    let url = format!("{base_url}/v1/models");
    let listed = fetch_json(http, &url, headers).await.and_then(|body| {
        body.get("data")?
            .get(0)?
            .get("id")?
            .as_str()
            .map(str::to_owned)
    });
    listed.with_context(|| {
        format!("no model found at {url}; pass --model, or check --host and --port")
    })
}

/// The tokenizer the server itself loaded, from `/model_info`.
async fn resolve_tokenizer_source(
    http: &reqwest::Client,
    base_url: &str,
    headers: &HeaderMap,
) -> Option<String> {
    let info = fetch_json(http, &format!("{base_url}/model_info"), headers).await?;
    for key in ["tokenizer_path", "model_path"] {
        if let Some(path) = info.get(key).and_then(serde_json::Value::as_str)
            && !path.is_empty()
        {
            return Some(path.to_owned());
        }
    }
    None
}

async fn fetch_json(
    http: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
) -> Option<serde_json::Value> {
    let response = http
        .get(url)
        .headers(headers.clone())
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.json().await.ok()
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    /// The civil-date arithmetic behind the default result filename, checked
    /// on epoch day 0, a leap day, and a year boundary.
    #[test]
    fn civil_dates_round_trip_known_days() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2020-02-29, a leap day.
        assert_eq!(civil_from_days(18_321), (2020, 2, 29));
        // 2024-12-31 and the next day.
        assert_eq!(civil_from_days(20_088), (2024, 12, 31));
        assert_eq!(civil_from_days(20_089), (2025, 1, 1));
    }

    #[test]
    fn headers_combine_auth_and_custom_values() {
        let args = Args::parse_from([
            "sglang-bench",
            "--header",
            "X-Route: a",
            "--header",
            "X-Other: b",
        ]);
        let headers = build_headers(&args).unwrap();
        assert_eq!(headers["x-route"], "a");
        assert_eq!(headers["x-other"], "b");
    }

    /// A malformed `--header` is rejected rather than dropped, since a header
    /// the server never sees would change what the run measured.
    #[test]
    fn a_header_without_a_colon_is_rejected() {
        let args = Args::parse_from(["sglang-bench", "--header", "no-colon"]);
        assert!(build_headers(&args).is_err());
    }

    /// CI flushes the cache for SGLang backends so the measured run does not
    /// inherit the warmup's prefixes; `--flush-cache` forces it everywhere.
    #[test]
    fn cache_flush_follows_the_backend_and_ci() {
        let plain = Args::parse_from(["sglang-bench"]);
        let forced = Args::parse_from(["sglang-bench", "--flush-cache"]);
        let vllm = Args::parse_from(["sglang-bench", "--backend", "vllm"]);
        assert!(should_flush_cache(&forced));
        // SGLANG_IS_IN_CI is not set in this test process.
        assert!(!should_flush_cache(&plain));
        assert!(!should_flush_cache(&vllm));
    }
}
