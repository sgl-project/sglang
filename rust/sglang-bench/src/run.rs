//! The benchmark run: set up, dispatch, measure, report.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand::rngs::StdRng;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use sglang_api_types::api::v1 as api;
use tokio::sync::Semaphore;

use crate::args::{Args, Protocol};
use crate::dataset::DatasetRow;
use crate::request::{GrpcClient, RequestOutput, RequestTemplate};
use crate::{dataset, hf, metrics, request};

/// aiohttp's benchmark session allows 6 hours for one request; a long
/// generation under heavy load legitimately takes minutes, and a shorter
/// timeout would report the client's impatience as a server failure.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);

/// The Rust server's gRPC listener accepts 64 MiB messages; a cumulative
/// `/generate` frame on a long request outgrows tonic's 4 MiB default.
const GRPC_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

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

    // gRPC speaks to the server through one multiplexed HTTP/2 channel; the
    // HTTP backends keep using `http`.
    let grpc = if args.protocol == Protocol::Grpc {
        Some(connect_grpc(&base_url)?)
    } else {
        None
    };
    let server = Server {
        http: &http,
        base_url: &base_url,
        headers: &headers,
        grpc: grpc.clone(),
    };

    if args.ready_check_timeout_sec > 0 {
        server
            .wait_until_ready(args.ready_check_timeout_sec)
            .await?;
    }

    let model_info = server.model_info().await;
    let model = match &args.model {
        Some(model) => model.clone(),
        None => model_info
            .as_ref()
            .and_then(|info| info.served_model_name.clone())
            .with_context(|| {
                format!("no model found at {base_url}; pass --model, or check --host and --port")
            })?,
    };
    let model_id = args
        .served_model_name
        .clone()
        .unwrap_or_else(|| model.clone());
    let tokenizer_source = match &args.tokenizer {
        Some(source) => source.clone(),
        None => model_info
            .as_ref()
            .and_then(|info| info.tokenizer_path.clone())
            .unwrap_or_else(|| model.clone()),
    };

    // `--vocab-size` is the one configuration that reads nothing from the
    // tokenizer, so none is loaded: the client then needs no model files.
    let tokenizer = if args.vocab_size.is_some() {
        println!("Running without a tokenizer (--vocab-size given); retokenized totals report 0.");
        None
    } else {
        println!("Loading tokenizer from {tokenizer_source}...");
        let tokenizer_path = hf::resolve_tokenizer(&http, &tokenizer_source).await?;
        Some(hf::load_tokenizer(&tokenizer_path)?)
    };
    let corpus = if dataset::needs_corpus(args.dataset_name) {
        Some(hf::resolve_sharegpt(&http, &args.dataset_path).await?)
    } else {
        None
    };
    let rows = dataset::load(&args, tokenizer.as_ref(), corpus.as_deref())?;

    let template = RequestTemplate {
        api: args.backend.api(),
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
        headers: headers.clone(),
        grpc,
    };

    // One time base for every request, so the peak-window buckets in `metrics`
    // place all of them on a single timeline.
    let base = Instant::now();
    warmup(&http, &template, &rows, &args, base).await?;
    // Cache flush is an HTTP-only control; a gRPC run skips it loudly rather
    // than measuring a run that silently kept its cache.
    if should_flush_cache(&args) {
        if server.grpc.is_some() {
            eprintln!("flush_cache is not exposed over gRPC; skipping");
        } else {
            flush_cache(
                &http,
                &base_url,
                &template.headers,
                args.flush_cache_timeout,
            )
            .await;
        }
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    // The profiler is an HTTP-only control; a gRPC run skips it loudly.
    if args.profile {
        if server.grpc.is_some() {
            eprintln!("the profiler is not exposed over gRPC; skipping");
        } else {
            profile(&http, &base_url, &template.headers, "start").await;
        }
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

    if args.profile && server.grpc.is_none() {
        profile(&http, &base_url, &template.headers, "stop").await;
    }

    let server_info = if metrics::reports_server_info(args.backend) {
        server.server_info().await
    } else {
        None
    };
    let accept_length = server_info
        .as_ref()
        .and_then(metrics::accept_length_from_server_info);

    let retokenized = retokenize(tokenizer.as_ref(), &outputs, &args)?;
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
    tokenizer: Option<&tokenizers::Tokenizer>,
    outputs: &[RequestOutput],
    args: &Args,
) -> Result<Vec<usize>> {
    let Some(tokenizer) = tokenizer.filter(|_| !args.disable_retokenize) else {
        return Ok(vec![0; outputs.len()]);
    };
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

/// Open the api.v1 channel. One connection carries every request stream,
/// multiplexed by HTTP/2; it connects on first use so a server that is still
/// starting is handled by the readiness poll rather than here.
fn connect_grpc(base_url: &str) -> Result<GrpcClient> {
    let channel = tonic::transport::Endpoint::from_shared(base_url.to_owned())
        .with_context(|| format!("{base_url} is not a valid gRPC endpoint"))?
        .connect_timeout(Duration::from_secs(5))
        .connect_lazy();
    Ok(GrpcClient::new(channel).max_decoding_message_size(GRPC_MAX_MESSAGE_SIZE))
}

/// What the server says about itself, in the shape both transports share.
struct ServerModelInfo {
    served_model_name: Option<String>,
    tokenizer_path: Option<String>,
}

/// The control-plane calls, each over whichever transport the run uses.
struct Server<'a> {
    http: &'a reqwest::Client,
    base_url: &'a str,
    headers: &'a HeaderMap,
    grpc: Option<GrpcClient>,
}

impl Server<'_> {
    /// Poll until the server answers, so a benchmark started alongside a
    /// server waits for it instead of reporting a wall of connection errors.
    async fn wait_until_ready(&self, timeout_sec: u64) -> Result<()> {
        let target = match &self.grpc {
            Some(_) => format!("{} (gRPC GetModelInfo)", self.base_url),
            None => format!("{}/v1/models", self.base_url),
        };
        println!("Waiting up to {timeout_sec}s for {target} to become ready...");
        let started = Instant::now();
        loop {
            if self.is_ready().await {
                println!("Server ready in {:.1}s.", started.elapsed().as_secs_f64());
                return Ok(());
            }
            anyhow::ensure!(
                started.elapsed() < Duration::from_secs(timeout_sec),
                "server at {target} did not become ready within {timeout_sec}s"
            );
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    async fn is_ready(&self) -> bool {
        match &self.grpc {
            Some(client) => client
                .clone()
                .get_model_info(api::GetModelInfoRequest {})
                .await
                .is_ok(),
            None => self
                .http
                .get(format!("{}/v1/models", self.base_url))
                .headers(self.headers.clone())
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success()),
        }
    }

    /// The served model name and the tokenizer the server itself loaded.
    /// HTTP reads `/v1/models` for the name and `/model_info` for the path;
    /// gRPC gets both from `GetModelInfo`.
    async fn model_info(&self) -> Option<ServerModelInfo> {
        if let Some(client) = &self.grpc {
            let info = client
                .clone()
                .get_model_info(api::GetModelInfoRequest {})
                .await
                .ok()?
                .into_inner();
            return Some(ServerModelInfo {
                served_model_name: non_empty(info.served_model_name)
                    .or_else(|| non_empty(info.model_path.clone())),
                tokenizer_path: non_empty(info.tokenizer_path)
                    .or_else(|| non_empty(info.model_path)),
            });
        }
        let served_model_name = fetch_json(
            self.http,
            &format!("{}/v1/models", self.base_url),
            self.headers,
        )
        .await
        .and_then(|body| {
            body.get("data")?
                .get(0)?
                .get("id")?
                .as_str()
                .map(str::to_owned)
        });
        let tokenizer_path = fetch_json(
            self.http,
            &format!("{}/model_info", self.base_url),
            self.headers,
        )
        .await
        .and_then(|info| {
            ["tokenizer_path", "model_path"].iter().find_map(|key| {
                info.get(key)
                    .and_then(serde_json::Value::as_str)
                    .filter(|path| !path.is_empty())
                    .map(str::to_owned)
            })
        });
        Some(ServerModelInfo {
            served_model_name,
            tokenizer_path,
        })
    }

    /// `/server_info`, or its gRPC twin rendered to the same JSON shape so
    /// the accept-length extractor reads both.
    async fn server_info(&self) -> Option<serde_json::Value> {
        match &self.grpc {
            Some(client) => {
                let info = client
                    .clone()
                    .get_server_info(api::GetServerInfoRequest {})
                    .await
                    .ok()?
                    .into_inner();
                serde_json::to_value(info).ok()
            }
            None => {
                fetch_json(
                    self.http,
                    &format!("{}/server_info", self.base_url),
                    self.headers,
                )
                .await
            }
        }
    }
}

fn non_empty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
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
