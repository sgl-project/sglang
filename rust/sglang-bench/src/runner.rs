//! Load generation: pacing, concurrency limiting, and result collection on a
//! dedicated tokio runtime, so the benchmark never contends with the GIL.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::Semaphore;

use crate::client::{self, RequestSpec, RunConfig};
use crate::output::RequestOutput;

/// 6 hours, matching `_create_bench_client_session` in serving.py.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);

pub struct RunState {
    pub total: usize,
    pub completed: AtomicUsize,
}

pub struct RunHandle {
    pub state: Arc<RunState>,
    pub join: Option<std::thread::JoinHandle<Vec<RequestOutput>>>,
}

/// Spawn the benchmark on a background thread and return immediately. The
/// anchor instant is captured here; the Python wrapper captures
/// `time.perf_counter()` at the same moment to re-base `start_time`.
pub fn start(cfg: RunConfig, specs: Vec<RequestSpec>) -> Result<RunHandle, String> {
    // One pooled client for the whole run. Deliberate difference from the
    // Python path (a fresh aiohttp session — i.e. a fresh connection — per
    // request): connection reuse is the realistic client behavior.
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| format!("failed to build http client: {e}"))?;

    let anchor = Instant::now();
    let state = Arc::new(RunState {
        total: specs.len(),
        completed: AtomicUsize::new(0),
    });
    let task_state = Arc::clone(&state);
    let join = std::thread::Builder::new()
        .name("sglang-bench".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            rt.block_on(run_all(client, cfg, specs, anchor, task_state))
        })
        .map_err(|e| format!("failed to spawn benchmark thread: {e}"))?;
    Ok(RunHandle {
        state,
        join: Some(join),
    })
}

async fn run_all(
    client: reqwest::Client,
    cfg: RunConfig,
    specs: Vec<RequestSpec>,
    anchor: Instant,
    state: Arc<RunState>,
) -> Vec<RequestOutput> {
    let cfg = Arc::new(cfg);
    let semaphore = cfg.max_concurrency.map(|n| Arc::new(Semaphore::new(n)));
    let start = tokio::time::Instant::from_std(anchor);

    let mut tasks = Vec::with_capacity(specs.len());
    for spec in specs {
        let client = client.clone();
        let cfg = Arc::clone(&cfg);
        let semaphore = semaphore.clone();
        let state = Arc::clone(&state);
        tasks.push(tokio::spawn(async move {
            // Arrival offsets are precomputed in Python (exponential
            // inter-arrival), replicating `get_request`: pacing happens
            // regardless of the concurrency limit, which gates only the send.
            let offset =
                Duration::try_from_secs_f64(spec.arrival_offset_s).unwrap_or(Duration::ZERO);
            tokio::time::sleep_until(start + offset).await;
            let _permit = match &semaphore {
                Some(s) => Some(s.acquire().await.expect("semaphore closed")),
                None => None,
            };
            let output = client::run_one(&client, &cfg, spec, anchor).await;
            state.completed.fetch_add(1, Ordering::Relaxed);
            output
        }));
    }

    let mut outputs = Vec::with_capacity(tasks.len());
    for task in tasks {
        outputs.push(task.await.expect("request task panicked"));
    }
    outputs
}
