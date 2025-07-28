use actix_web::{rt::System, test as actix_test, web, App};
use clap::{Arg, Command};
use futures::future::join_all;
use serde_json;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Import common benchmark utilities
mod common;
use common::LoadTestConfig as MetricsConfig;
use common::{create_test_payload, parse_endpoint, LoadTestMetrics};

#[path = "../tests/common/mod.rs"]
mod test_common;
use test_common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};

// Import router components
use sglang_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use sglang_router_rs::server::{
    add_worker, generate, v1_chat_completions, v1_completions, AppState,
};

#[derive(Clone)]
struct LoadTestConfig {
    requests: usize,
    workers: usize,
    batch_size: usize,
    worker_delay_ms: u64,
    #[allow(dead_code)]
    parse_responses: bool,
    router_port: u16,
    streaming: bool,
    endpoint: String,
    routing_mode: String,
    policy: String,
    prefill_workers: usize,
    decode_workers: usize,
}

fn display_progress(current: usize, total: usize, elapsed: Duration, failed: usize) {
    let progress = (current as f64 / total as f64) * 100.0;
    let bar_width = 50;
    let filled = (bar_width as f64 * progress / 100.0) as usize;
    let empty = bar_width - filled;

    let throughput = if elapsed.as_secs_f64() > 0.0 {
        current as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    print!(
        "\r[{}{}] {:.1}% | {} / {} | {:.0} req/s | {} failed",
        "█".repeat(filled),
        "░".repeat(empty),
        progress,
        current,
        total,
        throughput,
        failed
    );
    io::stdout().flush().unwrap();
}

async fn run_load_test(config: LoadTestConfig) {
    println!("=== SGLang Router Load Test ===");
    println!("Configuration:");
    println!("  Requests: {}", config.requests);
    println!("  Workers: {}", config.workers);
    println!("  Batch size: {}", config.batch_size);
    println!("  Worker delay: {}ms", config.worker_delay_ms);
    println!(
        "  Mode: {}",
        if config.streaming {
            "Streaming"
        } else {
            "Non-streaming"
        }
    );
    println!("  Endpoint: {}", config.endpoint);
    println!("  Routing mode: {}", config.routing_mode);
    println!("  Policy: {}", config.policy);
    if config.routing_mode == "pd" {
        println!("  Prefill workers: {}", config.prefill_workers);
        println!("  Decode workers: {}", config.decode_workers);
    }

    // Warn about large batch sizes and adjust if needed
    let effective_batch_size = if config.batch_size > 500 {
        println!(
            "\n⚠️  Warning: Large batch size ({}) may cause connection issues.",
            config.batch_size
        );
        println!("   Limiting concurrent connections to 500 for stability.");
        println!("   The test will run in multiple sub-batches.");
        500
    } else {
        config.batch_size
    };
    println!();

    // Start mock workers
    let total_workers = if config.routing_mode == "pd" {
        config.prefill_workers + config.decode_workers
    } else {
        config.workers
    };

    println!("Starting {} mock workers...", total_workers);
    let mut workers = Vec::new();
    let mut worker_urls = Vec::new();

    for i in 0..total_workers {
        let worker_type = if config.routing_mode == "pd" {
            if i < config.prefill_workers {
                WorkerType::Prefill
            } else {
                WorkerType::Decode
            }
        } else {
            WorkerType::Regular
        };

        let mut worker = MockWorker::new(MockWorkerConfig {
            port: 30000 + i as u16,
            worker_type: worker_type.clone(),
            health_status: HealthStatus::Healthy,
            response_delay_ms: config.worker_delay_ms,
            fail_rate: 0.0,
        });

        match worker.start().await {
            Ok(url) => {
                println!("  Worker {} ({:?}) started at {}", i + 1, worker_type, url);
                worker_urls.push(url);
                workers.push(worker);
            }
            Err(e) => {
                eprintln!("Failed to start worker {}: {}", i + 1, e);
                // Cleanup already started workers
                for w in &mut workers {
                    w.stop().await;
                }
                return;
            }
        }
    }

    tokio::time::sleep(Duration::from_millis(200)).await;

    let routing_mode = match config.routing_mode.as_str() {
        "pd" => {
            let prefill_urls: Vec<(String, Option<u16>)> = worker_urls[..config.prefill_workers]
                .iter()
                .map(|url| (url.clone(), None))
                .collect();
            let decode_urls: Vec<String> = worker_urls[config.prefill_workers..]
                .iter()
                .cloned()
                .collect();

            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
            }
        }
        _ => RoutingMode::Regular {
            worker_urls: if config.routing_mode == "regular" {
                vec![] // We'll add via API for regular mode
            } else {
                worker_urls.clone()
            },
        },
    };

    let policy = match config.policy.as_str() {
        "round_robin" => PolicyConfig::RoundRobin,
        "power_of_two" => PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        },
        "cache_aware" => PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 5,
            balance_rel_threshold: 0.2,
            eviction_interval_secs: 300,
            max_tree_size: 10000,
        },
        _ => PolicyConfig::Random,
    };

    let router_config = RouterConfig {
        mode: routing_mode,
        policy,
        host: "127.0.0.1".to_string(),
        port: config.router_port,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 600,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: None,
    };

    println!("\nConfiguring router...");

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(router_config.request_timeout_secs))
        .build()
        .unwrap();

    let app_state = match AppState::new(router_config, client) {
        Ok(state) => web::Data::new(state),
        Err(e) => {
            eprintln!("Failed to create app state: {}", e);
            // Cleanup workers
            for w in &mut workers {
                w.stop().await;
            }
            return;
        }
    };

    // Add workers to router (only for regular mode)
    if config.routing_mode == "regular" {
        // Create app with add_worker service to add mock workers
        let setup_app =
            actix_test::init_service(App::new().app_data(app_state.clone()).service(add_worker))
                .await;

        println!("Adding workers to router...");
        for (i, url) in worker_urls.iter().enumerate() {
            let req = actix_test::TestRequest::post()
                .uri(&format!("/add_worker?url={}", url))
                .to_request();

            let resp = actix_test::call_service(&setup_app, req).await;
            if !resp.status().is_success() {
                eprintln!("Failed to add worker {}: {:?}", i + 1, resp.status());
                // Cleanup
                for w in &mut workers {
                    w.stop().await;
                }
                return;
            }
            println!("  Added worker {}", i + 1);
        }
    } else {
        println!("PD mode: Workers configured at router creation time");
    }

    tokio::time::sleep(Duration::from_millis(500)).await;

    let app = actix_test::init_service(
        App::new()
            .app_data(app_state.clone())
            .service(generate)
            .service(v1_chat_completions)
            .service(v1_completions),
    )
    .await;

    let successful_requests = Arc::new(AtomicUsize::new(0));
    let failed_requests = Arc::new(AtomicUsize::new(0));
    let total_latency_ns = Arc::new(AtomicUsize::new(0));
    let worker_requests: Arc<Vec<AtomicUsize>> =
        Arc::new((0..total_workers).map(|_| AtomicUsize::new(0)).collect());

    // Create a mapping from worker_id to index
    let worker_id_to_index: Arc<std::collections::HashMap<String, usize>> = Arc::new(
        (0..total_workers)
            .map(|i| (format!("worker_{}", 30000 + i), i))
            .collect(),
    );

    println!("Starting load test...");
    let start = Instant::now();

    let mut processed = 0;

    while processed < config.requests {
        let _batch_start = Instant::now();
        let mut futures = Vec::new();
        let batch_size = effective_batch_size.min(config.requests - processed);

        for i in 0..batch_size {
            let req_num = processed + i;
            let app_ref = &app;
            let successful_ref = successful_requests.clone();
            let failed_ref = failed_requests.clone();
            let latency_ref = total_latency_ns.clone();
            let worker_reqs_ref = worker_requests.clone();
            let endpoint = config.endpoint.clone();
            let streaming = config.streaming;
            let worker_id_map = worker_id_to_index.clone();

            let future = async move {
                let req_start = Instant::now();

                let payload = create_test_payload(&endpoint, streaming, req_num);

                let req = actix_test::TestRequest::post()
                    .uri(&endpoint)
                    .set_json(&payload)
                    .to_request();

                let resp = actix_test::call_service(app_ref, req).await;
                let req_duration = req_start.elapsed();

                if resp.status().is_success() {
                    successful_ref.fetch_add(1, Ordering::Relaxed);
                    latency_ref.fetch_add(req_duration.as_nanos() as usize, Ordering::Relaxed);

                    // Parse the response to get worker_id
                    let body = actix_test::read_body(resp).await;
                    if let Ok(json_str) = std::str::from_utf8(&body) {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                            if let Some(worker_id) = json
                                .get("meta_info")
                                .and_then(|meta| meta.get("worker_id"))
                                .and_then(|id| id.as_str())
                            {
                                if let Some(&worker_idx) = worker_id_map.get(worker_id) {
                                    worker_reqs_ref[worker_idx].fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                } else {
                    eprintln!("Request {} failed with status: {}", req_num, resp.status());
                    failed_ref.fetch_add(1, Ordering::Relaxed);
                }
            };

            futures.push(future);
        }

        join_all(futures).await;

        processed += batch_size;
        let current_successful = successful_requests.load(Ordering::Relaxed);
        let current_failed = failed_requests.load(Ordering::Relaxed);
        let total_elapsed = start.elapsed();

        // Update progress bar
        display_progress(
            current_successful + current_failed,
            config.requests,
            total_elapsed,
            current_failed,
        );
    }

    let elapsed = start.elapsed();
    let total_successful = successful_requests.load(Ordering::Relaxed);
    let total_failed = failed_requests.load(Ordering::Relaxed);
    let total_latency = total_latency_ns.load(Ordering::Relaxed);

    println!();

    // Calculate metrics
    let avg_latency_ms = if total_successful > 0 {
        (total_latency as f64 / total_successful as f64) / 1_000_000.0
    } else {
        0.0
    };

    let (p50, p95, p99) = LoadTestMetrics::calculate_percentiles(avg_latency_ms);

    // Collect worker distribution
    let worker_dist: Vec<usize> = worker_requests
        .iter()
        .map(|count| count.load(Ordering::Relaxed))
        .collect();

    // Create metrics struct
    let metrics = LoadTestMetrics {
        total_requests: config.requests,
        successful_requests: total_successful,
        failed_requests: total_failed,
        total_time: elapsed,
        average_latency_ms: avg_latency_ms,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
        p99_latency_ms: p99,
        throughput_rps: total_successful as f64 / elapsed.as_secs_f64(),
        worker_distribution: worker_dist,
        config: MetricsConfig {
            workers: config.workers,
            batch_size: config.batch_size,
            worker_delay_ms: config.worker_delay_ms,
            router_port: config.router_port,
            routing_mode: config.routing_mode.clone(),
            policy: config.policy.clone(),
            prefill_workers: config.prefill_workers,
            decode_workers: config.decode_workers,
        },
    };

    // Print the summary
    metrics.print_summary();

    println!("\n🛑 Shutting down...");
    let cleanup_start = Instant::now();

    // Create a future for stopping all workers
    let stop_futures: Vec<_> = workers
        .iter_mut()
        .enumerate()
        .map(|(i, w)| async move {
            let worker_stop_start = Instant::now();
            match tokio::time::timeout(Duration::from_secs(5), w.stop()).await {
                Ok(_) => {
                    let stop_time = worker_stop_start.elapsed();
                    println!(
                        "  ✓ Worker {} stopped in {:.2}s",
                        i + 1,
                        stop_time.as_secs_f64()
                    );
                    Ok(())
                }
                Err(_) => {
                    eprintln!("  ✗ Worker {} failed to stop within 5s timeout", i + 1);
                    Err(format!("Worker {} timeout", i + 1))
                }
            }
        })
        .collect();

    let stop_results = join_all(stop_futures).await;
    let failed_stops = stop_results.iter().filter(|r| r.is_err()).count();

    let cleanup_elapsed = cleanup_start.elapsed();

    if failed_stops > 0 {
        eprintln!(
            "\n⚠️  Warning: {} workers failed to stop gracefully",
            failed_stops
        );
    } else {
        println!(
            "\n✅ All workers stopped successfully in {:.2}s",
            cleanup_elapsed.as_secs_f64()
        );
    }

    println!("\n🎉 Load test completed!");
}

fn main() {
    let matches = Command::new("SGLang Router Load Test")
        .version("1.0")
        .about("Load testing tool for SGLang router")
        .arg(
            Arg::new("requests")
                .help("Number of requests to send")
                .default_value("1000")
                .index(1),
        )
        .arg(
            Arg::new("workers")
                .help("Number of workers")
                .default_value("4")
                .index(2),
        )
        .arg(
            Arg::new("batch")
                .help("Batch size for concurrent requests")
                .default_value("100")
                .index(3),
        )
        .arg(
            Arg::new("delay")
                .short('d')
                .long("delay")
                .help("Worker response delay in ms")
                .default_value("0"),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .help("Router port")
                .default_value("3011"),
        )
        .arg(
            Arg::new("no-stream")
                .long("no-stream")
                .help("Test non-streaming requests (default: streaming)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("endpoint")
                .short('e')
                .long("endpoint")
                .help("Endpoint to test (generate, chat, completions)")
                .default_value("generate")
                .value_parser(["generate", "chat", "completions"])
                .value_name("ENDPOINT"),
        )
        .arg(
            Arg::new("routing-mode")
                .short('r')
                .long("routing-mode")
                .help("Routing mode (regular, pd)")
                .default_value("regular")
                .value_parser(["regular", "pd"])
                .value_name("MODE"),
        )
        .arg(
            Arg::new("policy")
                .long("policy")
                .help("Load balancing policy (random, round_robin, power_of_two, cache_aware)")
                .default_value("random")
                .value_parser(["random", "round_robin", "power_of_two", "cache_aware"])
                .value_name("POLICY"),
        )
        .arg(
            Arg::new("prefill-workers")
                .long("prefill-workers")
                .help("Number of prefill workers (for PD mode)")
                .default_value("2")
                .value_name("NUM"),
        )
        .arg(
            Arg::new("decode-workers")
                .long("decode-workers")
                .help("Number of decode workers (for PD mode)")
                .default_value("2")
                .value_name("NUM"),
        )
        .get_matches();

    let endpoint_name = matches.get_one::<String>("endpoint").unwrap();
    let endpoint = parse_endpoint(endpoint_name);

    let config = LoadTestConfig {
        requests: matches
            .get_one::<String>("requests")
            .unwrap()
            .parse()
            .unwrap(),
        workers: matches
            .get_one::<String>("workers")
            .unwrap()
            .parse()
            .unwrap(),
        batch_size: matches.get_one::<String>("batch").unwrap().parse().unwrap(),
        worker_delay_ms: matches.get_one::<String>("delay").unwrap().parse().unwrap(),
        parse_responses: false,
        router_port: matches.get_one::<String>("port").unwrap().parse().unwrap(),
        streaming: !matches.get_flag("no-stream"),
        endpoint: endpoint.to_string(),
        routing_mode: matches.get_one::<String>("routing-mode").unwrap().clone(),
        policy: matches.get_one::<String>("policy").unwrap().clone(),
        prefill_workers: matches
            .get_one::<String>("prefill-workers")
            .unwrap()
            .parse()
            .unwrap(),
        decode_workers: matches
            .get_one::<String>("decode-workers")
            .unwrap()
            .parse()
            .unwrap(),
    };

    let system = System::new();
    system.block_on(async {
        run_load_test(config).await;
    });
}
