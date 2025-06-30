use once_cell::sync::OnceCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::{fmt, io::Write};
use tokio::time::sleep;
use tracing_subscriber;

use sglang_router_rs::core::worker::{
    utils, worker_adapter, Worker, WorkerFactory, WorkerType, DEFAULT_HEALTH_CHECK_CACHE_TTL,
};
use sglang_router_rs::core::WorkerError;

static TRACING_INIT: OnceCell<()> = OnceCell::new();

#[ctor::ctor]
fn init_tracing() {
    TRACING_INIT.get_or_init(|| {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer() // directs logs to test output
            .init();
    });
}

// Mock worker for testing that can simulate different health states
#[derive(Debug, Clone)]
struct MockWorker {
    url: String,
    healthy: Arc<AtomicBool>,
    load_counter: Arc<AtomicUsize>,
    should_fail_health_check: Arc<AtomicBool>,
}

impl MockWorker {
    fn new(url: String, initially_healthy: bool) -> Self {
        Self {
            url,
            healthy: Arc::new(AtomicBool::new(initially_healthy)),
            load_counter: Arc::new(AtomicUsize::new(0)),
            should_fail_health_check: Arc::new(AtomicBool::new(false)),
        }
    }

    fn set_should_fail_health_check(&self, should_fail: bool) {
        self.should_fail_health_check
            .store(should_fail, Ordering::Relaxed);
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Relaxed);
    }
}

impl fmt::Display for MockWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MockWorker({})", self.url)
    }
}

impl Worker for MockWorker {
    fn url(&self) -> &str {
        &self.url
    }

    fn worker_type(&self) -> WorkerType {
        WorkerType::Regular
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    fn check_health(&self) -> futures::future::BoxFuture<'_, Result<(), WorkerError>> {
        let url = self.url.clone();
        let healthy = self.healthy.clone();
        let should_fail = self.should_fail_health_check.clone();

        Box::pin(async move {
            // Simulate some async work
            sleep(Duration::from_millis(10)).await;

            if should_fail.load(Ordering::Relaxed) {
                healthy.store(false, Ordering::Relaxed);
                return Err(WorkerError::HealthCheckFailed {
                    url,
                    reason: "Mock health check failure".to_string(),
                });
            }

            healthy.store(true, Ordering::Relaxed);
            Ok(())
        })
    }

    fn load(&self) -> Arc<AtomicUsize> {
        self.load_counter.clone()
    }

    fn update_health(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Relaxed);
    }
}

#[test]
fn test_regular_worker() {
    let worker = WorkerFactory::create_regular("http://localhost:8080".to_string());
    assert_eq!(worker.url(), "http://localhost:8080");
    assert_eq!(worker.worker_type(), WorkerType::Regular);
    assert!(!worker.is_healthy()); // starts as false to force health check
}

#[test]
fn test_prefill_worker() {
    let worker = WorkerFactory::create_decode("http://localhost:8080".to_string());
    assert_eq!(worker.url(), "http://localhost:8080");
    assert_eq!(worker.worker_type(), WorkerType::Decode);
    assert!(!worker.is_healthy()); // starts as false to force health check
}

#[test]
fn test_decode_worker() {
    let worker = WorkerFactory::create_prefill("http://localhost:8080".to_string(), Some(9000));
    assert_eq!(worker.url(), "http://localhost:8080");
    assert_eq!(worker.worker_type(), WorkerType::Prefill(Some(9000)));
    assert!(!worker.is_healthy()); // starts as false to force health check
}

// Tests for utility functions
#[tokio::test]
async fn test_wait_for_healthy_workers_async_success() {
    let workers: Vec<Arc<dyn Worker>> = vec![
        Arc::new(MockWorker::new("http://worker1:8080".to_string(), true)),
        Arc::new(MockWorker::new("http://worker2:8080".to_string(), true)),
    ];

    let result = utils::wait_for_healthy_workers(&workers, 1, 5).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_wait_for_healthy_workers_async_timeout() {
    let mock_worker = MockWorker::new("http://worker1:8080".to_string(), false);
    mock_worker.set_should_fail_health_check(true);

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(mock_worker)];

    let result = utils::wait_for_healthy_workers(&workers, 1, 2).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Timeout"));
}

#[tokio::test]
async fn test_wait_for_healthy_workers_async_eventually_healthy() {
    let mock_worker = MockWorker::new("http://worker1:8080".to_string(), false);
    mock_worker.set_should_fail_health_check(true);

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(mock_worker.clone())];

    // Start a task that will make the worker healthy after some time
    let mock_worker_clone = mock_worker.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(500)).await;
        mock_worker_clone.set_should_fail_health_check(false);
    });

    let result = utils::wait_for_healthy_workers(&workers, 1, 5).await;
    assert!(result.is_ok());
}

#[test]
fn test_wait_for_healthy_workers_sync_success() {
    let workers: Vec<Arc<dyn Worker>> = vec![
        Arc::new(MockWorker::new("http://worker1:8080".to_string(), true)),
        Arc::new(MockWorker::new("http://worker2:8080".to_string(), true)),
    ];

    let result = utils::wait_for_healthy_workers_sync(&workers, 5, 1);
    assert!(result.is_ok());
}

#[test]
fn test_wait_for_healthy_workers_sync_timeout() {
    let mock_worker = MockWorker::new("http://worker1:8080".to_string(), false);
    mock_worker.set_should_fail_health_check(true);

    let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(mock_worker)];

    let result = utils::wait_for_healthy_workers_sync(&workers, 2, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Timeout"));
}

#[tokio::test]
async fn test_get_worker_load_with_mock_server() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    // Start a mock HTTP server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn server task
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buffer = [0; 1024];
            let _ = stream.read(&mut buffer).await;

            let json_body = r#"{"load": 42}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json_body.len(),
                json_body
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.flush().await;
            // Explicitly shutdown the write side to signal end of response
            let _ = stream.shutdown().await;
        }
    });

    // Give server time to start
    sleep(Duration::from_millis(100)).await;

    let mock_worker = MockWorker::new(format!("http://127.0.0.1:{}", addr.port()), true);
    let worker: Arc<dyn Worker> = Arc::new(mock_worker);
    let client = reqwest::Client::new();

    let load = utils::get_worker_load(&client, &worker).await;
    assert_eq!(load, Some(42));
}

#[tokio::test]
async fn test_get_worker_load_connection_error() {
    // Use a non-existent port to simulate connection error
    let mock_worker = MockWorker::new("http://127.0.0.1:1".to_string(), true);
    let worker: Arc<dyn Worker> = Arc::new(mock_worker);
    let client = reqwest::Client::new();

    let load = utils::get_worker_load(&client, &worker).await;
    assert_eq!(load, None);
}

#[tokio::test]
async fn test_get_worker_load_invalid_json() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    // Start a mock HTTP server that returns invalid JSON
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn server task
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buffer = [0; 1024];
            let _ = stream.read(&mut buffer).await;

            let json_body = r#"{"invalid": json"#; // Invalid JSON
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json_body.len(),
                json_body
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.flush().await;
            let _ = stream.shutdown().await;
        }
    });

    // Give server time to start
    sleep(Duration::from_millis(100)).await;

    let mock_worker = MockWorker::new(format!("http://127.0.0.1:{}", addr.port()), true);
    let worker: Arc<dyn Worker> = Arc::new(mock_worker);
    let client = reqwest::Client::new();

    let load = utils::get_worker_load(&client, &worker).await;
    assert_eq!(load, None);
}

#[tokio::test]
async fn test_get_worker_load_missing_load_field() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    // Start a mock HTTP server that returns JSON without load field
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn server task
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buffer = [0; 1024];
            let _ = stream.read(&mut buffer).await;

            let json_body = r#"{"status": "ok"}"#; // Missing load field
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json_body.len(),
                json_body
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.flush().await;
            let _ = stream.shutdown().await;
        }
    });

    // Give server time to start
    sleep(Duration::from_millis(100)).await;

    let mock_worker = MockWorker::new(format!("http://127.0.0.1:{}", addr.port()), true);
    let worker: Arc<dyn Worker> = Arc::new(mock_worker);
    let client = reqwest::Client::new();

    let load = utils::get_worker_load(&client, &worker).await;
    assert_eq!(load, None);
}

#[tokio::test]
async fn test_get_worker_load_http_error() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    // Start a mock HTTP server that returns 500 error
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn server task
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buffer = [0; 1024];
            let _ = stream.read(&mut buffer).await;

            let response = "HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.flush().await;
            let _ = stream.shutdown().await;
        }
    });

    // Give server time to start
    sleep(Duration::from_millis(100)).await;

    let mock_worker = MockWorker::new(format!("http://127.0.0.1:{}", addr.port()), true);
    let worker: Arc<dyn Worker> = Arc::new(mock_worker);
    let client = reqwest::Client::new();

    let load = utils::get_worker_load(&client, &worker).await;
    assert_eq!(load, None);
}

// Worker implementation tests
#[test]
fn test_regular_worker_creation() {
    let worker = WorkerFactory::create_regular("http://localhost:8080".to_string());
    assert_eq!(worker.url(), "http://localhost:8080");
    assert_eq!(worker.worker_type(), WorkerType::Regular);
    assert!(!worker.is_healthy());
}

#[test]
fn test_decode_worker_creation() {
    let worker = WorkerFactory::create_decode("http://localhost:8080".to_string());
    assert_eq!(worker.url(), "http://localhost:8080");
    assert_eq!(worker.worker_type(), WorkerType::Decode);
    assert!(!worker.is_healthy());
}

#[test]
fn test_prefill_worker_creation_with_bootstrap_port() {
    let worker = WorkerFactory::create_prefill("http://localhost:8080".to_string(), Some(9000));
    assert_eq!(worker.url(), "http://localhost:8080");
    assert_eq!(worker.worker_type(), WorkerType::Prefill(Some(9000)));
    assert!(!worker.is_healthy());
}

#[test]
fn test_prefill_worker_creation_without_bootstrap_port() {
    let worker = WorkerFactory::create_prefill("http://localhost:8080".to_string(), None);
    assert_eq!(worker.url(), "http://localhost:8080");
    assert_eq!(worker.worker_type(), WorkerType::Prefill(None));
    assert!(!worker.is_healthy());
}

#[test]
fn test_worker_update_health() {
    let worker = WorkerFactory::create_regular("http://localhost:8080".to_string());

    // Initially not healthy
    assert!(!worker.is_healthy());

    // Update to healthy
    worker.update_health(true);
    assert!(worker.is_healthy());

    // Update to not healthy
    worker.update_health(false);
    assert!(!worker.is_healthy());
}

#[test]
fn test_worker_load_counter() {
    let worker = WorkerFactory::create_regular("http://localhost:8080".to_string());
    let load_counter = worker.load();

    // Initially should be 0
    assert_eq!(load_counter.load(Ordering::Relaxed), 0);

    // Increment and check
    load_counter.store(5, Ordering::Relaxed);
    assert_eq!(load_counter.load(Ordering::Relaxed), 5);
}

#[test]
fn test_worker_display_formatting() {
    let regular_worker = WorkerFactory::create_regular("http://worker1:8080".to_string());
    let decode_worker = WorkerFactory::create_decode("http://worker2:8080".to_string());
    let prefill_worker =
        WorkerFactory::create_prefill("http://worker3:8080".to_string(), Some(9000));

    let regular_display = format!("{}", regular_worker);
    let decode_display = format!("{}", decode_worker);
    let prefill_display = format!("{}", prefill_worker);

    assert!(regular_display.contains("http://worker1:8080"));
    assert!(decode_display.contains("http://worker2:8080"));
    assert!(prefill_display.contains("http://worker3:8080"));
}

#[test]
fn test_worker_type_display() {
    assert_eq!(format!("{}", WorkerType::Regular), "Regular");
    assert_eq!(format!("{}", WorkerType::Decode), "Decode");
    assert_eq!(
        format!("{}", WorkerType::Prefill(Some(9000))),
        "Prefill(bootstrap_port=9000)"
    );
    assert_eq!(format!("{}", WorkerType::Prefill(None)), "Prefill");
}

#[test]
fn test_worker_type_equality() {
    assert_eq!(WorkerType::Regular, WorkerType::Regular);
    assert_eq!(WorkerType::Decode, WorkerType::Decode);
    assert_eq!(
        WorkerType::Prefill(Some(9000)),
        WorkerType::Prefill(Some(9000))
    );
    assert_eq!(WorkerType::Prefill(None), WorkerType::Prefill(None));

    assert_ne!(WorkerType::Regular, WorkerType::Decode);
    assert_ne!(WorkerType::Prefill(Some(9000)), WorkerType::Prefill(None));
}

#[test]
fn test_worker_endpoints() {
    let regular_endpoints = WorkerType::Regular.get_endpoints();
    let decode_endpoints = WorkerType::Decode.get_endpoints();
    let prefill_endpoints = WorkerType::Prefill(Some(9000)).get_endpoints();

    // All worker types should have the same endpoints
    assert_eq!(regular_endpoints.health, "/health");
    assert_eq!(decode_endpoints.health, "/health");
    assert_eq!(prefill_endpoints.health, "/health");

    assert_eq!(regular_endpoints.load, "/get_load");
    assert_eq!(decode_endpoints.load, "/get_load");
    assert_eq!(prefill_endpoints.load, "/get_load");
}

// Worker adapter tests
#[test]
fn test_from_regular_vec() {
    let urls = vec![
        "http://worker1:8080".to_string(),
        "http://worker2:8080".to_string(),
        "http://worker3:8080".to_string(),
    ];

    let workers = worker_adapter::from_regular_vec(urls.clone());

    assert_eq!(workers.len(), 3);
    for (i, worker) in workers.iter().enumerate() {
        assert_eq!(worker.url(), urls[i]);
        assert_eq!(worker.worker_type(), WorkerType::Regular);
    }
}

#[test]
fn test_from_regular_vec_empty() {
    let workers = worker_adapter::from_regular_vec(vec![]);
    assert!(workers.is_empty());
}

#[test]
fn test_from_decode_vec() {
    let urls = vec![
        "http://decode1:8080".to_string(),
        "http://decode2:8080".to_string(),
    ];

    let workers = worker_adapter::from_decode_vec(urls.clone());

    assert_eq!(workers.len(), 2);
    for (i, worker) in workers.iter().enumerate() {
        assert_eq!(worker.url(), urls[i]);
        assert_eq!(worker.worker_type(), WorkerType::Decode);
    }
}

#[test]
fn test_from_decode_vec_empty() {
    let workers = worker_adapter::from_decode_vec(vec![]);
    assert!(workers.is_empty());
}

#[test]
fn test_from_prefill_vec() {
    let urls = vec![
        ("http://prefill1:8080".to_string(), Some(9001)),
        ("http://prefill2:8080".to_string(), None),
        ("http://prefill3:8080".to_string(), Some(9003)),
    ];

    let workers = worker_adapter::from_prefill_vec(urls.clone());

    assert_eq!(workers.len(), 3);
    for (i, worker) in workers.iter().enumerate() {
        assert_eq!(worker.url(), urls[i].0);
        assert_eq!(worker.worker_type(), WorkerType::Prefill(urls[i].1));
    }
}

#[test]
fn test_from_prefill_vec_empty() {
    let workers = worker_adapter::from_prefill_vec(vec![]);
    assert!(workers.is_empty());
}

#[test]
fn test_mixed_worker_types_in_collection() {
    let regular_workers = worker_adapter::from_regular_vec(vec!["http://regular:8080".to_string()]);
    let decode_workers = worker_adapter::from_decode_vec(vec!["http://decode:8080".to_string()]);
    let prefill_workers =
        worker_adapter::from_prefill_vec(vec![("http://prefill:8080".to_string(), Some(9000))]);

    let mut all_workers: Vec<Arc<dyn Worker>> = Vec::new();
    all_workers.extend(regular_workers);
    all_workers.extend(decode_workers);
    all_workers.extend(prefill_workers);

    assert_eq!(all_workers.len(), 3);
    assert_eq!(all_workers[0].worker_type(), WorkerType::Regular);
    assert_eq!(all_workers[1].worker_type(), WorkerType::Decode);
    assert_eq!(
        all_workers[2].worker_type(),
        WorkerType::Prefill(Some(9000))
    );
}

#[test]
fn test_worker_debug_formatting() {
    let worker = WorkerFactory::create_regular("http://debug-test:8080".to_string());
    let debug_string = format!("{:?}", worker);

    // Debug output should contain the URL
    assert!(debug_string.contains("http://debug-test:8080"));
}
