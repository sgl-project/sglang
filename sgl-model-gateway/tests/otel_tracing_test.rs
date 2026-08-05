mod common;

use std::{
    convert::Infallible,
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use axum::{body::Body, extract::Request, http::StatusCode, response::Response as AxumResponse};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use opentelemetry::{global, trace::TracerProvider as _};
use opentelemetry_proto::tonic::collector::trace::v1::{
    trace_service_server::{TraceService, TraceServiceServer},
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use opentelemetry_sdk::{
    export::trace::{ExportResult, SpanData, SpanExporter},
    propagation::TraceContextPropagator,
    trace::{SimpleSpanProcessor, TracerProvider as SdkTracerProvider},
};
use portpicker::pick_unused_port;
use serde_json::json;
use serial_test::serial;
use smg::{
    config::{RouterConfig, TraceConfig},
    core::Job,
    middleware,
    observability::{logging, otel_trace},
    routers::RouterFactory,
};
use tokio::sync::oneshot;
use tonic::metadata::MetadataMap;
use tonic_v12::{transport::Server, Request as TonicRequest, Response as TonicResponse, Status};
use tower::{service_fn, ServiceExt};
use tracing::info_span;
use tracing_subscriber::prelude::*;

#[derive(Clone, Default)]
struct TestSpanExporter {
    spans: Arc<Mutex<Vec<SpanData>>>,
}

impl TestSpanExporter {
    fn get_finished_spans(&self) -> Vec<SpanData> {
        self.spans.lock().expect("spans mutex poisoned").clone()
    }
}

impl fmt::Debug for TestSpanExporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestSpanExporter").finish_non_exhaustive()
    }
}

impl SpanExporter for TestSpanExporter {
    fn export(
        &mut self,
        mut batch: Vec<SpanData>,
    ) -> futures_util::future::BoxFuture<'static, ExportResult> {
        self.spans
            .lock()
            .expect("spans mutex poisoned")
            .append(&mut batch);
        Box::pin(std::future::ready(Ok(())))
    }
}

#[derive(Clone)]
struct TestOtelCollector {
    span_count: Arc<AtomicUsize>,
}

impl TestOtelCollector {
    fn new() -> Self {
        Self {
            span_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn get_span_count(&self) -> usize {
        self.span_count.load(Ordering::SeqCst)
    }
}

#[tonic_v12::async_trait]
impl TraceService for TestOtelCollector {
    async fn export(
        &self,
        request: TonicRequest<ExportTraceServiceRequest>,
    ) -> Result<TonicResponse<ExportTraceServiceResponse>, Status> {
        let req = request.into_inner();

        let mut total_spans = 0;

        for resource_span in &req.resource_spans {
            for scope_span in &resource_span.scope_spans {
                total_spans += scope_span.spans.len();
            }
        }

        self.span_count.fetch_add(total_spans, Ordering::SeqCst);

        Ok(TonicResponse::new(ExportTraceServiceResponse {
            partial_success: None,
        }))
    }
}

async fn start_collector(
    port: u16,
    shutdown_rx: oneshot::Receiver<()>,
) -> Result<TestOtelCollector, Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{}", port).parse()?;
    let collector = TestOtelCollector::new();
    let collector_clone = collector.clone();

    tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(TraceServiceServer::new(collector_clone))
            .serve_with_shutdown(addr, async {
                shutdown_rx.await.ok();
            })
            .await;
    });

    tokio::time::sleep(Duration::from_millis(200)).await;

    Ok(collector)
}

#[tokio::test(flavor = "current_thread")]
async fn test_000_http_request_span_extracts_inbound_traceparent() {
    const INBOUND_TRACE_ID: &str = "3dac15ca145787138ee68247fda9258b";
    const INBOUND_PARENT_SPAN_ID: &str = "892bf5320fe1ac06";

    global::set_text_map_propagator(TraceContextPropagator::new());

    let exporter = TestSpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter.clone())))
        .build();
    let tracer = provider.tracer("smg-test");
    let subscriber =
        tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));
    let _subscriber_guard = tracing::subscriber::set_default(subscriber);

    let app = tower::ServiceBuilder::new()
        .layer(middleware::create_logging_layer())
        .service(service_fn(|_request: Request<Body>| async {
            Ok::<AxumResponse, Infallible>(AxumResponse::new(Body::empty()))
        }));

    let traceparent = format!("00-{}-{}-01", INBOUND_TRACE_ID, INBOUND_PARENT_SPAN_ID);
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("traceparent", traceparent)
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK, "Request should succeed");
    drop(response);

    let spans = exporter.get_finished_spans();
    let http_span = spans
        .iter()
        .find(|span| span.name == "http_request")
        .expect("expected exported http_request span");

    assert_eq!(
        http_span.span_context.trace_id().to_string(),
        INBOUND_TRACE_ID
    );
    assert_eq!(http_span.parent_span_id.to_string(), INBOUND_PARENT_SPAN_ID);
}

#[tokio::test]
#[serial]
async fn test_router_with_tracing() {
    // 1. Start the OTLP collector
    let port = pick_unused_port().expect("Failed to pick unused port");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let collector = start_collector(port, shutdown_rx)
        .await
        .expect("Failed to start collector");
    let collector_endpoint = format!("0.0.0.0:{}", port);
    println!("OTLP Collector started on: {}", collector_endpoint);

    // 2. create the mock worker
    let mut mock_worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });

    let worker_url = mock_worker.start().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    println!("Mock worker started on: {}", worker_url);

    // 3. create router config and enable tracing
    let router_config = RouterConfig::builder()
        .regular_mode(vec![worker_url.clone()])
        .random_policy()
        .host("0.0.0.0")
        .port(0)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .enable_trace(&collector_endpoint)
        .build_unchecked();

    // 4. Initialize the OTLP client (check if already initialized by another test)
    let otel_initialized_by_this_test = if !otel_trace::is_otel_enabled() {
        let init_result = otel_trace::otel_tracing_init(true, Some(&collector_endpoint));
        assert!(
            init_result.is_ok(),
            "Failed to initialize OTEL: {:?}",
            init_result.err()
        );
        println!("OpenTelemetry initialized successfully");
        true
    } else {
        println!(
            "OpenTelemetry already initialized by previous test (spans will go to that collector)"
        );
        false
    };

    let trace_config = TraceConfig {
        enable_trace: true,
        otlp_traces_endpoint: collector_endpoint.clone(),
    };
    let _log_guard = logging::init_logging(
        logging::LoggingConfig {
            level: tracing::Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: false,
            log_file_name: "test-otel".to_string(),
            log_targets: Some(vec!["smg".to_string()]),
        },
        Some(trace_config),
    );
    println!("Logging initialized with OTEL layer");

    // 5. Create a span and sleep for a while
    let _span = info_span!(target: "smg::otel-trace", "test_router_with_tracing");
    tokio::time::sleep(Duration::from_secs(1)).await;
    drop(_span);

    // 6. create app context and router
    let app_context = common::create_test_context(router_config.clone()).await;

    // 7. initialize worker
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");

    let job = Job::InitializeWorkersFromConfig {
        router_config: Box::new(router_config.clone()),
    };

    job_queue
        .submit(job)
        .await
        .expect("Failed to submit worker init job");

    // 8. wait for worker initialization
    tokio::time::sleep(Duration::from_millis(1000)).await;
    println!("Workers initialized");

    // 9. create router
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("Failed to create router");

    println!("Router created");

    // 10. create app (middleware::create_logging_layer() will use the already initialized OTEL layer)
    let app =
        common::test_app::create_test_app_with_context(Arc::from(router), app_context.clone());

    println!("App created with logging middleware");

    // 10. send request
    let request_body = json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello, test OpenTelemetry tracing!"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(request_body.to_string()))
        .unwrap();

    println!("Sending request to router...");
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK, "Request should succeed");

    println!("Request completed successfully");
    drop(response);

    // 11. Wait for spans to be exported
    match otel_trace::flush_spans_async().await {
        Ok(_) => println!("Spans flushed successfully"),
        Err(e) => println!("Failed to flush spans: {:?}", e),
    }

    // 12. Verify that the spans were exported to the collector
    let span_count = collector.get_span_count();
    println!("Total spans received by collector: {}", span_count);

    // Only assert span count if we initialized OTEL with our own collector
    // When OTEL was pre-initialized by another test, spans go to that collector instead
    if otel_initialized_by_this_test {
        assert!(
            span_count == 2,
            "Expected to receive at least 2 span, but got {}. \
            This indicates that tracing data is not being exported to the OTLP collector.",
            span_count
        );
        println!("Test passed! Collector received {} spans", span_count);
    } else {
        println!(
            "Skipping span count assertion - OTEL was pre-initialized by another test. \
            Spans went to that collector. Received {} spans on this test's collector.",
            span_count
        );
    }

    // 13. cleanup
    let _ = shutdown_tx.send(());
    mock_worker.stop().await;

    println!("Cleanup completed");
}

// ============================================================================
// gRPC Trace Context Injection Tests
// ============================================================================

/// Comprehensive test for gRPC trace context injection.
///
/// This test validates:
/// 1. W3C trace context headers are properly injected into gRPC metadata
/// 2. traceparent format is correct (version-traceid-spanid-flags)
/// 3. All metadata keys are lowercase (gRPC requirement)
///
/// Note: This test handles the case where OTEL may already be initialized
/// by a previous test (since tests run sequentially with #[serial]).
#[tokio::test]
#[serial]
async fn test_grpc_trace_context_injection() {
    // 1. Start the OTLP collector (needed even if OTEL is already initialized,
    //    as a target for any spans that might be exported)
    let port = pick_unused_port().expect("Failed to pick unused port");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let _collector = start_collector(port, shutdown_rx)
        .await
        .expect("Failed to start collector");
    let collector_endpoint = format!("0.0.0.0:{}", port);

    // 2. Initialize OTEL if not already enabled
    // Note: otel_tracing_init will fail if already initialized (OnceLock),
    // but that's fine - we just need OTEL to be enabled
    let already_enabled = otel_trace::is_otel_enabled();
    if !already_enabled {
        let init_result = otel_trace::otel_tracing_init(true, Some(&collector_endpoint));
        assert!(
            init_result.is_ok(),
            "Failed to initialize OTEL: {:?}",
            init_result.err()
        );
    }

    // Verify OTEL is enabled (either from this test or a previous one)
    assert!(otel_trace::is_otel_enabled(), "OTEL should be enabled");

    // 3. Set up tracing subscriber with OTEL layer
    let otel_layer = otel_trace::get_otel_layer().expect("Failed to get OTEL layer");
    let subscriber = tracing_subscriber::registry().with(otel_layer);

    // 4. Test within a span context
    tracing::subscriber::with_default(subscriber, || {
        // Create a span that will be exported to OTEL
        let span = info_span!(target: "smg::otel-trace", "test_grpc_span");
        let _guard = span.enter();

        // Create empty gRPC metadata
        let mut metadata = MetadataMap::new();

        // Inject trace context
        otel_trace::inject_trace_context_grpc(&mut metadata);

        // === Test 1: Verify traceparent header was injected ===
        let traceparent = metadata.get("traceparent");
        assert!(
            traceparent.is_some(),
            "traceparent header should be present in gRPC metadata"
        );

        // === Test 2: Verify traceparent format (version-traceid-spanid-flags) ===
        let traceparent_value = traceparent.unwrap().to_str().unwrap();
        let parts: Vec<&str> = traceparent_value.split('-').collect();
        assert_eq!(
            parts.len(),
            4,
            "traceparent should have 4 parts: version-traceid-spanid-flags"
        );
        assert_eq!(parts[0], "00", "traceparent version should be 00");
        assert_eq!(parts[1].len(), 32, "trace ID should be 32 hex characters");
        assert_eq!(parts[2].len(), 16, "span ID should be 16 hex characters");

        println!("Successfully injected traceparent: {}", traceparent_value);

        // === Test 3: Verify all keys are lowercase (gRPC metadata requirement) ===
        for key_and_value in metadata.iter() {
            match key_and_value {
                tonic::metadata::KeyAndValueRef::Ascii(key, _) => {
                    let key_str = key.as_str();
                    assert_eq!(
                        key_str,
                        key_str.to_lowercase(),
                        "gRPC metadata key '{}' should be lowercase",
                        key_str
                    );
                }
                tonic::metadata::KeyAndValueRef::Binary(key, _) => {
                    let key_str = key.as_str();
                    assert_eq!(
                        key_str,
                        key_str.to_lowercase(),
                        "gRPC metadata key '{}' should be lowercase",
                        key_str
                    );
                }
            }
        }

        println!("All gRPC metadata keys are lowercase as required");
    });

    // Cleanup - don't shutdown OTEL since tests share global state (OnceLock)
    // and other tests may need to use the already-initialized OTEL
    let _ = shutdown_tx.send(());

    println!("test_grpc_trace_context_injection: All assertions passed!");
}
