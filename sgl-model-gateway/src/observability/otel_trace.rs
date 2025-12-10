use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    },
    time::Duration,
};

use anyhow::Result;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    runtime,
    trace::{BatchConfigBuilder, BatchSpanProcessor, Tracer as SdkTracer, TracerProvider},
    Resource,
};
use tokio::task::spawn_blocking;
use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};
use tracing::{Metadata, Subscriber};
use tracing_opentelemetry::{self, OpenTelemetrySpanExt};
use tracing_subscriber::{
    layer::{Context, Filter},
    Layer,
};

use super::events::get_module_path as events_module_path;

static ENABLED: AtomicBool = AtomicBool::new(false);

// Global tracer and provider
static TRACER: OnceLock<SdkTracer> = OnceLock::new();
static PROVIDER: OnceLock<TracerProvider> = OnceLock::new();

/// Targets allowed for OTEL export. Using a static slice avoids allocations.
/// Note: "sgl_model_gateway::otel-trace" is a custom target used for manual spans,
/// not the actual module path.
static ALLOWED_TARGETS: OnceLock<[&'static str; 3]> = OnceLock::new();

fn get_allowed_targets() -> &'static [&'static str; 3] {
    ALLOWED_TARGETS.get_or_init(|| {
        [
            "sgl_model_gateway::otel-trace", // Custom target for manual spans
            "sgl_model_gateway::observability::otel_trace",
            events_module_path(),
        ]
    })
}

/// Filter that only allows specific module targets to be exported to OTEL.
/// This reduces noise and cost by only exporting relevant spans.
#[derive(Clone)]
pub struct CustomOtelFilter;

impl CustomOtelFilter {
    pub fn new() -> Self {
        Self
    }

    #[inline]
    fn is_allowed(target: &str) -> bool {
        get_allowed_targets()
            .iter()
            .any(|allowed| target.starts_with(allowed))
    }
}

impl<S> Filter<S> for CustomOtelFilter
where
    S: Subscriber,
{
    fn enabled(&self, meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        Self::is_allowed(meta.target())
    }

    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> tracing::subscriber::Interest {
        if Self::is_allowed(meta.target()) {
            tracing::subscriber::Interest::always()
        } else {
            tracing::subscriber::Interest::never()
        }
    }
}

impl Default for CustomOtelFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize OpenTelemetry tracing with OTLP exporter.
///
/// # Arguments
/// * `enable` - Whether to enable OTEL tracing
/// * `otlp_endpoint` - OTLP collector endpoint (defaults to "localhost:4317")
pub fn otel_tracing_init(enable: bool, otlp_endpoint: Option<&str>) -> Result<()> {
    if !enable {
        ENABLED.store(false, Ordering::Relaxed);
        return Ok(());
    }

    let endpoint = otlp_endpoint.unwrap_or("localhost:4317");
    let endpoint = if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
        format!("http://{}", endpoint)
    } else {
        endpoint.to_string()
    };

    global::set_text_map_propagator(TraceContextPropagator::new());

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(opentelemetry_otlp::Protocol::Grpc)
        .build()
        .map_err(|e| {
            eprintln!("[tracing] Failed to create OTLP exporter: {}", e);
            anyhow::anyhow!("Failed to create OTLP exporter: {}", e)
        })?;

    let batch_config = BatchConfigBuilder::default()
        .with_scheduled_delay(Duration::from_millis(500))
        .with_max_export_batch_size(64)
        .build();

    let span_processor = BatchSpanProcessor::builder(exporter, runtime::Tokio)
        .with_batch_config(batch_config)
        .build();

    let resource = Resource::default().merge(&Resource::new(vec![KeyValue::new(
        "service.name",
        "sgl-router",
    )]));

    let provider = TracerProvider::builder()
        .with_span_processor(span_processor)
        .with_resource(resource)
        .build();

    PROVIDER
        .set(provider.clone())
        .map_err(|_| anyhow::anyhow!("Provider already initialized"))?;

    let tracer = provider.tracer("sgl-router");

    TRACER
        .set(tracer)
        .map_err(|_| anyhow::anyhow!("Tracer already initialized"))?;

    let _ = global::set_tracer_provider(provider);

    ENABLED.store(true, Ordering::Relaxed);

    eprintln!("[tracing] OpenTelemetry initialized successfully");
    Ok(())
}

/// Get the OpenTelemetry tracing layer to add to the subscriber.
///
/// Must be called after `otel_tracing_init` with `enable=true`.
pub fn get_otel_layer<S>() -> Result<Box<dyn Layer<S> + Send + Sync + 'static>>
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
{
    if !is_otel_enabled() {
        anyhow::bail!("OpenTelemetry is not enabled");
    }

    let tracer = TRACER
        .get()
        .ok_or_else(|| anyhow::anyhow!("Tracer not initialized. Call otel_tracing_init first."))?
        .clone();

    let layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(CustomOtelFilter::new());

    Ok(Box::new(layer))
}

/// Returns whether OpenTelemetry tracing is enabled.
#[inline]
pub fn is_otel_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Flush all pending spans to the OTLP collector.
///
/// This is useful before shutdown or when you need to ensure spans are exported.
pub async fn flush_spans_async() -> Result<()> {
    if !is_otel_enabled() {
        return Ok(());
    }

    let provider = PROVIDER
        .get()
        .ok_or_else(|| anyhow::anyhow!("Provider not initialized"))?
        .clone();

    spawn_blocking(move || provider.force_flush())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to flush spans: {}", e))?;

    Ok(())
}

/// Shutdown OpenTelemetry tracing and flush remaining spans.
pub fn shutdown_otel() {
    if ENABLED.load(Ordering::Relaxed) {
        global::shutdown_tracer_provider();
        ENABLED.store(false, Ordering::Relaxed);
        eprintln!("[tracing] OpenTelemetry shut down");
    }
}

/// Inject W3C trace context headers into an HTTP request.
///
/// This propagates the current span context to downstream services.
/// Does nothing if OTEL is not enabled.
pub fn inject_trace_context_http(headers: &mut HeaderMap) {
    if !is_otel_enabled() {
        return;
    }

    let context = tracing::Span::current().context();

    struct HeaderInjector<'a>(&'a mut HeaderMap);

    impl opentelemetry::propagation::Injector for HeaderInjector<'_> {
        fn set(&mut self, key: &str, value: String) {
            if let Ok(header_name) = HeaderName::from_bytes(key.as_bytes()) {
                if let Ok(header_value) = HeaderValue::from_str(&value) {
                    self.0.insert(header_name, header_value);
                }
            }
        }
    }

    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}

/// Inject W3C trace context into gRPC metadata.
///
/// This propagates the current span context to downstream gRPC services.
/// Does nothing if OTEL is not enabled.
pub fn inject_trace_context_grpc(metadata: &mut MetadataMap) {
    if !is_otel_enabled() {
        return;
    }

    let context = tracing::Span::current().context();

    struct MetadataInjector<'a>(&'a mut MetadataMap);

    impl opentelemetry::propagation::Injector for MetadataInjector<'_> {
        fn set(&mut self, key: &str, value: String) {
            // gRPC metadata keys must be lowercase ASCII
            if let Ok(metadata_key) = MetadataKey::from_bytes(key.to_lowercase().as_bytes()) {
                if let Ok(metadata_value) = MetadataValue::try_from(&value) {
                    self.0.insert(metadata_key, metadata_value);
                }
            }
        }
    }

    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut MetadataInjector(metadata));
    });
}
