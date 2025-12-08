use std::{
    collections::HashSet,
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
use tracing::{Metadata, Subscriber};
use tracing_opentelemetry::{self, OpenTelemetrySpanExt};
use tracing_subscriber::{
    layer::{Context, Filter},
    Layer,
};

use super::events::get_module_path as http_router_get_module_path;

static ENABLED: AtomicBool = AtomicBool::new(false);

// global tracer
static TRACER: OnceLock<SdkTracer> = OnceLock::new();
static PROVIDER: OnceLock<TracerProvider> = OnceLock::new();

pub struct CustomOtelFilter {
    allowed_targets: HashSet<String>,
}

impl CustomOtelFilter {
    pub fn new() -> Self {
        let mut allowed_targets = HashSet::new();
        allowed_targets.insert("sgl_model_gateway::otel-trace".to_string());
        allowed_targets.insert(http_router_get_module_path().to_string());

        Self { allowed_targets }
    }
}

impl<S> Filter<S> for CustomOtelFilter
where
    S: Subscriber,
{
    fn enabled(&self, meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        self.allowed_targets.contains(meta.target())
    }

    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> tracing::subscriber::Interest {
        if self.allowed_targets.contains(meta.target()) {
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

/// init OpenTelemetry connection
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

    let result = std::panic::catch_unwind(|| -> Result<()> {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_protocol(opentelemetry_otlp::Protocol::Grpc)
            .build()?;

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

        Ok(())
    });

    match result {
        Ok(Ok(())) => {
            eprintln!(
                "[tracing] OpenTelemetry initialized successfully, enabled: {}",
                ENABLED.load(Ordering::Relaxed)
            );
            Ok(())
        }
        Ok(Err(e)) => {
            eprintln!("[tracing] Failed to initialize OTLP tracer: {}", e);
            ENABLED.store(false, Ordering::Relaxed);
            Err(e)
        }
        Err(_) => {
            eprintln!("[tracing] Panic during OpenTelemetry initialization");
            ENABLED.store(false, Ordering::Relaxed);
            Err(anyhow::anyhow!("Panic during initialization"))
        }
    }
}

pub fn get_otel_layer<S>() -> Result<Box<dyn Layer<S> + Send + Sync + 'static>, &'static str>
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
{
    if !is_otel_enabled() {
        return Err("OpenTelemetry is not enabled");
    }

    let tracer = TRACER
        .get()
        .ok_or("Tracer not initialized. Call otel_tracing_init first.")?
        .clone();

    let custom_filter = CustomOtelFilter::new();

    let layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(custom_filter);

    Ok(Box::new(layer))
}

pub fn is_otel_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub async fn flush_spans_async() -> Result<()> {
    if !is_otel_enabled() {
        return Ok(());
    }

    if let Some(provider) = PROVIDER.get() {
        let provider = provider.clone();

        spawn_blocking(move || provider.force_flush())
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to join blocking task for flushing spans: {}", e)
            })?;

        Ok(())
    } else {
        Err(anyhow::anyhow!("Provider not initialized"))
    }
}

pub fn shutdown_otel() {
    if ENABLED.load(Ordering::Relaxed) {
        global::shutdown_tracer_provider();
        ENABLED.store(false, Ordering::Relaxed);
        eprintln!("[tracing] OpenTelemetry shut down");
    }
}

pub fn inject_trace_context_http(headers: &mut HeaderMap) -> Result<()> {
    if !is_otel_enabled() {
        return Err(anyhow::anyhow!("OTEL not enabled"));
    }

    let context = tracing::Span::current().context();

    struct HeaderInjector<'a>(&'a mut HeaderMap);

    impl<'a> opentelemetry::propagation::Injector for HeaderInjector<'a> {
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
    Ok(())
}
