use std::path::PathBuf;

use tracing::Level;
use tracing_appender::{
    non_blocking::WorkerGuard,
    rolling::{RollingFileAppender, Rotation},
};
use tracing_log::LogTracer;
use tracing_subscriber::{
    fmt::time::ChronoUtc, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

use super::otel_trace::get_otel_layer;
use crate::config::TraceConfig;

#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub level: Level,
    pub json_format: bool,
    pub log_dir: Option<String>,
    pub colorize: bool,
    pub log_file_name: String,
    pub log_targets: Option<Vec<String>>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: true,
            log_file_name: "sgl-model-gateway".to_string(),
            log_targets: Some(vec!["sgl_model_gateway".to_string()]),
        }
    }
}

#[allow(dead_code)]
pub struct LogGuard {
    _file_guard: Option<WorkerGuard>,
}

pub fn init_logging(config: LoggingConfig, otel_layer_config: Option<TraceConfig>) -> LogGuard {
    let _ = LogTracer::init();

    let level_filter = match config.level {
        Level::TRACE => "trace",
        Level::DEBUG => "debug",
        Level::INFO => "info",
        Level::WARN => "warn",
        Level::ERROR => "error",
    };

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let filter_string = if let Some(targets) = &config.log_targets {
            targets
                .iter()
                .enumerate()
                .map(|(i, target)| {
                    if i > 0 {
                        format!(",{}={}", target, level_filter)
                    } else {
                        format!("{}={}", target, level_filter)
                    }
                })
                .collect::<String>()
        } else {
            format!("sgl_model_gateway={}", level_filter)
        };

        EnvFilter::new(filter_string)
    });

    let mut layers = Vec::new();

    let time_format = "%Y-%m-%d %H:%M:%S".to_string();

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(config.colorize)
        .with_file(true)
        .with_line_number(true)
        .with_timer(ChronoUtc::new(time_format.clone()));

    let stdout_layer = if config.json_format {
        stdout_layer.json().flatten_event(true).boxed()
    } else {
        stdout_layer.boxed()
    };

    layers.push(stdout_layer);

    let mut file_guard = None;

    if let Some(log_dir) = &config.log_dir {
        let file_name = config.log_file_name.clone();
        let log_dir = PathBuf::from(log_dir);

        if !log_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                eprintln!("Failed to create log directory: {}", e);
                return LogGuard { _file_guard: None };
            }
        }

        let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, file_name);

        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        file_guard = Some(guard);

        let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_file(true)
            .with_line_number(true)
            .with_timer(ChronoUtc::new(time_format))
            .with_writer(non_blocking);

        let file_layer = if config.json_format {
            file_layer.json().flatten_event(true).boxed()
        } else {
            file_layer.boxed()
        };

        layers.push(file_layer);
    }

    if let Some(otel_layer_config) = &otel_layer_config {
        if otel_layer_config.enable_trace {
            match get_otel_layer() {
                Ok(otel_layer) => {
                    layers.push(otel_layer);
                }
                Err(e) => {
                    eprintln!("Failed to initialize OpenTelemetry: {}", e);
                }
            }
        }
    }

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .try_init();

    LogGuard {
        _file_guard: file_guard,
    }
}
