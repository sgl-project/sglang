use std::collections::HashMap;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use sgl_model_gateway::{
    config::{
        CircuitBreakerConfig, ConfigError, ConfigResult, DiscoveryConfig, HealthCheckConfig,
        HistoryBackend, MetricsConfig, OracleConfig, PolicyConfig, PostgresConfig, RetryConfig,
        RouterConfig, RoutingMode, TokenizerCacheConfig, TraceConfig,
    },
    core::ConnectionMode,
    observability::{
        metrics::PrometheusConfig,
        otel_trace::{is_otel_enabled, shutdown_otel},
    },
    server::{self, ServerConfig},
    service_discovery::ServiceDiscoveryConfig,
    version,
};
fn parse_prefill_args() -> Vec<(String, Option<u16>)> {
    let args: Vec<String> = std::env::args().collect();
    let mut prefill_entries = Vec::new();
    let mut i = 0;

    while i < args.len() {
        if args[i] == "--prefill" && i + 1 < args.len() {
            let url = args[i + 1].clone();
            let bootstrap_port = if i + 2 < args.len() && !args[i + 2].starts_with("--") {
                if let Ok(port) = args[i + 2].parse::<u16>() {
                    i += 1;
                    Some(port)
                } else if args[i + 2].to_lowercase() == "none" {
                    i += 1;
                    None
                } else {
                    None
                }
            } else {
                None
            };
            prefill_entries.push((url, bootstrap_port));
            i += 2;
        } else {
            i += 1;
        }
    }

    prefill_entries
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    #[value(name = "sglang")]
    Sglang,
    #[value(name = "vllm")]
    Vllm,
    #[value(name = "trtllm")]
    Trtllm,
    #[value(name = "openai")]
    Openai,
    #[value(name = "anthropic")]
    Anthropic,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Backend::Sglang => "sglang",
            Backend::Vllm => "vllm",
            Backend::Trtllm => "trtllm",
            Backend::Openai => "openai",
            Backend::Anthropic => "anthropic",
        };
        write!(f, "{}", s)
    }
}

#[derive(Parser, Debug)]
#[command(name = "sglang-router", alias = "smg", alias = "amg")]
#[command(about = "SGLang Model Gateway - High-performance inference gateway")]
#[command(args_conflicts_with_subcommands = true)]
#[command(long_about = r#"
SGLang Model Gateway - Rust-based inference gateway

Usage:
  smg launch [OPTIONS]     Launch router (short command)
  amg launch [OPTIONS]     Launch router (alternative)
  sglang-router [OPTIONS]  Launch router (full name)

Examples:
  # Regular mode
  smg launch --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode
  smg launch --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 9002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --policy cache_aware

  # With different policies
  smg launch --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --prefill-policy cache_aware --decode-policy power_of_two

"#)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    router_args: CliArgs,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the router (same as running without subcommand)
    #[command(visible_alias = "start")]
    Launch {
        #[command(flatten)]
        args: CliArgs,
    },
}

#[derive(Parser, Debug)]
struct CliArgs {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 30000)]
    port: u16,

    #[arg(long, num_args = 0..)]
    worker_urls: Vec<String>,

    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "cache_aware", "power_of_two"])]
    policy: String,

    #[arg(long, default_value_t = false)]
    pd_disaggregation: bool,

    #[arg(long, action = ArgAction::Append)]
    decode: Vec<String>,

    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two"])]
    prefill_policy: Option<String>,

    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two"])]
    decode_policy: Option<String>,

    #[arg(long, default_value_t = 1800)]
    worker_startup_timeout_secs: u64,

    #[arg(long, default_value_t = 30)]
    worker_startup_check_interval: u64,

    #[arg(long, default_value_t = 1)]
    worker_load_check_interval: u64,

    #[arg(long, default_value_t = 0.3)]
    cache_threshold: f32,

    #[arg(long, default_value_t = 64)]
    balance_abs_threshold: usize,

    #[arg(long, default_value_t = 1.5)]
    balance_rel_threshold: f32,

    #[arg(long, default_value_t = 120)]
    eviction_interval: u64,

    #[arg(long, default_value_t = 67108864)]
    max_tree_size: usize,

    #[arg(long, default_value_t = 536870912)]
    max_payload_size: usize,

    #[arg(long, default_value_t = false)]
    dp_aware: bool,

    #[arg(long, default_value_t = false)]
    dp_minimum_tokens_scheduler: bool,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long, value_enum, default_value_t = Backend::Sglang, alias = "runtime")]
    backend: Backend,

    #[arg(long)]
    log_dir: Option<String>,

    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"])]
    log_level: String,

    #[arg(long, default_value_t = false)]
    service_discovery: bool,

    #[arg(long, num_args = 0..)]
    selector: Vec<String>,

    #[arg(long, default_value_t = 80)]
    service_discovery_port: u16,

    #[arg(long)]
    service_discovery_namespace: Option<String>,

    #[arg(long, num_args = 0..)]
    prefill_selector: Vec<String>,

    #[arg(long, num_args = 0..)]
    decode_selector: Vec<String>,

    #[arg(long, default_value_t = 29000)]
    prometheus_port: u16,

    #[arg(long, default_value = "0.0.0.0")]
    prometheus_host: String,

    #[arg(long, num_args = 0..)]
    prometheus_duration_buckets: Vec<f64>,

    #[arg(long, num_args = 0..)]
    request_id_headers: Vec<String>,

    #[arg(long, default_value_t = 1800)]
    request_timeout_secs: u64,

    /// Grace period in seconds to wait for in-flight requests during shutdown.
    /// When the server receives SIGTERM/SIGINT, it will stop accepting new connections
    /// and wait up to this duration for existing streaming requests to complete.
    #[arg(long, default_value_t = 180)]
    shutdown_grace_period_secs: u64,

    #[arg(long, default_value_t = -1)]
    max_concurrent_requests: i32,

    #[arg(long, default_value_t = 100)]
    queue_size: usize,

    #[arg(long, default_value_t = 60)]
    queue_timeout_secs: u64,

    #[arg(long)]
    rate_limit_tokens_per_second: Option<i32>,

    #[arg(long, num_args = 0..)]
    cors_allowed_origins: Vec<String>,

    #[arg(long, default_value_t = 5)]
    retry_max_retries: u32,

    #[arg(long, default_value_t = 50)]
    retry_initial_backoff_ms: u64,

    #[arg(long, default_value_t = 30000)]
    retry_max_backoff_ms: u64,

    #[arg(long, default_value_t = 1.5)]
    retry_backoff_multiplier: f32,

    #[arg(long, default_value_t = 0.2)]
    retry_jitter_factor: f32,

    #[arg(long, default_value_t = false)]
    disable_retries: bool,

    #[arg(long, default_value_t = 10)]
    cb_failure_threshold: u32,

    #[arg(long, default_value_t = 3)]
    cb_success_threshold: u32,

    #[arg(long, default_value_t = 60)]
    cb_timeout_duration_secs: u64,

    #[arg(long, default_value_t = 120)]
    cb_window_duration_secs: u64,

    #[arg(long, default_value_t = false)]
    disable_circuit_breaker: bool,

    #[arg(long, default_value_t = 3)]
    health_failure_threshold: u32,

    #[arg(long, default_value_t = 2)]
    health_success_threshold: u32,

    #[arg(long, default_value_t = 5)]
    health_check_timeout_secs: u64,

    #[arg(long, default_value_t = 60)]
    health_check_interval_secs: u64,

    #[arg(long, default_value = "/health")]
    health_check_endpoint: String,

    #[arg(long, default_value_t = false)]
    enable_igw: bool,

    #[arg(long)]
    model_path: Option<String>,

    #[arg(long)]
    tokenizer_path: Option<String>,

    #[arg(long)]
    chat_template: Option<String>,

    #[arg(long, default_value_t = false)]
    tokenizer_cache_enable_l0: bool,

    #[arg(long, default_value_t = 10000)]
    tokenizer_cache_l0_max_entries: usize,

    #[arg(long, default_value_t = false)]
    tokenizer_cache_enable_l1: bool,

    #[arg(long, default_value_t = 52428800)]
    tokenizer_cache_l1_max_memory: usize,

    #[arg(long, default_value = "memory", value_parser = ["memory", "none", "oracle","postgres"])]
    history_backend: String,

    #[arg(long, env = "ATP_WALLET_PATH")]
    oracle_wallet_path: Option<String>,

    #[arg(long, env = "ATP_TNS_ALIAS")]
    oracle_tns_alias: Option<String>,

    #[arg(long, env = "ATP_DSN")]
    oracle_dsn: Option<String>,

    #[arg(long, env = "ATP_USER")]
    oracle_user: Option<String>,

    #[arg(long, env = "ATP_PASSWORD")]
    oracle_password: Option<String>,

    #[arg(long, env = "ATP_POOL_MIN")]
    oracle_pool_min: Option<usize>,

    #[arg(long, env = "ATP_POOL_MAX")]
    oracle_pool_max: Option<usize>,

    #[arg(long, env = "ATP_POOL_TIMEOUT_SECS")]
    oracle_pool_timeout_secs: Option<u64>,

    #[arg(long)]
    postgres_db_url: Option<String>,

    #[arg(long)]
    postgres_pool_max_size: Option<usize>,

    #[arg(long)]
    reasoning_parser: Option<String>,

    #[arg(long)]
    tool_call_parser: Option<String>,

    #[arg(long)]
    mcp_config_path: Option<String>,

    #[arg(long, default_value_t = false)]
    enable_wasm: bool,

    #[arg(long, default_value_t = false)]
    enable_trace: bool,

    #[arg(long, default_value = "localhost:4317")]
    otlp_traces_endpoint: String,

    #[arg(long)]
    tls_cert_path: Option<String>,

    #[arg(long)]
    tls_key_path: Option<String>,
}

enum OracleConnectSource {
    Dsn { descriptor: String },
    Wallet { path: String, alias: String },
}

impl CliArgs {
    fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
        for url in worker_urls {
            if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                return ConnectionMode::Grpc { port: None };
            }
        }
        ConnectionMode::Http
    }

    fn parse_selector(selector_list: &[String]) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for item in selector_list {
            if let Some(eq_pos) = item.find('=') {
                let key = item[..eq_pos].to_string();
                let value = item[eq_pos + 1..].to_string();
                map.insert(key, value);
            }
        }
        map
    }

    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5,
            },
            _ => PolicyConfig::RoundRobin,
        }
    }

    fn resolve_oracle_connect_details(&self) -> ConfigResult<OracleConnectSource> {
        if let Some(dsn) = self.oracle_dsn.clone() {
            return Ok(OracleConnectSource::Dsn { descriptor: dsn });
        }

        let wallet_path = self
            .oracle_wallet_path
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_wallet_path or ATP_WALLET_PATH".to_string(),
            })?;

        let tns_alias = self
            .oracle_tns_alias
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_tns_alias or ATP_TNS_ALIAS".to_string(),
            })?;

        Ok(OracleConnectSource::Wallet {
            path: wallet_path,
            alias: tns_alias,
        })
    }

    fn build_oracle_config(&self) -> ConfigResult<OracleConfig> {
        let (wallet_path, connect_descriptor) = match self.resolve_oracle_connect_details()? {
            OracleConnectSource::Dsn { descriptor } => (None, descriptor),
            OracleConnectSource::Wallet { path, alias } => (Some(path), alias),
        };
        let username = self
            .oracle_user
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_user or ATP_USER".to_string(),
            })?;
        let password = self
            .oracle_password
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_password or ATP_PASSWORD".to_string(),
            })?;

        let pool_min = self
            .oracle_pool_min
            .unwrap_or_else(OracleConfig::default_pool_min);
        let pool_max = self
            .oracle_pool_max
            .unwrap_or_else(OracleConfig::default_pool_max);

        if pool_min == 0 {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_min".to_string(),
                value: pool_min.to_string(),
                reason: "pool minimum must be at least 1".to_string(),
            });
        }

        if pool_max < pool_min {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_max".to_string(),
                value: pool_max.to_string(),
                reason: "pool maximum must be greater than or equal to minimum".to_string(),
            });
        }

        let pool_timeout_secs = self
            .oracle_pool_timeout_secs
            .unwrap_or_else(OracleConfig::default_pool_timeout_secs);

        Ok(OracleConfig {
            wallet_path,
            connect_descriptor,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
        })
    }

    fn build_postgres_config(&self) -> ConfigResult<PostgresConfig> {
        let db_url = self.postgres_db_url.clone().unwrap_or_default();
        let pool_max = self
            .postgres_pool_max_size
            .unwrap_or_else(PostgresConfig::default_pool_max);
        let pcf = PostgresConfig { db_url, pool_max };
        pcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(pcf)
    }

    fn to_router_config(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
        let mode = if self.enable_igw {
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if matches!(self.backend, Backend::Openai) {
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
            }
        } else if self.pd_disaggregation {
            let decode_urls = self.decode.clone();

            // Allow empty URLs to support dynamic worker addition
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
            }
        } else {
            // Allow empty URLs to support dynamic worker addition
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        let policy = self.parse_policy(&self.policy);

        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: Self::parse_selector(&self.selector),
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            })
        } else {
            None
        };

        let metrics = Some(MetricsConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        let trace_config = Some(TraceConfig {
            enable_trace: self.enable_trace,
            otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
        });

        let mut all_urls = Vec::new();
        match &mode {
            RoutingMode::Regular { worker_urls } => {
                all_urls.extend(worker_urls.clone());
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => {
                for (url, _) in prefill_urls {
                    all_urls.push(url.clone());
                }
                all_urls.extend(decode_urls.clone());
            }
            RoutingMode::OpenAI { .. } => {}
        }
        let connection_mode = match &mode {
            RoutingMode::OpenAI { .. } => ConnectionMode::Http,
            _ => Self::determine_connection_mode(&all_urls),
        };

        let history_backend = match self.history_backend.as_str() {
            "none" => HistoryBackend::None,
            "oracle" => HistoryBackend::Oracle,
            "postgres" => HistoryBackend::Postgres,
            _ => HistoryBackend::Memory,
        };

        let oracle = if history_backend == HistoryBackend::Oracle {
            Some(self.build_oracle_config()?)
        } else {
            None
        };
        let postgres = if history_backend == HistoryBackend::Postgres {
            Some(self.build_postgres_config()?)
        } else {
            None
        };

        let builder = RouterConfig::builder()
            .mode(mode)
            .policy(policy)
            .connection_mode(connection_mode)
            .host(&self.host)
            .port(self.port)
            .max_payload_size(self.max_payload_size)
            .request_timeout_secs(self.request_timeout_secs)
            .worker_startup_timeout_secs(self.worker_startup_timeout_secs)
            .worker_startup_check_interval_secs(self.worker_startup_check_interval)
            .worker_load_check_interval_secs(self.worker_load_check_interval)
            .max_concurrent_requests(self.max_concurrent_requests)
            .queue_size(self.queue_size)
            .queue_timeout_secs(self.queue_timeout_secs)
            .cors_allowed_origins(self.cors_allowed_origins.clone())
            .retry_config(RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            })
            .health_check_config(HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
            })
            .tokenizer_cache(TokenizerCacheConfig {
                enable_l0: self.tokenizer_cache_enable_l0,
                l0_max_entries: self.tokenizer_cache_l0_max_entries,
                enable_l1: self.tokenizer_cache_enable_l1,
                l1_max_memory: self.tokenizer_cache_l1_max_memory,
            })
            .history_backend(history_backend)
            .log_level(&self.log_level)
            .maybe_api_key(self.api_key.as_ref())
            .maybe_discovery(discovery)
            .maybe_metrics(metrics)
            .maybe_trace(trace_config)
            .maybe_log_dir(self.log_dir.as_ref())
            .maybe_request_id_headers(
                (!self.request_id_headers.is_empty()).then(|| self.request_id_headers.clone()),
            )
            .maybe_rate_limit_tokens_per_second(self.rate_limit_tokens_per_second)
            .maybe_model_path(self.model_path.as_ref())
            .maybe_tokenizer_path(self.tokenizer_path.as_ref())
            .maybe_chat_template(self.chat_template.as_ref())
            .maybe_oracle(oracle)
            .maybe_postgres(postgres)
            .maybe_reasoning_parser(self.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.tool_call_parser.as_ref())
            .maybe_mcp_config_path(self.mcp_config_path.as_ref())
            .dp_aware(self.dp_aware)
            .dp_minimum_tokens_scheduler(self.dp_minimum_tokens_scheduler)
            .retries(!self.disable_retries)
            .circuit_breaker(!self.disable_circuit_breaker)
            .enable_wasm(self.enable_wasm)
            .igw(self.enable_igw)
            .maybe_server_cert_and_key(self.tls_cert_path.as_ref(), self.tls_key_path.as_ref());

        builder.build()
    }

    fn to_server_config(&self, router_config: RouterConfig) -> ServerConfig {
        let service_discovery_config = if self.service_discovery {
            Some(ServiceDiscoveryConfig {
                enabled: true,
                selector: Self::parse_selector(&self.selector),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                pd_mode: self.pd_disaggregation,
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            })
        } else {
            None
        };

        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
            duration_buckets: if self.prometheus_duration_buckets.is_empty() {
                None
            } else {
                Some(self.prometheus_duration_buckets.clone())
            },
        });

        ServerConfig {
            host: self.host.clone(),
            port: self.port,
            router_config,
            max_payload_size: self.max_payload_size,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            service_discovery_config,
            prometheus_config,
            request_timeout_secs: self.request_timeout_secs,
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            shutdown_grace_period_secs: self.shutdown_grace_period_secs,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for version flags before parsing other args to avoid errors
    let args: Vec<String> = std::env::args().collect();
    for arg in &args {
        if arg == "--version" || arg == "-V" {
            println!("{}", version::get_version_string());
            return Ok(());
        }
        if arg == "--version-verbose" {
            println!("{}", version::get_verbose_version_string());
            return Ok(());
        }
    }

    let prefill_urls = parse_prefill_args();

    let mut filtered_args: Vec<String> = Vec::new();
    let raw_args: Vec<String> = std::env::args().collect();
    let mut i = 0;

    while i < raw_args.len() {
        if raw_args[i] == "--prefill" && i + 1 < raw_args.len() {
            i += 2;
            if i < raw_args.len()
                && !raw_args[i].starts_with("--")
                && (raw_args[i].parse::<u16>().is_ok() || raw_args[i].to_lowercase() == "none")
            {
                i += 1;
            }
        } else {
            filtered_args.push(raw_args[i].clone());
            i += 1;
        }
    }

    let cli = Cli::parse_from(filtered_args);

    // Handle subcommands or use direct args
    let cli_args = match cli.command {
        Some(Commands::Launch { args }) => args,
        None => cli.router_args,
    };

    println!("SGLang Router starting...");
    println!("Host: {}:{}", cli_args.host, cli_args.port);
    let mode_str = if cli_args.enable_igw {
        "IGW (Inference Gateway)".to_string()
    } else if matches!(cli_args.backend, Backend::Openai) {
        "OpenAI Backend".to_string()
    } else if cli_args.pd_disaggregation {
        "PD Disaggregated".to_string()
    } else {
        format!("Regular ({})", cli_args.backend)
    };
    println!("Mode: {}", mode_str);

    match cli_args.backend {
        Backend::Vllm | Backend::Trtllm | Backend::Anthropic => {
            println!(
                "WARNING: runtime '{}' not implemented yet; falling back to regular routing. \
Provide --worker-urls or PD flags as usual.",
                cli_args.backend
            );
        }
        Backend::Sglang | Backend::Openai => {}
    }

    if !cli_args.enable_igw {
        println!("Policy: {}", cli_args.policy);

        if cli_args.pd_disaggregation && !prefill_urls.is_empty() {
            println!("Prefill nodes: {:?}", prefill_urls);
            println!("Decode nodes: {:?}", cli_args.decode);
        }
    }

    let router_config = cli_args.to_router_config(prefill_urls)?;
    router_config.validate()?;
    let server_config = cli_args.to_server_config(router_config);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move { server::startup(server_config).await })?;
    if is_otel_enabled() {
        shutdown_otel();
    }
    Ok(())
}
