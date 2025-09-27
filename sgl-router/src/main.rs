use clap::{ArgAction, Parser, ValueEnum};
use sglang_router_rs::config::{
    CircuitBreakerConfig, ConfigError, ConfigResult, ConnectionMode, DiscoveryConfig,
    HealthCheckConfig, HistoryBackend, MetricsConfig, OracleConfig, PolicyConfig, RetryConfig,
    RouterConfig, RoutingMode,
};
use sglang_router_rs::metrics::PrometheusConfig;
use sglang_router_rs::server::{self, ServerConfig};
use sglang_router_rs::service_discovery::ServiceDiscoveryConfig;
use std::collections::HashMap;

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
#[command(name = "sglang-router")]
#[command(about = "SGLang Router - High-performance request distribution across worker nodes")]
#[command(long_about = r#"
SGLang Router - High-performance request distribution across worker nodes

Usage:
This launcher enables starting a router with individual worker instances. It is useful for
multi-node setups or when you want to start workers and router separately.

Examples:
  # Regular mode
  sglang-router --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode with same policy for both
  sglang-router --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 9002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --policy cache_aware

  # PD mode with different policies for prefill and decode
  sglang-router --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --prefill-policy cache_aware --decode-policy power_of_two

"#)]
struct CliArgs {
    #[arg(long, default_value = "127.0.0.1")]
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

    #[arg(long, default_value_t = 600)]
    worker_startup_timeout_secs: u64,

    #[arg(long, default_value_t = 30)]
    worker_startup_check_interval: u64,

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

    #[arg(long, default_value = "127.0.0.1")]
    prometheus_host: String,

    #[arg(long, num_args = 0..)]
    request_id_headers: Vec<String>,

    #[arg(long, default_value_t = 1800)]
    request_timeout_secs: u64,

    #[arg(long, default_value_t = 256)]
    max_concurrent_requests: usize,

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

    #[arg(long, default_value = "memory", value_parser = ["memory", "none", "oracle"])]
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
}

enum OracleConnectSource {
    Dsn { descriptor: String },
    Wallet { path: String, alias: String },
}

impl CliArgs {
    fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
        for url in worker_urls {
            if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                return ConnectionMode::Grpc;
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

            if !self.service_discovery && (prefill_urls.is_empty() || decode_urls.is_empty()) {
                return Err(ConfigError::ValidationFailed {
                    reason: "PD disaggregation mode requires --prefill and --decode URLs when not using service discovery".to_string(),
                });
            }

            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
            }
        } else {
            if !self.service_discovery && self.worker_urls.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "Regular mode requires --worker-urls when not using service discovery"
                        .to_string(),
                });
            }
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
            _ => HistoryBackend::Memory,
        };

        let oracle = if history_backend == HistoryBackend::Oracle {
            Some(self.build_oracle_config()?)
        } else {
            None
        };

        Ok(RouterConfig {
            mode,
            policy,
            connection_mode,
            host: self.host.clone(),
            port: self.port,
            max_payload_size: self.max_payload_size,
            request_timeout_secs: self.request_timeout_secs,
            worker_startup_timeout_secs: self.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: self.worker_startup_check_interval,
            dp_aware: self.dp_aware,
            api_key: self.api_key.clone(),
            discovery,
            metrics,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            max_concurrent_requests: self.max_concurrent_requests,
            queue_size: 100,
            queue_timeout_secs: 60,
            cors_allowed_origins: self.cors_allowed_origins.clone(),
            retry: RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            },
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            },
            disable_retries: self.disable_retries,
            disable_circuit_breaker: self.disable_circuit_breaker,
            health_check: HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
            },
            enable_igw: self.enable_igw,
            rate_limit_tokens_per_second: None,
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
            history_backend,
            oracle,
        })
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
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let cli_args = CliArgs::parse_from(filtered_args);

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

    Ok(())
}
