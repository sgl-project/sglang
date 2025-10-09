//! Unified Worker Management Module
//!
//! Handles all aspects of worker lifecycle including discovery, initialization,
//! runtime management, and health monitoring.

use crate::config::types::{
    CircuitBreakerConfig as ConfigCircuitBreakerConfig, ConnectionMode as ConfigConnectionMode,
    HealthCheckConfig, RouterConfig, RoutingMode,
};
use crate::core::{
    BasicWorkerBuilder, CircuitBreakerConfig, ConnectionMode, DPAwareWorkerBuilder, HealthConfig,
    Worker, WorkerFactory, WorkerRegistry, WorkerType,
};
use crate::policies::PolicyRegistry;
use crate::protocols::worker_spec::{
    FlushCacheResult, WorkerConfigRequest, WorkerLoadInfo, WorkerLoadsResult,
};
use crate::server::AppContext;
use futures::future;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{watch, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

static HTTP_CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
});

/// Server information returned from worker endpoints
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerInfo {
    pub model_id: Option<String>,
    pub model_path: Option<String>,
    pub dp_size: Option<usize>,
    pub version: Option<String>,
    pub max_batch_size: Option<usize>,
    pub max_total_tokens: Option<usize>,
    pub max_prefill_tokens: Option<usize>,
    pub max_running_requests: Option<usize>,
    pub max_num_reqs: Option<usize>,
}

/// DP (Data Parallel) information for a worker
#[derive(Debug, Clone)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

/// Unified worker management
pub struct WorkerManager;

impl WorkerManager {
    /// Get server info from /get_server_info endpoint
    pub async fn get_server_info(url: &str, api_key: Option<&str>) -> Result<ServerInfo, String> {
        let base_url = url.trim_end_matches('/');

        let server_info_url = format!("{}/get_server_info", base_url);
        let mut req = HTTP_CLIENT.get(&server_info_url);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        let response = req
            .send()
            .await
            .map_err(|e| format!("Failed to connect to {}: {}", server_info_url, e))?;

        if !response.status().is_success() {
            return Err(format!(
                "Server returned status {} from {}",
                response.status(),
                server_info_url
            ));
        }

        let json = response
            .json::<Value>()
            .await
            .map_err(|e| format!("Failed to parse response from {}: {}", server_info_url, e))?;

        info!(
            "Successfully retrieved server info from {}",
            server_info_url
        );
        Self::parse_server_info(json)
    }

    /// Get model info from /get_model_info endpoint
    pub async fn get_model_info(url: &str, api_key: Option<&str>) -> Result<Value, String> {
        let base_url = url.trim_end_matches('/');

        let model_info_url = format!("{}/get_model_info", base_url);
        let mut req = HTTP_CLIENT.get(&model_info_url);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        let response = req
            .send()
            .await
            .map_err(|e| format!("Failed to connect to {}: {}", model_info_url, e))?;

        if !response.status().is_success() {
            return Err(format!(
                "Server returned status {} from {}",
                response.status(),
                model_info_url
            ));
        }

        let json = response
            .json::<Value>()
            .await
            .map_err(|e| format!("Failed to parse response from {}: {}", model_info_url, e))?;

        info!("Successfully retrieved model info from {}", model_info_url);
        Ok(json)
    }

    /// Get DP info for a worker URL
    pub async fn get_dp_info(url: &str, api_key: Option<&str>) -> Result<DpInfo, String> {
        let info = Self::get_server_info(url, api_key).await?;

        let dp_size = info
            .dp_size
            .ok_or_else(|| format!("No dp_size in response from {}", url))?;

        let model_id = info
            .model_id
            .or_else(|| {
                info.model_path
                    .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
            })
            .unwrap_or_else(|| "unknown".to_string());

        Ok(DpInfo { dp_size, model_id })
    }

    /// Generate DP-aware worker URLs
    pub async fn get_dp_aware_urls(
        base_urls: &[String],
        api_key: Option<&str>,
    ) -> Result<Vec<String>, String> {
        let mut dp_urls = Vec::new();

        for base_url in base_urls {
            match Self::get_dp_info(base_url, api_key).await {
                Ok(dp_info) => {
                    info!(
                        "Discovered DP size {} for {} (model: {})",
                        dp_info.dp_size, base_url, dp_info.model_id
                    );

                    for rank in 0..dp_info.dp_size {
                        dp_urls.push(format!("{}@{}", base_url, rank));
                    }
                }
                Err(e) => {
                    return Err(format!("Failed to get DP info from {}: {}", base_url, e));
                }
            }
        }

        Ok(dp_urls)
    }

    /// Initialize workers from configuration at startup
    pub async fn initialize_workers(
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Starting worker initialization");

        // Determine connection mode from config
        let connection_mode = &config.connection_mode;

        match &config.mode {
            RoutingMode::Regular { worker_urls } => match connection_mode {
                ConfigConnectionMode::Http => {
                    Self::initialize_regular_workers(
                        worker_urls,
                        config,
                        registry,
                        policy_registry,
                    )
                    .await?;
                }
                ConfigConnectionMode::Grpc => {
                    Self::initialize_grpc_workers(worker_urls, config, registry, policy_registry)
                        .await?;
                }
            },
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => match connection_mode {
                ConfigConnectionMode::Http => {
                    let prefill_entries: Vec<(&String, &Option<u16>)> =
                        prefill_urls.iter().map(|(url, port)| (url, port)).collect();

                    Self::initialize_prefill_workers(
                        &prefill_entries,
                        config,
                        registry,
                        policy_registry,
                    )
                    .await?;
                    Self::initialize_decode_workers(decode_urls, config, registry, policy_registry)
                        .await?;
                }
                ConfigConnectionMode::Grpc => {
                    Self::initialize_grpc_pd_workers(
                        prefill_urls,
                        decode_urls,
                        config,
                        registry,
                        policy_registry,
                    )
                    .await?;
                }
            },
            RoutingMode::OpenAI { .. } => {
                info!("OpenAI routing mode - no workers to initialize");
            }
        }

        Self::wait_for_healthy_workers(
            registry,
            config.worker_startup_timeout_secs,
            config.health_check.check_interval_secs,
        )
        .await?;

        info!("Worker initialization completed successfully");
        Ok(())
    }

    /// Initialize regular workers
    async fn initialize_regular_workers(
        urls: &[String],
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Creating {} regular workers", urls.len());

        let connection_mode = Self::convert_connection_mode(&config.connection_mode, urls.first());
        let circuit_breaker_config =
            Self::convert_circuit_breaker_config(&config.effective_circuit_breaker_config());
        let health_config = Self::convert_health_config(&config.health_check);

        let mut registered_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        for url in urls {
            if config.dp_aware {
                match Self::get_dp_info(url, config.api_key.as_deref()).await {
                    Ok(dp_info) => {
                        info!(
                            "Discovered DP-aware worker {} with size {}",
                            url, dp_info.dp_size
                        );

                        for rank in 0..dp_info.dp_size {
                            let mut builder =
                                DPAwareWorkerBuilder::new(url.clone(), rank, dp_info.dp_size)
                                    .worker_type(WorkerType::Regular)
                                    .connection_mode(connection_mode.clone())
                                    .circuit_breaker_config(circuit_breaker_config.clone())
                                    .health_config(health_config.clone());

                            if let Some(ref key) = config.api_key {
                                builder = builder.api_key(key.clone());
                            }

                            let worker = Arc::new(builder.build()) as Arc<dyn Worker>;

                            let model_id = worker.model_id();
                            let worker_id = registry.register(Arc::clone(&worker));
                            info!(
                                "Registered DP-aware worker {}@{} with ID {:?}",
                                url, rank, worker_id
                            );

                            registered_workers
                                .entry(model_id.to_string())
                                .or_default()
                                .push(Arc::clone(&worker));

                            if let Some(policy_reg) = policy_registry {
                                policy_reg.on_worker_added(model_id, None);
                            }
                        }
                    }
                    Err(e) => {
                        return Err(format!(
                            "Failed to get DP info for worker {}: {}. DP-aware mode requires all workers to support DP.",
                            url, e
                        ));
                    }
                }
            } else {
                let worker = Self::create_basic_worker(
                    url.clone(),
                    WorkerType::Regular,
                    connection_mode.clone(),
                    config.api_key.clone(),
                    None,
                    circuit_breaker_config.clone(),
                    health_config.clone(),
                );
                Self::register_worker(worker, registry, &mut registered_workers, policy_registry);
            }
        }

        Self::initialize_cache_policies(&registered_workers, registry, policy_registry);
        Ok(())
    }

    /// Initialize prefill workers for PD mode
    async fn initialize_prefill_workers(
        prefill_entries: &[(&String, &Option<u16>)],
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Creating {} prefill workers", prefill_entries.len());

        let connection_mode = Self::convert_connection_mode(
            &config.connection_mode,
            prefill_entries.first().map(|(url, _)| *url),
        );
        let circuit_breaker_config =
            Self::convert_circuit_breaker_config(&config.effective_circuit_breaker_config());
        let health_config = Self::convert_health_config(&config.health_check);

        let mut registered_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        // TODO: Add proper DP-aware support for prefill workers in PD mode
        if config.dp_aware {
            warn!("DP-aware mode is not yet supported for prefill workers in PD mode. Creating regular prefill workers instead.");
        }

        for (url, bootstrap_port) in prefill_entries {
            let worker_type = WorkerType::Prefill {
                bootstrap_port: **bootstrap_port,
            };
            let worker = Self::create_basic_worker(
                (*url).clone(),
                worker_type,
                connection_mode.clone(),
                config.api_key.clone(),
                None,
                circuit_breaker_config.clone(),
                health_config.clone(),
            );
            Self::register_worker(worker, registry, &mut registered_workers, policy_registry);
        }

        if let Some(policy_reg) = policy_registry {
            let all_prefill_workers: Vec<Arc<dyn Worker>> = registered_workers
                .values()
                .flat_map(|workers| workers.iter().cloned())
                .collect();
            policy_reg.init_pd_cache_aware_policies(&all_prefill_workers, &[]);
        }

        Ok(())
    }

    /// Initialize decode workers for PD mode
    async fn initialize_decode_workers(
        urls: &[String],
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Creating {} decode workers", urls.len());

        let connection_mode = Self::convert_connection_mode(&config.connection_mode, urls.first());
        let circuit_breaker_config =
            Self::convert_circuit_breaker_config(&config.effective_circuit_breaker_config());
        let health_config = Self::convert_health_config(&config.health_check);

        let mut registered_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        // TODO: Add proper DP-aware support for decode workers in PD mode
        if config.dp_aware {
            warn!("DP-aware mode is not yet supported for decode workers in PD mode. Creating regular decode workers instead.");
        }

        for url in urls {
            let worker = Self::create_basic_worker(
                url.clone(),
                WorkerType::Decode,
                connection_mode.clone(),
                config.api_key.clone(),
                None,
                circuit_breaker_config.clone(),
                health_config.clone(),
            );
            Self::register_worker(worker, registry, &mut registered_workers, policy_registry);
        }

        if let Some(policy_reg) = policy_registry {
            let all_decode_workers: Vec<Arc<dyn Worker>> = registered_workers
                .values()
                .flat_map(|workers| workers.iter().cloned())
                .collect();
            policy_reg.init_pd_cache_aware_policies(&[], &all_decode_workers);
        }

        Ok(())
    }

    /// Initialize gRPC workers for regular mode
    async fn initialize_grpc_workers(
        urls: &[String],
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Creating {} gRPC regular workers", urls.len());

        let circuit_breaker_config =
            Self::convert_circuit_breaker_config(&config.effective_circuit_breaker_config());
        let health_config = Self::convert_health_config(&config.health_check);
        let connection_mode = ConnectionMode::Grpc { port: None };

        let mut registered_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        for url in urls {
            let worker = Self::create_basic_worker(
                url.clone(),
                WorkerType::Regular,
                connection_mode.clone(),
                config.api_key.clone(),
                None,
                circuit_breaker_config.clone(),
                health_config.clone(),
            );
            Self::register_worker(worker, registry, &mut registered_workers, policy_registry);
            info!(
                "Registered gRPC worker at {} (will connect on first use)",
                url
            );
        }

        Self::initialize_cache_policies(&registered_workers, registry, policy_registry);
        Ok(())
    }

    /// Initialize gRPC PD (Prefill-Decode) workers
    async fn initialize_grpc_pd_workers(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!(
            "Creating {} gRPC prefill workers and {} gRPC decode workers",
            prefill_urls.len(),
            decode_urls.len()
        );

        let circuit_breaker_config =
            Self::convert_circuit_breaker_config(&config.effective_circuit_breaker_config());
        let health_config = Self::convert_health_config(&config.health_check);

        let mut registered_prefill_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();
        let mut registered_decode_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        for (url, bootstrap_port) in prefill_urls {
            let worker_type = WorkerType::Prefill {
                bootstrap_port: *bootstrap_port,
            };
            let connection_mode = ConnectionMode::Grpc {
                port: *bootstrap_port,
            };

            let worker = Self::create_basic_worker(
                url.clone(),
                worker_type,
                connection_mode,
                config.api_key.clone(),
                None,
                circuit_breaker_config.clone(),
                health_config.clone(),
            );
            Self::register_worker(
                worker,
                registry,
                &mut registered_prefill_workers,
                policy_registry,
            );
            info!(
                "Registered gRPC prefill worker at {} (will connect on first use)",
                url
            );
        }

        // Create decode workers
        for url in decode_urls {
            let connection_mode = ConnectionMode::Grpc { port: None };

            let worker = Self::create_basic_worker(
                url.clone(),
                WorkerType::Decode,
                connection_mode,
                config.api_key.clone(),
                None,
                circuit_breaker_config.clone(),
                health_config.clone(),
            );
            Self::register_worker(
                worker,
                registry,
                &mut registered_decode_workers,
                policy_registry,
            );
            info!(
                "Registered gRPC decode worker at {} (will connect on first use)",
                url
            );
        }

        if let Some(policy_reg) = policy_registry {
            let all_prefill_workers: Vec<Arc<dyn Worker>> = registered_prefill_workers
                .values()
                .flat_map(|workers| workers.iter().cloned())
                .collect();
            let all_decode_workers: Vec<Arc<dyn Worker>> = registered_decode_workers
                .values()
                .flat_map(|workers| workers.iter().cloned())
                .collect();
            policy_reg.init_pd_cache_aware_policies(&all_prefill_workers, &all_decode_workers);
        }

        Ok(())
    }

    /// Add a worker from a configuration request
    pub async fn add_worker_from_config(
        config: &WorkerConfigRequest,
        context: &AppContext,
    ) -> Result<String, String> {
        let mut labels = config.labels.clone();

        let model_id = if let Some(ref model_id) = config.model_id {
            model_id.clone()
        } else {
            match Self::get_server_info(&config.url, config.api_key.as_deref()).await {
                Ok(info) => info
                    .model_id
                    .or_else(|| {
                        info.model_path
                            .as_ref()
                            .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
                    })
                    .unwrap_or_else(|| "unknown".to_string()),
                Err(e) => {
                    warn!("Failed to query server info from {}: {}", config.url, e);
                    "unknown".to_string()
                }
            }
        };

        labels.insert("model_id".to_string(), model_id.clone());
        if let Some(priority) = config.priority {
            labels.insert("priority".to_string(), priority.to_string());
        }
        if let Some(cost) = config.cost {
            labels.insert("cost".to_string(), cost.to_string());
        }
        if let Some(ref tokenizer_path) = config.tokenizer_path {
            labels.insert("tokenizer_path".to_string(), tokenizer_path.clone());
        }
        if let Some(ref reasoning_parser) = config.reasoning_parser {
            labels.insert("reasoning_parser".to_string(), reasoning_parser.clone());
        }
        if let Some(ref tool_parser) = config.tool_parser {
            labels.insert("tool_parser".to_string(), tool_parser.clone());
        }
        if let Some(ref chat_template) = config.chat_template {
            labels.insert("chat_template".to_string(), chat_template.clone());
        }

        let worker_type = config
            .worker_type
            .as_ref()
            .map(|t| match t.as_str() {
                "prefill" => WorkerType::Prefill {
                    bootstrap_port: config.bootstrap_port,
                },
                "decode" => WorkerType::Decode,
                _ => WorkerType::Regular,
            })
            .unwrap_or(WorkerType::Regular);

        let connection_mode = if config.url.starts_with("grpc://") {
            ConnectionMode::Grpc { port: None }
        } else {
            ConnectionMode::Http
        };

        let policy_hint = labels.get("policy").cloned();

        Self::add_worker_internal(
            &config.url,
            worker_type,
            connection_mode,
            config.api_key.clone(),
            Some(labels),
            policy_hint.as_deref(),
            context,
        )
        .await
    }

    /// Add a worker from URL (legacy endpoint)
    pub async fn add_worker(
        url: &str,
        api_key: &Option<String>,
        context: &AppContext,
    ) -> Result<String, String> {
        Self::add_worker_internal(
            url,
            WorkerType::Regular,
            ConnectionMode::Http,
            api_key.clone(),
            None,
            None,
            context,
        )
        .await
    }

    /// Remove a worker
    pub fn remove_worker(url: &str, context: &AppContext) -> Result<String, String> {
        if context.router_config.dp_aware {
            Self::remove_dp_aware_workers(url, context)
        } else {
            Self::remove_single_worker(url, context)
        }
    }

    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    /// Internal method to add a worker with all parameters
    async fn add_worker_internal(
        worker_url: &str,
        worker_type: WorkerType,
        connection_mode: ConnectionMode,
        api_key: Option<String>,
        labels: Option<HashMap<String, String>>,
        policy_hint: Option<&str>,
        context: &AppContext,
    ) -> Result<String, String> {
        WorkerFactory::validate_health(
            worker_url,
            context.router_config.worker_startup_timeout_secs,
        )
        .await
        .map_err(|e| format!("Health check failed: {}", e))?;

        let circuit_breaker_config = Self::convert_circuit_breaker_config(
            &context.router_config.effective_circuit_breaker_config(),
        );
        let health_config = Self::convert_health_config(&context.router_config.health_check);

        if context.router_config.dp_aware {
            let dp_urls = Self::get_dp_aware_urls(
                &[worker_url.to_string()],
                context.router_config.api_key.as_deref(),
            )
            .await?;
            let mut workers_added = 0;
            let mut model_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

            let dp_size_for_base = dp_urls.len();

            for (rank, dp_url) in dp_urls.iter().enumerate() {
                if context.worker_registry.get_by_url(dp_url).is_some() {
                    info!("Worker {} already exists, skipping", dp_url);
                    continue;
                }

                let base_url = dp_url.split('@').next().unwrap().to_string();
                let mut builder = DPAwareWorkerBuilder::new(base_url, rank, dp_size_for_base)
                    .worker_type(worker_type.clone())
                    .connection_mode(connection_mode.clone())
                    .circuit_breaker_config(circuit_breaker_config.clone())
                    .health_config(health_config.clone());

                if let Some(ref key) = api_key {
                    builder = builder.api_key(key.clone());
                }

                if let Some(ref worker_labels) = labels {
                    builder = builder.labels(worker_labels.clone());
                }

                let worker = Arc::new(builder.build()) as Arc<dyn Worker>;

                let model_id = worker.model_id().to_string();
                context.worker_registry.register(worker.clone());
                workers_added += 1;

                model_workers
                    .entry(model_id.clone())
                    .or_default()
                    .push(worker);

                context
                    .policy_registry
                    .on_worker_added(&model_id, policy_hint);
            }

            for model_id in model_workers.keys() {
                let all_model_workers = context.worker_registry.get_by_model_fast(model_id);
                if let Some(policy) = context.policy_registry.get_policy(model_id) {
                    if policy.name() == "cache_aware" {
                        context
                            .policy_registry
                            .init_cache_aware_policy(model_id, &all_model_workers);
                    }
                }
            }

            if workers_added == 0 {
                Ok(format!("All DP workers already exist for {}", worker_url))
            } else {
                Ok(format!(
                    "Added {} DP-aware workers for {}",
                    workers_added, worker_url
                ))
            }
        } else {
            if context.worker_registry.get_by_url(worker_url).is_some() {
                return Err(format!("Worker {} already exists", worker_url));
            }

            let worker = Self::create_basic_worker(
                worker_url.to_string(),
                worker_type,
                connection_mode,
                api_key,
                labels,
                circuit_breaker_config,
                health_config,
            );

            let model_id = worker.model_id().to_string();
            context.worker_registry.register(worker.clone());
            context
                .policy_registry
                .on_worker_added(&model_id, policy_hint);

            let workers = context.worker_registry.get_by_model_fast(&model_id);
            if let Some(policy) = context.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    context
                        .policy_registry
                        .init_cache_aware_policy(&model_id, &workers);
                }
            }

            Ok(format!("Worker {} added successfully", worker_url))
        }
    }

    /// Remove a single worker
    fn remove_single_worker(worker_url: &str, context: &AppContext) -> Result<String, String> {
        let worker = context
            .worker_registry
            .get_by_url(worker_url)
            .ok_or_else(|| format!("Worker {} not found", worker_url))?;
        let model_id = worker.model_id().to_string();

        context
            .policy_registry
            .remove_worker_from_cache_aware(&model_id, worker_url);
        context.worker_registry.remove_by_url(worker_url);
        context.policy_registry.on_worker_removed(&model_id);

        let remaining_workers = context.worker_registry.get_by_model_fast(&model_id);
        if let Some(policy) = context.policy_registry.get_policy(&model_id) {
            if policy.name() == "cache_aware" && !remaining_workers.is_empty() {
                context
                    .policy_registry
                    .init_cache_aware_policy(&model_id, &remaining_workers);
            }
        }

        Ok(format!("Worker {} removed successfully", worker_url))
    }

    /// Remove DP-aware workers with prefix matching
    fn remove_dp_aware_workers(worker_url: &str, context: &AppContext) -> Result<String, String> {
        let worker_url_prefix = format!("{}@", worker_url);
        let mut removed_workers = Vec::new();
        let mut affected_models = std::collections::HashSet::new();

        let all_workers = context.worker_registry.get_all();
        for worker in all_workers.iter() {
            if worker.url().starts_with(&worker_url_prefix) {
                let model_id = worker.model_id().to_string();
                affected_models.insert(model_id.clone());

                context
                    .policy_registry
                    .remove_worker_from_cache_aware(&model_id, worker.url());

                if context
                    .worker_registry
                    .remove_by_url(worker.url())
                    .is_some()
                {
                    removed_workers.push(worker.url().to_string());
                    context.policy_registry.on_worker_removed(&model_id);
                }
            }
        }

        for model_id in affected_models {
            let remaining_workers = context.worker_registry.get_by_model_fast(&model_id);
            if let Some(policy) = context.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" && !remaining_workers.is_empty() {
                    context
                        .policy_registry
                        .init_cache_aware_policy(&model_id, &remaining_workers);
                }
            }
        }

        if removed_workers.is_empty() {
            Err(format!(
                "No workers found with prefix {}",
                worker_url_prefix
            ))
        } else {
            Ok(format!(
                "Removed {} DP-aware workers: {:?}",
                removed_workers.len(),
                removed_workers
            ))
        }
    }

    /// Create a basic worker
    fn create_basic_worker(
        url: String,
        worker_type: WorkerType,
        connection_mode: ConnectionMode,
        api_key: Option<String>,
        labels: Option<HashMap<String, String>>,
        circuit_breaker_config: CircuitBreakerConfig,
        health_config: HealthConfig,
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .connection_mode(connection_mode)
            .circuit_breaker_config(circuit_breaker_config)
            .health_config(health_config);

        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }

        if let Some(worker_labels) = labels {
            builder = builder.labels(worker_labels);
        }

        let worker = builder.build();
        Arc::new(worker) as Arc<dyn Worker>
    }

    /// Register a worker and update policies
    fn register_worker(
        worker: Arc<dyn Worker>,
        registry: &Arc<WorkerRegistry>,
        registered_workers: &mut HashMap<String, Vec<Arc<dyn Worker>>>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) {
        let model_id = worker.model_id();
        let url = worker.url();
        let worker_id = registry.register(Arc::clone(&worker));
        info!("Registered worker {} with ID {:?}", url, worker_id);

        registered_workers
            .entry(model_id.to_string())
            .or_default()
            .push(Arc::clone(&worker));

        if let Some(policy_reg) = policy_registry {
            policy_reg.on_worker_added(model_id, None);
        }
    }

    /// Initialize cache-aware policies
    fn initialize_cache_policies(
        registered_workers: &HashMap<String, Vec<Arc<dyn Worker>>>,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) {
        if let Some(policy_reg) = policy_registry {
            for model_id in registered_workers.keys() {
                let all_model_workers = registry.get_by_model_fast(model_id);
                if let Some(policy) = policy_reg.get_policy(model_id) {
                    if policy.name() == "cache_aware" {
                        policy_reg.init_cache_aware_policy(model_id, &all_model_workers);
                    }
                }
            }
        }
    }

    /// Wait for workers to become healthy
    async fn wait_for_healthy_workers(
        registry: &Arc<WorkerRegistry>,
        timeout_secs: u64,
        check_interval_secs: u64,
    ) -> Result<(), String> {
        let timeout = Duration::from_secs(timeout_secs);
        let check_interval = Duration::from_secs(check_interval_secs);
        let start_time = std::time::Instant::now();

        info!(
            "Waiting for workers to become healthy (timeout: {}s)",
            timeout_secs
        );

        let workers = registry.get_all();
        if workers.is_empty() {
            info!("No workers to wait for, continuing");
            return Ok(());
        }

        // Mark all workers as unhealthy initially
        info!(
            "Marking {} workers as unhealthy before health checks",
            workers.len()
        );
        for worker in &workers {
            worker.set_healthy(false);
        }

        loop {
            // 1. Filter unhealthy workers
            let workers = registry.get_all();
            let unhealthy_workers: Vec<_> = workers
                .iter()
                .filter(|w| !w.is_healthy())
                .cloned()
                .collect();

            // 2. If all workers are healthy, return immediately
            if unhealthy_workers.is_empty() {
                let healthy_urls: Vec<_> = workers.iter().map(|w| w.url().to_string()).collect();
                info!(
                    "All {} workers are healthy: {:?}",
                    workers.len(),
                    healthy_urls
                );
                return Ok(());
            }

            // Check timeout
            if start_time.elapsed() > timeout {
                let healthy_workers: Vec<_> = workers
                    .iter()
                    .filter(|w| w.is_healthy())
                    .map(|w| w.url().to_string())
                    .collect();
                let unhealthy_urls: Vec<_> = unhealthy_workers
                    .iter()
                    .map(|w| w.url().to_string())
                    .collect();

                error!(
                    "Workers failed to become healthy after {}s. Unhealthy: {:?}, Healthy: {:?}",
                    timeout_secs, unhealthy_urls, healthy_workers
                );
                return Err(format!(
                    "Workers failed to become healthy after {}s. Unhealthy: {:?}",
                    timeout_secs, unhealthy_urls
                ));
            }

            let unhealthy_urls: Vec<_> = unhealthy_workers
                .iter()
                .map(|w| w.url().to_string())
                .collect();

            info!(
                "Waiting for {} workers to become healthy. Unhealthy: {:?}",
                unhealthy_workers.len(),
                unhealthy_urls
            );

            // 3. Check health of all unhealthy workers in parallel
            let health_check_futures: Vec<_> = unhealthy_workers
                .iter()
                .map(|worker| {
                    let w = worker.clone();
                    let url = worker.url().to_string();
                    async move {
                        match w.check_health_async().await {
                            Ok(_) => {
                                w.set_healthy(true);
                                debug!("Worker {} now healthy", url);
                            }
                            Err(e) => {
                                debug!("Worker {} health check failed: {}", url, e);
                            }
                        }
                    }
                })
                .collect();

            future::join_all(health_check_futures).await;

            // 4. Check if all workers are now healthy after health checks
            let still_unhealthy: Vec<_> = workers.iter().filter(|w| !w.is_healthy()).collect();

            // 5. If all workers are now healthy, return immediately without sleeping
            if still_unhealthy.is_empty() {
                let healthy_urls: Vec<_> = workers.iter().map(|w| w.url().to_string()).collect();
                info!(
                    "All {} workers are healthy: {:?}",
                    workers.len(),
                    healthy_urls
                );
                return Ok(());
            }

            // 6. Otherwise, sleep before next iteration
            tokio::time::sleep(check_interval).await;
        }
    }

    /// Parse server info from JSON response
    fn parse_server_info(json: Value) -> Result<ServerInfo, String> {
        Ok(ServerInfo {
            model_id: json
                .get("model_id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| json.get("model").and_then(|v| v.as_str()).map(String::from)),
            model_path: json
                .get("model_path")
                .and_then(|v| v.as_str())
                .map(String::from),
            dp_size: json
                .get("dp_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            version: json
                .get("version")
                .and_then(|v| v.as_str())
                .map(String::from),
            max_batch_size: json
                .get("max_batch_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            max_total_tokens: json
                .get("max_total_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            max_prefill_tokens: json
                .get("max_prefill_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            max_running_requests: json
                .get("max_running_requests")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            max_num_reqs: json
                .get("max_num_reqs")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        })
    }

    /// Convert config connection mode to core connection mode
    fn convert_connection_mode(
        config_mode: &ConfigConnectionMode,
        _sample_url: Option<&String>,
    ) -> ConnectionMode {
        match config_mode {
            ConfigConnectionMode::Http => ConnectionMode::Http,
            ConfigConnectionMode::Grpc => ConnectionMode::Grpc { port: None },
        }
    }

    /// Convert config circuit breaker to core circuit breaker
    fn convert_circuit_breaker_config(config: &ConfigCircuitBreakerConfig) -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold: config.failure_threshold,
            success_threshold: config.success_threshold,
            timeout_duration: Duration::from_secs(config.timeout_duration_secs),
            window_duration: Duration::from_secs(config.window_duration_secs),
        }
    }

    /// Convert config health check to core health config
    fn convert_health_config(config: &HealthCheckConfig) -> HealthConfig {
        HealthConfig {
            timeout_secs: config.timeout_secs,
            check_interval_secs: config.check_interval_secs,
            endpoint: config.endpoint.clone(),
            failure_threshold: config.failure_threshold,
            success_threshold: config.success_threshold,
        }
    }
    /// Flush cache on all workers
    ///
    /// Sends a POST request to /flush_cache endpoint on all HTTP workers.
    /// Returns detailed results showing which workers succeeded and which failed.
    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> Result<FlushCacheResult, String> {
        warn!("Flushing cache for ALL workers - this may impact performance temporarily");

        let workers = worker_registry.get_all();

        let http_workers: Vec<_> = workers
            .iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return Ok(FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers: workers.len(),
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            });
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total workers)",
            http_workers.len(),
            workers.len()
        );

        let mut tasks = Vec::new();
        for worker in &http_workers {
            let url = worker.url().to_string();
            let flush_url = format!("{}/flush_cache", url);
            let mut request = client.post(&flush_url);

            if let Some(api_key) = worker.api_key() {
                request = request.header("Authorization", format!("Bearer {}", api_key));
            }

            let worker_url = url.clone();
            tasks.push(async move {
                let result = request.send().await;
                (worker_url, result)
            });
        }

        let results = future::join_all(tasks).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for (url, result) in results {
            match result {
                Ok(response) if response.status().is_success() => {
                    debug!("Successfully flushed cache on worker: {}", url);
                    successful.push(url);
                }
                Ok(response) => {
                    let error = format!("HTTP {}", response.status());
                    warn!("Failed to flush cache on worker {}: {}", url, error);
                    failed.push((url, error));
                }
                Err(e) => {
                    let error = e.to_string();
                    error!("Failed to connect to worker {}: {}", url, error);
                    failed.push((url, error));
                }
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush completed: {} succeeded, {} failed (out of {} HTTP workers)",
                successful.len(),
                failed.len(),
                http_workers.len()
            )
        };

        info!("{}", message);

        Ok(FlushCacheResult {
            successful,
            failed,
            total_workers: workers.len(),
            http_workers: http_workers.len(),
            message,
        })
    }
    pub async fn get_worker_load(
        url: &str,
        api_key: Option<&str>,
        client: &reqwest::Client,
    ) -> Option<isize> {
        let load_url = format!("{}/get_load", url);
        let mut request = client.get(&load_url);

        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }

        match request.send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<Value>().await {
                    Ok(json) => {
                        // The /get_load endpoint returns an array of load info objects (one per DP rank)
                        // Each object has: {dp_rank, num_reqs, num_waiting_reqs, num_tokens}
                        if let Some(array) = json.as_array() {
                            let total_tokens: i64 = array
                                .iter()
                                .filter_map(|entry| {
                                    entry.get("num_tokens").and_then(|v| v.as_i64())
                                })
                                .sum();
                            debug!("Worker {} load (total tokens): {}", url, total_tokens);
                            Some(total_tokens as isize)
                        } else {
                            warn!(
                                "Invalid load response from {}: expected array, got {:?}",
                                url, json
                            );
                            None
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse load response from {}: {}", url, e);
                        None
                    }
                }
            }
            Ok(response) => {
                warn!(
                    "Failed to get load from {}: HTTP {}",
                    url,
                    response.status()
                );
                None
            }
            Err(e) => {
                warn!("Failed to connect to {} for load check: {}", url, e);
                None
            }
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        // Prepare tasks for parallel execution
        let mut tasks = Vec::new();
        for worker in &workers {
            let url = worker.url().to_string();
            let api_key = worker.api_key().clone();
            let worker_type = match worker.worker_type() {
                WorkerType::Regular => None,
                WorkerType::Prefill { .. } => Some("prefill".to_string()),
                WorkerType::Decode => Some("decode".to_string()),
            };
            let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
            let client = client.clone();

            tasks.push(async move {
                let load = if is_http {
                    Self::get_worker_load(&url, api_key.as_deref(), &client)
                        .await
                        .unwrap_or(-1)
                } else {
                    -1
                };

                WorkerLoadInfo {
                    worker: url,
                    worker_type,
                    load,
                }
            });
        }

        let loads = future::join_all(tasks).await;

        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    /// Create a new load monitor
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Start monitoring worker loads
    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    /// Stop monitoring worker loads
    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    /// Get a receiver for load updates
    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    /// The main monitoring loop
    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();

            if power_of_two_policies.is_empty() {
                debug!("No PowerOfTwo policies found, skipping load fetch");
                continue;
            }

            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads {
                loads.insert(load_info.worker, load_info.load);
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    /// Check if monitoring is currently active
    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_server_info() {
        let json = serde_json::json!({
            "model_id": "llama-3",
            "model_path": "/models/llama-3",
            "dp_size": 4,
            "version": "0.1.0"
        });

        let info = WorkerManager::parse_server_info(json).unwrap();
        assert_eq!(info.model_id, Some("llama-3".to_string()));
        assert_eq!(info.dp_size, Some(4));
    }

    #[test]
    fn test_parse_server_info_with_fallback() {
        let json = serde_json::json!({
            "model": "gpt-4",
            "dp_size": 2
        });

        let info = WorkerManager::parse_server_info(json).unwrap();
        assert_eq!(info.model_id, Some("gpt-4".to_string()));
        assert_eq!(info.dp_size, Some(2));
    }

    #[test]
    fn test_parse_server_info_minimal() {
        let json = serde_json::json!({});
        let info = WorkerManager::parse_server_info(json).unwrap();
        assert_eq!(info.model_id, None);
        assert_eq!(info.dp_size, None);
    }
}
