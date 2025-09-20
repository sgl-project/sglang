// Worker Initialization Module
// Separates worker lifecycle management from router construction

use crate::config::types::{ConnectionMode as ConfigConnectionMode, RouterConfig, RoutingMode};
use crate::core::{
    BasicWorkerBuilder, CircuitBreakerConfig, ConnectionMode, HealthConfig, Worker, WorkerRegistry,
    WorkerType,
};
use crate::policies::PolicyRegistry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// WorkerInitializer handles the creation and registration of workers
/// based on routing configuration, separating this concern from router constructors
pub struct WorkerInitializer;

impl WorkerInitializer {
    /// Initialize workers based on configuration and register them in the WorkerRegistry
    pub async fn initialize_workers(
        config: &RouterConfig,
        worker_registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Initializing workers for routing mode: {:?}", config.mode);

        match &config.mode {
            RoutingMode::Regular { worker_urls } => {
                Self::create_regular_workers(
                    worker_urls,
                    &config.connection_mode,
                    config,
                    worker_registry,
                    policy_registry,
                )
                .await?;
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => {
                Self::create_prefill_workers(
                    prefill_urls,
                    &config.connection_mode,
                    config,
                    worker_registry,
                    policy_registry,
                )
                .await?;
                Self::create_decode_workers(
                    decode_urls,
                    &config.connection_mode,
                    config,
                    worker_registry,
                    policy_registry,
                )
                .await?;
            }
            RoutingMode::OpenAI { .. } => {
                info!("OpenAI routing mode - no local workers to initialize");
            }
        }

        // Wait for workers to be healthy if any were registered
        if worker_registry.stats().total_workers > 0 {
            Self::wait_for_healthy_workers(
                worker_registry,
                config.worker_startup_timeout_secs,
                config.worker_startup_check_interval_secs,
            )
            .await?;
        }

        Ok(())
    }

    /// Create regular workers for standard routing mode
    async fn create_regular_workers(
        urls: &[String],
        config_connection_mode: &ConfigConnectionMode,
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Creating {} regular workers", urls.len());

        // Convert config connection mode to core connection mode
        let connection_mode = Self::convert_connection_mode(config_connection_mode, urls.first());

        // Convert circuit breaker config
        let circuit_breaker_config = config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Convert health check config
        let health_config = HealthConfig {
            timeout_secs: config.health_check.timeout_secs,
            check_interval_secs: config.health_check.check_interval_secs,
            endpoint: config.health_check.endpoint.clone(),
            failure_threshold: config.health_check.failure_threshold,
            success_threshold: config.health_check.success_threshold,
        };

        let mut registered_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        for url in urls {
            // TODO: Add DP-aware support when we have dp_rank/dp_size info
            let worker = BasicWorkerBuilder::new(url.clone())
                .worker_type(WorkerType::Regular)
                .connection_mode(connection_mode.clone())
                .circuit_breaker_config(core_cb_config.clone())
                .health_config(health_config.clone())
                .build();

            let worker_arc = Arc::new(worker) as Arc<dyn Worker>;
            let model_id = worker_arc.model_id();
            let worker_id = registry.register(Arc::clone(&worker_arc));
            info!("Registered regular worker {} with ID {:?}", url, worker_id);

            // Track workers by model for cache-aware policy initialization
            registered_workers
                .entry(model_id.to_string())
                .or_default()
                .push(Arc::clone(&worker_arc));

            // Notify policy registry about the worker
            if let Some(policy_reg) = policy_registry {
                policy_reg.on_worker_added(model_id, None);
            }
        }

        // Initialize cache-aware policies with all workers for each model
        if let Some(policy_reg) = policy_registry {
            for (model_id, workers) in registered_workers {
                policy_reg.init_cache_aware_policy(&model_id, &workers);
            }
        }

        Ok(())
    }

    /// Create prefill workers for disaggregated routing mode
    async fn create_prefill_workers(
        prefill_entries: &[(String, Option<u16>)],
        config_connection_mode: &ConfigConnectionMode,
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Creating {} prefill workers", prefill_entries.len());

        // Convert config connection mode to core connection mode
        let connection_mode = Self::convert_connection_mode(
            config_connection_mode,
            prefill_entries.first().map(|(url, _)| url),
        );

        // Convert circuit breaker config
        let circuit_breaker_config = config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Convert health check config
        let health_config = HealthConfig {
            timeout_secs: config.health_check.timeout_secs,
            check_interval_secs: config.health_check.check_interval_secs,
            endpoint: config.health_check.endpoint.clone(),
            failure_threshold: config.health_check.failure_threshold,
            success_threshold: config.health_check.success_threshold,
        };

        let mut registered_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        for (url, bootstrap_port) in prefill_entries {
            // TODO: Add DP-aware support when we have dp_rank/dp_size info
            let worker = BasicWorkerBuilder::new(url.clone())
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: *bootstrap_port,
                })
                .connection_mode(connection_mode.clone())
                .circuit_breaker_config(core_cb_config.clone())
                .health_config(health_config.clone())
                .build();

            let worker_arc = Arc::new(worker) as Arc<dyn Worker>;
            let model_id = worker_arc.model_id();
            let worker_id = registry.register(Arc::clone(&worker_arc));
            info!("Registered prefill worker {} with ID {:?}", url, worker_id);

            // Track workers by model for cache-aware policy initialization
            registered_workers
                .entry(model_id.to_string())
                .or_default()
                .push(Arc::clone(&worker_arc));

            // Notify policy registry about the worker
            if let Some(policy_reg) = policy_registry {
                policy_reg.on_worker_added(model_id, None);
            }
        }

        // Initialize cache-aware policies for PD mode
        if let Some(policy_reg) = policy_registry {
            // Collect all prefill workers
            let all_prefill_workers: Vec<Arc<dyn Worker>> = registered_workers
                .values()
                .flat_map(|workers| workers.iter().cloned())
                .collect();

            // Initialize PD policies (will handle both prefill and decode, but we only have prefill here)
            policy_reg.init_pd_cache_aware_policies(&all_prefill_workers, &[]);
        }

        Ok(())
    }

    /// Create decode workers for disaggregated routing mode
    async fn create_decode_workers(
        urls: &[String],
        config_connection_mode: &ConfigConnectionMode,
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
    ) -> Result<(), String> {
        info!("Creating {} decode workers", urls.len());

        // Convert config connection mode to core connection mode
        let connection_mode = Self::convert_connection_mode(config_connection_mode, urls.first());

        // Convert circuit breaker config
        let circuit_breaker_config = config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Convert health check config
        let health_config = HealthConfig {
            timeout_secs: config.health_check.timeout_secs,
            check_interval_secs: config.health_check.check_interval_secs,
            endpoint: config.health_check.endpoint.clone(),
            failure_threshold: config.health_check.failure_threshold,
            success_threshold: config.health_check.success_threshold,
        };

        let mut registered_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        for url in urls {
            // TODO: Add DP-aware support when we have dp_rank/dp_size info
            let worker = BasicWorkerBuilder::new(url.clone())
                .worker_type(WorkerType::Decode)
                .connection_mode(connection_mode.clone())
                .circuit_breaker_config(core_cb_config.clone())
                .health_config(health_config.clone())
                .build();

            let worker_arc = Arc::new(worker) as Arc<dyn Worker>;
            let model_id = worker_arc.model_id();
            let worker_id = registry.register(Arc::clone(&worker_arc));
            info!("Registered decode worker {} with ID {:?}", url, worker_id);

            // Track workers by model for cache-aware policy initialization
            registered_workers
                .entry(model_id.to_string())
                .or_default()
                .push(Arc::clone(&worker_arc));

            // Notify policy registry about the worker
            if let Some(policy_reg) = policy_registry {
                policy_reg.on_worker_added(model_id, None);
            }
        }

        // Initialize cache-aware policies for PD mode
        if let Some(policy_reg) = policy_registry {
            // Collect all decode workers
            let all_decode_workers: Vec<Arc<dyn Worker>> = registered_workers
                .values()
                .flat_map(|workers| workers.iter().cloned())
                .collect();

            // Initialize PD policies (will handle both prefill and decode, but we only have decode here)
            policy_reg.init_pd_cache_aware_policies(&[], &all_decode_workers);
        }

        Ok(())
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

        loop {
            let stats = registry.stats();

            if stats.healthy_workers > 0 {
                info!(
                    "Workers healthy: {}/{} workers are ready",
                    stats.healthy_workers, stats.total_workers
                );

                // If we have at least one healthy worker, we can proceed
                // This allows partial degradation rather than total failure
                return Ok(());
            }

            if start_time.elapsed() > timeout {
                let error_msg = format!(
                    "Timeout waiting for workers to become healthy after {}s. Total workers: {}, Healthy: {}",
                    timeout_secs, stats.total_workers, stats.healthy_workers
                );
                warn!("{}", error_msg);

                // If we have workers but none are healthy, it's still a failure
                if stats.total_workers > 0 {
                    return Err(error_msg);
                } else {
                    // No workers at all might be OK for some configurations
                    warn!("No workers registered, proceeding anyway");
                    return Ok(());
                }
            }

            tokio::time::sleep(check_interval).await;
        }
    }

    /// Initialize workers for gRPC connections specifically
    /// This is used when gRPC clients are pre-connected
    pub async fn initialize_grpc_workers(
        worker_urls: &[String],
        worker_type: WorkerType,
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        policy_registry: Option<&Arc<PolicyRegistry>>,
        grpc_clients: &mut HashMap<String, crate::grpc::SglangSchedulerClient>,
    ) -> Result<(), String> {
        info!(
            "Creating {} gRPC workers of type {:?}",
            worker_urls.len(),
            worker_type
        );

        // Convert circuit breaker config
        let circuit_breaker_config = config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Convert health check config
        let health_config = HealthConfig {
            timeout_secs: config.health_check.timeout_secs,
            check_interval_secs: config.health_check.check_interval_secs,
            endpoint: config.health_check.endpoint.clone(),
            failure_threshold: config.health_check.failure_threshold,
            success_threshold: config.health_check.success_threshold,
        };

        let mut registered_workers: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();

        for url in worker_urls {
            if let Some(client) = grpc_clients.remove(url) {
                let worker = BasicWorkerBuilder::new(url.clone())
                    .worker_type(worker_type.clone())
                    .connection_mode(ConnectionMode::Grpc { port: None })
                    .circuit_breaker_config(core_cb_config.clone())
                    .health_config(health_config.clone())
                    .grpc_client(client)
                    .build();

                let worker_arc = Arc::new(worker) as Arc<dyn Worker>;
                let model_id = worker_arc.model_id();
                let worker_id = registry.register(Arc::clone(&worker_arc));
                info!("Registered gRPC worker {} with ID {:?}", url, worker_id);

                // Track workers by model for cache-aware policy initialization
                registered_workers
                    .entry(model_id.to_string())
                    .or_default()
                    .push(Arc::clone(&worker_arc));

                // Notify policy registry about the worker
                if let Some(policy_reg) = policy_registry {
                    policy_reg.on_worker_added(model_id, None);
                }
            } else {
                warn!("No gRPC client available for worker {}, skipping", url);
            }
        }

        // Initialize cache-aware policies with all workers for each model
        if let Some(policy_reg) = policy_registry {
            for (model_id, workers) in registered_workers {
                policy_reg.init_cache_aware_policy(&model_id, &workers);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_connection_mode() {
        // HTTP mode
        assert!(matches!(
            WorkerInitializer::convert_connection_mode(
                &ConfigConnectionMode::Http,
                Some(&"http://localhost:8080".to_string())
            ),
            ConnectionMode::Http
        ));

        // gRPC mode
        assert!(matches!(
            WorkerInitializer::convert_connection_mode(
                &ConfigConnectionMode::Grpc,
                Some(&"grpc://localhost:50051".to_string())
            ),
            ConnectionMode::Grpc { .. }
        ));

        // No URL provided
        assert!(matches!(
            WorkerInitializer::convert_connection_mode(&ConfigConnectionMode::Http, None),
            ConnectionMode::Http
        ));
    }
}
