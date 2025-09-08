//! Router Manager for coordinating multiple routers and workers
//!
//! Provides centralized management based on enable_igw flag:
//! - Single Router Mode (enable_igw=false): Router owns workers directly
//! - Multi-Router Mode (enable_igw=true): RouterManager coordinates everything

use crate::config::RouterConfig;
use crate::core::{CircuitBreakerConfig, Worker, WorkerFactory, WorkerRegistry, WorkerType};
use crate::protocols::worker_spec::{
    ServerInfo, WorkerApiResponse, WorkerConfigRequest, WorkerErrorResponse, WorkerInfo,
    WorkerListResponse, WorkerStats, WorkerTypeStats,
};
use crate::routers::RouterTrait;
use dashmap::DashMap;
use std::sync::Arc;
use tracing::{info, warn};

/// Router identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(String);

impl RouterId {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Router Manager - Central coordinator for routers and workers
/// Only created when enable_igw=true
///
/// Phase 2 will support exactly 4 routers:
/// - HTTP Regular Router
/// - HTTP PD Router (Prefill/Decode separation)
/// - gRPC Regular Router  
/// - gRPC PD Router (Prefill/Decode separation)
pub struct RouterManager {
    /// Worker registry (single source of truth in multi-router mode)
    worker_registry: Arc<WorkerRegistry>,

    /// All routers managed by this manager (max 4 routers in Phase 2)
    /// RouterId examples: "http-regular", "http-pd", "grpc-regular", "grpc-pd"
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,

    /// Default router for requests without specific routing
    default_router: Option<RouterId>,

    /// Model to router mapping for model-aware routing
    /// Multiple models can be served by the same router
    model_routers: Arc<DashMap<String, Vec<RouterId>>>,

    /// HTTP client for querying worker info
    client: reqwest::Client,

    /// Configuration
    #[allow(dead_code)] // Will be used in future enhancements
    config: RouterConfig,
}

impl RouterManager {
    /// Create a new router manager
    pub fn new(config: RouterConfig, client: reqwest::Client) -> Self {
        Self {
            worker_registry: Arc::new(WorkerRegistry::new()),
            routers: Arc::new(DashMap::new()),
            default_router: None,
            model_routers: Arc::new(DashMap::new()),
            client,
            config,
        }
    }

    /// Register a router with the manager
    pub fn register_router(
        &mut self,
        id: RouterId,
        router: Arc<dyn RouterTrait>,
        models: Vec<String>,
    ) {
        // Store router
        self.routers.insert(id.clone(), router);

        // Update model mappings
        for model in models {
            self.model_routers
                .entry(model)
                .or_default()
                .push(id.clone());
        }

        // Set as default if first router
        if self.default_router.is_none() {
            self.default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    /// Set the default router
    pub fn set_default_router(&mut self, id: RouterId) {
        self.default_router = Some(id);
    }

    /// Get router for a specific model
    pub fn get_router_for_model(&self, model_id: &str) -> Option<Arc<dyn RouterTrait>> {
        // First try model-specific routers
        if let Some(router_ids) = self.model_routers.get(model_id) {
            if let Some(router_id) = router_ids.first() {
                if let Some(router) = self.routers.get(router_id) {
                    return Some(router.clone());
                }
            }
        }

        // Fall back to default router
        if let Some(ref default_id) = self.default_router {
            self.routers.get(default_id).map(|r| r.clone())
        } else {
            None
        }
    }

    /// Get workers for routing decision
    pub fn get_workers_for_request(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        if let Some(model) = model_id {
            self.worker_registry.get_by_model(model)
        } else {
            self.worker_registry.get_all()
        }
    }

    /// Add a worker to the registry
    pub async fn add_worker(
        &self,
        config: WorkerConfigRequest,
    ) -> Result<WorkerApiResponse, WorkerErrorResponse> {
        // Build labels from configuration
        let mut labels = config.labels.clone();

        // Query server info if model_id not provided
        let model_id = if let Some(model_id) = config.model_id {
            model_id
        } else {
            match self.query_server_info(&config.url).await {
                Ok(info) => {
                    // Extract model_id from server info
                    info.model_id
                        .or_else(|| {
                            info.model_path
                                .as_ref()
                                .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
                        })
                        .unwrap_or_else(|| "unknown".to_string())
                }
                Err(e) => {
                    warn!("Failed to query server info from {}: {}", config.url, e);
                    "unknown".to_string()
                }
            }
        };

        // Add configuration to labels
        labels.insert("model_id".to_string(), model_id.clone());

        if let Some(priority) = config.priority {
            labels.insert("priority".to_string(), priority.to_string());
        }

        if let Some(cost) = config.cost {
            labels.insert("cost".to_string(), cost.to_string());
        }

        // Add gRPC-specific configuration if provided
        if let Some(tokenizer_path) = config.tokenizer_path {
            labels.insert("tokenizer_path".to_string(), tokenizer_path);
        }

        if let Some(reasoning_parser) = config.reasoning_parser {
            labels.insert("reasoning_parser".to_string(), reasoning_parser);
        }

        if let Some(tool_parser) = config.tool_parser {
            labels.insert("tool_parser".to_string(), tool_parser);
        }

        if let Some(chat_template) = config.chat_template {
            labels.insert("chat_template".to_string(), chat_template);
        }

        // Create worker based on type
        // Note: For prefill and decode workers, we can't easily add labels after creation
        // since they return Box<dyn Worker>. We'll need to enhance WorkerFactory in the future.
        let worker = match config.worker_type.as_deref() {
            Some("prefill") => {
                // For now, prefill workers won't have custom labels
                // TODO: Enhance WorkerFactory to accept labels for prefill workers
                WorkerFactory::create_prefill(config.url.clone(), config.bootstrap_port)
            }
            Some("decode") => {
                // For now, decode workers won't have custom labels
                // TODO: Enhance WorkerFactory to accept labels for decode workers
                WorkerFactory::create_decode(config.url.clone())
            }
            _ => {
                // Regular workers can have labels
                WorkerFactory::create_regular_with_labels(
                    config.url.clone(),
                    labels.clone(),
                    CircuitBreakerConfig::default(),
                )
            }
        };

        // Register worker
        let worker_id = self.worker_registry.register(Arc::from(worker));

        info!(
            "Added worker {} with URL {} for model {}",
            worker_id.as_str(),
            config.url,
            model_id
        );

        // Return worker info
        let worker_arc = self.worker_registry.get(&worker_id).unwrap();
        let worker_info = self.worker_to_info(worker_id.as_str(), &worker_arc);

        Ok(WorkerApiResponse {
            success: true,
            message: format!("Worker {} added successfully", worker_id.as_str()),
            worker: Some(worker_info),
        })
    }

    /// Remove a worker from the registry
    pub fn remove_worker(&self, url: &str) -> Result<WorkerApiResponse, WorkerErrorResponse> {
        if let Some(_worker) = self.worker_registry.remove_by_url(url) {
            info!("Removed worker with URL {}", url);
            Ok(WorkerApiResponse {
                success: true,
                message: format!("Worker {} removed successfully", url),
                worker: None,
            })
        } else {
            Err(WorkerErrorResponse {
                error: format!("Worker with URL {} not found", url),
                code: "WORKER_NOT_FOUND".to_string(),
            })
        }
    }

    /// List all workers
    pub fn list_workers(&self) -> WorkerListResponse {
        let workers = self.worker_registry.get_all();
        let worker_infos: Vec<WorkerInfo> = workers
            .iter()
            .enumerate()
            .map(|(i, w)| self.worker_to_info(&format!("worker-{}", i), w))
            .collect();

        let total = worker_infos.len();
        let stats = self.calculate_stats(&workers);

        WorkerListResponse {
            workers: worker_infos,
            total,
            stats,
        }
    }

    /// Get worker by URL
    pub fn get_worker(&self, url: &str) -> Option<WorkerInfo> {
        self.worker_registry
            .get_by_url(url)
            .map(|w| self.worker_to_info("unknown", &w))
    }

    /// Query server info from a worker URL
    async fn query_server_info(&self, url: &str) -> Result<ServerInfo, String> {
        let info_url = format!("{}/get_server_info", url.trim_end_matches('/'));

        match self.client.get(&info_url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    response
                        .json::<ServerInfo>()
                        .await
                        .map_err(|e| format!("Failed to parse server info: {}", e))
                } else {
                    Err(format!("Server returned status: {}", response.status()))
                }
            }
            Err(e) => Err(format!("Failed to connect to server: {}", e)),
        }
    }

    /// Convert Worker to WorkerInfo
    fn worker_to_info(&self, id: &str, worker: &Arc<dyn Worker>) -> WorkerInfo {
        let metadata = worker.metadata();

        WorkerInfo {
            id: id.to_string(),
            url: worker.url().to_string(),
            model_id: worker.model_id().to_string(),
            priority: worker.priority(),
            cost: worker.cost(),
            worker_type: format!("{:?}", worker.worker_type()),
            is_healthy: worker.is_healthy(),
            load: worker.load(),
            connection_mode: format!("{:?}", worker.connection_mode()),
            tokenizer_path: worker.tokenizer_path().map(|s| s.to_string()),
            reasoning_parser: worker.reasoning_parser().map(|s| s.to_string()),
            tool_parser: worker.tool_parser().map(|s| s.to_string()),
            chat_template: worker.chat_template().map(|s| s.to_string()),
            metadata: metadata.labels.clone(),
        }
    }

    /// Calculate worker statistics
    fn calculate_stats(&self, workers: &[Arc<dyn Worker>]) -> WorkerStats {
        let total_workers = workers.len();
        let healthy_workers = workers.iter().filter(|w| w.is_healthy()).count();
        let total_models = self.worker_registry.get_models().len();
        let total_load: usize = workers.iter().map(|w| w.load()).sum();

        let mut regular = 0;
        let mut prefill = 0;
        let mut decode = 0;

        for worker in workers {
            match worker.worker_type() {
                WorkerType::Regular => regular += 1,
                WorkerType::Prefill { .. } => prefill += 1,
                WorkerType::Decode => decode += 1,
            }
        }

        WorkerStats {
            total_workers,
            healthy_workers,
            total_models,
            total_load,
            by_type: WorkerTypeStats {
                regular,
                prefill,
                decode,
            },
        }
    }
}

impl Default for RouterManager {
    fn default() -> Self {
        Self::new(RouterConfig::default(), reqwest::Client::new())
    }
}
