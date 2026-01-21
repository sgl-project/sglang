use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::{StreamExt, TryStreamExt};
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::{
        watcher::{watcher, Config},
        WatchStreamExt,
    },
    Client,
};
use rustls;
use tokio::{task, time};
use tracing::{debug, error, info, warn};

use crate::{
    app_context::AppContext,
    core::Job,
    mesh::service::{
        gossip::{NodeState, NodeStatus},
        ClusterState,
    },
    observability::metrics::{metrics_labels, Metrics},
    protocols::worker_spec::WorkerConfigRequest,
};

#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    pub enabled: bool,
    pub selector: HashMap<String, String>,
    pub check_interval: Duration,
    pub port: u16,
    pub namespace: Option<String>,
    // PD mode specific configuration
    pub pd_mode: bool,
    pub prefill_selector: HashMap<String, String>,
    pub decode_selector: HashMap<String, String>,
    // Bootstrap port annotation specific to mooncake implementation
    pub bootstrap_port_annotation: String,
    // Router node discovery for mesh
    pub router_selector: HashMap<String, String>,
    pub router_mesh_port_annotation: String,
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            enabled: false,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8000,
            namespace: None,
            pd_mode: false,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/ha-port".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PodType {
    Prefill,
    Decode,
    Regular,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodInfo {
    pub name: String,
    pub ip: String,
    pub status: String,
    pub is_ready: bool,
    pub pod_type: Option<PodType>,
    pub bootstrap_port: Option<u16>,
    pub is_router: bool,
    pub mesh_port: Option<u16>,
}

impl PodInfo {
    fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
        if selector.is_empty() {
            return false;
        }

        pod.metadata
            .labels
            .as_ref()
            .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
    }

    pub fn should_include(pod: &Pod, config: &ServiceDiscoveryConfig) -> bool {
        if config.pd_mode {
            if config.prefill_selector.is_empty() && config.decode_selector.is_empty() {
                warn!("PD mode enabled but both prefill_selector and decode_selector are empty");
                return false;
            }
            Self::matches_selector(pod, &config.prefill_selector)
                || Self::matches_selector(pod, &config.decode_selector)
        } else {
            if config.selector.is_empty() {
                warn!("Regular mode enabled but selector is empty");
                return false;
            }
            Self::matches_selector(pod, &config.selector)
        }
    }

    pub fn from_pod(pod: &Pod, config: Option<&ServiceDiscoveryConfig>) -> Option<Self> {
        let name = pod.metadata.name.clone()?;
        let status = pod.status.clone()?;
        let pod_ip = status.pod_ip?;

        let is_ready = if let Some(conditions) = &status.conditions {
            conditions
                .iter()
                .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        } else {
            false
        };

        let pod_status = status.phase.unwrap_or_else(|| "Unknown".to_string());

        let pod_type = if let Some(config) = config {
            if config.pd_mode {
                if Self::matches_selector(pod, &config.prefill_selector) {
                    Some(PodType::Prefill)
                } else if Self::matches_selector(pod, &config.decode_selector) {
                    Some(PodType::Decode)
                } else {
                    Some(PodType::Regular)
                }
            } else {
                Some(PodType::Regular)
            }
        } else {
            None
        };

        let bootstrap_port = if matches!(pod_type, Some(PodType::Prefill)) {
            if let Some(config) = config {
                pod.metadata
                    .annotations
                    .as_ref()
                    .and_then(|annotations| annotations.get(&config.bootstrap_port_annotation))
                    .and_then(|port_str| port_str.parse::<u16>().ok())
            } else {
                None
            }
        } else {
            None
        };

        // Check if this is a router pod
        let is_router = if let Some(config) = config {
            !config.router_selector.is_empty()
                && Self::matches_selector(pod, &config.router_selector)
        } else {
            false
        };

        // Extract mesh port from annotation if this is a router pod
        let mesh_port = if is_router {
            if let Some(config) = config {
                pod.metadata
                    .annotations
                    .as_ref()
                    .and_then(|annotations| annotations.get(&config.router_mesh_port_annotation))
                    .and_then(|port_str| port_str.parse::<u16>().ok())
            } else {
                None
            }
        } else {
            None
        };

        Some(PodInfo {
            name,
            ip: pod_ip,
            status: pod_status,
            is_ready,
            pod_type,
            bootstrap_port,
            is_router,
            mesh_port,
        })
    }

    pub fn is_healthy(&self) -> bool {
        self.is_ready && self.status == "Running"
    }

    pub fn worker_url(&self, port: u16) -> String {
        // Default to http:// prefix; workflow will detect actual protocol (HTTP vs gRPC)
        format!("http://{}:{}", self.ip, port)
    }
}

pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
    mesh_cluster_state: Option<ClusterState>,
    mesh_port: Option<u16>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    if !config.enabled {
        return Err(kube::Error::Api(kube::error::ErrorResponse {
            status: "Disabled".to_string(),
            message: "Service discovery is disabled".to_string(),
            reason: "ConfigurationError".to_string(),
            code: 400,
        }));
    }

    let _ = rustls::crypto::ring::default_provider().install_default();

    let client = Client::try_default().await?;

    // Log the appropriate selectors based on mode
    if config.pd_mode {
        let prefill_selector = config
            .prefill_selector
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");

        let decode_selector = config
            .decode_selector
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "Starting K8s service discovery | PD mode | prefill: '{}' | decode: '{}'",
            prefill_selector, decode_selector
        );
    } else {
        let label_selector = config
            .selector
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "Starting K8s service discovery | selector: '{}'",
            label_selector
        );
    }

    // Log router discovery if enabled
    if !config.router_selector.is_empty() {
        let router_selector = config
            .router_selector
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");
        info!(
            "Router node discovery enabled | selector: '{}' | mesh port annotation: '{}'",
            router_selector, config.router_mesh_port_annotation
        );
    }

    let handle = task::spawn(async move {
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));

        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        debug!("K8s service discovery initialized");

        let config_arc = Arc::new(config.clone());
        let port = config.port;

        // Spawn router discovery task if enabled and mesh is available
        // Router discovery requires mesh to be enabled to update cluster state
        // If mesh is not enabled, router discovery is skipped and service discovery works independently
        if !config_arc.router_selector.is_empty() {
            if let (Some(cluster_state), Some(mesh_port)) = (mesh_cluster_state.clone(), mesh_port)
            {
                let router_config = config_arc.clone();
                let router_pods = pods.clone();
                tokio::spawn(async move {
                    start_router_discovery(router_config, router_pods, cluster_state, mesh_port)
                        .await;
                });
                info!("Router discovery enabled (requires mesh to be enabled)");
            } else {
                warn!(
                    "Router selector configured but mesh is not enabled (mesh cluster state or mesh port not provided). \
                    Router discovery requires mesh to be enabled. Skipping router discovery."
                );
            }
        }

        let mut retry_delay = Duration::from_secs(1);
        const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

        loop {
            let watcher_config = Config::default();
            let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();

            let config_clone = Arc::clone(&config_arc);
            let tracked_pods_clone = Arc::clone(&tracked_pods);

            let filtered_stream = watcher_stream.filter_map(move |obj_res| {
                let config_inner = Arc::clone(&config_clone);

                async move {
                    match obj_res {
                        Ok(pod) => {
                            if PodInfo::should_include(&pod, &config_inner) {
                                Some(Ok(pod))
                            } else {
                                None
                            }
                        }
                        Err(e) => Some(Err(e)),
                    }
                }
            });

            let tracked_pods_clone2 = Arc::clone(&tracked_pods_clone);
            let app_context_clone = Arc::clone(&app_context);
            let config_clone2 = Arc::clone(&config_arc);

            match filtered_stream
                .try_for_each(move |pod| {
                    let tracked_pods_inner = Arc::clone(&tracked_pods_clone2);
                    let app_context_inner = Arc::clone(&app_context_clone);
                    let config_inner = Arc::clone(&config_clone2);

                    async move {
                        let pod_info = PodInfo::from_pod(&pod, Some(&config_inner));

                        if let Some(pod_info) = pod_info {
                            if pod.metadata.deletion_timestamp.is_some() {
                                handle_pod_deletion(
                                    &pod_info,
                                    tracked_pods_inner,
                                    app_context_inner,
                                    port,
                                )
                                .await;
                            } else {
                                handle_pod_event(
                                    &pod_info,
                                    tracked_pods_inner,
                                    app_context_inner,
                                    port,
                                    config_inner.pd_mode,
                                )
                                .await;
                            }
                        }
                        Ok(())
                    }
                })
                .await
            {
                Ok(_) => {
                    retry_delay = Duration::from_secs(1);
                }
                Err(err) => {
                    error!("Error in Kubernetes watcher: {}", err);
                    warn!(
                        "Retrying in {} seconds with exponential backoff",
                        retry_delay.as_secs()
                    );
                    time::sleep(retry_delay).await;

                    retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
                }
            }

            warn!(
                "Kubernetes watcher exited, restarting in {} seconds",
                config_arc.check_interval.as_secs()
            );
            time::sleep(config_arc.check_interval).await;
        }
    });

    Ok(handle)
}

async fn handle_pod_event(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashSet<PodInfo>>>,
    app_context: Arc<AppContext>,
    port: u16,
    pd_mode: bool,
) {
    let worker_url = pod_info.worker_url(port);

    if pod_info.is_healthy() {
        // Track whether to add and get count in single lock acquisition
        let (should_add, tracked_count) = {
            let mut tracker = match tracked_pods.lock() {
                Ok(tracker) => tracker,
                Err(e) => {
                    error!("Failed to acquire tracked_pods lock: {}", e);
                    return;
                }
            };

            if tracker.contains(pod_info) {
                (false, tracker.len())
            } else {
                tracker.insert(pod_info.clone());
                (true, tracker.len())
            }
        };

        if should_add {
            info!(
                "Adding pod: {} | type: {:?} | url: {}",
                pod_info.name, pod_info.pod_type, worker_url
            );

            let worker_type = if pd_mode {
                match &pod_info.pod_type {
                    Some(PodType::Prefill) => Some("prefill".to_string()),
                    Some(PodType::Decode) => Some("decode".to_string()),
                    Some(PodType::Regular) | None => None,
                }
            } else {
                None
            };

            let bootstrap_port = if pd_mode {
                match &pod_info.pod_type {
                    Some(PodType::Prefill) => pod_info.bootstrap_port,
                    _ => None,
                }
            } else {
                None
            };

            let config = WorkerConfigRequest {
                url: worker_url.clone(),
                model_id: None,
                worker_type,
                priority: None,
                cost: None,
                runtime: None,
                labels: HashMap::new(),
                bootstrap_port,
                tokenizer_path: None,
                reasoning_parser: None,
                tool_parser: None,
                chat_template: None,
                api_key: app_context.router_config.api_key.clone(),
                health_check_timeout_secs: app_context.router_config.health_check.timeout_secs,
                health_check_interval_secs: app_context
                    .router_config
                    .health_check
                    .check_interval_secs,
                health_success_threshold: app_context.router_config.health_check.success_threshold,
                health_failure_threshold: app_context.router_config.health_check.failure_threshold,
                disable_health_check: app_context.router_config.health_check.disable_health_check,
                max_connection_attempts: app_context.router_config.health_check.success_threshold
                    * 20,
                dp_aware: app_context.router_config.dp_aware,
            };

            let job = Job::AddWorker {
                config: Box::new(config.clone()),
            };

            if let Some(job_queue) = app_context.worker_job_queue.get() {
                match job_queue.submit(job).await {
                    Ok(_) => {
                        debug!("Worker addition job submitted for: {}", worker_url);

                        // Layer 4: Record successful registration from K8s discovery
                        Metrics::record_discovery_registration(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            metrics_labels::REGISTRATION_SUCCESS,
                        );

                        // Update workers discovered gauge (using count from initial lock)
                        Metrics::set_discovery_workers_discovered(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            tracked_count,
                        );
                    }
                    Err(e) => {
                        error!(
                            "Failed to submit worker addition job for {}: {}",
                            worker_url, e
                        );

                        // Layer 4: Record failed registration
                        Metrics::record_discovery_registration(
                            metrics_labels::DISCOVERY_KUBERNETES,
                            metrics_labels::REGISTRATION_FAILED,
                        );

                        if let Ok(mut tracker) = tracked_pods.lock() {
                            tracker.remove(pod_info);
                        }
                    }
                }
            } else {
                debug!(
                    "JobQueue not initialized, skipping async worker addition for: {}",
                    worker_url
                );
            }
        } else {
            // Pod already tracked - this is a duplicate event
            Metrics::record_discovery_registration(
                metrics_labels::DISCOVERY_KUBERNETES,
                metrics_labels::REGISTRATION_DUPLICATE,
            );
        }
    }
}

async fn handle_pod_deletion(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashSet<PodInfo>>>,
    app_context: Arc<AppContext>,
    port: u16,
) {
    let worker_url = pod_info.worker_url(port);

    // Remove pod and get remaining count in single lock acquisition
    let (was_tracked, remaining_count) = {
        let mut tracked = match tracked_pods.lock() {
            Ok(tracked) => tracked,
            Err(e) => {
                error!("Failed to acquire tracked_pods lock during deletion: {}", e);
                return;
            }
        };
        let removed = tracked.remove(pod_info);
        (removed, tracked.len())
    };

    if was_tracked {
        info!(
            "Removing pod: {} | type: {:?} | url: {}",
            pod_info.name, pod_info.pod_type, worker_url
        );

        let job = Job::RemoveWorker {
            url: worker_url.clone(),
        };

        if let Some(job_queue) = app_context.worker_job_queue.get() {
            if let Err(e) = job_queue.submit(job).await {
                error!(
                    "Failed to submit worker removal job for {}: {}",
                    worker_url, e
                );
            } else {
                debug!("Submitted worker removal job for {}", worker_url);

                // Layer 4: Record deregistration from K8s pod deletion
                Metrics::record_discovery_deregistration(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    metrics_labels::DEREGISTRATION_POD_DELETED,
                );

                // Update workers discovered gauge (using count from initial lock)
                Metrics::set_discovery_workers_discovered(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    remaining_count,
                );
            }
        } else {
            error!(
                "JobQueue not initialized, cannot remove worker {}",
                worker_url
            );
        }
    } else {
        debug!(
            "Pod deletion event for untracked/already removed pod: {} (type: {:?}). Worker URL: {}",
            pod_info.name, pod_info.pod_type, worker_url
        );
    }
}

/// Start router node discovery for mesh cluster
async fn start_router_discovery(
    config: Arc<ServiceDiscoveryConfig>,
    pods: Api<Pod>,
    cluster_state: ClusterState,
    default_mesh_port: u16,
) {
    use std::collections::HashMap;

    let mut retry_delay = Duration::from_secs(1);
    const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

    loop {
        let watcher_config = Config::default();
        let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();

        let config_clone = Arc::clone(&config);

        let filtered_stream = watcher_stream.filter_map(move |obj_res| {
            let config_inner = Arc::clone(&config_clone);

            async move {
                match obj_res {
                    Ok(pod) => {
                        // Check if this pod matches router selector
                        if PodInfo::matches_selector(&pod, &config_inner.router_selector) {
                            Some(Ok(pod))
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            }
        });

        let config_clone2 = Arc::clone(&config);
        let cluster_state_clone2 = cluster_state.clone();

        match filtered_stream
            .try_for_each(move |pod| {
                let config_inner = Arc::clone(&config_clone2);
                let cluster_state_inner = cluster_state_clone2.clone();

                async move {
                    let pod_info = PodInfo::from_pod(&pod, Some(&config_inner));

                    if let Some(pod_info) = pod_info {
                        if pod_info.is_router {
                            let mesh_port = pod_info.mesh_port.unwrap_or(default_mesh_port);
                            let node_address = format!("{}:{}", pod_info.ip, mesh_port);

                            if pod.metadata.deletion_timestamp.is_some() {
                                // Pod is being deleted, mark node as Down
                                let mut state = cluster_state_inner.write();
                                if let Some(node) = state.get_mut(&pod_info.name) {
                                    node.status = NodeStatus::Down as i32;
                                    node.version += 1;
                                    info!(
                                        "Router node {} marked as Down (pod deleted)",
                                        pod_info.name
                                    );
                                } else {
                                    debug!(
                                        "Router node {} not found in cluster state (already removed)",
                                        pod_info.name
                                    );
                                }
                            } else if pod_info.is_healthy() {
                                // Pod is healthy, add or update node in cluster state
                                let mut state = cluster_state_inner.write();
                                let existing_version = state
                                    .get(&pod_info.name)
                                    .map(|n| n.version)
                                    .unwrap_or(0);

                                let node_state = NodeState {
                                    name: pod_info.name.clone(),
                                    address: node_address,
                                    status: NodeStatus::Alive as i32,
                                    version: existing_version + 1,
                                    metadata: HashMap::new(),
                                };

                                state.insert(pod_info.name.clone(), node_state.clone());
                                info!(
                                    "Router node {} added/updated in mesh cluster (address: {})",
                                    pod_info.name, node_state.address
                                );
                            } else {
                                // Pod is not healthy, mark as Suspected
                                let mut state = cluster_state_inner.write();
                                if let Some(node) = state.get_mut(&pod_info.name) {
                                    if node.status != NodeStatus::Down as i32 {
                                        node.status = NodeStatus::Suspected as i32;
                                        node.version += 1;
                                        debug!(
                                            "Router node {} marked as Suspected (pod not healthy)",
                                            pod_info.name
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                }
            })
            .await
        {
            Ok(_) => {
                retry_delay = Duration::from_secs(1);
            }
            Err(err) => {
                error!("Error in router discovery watcher: {}", err);
                warn!(
                    "Retrying router discovery in {} seconds with exponential backoff",
                    retry_delay.as_secs()
                );
                time::sleep(retry_delay).await;

                retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
            }
        }

        warn!(
            "Router discovery watcher exited, restarting in {} seconds",
            config.check_interval.as_secs()
        );
        time::sleep(config.check_interval).await;
    }
}

#[cfg(test)]
mod tests {
    use k8s_openapi::{
        api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
        apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time},
    };

    use super::*;

    fn create_k8s_pod(
        name: Option<&str>,
        ip: Option<&str>,
        phase: Option<&str>,
        ready_status: Option<&str>,
        deletion_timestamp: Option<Time>,
    ) -> Pod {
        let mut pod = Pod {
            metadata: ObjectMeta {
                name: name.map(String::from),
                deletion_timestamp,
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };

        if ip.is_some() || phase.is_some() || ready_status.is_some() {
            let mut pod_status = PodStatus {
                pod_ip: ip.map(String::from),
                phase: phase.map(String::from),
                conditions: None,
                ..Default::default()
            };

            if let Some(status_str) = ready_status {
                let condition = PodCondition {
                    type_: "Ready".to_string(),
                    status: status_str.to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                };
                pod_status.conditions = Some(vec![condition]);
            }
            pod.status = Some(pod_status);
        }
        pod
    }

    fn create_pd_k8s_pod(name: &str, ip: &str, pod_type: &str, bootstrap_port: Option<u16>) -> Pod {
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("app".to_string(), "sglang".to_string());
        labels.insert("component".to_string(), pod_type.to_string());

        let mut annotations = std::collections::BTreeMap::new();
        if let Some(port) = bootstrap_port {
            annotations.insert("sglang.ai/bootstrap-port".to_string(), port.to_string());
        }

        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                labels: Some(labels),
                annotations: Some(annotations),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: Some(PodStatus {
                pod_ip: Some(ip.to_string()),
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                }]),
                ..Default::default()
            }),
        }
    }

    async fn create_test_app_context() -> Arc<AppContext> {
        use crate::{
            config::RouterConfig, core::WorkerService, middleware::TokenBucket,
            observability::inflight_tracker::InFlightRequestTracker,
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();

        let worker_registry = Arc::new(crate::core::WorkerRegistry::new());
        let worker_job_queue = Arc::new(std::sync::OnceLock::new());

        // Note: Using uninitialized queue for tests to avoid spawning background workers
        // Jobs submitted during tests will queue but not be processed
        Arc::new(AppContext {
            client: reqwest::Client::new(),
            router_config: router_config.clone(),
            rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
            worker_registry: worker_registry.clone(),
            policy_registry: Arc::new(crate::policies::PolicyRegistry::new(
                router_config.policy.clone(),
            )),
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            router_manager: None,
            response_storage: Arc::new(crate::data_connector::MemoryResponseStorage::new()),
            conversation_storage: Arc::new(crate::data_connector::MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(
                crate::data_connector::MemoryConversationItemStorage::new(),
            ),
            load_monitor: None,
            configured_reasoning_parser: None,
            configured_tool_parser: None,
            worker_job_queue: worker_job_queue.clone(),
            workflow_engines: Arc::new(std::sync::OnceLock::new()),
            mcp_manager: Arc::new(std::sync::OnceLock::new()),
            tokenizer_registry: Arc::new(crate::tokenizer::registry::TokenizerRegistry::new()),
            wasm_manager: None,
            worker_service: Arc::new(WorkerService::new(
                worker_registry,
                worker_job_queue,
                router_config,
            )),
            inflight_tracker: InFlightRequestTracker::new(),
        })
    }

    fn create_pd_config() -> ServiceDiscoveryConfig {
        let mut prefill_selector = HashMap::new();
        prefill_selector.insert("app".to_string(), "sglang".to_string());
        prefill_selector.insert("component".to_string(), "prefill".to_string());

        let mut decode_selector = HashMap::new();
        decode_selector.insert("app".to_string(), "sglang".to_string());
        decode_selector.insert("component".to_string(), "decode".to_string());

        ServiceDiscoveryConfig {
            enabled: true,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8080,
            namespace: None,
            pd_mode: true,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/ha-port".to_string(),
        }
    }

    #[test]
    fn test_pod_info_should_include() {
        let config = create_pd_config();

        let prefill_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        assert!(PodInfo::should_include(&prefill_pod, &config));

        let decode_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        assert!(PodInfo::should_include(&decode_pod, &config));

        let unmatched_pod = create_pd_k8s_pod("other-pod", "10.0.0.3", "other", None);
        assert!(!PodInfo::should_include(&unmatched_pod, &config));

        let mut regular_config = ServiceDiscoveryConfig::default();
        regular_config
            .selector
            .insert("app".to_string(), "sglang".to_string());
        regular_config.pd_mode = false;

        let regular_pod = create_pd_k8s_pod("worker-pod", "10.0.0.4", "worker", None);
        assert!(PodInfo::should_include(&regular_pod, &regular_config));
    }

    #[test]
    fn test_service_discovery_config_default() {
        let config = ServiceDiscoveryConfig::default();
        assert!(!config.enabled);
        assert!(config.selector.is_empty());
        assert_eq!(config.check_interval, Duration::from_secs(60));
        assert_eq!(config.port, 8000);
        assert!(config.namespace.is_none());
        assert!(!config.pd_mode);
        assert!(config.prefill_selector.is_empty());
        assert!(config.decode_selector.is_empty());
        assert_eq!(config.bootstrap_port_annotation, "sglang.ai/bootstrap-port");
    }

    #[test]
    fn test_pod_type_enum() {
        let prefill = PodType::Prefill;
        let decode = PodType::Decode;
        let regular = PodType::Regular;

        assert_eq!(format!("{:?}", prefill), "Prefill");
        assert_eq!(format!("{:?}", decode), "Decode");
        assert_eq!(format!("{:?}", regular), "Regular");
    }

    #[test]
    fn test_pod_info_from_pod_valid() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(pod_info.name, "test-pod");
        assert_eq!(pod_info.ip, "10.0.0.1");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert!(pod_info.pod_type.is_none());
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_prefill() {
        let k8s_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "prefill-pod");
        assert_eq!(pod_info.ip, "10.0.0.1");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert_eq!(pod_info.bootstrap_port, Some(8081));
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_decode() {
        let k8s_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "decode-pod");
        assert_eq!(pod_info.ip, "10.0.0.2");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Decode));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_regular_mode() {
        let k8s_pod = create_pd_k8s_pod("regular-pod", "10.0.0.3", "worker", None);
        let mut config = create_pd_config();
        config.pd_mode = false;

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "regular-pod");
        assert_eq!(pod_info.ip, "10.0.0.3");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_unmatched_labels() {
        let k8s_pod = create_pd_k8s_pod("unknown-pod", "10.0.0.4", "unknown", None);
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "unknown-pod");
        assert_eq!(pod_info.ip, "10.0.0.4");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_invalid_bootstrap_port() {
        let mut pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", None);
        pod.metadata.annotations.as_mut().unwrap().insert(
            "sglang.ai/bootstrap-port".to_string(),
            "invalid".to_string(),
        );
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_not_ready() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("False"),
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_no_conditions() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            None,
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_missing_name() {
        let k8s_pod = create_k8s_pod(None, Some("10.0.0.1"), Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_ip() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), None, Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_status_phase() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), Some("10.0.0.1"), None, Some("True"), None);
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(pod_info.status, "Unknown");
    }

    #[test]
    fn test_pod_info_from_pod_no_status_object() {
        let mut k8s_pod = create_k8s_pod(Some("test-pod"), None, None, None, None);
        k8s_pod.status = None;
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_is_healthy() {
        let healthy_pod = PodInfo {
            name: "p1".into(),
            ip: "1.1.1.1".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };
        assert!(healthy_pod.is_healthy());

        let not_ready_pod = PodInfo {
            name: "p2".into(),
            ip: "1.1.1.2".into(),
            status: "Running".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };
        assert!(!not_ready_pod.is_healthy());

        let not_running_pod = PodInfo {
            name: "p3".into(),
            ip: "1.1.1.3".into(),
            status: "Pending".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };
        assert!(!not_running_pod.is_healthy());
    }

    #[test]
    fn test_pod_info_equality_with_pod_type() {
        let pod1 = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
        };

        let pod2 = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
        };

        let pod3 = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };

        assert_eq!(pod1, pod2);
        assert_ne!(pod1, pod3);
    }

    #[tokio::test]
    async fn test_handle_pod_event_add_unhealthy_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Pending".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            false, // pd_mode = false
        )
        .await;

        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }

    #[tokio::test]
    async fn test_handle_pod_deletion_non_existing_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };
        let port = 8080u16;

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
        )
        .await;

        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_prefill_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "prefill-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            true, // pd_mode = true for PD pod
        )
        .await;

        // With fully async control plane, pod is tracked and job is queued
        // Worker registration and validation happen in background job
        assert!(tracked_pods.lock().unwrap().contains(&pod_info));

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_decode_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "decode-pod".into(),
            ip: "1.2.3.5".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            true, // pd_mode = true for PD pod
        )
        .await;

        // With fully async control plane, pod is tracked and job is queued
        // Worker registration and validation happen in background job
        assert!(tracked_pods.lock().unwrap().contains(&pod_info));

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_tracked_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "test-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
        };

        // Add pod to tracked set first
        {
            let mut tracked = tracked_pods.lock().unwrap();
            tracked.insert(pod_info.clone());
        }

        let port = 8080u16;

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
        )
        .await;

        // Pod should be removed from tracking
        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_untracked_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "untracked-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };
        let port = 8080u16;

        // Don't add pod to tracked set

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
        )
        .await;

        // Tracked set should remain empty
        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_unified_handler_regular_mode() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "regular-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            false, // pd_mode = false
        )
        .await;

        // With fully async control plane, pod is tracked and job is queued
        // In regular mode (pd_mode=false), worker_type defaults to Regular
        // Worker registration and validation happen in background job
        assert!(tracked_pods.lock().unwrap().contains(&pod_info));

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_unified_handler_pd_mode_with_prefill() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "prefill-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
            is_router: false,
            mesh_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
            true, // pd_mode = true
        )
        .await;

        // With fully async control plane, pod is tracked and job is queued
        // Worker registration and validation happen in background job
        assert!(tracked_pods.lock().unwrap().contains(&pod_info));

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_unified_handler_deletion_with_pd_mode() {
        let app_context = create_test_app_context().await;
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "decode-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        };

        // Add pod to tracked set first
        {
            let mut tracked = tracked_pods.lock().unwrap();
            tracked.insert(pod_info.clone());
        }

        let port = 8080u16;

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
        )
        .await;

        // Pod should be removed from tracking
        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }
}
