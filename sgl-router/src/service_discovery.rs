use crate::router::Router;

use futures::{StreamExt, TryStreamExt};
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::watcher::{watcher, Config},
    runtime::WatchStreamExt,
    Client,
};
use std::collections::{HashMap, HashSet};

use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task;
use tokio::time;
use tracing::{debug, error, info, warn};

/// Represents the service discovery configuration
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
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            enabled: false,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8000,      // Standard port for modern services
            namespace: None, // None means watch all namespaces
            // PD mode defaults
            pd_mode: false,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
        }
    }
}

/// Pod type for PD mode service discovery
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PodType {
    Prefill,
    Decode,
    Regular,
}

/// Represents a Kubernetes pod's information used for worker management
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodInfo {
    pub name: String,
    pub ip: String,
    pub status: String,
    pub is_ready: bool,
    pub pod_type: Option<PodType>,
    pub bootstrap_port: Option<u16>,
}

impl PodInfo {
    /// Check if a pod matches any of the given selectors
    fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
        if selector.is_empty() {
            return false;
        }

        pod.metadata.labels.as_ref().map_or(false, |labels| {
            selector
                .iter()
                .all(|(k, v)| labels.get(k).map_or(false, |label_value| label_value == v))
        })
    }

    /// Check if a pod should be included in service discovery
    pub fn should_include(pod: &Pod, config: &ServiceDiscoveryConfig) -> bool {
        if config.pd_mode {
            // In PD mode, at least one selector must be non-empty
            if config.prefill_selector.is_empty() && config.decode_selector.is_empty() {
                warn!("PD mode enabled but both prefill_selector and decode_selector are empty");
                return false;
            }
            // In PD mode, pod must match either prefill or decode selector
            Self::matches_selector(pod, &config.prefill_selector)
                || Self::matches_selector(pod, &config.decode_selector)
        } else {
            // In regular mode, pod must match the general selector
            if config.selector.is_empty() {
                warn!("Regular mode enabled but selector is empty");
                return false;
            }
            Self::matches_selector(pod, &config.selector)
        }
    }

    /// Unified PodInfo creation with optional PD configuration
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

        // Determine pod type based on labels if config is provided and in PD mode
        let pod_type = if let Some(config) = config {
            if config.pd_mode {
                // Use simplified helper methods for cleaner logic
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
            // No config provided, default to None (for backwards compatibility)
            None
        };

        // Extract bootstrap port from annotations for prefill pods
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

        Some(PodInfo {
            name,
            ip: pod_ip,
            status: pod_status,
            is_ready,
            pod_type,
            bootstrap_port,
        })
    }

    /// Returns true if the pod is in a state where it can accept traffic
    pub fn is_healthy(&self) -> bool {
        self.is_ready && self.status == "Running"
    }

    /// Generates a worker URL for this pod
    pub fn worker_url(&self, port: u16) -> String {
        format!("http://{}:{}", self.ip, port)
    }
}

pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    router: Arc<Router>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    // Don't initialize anything if service discovery is disabled
    if !config.enabled {
        // Return a generic error when service discovery is disabled
        return Err(kube::Error::Api(kube::error::ErrorResponse {
            status: "Disabled".to_string(),
            message: "Service discovery is disabled".to_string(),
            reason: "ConfigurationError".to_string(),
            code: 400,
        }));
    }

    // Initialize Kubernetes client
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
            "Starting Kubernetes service discovery in PD mode with prefill_selector: '{}', decode_selector: '{}'",
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
            "Starting Kubernetes service discovery with selector: '{}'",
            label_selector
        );
    }

    // Create the task that will run in the background
    let handle = task::spawn(async move {
        // We'll track pods we've already added to avoid duplicates
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));

        // Create a watcher for pods
        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        info!("Kubernetes service discovery initialized successfully");

        // Create Arcs for configuration data
        let config_arc = Arc::new(config.clone());
        let port = config.port;

        let mut retry_delay = Duration::from_secs(1);
        const MAX_RETRY_DELAY: Duration = Duration::from_secs(300); // 5 minutes max

        loop {
            // Create a watcher with the proper parameters according to the kube-rs API
            let watcher_config = Config::default();
            let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();

            // Clone Arcs for the closures
            let config_clone = Arc::clone(&config_arc);
            let tracked_pods_clone = Arc::clone(&tracked_pods);

            // Simplified label selector filter using helper method
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

            // Clone again for the next closure
            let tracked_pods_clone2 = Arc::clone(&tracked_pods_clone);
            let router_clone = Arc::clone(&router);
            let config_clone2 = Arc::clone(&config_arc);

            match filtered_stream
                .try_for_each(move |pod| {
                    let tracked_pods_inner = Arc::clone(&tracked_pods_clone2);
                    let router_inner = Arc::clone(&router_clone);
                    let config_inner = Arc::clone(&config_clone2);

                    async move {
                        let pod_info = PodInfo::from_pod(&pod, Some(&config_inner));

                        if let Some(pod_info) = pod_info {
                            if pod.metadata.deletion_timestamp.is_some() {
                                handle_pod_deletion(
                                    &pod_info,
                                    tracked_pods_inner,
                                    router_inner,
                                    port,
                                    config_inner.pd_mode,
                                )
                                .await;
                            } else {
                                handle_pod_event(
                                    &pod_info,
                                    tracked_pods_inner,
                                    router_inner,
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
                    // Reset retry delay on success
                    retry_delay = Duration::from_secs(1);
                }
                Err(err) => {
                    error!("Error in Kubernetes watcher: {}", err);
                    warn!(
                        "Retrying in {} seconds with exponential backoff",
                        retry_delay.as_secs()
                    );
                    time::sleep(retry_delay).await;

                    // Exponential backoff with jitter
                    retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
                }
            }

            // If the watcher exits for some reason, wait a bit before restarting
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
    router: Arc<Router>,
    port: u16,
    pd_mode: bool,
) {
    let worker_url = pod_info.worker_url(port);

    // If pod is healthy, try to add it (with atomic check-and-insert)
    if pod_info.is_healthy() {
        // Atomic check-and-insert to prevent race conditions
        let should_add = {
            let mut tracker = match tracked_pods.lock() {
                Ok(tracker) => tracker,
                Err(e) => {
                    error!("Failed to acquire tracked_pods lock: {}", e);
                    return;
                }
            };

            if tracker.contains(pod_info) {
                false // Already tracked
            } else {
                // Reserve the spot to prevent other threads from adding the same pod
                tracker.insert(pod_info.clone());
                true
            }
        };

        if should_add {
            info!(
                "Healthy pod found: {} (type: {:?}). Adding worker: {}",
                pod_info.name, pod_info.pod_type, worker_url
            );

            let result = if pd_mode && pod_info.pod_type.is_some() {
                // Use PD-aware worker management
                if let Some(pod_type) = &pod_info.pod_type {
                    router
                        .add_pd_worker(&worker_url, pod_type.clone(), pod_info.bootstrap_port)
                        .await
                } else {
                    Err("Pod type is None in PD mode".to_string())
                }
            } else {
                // Fallback to regular worker management
                router.add_worker(&worker_url).await
            };

            match result {
                Ok(msg) => {
                    info!("Successfully added worker: {}", msg);
                }
                Err(e) => {
                    error!("Failed to add worker {} to router: {}", worker_url, e);
                    // Remove from tracking since addition failed
                    if let Ok(mut tracker) = tracked_pods.lock() {
                        tracker.remove(pod_info);
                    }
                }
            }
        }
    }
}

async fn handle_pod_deletion(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashSet<PodInfo>>>,
    router: Arc<Router>,
    port: u16,
    pd_mode: bool,
) {
    let worker_url = pod_info.worker_url(port);

    let was_tracked = {
        let mut tracked = match tracked_pods.lock() {
            Ok(tracked) => tracked,
            Err(e) => {
                error!("Failed to acquire tracked_pods lock during deletion: {}", e);
                return;
            }
        };
        tracked.remove(pod_info)
    };

    if was_tracked {
        info!(
            "Pod deleted: {} (type: {:?}). Removing worker: {}",
            pod_info.name, pod_info.pod_type, worker_url
        );

        if pd_mode && pod_info.pod_type.is_some() {
            // Use PD-aware worker removal
            if let Some(pod_type) = &pod_info.pod_type {
                if let Err(e) = router.remove_pd_worker(&worker_url, pod_type.clone()).await {
                    error!(
                        "Failed to remove PD worker {} from router: {}",
                        worker_url, e
                    );
                }
            }
        } else {
            // Fallback to regular worker removal
            router.remove_worker(&worker_url);
        }
    } else {
        // This case might occur if a pod is deleted before it was ever marked healthy and added.
        // Or if the event is duplicated. No action needed on the router if it wasn't tracked (and thus not added).
        debug!(
            "Pod deletion event for untracked/already removed pod: {} (type: {:?}). Worker URL: {}",
            pod_info.name, pod_info.pod_type, worker_url
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::Router;
    use k8s_openapi::api::core::v1::{Pod, PodCondition, PodSpec, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::Time;
    use std::sync::RwLock;

    // Helper function to create a Pod for testing PodInfo::from_pod
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
                };
                pod_status.conditions = Some(vec![condition]);
            }
            pod.status = Some(pod_status);
        }
        pod
    }

    // Helper function to create a Pod with PD-specific labels and annotations
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
                }]),
                ..Default::default()
            }),
        }
    }

    // Helper to create a Router instance for testing event handlers
    fn create_test_router() -> Arc<Router> {
        let worker_urls = Arc::new(RwLock::new(Vec::new()));
        Arc::new(Router::Random {
            worker_urls,
            timeout_secs: 5,
            interval_secs: 1,
        })
    }

    // Helper to create a PD config for testing
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
        }
    }

    #[test]
    fn test_pod_info_should_include() {
        let config = create_pd_config();

        // Test prefill pod should be included
        let prefill_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        assert!(PodInfo::should_include(&prefill_pod, &config));

        // Test decode pod should be included
        let decode_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        assert!(PodInfo::should_include(&decode_pod, &config));

        // Test unmatched pod should not be included
        let unmatched_pod = create_pd_k8s_pod("other-pod", "10.0.0.3", "other", None);
        assert!(!PodInfo::should_include(&unmatched_pod, &config));

        // Test regular mode
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
        // Test that PodType enum has expected variants
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
        config.pd_mode = false; // Set to regular mode

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
        // Add invalid bootstrap port annotation
        pod.metadata.annotations.as_mut().unwrap().insert(
            "sglang.ai/bootstrap-port".to_string(),
            "invalid".to_string(),
        );
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert!(pod_info.bootstrap_port.is_none()); // Should be None for invalid port
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
        };
        assert!(healthy_pod.is_healthy());

        let not_ready_pod = PodInfo {
            name: "p2".into(),
            ip: "1.1.1.2".into(),
            status: "Running".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
        };
        assert!(!not_ready_pod.is_healthy());

        let not_running_pod = PodInfo {
            name: "p3".into(),
            ip: "1.1.1.3".into(),
            status: "Pending".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        assert!(!not_running_pod.is_healthy());
    }

    #[test]
    fn test_pod_info_worker_url() {
        let pod_info = PodInfo {
            name: "p1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        assert_eq!(pod_info.worker_url(8080), "http://1.2.3.4:8080");
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
        };

        let pod2 = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
        };

        let pod3 = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };

        assert_eq!(pod1, pod2);
        assert_ne!(pod1, pod3);
    }

    #[tokio::test]
    async fn test_handle_pod_event_add_unhealthy_pod() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Pending".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false
        )
        .await;

        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
        assert!(!router
            .get_worker_urls()
            .read()
            .unwrap()
            .contains(&pod_info.worker_url(port)));
    }

    #[tokio::test]
    async fn test_handle_pod_deletion_non_existing_pod() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "pod1".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        let port = 8080u16;

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false
        )
        .await;

        assert!(tracked_pods.lock().unwrap().is_empty());
        assert!(router.get_worker_urls().read().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_prefill_pod() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "prefill-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
        };
        let port = 8080u16;

        // This test validates the structure but won't actually add workers since
        // we're using a regular router instead of PD router
        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false, so it should fallback to regular handling
        )
        .await;

        // Pod should not be tracked since router.add_worker will fail for non-running server
        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_decode_pod() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "decode-pod".into(),
            ip: "1.2.3.5".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false, so it should fallback to regular handling
        )
        .await;

        // Pod should not be tracked since router.add_worker will fail for non-running server
        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_tracked_pod() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "test-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
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
            Arc::clone(&router),
            port,
            false, // pd_mode = false
        )
        .await;

        // Pod should be removed from tracking
        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_untracked_pod() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "untracked-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };
        let port = 8080u16;

        // Don't add pod to tracked set

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            true, // pd_mode = true
        )
        .await;

        // Tracked set should remain empty
        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_unified_handler_regular_mode() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "regular-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
        };
        let port = 8080u16;

        // Test that unified handler works for regular mode
        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false
        )
        .await;

        // Pod should not be tracked since router.add_worker will fail for non-running server
        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }

    #[tokio::test]
    async fn test_unified_handler_pd_mode_with_prefill() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "prefill-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
        };
        let port = 8080u16;

        // Test that unified handler works for PD mode with prefill
        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            true, // pd_mode = true
        )
        .await;

        // Pod should not be tracked since router.add_pd_worker will fail for regular router
        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }

    #[tokio::test]
    async fn test_unified_handler_deletion_with_pd_mode() {
        let router = create_test_router();
        let tracked_pods = Arc::new(Mutex::new(HashSet::new()));
        let pod_info = PodInfo {
            name: "decode-pod".into(),
            ip: "1.2.3.4".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };

        // Add pod to tracked set first
        {
            let mut tracked = tracked_pods.lock().unwrap();
            tracked.insert(pod_info.clone());
        }

        let port = 8080u16;

        // Test that unified handler works for deletion in PD mode
        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            true, // pd_mode = true
        )
        .await;

        // Pod should be removed from tracking
        assert!(!tracked_pods.lock().unwrap().contains(&pod_info));
    }
}
