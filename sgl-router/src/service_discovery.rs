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
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            enabled: false,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 80,        // Default port to connect to pods
            namespace: None, // None means watch all namespaces
        }
    }
}

/// Represents a Kubernetes pod's information used for worker management
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodInfo {
    pub name: String,
    pub ip: String,
    pub status: String,
    pub is_ready: bool,
}

impl PodInfo {
    pub fn from_pod(pod: &Pod) -> Option<Self> {
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

        Some(PodInfo {
            name,
            ip: pod_ip,
            status: pod_status,
            is_ready,
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

    // Construct label selector string from map
    let label_selector = config
        .selector
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",");

    info!(
        "Starting Kubernetes service discovery with selector: {}",
        label_selector
    );

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

        // Create an Arc for the selector map
        let selector = Arc::new(config.selector);
        let port = config.port;

        loop {
            // Create a watcher with the proper parameters according to the kube-rs API
            let watcher_config = Config::default();
            let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();

            // Clone Arcs for the closures
            let selector_clone = Arc::clone(&selector);
            let tracked_pods_clone = Arc::clone(&tracked_pods);

            // Apply label selector filter separately since we can't do it directly with the watcher anymore
            let filtered_stream = watcher_stream.filter_map(move |obj_res| {
                let selector_inner = Arc::clone(&selector_clone);

                async move {
                    match obj_res {
                        Ok(pod) => {
                            // Only process pods matching our label selector
                            if pod.metadata.labels.as_ref().map_or(false, |labels| {
                                // Check if the pod has all the labels from our selector
                                selector_inner.iter().all(|(k, v)| {
                                    labels.get(k).map_or(false, |label_value| label_value == v)
                                })
                            }) {
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

            match filtered_stream
                .try_for_each(move |pod| {
                    let tracked_pods_inner = Arc::clone(&tracked_pods_clone2);
                    let router_inner = Arc::clone(&router_clone);

                    async move {
                        if let Some(pod_info) = PodInfo::from_pod(&pod) {
                            if pod.metadata.deletion_timestamp.is_some() {
                                handle_pod_deletion(
                                    &pod_info,
                                    tracked_pods_inner,
                                    router_inner,
                                    port,
                                )
                                .await;
                            } else {
                                handle_pod_event(&pod_info, tracked_pods_inner, router_inner, port)
                                    .await;
                            }
                        }
                        Ok(())
                    }
                })
                .await
            {
                Ok(_) => {}
                Err(err) => {
                    error!("Error in Kubernetes watcher: {}", err);
                    // Wait a bit before retrying
                    time::sleep(Duration::from_secs(5)).await;
                }
            }

            // If the watcher exits for some reason, wait a bit before restarting
            warn!(
                "Kubernetes watcher exited, restarting in {} seconds",
                config.check_interval.as_secs()
            );
            time::sleep(config.check_interval).await;
        }
    });

    Ok(handle)
}

async fn handle_pod_event(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashSet<PodInfo>>>,
    router: Arc<Router>,
    port: u16,
) {
    let worker_url = pod_info.worker_url(port);

    // Check if pod is already tracked
    let already_tracked = {
        let tracker = tracked_pods.lock().unwrap();
        tracker.contains(pod_info)
    };

    // If pod is healthy and not already tracked, add it
    if pod_info.is_healthy() {
        if !already_tracked {
            info!(
                "Healthy pod found: {}. Adding worker: {}",
                pod_info.name, worker_url
            );
            match router.add_worker(&worker_url).await {
                Ok(msg) => {
                    info!("Router add_worker: {}", msg);
                    let mut tracker = tracked_pods.lock().unwrap();
                    tracker.insert(pod_info.clone());
                }
                Err(e) => error!("Failed to add worker {} to router: {}", worker_url, e),
            }
        }
    } else if already_tracked {
        // If pod was healthy before but not anymore, remove it
        handle_pod_deletion(pod_info, tracked_pods, router, port).await;
    }
}

async fn handle_pod_deletion(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashSet<PodInfo>>>,
    router: Arc<Router>,
    port: u16,
) {
    let worker_url = pod_info.worker_url(port);
    let mut tracked = tracked_pods.lock().unwrap();

    if tracked.remove(pod_info) {
        info!(
            "Pod deleted: {}. Removing worker: {}",
            pod_info.name, worker_url
        );
        router.remove_worker(&worker_url);
    } else {
        // This case might occur if a pod is deleted before it was ever marked healthy and added.
        // Or if the event is duplicated. No action needed on the router if it wasn't tracked (and thus not added).
        debug!(
            "Pod deletion event for untracked/already removed pod: {}. Worker URL: {}",
            pod_info.name, worker_url
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

    // Helper to create a Router instance for testing event handlers
    fn create_test_router() -> Arc<Router> {
        let worker_urls = Arc::new(RwLock::new(Vec::new()));
        Arc::new(Router::Random {
            worker_urls,
            timeout_secs: 5,
            interval_secs: 1,
        })
    }

    #[test]
    fn test_service_discovery_config_default() {
        let config = ServiceDiscoveryConfig::default();
        assert!(!config.enabled);
        assert!(config.selector.is_empty());
        assert_eq!(config.check_interval, Duration::from_secs(60));
        assert_eq!(config.port, 80);
        assert!(config.namespace.is_none());
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
        let pod_info = PodInfo::from_pod(&k8s_pod).unwrap();
        assert_eq!(pod_info.name, "test-pod");
        assert_eq!(pod_info.ip, "10.0.0.1");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
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
        let pod_info = PodInfo::from_pod(&k8s_pod).unwrap();
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
        let pod_info = PodInfo::from_pod(&k8s_pod).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_missing_name() {
        let k8s_pod = create_k8s_pod(None, Some("10.0.0.1"), Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_ip() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), None, Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_status_phase() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), Some("10.0.0.1"), None, Some("True"), None);
        let pod_info = PodInfo::from_pod(&k8s_pod).unwrap();
        assert_eq!(pod_info.status, "Unknown");
    }

    #[test]
    fn test_pod_info_from_pod_no_status_object() {
        let mut k8s_pod = create_k8s_pod(Some("test-pod"), None, None, None, None);
        k8s_pod.status = None;
        assert!(PodInfo::from_pod(&k8s_pod).is_none());
    }

    #[test]
    fn test_pod_info_is_healthy() {
        let healthy_pod = PodInfo {
            name: "p1".into(),
            ip: "1.1.1.1".into(),
            status: "Running".into(),
            is_ready: true,
        };
        assert!(healthy_pod.is_healthy());

        let not_ready_pod = PodInfo {
            name: "p2".into(),
            ip: "1.1.1.2".into(),
            status: "Running".into(),
            is_ready: false,
        };
        assert!(!not_ready_pod.is_healthy());

        let not_running_pod = PodInfo {
            name: "p3".into(),
            ip: "1.1.1.3".into(),
            status: "Pending".into(),
            is_ready: true,
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
        };
        assert_eq!(pod_info.worker_url(8080), "http://1.2.3.4:8080");
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
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
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
        };
        let port = 8080u16;

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
        )
        .await;

        assert!(tracked_pods.lock().unwrap().is_empty());
        assert!(router.get_worker_urls().read().unwrap().is_empty());
    }
}
