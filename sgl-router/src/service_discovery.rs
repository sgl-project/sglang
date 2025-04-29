use futures::{StreamExt, TryStreamExt};
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::watcher::{watcher, Config},
    runtime::WatchStreamExt,
    Client,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::task;
use tokio::time;
use tracing::{error, info, warn};

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
    worker_urls: Arc<RwLock<Vec<String>>>,
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
            let worker_urls_clone = Arc::clone(&worker_urls);

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
            let worker_urls_clone2 = Arc::clone(&worker_urls_clone);

            match filtered_stream
                .try_for_each(move |pod| {
                    let tracked_pods_inner = Arc::clone(&tracked_pods_clone2);
                    let worker_urls_inner = Arc::clone(&worker_urls_clone2);

                    async move {
                        if let Some(pod_info) = PodInfo::from_pod(&pod) {
                            if pod.metadata.deletion_timestamp.is_some() {
                                handle_pod_deletion(
                                    &pod_info,
                                    tracked_pods_inner,
                                    worker_urls_inner,
                                    port,
                                )
                                .await;
                            } else {
                                handle_pod_event(
                                    &pod_info,
                                    tracked_pods_inner,
                                    worker_urls_inner,
                                    port,
                                )
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
    worker_urls: Arc<RwLock<Vec<String>>>,
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
                "Adding healthy pod {} ({}) as worker",
                pod_info.name, pod_info.ip
            );

            // Add URL to worker list
            let mut urls = worker_urls.write().unwrap();
            if !urls.contains(&worker_url) {
                urls.push(worker_url.clone());
                info!("Added new worker URL: {}", worker_url);
            }

            // Track this pod
            let mut tracker = tracked_pods.lock().unwrap();
            tracker.insert(pod_info.clone());
        }
    } else if already_tracked {
        // If pod was healthy before but not anymore, remove it
        handle_pod_deletion(pod_info, tracked_pods, worker_urls, port).await;
    }
}

async fn handle_pod_deletion(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashSet<PodInfo>>>,
    worker_urls: Arc<RwLock<Vec<String>>>,
    port: u16,
) {
    let worker_url = pod_info.worker_url(port);

    // Remove the pod from our tracking
    let was_tracked = {
        let mut tracker = tracked_pods.lock().unwrap();
        tracker.remove(pod_info)
    };

    if was_tracked {
        info!(
            "Removing pod {} ({}) from workers",
            pod_info.name, pod_info.ip
        );

        // Remove URL from worker list
        let mut urls = worker_urls.write().unwrap();
        if let Some(idx) = urls.iter().position(|url| url == &worker_url) {
            urls.remove(idx);
            info!("Removed worker URL: {}", worker_url);
        }
    }
}
