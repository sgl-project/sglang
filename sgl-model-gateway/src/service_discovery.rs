use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use futures::{StreamExt, TryStreamExt};
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::{Api, ListParams},
    runtime::{
        watcher::{watcher, Config},
        WatchStreamExt,
    },
    Client,
};
use rustls;
use smg_mesh::service::{
    gossip::{NodeState, NodeStatus},
    ClusterState,
};
use tokio::{task, time};
use tracing::{debug, error, info, warn};

use crate::{
    app_context::AppContext,
    core::{steps::worker::local::find_workers_by_url, Job},
    observability::metrics::{metrics_labels, Metrics},
    protocols::worker_spec::WorkerConfigRequest,
};

#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    pub enabled: bool,
    pub selector: HashMap<String, String>,
    pub check_interval: Duration,
    /// Period of the authoritative full-LIST reconcile that repairs the
    /// worker set independently of watch health. Keep this comfortably below
    /// the pod termination-drain budget so a draining pod stops receiving new
    /// traffic early enough for its in-flight requests to complete.
    pub resync_interval: Duration,
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
    // When true (IGW mode), also discover selector pods as Regular workers alongside PD workers
    pub igw_mode: bool,
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            enabled: false,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            resync_interval: Duration::from_secs(30),
            port: 8000,
            namespace: None,
            pd_mode: false,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/ha-port".to_string(),
            igw_mode: false,
        }
    }
}

impl ServiceDiscoveryConfig {
    pub fn warn_if_misconfigured(&self) {
        if self.pd_mode && !self.igw_mode && !self.selector.is_empty() {
            warn!(
                "--selector is set in PD mode without IGW mode enabled; \
                regular worker discovery alongside PD workers requires IGW mode, \
                selector will be ignored"
            );
        }
    }
}

/// Pods this discovery loop believes are registered, keyed by
/// [`PodInfo::identity`].
///
/// A map rather than a set of `PodInfo`: every lookup here — the add path's
/// dedup, the delete path's removal, the reconcile's diff — asks "is THIS pod
/// tracked", which is a question about identity. Keying by the whole struct
/// answers a different question ("is this pod tracked in exactly this state")
/// and gets it wrong the moment a pod's status or readiness flips.
type TrackedPods = Arc<Mutex<HashMap<String, PodInfo>>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PodType {
    Prefill,
    Decode,
    Regular,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodInfo {
    pub name: String,
    /// The API server's UID for this pod, used as its identity.
    ///
    /// `None` only for hand-built `Pod` objects; every pod returned by a LIST
    /// or WATCH carries one. See [`PodInfo::identity`] for why the identity is
    /// the UID and not the name.
    pub uid: Option<String>,
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
            if config.prefill_selector.is_empty()
                && config.decode_selector.is_empty()
                && (!config.igw_mode || config.selector.is_empty())
            {
                warn!("PD mode enabled but both prefill_selector and decode_selector are empty");
                return false;
            }
            let matches_pd = Self::matches_selector(pod, &config.prefill_selector)
                || Self::matches_selector(pod, &config.decode_selector);
            // In IGW mode, also discover regular workers via the selector field
            let matches_regular = config.igw_mode
                && !config.selector.is_empty()
                && Self::matches_selector(pod, &config.selector);
            matches_pd || matches_regular
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
        let uid = pod.metadata.uid.clone();
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
            uid,
            ip: pod_ip,
            status: pod_status,
            is_ready,
            pod_type,
            bootstrap_port,
            is_router,
            mesh_port,
        })
    }

    /// Stable identity of the pod, used to key the tracked set.
    ///
    /// The UID is the only key with both properties the tracked set needs: it
    /// does not change when the pod's status or readiness flips, and it
    /// differs for a pod recreated under the same name. Neither the name nor
    /// the full struct has both — the name is shared by a replaced pod (and,
    /// under `Api::all`, by same-named pods in other namespaces), while full
    /// struct equality breaks on the first status flip.
    ///
    /// Falls back to the name when the object carries no UID, which degrades
    /// to the previous name-keyed behaviour rather than dropping the pod from
    /// discovery altogether.
    pub fn identity(&self) -> &str {
        self.uid.as_deref().unwrap_or(&self.name)
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
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));

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

        // Authoritative periodic full-LIST reconcile. The watch alone cannot
        // guarantee convergence: `.applied_objects()` drops the watcher's
        // Delete events, and environments that frequently drop long-lived
        // watch connections leave extended windows where nothing is observed.
        // This task repairs both missed adds and missed removes on a fixed
        // cadence, independent of watch health. `time::interval` fires
        // immediately, so the first reconcile also seeds the worker set at
        // startup.
        //
        // Both this timer and the watch-restart path below drive reconciles,
        // so they share state (a lock taken inside `reconcile_from_list` to
        // keep the passes from interleaving, plus the failure counter).
        let reconcile_state = Arc::new(ReconcileState::default());
        {
            let pods_resync = pods.clone();
            let config_resync = Arc::clone(&config_arc);
            let tracked_resync = Arc::clone(&tracked_pods);
            let app_context_resync = Arc::clone(&app_context);
            let state_resync = Arc::clone(&reconcile_state);
            // Clamp to >= 1s: time::interval panics on a zero period.
            let resync_period = config_arc.resync_interval.max(Duration::from_secs(1));
            tokio::spawn(async move {
                let mut ticker = time::interval(resync_period);
                // Delay, not the default Burst: if a LIST ever takes longer
                // than the period, Burst fires every missed tick with no gap,
                // so a slow API server would be answered with back-to-back
                // cluster-wide LISTs — a feedback loop that cannot recover on
                // its own. Delay keeps consecutive reconciles at least
                // `resync_period` apart no matter how slow the LIST is.
                ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
                loop {
                    ticker.tick().await;
                    reconcile_from_list(
                        &pods_resync,
                        &config_resync,
                        Arc::clone(&tracked_resync),
                        Arc::clone(&app_context_resync),
                        config_resync.port,
                        &state_resync,
                    )
                    .await;
                }
            });
        }

        const INITIAL_RETRY_DELAY: Duration = Duration::from_secs(1);
        const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);
        // A watch session that lasted at least this long counts as healthy, so
        // the drop that ended it is treated as a fresh incident rather than a
        // continuation of an earlier one.
        const HEALTHY_STREAM_THRESHOLD: Duration = Duration::from_secs(60);

        let mut retry_delay = INITIAL_RETRY_DELAY;

        loop {
            // Reconcile against a fresh LIST before (re)starting the watch so
            // a reconnect immediately repairs the adds/removes missed during
            // the outage, rather than waiting for the next periodic tick.
            reconcile_from_list(
                &pods,
                &config_arc,
                Arc::clone(&tracked_pods),
                Arc::clone(&app_context),
                port,
                &reconcile_state,
            )
            .await;

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

            let stream_started = time::Instant::now();
            let watch_result = filtered_stream
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
                .await;

            // `watcher()` is an infinite stream — it recovers internally and
            // surfaces failures as Err items — so `try_for_each` returns only
            // by aborting on an error. A clean end is not expected, but it is
            // handled on the same path so the restart is never a hot loop if
            // that ever changes.
            match watch_result {
                Ok(()) => warn!("Kubernetes watcher stream ended unexpectedly"),
                Err(err) => error!("Error in Kubernetes watcher: {}", err),
            }

            // Reset the backoff after a session that ran long enough to be
            // considered healthy. Without this, retry_delay only ever grows:
            // in environments that drop long-lived watches every few minutes,
            // a handful of unrelated drops ratchets the reconnect interval to
            // MAX_RETRY_DELAY and pins it there for the lifetime of the
            // process, leaving the periodic reconcile as the only thing still
            // observing the cluster.
            if stream_started.elapsed() >= HEALTHY_STREAM_THRESHOLD {
                retry_delay = INITIAL_RETRY_DELAY;
            }

            warn!("Restarting Kubernetes watcher in {:?}", retry_delay);
            time::sleep(retry_delay).await;
            retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
        }
    });

    Ok(handle)
}

async fn handle_pod_event(
    pod_info: &PodInfo,
    tracked_pods: TrackedPods,
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

            if tracker.contains_key(pod_info.identity()) {
                (false, tracker.len())
            } else {
                tracker.insert(pod_info.identity().to_string(), pod_info.clone());
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
                            tracker.remove(pod_info.identity());
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

/// Deregister a pod's worker, resolving the tracked entry by identity.
///
/// `pod_info` only has to identify the pod: the URL that gets deregistered
/// comes from the STORED entry, which is the one that was registered. The
/// caller may therefore pass a freshly parsed `PodInfo` — as the watch path
/// does, where the pod's readiness has usually already flipped to false by the
/// time the terminating object arrives.
async fn handle_pod_deletion(
    pod_info: &PodInfo,
    tracked_pods: TrackedPods,
    app_context: Arc<AppContext>,
    port: u16,
) {
    // Remove pod and get remaining count in single lock acquisition
    let (stored, remaining_count) = {
        let mut tracked = match tracked_pods.lock() {
            Ok(tracked) => tracked,
            Err(e) => {
                error!("Failed to acquire tracked_pods lock during deletion: {}", e);
                return;
            }
        };
        let removed = tracked.remove(pod_info.identity());
        (removed, tracked.len())
    };

    if let Some(stored) = stored {
        let worker_url = stored.worker_url(port);
        info!(
            "Removing pod: {} | type: {:?} | url: {}",
            stored.name, stored.pod_type, worker_url
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
            pod_info.name,
            pod_info.pod_type,
            pod_info.worker_url(port)
        );
    }
}

/// Page size for the reconcile's LIST, matching the watcher's own initial
/// list (`watcher::Config::default().page_size`) and client-go's default.
const LIST_PAGE_SIZE: u32 = 500;

/// Consecutive failed resyncs after which the log is escalated to `error!`.
/// A single failed LIST is unremarkable; a run of them means the worker set
/// is frozen and may be routing to pods that no longer exist.
const RESYNC_FAILURES_BEFORE_ESCALATION: u32 = 3;

/// State shared by the two reconcile drivers (the periodic timer and the
/// watch-restart path).
#[derive(Default)]
struct ReconcileState {
    /// Held for the duration of a pass so the two drivers never interleave.
    lock: tokio::sync::Mutex<()>,
    /// Consecutive failed passes, used only to escalate the log level.
    consecutive_failures: AtomicU32,
}

/// Full LIST + reconcile of the tracked worker set against the API server.
///
/// Adds any healthy, selector-matching pod not yet tracked, and removes any
/// tracked pod that has disappeared from the API server or begun terminating
/// (mirroring the watch path's deletion_timestamp handling). This is the
/// authoritative resync that makes worker correctness independent of
/// watch-stream reliability: `.applied_objects()` drops Delete events and
/// unstable environments drop the long-lived watch, so the watch alone
/// cannot converge.
///
/// The LIST is paged; any page failing aborts the whole pass and leaves the
/// worker set untouched, so a partial view can never drive removals.
async fn reconcile_from_list(
    pods: &Api<Pod>,
    config: &ServiceDiscoveryConfig,
    tracked_pods: TrackedPods,
    app_context: Arc<AppContext>,
    port: u16,
    state: &ReconcileState,
) {
    // One reconcile at a time. The periodic timer and the watch-restart path
    // both call this, and letting two passes interleave lets the one holding
    // the older LIST snapshot delete a pod the newer one has just added.
    let _guard = state.lock.lock().await;
    let started = time::Instant::now();

    // Snapshot the tracked set BEFORE issuing the LIST; only pods that were
    // already tracked at that instant are eligible for removal below. The
    // watch runs concurrently, so a pod it registers while the LIST is in
    // flight is legitimately missing from the response through no fault of
    // its own. Removing it would open exactly the zero-worker window this
    // reconcile exists to close, and the watch would not re-emit an Apply to
    // repair it — the pod would stay unregistered until the next resync.
    let tracked_before: HashMap<String, PodInfo> = match tracked_pods.lock() {
        Ok(tracker) => tracker.clone(),
        Err(e) => {
            error!("SD resync: failed to lock tracked_pods: {}", e);
            return;
        }
    };

    // Identities currently present in the API for matching pods, and
    // add-missing.
    // A pod with a deletion_timestamp is deliberately NOT counted as present:
    // the watch path deregisters a pod as soon as it starts terminating, and
    // the reconcile must mirror that semantic watch-independently. If it
    // waited for the pod to disappear from the API instead, a draining pod
    // would keep receiving new traffic whenever the watch is down — which
    // both fails those requests once the pod finishes shutting down and can
    // prevent a graceful-drain hook from ever reaching zero in-flight.
    // handle_pod_deletion is idempotent (removes only if tracked), so the
    // watch and the reconcile processing the same termination never conflict.
    let mut present: HashSet<String> = HashSet::new();

    // Worker URLs claimed by a live, healthy pod in this LIST. Used to tell a
    // same-URL replacement apart from a pod that is simply gone; see the
    // removal pass below.
    let mut claimed_urls: HashSet<String> = HashSet::new();

    let dp_aware = app_context.router_config.dp_aware;

    // Page through the LIST rather than pulling the whole collection in one
    // response. With no namespace configured this is an Api::all over every
    // pod in the cluster, once per resync per router replica, so an unbounded
    // list would hand the API server a multi-megabyte response on a large
    // cluster; the watch's own initial list already pages at the same size.
    // Chunked list is still a consistent snapshot: the server pins the
    // resourceVersion to the continue token.
    //
    // Only `present` is accumulated across pages — pods are processed and
    // dropped page by page, so peak memory is one page rather than the whole
    // cluster.
    //
    // Same client-side filtering as the watch (Config::default() => no server
    // label filter); we match via should_include below.
    let mut params = ListParams::default().limit(LIST_PAGE_SIZE);
    loop {
        let page = match pods.list(&params).await {
            Ok(page) => page,
            Err(e) => {
                // Bail before the removal pass. Adds already applied are
                // harmless (those pods really do exist), but `present` is now
                // partial and removing against it would deregister live
                // workers. An expired continue token (410) lands here too and
                // is simply retried by the next resync.
                Metrics::record_discovery_sync(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    metrics_labels::RESULT_ERROR,
                );
                let failures = state.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= RESYNC_FAILURES_BEFORE_ESCALATION {
                    // The gauge cannot show this: it just stops moving, so a
                    // frozen worker set looks identical to a quiet cluster.
                    error!(
                        "SD resync LIST has failed {} times in a row; the worker set is \
                         frozen and may still route to pods that no longer exist: {}",
                        failures, e
                    );
                } else {
                    warn!(
                        "SD resync LIST failed ({} in a row), keeping current worker set: {}",
                        failures, e
                    );
                }
                return;
            }
        };

        let continue_token = page.metadata.continue_.clone();

        for pod in &page.items {
            if !PodInfo::should_include(pod, config) {
                continue;
            }
            if pod.metadata.deletion_timestamp.is_some() {
                continue;
            }
            // Read the identity off the raw object so a pod that is present
            // but not yet parseable into a PodInfo (no IP assigned) still
            // counts as present and cannot be removed below.
            if let Some(identity) = pod_identity(pod) {
                present.insert(identity);
            }
            if let Some(pod_info) = PodInfo::from_pod(pod, Some(config)) {
                if pod_info.is_healthy() {
                    claimed_urls.insert(pod_info.worker_url(port));
                    forget_lost_registration(
                        &pod_info,
                        &tracked_pods,
                        &app_context,
                        port,
                        dp_aware,
                    );
                }
                // handle_pod_event dedups via tracked_pods and gates on
                // is_healthy(), so re-adds are cheap no-ops.
                handle_pod_event(
                    &pod_info,
                    Arc::clone(&tracked_pods),
                    Arc::clone(&app_context),
                    port,
                    config.pd_mode,
                )
                .await;
            }
        }

        match continue_token {
            Some(token) if !token.is_empty() => params = params.continue_token(&token),
            _ => break,
        }
    }

    // Every page landed, so `present` is a complete view and the removal pass
    // below is safe to run.
    state.consecutive_failures.store(0, Ordering::Relaxed);
    Metrics::record_discovery_sync(
        metrics_labels::DISCOVERY_KUBERNETES,
        metrics_labels::RESULT_SUCCESS,
    );
    Metrics::record_discovery_sync_duration(
        metrics_labels::DISCOVERY_KUBERNETES,
        started.elapsed(),
    );

    // Remove-absent: any pod tracked before the LIST whose identity is no
    // longer in the API. handle_pod_deletion re-checks membership under the
    // lock, so a pod the watch removed in the meantime is a no-op rather than
    // a duplicate deregistration.
    for pod_info in stale_tracked_pods(&tracked_before, &present) {
        // A departed pod whose worker URL is now claimed by a live, healthy
        // pod is a same-URL replacement, not a worker to deregister: with
        // hostNetwork a pod's IP is its node's, so a rollout on that node
        // brings up a new pod (new UID) at the address the old one had. The
        // replacement's AddWorker in the pass above has already re-registered
        // that URL — the registry upserts by URL — so submitting a removal for
        // it here would tear the replacement straight back out and leave the
        // model with no worker at all. Forget the old entry and leave the
        // registry alone.
        if claimed_urls.contains(&pod_info.worker_url(port)) {
            match tracked_pods.lock() {
                Ok(mut tracker) => {
                    if tracker.remove(pod_info.identity()).is_some() {
                        info!(
                            "Replaced pod: {} | url: {} is now served by a live pod",
                            pod_info.name,
                            pod_info.worker_url(port)
                        );
                    }
                }
                Err(e) => error!("SD resync: failed to lock tracked_pods: {}", e),
            }
            continue;
        }

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&app_context),
            port,
        )
        .await;
    }
}

/// Identity of a raw API object, matching [`PodInfo::identity`].
fn pod_identity(pod: &Pod) -> Option<String> {
    pod.metadata
        .uid
        .clone()
        .or_else(|| pod.metadata.name.clone())
}

/// Drop a tracked pod whose worker never reached the registry, so the add path
/// stops treating it as registered and submits it again.
///
/// Membership in `tracked_pods` records that an AddWorker job was SUBMITTED,
/// not that it succeeded: `handle_pod_event` inserts before submitting and
/// rolls back only when `submit()` itself fails. A job that fails after that —
/// a `detect_connection_mode` exhausting `worker_startup_timeout_secs` (30
/// minutes by default), a failing DP discovery, a workflow that never starts —
/// only records a failed `JobStatus`, and nothing untracks the pod. Because the
/// add path short-circuits on tracked pods, the pod then stays absent from the
/// registry for the life of the process while the reconcile believes it is
/// registered. That is the same unrecoverable state this reconcile closes on
/// the delete side; the pass is already paying for a full LIST, so it is the
/// natural place to close it on the add side too.
fn forget_lost_registration(
    pod_info: &PodInfo,
    tracked_pods: &TrackedPods,
    app_context: &Arc<AppContext>,
    port: u16,
    dp_aware: bool,
) {
    // Only pods this loop believes it already registered are candidates; a
    // pod that is not tracked is about to be added by the caller anyway.
    match tracked_pods.lock() {
        Ok(tracker) => {
            if !tracker.contains_key(pod_info.identity()) {
                return;
            }
        }
        Err(e) => {
            error!("SD resync: failed to lock tracked_pods: {}", e);
            return;
        }
    }

    // Reconcile against the registry, not just the shadow set. In dp-aware
    // mode workers are registered as `<url>@<rank>`, so an exact-URL lookup
    // never matches one and every dp-aware pod would look unregistered.
    let worker_url = pod_info.worker_url(port);
    if !find_workers_by_url(&app_context.worker_registry, &worker_url, dp_aware).is_empty() {
        return;
    }

    // An add that is still queued or running has not failed yet. `submit()`
    // does not dedup by URL, so without this the reconcile would pile up a
    // fresh AddWorker on every tick for the whole 30-minute startup window.
    if let Some(job_queue) = app_context.worker_job_queue.get() {
        if job_queue.has_add_worker_in_flight_for(&worker_url) {
            return;
        }
    }

    match tracked_pods.lock() {
        Ok(mut tracker) => {
            if tracker.remove(pod_info.identity()).is_some() {
                warn!(
                    "SD resync: pod {} is tracked but no worker is registered at {};                      its registration must have failed after being submitted, re-submitting",
                    pod_info.name, worker_url
                );
            }
        }
        Err(e) => error!("SD resync: failed to lock tracked_pods: {}", e),
    }
}

/// Tracked pods whose identity is absent from a fresh LIST of the API server.
///
/// Selection is by [`PodInfo::identity`] — never by name and never by full
/// `PodInfo` equality. Full equality is too strict: a tracked pod whose
/// status or readiness has flipped since it was added must still be treated as
/// present. The name is too loose in the other direction, and both of its
/// failures leave a dead worker registered for the life of the process, since
/// the registry's health checker only marks workers unhealthy and never
/// deregisters them:
///
/// - under `Api::all` (the default — no `--service-discovery-namespace`) pod
///   names are unique only per namespace, so deleting `ns-a/engine-0` while
///   `ns-b/engine-0` lives leaves the name present and the dead pod tracked;
/// - a pod replaced under the same name (StatefulSet, LeaderWorkerSet, or
///   force-delete and recreate) keeps the name present, so the replacement is
///   tracked alongside the pod it replaced instead of evicting it.
///
/// The returned values are the stored `PodInfo` structs, which carry the IP
/// the worker was actually registered under.
fn stale_tracked_pods(
    tracked: &HashMap<String, PodInfo>,
    present: &HashSet<String>,
) -> Vec<PodInfo> {
    tracked
        .iter()
        .filter(|(identity, _)| !present.contains(*identity))
        .map(|(_, pod_info)| pod_info.clone())
        .collect()
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
        create_test_app_context_with_dp_aware(false).await
    }

    async fn create_test_app_context_with_dp_aware(dp_aware: bool) -> Arc<AppContext> {
        use crate::{
            config::RouterConfig, core::WorkerService, middleware::TokenBucket,
            observability::inflight_tracker::InFlightRequestTracker,
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .dp_aware(dp_aware)
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
            response_storage: Arc::new(data_connector::MemoryResponseStorage::new()),
            conversation_storage: Arc::new(data_connector::MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(
                data_connector::MemoryConversationItemStorage::new(),
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
            resync_interval: Duration::from_secs(30),
            port: 8080,
            namespace: None,
            pd_mode: true,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/ha-port".to_string(),
            igw_mode: false,
        }
    }

    fn create_regular_k8s_pod(name: &str, ip: &str) -> Pod {
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("app".to_string(), "regular-worker".to_string());

        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                labels: Some(labels),
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
    fn test_should_include_regular_pod_in_pd_igw_mode() {
        let mut config = create_pd_config();
        config.igw_mode = true;
        config
            .selector
            .insert("app".to_string(), "regular-worker".to_string());

        let regular_pod = create_regular_k8s_pod("regular-pod", "10.0.0.5");
        assert!(PodInfo::should_include(&regular_pod, &config));

        let pod_info = PodInfo::from_pod(&regular_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
    }

    #[test]
    fn test_should_exclude_regular_pod_in_pd_mode_without_igw() {
        let mut config = create_pd_config();
        config.igw_mode = false;
        config
            .selector
            .insert("app".to_string(), "regular-worker".to_string());

        let regular_pod = create_regular_k8s_pod("regular-pod", "10.0.0.5");
        assert!(!PodInfo::should_include(&regular_pod, &config));
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
            uid: None,
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
            uid: None,
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
            uid: None,
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
            uid: None,
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
            uid: None,
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
            uid: None,
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
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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

        assert!(!is_tracked(&tracked_pods, &pod_info));
    }

    #[tokio::test]
    async fn test_handle_pod_deletion_non_existing_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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
        assert!(is_tracked(&tracked_pods, &pod_info));

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_decode_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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
        assert!(is_tracked(&tracked_pods, &pod_info));

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_tracked_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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
            tracked.insert(pod_info.identity().to_string(), pod_info.clone());
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
        assert!(!is_tracked(&tracked_pods, &pod_info));
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_untracked_pod() {
        let app_context = create_test_app_context().await;
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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
        assert!(is_tracked(&tracked_pods, &pod_info));

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_unified_handler_pd_mode_with_prefill() {
        let app_context = create_test_app_context().await;
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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
        assert!(is_tracked(&tracked_pods, &pod_info));

        // Note: In tests with uninitialized queue, background jobs don't process
        // Worker won't appear in registry until background job runs (in production)
    }

    #[tokio::test]
    async fn test_unified_handler_deletion_with_pd_mode() {
        let app_context = create_test_app_context().await;
        let tracked_pods: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            uid: None,
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
            tracked.insert(pod_info.identity().to_string(), pod_info.clone());
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
        assert!(!is_tracked(&tracked_pods, &pod_info));
    }

    #[test]
    fn test_should_include_mixed_pd_igw_regular_pod_included() {
        let mut regular_selector = HashMap::new();
        regular_selector.insert("app".to_string(), "regular-worker".to_string());

        let mut prefill_selector = HashMap::new();
        prefill_selector.insert("app".to_string(), "sglang".to_string());
        prefill_selector.insert("component".to_string(), "prefill".to_string());

        let mut decode_selector = HashMap::new();
        decode_selector.insert("app".to_string(), "sglang".to_string());
        decode_selector.insert("component".to_string(), "decode".to_string());

        let config = ServiceDiscoveryConfig {
            enabled: true,
            selector: regular_selector,
            check_interval: Duration::from_secs(60),
            resync_interval: Duration::from_secs(30),
            port: 8080,
            namespace: None,
            pd_mode: true,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/ha-port".to_string(),
            igw_mode: true,
        };

        let regular_pod = create_regular_k8s_pod("regular-pod", "10.0.1.1");
        assert!(PodInfo::should_include(&regular_pod, &config));

        let pod_info = PodInfo::from_pod(&regular_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
    }

    #[test]
    fn test_should_include_mixed_pd_no_igw_regular_pod_excluded() {
        let mut regular_selector = HashMap::new();
        regular_selector.insert("app".to_string(), "regular-worker".to_string());

        let mut prefill_selector = HashMap::new();
        prefill_selector.insert("app".to_string(), "sglang".to_string());
        prefill_selector.insert("component".to_string(), "prefill".to_string());

        let mut decode_selector = HashMap::new();
        decode_selector.insert("app".to_string(), "sglang".to_string());
        decode_selector.insert("component".to_string(), "decode".to_string());

        let config = ServiceDiscoveryConfig {
            enabled: true,
            selector: regular_selector,
            check_interval: Duration::from_secs(60),
            resync_interval: Duration::from_secs(30),
            port: 8080,
            namespace: None,
            pd_mode: true,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/ha-port".to_string(),
            igw_mode: false,
        };

        let regular_pod = create_regular_k8s_pod("regular-pod", "10.0.1.1");
        assert!(!PodInfo::should_include(&regular_pod, &config));
    }

    /// A pod carrying an explicit UID and namespace, as the API server
    /// returns them. `create_regular_k8s_pod` deliberately leaves both unset,
    /// which exercises the name fallback instead.
    fn create_identified_k8s_pod(name: &str, uid: &str, ip: &str, namespace: &str) -> Pod {
        let mut pod = create_regular_k8s_pod(name, ip);
        pod.metadata.uid = Some(uid.to_string());
        pod.metadata.namespace = Some(namespace.to_string());
        pod
    }

    /// The `PodInfo` the reconcile would have stored for such a pod.
    fn identified_pod_info(name: &str, uid: &str, ip: &str) -> PodInfo {
        PodInfo {
            uid: Some(uid.to_string()),
            ..tracked_pod_info(name, ip, "Running", true)
        }
    }

    /// An `AppContext` whose job queue accepts submissions but never runs
    /// them: `max_concurrent_jobs: 0` leaves the dispatcher parked on the
    /// semaphore, so every submitted job stays observable in the status map.
    /// This is what lets a test tell "a job was submitted" apart from "no job
    /// was submitted" — the shared `create_test_app_context` leaves the queue
    /// uninitialized, where submissions vanish.
    async fn app_context_with_parked_queue(dp_aware: bool) -> Arc<AppContext> {
        let app_context = create_test_app_context_with_dp_aware(dp_aware).await;
        let queue = crate::core::JobQueue::new(
            crate::core::job_queue::JobQueueConfig {
                queue_capacity: 64,
                max_concurrent_jobs: 0,
            },
            Arc::downgrade(&app_context),
        );
        let _ = app_context.worker_job_queue.set(queue);
        app_context
    }

    fn submitted_job(app_context: &Arc<AppContext>, worker_url: &str) -> Option<String> {
        app_context
            .worker_job_queue
            .get()
            .expect("queue must be initialized")
            .get_status(worker_url)
            .map(|status| status.job_type.to_string())
    }

    fn tracked_identities(tracked: &TrackedPods) -> HashSet<String> {
        tracked.lock().unwrap().keys().cloned().collect()
    }

    /// The tracked set, seeded with `pods` keyed by identity.
    fn tracked_from(pods: &[PodInfo]) -> TrackedPods {
        Arc::new(Mutex::new(tracked_map(pods)))
    }

    fn tracked_map(pods: &[PodInfo]) -> HashMap<String, PodInfo> {
        pods.iter()
            .map(|pod_info| (pod_info.identity().to_string(), pod_info.clone()))
            .collect()
    }

    fn is_tracked(tracked: &TrackedPods, pod_info: &PodInfo) -> bool {
        tracked.lock().unwrap().contains_key(pod_info.identity())
    }

    fn tracked_pod_info(name: &str, ip: &str, status: &str, is_ready: bool) -> PodInfo {
        PodInfo {
            uid: None,
            name: name.to_string(),
            ip: ip.to_string(),
            status: status.to_string(),
            is_ready,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
            is_router: false,
            mesh_port: None,
        }
    }

    #[test]
    fn test_stale_tracked_pods_removes_only_absent_names() {
        let tracked = tracked_map(&[
            tracked_pod_info("pod-a", "10.0.0.1", "Running", true),
            tracked_pod_info("pod-b", "10.0.0.2", "Running", true),
            tracked_pod_info("pod-c", "10.0.0.3", "Running", true),
        ]);

        let mut present = HashSet::new();
        present.insert("pod-a".to_string());
        present.insert("pod-c".to_string());

        let stale = stale_tracked_pods(&tracked, &present);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].name, "pod-b");
    }

    #[test]
    fn test_stale_tracked_pods_keeps_pod_with_changed_status() {
        // The stored PodInfo was captured while the pod was Ready/Running; the
        // live pod has since flipped (e.g. is_ready=false). Presence is keyed
        // by name, so the tracked entry must NOT be selected for removal —
        // full-struct equality would wrongly treat it as gone.
        let tracked = tracked_map(&[tracked_pod_info("pod-a", "10.0.0.1", "Running", true)]);

        let mut present = HashSet::new();
        present.insert("pod-a".to_string());

        let stale = stale_tracked_pods(&tracked, &present);
        assert!(stale.is_empty());
    }

    #[test]
    fn test_stale_tracked_pods_empty_list_removes_everything() {
        // A fresh LIST that returns no matching pods must drain the whole
        // tracked set (e.g. all workers were deleted while the watch was
        // down).
        let tracked = tracked_map(&[
            tracked_pod_info("pod-a", "10.0.0.1", "Running", true),
            tracked_pod_info("pod-b", "10.0.0.2", "Pending", false),
        ]);

        let stale = stale_tracked_pods(&tracked, &HashSet::new());
        assert_eq!(stale.len(), 2);
    }

    #[test]
    fn test_stale_tracked_pods_returns_stored_struct() {
        // The returned PodInfo must be the STORED struct (not a re-parsed
        // one) so the caller's exact-struct HashSet::remove always matches.
        let stored = tracked_pod_info("pod-gone", "10.0.0.9", "Running", true);
        let tracked = tracked_map(std::slice::from_ref(&stored));

        let stale = stale_tracked_pods(&tracked, &HashSet::new());
        assert_eq!(stale, vec![stored]);
    }

    // ---- reconcile_from_list ----
    //
    // These drive the real `reconcile_from_list` against a fake API server
    // built from a `tower::service_fn`, which `kube::Client::new` accepts
    // directly — no cluster and no extra dependencies.
    //
    // Two traps worth knowing if you extend these: a `k8s_openapi::List<Pod>`
    // serialized with serde_json does not round-trip into kube's `ObjectList`
    // (the items come back empty), so the body is assembled as explicit
    // PodList JSON; and the config selector must actually match the pod
    // labels, or `should_include` filters everything out and the test
    // silently asserts nothing.

    fn reconcile_test_config() -> ServiceDiscoveryConfig {
        // Matches the labels set by `create_regular_k8s_pod`.
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "regular-worker".to_string());
        ServiceDiscoveryConfig {
            enabled: true,
            selector,
            ..Default::default()
        }
    }

    fn pod_list_json(pods: &[Pod], continue_token: Option<&str>) -> String {
        let mut metadata = serde_json::json!({ "resourceVersion": "1" });
        if let Some(token) = continue_token {
            metadata["continue"] = serde_json::Value::String(token.to_string());
        }
        serde_json::json!({
            "apiVersion": "v1",
            "kind": "PodList",
            "metadata": metadata,
            "items": pods
                .iter()
                .map(|pod| serde_json::to_value(pod).unwrap())
                .collect::<Vec<_>>(),
        })
        .to_string()
    }

    fn json_response(
        status: u16,
        body: String,
    ) -> http::Response<http_body_util::Full<bytes::Bytes>> {
        http::Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(http_body_util::Full::new(bytes::Bytes::from(body)))
            .unwrap()
    }

    /// Fake `Api<Pod>` that replays `responses` in order and records the
    /// request URIs it was asked for.
    fn fake_pod_api(responses: Vec<(u16, String)>) -> (Api<Pod>, Arc<Mutex<Vec<String>>>) {
        let pending = Arc::new(Mutex::new(std::collections::VecDeque::from(responses)));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_for_service = Arc::clone(&seen);

        let service = tower::service_fn(move |req: http::Request<kube::client::Body>| {
            let pending = Arc::clone(&pending);
            let seen = Arc::clone(&seen_for_service);
            async move {
                seen.lock().unwrap().push(req.uri().to_string());
                let (status, body) = pending
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("fake API server got more requests than it had responses");
                Ok::<_, std::convert::Infallible>(json_response(status, body))
            }
        });

        (Api::all(Client::new(service, "default")), seen)
    }

    fn tracked_names(tracked: &TrackedPods) -> HashSet<String> {
        tracked
            .lock()
            .unwrap()
            .values()
            .map(|pod_info| pod_info.name.clone())
            .collect()
    }

    #[tokio::test]
    async fn test_reconcile_adds_listed_pods_and_removes_absent_ones() {
        let tracked = tracked_from(&[tracked_pod_info("pod-gone", "10.0.0.1", "Running", true)]);

        let live = create_regular_k8s_pod("pod-live", "10.0.0.2");
        let (api, _) = fake_pod_api(vec![(200, pod_list_json(&[live], None))]);

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            create_test_app_context().await,
            8000,
            &ReconcileState::default(),
        )
        .await;

        let names = tracked_names(&tracked);
        assert!(
            names.contains("pod-live"),
            "a listed healthy pod should be tracked"
        );
        assert!(
            !names.contains("pod-gone"),
            "a tracked pod absent from the LIST should be removed"
        );
    }

    #[tokio::test]
    async fn test_reconcile_treats_terminating_pod_as_absent() {
        // The pod is still in the API but has begun terminating. It must be
        // deregistered now rather than when it finally disappears, so it stops
        // receiving new traffic while its drain hook runs.
        let tracked = tracked_from(&[tracked_pod_info(
            "pod-draining",
            "10.0.0.3",
            "Running",
            true,
        )]);

        let mut draining = create_regular_k8s_pod("pod-draining", "10.0.0.3");
        draining.metadata.deletion_timestamp = Some(Time(chrono::Utc::now()));
        let (api, _) = fake_pod_api(vec![(200, pod_list_json(&[draining], None))]);

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            create_test_app_context().await,
            8000,
            &ReconcileState::default(),
        )
        .await;

        assert!(
            !tracked_names(&tracked).contains("pod-draining"),
            "a terminating pod should be treated as absent"
        );
    }

    #[tokio::test]
    async fn test_reconcile_keeps_worker_set_when_list_fails() {
        // The most consequential branch: a failed LIST must never be read as
        // "no pods exist" and drain every worker.
        let tracked = tracked_from(&[tracked_pod_info("pod-a", "10.0.0.1", "Running", true)]);

        let forbidden = serde_json::json!({
            "kind": "Status",
            "apiVersion": "v1",
            "status": "Failure",
            "message": "pods is forbidden",
            "reason": "Forbidden",
            "code": 403,
        })
        .to_string();
        let (api, _) = fake_pod_api(vec![(403, forbidden)]);

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            create_test_app_context().await,
            8000,
            &ReconcileState::default(),
        )
        .await;

        assert!(
            tracked_names(&tracked).contains("pod-a"),
            "a failed LIST must leave the worker set untouched"
        );
    }

    #[tokio::test]
    async fn test_reconcile_does_not_remove_pod_registered_during_the_list() {
        // The watch registers a pod while the LIST is in flight. It cannot
        // appear in that response through no fault of its own, and the watch
        // will not re-emit an Apply for it, so removing it here would strand
        // the worker until the next resync.
        let tracked: TrackedPods = Arc::new(Mutex::new(HashMap::new()));
        let tracked_for_service = Arc::clone(&tracked);

        let service = tower::service_fn(move |_req: http::Request<kube::client::Body>| {
            let tracked = Arc::clone(&tracked_for_service);
            async move {
                let late = tracked_pod_info("late-pod", "10.0.0.9", "Running", true);
                tracked
                    .lock()
                    .unwrap()
                    .insert(late.identity().to_string(), late);
                Ok::<_, std::convert::Infallible>(json_response(200, pod_list_json(&[], None)))
            }
        });
        let api: Api<Pod> = Api::all(Client::new(service, "default"));

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            create_test_app_context().await,
            8000,
            &ReconcileState::default(),
        )
        .await;

        assert!(
            tracked_names(&tracked).contains("late-pod"),
            "a pod registered while the LIST was in flight must survive the reconcile"
        );
    }

    #[tokio::test]
    async fn test_reconcile_follows_continue_token_across_pages() {
        let tracked = tracked_from(&[tracked_pod_info("pod-gone", "10.0.0.1", "Running", true)]);

        let first = create_regular_k8s_pod("pod-page1", "10.0.0.2");
        let second = create_regular_k8s_pod("pod-page2", "10.0.0.3");
        let (api, seen) = fake_pod_api(vec![
            (200, pod_list_json(&[first], Some("next-page-token"))),
            (200, pod_list_json(&[second], None)),
        ]);

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            create_test_app_context().await,
            8000,
            &ReconcileState::default(),
        )
        .await;

        let names = tracked_names(&tracked);
        assert!(names.contains("pod-page1"), "first page should be applied");
        assert!(names.contains("pod-page2"), "second page should be applied");
        assert!(
            !names.contains("pod-gone"),
            "removal should consider every page, not just the last"
        );

        let requests = seen.lock().unwrap().clone();
        assert_eq!(requests.len(), 2, "expected exactly two pages");
        assert!(
            requests[0].contains(&format!("limit={}", LIST_PAGE_SIZE)),
            "first page should be bounded, got {}",
            requests[0]
        );
        assert!(
            requests[1].contains("continue=next-page-token"),
            "second page should carry the continue token, got {}",
            requests[1]
        );
    }

    // ---- identity: replaced and same-named pods ----

    #[tokio::test]
    async fn test_reconcile_removes_pod_replaced_under_the_same_name() {
        // A pod recreated under the same name with a new IP: StatefulSet or
        // LeaderWorkerSet rollout, or force-delete plus recreate. Keyed by
        // name the replacement is added alongside the pod it replaced and the
        // dead entry is never removed, leaving that worker registered for the
        // life of the process.
        let replaced = identified_pod_info("engine-0", "uid-a", "10.0.0.1");
        let tracked = tracked_from(&[replaced]);

        let replacement = create_identified_k8s_pod("engine-0", "uid-b", "10.0.0.2", "ns-a");
        let (api, _) = fake_pod_api(vec![(200, pod_list_json(&[replacement], None))]);

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            create_test_app_context().await,
            8000,
            &ReconcileState::default(),
        )
        .await;

        let identities = tracked_identities(&tracked);
        assert!(
            identities.contains("uid-b"),
            "the replacement pod must be tracked"
        );
        assert!(
            !identities.contains("uid-a"),
            "the pod it replaced must be removed, not kept because the name is reused"
        );
    }

    #[tokio::test]
    async fn test_reconcile_removes_dead_pod_whose_name_lives_in_another_namespace() {
        // With no --service-discovery-namespace the API is Api::all, where pod
        // names are unique only per namespace — the configuration
        // gateway-cluster-scoped.yaml ships. `ns-a/engine-0` is gone but
        // `ns-b/engine-0` is alive, so a name key reports the dead pod as
        // still present and nothing ever deregisters it: the registry's health
        // checker only marks workers unhealthy.
        let tracked = tracked_from(&[
            identified_pod_info("engine-0", "uid-ns-a", "10.0.0.1"),
            identified_pod_info("engine-0", "uid-ns-b", "10.0.0.2"),
        ]);

        let survivor = create_identified_k8s_pod("engine-0", "uid-ns-b", "10.0.0.2", "ns-b");
        let (api, _) = fake_pod_api(vec![(200, pod_list_json(&[survivor], None))]);

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            create_test_app_context().await,
            8000,
            &ReconcileState::default(),
        )
        .await;

        let identities = tracked_identities(&tracked);
        assert!(
            identities.contains("uid-ns-b"),
            "the surviving pod must stay tracked"
        );
        assert!(
            !identities.contains("uid-ns-a"),
            "a same-named pod in another namespace must not keep a dead pod registered"
        );
    }

    #[tokio::test]
    async fn test_reconcile_hands_over_a_shared_url_without_deregistering_it() {
        // hostNetwork makes a pod's IP its node's, so a rollout on that node
        // brings up a new pod (new UID) at the address the old one had. Both
        // map to one worker URL, and the registry upserts by URL, so the
        // replacement's add has already taken that slot by the time the old
        // entry is found stale. Submitting a removal for it would tear the
        // replacement straight back out and leave the model with no worker.
        let replaced = identified_pod_info("engine-0", "uid-a", "10.0.0.5");
        let tracked = tracked_from(std::slice::from_ref(&replaced));

        let replacement = create_identified_k8s_pod("engine-0", "uid-b", "10.0.0.5", "ns-a");
        let (api, _) = fake_pod_api(vec![(200, pod_list_json(&[replacement], None))]);
        let app_context = app_context_with_parked_queue(false).await;
        let shared_url = replaced.worker_url(8000);

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            Arc::clone(&app_context),
            8000,
            &ReconcileState::default(),
        )
        .await;

        let identities = tracked_identities(&tracked);
        assert!(
            identities.contains("uid-b"),
            "the replacement must be tracked"
        );
        assert!(
            !identities.contains("uid-a"),
            "the pod it replaced must be untracked"
        );
        assert_eq!(
            submitted_job(&app_context, &shared_url).as_deref(),
            Some("AddWorker"),
            "the last job for the shared URL must be the replacement's add, \
             never a removal that would undo it"
        );
    }

    // ---- add-side repair: tracked but never registered ----

    #[tokio::test]
    async fn test_reconcile_resubmits_a_tracked_pod_missing_from_the_registry() {
        // handle_pod_event inserts into the tracked set before submitting and
        // rolls back only if submit() itself fails. An AddWorker that fails
        // afterwards — detect_connection_mode exhausting its 30-minute budget,
        // a failing DP discovery — leaves the pod tracked with nothing
        // registered, and the add path then short-circuits on it forever. The
        // reconcile must notice the registry is empty and submit again.
        let stuck = identified_pod_info("engine-0", "uid-a", "10.0.0.1");
        let tracked = tracked_from(std::slice::from_ref(&stuck));
        let worker_url = stuck.worker_url(8000);

        let live = create_identified_k8s_pod("engine-0", "uid-a", "10.0.0.1", "ns-a");
        let (api, _) = fake_pod_api(vec![(200, pod_list_json(&[live], None))]);
        let app_context = app_context_with_parked_queue(false).await;
        assert!(
            app_context
                .worker_registry
                .get_by_url(&worker_url)
                .is_none(),
            "precondition: the worker never reached the registry"
        );

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            Arc::clone(&app_context),
            8000,
            &ReconcileState::default(),
        )
        .await;

        assert_eq!(
            submitted_job(&app_context, &worker_url).as_deref(),
            Some("AddWorker"),
            "a tracked pod with no registered worker must be submitted again"
        );
        assert!(
            tracked_identities(&tracked).contains("uid-a"),
            "the pod stays tracked once resubmitted"
        );
    }

    #[tokio::test]
    async fn test_reconcile_does_not_resubmit_a_registered_worker() {
        // The mirror of the test above: a pod whose worker did reach the
        // registry must not be resubmitted, or every resync would re-register
        // every worker.
        let registered = identified_pod_info("engine-0", "uid-a", "10.0.0.1");
        let tracked = tracked_from(std::slice::from_ref(&registered));
        let worker_url = registered.worker_url(8000);

        let live = create_identified_k8s_pod("engine-0", "uid-a", "10.0.0.1", "ns-a");
        let (api, _) = fake_pod_api(vec![(200, pod_list_json(&[live], None))]);
        let app_context = app_context_with_parked_queue(false).await;
        app_context.worker_registry.register(Arc::new(
            crate::core::BasicWorkerBuilder::new(worker_url.clone())
                .worker_type(crate::core::WorkerType::Regular)
                .build(),
        ));

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            Arc::clone(&app_context),
            8000,
            &ReconcileState::default(),
        )
        .await;

        assert_eq!(
            submitted_job(&app_context, &worker_url),
            None,
            "an already-registered worker must not be resubmitted"
        );
        assert!(
            tracked_identities(&tracked).contains("uid-a"),
            "and it must stay tracked"
        );
    }

    #[tokio::test]
    async fn test_reconcile_does_not_resubmit_a_dp_aware_worker() {
        // In dp-aware mode a worker is registered as `<url>@<rank>`, so an
        // exact-URL lookup never finds one. Reconciling the add side against
        // the registry with `get_by_url` would therefore read every dp-aware
        // pod as unregistered and resubmit it on every single resync.
        let registered = identified_pod_info("engine-0", "uid-a", "10.0.0.1");
        let tracked = tracked_from(std::slice::from_ref(&registered));
        let worker_url = registered.worker_url(8000);

        let live = create_identified_k8s_pod("engine-0", "uid-a", "10.0.0.1", "ns-a");
        let (api, _) = fake_pod_api(vec![(200, pod_list_json(&[live], None))]);
        let app_context = app_context_with_parked_queue(true).await;
        app_context.worker_registry.register(Arc::new(
            crate::core::BasicWorkerBuilder::new(format!("{}@0", worker_url))
                .worker_type(crate::core::WorkerType::Regular)
                .build(),
        ));

        reconcile_from_list(
            &api,
            &reconcile_test_config(),
            Arc::clone(&tracked),
            Arc::clone(&app_context),
            8000,
            &ReconcileState::default(),
        )
        .await;

        assert_eq!(
            submitted_job(&app_context, &worker_url),
            None,
            "a dp-aware worker registered as <url>@<rank> must count as registered"
        );
    }
}
