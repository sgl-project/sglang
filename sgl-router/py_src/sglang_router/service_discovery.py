"""
Kubernetes service discovery for mini_lb.
This module provides service discovery functionality similar to the Rust implementation.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os

try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

logger = logging.getLogger(__name__)

if not KUBERNETES_AVAILABLE:
    logger.warning("Kubernetes library not available. Service discovery will be disabled.")


class PodType(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    REGULAR = "regular"


@dataclass
class PodInfo:
    name: str
    ip: str
    status: str
    is_ready: bool
    pod_type: Optional[PodType]
    bootstrap_port: Optional[int]

    def is_healthy(self) -> bool:
        return self.is_ready and self.status == "Running"

    def worker_url(self, port: int) -> str:
        return f"http://{self.ip}:{port}"


@dataclass
class ServiceDiscoveryConfig:
    enabled: bool = False
    selector: Dict[str, str] = None
    check_interval: int = 30  # seconds
    port: int = 80
    namespace: Optional[str] = None
    prefill_selector: Dict[str, str] = None
    decode_selector: Dict[str, str] = None
    bootstrap_port_annotation: str = "sglang.ai/bootstrap-port"

    def __post_init__(self):
        if self.selector is None:
            self.selector = {}
        if self.prefill_selector is None:
            self.prefill_selector = {}
        if self.decode_selector is None:
            self.decode_selector = {}


def create_service_discovery_config(router_args) -> ServiceDiscoveryConfig:
    """Create service discovery config from router args."""
    return ServiceDiscoveryConfig(
        enabled=router_args.service_discovery,
        selector=router_args.selector,
        port=router_args.service_discovery_port,
        namespace=router_args.service_discovery_namespace,
        prefill_selector=router_args.prefill_selector,
        decode_selector=router_args.decode_selector,
        bootstrap_port_annotation=router_args.bootstrap_port_annotation,
    )


class KubernetesServiceDiscovery:
    """
    Kubernetes service discovery implementation for mini_lb using threading.
    This is a simplified version of the Rust implementation.
    """

    def __init__(self, config: ServiceDiscoveryConfig):
        self.config = config
        self.discovered_pods: Dict[str, PodInfo] = {}
        self.prefill_urls: List[Tuple[str, Optional[int]]] = []
        self.decode_urls: List[str] = []
        self._running = False
        self._watch_thread: Optional[threading.Thread] = None
        self._watcher = None
        self._v1 = None
        self._events_processed = 0
        self._events_ignored = 0
        self._lock = threading.Lock()  # Thread safety for shared data

    def start(self):
        """Start the service discovery process."""
        if not self.config.enabled:
            logger.info("Service discovery is disabled")
            return

        if not KUBERNETES_AVAILABLE:
            logger.error("Kubernetes library not available. Cannot start service discovery.")
            return

        if self._running:
            logger.warning("Service discovery is already running")
            return

        try:
            # Load Kubernetes configuration
            if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token'):
                # Running inside a pod
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            else:
                # Running outside cluster
                config.load_kube_config()
                logger.info("Loaded local Kubernetes configuration")

            self._v1 = client.CoreV1Api()
            self._running = True
            
            # Start the watch thread
            self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._watch_thread.start()
            logger.info("Started Kubernetes service discovery with threading")
        except Exception as e:
            logger.error(f"Failed to start service discovery: {e}")
            raise

    def stop(self):
        """Stop the service discovery process."""
        if not self._running:
            return

        logger.info("Stopping Kubernetes service discovery...")
        self._running = False
        
        # Stop the watcher first
        if self._watcher:
            try:
                self._watcher.stop()
                logger.debug("Stopped Kubernetes watcher")
            except Exception as e:
                logger.warning(f"Error stopping watcher: {e}")
            finally:
                self._watcher = None
        
        # Wait for the watch thread to finish
        if self._watch_thread and self._watch_thread.is_alive():
            try:
                self._watch_thread.join(timeout=5.0)
                if self._watch_thread.is_alive():
                    logger.warning("Watch thread did not stop within timeout")
            except Exception as e:
                logger.warning(f"Error stopping watch thread: {e}")
            
        logger.info("Stopped Kubernetes service discovery")

    def _watch_loop(self):
        """Main watch loop that watches for pod changes using threading."""
        retry_count = 0
        max_retries = 5
        
        while self._running:
            try:
                self._watch_pods()
                # Reset retry count on successful watch
                retry_count = 0
            except Exception as e:
                retry_count += 1
                logger.error(f"Error in watch loop (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    logger.error(f"Max retries ({max_retries}) reached, stopping service discovery")
                    self._running = False
                    break
                
                if self._running:
                    try:
                        # Exponential backoff, but with shorter intervals for better responsiveness
                        sleep_time = min(5 * (2 ** retry_count), 60)  # Max 60 seconds
                        logger.info(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    except Exception:
                        logger.info("Watch loop interrupted during sleep")
                        self._running = False
                        break

    def _watch_pods(self):
        """Watch for pod changes using Kubernetes API."""
        try:
            # Create label selector string
            selector_parts = []
            for key, value in self.config.selector.items():
                selector_parts.append(f"{key}={value}")
            label_selector = ",".join(selector_parts) if selector_parts else None

            # Watch for pod changes
            self._watcher = watch.Watch()
            
            # Only watch pods in specified namespace
            if not self.config.namespace:
                raise ValueError("Service discovery requires a namespace to be specified")
            
            # Watch pods in specific namespace
            watch_method = lambda **kwargs: self._v1.list_namespaced_pod(
                namespace=self.config.namespace, **kwargs
            )
            logger.debug(f"Watching pods in namespace: {self.config.namespace}")
            
            # Use a shorter timeout to allow for more responsive cancellation
            for event in self._watcher.stream(
                watch_method,
                label_selector=label_selector,
                timeout_seconds=5  # Even shorter timeout for better responsiveness
            ):
                # Check if we should stop before processing each event
                if not self._running:
                    logger.info("Service discovery stopped, breaking watch loop")
                    break
                
                try:
                    # Handle the event properly - Kubernetes watch events are always dicts
                    if not isinstance(event, dict):
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Unexpected event format: {type(event)}, expected dict")
                        continue
                    
                    event_type = event.get('type')
                    pod = event.get('object')
                    
                    if not event_type:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Event missing 'type' field: {event}")
                        continue
                    
                    if not pod:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Event missing 'object' field: {event}")
                        continue
                    
                    # Handle both dict and object formats for pod
                    if isinstance(pod, dict):
                        # Pod is a dictionary
                        metadata = pod.get('metadata')
                        if not metadata:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Pod dict missing metadata: {pod}")
                            continue
                        
                        pod_name = metadata.get('name')
                        if not pod_name:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Pod metadata missing name: {metadata}")
                            continue
                        
                        # Convert dict to a simple object for consistent handling
                        pod = self._dict_to_pod_object(pod)
                        
                    else:
                        # Pod is a Python object
                        if not hasattr(pod, 'metadata'):
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Pod object missing metadata attribute: {type(pod)}")
                            continue
                        
                        if pod.metadata is None:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Pod metadata is None: {pod}")
                            continue
                        
                        # Check if metadata has required fields
                        if not hasattr(pod.metadata, 'name') or not pod.metadata.name:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Pod metadata missing name: {pod.metadata}")
                            continue
                    
                    # Check if this pod matches our selectors before processing
                    if not self._pod_matches_selectors(pod):
                        with self._lock:
                            self._events_ignored += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Pod {pod.metadata.name} does not match selectors, ignoring")
                        continue
                    
                    with self._lock:
                        self._events_processed += 1
                    
                    # Only log at debug level to reduce noise
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Pod event: {event_type} - {pod.metadata.name} in {pod.metadata.namespace}")
                    
                    if event_type in ['ADDED', 'MODIFIED']:
                        self._handle_pod_added_or_modified(pod)
                    elif event_type == 'DELETED':
                        self._handle_pod_deleted(pod)
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Ignoring event type: {event_type}")
                        
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Event that caused error: {event}")
                    continue
                    
        except ApiException as e:
            logger.error(f"Kubernetes API error: {e}")
        except Exception as e:
            logger.error(f"Error watching pods: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _handle_pod_added_or_modified(self, pod):
        """Handle pod added or modified event."""
        pod_info = self._parse_pod_info(pod)
        if pod_info and pod_info.is_healthy():
            with self._lock:
                old_pod = self.discovered_pods.get(pod_info.name)
                
                # Check if there's actually a change
                if old_pod is None:
                    # New pod added
                    self.discovered_pods[pod_info.name] = pod_info
                    self._update_urls()
                    logger.info(f"Added pod: {pod_info.name} ({pod_info.pod_type}) - {pod_info.ip}:{self.config.port}")
                elif (old_pod.ip != pod_info.ip or 
                      old_pod.pod_type != pod_info.pod_type or 
                      old_pod.bootstrap_port != pod_info.bootstrap_port):
                    # Pod actually changed
                    self.discovered_pods[pod_info.name] = pod_info
                    self._update_urls()
                    logger.info(f"Updated pod: {pod_info.name} ({pod_info.pod_type}) - {pod_info.ip}:{self.config.port}")
                # If no change, do nothing (no log)
                
        elif pod_info and pod_info.name in self.discovered_pods:
            # Pod became unhealthy, remove it
            with self._lock:
                old_pod = self.discovered_pods[pod_info.name]
                del self.discovered_pods[pod_info.name]
                self._update_urls()
            logger.info(f"Removed unhealthy pod: {pod_info.name} ({old_pod.pod_type}) - {pod_info.ip}:{self.config.port}")

    def _handle_pod_deleted(self, pod):
        """Handle pod deleted event."""
        pod_name = pod.metadata.name
        with self._lock:
            if pod_name in self.discovered_pods:
                old_pod = self.discovered_pods[pod_name]
                del self.discovered_pods[pod_name]
                self._update_urls()
                logger.info(f"Removed deleted pod: {pod_name} ({old_pod.pod_type}) - {old_pod.ip}:{self.config.port}")

    def _parse_pod_info(self, pod) -> Optional[PodInfo]:
        """Parse pod information from Kubernetes API response."""
        try:
            name = pod.metadata.name
            pod_ip = pod.status.pod_ip
            
            if not pod_ip:
                return None

            # Check if pod is ready
            is_ready = False
            if pod.status.conditions:
                for condition in pod.status.conditions:
                    # Handle both dict and object formats for condition
                    if isinstance(condition, dict):
                        condition_type = condition.get('type')
                        condition_status = condition.get('status')
                    else:
                        condition_type = getattr(condition, 'type', None)
                        condition_status = getattr(condition, 'status', None)
                    
                    if condition_type == 'Ready' and condition_status == 'True':
                        is_ready = True
                        break

            pod_status = pod.status.phase

            # Determine pod type based on labels
            pod_type = PodType.REGULAR  # Default to REGULAR
            labels = pod.metadata.labels or {}
            
            # Check for prefill selector match
            if self._matches_selector(labels, self.config.prefill_selector):
                pod_type = PodType.PREFILL
            # Check for decode selector match
            elif self._matches_selector(labels, self.config.decode_selector):
                pod_type = PodType.DECODE
            # If no specific selectors match, it remains REGULAR (no need to check general selector)

            # Extract bootstrap port from annotations for prefill pods only
            bootstrap_port = None
            if pod_type == PodType.PREFILL and pod.metadata.annotations:
                bootstrap_port_str = pod.metadata.annotations.get(self.config.bootstrap_port_annotation)
                if bootstrap_port_str:
                    try:
                        bootstrap_port = int(bootstrap_port_str)
                    except ValueError:
                        logger.warning(f"Invalid bootstrap port annotation for pod {name}: {bootstrap_port_str}")

            return PodInfo(
                name=name,
                ip=pod_ip,
                status=pod_status,
                is_ready=is_ready,
                pod_type=pod_type,
                bootstrap_port=bootstrap_port
            )

        except Exception as e:
            logger.error(f"Error parsing pod info: {e}")
            return None

    def _matches_selector(self, labels: Dict[str, str], selector: Dict[str, str]) -> bool:
        """Check if pod labels match the selector."""
        if not selector:
            return False
        
        for key, value in selector.items():
            if labels.get(key) != value:
                return False
        return True

    def _dict_to_pod_object(self, pod_dict):
        """Convert a pod dictionary to a simple object for consistent handling."""
        class PodObject:
            def __init__(self, pod_dict):
                self.metadata = PodMetadata(pod_dict.get('metadata', {}))
                self.status = PodStatus(pod_dict.get('status', {}))
        
        class PodMetadata:
            def __init__(self, metadata_dict):
                self.name = metadata_dict.get('name')
                self.namespace = metadata_dict.get('namespace')
                self.labels = metadata_dict.get('labels', {})
                self.annotations = metadata_dict.get('annotations', {})
        
        class PodStatus:
            def __init__(self, status_dict):
                self.phase = status_dict.get('phase')
                self.pod_ip = status_dict.get('podIP')  # Kubernetes API uses 'podIP'
                # Keep conditions as-is (they might be dicts or objects)
                self.conditions = status_dict.get('conditions', [])
        
        return PodObject(pod_dict)

    def _pod_matches_selectors(self, pod) -> bool:
        """Check if pod matches any of our selectors."""
        if not hasattr(pod, 'metadata') or not pod.metadata or not hasattr(pod.metadata, 'labels'):
            return False
        
        labels = pod.metadata.labels or {}
        
        # Check if pod matches any of our selectors
        selectors_to_check = [
            self.config.selector,
            self.config.prefill_selector,
            self.config.decode_selector
        ]
        
        for selector in selectors_to_check:
            if self._matches_selector(labels, selector):
                return True
        
        return False

    def _update_urls(self):
        """Update the URL lists based on discovered pods using service_discovery_port."""
        # Store previous URLs for comparison
        old_prefill_urls = set(self.prefill_urls)
        old_decode_urls = set(self.decode_urls)
        
        prefill_urls = []
        decode_urls = []
        
        for pod_info in self.discovered_pods.values():
            if pod_info.pod_type == PodType.PREFILL:
                # Use service_discovery_port for prefill servers
                prefill_urls.append((pod_info.worker_url(self.config.port), pod_info.bootstrap_port))
            elif pod_info.pod_type == PodType.DECODE:
                # Use service_discovery_port for decode servers
                decode_urls.append(pod_info.worker_url(self.config.port))
            elif pod_info.pod_type == PodType.REGULAR:
                # For regular pods, add to both lists (they can serve both prefill and decode)
                prefill_urls.append((pod_info.worker_url(self.config.port), pod_info.bootstrap_port))
                decode_urls.append(pod_info.worker_url(self.config.port))

        # Convert to sets for comparison
        new_prefill_urls = set(prefill_urls)
        new_decode_urls = set(decode_urls)
        
        # Log prefill address changes
        added_prefill = new_prefill_urls - old_prefill_urls
        removed_prefill = old_prefill_urls - new_prefill_urls
        
        for url_tuple in added_prefill:
            url, bootstrap_port = url_tuple
            logger.info(f"Prefill addr ADD: {url} (bootstrap: {bootstrap_port})")
        
        for url_tuple in removed_prefill:
            url, bootstrap_port = url_tuple
            logger.info(f"Prefill addr DEL: {url} (bootstrap: {bootstrap_port})")
        
        # Log decode address changes
        added_decode = new_decode_urls - old_decode_urls
        removed_decode = old_decode_urls - new_decode_urls
        
        for url in added_decode:
            logger.info(f"Decode addr ADD: {url}")
        
        for url in removed_decode:
            logger.info(f"Decode addr DEL: {url}")
        
        # Update the URLs
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls
        
        # Log summary only if there were changes
        if added_prefill or removed_prefill or added_decode or removed_decode:
            logger.info(f"URLs updated - Prefill: {len(prefill_urls)}, Decode: {len(decode_urls)} (port {self.config.port})")

    def get_prefill_urls(self) -> List[Tuple[str, Optional[int]]]:
        """Get current prefill URLs."""
        return self.prefill_urls.copy()

    def get_decode_urls(self) -> List[str]:
        """Get current decode URLs."""
        return self.decode_urls.copy()

    def get_pod_count(self) -> int:
        """Get current number of discovered pods."""
        return len(self.discovered_pods)

    def get_stats(self) -> Dict[str, int]:
        """Get service discovery statistics."""
        with self._lock:
            return {
                "pods_discovered": len(self.discovered_pods),
                "prefill_servers": len(self.prefill_urls),
                "decode_servers": len(self.decode_urls),
                "events_processed": self._events_processed,
                "events_ignored": self._events_ignored
            }


class MockServiceDiscovery:
    """
    Mock service discovery for testing and development.
    """

    def __init__(self, config: ServiceDiscoveryConfig):
        self.config = config
        self.prefill_urls: List[Tuple[str, Optional[int]]] = []
        self.decode_urls: List[str] = []

    def start(self):
        """Start mock service discovery."""
        if not self.config.namespace:
            logger.warning("Mock service discovery: namespace not specified")
        else:
            logger.info(f"Started mock service discovery for namespace: {self.config.namespace}")
        # In mock mode, we don't discover anything automatically
        # URLs should be provided via other means

    def stop(self):
        """Stop mock service discovery."""
        logger.info("Stopped mock service discovery")

    def get_prefill_urls(self) -> List[Tuple[str, Optional[int]]]:
        """Get current prefill URLs."""
        return self.prefill_urls.copy()

    def get_decode_urls(self) -> List[str]:
        """Get current decode URLs."""
        return self.decode_urls.copy()

    def get_pod_count(self) -> int:
        """Get current number of discovered pods."""
        return 0

    def set_urls(self, prefill_urls: List[Tuple[str, Optional[int]]], decode_urls: List[str]):
        """Set URLs manually for testing."""
        self.prefill_urls = prefill_urls.copy()
        self.decode_urls = decode_urls.copy()
        logger.info(f"Set mock URLs - Prefill: {len(prefill_urls)}, Decode: {len(decode_urls)}")


def create_service_discovery(router_args):
    """Create appropriate service discovery instance."""
    config = create_service_discovery_config(router_args)
    
    if config.enabled and KUBERNETES_AVAILABLE:
        return KubernetesServiceDiscovery(config)
    else:
        return MockServiceDiscovery(config)