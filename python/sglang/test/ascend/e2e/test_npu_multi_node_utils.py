import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
from functools import wraps
from typing import Iterable, Optional, Union

import requests

from sglang.test.server_fixtures.disaggregation_fixture import (
    SGL_ROUTER_REQUEST_TIMEOUT_SECS,
    find_sgl_router_binary,
    sgl_router_command,
)
from sglang.test.test_utils import popen_launch_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

NAMESPACE = os.environ.get("NAMESPACE")
CONFIGMAP_NAME = os.environ.get("KUBE_CONFIG_MAP")
ACTIVE_TEST_CLASS = "active-test-class"

LOCAL_TIMEOUT = 3600
ALL_ROLE_SET = {"prefill", "decode", "router", "master", "worker"}

# Port numbers
ASCEND_RT_VISIBLE_DEVICES = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
SERVICE_PORT = (
    6677
    if not ASCEND_RT_VISIBLE_DEVICES
    else 6677 + int(ASCEND_RT_VISIBLE_DEVICES.strip().split(",")[0])
)
PREFILL_DECODE_PORT = 8000
BOOTSTRAP_INIT_PORT = 8995

# Timeouts and delays
ROUTER_CONFIGMAP_TIMEOUT = 300
PD_WORKER_READY_TIMEOUT = 600
MOONCAKE_MASTER_READY_TIMEOUT = 300
SERVER_INITIALIZATION_DELAY = 30
SERVICE_EXIT_WAIT_SECONDS = 120


def get_nic_name():
    """
    Automatically identify the optimal network interface for SGLang multi-machine deployment
    Returns: str - Valid network interface name; None - No valid interface found
    """
    # Define virtual interface prefixes to exclude (k8s/docker common)
    exclude_prefixes = [
        "lo",
        "docker",
        "tunl",
        "cali",
        "veth",
        "br-",
        "virbr",
        "eth0@if",
        "kube-",
        "flannel",
        "weave",
        "cilium",
    ]

    proc_net_dev = "/proc/net/dev"
    if not os.path.exists(proc_net_dev):
        logger.error("Error: /proc/net/dev not found (not a Linux system)")
        return None

    # Store interfaces with traffic (rx_bytes + tx_bytes > 0)
    interfaces_with_traffic = {}

    with open(proc_net_dev, "r") as f:
        # Skip header lines (first 2 lines)
        lines = f.readlines()[2:]
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Split interface name and stats (format: "ifname: rx_bytes rx_packets ... tx_bytes ...")
            parts = re.split(r"\s+", line)
            if len(parts) < 10:  # Ensure enough stats fields
                continue

            ifname = parts[0].rstrip(":")

            # Skip virtual interfaces
            if any(ifname.startswith(prefix) for prefix in exclude_prefixes):
                continue

            # Get rx/tx bytes (2nd field: rx_bytes, 10th field: tx_bytes)
            try:
                rx_bytes = int(parts[1])
                tx_bytes = int(parts[9])
                total_bytes = rx_bytes + tx_bytes
            except (ValueError, IndexError):
                continue

            # Only keep interfaces with traffic (active link)
            if total_bytes > 0:
                interfaces_with_traffic[ifname] = total_bytes

    # Priority 1: Select interface with most traffic (most active)
    if interfaces_with_traffic:
        # Sort by total bytes (descending) and pick first
        sorted_interfaces = sorted(
            interfaces_with_traffic.items(), key=lambda x: x[1], reverse=True
        )
        nic_name = sorted_interfaces[0][0]
        logger.info(f"The nic name matched is {nic_name}")
        return nic_name

    # Priority 2: Fallback to first non-virtual interface (no traffic but exists)
    # Re-read to get non-virtual interfaces (even with no traffic)
    all_non_virtual = []
    with open(proc_net_dev, "r") as f:
        lines = f.readlines()[2:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            ifname = re.split(r"\s+", line)[0].rstrip(":")
            if not any(ifname.startswith(p) for p in exclude_prefixes):
                all_non_virtual.append(ifname)

    if all_non_virtual:
        nic_name = all_non_virtual[0]
        logger.info(f"The nic name matched is {nic_name}")
        return nic_name

    # No valid interface found
    logger.error("No valid interface found")
    return None


nic = get_nic_name()
NIC_NAME = "lo" if nic is None else nic


def get_host_name():
    host_name = os.getenv("HOSTNAME")
    if not host_name:
        raise RuntimeError(
            f"Missing required environment variables: HOSTNAME={host_name}"
        )
    return host_name


def get_host_ip():
    host_ip = os.getenv("POD_IP")
    if not host_ip:
        raise RuntimeError(f"Missing required environment variables: POD_IP={host_ip}")
    return host_ip


def get_k8s_api():
    from kubernetes import client, config

    kube_config = os.environ.get("KUBECONFIG")
    config.load_kube_config(kube_config)
    return client.CoreV1Api()


# Query ConfigMap from Kubernetes
def query_configmap(name, namespace):
    """Query ConfigMap from Kubernetes.

    Args:
        name (str): ConfigMap name.
        namespace (str): Kubernetes namespace.

    Returns:
        V1ConfigMap: ConfigMap object, or None if failed.
    """
    from kubernetes.client.rest import ApiException

    k8s_api = get_k8s_api()
    try:
        configmap = k8s_api.read_namespaced_config_map(name, namespace)
        logger.info(f"Successfully queried ConfigMap {name} in namespace {namespace}")
        return configmap
    except ApiException as e:
        logger.error(f"Failed to query ConfigMap {name} in namespace {namespace}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error querying ConfigMap: {e}")
        return None


def upsert_configmap_field_strict(
    name: str,
    namespace: str,
    key: str,
    value: str,
):
    """
    Add or update a field in ConfigMap using patch.
    Strict mode: fail if ConfigMap does not exist.
    """
    from kubernetes.client.rest import ApiException

    k8s_api = get_k8s_api()
    patch = {"data": {key: value}}

    try:
        k8s_api.patch_namespaced_config_map(name=name, namespace=namespace, body=patch)
        logger.info(f"Upserted ConfigMap {name}: {key}={value}")
    except ApiException as e:
        if e.status == 404:
            raise RuntimeError(
                f"ConfigMap {name} does not exist in namespace {namespace}"
            )
        logger.error(f"Failed to upsert ConfigMap {name}: {e}")
        raise


def wait_for_prefill_decode_exit(
    key: str,
    value: str,
    timeout: int = ROUTER_CONFIGMAP_TIMEOUT,
    poll_interval: int = 15,
):
    start_time = time.time()

    while time.time() - start_time < timeout:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            logger.info(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(poll_interval)
            continue

        existing_value = configmap.data.get(key)

        upsert_configmap_field_strict(CONFIGMAP_NAME, NAMESPACE, key, value)

        if existing_value is not None:
            logger.info(
                "%s already set (%s), waiting 120s for prefill/decode to exit ...",
                key,
                existing_value,
            )
            time.sleep(SERVICE_EXIT_WAIT_SECONDS)
        else:
            logger.info("%s set for the first time (%s)", key, value)

        return


# Get node count from Kubernetes
def discover_worker_nodes():
    """Discover worker nodes from Kubernetes.

    Returns:
        int: Number of worker nodes, or 0 if failed.
    """
    k8s_api = get_k8s_api()
    try:
        prefill_pods = k8s_api.list_namespaced_pod(
            namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-prefill"
        )
        decode_pods = k8s_api.list_namespaced_pod(
            namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-decode"
        )

        prefill_count = len(prefill_pods.items)
        decode_count = len(decode_pods.items)
        nodes_count = prefill_count + decode_count

        logger.info(
            f"Discovered {nodes_count} worker nodes (prefill: {prefill_count}, decode: {decode_count})"
        )
        return nodes_count

    except Exception as e:
        logger.error(f"Unexpected error discovering worker nodes: {e}")
        return 0


def set_environment_variables(env_vars):
    """Set environment variables.

    Args:
        env_vars (dict): Environment variables dictionary.

    Returns:
        dict: Updated environment variables.
    """
    if not env_vars:
        return {}

    for key, value in env_vars.items():
        logger.info(f"Setting ENV_VAR {key}={value}")
        os.environ[key] = value

    return env_vars


def check_port_availability(host, port, timeout=3):
    """Check if the port is available.

    Args:
        host (str): Host IP address.
        port (int): Port number.
        timeout (int): Connection timeout in seconds.

    Returns:
        bool: True if port is available, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, int(port)))
            if result == 0:
                return True
            else:
                return False

    except socket.timeout:
        logger.error(f"Port check timeout for {host}:{port} after {timeout}s")
        return False
    except socket.gaierror as e:
        logger.error(f"Port check address resolution error for {host}:{port}: {e}")
        return False
    except socket.error as e:
        logger.error(f"Port check socket error for {host}:{port}: {e}")
        return False
    except ValueError as e:
        logger.error(f"Port check invalid value for {host}:{port}: {e}")
        return False
    except Exception as e:
        logger.error(f"Port check unexpected error for {host}:{port}: {e}")
        return False


def _resolve_mooncake_master_address(args_list, default_address):
    """Return the master address the server launched with ``args_list`` uses.

    ``--hicache-storage-backend-extra-config`` wins over the ``MOONCAKE_MASTER``
    environment variable, so its ``master_server_address`` is the endpoint to
    poll when the case passes one; otherwise fall back to the environment form.

    Args:
        args_list (list): Server launch args.
        default_address (str): Address set through ``MOONCAKE_MASTER``.

    Returns:
        str: ``<host>:<port>`` of the Mooncake master.
    """
    from sglang.test.ascend.test_ascend_utils import get_server_args_value

    extra_config = get_server_args_value(
        args_list, "--hicache-storage-backend-extra-config"
    )
    if extra_config:
        try:
            address = json.loads(extra_config).get("master_server_address")
        except (json.JSONDecodeError, TypeError):
            address = None
        if address:
            return address
    return default_address


def wait_for_mooncake_master(
    address, timeout=MOONCAKE_MASTER_READY_TIMEOUT, poll_interval=2
):
    """Wait until the Mooncake master at ``address`` accepts connections.

    Only the node that starts the master waits for its port locally (see
    ``maybe_start_hicache_storage_service``). The other nodes of a multi-node
    instance launch concurrently and can reach the store setup before the
    master is listening; the store does not retry, so they wait here first.

    Args:
        address (str): ``<host>:<port>`` of the master.
        timeout (int): Total timeout in seconds.
        poll_interval (int): Interval between connection attempts.

    Raises:
        RuntimeError: If the master does not open its port within ``timeout``.
    """
    host, _, port = address.rpartition(":")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_port_availability(host, int(port)):
            logger.info(f"Mooncake master {address} is ready")
            return
        logger.info(f"Mooncake master {address} is not ready yet, waiting ...")
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Mooncake master {address} did not open its port within {timeout}s"
    )


def wait_for_all_ports_ready(ips, port, timeout=LOCAL_TIMEOUT, check_interval=15):
    """Wait for all nodes' ports to be ready.

    Args:
        ips (list): List of IP addresses.
        port (int): Port number to check.
        timeout (int): Total timeout in seconds.
        check_interval (int): Interval between checks in seconds.

    Returns:
        bool: True if all ports are ready, False if timeout.
    """
    start_time = time.time()
    node_status = {ip: False for ip in ips}

    while time.time() - start_time < timeout:
        ready_nodes = 0
        status_changed = False

        for ip in ips:
            is_ready = check_port_availability(ip, port)
            if is_ready != node_status[ip]:
                node_status[ip] = is_ready
                status_changed = True
                if is_ready:
                    logger.info(f"Node {ip}:{port} is ready")
                else:
                    logger.info(f"Node {ip}:{port} is not ready yet")
            if is_ready:
                ready_nodes += 1

        if ready_nodes == len(ips):
            logger.info(f"All {len(ips)} nodes' ports are ready!")
            return True

        if status_changed:
            remaining_nodes = len(ips) - ready_nodes
            logger.info(f"Waiting for {remaining_nodes} more nodes to be ready...")

        time.sleep(check_interval)

    logger.info(f"Timeout: Not all nodes are ready after {timeout} seconds")
    return False


def wait_for_pd_worker_roles(
    expected_roles,
    port,
    timeout=PD_WORKER_READY_TIMEOUT,
    check_interval=5,
    report_interval=30,
):
    """Wait until every PD worker is serving and reports its own role.

    sgl-router classifies a worker as prefill/decode by introspecting
    `/server_info` once when it starts, and static worker URLs are never
    re-introspected. Two windows have to close before that introspection can
    be trusted:

    * The engine may still be starting. `/server_info` is answered as soon as
      the HTTP server binds, while `/health` stays 503 until the engine
      finishes its warmup, so a worker can pass a role-only check and then
      stop answering for the seconds its warmup owns. The router retries the
      introspection for ~5s, falls back to the discovery backend's `Plain`
      classification, and rejects the worker when the model already has PD
      workers ("does not support mixed pools") — a rejection that is never
      retried, leaving `/readiness` at 503 forever.
    * A worker that is too busy to answer HTTP a few seconds later would
      likewise stay registered as `Plain`, and the decode side then rejects
      every request with "Disaggregated request received without bootstrap
      room id".

    Requiring `/health` == 200 before reading `/server_info` removes both
    races before the router is launched.

    The last `/server_info` snapshot of every pending worker is also logged
    every ``report_interval`` seconds, so a worker that is wedged during warmup
    leaves a trail in the router log instead of only the final timeout.

    Args:
        expected_roles (dict): {pod_ip: "prefill" | "decode"}.
        port (int): Worker HTTP port.
        timeout (int): Total timeout in seconds.
        check_interval (int): Interval between checks in seconds.
        report_interval (int): Interval between state snapshots in seconds.

    Raises:
        RuntimeError: If a worker never becomes healthy or never reports the
            expected role.
    """
    start_time = time.time()
    pending = dict(expected_roles)
    last_seen = {}
    next_report = start_time + report_interval

    while time.time() - start_time < timeout:
        for ip, role in list(pending.items()):
            health_url = f"http://{ip}:{port}/health"
            try:
                health = requests.get(health_url, timeout=10)
            except Exception as e:
                last_seen[ip] = f"health request failed: {e}"
                continue
            if health.status_code != 200:
                last_seen[ip] = f"health returned {health.status_code}"
                continue

            url = f"http://{ip}:{port}/server_info"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                info = response.json()
            except Exception as e:
                last_seen[ip] = f"request failed: {e}"
                continue

            mode = info.get("disaggregation_mode")
            bootstrap_port = info.get("disaggregation_bootstrap_port")
            # A responsive worker answers with one internal state per DP rank,
            # so the count shows whether the scheduler behind it is alive.
            last_seen[ip] = (
                f"disaggregation_mode={mode!r} bootstrap_port={bootstrap_port!r} "
                f"dp_ranks={len(info.get('internal_states') or [])}"
            )
            if mode != role:
                continue
            if mode == "prefill" and not bootstrap_port:
                last_seen[ip] += " (bootstrap port missing)"
                continue
            del pending[ip]
            logger.info(
                f"Worker {url} reports disaggregation_mode={mode} "
                f"bootstrap_port={bootstrap_port}"
            )

        if not pending:
            logger.info(f"All {len(expected_roles)} PD workers report their role")
            return

        now = time.time()
        if now >= next_report:
            logger.info(
                f"PD worker roles pending after {now - start_time:.0f}s: "
                f"{ {ip: _sample_worker_server_info(ip, port) for ip in pending} }"
            )
            next_report = now + report_interval

        time.sleep(check_interval)

    raise RuntimeError(
        f"PD workers did not report their disaggregation role within {timeout}s: "
        f"{ {ip: last_seen.get(ip, 'no response') for ip in pending} }"
    )


def _sample_worker_server_info(ip, port, timeout=5):
    """One-line `/server_info` snapshot of a worker that is still pending.

    The readiness wait may short-circuit on a 503 `/health`, which says nothing
    about whether the scheduler behind the tokenizer is alive. Probing
    `/server_info` directly (it carries one internal state per DP rank) tells
    a wedged worker (request times out) from a busy one (it answers).
    """
    try:
        info = requests.get(f"http://{ip}:{port}/server_info", timeout=timeout).json()
    except Exception as e:
        return f"server_info failed: {e}"
    return (
        f"disaggregation_mode={info.get('disaggregation_mode')!r} "
        f"bootstrap_port={info.get('disaggregation_bootstrap_port')!r} "
        f"dp_ranks={len(info.get('internal_states') or [])}"
    )


def _get_nnodes_from_args(args_list):
    from sglang.test.ascend.test_ascend_utils import get_server_args_value

    nnodes = get_server_args_value(args_list, "--nnodes")
    return int(nnodes) if nnodes is not None else None


def _is_instance_head(pod_index, is_multi_node_instance, nnodes):
    if not is_multi_node_instance:
        return True
    if nnodes is not None and nnodes >= 1:
        return pod_index % nnodes == 0
    return pod_index == 0


def _get_served_model_name(model_config):
    args = model_config["prefill_args"]
    if "--served-model-name" in args:
        return str(args[args.index("--served-model-name") + 1])
    return model_config["model_path"]


def _get_pod_ip_by_keyword(configmap_data, keyword):
    for pod_name, pod_ip in configmap_data.items():
        if keyword in pod_name:
            return pod_ip
    return None


# Pod name suffixes of the node that runs the HiCache L3 storage service
# (Ascend MemCache MetaService or Mooncake master), checked in order. PD
# separation uses the first prefill pod, PD mix uses the master node.
STORAGE_SERVICE_NODE_SUFFIXES = ("prefill-0", "sglang-node-0")


def _get_storage_service_node_ip(timeout: int = LOCAL_TIMEOUT, poll_interval: int = 15):
    """Get the IP of the node that hosts the HiCache L3 storage service.

    The service always runs on the first prefill node:
    - single-machine run: the local node itself;
    - multi-node PD mix: the master node (``*-sglang-node-0``);
    - multi-node PD separation: the first prefill node (``*-prefill-0``).

    Args:
        timeout (int): Total timeout in seconds.
        poll_interval (int): Interval between checks in seconds.

    Returns:
        str: IP of the node that hosts the storage service.

    Raises:
        RuntimeError: If the node entry is not published in time.
    """
    if not (CONFIGMAP_NAME and NAMESPACE):
        # Single-machine run: the service binds the local pod IP.
        from sglang.test.ascend.test_ascend_utils import resolve_local_ip

        node_ip = resolve_local_ip()
        logger.info(f"Single-machine run, HiCache storage service IP: {node_ip}")
        return node_ip

    start_time = time.time()
    while time.time() - start_time < timeout:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap and configmap.data:
            for suffix in STORAGE_SERVICE_NODE_SUFFIXES:
                node_ip = _get_pod_ip_by_keyword(configmap.data, suffix)
                if node_ip:
                    logger.info(f"HiCache storage service IP ({suffix}): {node_ip}")
                    return node_ip
        logger.info(
            "The HiCache storage service node is not published in ConfigMap yet, "
            "waiting ..."
        )
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Failed to get the HiCache storage service IP "
        f"({', '.join(STORAGE_SERVICE_NODE_SUFFIXES)}) "
        f"from ConfigMap within {timeout}s"
    )


def get_memcache_meta_service_url(
    port: Optional[int] = None,
    timeout: int = LOCAL_TIMEOUT,
    poll_interval: int = 15,
):
    """Get the URL of a memcache meta service endpoint.

    The host is the node that runs the storage service (see
    ``_get_storage_service_node_ip``). The port is owned by the common helpers:
    it defaults to ``MEMCACHE_META_SERVICE_PORT``, and callers pass
    ``MEMCACHE_CONFIG_STORE_PORT`` for the config store endpoint, so the ports
    stay in one place instead of being hardcoded in each test case.

    Args:
        port (int): Endpoint port. Defaults to the meta service port.
        timeout (int): Total timeout when resolving the host IP.
        poll_interval (int): Interval between host IP checks.

    Returns:
        str: ``tcp://<host ip>:<port>``.
    """
    if port is None:
        from sglang.test.ascend.test_ascend_utils import MEMCACHE_META_SERVICE_PORT

        port = MEMCACHE_META_SERVICE_PORT
    host_ip = _get_storage_service_node_ip(timeout=timeout, poll_interval=poll_interval)
    return f"tcp://{host_ip}:{port}"


def get_mooncake_master_server_address(
    port: Optional[int] = None,
    timeout: int = LOCAL_TIMEOUT,
    poll_interval: int = 15,
):
    """Get the ``<host>:<port>`` address of the Mooncake master.

    The host is the node that runs the storage service (see
    ``_get_storage_service_node_ip``), i.e. the first prefill pod rather than
    whichever pod asks. This is the value for ``master_server_address`` in the
    ``--hicache-storage-backend-extra-config`` JSON.

    Args:
        port (int): Master port. Defaults to ``MOONCAKE_MASTER_PORT`` so it
            stays in step with the master launch helper.
        timeout (int): Total timeout when resolving the host IP.
        poll_interval (int): Interval between host IP checks.

    Returns:
        str: ``<host ip>:<port>``.
    """
    if port is None:
        from sglang.test.ascend.test_ascend_utils import MOONCAKE_MASTER_PORT

        port = MOONCAKE_MASTER_PORT
    host_ip = _get_storage_service_node_ip(timeout=timeout, poll_interval=poll_interval)
    return f"{host_ip}:{port}"


def check_role(allowed_roles: Union[str, Iterable[str]]):
    if isinstance(allowed_roles, str):
        allowed_roles = {allowed_roles}
    else:
        allowed_roles = set(allowed_roles)

    if not allowed_roles.issubset(ALL_ROLE_SET):
        raise ValueError(f"Invalid allowed roles: {allowed_roles}")

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            current_role = getattr(self, "role", None)
            if current_role in allowed_roles:
                return func(self, *args, **kwargs)
            else:
                logger.info(
                    f"The current node is {current_role}, skip this function {func.__name__}."
                )
                return None

        return wrapper

    return decorator


# Launch master/worker node
def launch_pd_mix_node(model_config):
    logger.info(f"Launch pd mix node start ......")
    host_name = get_host_name()
    last_part = host_name.rsplit("-", 1)[-1]
    if not last_part.isdigit():
        raise RuntimeError(
            f"Unexpected hostname format, expected numeric suffix: {host_name}"
        )
    pod_index = int(last_part)

    # Monitor ConfigMap to generate dist-init-addr and node-rank
    is_ready = False
    dist_init_addr = None
    start_time = time.time()
    while not is_ready and time.time() - start_time < LOCAL_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or configmap.data is None:
            logger.info(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue
        logger.info(f"monitor {configmap.data=}")

        master_node_ip = None
        for pod_name in configmap.data:
            if pod_name.endswith("sglang-node-0"):
                master_node_ip = configmap.data[pod_name]
                break
        if master_node_ip is None:
            logger.info(f"Can not find master node in configmap: {configmap.data=}")
            time.sleep(15)
            continue

        dist_init_addr = f"{master_node_ip}:5000"
        logger.info(f"launch_node {dist_init_addr=}")
        is_ready = True

    if not is_ready:
        raise RuntimeError(
            f"Timeout: Failed to get master node information from ConfigMap after {LOCAL_TIMEOUT} seconds"
        )

    special_args = [
        "--dist-init-addr",
        dist_init_addr,
        "--node-rank",
        str(pod_index),
    ]
    other_args = model_config["other_args"]
    for sa in special_args:
        other_args.append(sa)

    # if not "--model-type" in other_args:
    #     other_args += ["--model-type", "llm"]

    for key, value in model_config["node_envs"].items():
        logger.info(f"ENV_VAR_CASE {key}:{value}")
        os.environ[key] = value

    from sglang.test.ascend.test_ascend_utils import (
        HICACHE_STORAGE_BACKEND_ARG,
        get_server_args_value,
        resolve_local_ip,
        set_mooncake_master_env,
    )

    if get_server_args_value(other_args, HICACHE_STORAGE_BACKEND_ARG) == "mooncake":
        master_address = set_mooncake_master_env(master_node_ip)
        # Each node registers its own address with the store instead of the
        # shared "localhost" default, which other nodes cannot reach.
        os.environ.setdefault("MOONCAKE_LOCAL_HOSTNAME", resolve_local_ip())
        # Nodes other than the one hosting the master start concurrently with
        # it and would otherwise race its listen port.
        wait_for_mooncake_master(
            _resolve_mooncake_master_address(other_args, master_address)
        )

    host_ip = get_host_ip()
    logger.info(f"Starting node, {host_ip=} {other_args=}")
    try:
        process = popen_launch_server(
            model_config["model_path"],
            f"http://{host_ip}:{SERVICE_PORT}",
            timeout=LOCAL_TIMEOUT,
            other_args=[
                *other_args,
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start node on {host_ip}: {e}")

    return process


# Launch prefill/decode separation node
def launch_pd_separation_node(model_config):
    logger.info(f"Launch pd separation node start ......")
    host_name = get_host_name()
    last_part = host_name.rsplit("-", 1)[-1]
    if not last_part.isdigit():
        raise RuntimeError(
            f"Unexpected hostname format, expected numeric suffix: {host_name}"
        )
    pod_index = int(last_part)
    role = "prefill" if "prefill" in host_name else "decode"

    bootstrap_init_port = BOOTSTRAP_INIT_PORT
    master_prefill_ip = None
    master_decode_ip = None

    is_prefill_instance_multi_node = "--node-rank" not in model_config["prefill_args"]
    is_decode_instance_multi_node = "--node-rank" not in model_config["decode_args"]

    # Monitor ConfigMap ready
    is_ready = False
    start_time = time.time()
    configmap_data = None
    while not is_ready and time.time() - start_time < LOCAL_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            logger.info(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(15)
            continue

        configmap_data = configmap.data
        logger.info(f"Retrieved ConfigMap data: {configmap_data}")

        for pod_name, pod_ip in configmap_data.items():
            if pod_name.endswith("prefill-0"):
                master_prefill_ip = pod_ip
            if pod_name.endswith("decode-0"):
                master_decode_ip = pod_ip

        if master_prefill_ip and master_decode_ip:
            is_ready = True
        else:
            logger.info(
                f"Missing master node information - prefill: {master_prefill_ip}, decode: {master_decode_ip}"
            )
            logger.info("Retrying in 15s...")
            time.sleep(15)
    if not is_ready:
        raise RuntimeError(
            f"Timeout: Failed to get master node information from ConfigMap"
        )

    # Generate prefill/decode run command
    service_args = list()

    # ASCEND_MF_STORE_URL belongs to the Ascend memfabric transfer path. Other
    # backends (e.g. mooncake with TENT) configure their metadata through their
    # own config (MC_TENT_CONF), so the legacy store URL must not leak into the
    # launched process.
    role_args = (
        model_config["prefill_args"]
        if role == "prefill"
        else model_config["decode_args"]
    )
    from sglang.test.ascend.test_ascend_utils import get_server_args_value

    transfer_backend = get_server_args_value(
        role_args, "--disaggregation-transfer-backend"
    )
    if transfer_backend in (None, "ascend"):
        mf_addr = f"tcp://{master_prefill_ip}:24666"
        os.environ["ASCEND_MF_STORE_URL"] = mf_addr
        logger.info(f"Setting ENV_VAR ASCEND_MF_STORE_URL={mf_addr}")
    else:
        os.environ.pop("ASCEND_MF_STORE_URL", None)
        logger.info(f"Skip ASCEND_MF_STORE_URL for transfer backend {transfer_backend}")

    if role == "prefill":
        # Current node is prefill
        set_environment_variables(model_config.get("prefill_envs"))

        prefill_args = model_config["prefill_args"]
        if is_prefill_instance_multi_node:
            nnodes = _get_nnodes_from_args(prefill_args)
            if nnodes and nnodes > 1:
                instance_master_index = (pod_index // nnodes) * nnodes
                node_rank = pod_index % nnodes
                instance_group_index = pod_index // nnodes
                master_pod_keyword = f"prefill-{instance_master_index}"
                instance_master_ip = _get_pod_ip_by_keyword(
                    configmap_data, master_pod_keyword
                )
                if not instance_master_ip:
                    raise RuntimeError(
                        f"Failed to find instance master {master_pod_keyword} in ConfigMap"
                    )
                dist_init_addr = f"{instance_master_ip}:5000"
                logger.info(
                    f"Multi-node prefill with nnodes={nnodes}: "
                    f"pod_index={pod_index}, node_rank={node_rank}, "
                    f"instance_master={master_pod_keyword}, dist_init_addr={dist_init_addr}"
                )
                prefill_args.extend(
                    [
                        "--node-rank",
                        node_rank,
                        "--dist-init-addr",
                        dist_init_addr,
                        "--disaggregation-bootstrap-port",
                        str(bootstrap_init_port + instance_group_index),
                    ]
                )
            else:
                logger.info(
                    "No node-rank specified - each prefill node is an independent instance."
                )
                prefill_args.extend(
                    [
                        "--node-rank",
                        0,
                        "--disaggregation-bootstrap-port",
                        str(bootstrap_init_port + pod_index),
                    ]
                )
        else:
            logger.info("Node-rank specified - each prefill node is an instance.")
            prefill_args.extend(
                [
                    "--disaggregation-bootstrap-port",
                    str(bootstrap_init_port + pod_index),
                ]
            )

        service_args.extend(prefill_args)

    elif role == "decode":
        set_environment_variables(model_config.get("decode_envs"))

        decode_args = model_config["decode_args"]
        if is_decode_instance_multi_node:
            nnodes = _get_nnodes_from_args(decode_args)
            if nnodes and nnodes > 1:
                instance_master_index = (pod_index // nnodes) * nnodes
                node_rank = pod_index % nnodes
                master_pod_keyword = f"decode-{instance_master_index}"
                instance_master_ip = _get_pod_ip_by_keyword(
                    configmap_data, master_pod_keyword
                )
                if not instance_master_ip:
                    raise RuntimeError(
                        f"Failed to find instance master {master_pod_keyword} in ConfigMap"
                    )
                dist_init_addr = f"{instance_master_ip}:5000"
                logger.info(
                    f"Multi-node decode with nnodes={nnodes}: "
                    f"pod_index={pod_index}, node_rank={node_rank}, "
                    f"instance_master={master_pod_keyword}, dist_init_addr={dist_init_addr}"
                )
                decode_args.extend(
                    [
                        "--node-rank",
                        str(node_rank),
                        "--dist-init-addr",
                        dist_init_addr,
                    ]
                )
            else:
                logger.info(
                    "No node-rank specified - each decode node is an independent instance."
                )
                decode_args.extend(["--node-rank", "0"])
        else:
            logger.info("Node-rank specified - each decode node is an instance.")

        service_args.extend(decode_args)

    host_ip = get_host_ip()
    logger.info(f"Starting {role} node on {host_ip} with args: {service_args}")

    other_args = list()
    if "--trust-remote-code" not in service_args:
        other_args.extend(["--trust-remote-code"])
    if "--attention-backend" not in service_args:
        other_args.extend(["--attention-backend", "ascend"])
    if "--device" not in service_args:
        other_args.extend(["--device", "npu"])
    if "--disaggregation-transfer-backend" not in service_args:
        other_args.extend(["--disaggregation-transfer-backend", "ascend"])
    # if "--model-type" not in service_args:
    #     other_args.extend(["--model-type", "llm"])

    other_args.extend(service_args)

    from sglang.test.ascend.test_ascend_utils import (
        HICACHE_STORAGE_BACKEND_ARG,
        get_server_args_value,
        resolve_local_ip,
        set_mooncake_master_env,
    )

    if get_server_args_value(service_args, HICACHE_STORAGE_BACKEND_ARG) == "mooncake":
        master_address = set_mooncake_master_env(master_prefill_ip)
        # Each node registers its own address with the store instead of the
        # shared "localhost" default, which other nodes cannot reach.
        os.environ.setdefault("MOONCAKE_LOCAL_HOSTNAME", resolve_local_ip())
        # Nodes other than the one hosting the master start concurrently with
        # it and would otherwise race its listen port.
        wait_for_mooncake_master(
            _resolve_mooncake_master_address(service_args, master_address)
        )

    try:
        process = popen_launch_server(
            model_config["model_path"],
            f"http://{host_ip}:{PREFILL_DECODE_PORT}",
            timeout=LOCAL_TIMEOUT,
            other_args=other_args,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start {role} node on {host_ip}: {e}")

    return process


# Launch router node
def launch_router(model_config, router_binary, process_holder=None):
    logger.info(f"launch_router start ......")
    discover_worker_nodes()

    # Monitor to generate prefill/decode URL
    prefill_url = []
    decode_url = []
    node_ip_list = []
    roles = []
    for role, urls in (("prefill", prefill_url), ("decode", decode_url)):
        args = model_config[f"{role}_args"]
        is_multi_node_instance = "--node-rank" not in args
        nnodes = _get_nnodes_from_args(args) if is_multi_node_instance else None
        roles.append((role, urls, is_multi_node_instance, nnodes))

    is_ready = False
    start_time = time.time()
    while not is_ready and time.time() - start_time < ROUTER_CONFIGMAP_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            logger.info(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(15)
            continue
        logger.info(f"Retrieved ConfigMap data: {configmap.data}")

        prefill_url.clear()
        decode_url.clear()
        node_ip_list.clear()

        for pod_name, pod_ip in configmap.data.items():
            # Skip unexpected entries that don't end with a numeric index
            last_part = pod_name.rsplit("-", 1)[-1]
            if not last_part.isdigit():
                logger.info(
                    "Skipping ConfigMap entry with non-numeric suffix: %s", pod_name
                )
                continue
            pod_index = int(last_part)

            for role, urls, is_multi_node_instance, nnodes in roles:
                if role in pod_name and _is_instance_head(
                    pod_index, is_multi_node_instance, nnodes
                ):
                    urls.append(f"{pod_ip}:{PREFILL_DECODE_PORT}")
                    node_ip_list.append(pod_ip)

        if prefill_url and decode_url:
            is_ready = True
        else:
            logger.info("Incomplete node information in ConfigMap, waiting for 15s...")
            time.sleep(15)

    if not is_ready:
        raise RuntimeError(
            f"Timeout: Failed to get complete node information from ConfigMap"
        )
    logger.info(
        f"ConfigMap monitoring complete: prefill_url={prefill_url}, decode_url={decode_url}, "
        f"node_ip_list={node_ip_list}"
    )

    # Check all node port ready
    if not wait_for_all_ports_ready(
        ips=node_ip_list, port=PREFILL_DECODE_PORT, timeout=LOCAL_TIMEOUT
    ):
        raise RuntimeError("Failed to wait for all nodes to be ready")

    # PD separation only: the router reads each worker's role from
    # /server_info exactly once when it starts, so wait until every worker is
    # both healthy (/health == 200, i.e. warmup done) and reporting its role.
    # An open TCP port is not enough, and neither is /server_info on its own:
    # it answers during warmup, and the router's introspection lands in the
    # warmup window — it then falls back to Plain and rejects the worker as a
    # mixed pool, which never recovers.
    if "prefill_args" in model_config and "decode_args" in model_config:
        expected_roles = {}
        for role, urls in (("prefill", prefill_url), ("decode", decode_url)):
            for url in urls:
                expected_roles[url.rsplit(":", 1)[0]] = role
        wait_for_pd_worker_roles(expected_roles, port=PREFILL_DECODE_PORT)

    # Set environment variables
    set_environment_variables(model_config.get("router_envs"))

    router_args = [str(x) for x in model_config["router_args"]]
    request_timeout_secs = SGL_ROUTER_REQUEST_TIMEOUT_SECS
    if "--request-timeout-secs" in router_args:
        i = router_args.index("--request-timeout-secs")
        request_timeout_secs = router_args[i + 1]
        del router_args[i : i + 2]
    router_command = sgl_router_command(
        router_binary,
        _get_served_model_name(model_config),
        [f"http://{url}" for url in prefill_url + decode_url],
        host="0.0.0.0",
        port=SERVICE_PORT,
        request_timeout_secs=request_timeout_secs,
        extra_args=router_args,
    )

    logger.info(f"Starting router with command: {' '.join(router_command)}")
    try:
        router_process = subprocess.Popen(router_command)
        logger.info(f"Router process started with PID: {router_process.pid}")
        if process_holder is not None:
            process_holder.append(router_process)
    except Exception as e:
        raise RuntimeError(f"Failed to start router process: {e}")


def start_router_thread(model_config, base_url):
    logger.info(f"Starting router in thread...")
    # Looked up here so a missing binary fails the test, not the daemon thread.
    router_binary = find_sgl_router_binary()
    # Filled by launch_router so the readiness wait can fail fast if the router
    # exits early (e.g. rejected arguments) instead of polling until timeout.
    process_holder = []
    router_thread = threading.Thread(
        target=launch_router, args=(model_config, router_binary, process_holder)
    )
    router_thread.daemon = True
    router_thread.start()

    # /health answers once the router listens; /readiness waits for the
    # prefill and decode workers to be registered.
    health_check_url = f"{base_url}/readiness"
    logger.info(f"Waiting for router to be ready at {health_check_url}")
    wait_server_ready(
        health_check_url,
        process_holder=process_holder,
        launch_thread=router_thread,
    )
    return router_thread


def wait_server_ready(
    url, timeout=LOCAL_TIMEOUT, process_holder=None, launch_thread=None
):
    """Wait for the server to be ready.

    Args:
        url (str): Server URL to check.
        timeout (int): Timeout in seconds.
        process_holder (list): Optional single-element list holding the router
            subprocess, so an early exit fails fast instead of polling.
        launch_thread (threading.Thread): Optional thread running the launcher.
            A launcher cannot outlive its thread, so a thread that exited
            before the subprocess was created means the launcher raised (its
            traceback is printed by the thread) — fail fast instead of polling.

    Raises:
        RuntimeError: If server fails to start within timeout.
    """
    logger.info(f"Waiting for the server to start at {url}...")
    start_time = time.perf_counter()
    check_interval = 10

    while True:
        if process_holder and process_holder[0].poll() is not None:
            raise RuntimeError(
                f"Router exited with code {process_holder[0].returncode} before "
                f"serving {url}; check the router output above for the reason"
            )
        if (
            launch_thread is not None
            and not launch_thread.is_alive()
            and not process_holder
        ):
            raise RuntimeError(
                f"Server launcher exited before starting the server for {url}; "
                "check the traceback printed by the launcher thread above"
            )
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                logger.info(f"Server {url} is ready!")
                return
            else:
                logger.info(
                    f"Server {url} returned status code: {response.status_code}"
                )
        except Exception:
            # logger.error(f"Server {url} request error: {e}, retrying...")
            pass

        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > timeout:
            raise RuntimeError(
                f"Server {url} failed to start in {timeout}s (elapsed: {elapsed_time:.2f}s)"
            )
        time.sleep(check_interval)


def kill_process_group(process):
    """SIGKILL the whole process group led by ``process.pid`` (== pgid).

    ``process`` is a ``subprocess.Popen`` launched with ``start_new_session=True``,
    so its pid equals the process-group id. A ``None`` process is ignored.
    """
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
        logger.info(f"killed process group pgid={process.pid}")
    except ProcessLookupError:
        logger.info(f"process group pgid={process.pid} already gone")
    except PermissionError:
        logger.warning(f"no permission to kill process group pgid={process.pid}")
    except OSError as e:
        logger.warning(f"failed to kill process group pgid={process.pid}: {e}")
