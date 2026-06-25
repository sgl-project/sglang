import logging
import os
import socket
import subprocess
import threading
import time
from functools import wraps
from types import SimpleNamespace
from typing import Iterable, Union

import psutil
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k
from sglang.test.test_utils import CustomTestCase, popen_launch_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

NAMESPACE = os.environ.get("NAMESPACE")
CONFIGMAP_NAME = os.environ.get("KUBE_CONFIG_MAP")

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
SERVER_INITIALIZATION_DELAY = 30

# Network configuration
NETWORK_ADDRESS_PREFIXES = ["172.", "192."]


def get_nic_name():
    """Get network interface name.

    Returns:
        str: Network interface name, or None if not found.
    """
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and any(
                addr.address.startswith(prefix) for prefix in NETWORK_ADDRESS_PREFIXES
            ):
                logger.info(f"The nic name matched is {nic}")
                return nic
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
    pod_index = int(host_name.rsplit("-", 1)[-1])

    # Monitor ConfigMap to generate dist-init-addr and node-rank
    is_ready = False
    dist_init_addr = None
    start_time = time.time()
    while not is_ready and time.time() - start_time < LOCAL_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data is None:
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

    for key, value in model_config["node_envs"].items():
        logger.info(f"ENV_VAR_CASE {key}:{value}")
        os.environ[key] = value

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
    pod_index = int(host_name.rsplit("-", 1)[-1])
    role = "prefill" if "prefill" in host_name else "decode"

    bootstrap_init_port = BOOTSTRAP_INIT_PORT
    master_prefill_ip = None
    master_decode_ip = None

    is_prefill_instance_multi_node = "--node-rank" not in model_config["prefill_args"]
    is_decode_instance_multi_node = "--node-rank" not in model_config["decode_args"]

    # Monitor ConfigMap ready
    is_ready = False
    start_time = time.time()
    while not is_ready and time.time() - start_time < LOCAL_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            logger.info(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(15)
            continue

        logger.info(f"Retrieved ConfigMap data: {configmap.data}")

        for pod_name, pod_ip in configmap.data.items():
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
    common_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--disaggregation-transfer-backend",
        "ascend",
    ]
    service_args = list(common_args)

    mf_addr = f"tcp://{master_prefill_ip}:24666"
    os.environ["ASCEND_MF_STORE_URL"] = mf_addr
    logger.info(f"Setting ENV_VAR ASCEND_MF_STORE_URL={mf_addr}")

    if role == "prefill":
        # Current node is prefill
        dist_init_addr = f"{master_prefill_ip}:5000"
        logger.info(f"Launching prefill node with dist_init_addr={dist_init_addr}")

        set_environment_variables(model_config.get("prefill_envs"))

        prefill_args = model_config["prefill_args"]
        if is_prefill_instance_multi_node:
            logger.info(
                "No node-rank specified - all prefill nodes will form a single instance."
            )
            prefill_args.extend(
                [
                    "--node-rank",
                    pod_index,
                    "--dist-init-addr",
                    dist_init_addr,
                    "--disaggregation-bootstrap-port",
                    bootstrap_init_port,
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
        dist_init_addr = f"{master_decode_ip}:5000"
        logger.info(f"Launching decode node with dist_init_addr={dist_init_addr}")

        set_environment_variables(model_config.get("decode_envs"))

        decode_args = model_config["decode_args"]
        if is_decode_instance_multi_node:
            logger.info(
                "No node-rank specified - all decode nodes will form a single instance."
            )
            decode_args.extend(
                [
                    "--node-rank",
                    str(pod_index),
                    "--dist-init-addr",
                    dist_init_addr,
                ]
            )
        else:
            logger.info("Node-rank specified - each decode node is an instance.")

        service_args.extend(decode_args)

    host_ip = get_host_ip()
    logger.info(f"Starting {role} node on {host_ip} with args: {service_args}")

    try:
        process = popen_launch_server(
            model_config["model_path"],
            f"http://{host_ip}:{PREFILL_DECODE_PORT}",
            timeout=LOCAL_TIMEOUT,
            other_args=[
                *service_args,
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start {role} node on {host_ip}: {e}")

    return process


# Launch router node
def launch_router(model_config):
    logger.info(f"launch_router start ......")
    discover_worker_nodes()

    # Monitor to generate prefill/decode URL
    prefill_url = []
    decode_url = []
    bootstrap_ports = []
    node_ip_list = []
    is_multi_node_prefill_instance = "--node-rank" not in model_config["prefill_args"]
    is_multi_node_decode_instance = "--node-rank" not in model_config["decode_args"]

    is_ready = False
    bootstrap_init_port = BOOTSTRAP_INIT_PORT
    start_time = time.time()
    while not is_ready and time.time() - start_time < ROUTER_CONFIGMAP_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            logger.info(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(15)
            continue
        logger.info(f"Retrieved ConfigMap data: {configmap.data}")
        for pod_name, pod_ip in configmap.data.items():
            pod_index = int(pod_name.rsplit("-", 1)[-1])
            prefill_keyword = (
                "prefill-0" if is_multi_node_prefill_instance else "prefill"
            )
            if prefill_keyword in pod_name:
                prefill_url.append(f"{pod_ip}:{PREFILL_DECODE_PORT}")
                bootstrap_port = (
                    bootstrap_init_port
                    if is_multi_node_prefill_instance
                    else bootstrap_init_port + pod_index
                )
                bootstrap_ports.append(str(bootstrap_port))
                node_ip_list.append(pod_ip)
            decode_keyword = "decode-0" if is_multi_node_decode_instance else "decode"
            if decode_keyword in pod_name:
                decode_url.append(f"{pod_ip}:{PREFILL_DECODE_PORT}")
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
        f"bootstrap_ports={bootstrap_ports}, node_ip_list={node_ip_list}"
    )

    # Check all node port ready
    if not wait_for_all_ports_ready(
        ips=node_ip_list, port=PREFILL_DECODE_PORT, timeout=LOCAL_TIMEOUT
    ):
        raise RuntimeError("Failed to wait for all nodes to be ready")

    # Set environment variables
    set_environment_variables(model_config.get("router_envs"))

    router_args = model_config["router_args"]
    # Router server params
    router_command = [
        "python3",
        "-u",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVICE_PORT),
        "--pd-disaggregation",
        "--policy",
        "cache_aware",
        *[str(x) for x in router_args],
    ]

    for index, url in enumerate(prefill_url):
        router_command.extend(
            ["--prefill", f"http://{url}", f"{bootstrap_ports[index]}"]
        )

    for url in decode_url:
        router_command.extend(["--decode", f"http://{url}"])

    logger.info(f"Starting router with command: {' '.join(router_command)}")
    try:
        router_process = subprocess.Popen(router_command)
        logger.info(f"Router process started with PID: {router_process.pid}")
    except Exception as e:
        raise RuntimeError(f"Failed to start router process: {e}")


def wait_server_ready(url, timeout=LOCAL_TIMEOUT):
    """Wait for the server to be ready.

    Args:
        url (str): Server URL to check.
        timeout (int): Timeout in seconds.

    Raises:
        RuntimeError: If server fails to start within timeout.
    """
    logger.info(f"Waiting for the server to start at {url}...")
    start_time = time.perf_counter()
    check_interval = 10

    while True:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                logger.info(f"Server {url} is ready!")
                return
            else:
                logger.info(
                    f"Server {url} returned status code: {response.status_code}"
                )
        except Exception as e:
            # logger.error(f"Server {url} request error: {e}, retrying...")
            pass

        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > timeout:
            raise RuntimeError(
                f"Server {url} failed to start in {timeout}s (elapsed: {elapsed_time:.2f}s)"
            )
        time.sleep(check_interval)


class TestAscendMultiNodePdMixTestCaseBase(CustomTestCase):
    model_config = None

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = "master" if cls.hostname.endswith("sglang-node-0") else "worker"
        logger.info(f"Init {cls.host} {cls.role=}!")
        cls.sglang_thread = None
        cls.stop_event = threading.Event()

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    @classmethod
    @check_role(allowed_roles=["master"])
    def launch_pd_mix_master_node(cls):
        logger.info(f"Starting master node in thread...")
        cls.sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        cls.sglang_thread.daemon = True
        cls.sglang_thread.start()

        health_check_url = f"{cls.base_url}/health"
        logger.info(f"Waiting for router to be ready at {health_check_url}")
        wait_server_ready(health_check_url)

        logger.info(
            f"Waiting {SERVER_INITIALIZATION_DELAY} seconds for the server to fully initialize..."
        )
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["worker"])
    def launch_pd_mix_worker_node(cls):
        logger.info(f"Starting master node in thread...")
        cls.sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        cls.sglang_thread.daemon = True
        cls.sglang_thread.start()
        keep_alive_time = 1800
        logger.info(
            f"{cls.role} node started, keeping test alive for {keep_alive_time} seconds"
        )
        time.sleep(keep_alive_time)

    @classmethod
    @check_role(allowed_roles=["master", "worker"])
    def stop_sglang_thread(cls):
        if cls.sglang_thread:
            logger.info(f"Stopping sglang thread {cls.sglang_thread}")
            if cls.sglang_thread.is_alive():
                logger.info("Notifying stop event...")
                cls.stop_event.set()
                cls.sglang_thread.join(timeout=5)
                if cls.sglang_thread.is_alive():
                    logger.info(
                        "Warning: subprocess is not terminated normally, it may has been already force stopped."
                    )
                else:
                    logger.info("Subprocess has been Stopped.")
        else:
            logger.info("No running sglang thread.")

    @check_role(allowed_roles=["master"])
    def run_gsm8k_test(
        self,
        expect_accuracy,
        num_shots=8,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
    ):
        args = SimpleNamespace(
            num_shots=num_shots,
            data_path=data_path,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens,
            parallel=parallel,
            host=f"http://{self.host}",
            port=self.port,
        )
        logger.info("Starting gsm8k test...")
        metrics = run_eval_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )


class TestAscendMultiNodePdSepTestCaseBase(CustomTestCase):
    model_config = None

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = (
            "router"
            if "router" in cls.hostname
            else "prefill" if "prefill" in cls.hostname else "decode"
        )
        logger.info(f"Init {cls.host} {cls.role=}!")
        cls.sglang_thread = None
        cls.stop_event = threading.Event()

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    @classmethod
    @check_role(allowed_roles=["router"])
    def start_router_server(cls):
        logger.info(f"Starting router in thread...")
        cls.sglang_thread = threading.Thread(
            target=launch_router, args=(cls.model_config,)
        )
        cls.sglang_thread.daemon = True
        cls.sglang_thread.start()

        health_check_url = f"{cls.base_url}/health"
        logger.info(f"Waiting for router to be ready at {health_check_url}")
        wait_server_ready(health_check_url)

        logger.info(
            f"Waiting {SERVER_INITIALIZATION_DELAY} seconds for the server to fully initialize..."
        )
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["prefill", "decode"])
    def start_pd_server(cls):
        logger.info(f"Starting pd separation node in thread...")
        cls.sglang_thread = threading.Thread(
            target=launch_pd_separation_node, args=(cls.model_config,)
        )
        cls.sglang_thread.daemon = True
        cls.sglang_thread.start()
        keep_alive_time = 1800
        logger.info(
            f"{cls.role} node started, keeping test alive for {keep_alive_time} seconds"
        )
        time.sleep(keep_alive_time)

    @classmethod
    @check_role(allowed_roles=["prefill", "decode", "router"])
    def stop_sglang_thread(cls):
        if cls.sglang_thread:
            logger.info(f"Stopping sglang thread {cls.sglang_thread}")
            if cls.sglang_thread.is_alive():
                logger.info("Notifying stop event...")
                cls.stop_event.set()
                cls.sglang_thread.join(timeout=5)
                if cls.sglang_thread.is_alive():
                    logger.info(
                        "Warning: subprocess is not terminated normally, it may has been already force stopped."
                    )
                else:
                    logger.info("Subprocess has been Stopped.")
        else:
            logger.info("No running sglang thread.")

    @check_role(allowed_roles=["router"])
    def run_gsm8k_test(
        self,
        expect_accuracy,
        num_shots=8,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
    ):
        args = SimpleNamespace(
            num_shots=num_shots,
            data_path=data_path,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens,
            parallel=parallel,
            host=f"http://{self.host}",
            port=self.port,
        )
        logger.info("Starting gsm8k test...")
        metrics = run_eval_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )
