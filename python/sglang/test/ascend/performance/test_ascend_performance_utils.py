import os
import subprocess
import threading
import time
import psutil
import socket
import requests
from urllib.parse import urlparse
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEEPSEEK_R1_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"
DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"
DEEPSEEK_V32_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8"
QWEN3_30B_A3B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8"
QWEN3_A3B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-a3B_eagle3"
QWEN3_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
QWEN3_32B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
QWEN3_32B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Eagle3-Qwen3-32B-zh"
QWEN3_235B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B"
QWEN3_235B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"
QWEN3_235B_A22B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"
QWEN3_480B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"
QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8"
GLM_4_6_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/GLM-4.6-w8a8_WITH_MTP"

KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')

LOCAL_HOST_IP = os.getenv("POD_IP")
LOCAL_HOST_NAME = os.getenv("HOSTNAME")
if not LOCAL_HOST_IP or not LOCAL_HOST_NAME:
    raise RuntimeError(f"Missing required environment variables: POD_IP={LOCAL_HOST_IP}, HOSTNAME={LOCAL_HOST_NAME}")

ROUND_ROBIN = "round_robin"

LOCAL_TIMEOUT = 3600

# Port numbers
SERVICE_PORT = 6677
PREFILL_DECODE_PORT = 8000
BOOTSTRAP_INIT_PORT = 8995

# Timeouts and delays
ROUTER_CONFIGMAP_TIMEOUT = 300
SERVER_INITIALIZATION_DELAY = 120

# Test parameters
DEFAULT_RUN_CYCLES = 2
PROMPTS_MULTIPLIER = 4

# Metrics thresholds
TPOT_THRESHOLD = 50
TPOT_TOLERANCE_LOW = 1.0  # +1 second
TPOT_TOLERANCE_HIGH = 1.02  # +2%
TTFT_TOLERANCE = 1.02  # +2%
OUTPUT_TOKEN_THROUGHPUT_TOLERANCE = 0.98  # -2%

# Network configuration
NETWORK_ADDRESS_PREFIXES = ["172.", "192."]

# Package filtering keywords
PACKAGE_FILTER_KEYWORDS = ['sglang', 'sgl', 'torch', 'transformers', 'deep-ep', 'memfabric_hybrid']

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()

def get_nic_name():
    """Get network interface name.

    Returns:
        str: Network interface name, or None if not found.
    """
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and any(addr.address.startswith(prefix) for prefix in NETWORK_ADDRESS_PREFIXES):
                print(f"The nic name matched is {nic}")
                return nic
    return None

nic = get_nic_name()
NIC_NAME = "lo" if nic is None else nic

def get_cann_version():
    """Get CANN version info.

    Returns:
        str: CANN version info string.
    """
    cann_info_file = "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
    cann_ver_num = None

    try:
        with open(cann_info_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('version='):
                    cann_ver_num = line.strip()
                    break

        if cann_ver_num:
            cann_version_info = f"CANN: {cann_ver_num}"
            print(cann_version_info)
            return cann_version_info
        else:
            print("CANN version not found")
            return f"CANN: {cann_ver_num}"

    except FileNotFoundError:
        print(f"CANN info file not found: {cann_info_file}")
        return f"CANN: {cann_ver_num}"
    except Exception as e:
        print(f"Error reading CANN info: {e}")
        return f"CANN: {cann_ver_num}"

def write_pkg_info_to_file(result_file):
    """Write package information to result file.

    Args:
        result_file (str): Path to the result file.
    """
    try:
        pip_output = subprocess.run(["pip", "list"], capture_output=True, text=True, check=False)
        packages = pip_output.stdout

        # Filter relevant packages using list comprehension
        filtered_packages = [
            line for line in packages.split('\n')
            if any(keyword in line for keyword in PACKAGE_FILTER_KEYWORDS)
        ]

        # Write to result file
        with open(result_file, 'w', encoding='utf-8') as f:
            for pkg in filtered_packages:
                f.write(pkg + '\n')
                print(pkg)
            f.write(get_cann_version() + '\n')

    except Exception as e:
        print(f"Error getting packages: {e}")

# Query ConfigMap from Kubernetes
def query_configmap(name, namespace):
    """Query ConfigMap from Kubernetes.

    Args:
        name (str): ConfigMap name.
        namespace (str): Kubernetes namespace.

    Returns:
        V1ConfigMap: ConfigMap object, or None if failed.
    """
    try:
        configmap = v1.read_namespaced_config_map(name, namespace)
        print(f"Successfully queried ConfigMap {name} in namespace {namespace}")
        return configmap
    except ApiException as e:
        print(f"Failed to query ConfigMap {name} in namespace {namespace}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error querying ConfigMap: {e}")
        return None

# Get node count from Kubernetes
def discover_worker_nodes():
    """Discover worker nodes from Kubernetes.

    Returns:
        int: Number of worker nodes, or 0 if failed.
    """
    try:
        # Use existing config instead of loading incluster config again
        prefill_pods = v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-prefill"
        )
        decode_pods = v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-decode"
        )

        prefill_count = len(prefill_pods.items)
        decode_count = len(decode_pods.items)
        nodes_count = prefill_count + decode_count

        print(f"Discovered {nodes_count} worker nodes (prefill: {prefill_count}, decode: {decode_count})")
        return nodes_count

    except Exception as e:
        print(f"Unexpected error discovering worker nodes: {e}")
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
        print(f"Setting ENV_VAR {key}={value}")
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
                print(f"Port check failed for {host}:{port} - socket error code: {result}")
                return False

    except socket.timeout:
        print(f"Port check timeout for {host}:{port} after {timeout}s")
        return False
    except socket.gaierror as e:
        print(f"Port check address resolution error for {host}:{port}: {e}")
        return False
    except socket.error as e:
        print(f"Port check socket error for {host}:{port}: {e}")
        return False
    except ValueError as e:
        print(f"Port check invalid value for {host}:{port}: {e}")
        return False
    except Exception as e:
        print(f"Port check unexpected error for {host}:{port}: {e}")
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
                    print(f"Node {ip}:{port} is ready")
                else:
                    print(f"Node {ip}:{port} is not ready yet")
            if is_ready:
                ready_nodes += 1

        if ready_nodes == len(ips):
            print(f"All {len(ips)} nodes' ports are ready!")
            return True

        if status_changed:
            remaining_nodes = len(ips) - ready_nodes
            print(f"Waiting for {remaining_nodes} more nodes to be ready...")

        time.sleep(check_interval)

    print(f"Timeout: Not all nodes are ready after {timeout} seconds")
    return False

# Launch master/worker node
def launch_pd_mix_node(model_config):
    print(f"Launch pd mix node start ......")
    pod_index = int(LOCAL_HOST_NAME.rsplit("-", 1)[-1])

    # Monitor ConfigMap to generate dist-init-addr and node-rank
    is_ready = False
    dist_init_addr = None
    start_time = time.time()
    while not is_ready and time.time() - start_time < LOCAL_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data is None:
            print(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue
        print(f"monitor {configmap.data=}")

        master_node_ip = None
        for pod_name in configmap.data:
            if pod_name.endswith("sglang-node-0"):
                master_node_ip = configmap.data[pod_name]
                break
        if master_node_ip is None:
            print(f"Can not find master node in configmap: {configmap.data=}")
            time.sleep(15)
            continue

        dist_init_addr = f"{master_node_ip}:5000"
        print(f"launch_node {dist_init_addr=}")
        is_ready = True

    if not is_ready:
        raise RuntimeError(f"Timeout: Failed to get master node information from ConfigMap after {LOCAL_TIMEOUT} seconds")

    special_args = [
        "--dist-init-addr", dist_init_addr,
        "--node-rank", str(pod_index),
    ]
    other_args = model_config["other_args"]
    for sa in special_args:
        other_args.append(sa)

    for key, value in model_config["node_envs"].items():
        print(f"ENV_VAR_CASE {key}:{value}")
        os.environ[key] = value

    print(f"Starting node, {LOCAL_HOST_IP=} {other_args=}")
    try:
        process = popen_launch_server(
            model_config["model_path"],
            f"http://{LOCAL_HOST_IP}:{SERVICE_PORT}",
            timeout=LOCAL_TIMEOUT,
            other_args=[
                *other_args,
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start node on {LOCAL_HOST_IP}: {e}")

    return process

# Launch prefill/decode separation node
def launch_pd_seperation_node(model_config):
    print(f"Launch pd seperation node start ......")
    pod_index = int(LOCAL_HOST_NAME.rsplit("-", 1)[-1])
    role = "prefill" if "prefill" in LOCAL_HOST_NAME else "decode"

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
            print(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(15)
            continue

        print(f"Retrieved ConfigMap data: {configmap.data}")

        for pod_name, pod_ip in configmap.data.items():
            if pod_name.endswith("prefill-0"):
                master_prefill_ip = pod_ip
            if pod_name.endswith("decode-0"):
                master_decode_ip = pod_ip

        if master_prefill_ip and master_decode_ip:
            is_ready = True
        else:
            print(f"Missing master node information - prefill: {master_prefill_ip}, decode: {master_decode_ip}")
            print("Retrying in 15s...")
            time.sleep(15)
    if not is_ready:
        raise RuntimeError(f"Timeout: Failed to get master node information from ConfigMap")

    # Generate prefill/decode run command
    common_args = [
        "--trust-remote-code",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--disaggregation-transfer-backend", "ascend",
    ]
    service_args = list(common_args)

    mf_addr = f"tcp://{master_prefill_ip}:24666"
    os.environ["ASCEND_MF_STORE_URL"] = mf_addr
    print(f"Setting ENV_VAR ASCEND_MF_STORE_URL={mf_addr}")

    if role == "prefill":
        # Current node is prefill
        dist_init_addr = f"{master_prefill_ip}:5000"
        print(f"Launching prefill node with dist_init_addr={dist_init_addr}")

        set_environment_variables(model_config.get("prefill_envs"))

        prefill_args = model_config["prefill_args"]
        if is_prefill_instance_multi_node:
            print("No node-rank specified - all prefill nodes will form a single instance.")
            prefill_args.extend(
                [
                    "--node-rank", pod_index,
                    "--dist-init-addr", dist_init_addr,
                    "--disaggregation-bootstrap-port", bootstrap_init_port,
                ]
            )
        else:
            print("Node-rank specified - each prefill node is an instance.")
            prefill_args.extend([
                    "--disaggregation-bootstrap-port", str(bootstrap_init_port + pod_index),
            ])

        service_args.extend(prefill_args)

    elif role == "decode":
        dist_init_addr = f"{master_decode_ip}:5000"
        print(f"Launching decode node with dist_init_addr={dist_init_addr}")

        set_environment_variables(model_config.get("decode_envs"))

        decode_args = model_config["decode_args"]
        if is_decode_instance_multi_node:
            print("No node-rank specified - all decode nodes will form a single instance.")
            decode_args.extend([
                "--node-rank", str(pod_index),
                "--dist-init-addr", dist_init_addr,
            ])
        else:
            print("Node-rank specified - each decode node is an instance.")

        service_args.extend(decode_args)

    print(f"Starting {role} node on {LOCAL_HOST_IP} with args: {service_args}")

    try:
        process = popen_launch_server(
            model_config["model_path"],
            f"http://{LOCAL_HOST_IP}:{PREFILL_DECODE_PORT}",
            timeout=LOCAL_TIMEOUT,
            other_args=[
                *service_args,
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start {role} node on {LOCAL_HOST_IP}: {e}")

    return process

# Launch router node
def launch_router(model_config):
    print(f"launch_router start ......")
    nodes_count = discover_worker_nodes()
    print(f"Discovered {nodes_count} worker nodes")

    # Monitor to generate prefill/decode URL
    prefill_url = []
    decode_url = []
    bootstrap_ports = []
    node_ip_list = []
    is_prefill_instance_multi_node = "--node-rank" not in model_config["prefill_args"]
    is_decode_instance_multi_node = "--node-rank" not in model_config["decode_args"]

    is_ready = False
    bootstrap_init_port = BOOTSTRAP_INIT_PORT
    start_time = time.time()
    while not is_ready and time.time() - start_time < ROUTER_CONFIGMAP_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            print(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(15)
            continue
        print(f"Retrieved ConfigMap data: {configmap.data}")
        for pod_name, pod_ip in configmap.data.items():
            pod_index = int(pod_name.rsplit("-", 1)[-1])
            prefill_keyword = "prefill-0" if is_prefill_instance_multi_node else "prefill"
            if prefill_keyword in pod_name:
                prefill_url.append(f"{pod_ip}:{PREFILL_DECODE_PORT}")
                bootstrap_port = (bootstrap_init_port if is_prefill_instance_multi_node else bootstrap_init_port + pod_index)
                bootstrap_ports.append(str(bootstrap_port))
                node_ip_list.append(pod_ip)
            decode_keyword = "decode-0" if is_decode_instance_multi_node else "decode"
            if decode_keyword in pod_name:
                decode_url.append(f"{pod_ip}:{PREFILL_DECODE_PORT}")
                node_ip_list.append(pod_ip)
        if prefill_url and decode_url:
            is_ready = True
        else:
            print("Incomplete node information in ConfigMap, waiting for 15s...")
            time.sleep(15)

    if not is_ready:
        raise RuntimeError(f"Timeout: Failed to get complete node information from ConfigMap")
    print(
        f"ConfigMap monitoring complete: prefill_url={prefill_url}, decode_url={decode_url}, "
        f"bootstrap_ports={bootstrap_ports}, node_ip_list={node_ip_list}"
    )

    # Check all node port ready
    if not wait_for_all_ports_ready(ips=node_ip_list, port=PREFILL_DECODE_PORT, timeout=LOCAL_TIMEOUT):
        raise RuntimeError("Failed to wait for all nodes to be ready")

    # Set environment variables
    set_environment_variables(model_config.get("router_envs"))

    router_args = model_config["router_args"]
    # Router server params
    router_command = [
        "python3", "-u", "-m", "sglang_router.launch_router",
        "--host", "127.0.0.1",
        "--port", str(SERVICE_PORT),
        "--pd-disaggregation",
        "--policy", "cache_aware",
        *[str(x) for x in router_args],
    ]

    for index, url in enumerate(prefill_url):
        router_command.extend(["--prefill", f"http://{url}", f"{bootstrap_ports[index]}"])

    for url in decode_url:
        router_command.extend(["--decode", f"http://{url}"])

    print(f"Starting router with command: {' '.join(router_command)}")
    try:
        router_process = subprocess.Popen(router_command)
        print(f"Router process started with PID: {router_process.pid}")
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
    print(f"Waiting for the server to start at {url}...")
    start_time = time.perf_counter()
    check_interval = 10

    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Server {url} is ready!")
                return
            else:
                print(f"Server {url} returned status code: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"Server {url} request timeout, retrying...")
        except requests.exceptions.ConnectionError as e:
            print(f"Server {url} connection error: {e}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Server {url} request error: {e}, retrying...")
        except Exception as e:
            print(f"Server {url} unexpected error: {e}, retrying...")

        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > timeout:
            raise RuntimeError(f"Server {url} failed to start in {timeout}s (elapsed: {elapsed_time:.2f}s)")

        print(f"Waiting... ({elapsed_time:.2f}s elapsed, {timeout - elapsed_time:.2f}s remaining)")
        time.sleep(check_interval)

def run_bench_serving(host, port, model_path=None, backend="sglang", dataset_name=None, request_rate=None,
                      max_concurrency=None, num_prompts=None, input_len=None, output_len=None, random_range_ratio=1, dataset_path=None):
    metrics_file = os.getenv("METRICS_DATA_FILE")
    result_file = "./bench_log.txt" if not metrics_file else metrics_file
    print(f"The metrics result file: {result_file}")

    write_pkg_info_to_file(result_file)

    cmd_args = ["python3", "-m", "sglang.bench_serving", "--host", host, "--port", str(port), "--model", model_path, "--backend", backend]

    if dataset_name:
        cmd_args.extend(["--dataset-name", str(dataset_name)])
    if dataset_path:
        cmd_args.extend(["--dataset-path", str(dataset_path)])
    if request_rate:
        cmd_args.extend(["--request-rate", str(request_rate)])
    if max_concurrency:
        cmd_args.extend(["--max-concurrency", str(max_concurrency)])
    if num_prompts:
        cmd_args.extend(["--num-prompts", str(num_prompts)])
    if input_len:
        cmd_args.extend(["--random-input-len", str(input_len)])
    if output_len:
        cmd_args.extend(["--random-output-len", str(output_len)])
    if random_range_ratio:
        cmd_args.extend(["--random-range-ratio", str(random_range_ratio)])
    print(f"Command: {' '.join(cmd_args)}")

    # Run benchmark command and capture output
    metrics = {
        'mean_ttft': None,
        'mean_tpot': None,
        'total_tps': None
    }

    process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        # Read output line by line
        with open(result_file, 'a', encoding='utf-8') as f:
            for line in process.stdout:
                f.write(line)
                stripped_line = line.strip()
                print(stripped_line)

                # Extract metrics
                if 'Mean TTFT' in stripped_line:
                    parts = stripped_line.split()
                    if len(parts) >= 4:
                        metrics['mean_ttft'] = parts[3]
                elif 'Mean TPOT' in stripped_line:
                    parts = stripped_line.split()
                    if len(parts) >= 4:
                        metrics['mean_tpot'] = parts[3]
                elif 'Output token throughput' in stripped_line:
                    parts = stripped_line.split()
                    if len(parts) >= 5:
                        metrics['total_tps'] = parts[4]
        process.wait()
        if process.returncode != 0:
            print(f"Benchmark command failed with return code: {process.returncode}")
    except Exception as e:
        print(f"Error running benchmark: {e}")
    finally:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()

    return metrics

class TestPerformanceTestCaseBase(CustomTestCase):
    model = None
    backend = "sglang"
    dataset_name = None
    dataset_path = None
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 10
    envs = None
    request_rate = None
    max_concurrency = 8
    num_prompts = int(max_concurrency) * PROMPTS_MULTIPLIER
    input_len = None
    output_len = None
    random_range_ratio = None
    ttft = None
    tpot = None
    output_token_throughput = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        for key, value in env.items():
            print(f"ENV_VAR_SYS {key}:{value}")
        if cls.envs:
            for key, value in cls.envs.items():
                print(f"ENV_VAR_CASE {key}:{value}")
                env[key] = value

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'process') and cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                print(f"Error during tearDown: {e}")

    def _assert_metrics(self, metrics):
        """Assert benchmark metrics against expected values.

        Args:
            metrics (dict): Benchmark metrics dictionary.
        """
        if not metrics:
            self.fail("No metrics obtained from benchmark")

        if self.tpot:
            if self.tpot < TPOT_THRESHOLD:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot + TPOT_TOLERANCE_LOW,
                )
            else:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot * TPOT_TOLERANCE_HIGH,
                )
        if self.output_token_throughput:
            self.assertGreaterEqual(
                float(metrics['total_tps']),
                self.output_token_throughput * OUTPUT_TOKEN_THROUGHPUT_TOLERANCE,
            )
        if self.ttft:
            self.assertLessEqual(
                float(metrics['mean_ttft']),
                self.ttft * TTFT_TOLERANCE,
            )

    def run_throughput(self, run_cycles=2):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        bench_params = {
            'host': host,
            'port': port,
            'model_path': self.model,
            'backend': self.backend,
            'dataset_name': self.dataset_name,
            'request_rate': self.request_rate,
            'max_concurrency': self.max_concurrency,
            'num_prompts': self.num_prompts,
            'input_len': self.input_len,
            'output_len': self.output_len,
            'random_range_ratio': self.random_range_ratio,
            'dataset_path': self.dataset_path,
        }
        print(f"Starting benchmark with parameters: {bench_params}")

        metrics = None
        for i in range(run_cycles):
            print(f"Running benchmark, {i + 1}/{run_cycles}")
            metrics = run_bench_serving(**bench_params)

        self._assert_metrics(metrics)

class TestMultiNodePdMixTestCaseBase(CustomTestCase):
    model_config = None
    backend = "sglang"
    dataset_name = None
    dataset_path = None
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = None
    ttft = None
    tpot = None
    output_token_throughput = None

    @classmethod
    def setUpClass(cls):
        cls.local_ip = os.getenv("POD_IP")
        hostname = os.getenv("HOSTNAME")
        cls.role = "master" if hostname.endswith("sglang-node-0") else "worker"
        print(f"Init {cls.local_ip} {cls.role=}!")

    @classmethod
    def tearDownClass(cls):
        pass

    def _assert_metrics(self, metrics):
        """Assert benchmark metrics against expected values.

        Args:
            metrics (dict): Benchmark metrics dictionary.
        """
        if not metrics:
            self.fail("No metrics obtained from benchmark")

        if self.tpot:
            if self.tpot < TPOT_THRESHOLD:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot + TPOT_TOLERANCE_LOW,
                )
            else:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot * TPOT_TOLERANCE_HIGH,
                )
        if self.output_token_throughput:
            self.assertGreaterEqual(
                float(metrics['total_tps']),
                self.output_token_throughput * OUTPUT_TOKEN_THROUGHPUT_TOLERANCE,
            )
        if self.ttft:
            self.assertLessEqual(
                float(metrics['mean_ttft']),
                self.ttft * TTFT_TOLERANCE,
            )

    def run_throughput(self, run_cycles=2):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(self.model_config,)
        )
        sglang_thread.start()

        if self.role == "master":
            wait_server_ready(f"http://{self.local_ip}:{SERVICE_PORT}" + "/health")

            print(f"Wait {SERVER_INITIALIZATION_DELAY}s, starting run benchmark ......")
            time.sleep(SERVER_INITIALIZATION_DELAY)

            bench_params = {
                'host': self.local_ip,
                'port': str(SERVICE_PORT),
                'model_path': self.model_config.get("model_path"),
                'backend': self.backend,
                'dataset_name': self.dataset_name,
                'request_rate': self.request_rate,
                'max_concurrency': self.max_concurrency,
                'num_prompts': self.num_prompts,
                'input_len': self.input_len,
                'output_len': self.output_len,
                'random_range_ratio': self.random_range_ratio,
            }
            print(f"Starting benchmark with parameters: {bench_params}")

            metrics = None
            for i in range(run_cycles):
                print(f"Running benchmark, {i+1}/{run_cycles}")
                metrics = run_bench_serving(**bench_params)

            self._assert_metrics(metrics)
        else:
            print("Worker node is running.")
            time.sleep(LOCAL_TIMEOUT)

class TestAscendMultiNodePdSepTestCaseBase(CustomTestCase):
    model_config = None
    backend = "sglang"
    dataset_name = None
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = 1
    ttft = None
    tpot = None
    output_token_throughput = None

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = os.getenv("POD_IP")
        hostname = os.getenv("HOSTNAME")
        cls.role = "router" if "router" in hostname else "prefill" if "prefill" in hostname else "decode"
        print(f"Init {cls.local_ip} {cls.role=}!")

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                print(f"Error during tearDown: {e}")

    def _assert_metrics(self, metrics):
        """Assert benchmark metrics against expected values.

        Args:
            metrics (dict): Benchmark metrics dictionary.
        """
        if not metrics:
            self.fail("No metrics obtained from benchmark")

        if self.tpot:
            if self.tpot < TPOT_THRESHOLD:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot + TPOT_TOLERANCE_LOW,
                )
            else:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot * TPOT_TOLERANCE_HIGH,
                )
        if self.output_token_throughput:
            self.assertGreaterEqual(
                float(metrics['total_tps']),
                self.output_token_throughput * OUTPUT_TOKEN_THROUGHPUT_TOLERANCE,
            )
        if self.ttft:
            self.assertLessEqual(
                float(metrics['mean_ttft']),
                self.ttft * TTFT_TOLERANCE,
            )

    def run_throughput(self, run_cycles=2):
        if self.role == "router":
            print(f"Starting router in thread...")
            router_thread = threading.Thread(
                target=launch_router, args=(self.model_config,)
            )
            router_thread.start()

            health_check_url = f"http://127.0.0.1:{SERVICE_PORT}/health"
            print(f"Waiting for router to be ready at {health_check_url}")
            wait_server_ready(health_check_url)

            print(f"Waiting {SERVER_INITIALIZATION_DELAY} seconds for the server to fully initialize...")
            time.sleep(SERVER_INITIALIZATION_DELAY)

            bench_params = {
                'host': "127.0.0.1",
                'port': str(SERVICE_PORT),
                'model_path': self.model_config.get("model_path"),
                'backend': self.backend,
                'dataset_name': self.dataset_name,
                'request_rate': self.request_rate,
                'max_concurrency': self.max_concurrency,
                'num_prompts': self.num_prompts,
                'input_len': self.input_len,
                'output_len': self.output_len,
                'random_range_ratio': self.random_range_ratio,
            }
            print(f"Starting benchmark with parameters: {bench_params}")

            metrics = None
            for i in range(run_cycles):
                print(f"Running benchmark, {i+1}/{run_cycles}")
                metrics = run_bench_serving(**bench_params)

            self._assert_metrics(metrics)
        else:
            # launch p/d node
            sglang_thread = threading.Thread(
                target=launch_pd_seperation_node, args=(self.model_config,)
            )
            sglang_thread.start()
            keep_alive_time = LOCAL_TIMEOUT * 2
            print(f"{self.role} node started, keeping test alive for {keep_alive_time} seconds")
            time.sleep(keep_alive_time)
