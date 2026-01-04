import os
import socket
import subprocess
import threading
import time
import requests

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from sglang.test.test_utils import CustomTestCase, popen_launch_server

from test_ascend_single_mix_utils import run_bench_serving


KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')
LOCAL_TIMEOUT = 3600
SERVICE_PORT = "6688"

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()

# query configmap
def query_configmap(name, namespace):
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

# get node count from k8s
def discover_worker_nodes():
    try:
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        prefill_pods = v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-prefill"
        )
        decode_pods = v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-decode"
        )

        nodes_count = len(prefill_pods.items) + len(decode_pods.items)
        print(f"Discovered {nodes_count} worker nodes (prefill: {len(prefill_pods.items)}, decode: {len(decode_pods.items)})")
        return nodes_count
    except Exception as e:
        print(f"Unexpected error discovering worker nodes: {e}")
        return 0

def set_environment_variables(env_vars):
    if not env_vars:
        return
    for key, value in env_vars.items():
        print(f"Setting ENV_VAR {key}={value}")
        os.environ[key] = value

def check_port_availability(host, port, timeout=3):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, int(port)))
            return result == 0
    except (socket.error, ValueError) as e:
        print(f"Port check error for {host}:{port}: {e}")
        return False

def wait_for_all_ports_ready(ips, port, timeout=LOCAL_TIMEOUT):
    start_time = time.time()
    while time.time() - start_time < timeout:
        ready_nodes = 0
        for ip in ips:
            if check_port_availability(ip, port):
                print(f"Node {ip}:{port} is ready")
                ready_nodes += 1
            else:
                print(f"Node {ip}:{port} is not ready yet")
        
        if ready_nodes == len(ips):
            print(f"All {len(ips)} nodes' ports are ready!")
            return True
        
        print(f"Waiting for {len(ips) - ready_nodes} more nodes to be ready...")
        time.sleep(15)
    
    print(f"Timeout: Not all nodes are ready after {timeout} seconds")
    return False

# launch router
def launch_router(config):
    print(f"launch_router start ......")
    nodes_count = discover_worker_nodes()
    print(f"Discovered {nodes_count} worker nodes")

    # monitor  to generate p/d url
    prefill_url = []
    decode_url = []
    bootstrap_ports = []
    node_ip_list = []
    is_prefill_instance_multi_node = True if "--node-rank" not in config["prefill_args"] else False
    is_decode_instance_multi_node = True if "--node-rank" not in config["decode_args"] else False

    is_ready = False
    bootstrap_init_port = 8995
    start_time = time.time()
    while not is_ready and time.time() - start_time < 300:
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
                prefill_url.append(f"{pod_ip}:8000")
                bootstrap_port = (bootstrap_init_port if is_prefill_instance_multi_node else bootstrap_init_port + pod_index)
                bootstrap_ports.append(str(bootstrap_port))
                node_ip_list.append(pod_ip)
            decode_keyword = "decode-0" if is_decode_instance_multi_node else "decode"
            if decode_keyword in pod_name:
                decode_url.append(f"{pod_ip}:8000")
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

    # checkout all node port ready
    if not wait_for_all_ports_ready(node_ip_list, 8000):
        raise RuntimeError("Failed to wait for all nodes to be ready")

    # set env var
    set_environment_variables(config.get("router_envs"))

    router_args = config["router_args"]
    # router server params
    router_command = [
        "python3",
        "-u",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        SERVICE_PORT,
        "--pd-disaggregation",
        "--policy",
        "cache_aware",
        *[str(x) for x in router_args],
    ]

    for index, url in enumerate(prefill_url):
        router_command.append("--prefill")
        router_command.append(f"http://{url}")
        router_command.append(f"{bootstrap_ports[index]}")

    for url in decode_url:
        router_command.append("--decode")
        router_command.append("http://" + url)
    router_command_str = " ".join(router_command)
    print(f"Starting router with command: {router_command_str}")
    try:
        router_process = subprocess.Popen(router_command_str, shell=True)
        print(f"Router process started with PID: {router_process.pid}")
    except Exception as e:
        raise RuntimeError(f"Failed to start router process: {e}")

# launch p/d node
def launch_node(config):
    print(f"launch_node start ......")
    node_ip = os.getenv("POD_IP")
    hostname = os.getenv("HOSTNAME")
    if not node_ip or not hostname:
        raise RuntimeError(f"Missing required environment variables: POD_IP={node_ip}, HOSTNAME={hostname}")

    pod_index = int(hostname.rsplit("-", 1)[-1])
    role = "prefill" if "prefill" in hostname else "decode"
    bootstrap_init_port = 8995
    master_prefill_ip = None
    master_decode_ip = None

    is_prefill_instance_multi_node = True if "--node-rank" not in config["prefill_args"] else False
    is_decode_instance_multi_node = True if "--node-rank" not in config["decode_args"] else False

    # monitor configmap ready
    is_ready = False
    start_time = time.time()
    while not is_ready and time.time() - start_time < 1800:
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

    # generate p/d run command
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

        set_environment_variables(config.get("prefill_envs"))

        prefill_args = config["prefill_args"]
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

        set_environment_variables(config.get("decode_envs"))

        decode_args = config["decode_args"]
        if is_decode_instance_multi_node:
            print("No node-rank specified - all decode nodes will form a single instance.")
            decode_args.extend([
                    "--dist-init-addr", dist_init_addr,
                    "--node-rank", str(pod_index),
            ])
        else:
            print("Node-rank specified - each decode node is an instance.")

        service_args.extend(decode_args)

    print(f"Starting {role} node on {node_ip} with args: {service_args}")

    try:
        process = popen_launch_server(
            config["model_path"],
            f"http://{node_ip}:{8000}",
            timeout=LOCAL_TIMEOUT,
            other_args=[
                *service_args,
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start {role} node on {node_ip}: {e}")
    return process

def wait_router_ready(url, timeout=LOCAL_TIMEOUT):
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Router {url} is ready!")
                return
        except Exception:
            pass

        if time.perf_counter() - start_time > timeout:
            raise RuntimeError(f"Server {url} failed to start in {timeout}s")
        time.sleep(10)


class TestAscendMultiNodePdSepTestCaseBase(CustomTestCase):
    model_config = None
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
    metrics_data_file = os.getenv("METRICS_DATA_FILE")

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = os.getenv("POD_IP")
        hostname = os.getenv("HOSTNAME")
        cls.role = "router" if "router" in hostname else None
        print(f"Init {cls.local_ip} {cls.role=}!")

    def run_throughput(self, retry=True):
        if self.role == "router":
            print(f"Starting router in thread...")
            router_thread = threading.Thread(
                target=launch_router, args=(self.model_config,)
            )
            router_thread.start()

            health_check_url = f"http://127.0.0.1:{SERVICE_PORT}/health"
            print(f"Waiting for router to be ready at {health_check_url}")
            wait_router_ready(health_check_url)

            print(f"Waiting 120 seconds for the server to fully initialize...")
            time.sleep(120)

            bench_params = {
                'host': "127.0.0.1",
                'port': SERVICE_PORT,
                'model_path': self.model_config.get("model_path"),
                'dataset_name': self.dataset_name,
                'request_rate': self.request_rate,
                'max_concurrency': self.max_concurrency,
                'num_prompts': self.num_prompts,
                'input_len': self.input_len,
                'output_len': self.output_len,
                'random_range_ratio': self.random_range_ratio,
                'result_file': self.metrics_data_file,
            }
            bench_params = {
                'host': "127.0.0.1",
                'port': SERVICE_PORT,
                'model_path': self.model_config.get("model_path"),
                'dataset_name': self.dataset_name,
                'request_rate': self.request_rate,
                'max_concurrency': self.max_concurrency,
                'num_prompts': self.num_prompts,
                'input_len': self.input_len,
                'output_len': self.output_len,
                'random_range_ratio': self.random_range_ratio,
                'result_file': self.metrics_data_file,
            }
            print(f"Starting benchmark with parameters: {bench_params}")
            metrics = run_bench_serving(**bench_params)

            if retry:
                print(f"Retrying benchmark...")
                metrics = run_bench_serving(**bench_params)

            if self.tpot:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot + 1 if self.tpot < 50 else self.tpot * 1.02,
                )
            if self.output_token_throughput:
                self.assertGreaterEqual(
                    float(metrics['total_tps']),
                    self.output_token_throughput * 0.98,
                )
            if self.ttft:
                self.assertLessEqual(
                    float(metrics['mean_ttft']),
                    self.ttft * 1.02,
                )
        else:
            # launch p/d node
            sglang_thread = threading.Thread(
                target=launch_node, args=(self.model_config,)
            )
            sglang_thread.start()
            print(f"{self.role} node started, keeping test alive for {LOCAL_TIMEOUT} seconds")
            time.sleep(LOCAL_TIMEOUT)
