import os
import time
import requests
import threading

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
)
from test_ascend_single_mix_utils import run_bench_serving


KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')
LOCAL_TIMEOUT = 3600
SERVICE_PORT = "6677"

# query configmap
def query_configmap(name, namespace):
    try:
        config.load_kube_config(KUBE_CONFIG)
        v1 = client.CoreV1Api()
        configmap = v1.read_namespaced_config_map(name, namespace)
        print(f"query_configmap successfully!")
        return configmap
    except ApiException as e:
        print(f"query_configmap error {e=}")
        return None

# launch node
def launch_node(config):
    print(f"launch_node start ......")
    node_ip = os.getenv("POD_IP")
    hostname = os.getenv("HOSTNAME")
    pod_index = int(hostname.rsplit("-", 1)[-1])

    # monitor configmap to generate dist-init-addr and node-rank
    is_ready = False
    dist_init_addr = None
    while not is_ready:
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
            continue

        dist_init_addr = f"{master_node_ip}:5000"
        print(f"launch_node {dist_init_addr=}")
        is_ready = True

    special_args = [
        "--dist-init-addr",
        dist_init_addr,
        "--node-rank",
        pod_index,
    ]
    other_args = config["other_args"]
    for sa in special_args:
        other_args.append(sa)

    for key, value in config["node_envs"].items():
        print(f"ENV_VAR {key}:{value}")
        os.environ[key] = value

    print(f"Starting node, {node_ip=} {other_args=}")
    return popen_launch_server(
        config["model_path"],
        f"http://{node_ip}:{SERVICE_PORT}",
        timeout=LOCAL_TIMEOUT,
        other_args=[
            *other_args,
        ],
    )

def wait_server_ready(url, timeout=LOCAL_TIMEOUT):
    print(f"Waiting for the server to start...")
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Server {url} is ready!")
                return
        except Exception:
            pass

        if time.perf_counter() - start_time > timeout:
            raise RuntimeError(f"Server {url} failed to start in {timeout}s")
        time.sleep(10)


class TestMultiNodePdMixTestCaseBase(CustomTestCase):
    model_config = None
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
    metrics_data_file = os.getenv("METRICS_DATA_FILE")

    @classmethod
    def setUpClass(cls):
        cls.local_ip = os.getenv("POD_IP")
        hostname = os.getenv("HOSTNAME")
        cls.role = "master" if hostname.endswith("sglang-node-0") else "worker"
        print(f"Init {cls.local_ip} {cls.role=}!")

    def run_throughput(self, retry=True):
        sglang_thread = threading.Thread(
            target=launch_node, args=(self.model_config,)
        )
        sglang_thread.start()

        if self.role == "master":
            master_node_ip = os.getenv("POD_IP")
            wait_server_ready(f"http://{master_node_ip}:{SERVICE_PORT}" + "/health")
            print(f"Wait 120s, starting run benchmark ......")
            time.sleep(120)

            metrics = run_bench_serving(
                host=master_node_ip,
                port=SERVICE_PORT,
                model_path=self.model_config.get("model_path"),
                dataset_name=self.dataset_name,
                request_rate=self.request_rate,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
                input_len=self.input_len,
                output_len=self.output_len,
                random_range_ratio=self.random_range_ratio,
                result_file=self.metrics_data_file,
            )
            if retry:
                metrics = run_bench_serving(
                    host=master_node_ip,
                    port=SERVICE_PORT,
                    model_path=self.model_config.get("model_path"),
                    dataset_name=self.dataset_name,
                    request_rate=self.request_rate,
                    max_concurrency=self.max_concurrency,
                    num_prompts=self.num_prompts,
                    input_len=self.input_len,
                    output_len=self.output_len,
                    random_range_ratio=self.random_range_ratio,
                    result_file=self.metrics_data_file,
                )
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
            print("Worker node is running.")
            time.sleep(LOCAL_TIMEOUT)
