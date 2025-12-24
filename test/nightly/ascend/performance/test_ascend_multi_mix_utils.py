import os
import subprocess
import time
import requests
import threading

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
)
from test_ascend_single_mix_utils import NIC_NAME


KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')
LOCAL_TIMEOUT = 6000
SERVICE_PORT = "6677"

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"command error: {e}")
        return None

# query configmap
def query_configmap(name, namespace):
    try:
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
    isReady = False
    dist_init_addr = None
    while not isReady:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data == None:
            print(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue
        print(f"monitor {configmap.data=}")

        master_node_ip = None
        for pod_name in configmap.data:
            if pod_name.endswith("sglang-node-0"):
                master_node_ip = configmap.data[pod_name]
                break
        if master_node_ip == None:
            print(f"Can not find master node in configmap: {configmap.data=}")
            continue

        dist_init_addr = f"{master_node_ip}:5000"
        print(f"launch_node {dist_init_addr=}")
        isReady = True

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
        timeout=LOCAL_TIMEOUT * 10,
        other_args=[
            *other_args,
        ],
    )

def run_bench_serving(host, port, model_path, dataset_name="random", request_rate=None, max_concurrency=None, num_prompts=None, input_len=None, output_len=None,
                      random_range_ratio=1):
    request_configs = "" if request_rate==None else (f"--request-rate {request_rate}")
    random_configs = (f"--random-input-len {input_len} --random-output-len {output_len} --random-range-ratio {random_range_ratio}")
    command = (f"python3 -m sglang.bench_serving --backend sglang --model {model_path} --host {host} --port {port} --dataset-name {dataset_name} {request_configs} "
               f"--max-concurrency {max_concurrency} --num-prompts {num_prompts} {random_configs}")
    print(f"command:{command}")
    metrics = run_command(f"{command} | tee ./bench_log.txt")
    print("metrics is " + str(metrics))
    mean_ttft = run_command(
        "cat ./bench_log.txt | grep 'Mean TTFT' | awk '{print $4}'"
    )
    mean_tpot = run_command(
        "cat ./bench_log.txt | grep 'Mean TPOT' | awk '{print $4}'"
    )
    total_tps = run_command(
        "cat ./bench_log.txt | grep 'Output token throughput' | awk '{print $5}'"
    )
    result = {
        'mean_ttft': mean_ttft,
        'mean_tpot': mean_tpot,
        'total_tps': total_tps
    }
    return result

class TestMultiMixUtils(CustomTestCase):
    model = None
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

    def wait_server_ready(self, url, timeout=LOCAL_TIMEOUT):
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

    def run_throughput(self):
        sglang_thread = threading.Thread(
            target=launch_node, args=(self.model_config,)
        )
        sglang_thread.start()

        if self.role == "master":
            master_node_ip = os.getenv("POD_IP")
            self.wait_server_ready(f"http://{master_node_ip}:{SERVICE_PORT}" + "/health")
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
            )
            self.assertLessEqual(
                float(metrics['mean_ttft']),
                self.ttft * 1.02,
            )
            self.assertLessEqual(
                float(metrics['mean_tpot']),
                self.tpot * 1.02,
            )
            self.assertGreaterEqual(
                float(metrics['total_tps']),
                self.output_token_throughput * 0.98,
            )
        else:
            print("Worker node is running.")
            time.sleep(LOCAL_TIMEOUT)
