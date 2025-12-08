import os
import re
import socket
import subprocess
import threading
import time

import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
)


KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')
LOACL_TIMEOUT = 6000

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()


def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"execute command error: {e}")
        return None


def checkout_port(host, port, timeout=3):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


# query configmap
def query_configmap(name, namespace):
    try:
        configmap = v1.read_namespaced_config_map(name, namespace)
        print(f"query_configmap successfully!")
        return configmap
    except ApiException as e:
        print(f"query_configmap error {e=}")
        return None


# get node count from k8s
def discover_worker_nodes():
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    prefill_pods = v1.list_namespaced_pod(
        namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-prefill"
    )
    docode_pods = v1.list_namespaced_pod(
        namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-decode"
    )
    nodes_count = len(prefill_pods.items) + len(docode_pods.items)
    return nodes_count


# launch router
def launch_router():
    print(f"launch_router start ......")
    node_ip = os.getenv("POD_IP")
    nodes_count = discover_worker_nodes()
    print(f"launch_router nodes_count {nodes_count=}")

    # monitor  to generate p/d url
    prefill_url = []
    decode_url = []
    bootstrap_ports = []
    node_ip_list = []

    isReady = False
    bootstrap_init_port = 8995
    while not isReady:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data == None:
            print(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue
        print(f"launch_router query_configmap {configmap.data=}")
        for pod_name in configmap.data:
            pod_ip = configmap.data[pod_name]
            if "prefill" in pod_name:
                prefill_url.append(f"{pod_ip}:8000")
                bootstrap_ports.append(str(bootstrap_init_port + int(pod_name[-1])))
                node_ip_list.append(pod_ip)
            if "decode-0" in pod_name:
                decode_url.append(f"{pod_ip}:8000")
                node_ip_list.append(pod_ip)
        isReady = True
    print(
        f"monitor configmap end, {prefill_url=} {decode_url=} {bootstrap_ports=} {node_ip_list=}"
    )

    # checkout all node port ready
    while True:
        success_nodes = 0
        port = 8000
        print(f"==================================")
        for ip in node_ip_list:
            if checkout_port(ip, port):
                print(f"{ip=} {port} is ready")
                success_nodes = success_nodes + 1
            else:
                print(f"{ip=} {port} is not ready")
        if success_nodes == len(node_ip_list):
            print(f"launch_router all node port are ready!")
            break
        time.sleep(15)

    router_command = [
        "python3",
        "-u",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--host",
        "127.0.0.1",
        "--port",
        "6688",
    ]

    for index, url in enumerate(prefill_url):
        router_command.append("--prefill")
        router_command.append(f"http://{url}")
        router_command.append(f"{bootstrap_ports[index]}")

    for url in decode_url:
        router_command.append("--decode")
        router_command.append("http://" + url)
    router_command_str = " ".join(router_command)
    print(f"Starting router, {router_command_str=}")
    # subprocess.Popen(router_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.Popen(router_command_str, shell=True)


# launch p/d node
def launch_node(config):
    print(f"launch_node start ......")
    node_ip = os.getenv("POD_IP")
    hostname = os.getenv("HOSTNAME")
    pod_index = int(hostname[-1])
    role = "prefill" if "prefill" in hostname else "decode"
    bootstrap_ports = 8995 + pod_index if role == "prefill" else None

    # monitor configmap to generate ASCEND_MF_STORE_URL and dist_init_addr
    isReady = False
    dist_init_addr = None
    while not isReady:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data == None:
            print(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue

        print(f"monitor {configmap.data=}")
        for pod_name in configmap.data:
            pod_ip = configmap.data[pod_name]
            if "prefill-0" in pod_name:
                mf_addr = f"tcp://{pod_ip}:24666"
                os.environ["ASCEND_MF_STORE_URL"] = mf_addr
                print(f"launch_node {mf_addr=}")
            if role == "decode" and "decode-0" in pod_name:
                dist_init_addr = f"{pod_ip}:5000"
                print(f"launch_node {dist_init_addr=}")
        isReady = True

    # generate p/d run command
    common_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--disaggregation-transfer-backend",
        "ascend",
    ]

    if role == "prefill":
        for key, value in config["prefill_envs"].items():
            os.environ[key] = value

        dist_init_addr = f"{node_ip}:5000"
        prefill_args = config["prefill_args"]
        prefill_args.extend(
            [
                "--dist-init-addr",
                dist_init_addr,
                "--disaggregation-bootstrap-port",
                bootstrap_ports,
            ]
        )

        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        for pod_name in configmap.data:
            pod_ip = configmap.data[pod_name]
            if pod_ip != node_ip:
                continue

            match = re.search(r"prefill-(\d+)", pod_name)
            if match:
                idx = match.group(1)
                hot_map_addr = f"/data/.cache/hot_map/aisbench_hot_map_p{idx}.pt"
                # prefill_args.extend(["--init-expert-location", hot_map_addr])
                print(f"{pod_name} get hot map in {hot_map_addr}")

        for pa in prefill_args:
            common_args.append(pa)

    if role == "decode":
        for key, value in config["decode_envs"].items():
            os.environ[key] = value

        decode_args = config["decode_args"]
        decode_args.extend(
            [
                "--dist-init-addr",
                dist_init_addr,
                # "--nnodes",
                # int(discover_worker_nodes() / 2),
                "--node-rank",
                pod_index,
            ]
        )

        for da in decode_args:
            common_args.append(da)

    print(f"Starting node, {node_ip=} {common_args=}")
    return popen_launch_server(
        config["model_path"],
        f"http://{node_ip}:{8000}",
        timeout=LOACL_TIMEOUT * 10,
        other_args=[
            *common_args,
        ],
    )


def run_bench_serving(host, port, model_path, dataset_name="random", request_rate=8, max_concurrency=8, num_prompts=32, input_len=1024, output_len=1024,
                      random_range_ratio=1):
    command = (f"python3 -m sglang.bench_serving --backend sglang --model {model_path} --host {host} --port {port} --dataset-name {dataset_name} --request-rate {request_rate} "
               f"--max-concurrency {max_concurrency} --num-prompts {num_prompts} --random-input-len {input_len} "
               f"--random-output-len {output_len} --random-range-ratio {random_range_ratio}")
    print(f"command:{command}")
    metrics = run_command(f"{command} | tee ./bench_log.txt")
    return metrics


class TestAscendDisaggregationUtils(CustomTestCase):
    model_config = None
    dataset_name = None
    request_rate = None
    max_concurrency = 8
    num_prompts = int(max_concurrency) * 4
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
        cls.role = "router" if "router" in hostname else None
        print(f"Init {cls.local_ip} {cls.role=}!")

    def wait_router_ready(self, url, timeout=LOACL_TIMEOUT):
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

    def run_throughput(self):
        if self.role == "router":
            router_thread = threading.Thread(target=launch_router)
            router_thread.start()
            self.wait_router_ready(f"http://127.0.0.1:6688" + "/health")

            print(f"Wait 120s, starting run benchmark ......")
            time.sleep(120)

            metrics = run_bench_serving(
                host="127.0.0.1",
                port="6688",
                model_path = self.model_config.get("model_path"),
                dataset_name=self.dataset_name,
                request_rate=self.request_rate,
                max_concurrency=self.max_concurrency,
                input_len=self.input_len,
                output_len=self.output_len,
                random_range_ratio=self.random_range_ratio,
            )
            print("metrics is " + str(metrics))
            res_ttft = run_command(
                "cat ./bench_log.txt | grep 'Mean TTFT' | awk '{print $4}'"
            )
            res_tpot = run_command(
                "cat ./bench_log.txt | grep 'Mean TPOT' | awk '{print $4}'"
            )
            res_output_token_throughput = run_command(
                "cat ./bench_log.txt | grep 'Output token throughput' | awk '{print $5}'"
            )
            # self.assertLessEqual(
            #     float(res_ttft),
            #     self.ttft,
            # )
            # self.assertLessEqual(
            #     float(res_tpot),
            #     self.tpot,
            # )
            # self.assertGreaterEqual(
            #     float(res_output_token_throughput),
            #     self.output_token_throughput,
            # )
            self.assertGreater(
                float(res_ttft),
                0,
            )
            self.assertGreater(
                float(res_tpot),
                0,
            )
            self.assertGreater(
                float(res_output_token_throughput),
                0,
            )
        else:
            # launch p/d node
            sglang_thread = threading.Thread(
                target=launch_node, args=(self.model_config,)
            )
            sglang_thread.start()
            time.sleep(LOACL_TIMEOUT)
