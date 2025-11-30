import argparse
import os
import re
import socket
import subprocess
import threading
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests
from kubernetes import client, config
from kubernetes.client import V1ConfigMap, V1ObjectMeta
from kubernetes.client.rest import ApiException

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

# MODEL_PATH = "/data/.cache/dsv3_layer5"
MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"
KUBE_CONFIG = "/data/.cache/kb.yaml"
DATA_PATH = "/data/.cache/GSM8K-in3584-bs8192.jsonl"
NAMESPACE = "kube-system"
CONFIGMAP_NAME = "sglang-info"
LOACL_TIMEOUT = 6000

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()

DEEPSEEK_R1_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "SGLANG_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        # "ENABLE_MOE_NZ": "1",
        "SGLANG_USE_AG_AFTER_QLORA": "1",
        "HCCL_BUFFSIZE": "1536",
        "TASK_QUEUE_ENABLE": "2",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
    },
    "decode_envs": {
        "SGLANG_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        # "ENABLE_MOE_NZ": "1",
        # "NO_DP_ALL_GATHER": "1",
        # "ENABLE_FUSED_MOE": "1",
        "DP_ROUND_ROBIN": "1",
        "HCCL_BUFFSIZE": "1200",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        # "HCCL_SOCKET_IFNAME": "data0.3001",
        # "GLOO_SOCKET_IFNAME": "data0.3001",
    },
    "prefill_args": [
        "--disaggregation-mode",
        "prefill",
        "--nnodes",
        1,
        "--node-rank",
        0,
        "--tp-size",
        16,
        "--dp-size",
        2,
        "--mem-fraction-static",
        0.81,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        32768,
        "--max-prefill-tokens",
        28680,
        "--max-running-requests",
        8,
        "--context-length",
        8192,
        "--disable-overlap-schedule",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        # "--ep-dispatch-algorithm",
        # "static",
        # "--init-expert-location",
        # "/home/fuyong/codes/sglang/sgl_ascend/hot_map/aisbench_hot_map.pt",
        "--enable-dp-attention",
        # "--tokenizer-worker-num",
        # 4,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        # "--enable-expert-distribution-metrics",
        "--disable-shared-experts-fusion",
        "--disable-cuda-graph",
    ],
    "decode_args": [
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        32,
        "--dp-size",
        32,
        "--mem-fraction-static",
        0.9,
        "--moe-a2a-backend",
        "deepep",
        "--enable-dp-attention",
        "--deepep-mode",
        "low_latency",
        "--enable-dp-lm-head",
        "--moe-dense-tp-size",
        1,
        "--disable-cuda-graph",
        "--watchdog-timeout",
        9000,
        "--context-length",
        8192,
        "--max-running-requests",
        768,
        "--prefill-round-robin-balance",
        "--cuda-graph-bs",
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        # "--ep-dispatch-algorithm",
        # "static",
        # "--init-expert-location",
        # "/data/.cache/hot_map/aisbench_hot_map_decode.pt",
        # "--ep-num-redundant-experts",
        # 64,
        "--tokenizer-worker-num",
        4,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        # "--enable-expert-distribution-metrics",
        "--disable-shared-experts-fusion",
    ],
}


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
        namespace="kube-system", label_selector="volcano.sh/task-spec=sglang-prefill"
    )
    docode_pods = v1.list_namespaced_pod(
        namespace="kube-system", label_selector="volcano.sh/task-spec=sglang-decode"
    )
    nodes_count = len(prefill_pods.items) + len(docode_pods.items)
    return nodes_count


# launch mini_lb
def launch_router():
    print(f"launch_router start ......")
    node_ip = os.getenv("POD_IP")
    nodes_count = discover_worker_nodes()
    print(f"launch_router nodes_count {nodes_count=}")

    # monitor configmap to generate p/d url
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

    lb_command = [
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
        lb_command.append("--prefill")
        lb_command.append(f"http://{url}")
        lb_command.append(f"{bootstrap_ports[index]}")

    for url in decode_url:
        lb_command.append("--decode")
        lb_command.append("http://" + url)
    lb_command_str = " ".join(lb_command)
    print(f"Starting router, {lb_command_str=}")
    # subprocess.Popen(lb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.Popen(lb_command_str, shell=True)


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
        "--quantization",
        "w8a8_int8",
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
                prefill_args.extend(["--init-expert-location", hot_map_addr])
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
                "--nnodes",
                int(discover_worker_nodes() / 2),
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


class TestAscend_DISAGGREGATION_DEEPEP(CustomTestCase):

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

    def test_a_gsm8k(self):
        if self.role == "router":
            # launch router
            router_thread = threading.Thread(target=launch_router)
            router_thread.start()
            self.wait_router_ready(f"http://127.0.0.1:6688" + "/health")

            print(f"Wait 120s, starting run benchmark ......")
            time.sleep(120)
            # print(f"Starting run benchmark ......")
            # args = SimpleNamespace(
            #     num_shots=5,
            #     data_path=DATA_PATH,
            #     num_questions=8192,
            #     max_new_tokens=512,
            #     parallel=128,
            #     host=f"http://{self.local_ip}",
            #     port=6688,
            # )
            # metrics = run_eval_few_shot_gsm8k(args)
            # parser = argparse.ArgumentParser()
            # parser.add_argument("--dataset_path", type=str, default=DATA_PATH)
            # parser.add_argument("--dataset_name", type=str, default="gsm8k")
            # parser.add_argument("--backend", type=str, default="sglang")
            # parser.add_argument("--host", type=str, default=f"http://{self.local_ip}")
            # parser.add_argument("--port", type=int, default=6688)
            # parser.add_argument("--max-concurrency", type=int, default=2100)
            # parser.add_argument("--random-output-len", type=int, default=1020)
            # parser.add_argument("--random-input-len", type=int, default=3587)
            # parser.add_argument("--num-prompts", type=int, default=8192)
            # parser.add_argument("--seed", type=int, default=42, help="Random seed")
            # parser.add_argument(
            #     "--extra_request_body",
            #     type=str,
            #     default=None,
            #     help="Extra request body",
            # )
            # parser.add_argument(
            #     "--tokenize_prompt", type=int, default=None, help="Tokenize prompt"
            # )
            # parser.add_argument("--tokenizer", type=int, default=None, help="Tokenizer")
            # parser.add_argument("--model", type=str, default=MODEL_PATH)
            # parser.add_argument("--base_url", type=str, default=None)
            # parser.add_argument("--request_rate", type=str, default=None)
            # parser.add_argument("--disable_tqdm", type=str, default=None)
            # parser.add_argument("--lora_name", type=str, default=None)
            # parser.add_argument("--profile", type=str, default=None)
            # parser.add_argument("--pd_separated", type=str, default=None)
            # parser.add_argument("--flush_cache", type=str, default=None)
            # parser.add_argument("--warmup_requests", type=str, default=None)
            # args = parser.parse_args()
            # metrics = run_benchmark(args)
            ais_res = run_command("pip3 install -e /data/.cache/benchmark/")
            print(str(ais_res))
            metrics = run_command(
                "ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf | tee /data/.cache/gsm8k_deepseek_log.txt"
            )
            print(str(metrics))
            run_command("cat /data/.cache/gsm8k_deepseek_log.txt")
            tpot = run_command(
                'cat /data/.cache/gsm8k_deepseek_log.txt | grep "TPOT" | awk "{print $6}"'
            )
            request_throughput = run_command(
                'cat /data/.cache/gsm8k_deepseek_log.txt | grep "Request Throughput" | awk "{print $7}"'
            )
            print(str(tpot))
            print(str(request_throughput))
            # metrics = run_command("bash run_ascend_ais_bench.sh")
            # self.assertGreaterEqual(
            #     metrics["accuracy"],
            #     0.90,
            # )
            # self.assertLessEqual(
            #     metrics["latency"],
            #     200,
            # )
        else:
            # launch p/d node
            sglang_thread = threading.Thread(
                target=launch_node, args=(DEEPSEEK_R1_CONFIG,)
            )
            sglang_thread.start()
            time.sleep(LOACL_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
