import os
import subprocess
import psutil
import socket

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"
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

def get_nic_name():
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and (addr.address.startswith("172.") or addr.address.startswith("192.")):
                print("The nic name matched is {}".format(nic))
                return nic
    return None

NIC = get_nic_name()
NIC_NAME = "lo" if NIC is None else NIC

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command error: {e}")
        return None

def run_bench_serving(host, port, model_path=None, backend="sglang", dataset_name=None, request_rate=None,
                      max_concurrency=None, num_prompts=None, input_len=None, output_len=None, random_range_ratio=1,
                      dataset_path=None):
    cmd_args = ["python3", "-m", "sglang.bench_serving", "--host", host, "--port", str(port),
                "--model", model_path, "--backend", backend]

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

    result_file = os.getenv("METRICS_DATA_FILE")
    result_file = "./bench_log.txt" if not result_file else result_file
    print(f"The metrics result file: {result_file}")
    run_command(f"pip list | grep -E 'sglang|sgl|torch|transformers|deep-ep|memfabric_hybrid' | tee {result_file}")
    cann_info="/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
    run_command(f"echo \"CANN: $(cat {cann_info} | grep '^version=')\" | tee -a {result_file}")

    command = " " .join(cmd_args)
    print(f"Command: {command}")

    metrics = run_command(f"{command} | tee -a {result_file}")
    print(f"metrics is {metrics}")

    mean_ttft = run_command(f"grep 'Mean TTFT' {result_file} | awk '{{print $4}}'")
    mean_tpot = run_command(f"grep 'Mean TPOT' {result_file} | awk '{{print $4}}'")
    total_tps = run_command(f"grep 'Output token throughput' {result_file} | awk '{{print $5}}'")

    return {
        'mean_ttft': mean_ttft,
        'mean_tpot': mean_tpot,
        'total_tps': total_tps
    }

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
    num_prompts = int(max_concurrency) * 4
    input_len = None
    output_len = None
    random_range_ratio = None
    ttft = None
    tpot = None
    output_token_throughput = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        if cls.envs:
            for key, value in cls.envs.items():
                print(f"ENV_VAR_CASE {key}:{value}")
        env = os.environ.copy()
        for key, value in cls.envs.items():
            print(f"ENV_VAR_OTHER {key}:{value}")
        env.update(cls.envs)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_throughput(self, run_cycles=2):
        _, host, port = self.base_url.split(":")
        host = host[2:]
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

