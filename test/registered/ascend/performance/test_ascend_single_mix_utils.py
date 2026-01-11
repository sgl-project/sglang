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

def get_nic_name():
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and (addr.address.startswith("172.") or addr.address.startswith("192.")):
                print("The nic name matched is {}".format(nic))
                return nic
    return None

NIC_NAME = "lo" if get_nic_name() is None else get_nic_name()

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command error: {e}")
        return None

def run_bench_serving(host, port, model_path=None, dataset_name=None, request_rate=None, max_concurrency=None, num_prompts=None, input_len=None, output_len=None,
                      random_range_ratio=1, dataset_path=None, result_file=None):
    cmd_args = ["python3", "-m", "sglang.bench_serving", "--backend", "sglang", 
                "--model", model_path, "--host", host, "--port", str(port)]
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

    result_file = "./bench_log.txt" if not result_file else result_file
    print(f"The metrics result file: {result_file}")
    run_command(f"pip list | grep -E 'sglang|sgl|torch|transformers' | tee {result_file}")

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

class TestSingleNodeTestCaseBase(CustomTestCase):
    model = None
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
    metrics_data_file = os.getenv("METRICS_DATA_FILE")

    print("Nic name: {}".format(NIC_NAME))

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

    def run_throughput(self, retry=True):
        _, host, port = self.base_url.split(":")
        host = host[2:]
        bench_params = {
            'host': host,
            'port': port,
            'model_path': self.model,
            'dataset_name': self.dataset_name,
            'request_rate': self.request_rate,
            'max_concurrency': self.max_concurrency,
            'num_prompts': self.num_prompts,
            'input_len': self.input_len,
            'output_len': self.output_len,
            'random_range_ratio': self.random_range_ratio,
            'dataset_path': self.dataset_path,
            'result_file': self.metrics_data_file,
        }
        metrics = run_bench_serving(**bench_params)

        if retry:
            metrics = run_bench_serving(**bench_params)
        if self.tpot:
            self.assertLessEqual(
                float(metrics['mean_tpot']),
                self.tpot + 1 if self.tpot < 50 else self.tpot * 1,
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

