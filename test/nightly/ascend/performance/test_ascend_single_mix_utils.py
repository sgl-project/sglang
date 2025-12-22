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
            if addr.family == socket.AF_INET and addr.address.startswith("192."):
                print("The nic name matched is {}".format(nic))
                return nic
    return None

NIC_NAME = "lo" if get_nic_name() == None else get_nic_name()

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"command error: {e}")
        return None

def run_bench_serving(host, port, model_path=None, dataset_name=None, request_rate=None, max_concurrency=None, num_prompts=None, input_len=None, output_len=None,
                      random_range_ratio=1, dataset_path=None):
    dataset_configs = (f"--dataset-name {dataset_name}")
    request_configs = "" if request_rate==None else (f"--request-rate {request_rate}")
    random_configs = (f"--random-input-len {input_len} --random-output-len {output_len} --random-range-ratio {random_range_ratio}")
    if dataset_name == "gsm8k":
        dataset_configs = (f"{dataset_configs} --dataset-path {dataset_path}")
        random_configs = (f"--random-input-len {input_len} --random-output-len {output_len}")

    command = (f"python3 -m sglang.bench_serving --backend sglang --model {model_path} --host {host} --port {port} {dataset_configs} {request_configs} "
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

class TestSingleMixUtils(CustomTestCase):
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

    print("Nic name: {}".format(NIC_NAME))

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        for key, value in cls.envs.items():
            print(f"ENV_VAR {key}:{value}")
        env = os.environ.copy()
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

    def run_throughput(self):
        _, host, port = self.base_url.split(":")
        host = host[2:]
        metrics = run_bench_serving(
            host=host,
            port=port,
            model_path=self.model,
            dataset_name=self.dataset_name,
            request_rate=self.request_rate,
            max_concurrency=self.max_concurrency,
            num_prompts=self.num_prompts,
            input_len=self.input_len,
            output_len=self.output_len,
            random_range_ratio=self.random_range_ratio,
            dataset_path=self.dataset_path
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

