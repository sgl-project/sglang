import os
import subprocess
import threading
import time

from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_ascend_multi_node_utils import (
    launch_pd_mix_node,
    launch_router,
    wait_server_ready,
    launch_pd_seperation_node,
    SERVICE_PORT, check_role
)
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

ROUND_ROBIN = "round_robin"

LOCAL_TIMEOUT = 3600
MAX_SERVER_KEEP_ALIVE_TIME = 3600

# Timeouts and delays
SERVER_INITIALIZATION_DELAY = 120

# Test parameters
PROMPTS_MULTIPLIER = 4

# Metrics thresholds
TPOT_THRESHOLD = 50
TPOT_TOLERANCE_LOW = 1.0  # +1 second
TPOT_TOLERANCE_HIGH = 1.02  # +2%
TTFT_TOLERANCE = 1.02  # +2%
OUTPUT_TOKEN_THROUGHPUT_TOLERANCE = 0.98  # -2%

# Package filtering keywords
PACKAGE_FILTER_KEYWORDS = ['sglang', 'sgl', 'torch', 'transformers', 'deep-ep', 'memfabric_hybrid']

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

class TestAscendPerformanceTestCaseBase(CustomTestCase):
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

class TestAscendPerfMultiNodePdMixTestCaseBase(CustomTestCase):
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
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = "master" if cls.hostname.endswith("sglang-node-0") else "worker"
        print(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_mix_master_node()
        cls.start_pd_mix_worker_node()

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

    @classmethod
    @check_role(allowed_roles=["master"])
    def start_pd_mix_master_node(cls):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        sglang_thread.start()

        wait_server_ready(f"{cls.base_url}/health")

        print(f"Wait {SERVER_INITIALIZATION_DELAY}s, starting run benchmark ......")
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["worker"])
    def start_pd_mix_worker_node(cls):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        sglang_thread.start()

        print(f"{cls.role} node started, keeping test alive for {MAX_SERVER_KEEP_ALIVE_TIME} seconds")
        time.sleep(MAX_SERVER_KEEP_ALIVE_TIME)

    @check_role(allowed_roles=["master", "worker"])
    def run_throughput(self, run_cycles=2):
        bench_params = {
            'host': self.host,
            'port': str(self.port),
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

class TestAscendPerfMultiNodePdSepTestCaseBase(CustomTestCase):
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
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = "router" if "router" in cls.hostname else "prefill" if "prefill" in cls.hostname else "decode"
        print(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_server()
        cls.start_router_server()

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

    @classmethod
    @check_role(allowed_roles=["router"])
    def start_router_server(cls):
        print(f"Starting router in thread...")
        cls.sglang_thread = threading.Thread(
            target=launch_router, args=(cls.model_config,)
        )
        cls.sglang_thread.daemon = True
        cls.sglang_thread.start()

        health_check_url = f"{cls.base_url}/health"
        print(f"Waiting for router to be ready at {health_check_url}")
        wait_server_ready(health_check_url)

        print(f"Waiting {SERVER_INITIALIZATION_DELAY} seconds for the server to fully initialize...")
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["prefill", "decode"])
    def start_pd_server(cls):
        print(f"Starting pd seperation node in thread...")
        cls.sglang_thread = threading.Thread(
            target=launch_pd_seperation_node, args=(cls.model_config,)
        )
        cls.sglang_thread.daemon = True
        cls.sglang_thread.start()
        print(f"{cls.role} node started, keeping test alive for {MAX_SERVER_KEEP_ALIVE_TIME} seconds")
        time.sleep(MAX_SERVER_KEEP_ALIVE_TIME)

    @check_role(allowed_roles=["router"])
    def run_throughput(self, run_cycles=2):
        bench_params = {
            'host': self.host,
            'port': str(self.port),
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

