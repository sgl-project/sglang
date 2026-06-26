import json
import logging
import os
import re
import subprocess
import threading
import time
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    SERVICE_PORT,
    check_role,
    launch_pd_mix_node,
    launch_pd_separation_node,
    launch_router,
    wait_server_ready,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    dump_metric,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

EVALSCOPE = "evalscope"
BENCHMARK_TOOL_DEFAULT = EVALSCOPE

PYTHON_FOR_TEST_TOOL = "test_env_transformers_tool/bin/python"
if not os.path.exists(PYTHON_FOR_TEST_TOOL) or not os.access(
    PYTHON_FOR_TEST_TOOL, os.X_OK
):
    PYTHON_FOR_TEST_TOOL = "python3"
logger.info(f"PYTHON_FOR_TEST_TOOL: {PYTHON_FOR_TEST_TOOL}")

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3600
MAX_SERVER_KEEP_ALIVE_TIME = 3600

ACCURACY_TOLERANCE = 0.99

SERVER_INITIALIZATION_DELAY = 120

if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
    )
else:
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_VISIBLE_DEVICES", "0")[0]) * 100
    )
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_SERVER_PORT_FOR_TEST + 66}"


def run_evalscope(
    host,
    port,
    model,
    datasets,
    dataset_args=None,
    eval_batch_size=16,
    limit=100000,
    generation_config=None,
    dataset_dir=None,
    timeout=60000,
    stream=True,
    eval_type="openai_api",
):

    metrics_path = os.getenv("METRICS_DATA_FILE")
    result_path = "./evalscope_result" if not metrics_path else metrics_path
    logger.info(f"The metrics result file: {result_path}")

    api_url = f"http://{host}:{port}/v1/chat/completions"

    if generation_config is None:
        generation_config = {"max_tokens": 512}

    config_dict = {
        "model": model,
        "api_url": api_url,
        "eval_type": eval_type,
        "datasets": datasets,
        "eval_batch_size": eval_batch_size,
        "generation_config": generation_config,
        "timeout": timeout,
        "stream": stream,
        "limit": limit,
        "work_dir": result_path,
    }
    if dataset_args:
        config_dict["dataset_args"] = dataset_args
    if dataset_dir:
        config_dict["dataset_dir"] = dataset_dir

    config_json = json.dumps(config_dict, ensure_ascii=False, indent=2)
    config_json_escaped = config_json.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")

    script_content = "import json\n"
    script_content += "from evalscope import TaskConfig, run_task\n\n"
    script_content += f"config = json.loads('''{config_json_escaped}''')\n"
    script_content += "task_cfg = TaskConfig(**config)\n"
    script_content += "run_task(task_cfg=task_cfg)\n"

    script_path = f"/tmp/evalscope_run_{model}_{'_'.join(datasets)}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    logger.info(f"Generated evalscope script: {script_path}")

    install_cmd = (
        "/bin/bash /root/sglang/python/sglang/test/ascend/e2e/run_evalscope.sh"
    )
    subprocess.run(install_cmd, shell=True, check=True)

    python_bin = "test_env_evalscope/bin/python"
    cmd = f"{python_bin} {script_path}"

    logger.info(f"Command: {cmd}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True,
    )

    output_lines = []
    try:
        for line in iter(process.stdout.readline, ""):
            if line.strip():
                print(line, end="")
            output_lines.append(line.strip())

        process.wait()

        if process.returncode != 0:
            logger.error(f"Command failed with return code: {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd)

        logger.info("Command executed successfully")

        metrics = {}
        full_output = "\n".join(output_lines)

        report_match = re.search(r"Dump report to:\s*(\S+)", full_output)
        if report_match:
            report_path = report_match.group(1)
            logger.info(f"Found evalscope report file: {report_path}")
            try:
                with open(report_path, "r") as rf:
                    report_data = json.load(rf)
                for item in report_data:
                    score = item.get("score")
                    if score is not None:
                        metrics["accuracy"] = float(score)
                        logger.info(f"The Final Accuracy from report: {score}")
                        break
            except Exception as e:
                logger.warning(f"Failed to read report file {report_path}: {e}")

        if "accuracy" not in metrics:
            accuracy_patterns = [
                r"mean_acc\s*.*?鈹俓s*\d+\s*鈹俓s*([\d.]+)\s*鈹?,
                r"鈹俓s+([\d.]+)\s+鈹俓s+\S+\s+鈹俓s*$",
                r"accuracy\s*[:=]?\s*([\d.]+)",
                r"Accuracy\s*[:=]?\s*([\d.]+)",
                r"score\s*[:=]?\s*([\d.]+)",
            ]

            for pattern in accuracy_patterns:
                matches = re.findall(pattern, full_output)
                if matches:
                    final_accuracy = float(matches[-1])
                    metrics["accuracy"] = final_accuracy
                    logger.info(f"The Final Accuracy from output: {final_accuracy}")
                    break

        if "accuracy" not in metrics:
            logger.info("Can Not Find The Accuracy in evalscope output")

        return metrics

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating process...")
        process.terminate()
        try:
            process.wait(timeout=5)
            logger.info("Process terminated")
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, killing it...")
            process.kill()
            logger.info("Process killed")
        raise
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        process.terminate()
        process.wait(timeout=5)
        raise


def assert_metrics(self, metrics):
    if not metrics:
        raise Exception("No metrics obtained from benchmark")

    if self.accuracy is not None:
        dump_metric(
            "accuracy",
            float(metrics["accuracy"]),
            labels={"test_case": self.__class__.__name__, "type": "accuracy"},
        )
        dump_metric(
            "accuracy_baseline",
            float(self.accuracy),
            labels={"test_case": self.__class__.__name__, "type": "accuracy"},
        )
        self.assertGreaterEqual(
            float(metrics["accuracy"]),
            self.accuracy * ACCURACY_TOLERANCE,
            f"Accuracy check failed. Expected >= {self.accuracy * ACCURACY_TOLERANCE}, Got: {metrics['accuracy']}",
        )


MMMU_LOCAL_PATH = "/root/.cache/modelscope/hub/datasets/AI-ModelScope___mmmu"


class TestNpuAccuracyTestCaseBase(CustomTestCase):
    model = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    datasets = ["gsm8k"]
    dataset_args = None
    eval_batch_size = 16
    limit = 100000
    generation_config = None
    dataset_dir = None
    stream = True
    timeout = 60000
    eval_type = "openai_api"
    other_args = None
    server_timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    max_attempts = 2
    n_runs = 3
    accuracy = 0.1

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        for key, value in env.items():
            logger.info(f"ENV_VAR_SYS {key}:{value}")
        if cls.envs:
            for key, value in cls.envs.items():
                logger.info(f"ENV_VAR_CASE {key}:{value}")
                env[key] = value

        other_args = list(cls.other_args)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.server_timeout,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    def _get_dataset_args(self):
        if "mmmu" in self.datasets:
            base_args = {"mmmu": {"dataset_id": MMMU_LOCAL_PATH}}
            if self.dataset_args:
                if isinstance(self.dataset_args, dict):
                    base_args.update(self.dataset_args)
                elif isinstance(self.dataset_args, str):
                    base_args.update(json.loads(self.dataset_args))
            return base_args
        return self.dataset_args

    def run_accuracy(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == EVALSCOPE:
            model_name = os.path.basename(self.model)
            metrics = run_evalscope(
                host=host,
                port=port,
                model=model_name,
                datasets=self.datasets,
                dataset_args=self._get_dataset_args(),
                eval_batch_size=self.eval_batch_size,
                limit=self.limit,
                generation_config=self.generation_config,
                dataset_dir=self.dataset_dir,
                stream=self.stream,
                timeout=self.timeout,
                eval_type=self.eval_type,
            )
            assert_metrics(self, metrics)

    def run_accuracy_multiple(self, n_runs=None):
        if n_runs is None:
            n_runs = self.n_runs

        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port

        if self.benchmark_tool != EVALSCOPE:
            raise Exception(
                "run_accuracy_multiple only supports evalscope benchmark tool"
            )

        model_name = os.path.basename(self.model)
        all_metrics = []

        for i in range(n_runs):
            logger.info(f"=== Accuracy run {i + 1}/{n_runs} ===")
            metrics = run_evalscope(
                host=host,
                port=port,
                model=model_name,
                datasets=self.datasets,
                dataset_args=self._get_dataset_args(),
                eval_batch_size=self.eval_batch_size,
                limit=self.limit,
                generation_config=self.generation_config,
                dataset_dir=self.dataset_dir,
                stream=self.stream,
                timeout=self.timeout,
                eval_type=self.eval_type,
            )
            all_metrics.append(metrics)
            if metrics and "accuracy" in metrics:
                logger.info(f"Run {i + 1} accuracy: {metrics['accuracy']}")
            else:
                logger.warning(f"Run {i + 1} failed to get accuracy metric")

        valid_metrics = [m for m in all_metrics if m and "accuracy" in m]
        if not valid_metrics:
            raise Exception("No valid accuracy metrics obtained from any run")

        avg_accuracy = sum(float(m["accuracy"]) for m in valid_metrics) / len(
            valid_metrics
        )

        logger.info("=" * 60)
        logger.info("Multiple Run Accuracy Results:")
        for i, m in enumerate(valid_metrics):
            logger.info(f"  Run {i + 1}: {m['accuracy']}")
        logger.info(f"  Average: {avg_accuracy}")
        logger.info("=" * 60)

        avg_metrics = {"accuracy": avg_accuracy}
        dump_metric(
            "accuracy_avg",
            avg_accuracy,
            labels={"test_case": self.__class__.__name__, "type": "accuracy"},
        )
        assert_metrics(self, avg_metrics)


class TestNpuAccuracyMultiNodePdMixTestCaseBase(CustomTestCase):
    model_config = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    datasets = ["gsm8k"]
    dataset_args = None
    eval_batch_size = 16
    limit = 100000
    generation_config = None
    dataset_dir = None
    stream = True
    timeout = 60000
    eval_type = "openai_api"
    other_args = None
    server_timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    max_attempts = 2
    accuracy = 0.1

    @classmethod
    def setUpClass(cls):
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = "master" if cls.hostname.endswith("sglang-node-0") else "worker"
        logger.info(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_mix_master_node()
        cls.start_pd_mix_worker_node()

    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    @check_role(allowed_roles=["master"])
    def start_pd_mix_master_node(cls):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        sglang_thread.start()

        wait_server_ready(f"{cls.base_url}/health")

        logger.info(
            f"Wait {SERVER_INITIALIZATION_DELAY}s, starting run benchmark ......"
        )
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["worker"])
    def start_pd_mix_worker_node(cls):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        sglang_thread.start()

        logger.info(
            f"{cls.role} node started, keeping test alive for {MAX_SERVER_KEEP_ALIVE_TIME} seconds"
        )
        time.sleep(MAX_SERVER_KEEP_ALIVE_TIME)

    def _get_dataset_args(self):
        if "mmmu" in self.datasets:
            base_args = {"mmmu": {"dataset_id": MMMU_LOCAL_PATH}}
            if self.dataset_args:
                if isinstance(self.dataset_args, dict):
                    base_args.update(self.dataset_args)
                elif isinstance(self.dataset_args, str):
                    base_args.update(json.loads(self.dataset_args))
            return base_args
        return self.dataset_args

    @check_role(allowed_roles=["master", "worker"])
    def run_accuracy(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == EVALSCOPE:
            model_name = os.path.basename(self.model_config.get("model_path"))
            metrics = run_evalscope(
                host=self.host,
                port=self.port,
                model=model_name,
                datasets=self.datasets,
                dataset_args=self._get_dataset_args(),
                eval_batch_size=self.eval_batch_size,
                limit=self.limit,
                generation_config=self.generation_config,
                dataset_dir=self.dataset_dir,
                stream=self.stream,
                timeout=self.timeout,
                eval_type=self.eval_type,
            )
            assert_metrics(self, metrics)


class TestNpuAccuracyMultiNodePdSepTestCaseBase(CustomTestCase):
    model_config = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    datasets = ["gsm8k"]
    dataset_args = None
    eval_batch_size = 16
    limit = 100000
    generation_config = None
    dataset_dir = None
    stream = True
    timeout = 60000
    eval_type = "openai_api"
    other_args = None
    server_timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    max_attempts = 2
    accuracy = 0.1

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = (
            "router"
            if "router" in cls.hostname
            else "prefill" if "prefill" in cls.hostname else "decode"
        )
        logger.info(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_server()
        cls.start_router_server()

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    @classmethod
    @check_role(allowed_roles=["router"])
    def start_router_server(cls):
        logger.info(f"Starting router in thread...")
        sglang_thread = threading.Thread(target=launch_router, args=(cls.model_config,))
        sglang_thread.daemon = True
        sglang_thread.start()

        health_check_url = f"{cls.base_url}/health"
        logger.info(f"Waiting for router to be ready at {health_check_url}")
        wait_server_ready(health_check_url)

        logger.info(
            f"Waiting {SERVER_INITIALIZATION_DELAY} seconds for the server to fully initialize..."
        )
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["prefill", "decode"])
    def start_pd_server(cls):
        logger.info(f"Starting pd separation node...")
        cls.process = launch_pd_separation_node(cls.model_config)
        logger.info(f"Pd separation node started with PID: {cls.process.pid}")

        while True:
            if cls.process.poll() is None:
                time.sleep(30)
            else:
                exit_code = cls.process.poll()
                raise Exception(
                    f"Sglang process exited on node {cls.host} {cls.hostname} with exit code: {exit_code}"
                )

    def _get_dataset_args(self):
        if "mmmu" in self.datasets:
            base_args = {"mmmu": {"dataset_id": MMMU_LOCAL_PATH}}
            if self.dataset_args:
                if isinstance(self.dataset_args, dict):
                    base_args.update(self.dataset_args)
                elif isinstance(self.dataset_args, str):
                    base_args.update(json.loads(self.dataset_args))
            return base_args
        return self.dataset_args

    @check_role(allowed_roles=["router"])
    def run_accuracy(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == EVALSCOPE:
            model_name = os.path.basename(self.model_config.get("model_path"))
            metrics = run_evalscope(
                host=host,
                port=port,
                model=model_name,
                datasets=self.datasets,
                dataset_args=self._get_dataset_args(),
                eval_batch_size=self.eval_batch_size,
                limit=self.limit,
                generation_config=self.generation_config,
                dataset_dir=self.dataset_dir,
                stream=self.stream,
                timeout=self.timeout,
                eval_type=self.eval_type,
            )
            assert_metrics(self, metrics)
