import logging
import os
import re
import subprocess
import threading
import time
from functools import wraps
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.gen_dataset_fixed_len import (
    generate_gsm8k_dataset,
    generate_mm_dataset,
    generate_random_dataset,
    save_jsonl,
)
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
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

AISBENCHMARK = "aisbench"
BENCHSERVING = "bench-serving"
BENCHMARK_TOOL_DEFAULT = BENCHSERVING
AISBENCHMARK_DATASET_GSM8K = "gsm8k"
AISBENCHMARK_DATASET_SHAREGPT = "sharegpt"
AISBENCHMARK_DATASET_MM_CUSTOM_GEN = "mm-custom-gen"
AISBENCHMARK_DATASET_DEFAULT = AISBENCHMARK_DATASET_GSM8K

SHAREGPT_DATASET_TEST_FILE = "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
GSM8K_DATASET_TEST_FILE = (
    "/root/.cache/modelscope/hub/datasets/grade_school_math/test.jsonl"
)
GSM8K_DATASET_TRAIN_FILE = (
    "/root/.cache/modelscope/hub/datasets/grade_school_math/train.jsonl"
)

PYTHON_FOR_TEST_TOOL = "python_venv_for_test_tool/bin/python"
if not os.path.exists(PYTHON_FOR_TEST_TOOL) or not os.access(
    PYTHON_FOR_TEST_TOOL, os.X_OK
):
    PYTHON_FOR_TEST_TOOL = "python3"
logger.info(f"PYTHON_FOR_TEST_TOOL: {PYTHON_FOR_TEST_TOOL}")

DEEPSEEK_R1_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"
)
DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"
)
DEEPSEEK_V32_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8"
)
QWEN3_8B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B-W8A8"
QWEN3_8B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Eagle3-Qwen3-8B-zh"
QWEN3_14B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-14B"
QWEN3_14B_LORA_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-14B-Lora/Qwen3-14B_lora"
)
QWEN3_14B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-14B-W8A8-Dynamic2"
)
QWEN3_14B_EAGLE_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/AngelSlim/Qwen3-14B_eagle3"
)
QWEN3_5_27B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B"
QWEN3_5_27B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.5-27B-W8A8"
)
QWEN3_30B_A3B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
)
QWEN3_6_35B_A3B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B"
QWEN3_6_27B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3.6-27B"
QWEN3_6_27B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.6-27B-w8a8"
)
QWEN3_30B_A3B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8"
)
QWEN3_30B_A3B_W8A8_VLLM_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-30B-A3B-W8A8"
)
QWEN3_A3B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-a3B_eagle3"
QWEN3_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
QWEN3_32B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
)
QWEN3_32B_EAGLE_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Eagle3-Qwen3-32B-zh"
)
QWEN3_235B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B"
QWEN3_235B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"
)
QWEN3_235B_A22B_EAGLE_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"
)
QWEN3_235B_A22B_INSTRUCT_2507_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/zcgy26/Qwen3-235B-A22B-Instruct-2507-w8a8"
)
QWEN3_480B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"
)
QWEN3_NEXT_80B_A3B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Instruct"
)
QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8"
)
QWEN3_CODER_NEXT_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-Next-W8A8"
)
GLM_4_6_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/GLM-4.6-w8a8_WITH_MTP"

QWEN3_VL_8B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_30B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-30B-A3B-Instruct"
)
QWEN3_VL_235B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-235B-A22B-Instruct"
)
QWEN2_5_VL_72B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-72B-Instruct-w8a8"
)
KIMI_K2_5_W4A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Eco-Tech/Kimi-K2.5-w4a8"
KIMI_K2_5_EAGLE3_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/lightseekorg/kimi-k2.5-eagle3"
)
GLM_4_7_FLASH_MODEL_PATH = "/root/.cache/modelscope/hub/models/ZhipuAI/GLM-4.7-Flash"
GLM_5_1_W4A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Eco-Tech/GLM-5.1-w4a8"
MINIMAX_M2_5_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Eco-Tech/MiniMax-M2.5-w8a8-QuaRot"
)
MINIMAX_M2_5_EAGLE3_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/sgl-npu/MiniMax-M2.5-eagel-model-0318"
)

QWEN3_5_397B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp"
)
QWEN3_5_397B_W4A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp"
)

ROUND_ROBIN = "round_robin"

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3600
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
E2E_TOLERANCE = 1.02  # +2%
OUTPUT_TOKEN_THROUGHPUT_TOLERANCE = 0.98  # -2%

# Package filtering keywords
PACKAGE_FILTER_KEYWORDS = [
    "sglang",
    "sgl",
    "torch",
    "deep-ep",
    "memfabric_hybrid",
]

if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
    )
else:
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_VISIBLE_DEVICES", "0")[0]) * 100
    )
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_SERVER_PORT_FOR_TEST + 66}"


def retry(max_attempts: int = None):
    """
        Test case retry decorator
    Args:
        max_attempts (int): Maximum number of execution attempts. If None, use self.max_attempts.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Store the last exception for final reporting
            last_exception = None

            # Get max_attempts from instance if not provided in decorator
            attempts = max_attempts or getattr(self, "max_attempts", 2)

            # Execute the test up to max_attempts times
            for attempt in range(1, attempts + 1):
                try:
                    logger.info(f"Executing test attempt {attempt}/{attempts}")
                    return func(
                        self, *args, **kwargs
                    )  # Return immediately if test passes
                except (AssertionError, Exception) as e:
                    last_exception = e
                    logger.info(f"Test failed on attempt {attempt}")

            # Raise the last exception if all attempts failed
            raise last_exception

        return wrapper

    return decorator


def get_cann_version():
    """Get CANN version info.

    Returns:
        str: CANN version info string.
    """
    cann_info_file = "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
    cann_ver_num = None

    try:
        with open(cann_info_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    cann_ver_num = line.strip().split("=")[-1]
                    break

        if cann_ver_num:
            cann_version_info = f"CANN: {cann_ver_num}"
            logger.info(cann_version_info)
            return cann_version_info
        else:
            logger.info("CANN version not found")
            return f"CANN: {cann_ver_num}"

    except FileNotFoundError:
        logger.error(f"CANN info file not found: {cann_info_file}")
        return f"CANN: {cann_ver_num}"
    except Exception as e:
        logger.error(f"Error reading CANN info: {e}")
        return f"CANN: {cann_ver_num}"


def write_pkg_info_to_file(result_file):
    """Write package information to result file.

    Args:
        result_file (str): Path to the result file.
    """
    import transformers

    try:
        pip_output = subprocess.run(
            ["pip", "list"], capture_output=True, text=True, check=False
        )
        packages = pip_output.stdout

        # Filter relevant packages using list comprehension
        filtered_packages = [
            line
            for line in packages.split("\n")
            if any(keyword in line for keyword in PACKAGE_FILTER_KEYWORDS)
        ]

        # Write to result file
        os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            for pkg in filtered_packages:
                f.write(pkg + "\n")
                logger.info(pkg)
            f.write(get_cann_version() + "\n")
            transformers_version_info = (
                "transformers: " + transformers.__version__ + "\n"
            )
            f.write(transformers_version_info)
            logger.info(transformers_version_info)

    except Exception as e:
        logger.error(f"Error getting packages: {e}")


def run_bench_serving(
    host,
    port,
    model_path=None,
    backend="sglang",
    dataset_name=None,
    dataset_path=None,
    request_rate=None,
    max_concurrency=None,
    num_prompts=None,
    input_len=None,
    output_len=None,
    random_range_ratio=1,
    image_resolution=None,
    image_count=None,
    warmup_requests=None,
    seed=None,
    output_file=None,
):
    metrics_path = os.getenv("METRICS_DATA_FILE")
    result_file = (
        "./bench_log.txt"
        if not metrics_path
        else f"{metrics_path}/bench_serving_metrics.txt"
    )
    logger.info(f"The metrics result file: {result_file}")

    write_pkg_info_to_file(result_file)

    cmd_args = [
        PYTHON_FOR_TEST_TOOL,
        "-m",
        "sglang.bench_serving",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model_path,
        "--backend",
        backend,
    ]

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
    if image_resolution:
        cmd_args.extend(["--image-resolution", str(image_resolution)])
    if image_count:
        cmd_args.extend(["--image-count", str(image_count)])
    if warmup_requests:
        cmd_args.extend(["--warmup-requests", str(warmup_requests)])
    if seed:
        cmd_args.extend(["--seed", str(seed)])
    if output_file:
        cmd_args.extend(["--output-file", str(output_file)])
    logger.info(f"Command: {' '.join(cmd_args)}")

    # Run benchmark command and capture output
    metrics = {"mean_ttft": None, "mean_tpot": None, "total_tps": None}

    process = subprocess.Popen(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    try:
        # Read output line by line
        with open(result_file, "a", encoding="utf-8") as f:
            for line in process.stdout:
                if line.strip():
                    print(line, end="")
                f.write(line)
                stripped_line = line.strip()

                # Extract metrics
                if "Mean TTFT" in stripped_line:
                    parts = stripped_line.split()
                    if len(parts) >= 4:
                        metrics["mean_ttft"] = parts[3]
                elif "Mean TPOT" in stripped_line:
                    parts = stripped_line.split()
                    if len(parts) >= 4:
                        metrics["mean_tpot"] = parts[3]
                elif "Output token throughput" in stripped_line:
                    parts = stripped_line.split()
                    if len(parts) >= 5:
                        metrics["total_tps"] = parts[4]
                elif "Mean E2E Latency" in stripped_line:
                    parts = stripped_line.split()
                    if len(parts) >= 5:
                        metrics["mean_e2e_latency"] = parts[4]
        process.wait()
        if process.returncode != 0:
            logger.error(
                f"Benchmark command failed with return code: {process.returncode}"
            )
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
    finally:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()

    return metrics


def run_aisbench(
    host,
    port,
    model_path,
    dataset_type,
    dataset_path,
    input_len,
    output_len,
    max_concurrency,
    num_prompts,
    image_resolution=None,
    random_range_ratio=1,
    prefix_hit_rate=None,
    aisbench_request_rate=None,
    aisbench_repeat_rate=None,
    dp=None,
    generation_kwargs=None,
):

    if dataset_type == "sharegpt":
        dataset_file = f"/tmp/datasets/test.jsonl"
        if not os.path.exists(dataset_file):
            logger.info(
                f"Generating random dataset from ShareGPT: {dataset_file}, "
                f"model_path={model_path}, batch_size={num_prompts}, input_len={input_len}"
            )
            generate_random_dataset(
                model_path=model_path,
                source_dataset_path=SHAREGPT_DATASET_TEST_FILE,
                batch_size=num_prompts,
                input_len=input_len,
                output_file=dataset_file,
                output_len=output_len,
                range_ratio=random_range_ratio,
            )
        dataset_path = dataset_file
        logger.info(f"Dataset generated: {dataset_path}")

    elif dataset_type == AISBENCHMARK_DATASET_GSM8K and not dataset_path:
        dataset_file = f"/tmp/datasets/test.jsonl"
        if not os.path.exists(dataset_file):
            logger.info(
                f"Generating gsm8k dataset: {dataset_file}, "
                f"model_path={model_path}, batch_size={num_prompts}, input_len={input_len}"
            )
            generate_gsm8k_dataset(
                model_path=model_path,
                source_dataset_path=GSM8K_DATASET_TEST_FILE,
                batch_size=num_prompts,
                input_len=input_len,
                output_file=dataset_file,
            )
        dataset_path = dataset_file
        logger.info(f"Dataset generated: {dataset_path}")

    elif dataset_type == AISBENCHMARK_DATASET_MM_CUSTOM_GEN and not dataset_path:
        dataset_file = f"/tmp/datasets/mm.jsonl"
        if not os.path.exists(dataset_file):
            image_dir = f"/tmp/datasets/images"
            data = generate_mm_dataset(
                train_path=GSM8K_DATASET_TRAIN_FILE,
                test_path=GSM8K_DATASET_TEST_FILE,
                tokenizer_path=model_path,
                target_tokens=input_len,
                num_prompts=num_prompts,
                image_dir=image_dir,
                size=image_resolution,
                trust_remote_code=True,
            )
            save_jsonl(data, dataset_file)
        dataset_path = dataset_file
        logger.info(f"Dataset generated: {dataset_file}")

    else:
        logger.info(f"Use exist dataset: {dataset_path}")

    metrics_path = os.getenv("METRICS_DATA_FILE")
    result_path = "./aisbench_result" if not metrics_path else metrics_path
    logger.info(f"The metrics result file: {result_path}")

    cmd = f"/bin/bash /root/sglang/python/sglang/test/ascend/e2e/run_aisbench.sh "
    cmd += f"--mode perf "
    cmd += f"--ip {host} "
    cmd += f"--port {str(port)} "
    cmd += f"--model {os.path.basename(model_path)} "
    cmd += f"--model-path {model_path} "
    cmd += f"--dataset-type {dataset_type} "
    cmd += f"--dataset-path {dataset_path} "
    cmd += f"--input-len {str(input_len)} "
    cmd += f"--output-len {str(output_len)} "
    cmd += f"--batch-size {str(max_concurrency)} "
    cmd += f"--num-prompts {str(num_prompts)} "
    cmd += f"--output-path {result_path}"

    if prefix_hit_rate is not None:
        cmd += f" --prefix-hit-rate {prefix_hit_rate}"
    if aisbench_request_rate is not None:
        cmd += f" --request_rate {aisbench_request_rate}"
    if aisbench_repeat_rate is not None:
        cmd += f" --repeat_rate {aisbench_repeat_rate}"
    if dp is not None:
        cmd += f" --dp {dp}"
    if generation_kwargs:
        cmd += f" --generation-kwargs '{generation_kwargs}'"

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

        simplified_output = re.sub(r"[^\w\s.]", " ", full_output)

        tpot_match = re.search(r"TPOT\s+total\s+([\d.]+)\s+ms", simplified_output)
        if tpot_match:
            metrics["mean_tpot"] = tpot_match.group(1)
            logger.info(f"Extracted mean_tpot: {metrics['mean_tpot']} ms")
        else:
            logger.warning("Could not extract mean_tpot from output")
            logger.error(
                f"Simplified output snippet around TPOT: {simplified_output[simplified_output.find('TPOT')-20:simplified_output.find('TPOT')+50] if 'TPOT' in simplified_output else 'TPOT not found'}"
            )

        tps_matches = re.findall(
            r"Output\s+Token\s+Throughput\s+total\s+([\d.]+)\s+token\s*/?\s*s",
            simplified_output,
        )
        if len(tps_matches) < 2:
            tps_matches += re.findall(
                r"OutputTokenThroughput\s+total\s+([\d.]+)\s+token\s*/?\s*s",
                simplified_output,
            )

        logger.info(
            f"Found {len(tps_matches)} matches for Output Token Throughput: {tps_matches}"
        )
        if tps_matches:
            # The first match is from the Common Metric section, which is the total throughput
            metrics["total_tps"] = tps_matches[0]
            if len(tps_matches) >= 2:
                logger.info(
                    f"Extracted total_tps: {metrics['total_tps']} token/s (from Common Metric section)"
                )
            else:
                logger.info(
                    f"Extracted total_tps: {metrics['total_tps']} token/s (only one match found)"
                )
        else:
            logger.warning("Could not extract total_tps from output")
            logger.warning(
                f"Simplified output snippet around Output Token Throughput: {simplified_output[simplified_output.find('Output')-20:simplified_output.find('Output')+100] if 'Output' in simplified_output else 'Output not found'}"
            )

        ttft_match = re.search(r"TTFT\s+total\s+([\d.]+)\s+ms", simplified_output)
        if ttft_match:
            metrics["mean_ttft"] = ttft_match.group(1)
            logger.info(f"Extracted mean_ttft: {metrics['mean_ttft']} ms")
        else:
            logger.warning("Could not extract mean_ttft from output")
            logger.warning(
                f"Simplified output snippet around TTFT: {simplified_output[simplified_output.find('TTFT')-20:simplified_output.find('TTFT')+50] if 'TTFT' in simplified_output else 'TTFT not found'}"
            )

        e2el_match = re.search(r"E2EL\s+total\s+([\d.]+)\s+ms", simplified_output)
        if e2el_match:
            metrics["mean_e2e_latency"] = e2el_match.group(1)
            logger.info(f"Extracted mean_e2e_latency: {metrics['mean_e2e_latency']} ms")
        else:
            logger.warning("Could not extract mean_e2e_latency from output")
            logger.warning(
                f"Simplified output snippet around E2EL: {simplified_output[simplified_output.find('E2EL')-20:simplified_output.find('E2EL')+50] if 'E2EL' in simplified_output else 'E2EL not found'}"
            )

        concurrency_match = re.search(
            r"Concurrency\s+total\s+([\d.]+)", simplified_output
        )
        if concurrency_match:
            metrics["concurrency"] = concurrency_match.group(1)
            logger.info(f"Extracted concurrency: {metrics['concurrency']}")
        else:
            logger.warning("Could not extract concurrency from output")
            logger.warning(
                f"Simplified output snippet around Concurrency: {simplified_output[simplified_output.find('Concurrency')-20:simplified_output.find('Concurrency')+50] if 'Concurrency' in simplified_output else 'Concurrency not found'}"
            )

        max_concurrency_match = re.search(
            r"Max\s+Concurrency\s+total\s+([\d.]+)", simplified_output
        )
        if max_concurrency_match:
            metrics["max_concurrency"] = max_concurrency_match.group(1)
            logger.info(f"Extracted max_concurrency: {metrics['max_concurrency']}")
        else:
            logger.warning("Could not extract max_concurrency from output")
            logger.warning(
                f"Simplified output snippet around Max Concurrency: {simplified_output[simplified_output.find('Max Concurrency')-20:simplified_output.find('Max Concurrency')+50] if 'Max Concurrency' in simplified_output else 'Max Concurrency not found'}"
            )

        req_throughput_match = re.search(
            r"Request\s+Throughput\s+total\s+([\d.]+)\s+req\s*/?\s*s",
            simplified_output,
        )
        if req_throughput_match:
            metrics["request_throughput"] = req_throughput_match.group(1)
            logger.info(
                f"Extracted request_throughput: {metrics['request_throughput']} req/s"
            )
        else:
            logger.warning("Could not extract request_throughput from output")
            logger.warning(
                f"Simplified output snippet around Request Throughput: {simplified_output[simplified_output.find('Request')-20:simplified_output.find('Request')+50] if 'Request' in simplified_output else 'Request not found'}"
            )

        total_requests_match = re.search(
            r"Total\s+Requests\s+total\s+(\d+)", simplified_output
        )
        if total_requests_match:
            metrics["total_requests"] = total_requests_match.group(1)
            logger.info(f"Extracted total_requests: {metrics['total_requests']}")
        else:
            logger.warning("Could not extract total_requests from output")
            logger.warning(
                f"Simplified output snippet around Total Requests: {simplified_output[simplified_output.find('Total Requests')-20:simplified_output.find('Total Requests')+50] if 'Total Requests' in simplified_output else 'Total Requests not found'}"
            )

        failed_requests_match = re.search(
            r"Failed\s+Requests\s+total\s+(\d+)", simplified_output
        )
        if failed_requests_match:
            metrics["failed_requests"] = failed_requests_match.group(1)
            logger.info(f"Extracted failed_requests: {metrics['failed_requests']}")
        else:
            logger.warning("Could not extract failed_requests from output")
            logger.warning(
                f"Simplified output snippet around Failed Requests: {simplified_output[simplified_output.find('Failed Requests')-20:simplified_output.find('Failed Requests')+50] if 'Failed Requests' in simplified_output else 'Failed Requests not found'}"
            )

        logger.info(f"All extracted metrics: {metrics}")

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
    """Assert benchmark metrics against expected values.

    Args:
        metrics (dict): Benchmark metrics dictionary.
    """
    if not metrics:
        raise Exception("No metrics obtained from benchmark")

    if self.tpot:
        if self.tpot < TPOT_THRESHOLD:
            self.assertLessEqual(
                float(metrics["mean_tpot"]),
                self.tpot + TPOT_TOLERANCE_LOW,
            )
        else:
            self.assertLessEqual(
                float(metrics["mean_tpot"]),
                self.tpot * TPOT_TOLERANCE_HIGH,
            )
    if self.output_token_throughput:
        self.assertGreaterEqual(
            float(metrics["total_tps"]),
            self.output_token_throughput * OUTPUT_TOKEN_THROUGHPUT_TOLERANCE,
        )
    if self.ttft:
        self.assertLessEqual(
            float(metrics["mean_ttft"]),
            self.ttft * TTFT_TOLERANCE,
        )
    if self.mean_e2e_latency:
        self.assertLessEqual(
            float(metrics["mean_e2e_latency"]),
            self.mean_e2e_latency * E2E_TOLERANCE,
        )


class TestAscendPerformanceTestCaseBase(CustomTestCase):
    model = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "random"
    dataset_path = SHAREGPT_DATASET_TEST_FILE
    aisbench_dataset_type = "gsm8k"  # gsm8k | mm-custom-gen
    aisbench_dataset_path = None  # auto generate dataset if none
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    max_attempts = 2
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = 1
    image_resolution = None
    image_count = None
    warmup_requests = None
    seed = None
    ttft = None
    tpot = None
    mean_e2e_latency = None
    output_token_throughput = None

    prefix_hit_rate = None
    aisbench_request_rate = None
    aisbench_repeat_rate = None
    dp = None
    generation_kwargs = None

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
            timeout=cls.timeout,
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

    @retry()
    def run_throughput(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=host,
                port=port,
                model_path=self.model,
                dataset_type=self.aisbench_dataset_type,
                dataset_path=self.aisbench_dataset_path,
                input_len=self.input_len,
                output_len=self.output_len,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
                image_resolution=self.image_resolution,
                random_range_ratio=self.random_range_ratio,
                prefix_hit_rate=self.prefix_hit_rate,
                aisbench_request_rate=self.aisbench_request_rate,
                aisbench_repeat_rate=self.aisbench_repeat_rate,
                dp=self.dp,
                generation_kwargs=self.generation_kwargs,
            )
            assert_metrics(self, metrics)

        else:
            bench_params = {
                "host": host,
                "port": port,
                "model_path": self.model,
                "backend": self.backend,
                "dataset_name": self.dataset_name,
                "dataset_path": self.dataset_path,
                "request_rate": self.request_rate,
                "max_concurrency": self.max_concurrency,
                "num_prompts": self.num_prompts,
                "input_len": self.input_len,
                "output_len": self.output_len,
                "random_range_ratio": self.random_range_ratio,
                "image_resolution": self.image_resolution,
                "image_count": self.image_count,
                "warmup_requests": self.warmup_requests,
                "seed": self.seed,
            }
            logger.info(f"Starting benchmark with parameters: {bench_params}")
            metrics = run_bench_serving(**bench_params)
            assert_metrics(self, metrics)


class TestAscendPerfMultiNodePdMixTestCaseBase(CustomTestCase):
    model_config = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "random"
    dataset_path = SHAREGPT_DATASET_TEST_FILE
    aisbench_dataset_type = "gsm8k"  # gsm8k | mm-custom-gen
    aisbench_dataset_path = None  # auto generate dataset if none
    max_attempts = 2
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = 1
    image_resolution = None
    image_count = None
    warmup_requests = None
    seed = None
    ttft = None
    tpot = None
    mean_e2e_latency = None
    output_token_throughput = None

    prefix_hit_rate = None
    aisbench_request_rate = None
    aisbench_repeat_rate = None
    dp = None
    generation_kwargs = None

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

    @retry()
    @check_role(allowed_roles=["master", "worker"])
    def run_throughput(self):
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=self.host,
                port=str(self.port),
                model_path=self.model_config.get("model_path"),
                dataset_type=self.aisbench_dataset_type,
                dataset_path=self.aisbench_dataset_path,
                input_len=self.input_len,
                output_len=self.output_len,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
                image_resolution=self.image_resolution,
                random_range_ratio=self.random_range_ratio,
                prefix_hit_rate=self.prefix_hit_rate,
                aisbench_request_rate=self.aisbench_request_rate,
                aisbench_repeat_rate=self.aisbench_repeat_rate,
                dp=self.dp,
                generation_kwargs=self.generation_kwargs,
            )
            assert_metrics(self, metrics)

        else:
            bench_params = {
                "host": self.host,
                "port": str(self.port),
                "model_path": self.model_config.get("model_path"),
                "backend": self.backend,
                "dataset_name": self.dataset_name,
                "dataset_path": self.dataset_path,
                "request_rate": self.request_rate,
                "max_concurrency": self.max_concurrency,
                "num_prompts": self.num_prompts,
                "input_len": self.input_len,
                "output_len": self.output_len,
                "random_range_ratio": self.random_range_ratio,
                "image_resolution": self.image_resolution,
                "image_count": self.image_count,
                "warmup_requests": self.warmup_requests,
                "seed": self.seed,
            }
            logger.info(f"Starting benchmark with parameters: {bench_params}")
            metrics = run_bench_serving(**bench_params)
            assert_metrics(self, metrics)


class TestAscendPerfMultiNodePdSepTestCaseBase(CustomTestCase):
    model_config = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "random"
    dataset_path = SHAREGPT_DATASET_TEST_FILE
    aisbench_dataset_type = "gsm8k"  # gsm8k | mm-custom-gen
    aisbench_dataset_path = None  # auto generate dataset if none
    max_attempts = 2
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = 1
    image_resolution = None
    image_count = None
    warmup_requests = None
    seed = None
    ttft = None
    tpot = None
    mean_e2e_latency = None
    output_token_throughput = None

    prefix_hit_rate = None
    aisbench_request_rate = None
    aisbench_repeat_rate = None
    dp = None
    generation_kwargs = None

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

        # Loop to check if the process is still running
        while True:
            if cls.process.poll() is None:
                # Process is still running
                time.sleep(30)
            else:
                # Process has exited
                exit_code = cls.process.poll()
                raise Exception(
                    f"Sglang process exited on node {cls.host} {cls.hostname} with exit code: {exit_code}"
                )

    @retry()
    @check_role(allowed_roles=["router"])
    def run_throughput(self):
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=self.host,
                port=str(self.port),
                model_path=self.model_config.get("model_path"),
                dataset_type=self.aisbench_dataset_type,
                dataset_path=self.aisbench_dataset_path,
                input_len=self.input_len,
                output_len=self.output_len,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
                image_resolution=self.image_resolution,
                random_range_ratio=self.random_range_ratio,
                prefix_hit_rate=self.prefix_hit_rate,
                aisbench_request_rate=self.aisbench_request_rate,
                aisbench_repeat_rate=self.aisbench_repeat_rate,
                dp=self.dp,
                generation_kwargs=self.generation_kwargs,
            )
            assert_metrics(self, metrics)

        else:
            bench_params = {
                "host": self.host,
                "port": str(self.port),
                "model_path": self.model_config.get("model_path"),
                "backend": self.backend,
                "dataset_name": self.dataset_name,
                "dataset_path": self.dataset_path,
                "request_rate": self.request_rate,
                "max_concurrency": self.max_concurrency,
                "num_prompts": self.num_prompts,
                "input_len": self.input_len,
                "output_len": self.output_len,
                "random_range_ratio": self.random_range_ratio,
                "image_resolution": self.image_resolution,
                "image_count": self.image_count,
                "warmup_requests": self.warmup_requests,
                "seed": self.seed,
            }
            logger.info(f"Starting benchmark with parameters: {bench_params}")
            metrics = run_bench_serving(**bench_params)
            assert_metrics(self, metrics)
