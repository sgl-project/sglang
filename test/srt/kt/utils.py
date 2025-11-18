"""
Utility functions for KT-kernel tests

Provides common functionality for:
- Building KT-specific server arguments
- Running inference requests
- Test configuration
"""

import os
from typing import List, Optional

import requests


def get_kt_model_paths():
    """Get model paths from environment variables"""
    model_dir = os.getenv("SGLANG_CI_MODEL_DIR", "models")
    kt_weight_dir = os.getenv("SGLANG_CI_KT_WEIGHT_DIR", "models")

    return {
        "gpu_model_path": os.path.join(model_dir, "DeepSeek-R1-0528-GPU-weight"),
        "cpu_model_path": os.path.join(kt_weight_dir, "DeepSeek-R1-0528-CPU-weight"),
        "served_model_name": os.getenv("SERVED_MODEL_NAME", "DeepSeek-R1-0528-FP8"),
    }


def get_kt_server_args(
    kt_weight_path: str,
    kt_num_gpu_experts: int = 200,
    kt_cpuinfer: int = 60,
    kt_threadpool_count: int = 2,
    kt_method: str = "AMXINT4",
    tensor_parallel_size: int = 1,
    served_model_name: Optional[str] = None,
    max_running_requests: int = 40,
    max_total_tokens: int = 40000,
    additional_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Build server arguments for KT-kernel configuration

    Args:
        kt_weight_path: Path to CPU weight for KT
        kt_num_gpu_experts: Number of GPU experts
        kt_cpuinfer: Number of CPU experts
        kt_threadpool_count: Thread pool count
        kt_method: KT method (e.g., AMXINT4)
        tensor_parallel_size: Tensor parallel size
        served_model_name: Model name to serve
        max_running_requests: Max running requests
        max_total_tokens: Max total tokens
        additional_args: Additional arguments

    Returns:
        List of server arguments
    """
    args = [
        "--attention-backend",
        "triton",
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.98",
        "--chunked-prefill-size",
        "4096",
        "--max-running-requests",
        str(max_running_requests),
        "--max-total-tokens",
        str(max_total_tokens),
        "--enable-mixed-chunk",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--enable-p2p-check",
        "--disable-shared-experts-fusion",
    ]

    if served_model_name:
        args.extend(["--served-model-name", served_model_name])

    # KT-specific arguments
    args.extend(
        [
            "--kt-weight-path",
            kt_weight_path,
            "--kt-cpuinfer",
            str(kt_cpuinfer),
            "--kt-threadpool-count",
            str(kt_threadpool_count),
            "--kt-num-gpu-experts",
            str(kt_num_gpu_experts),
            "--kt-method",
            kt_method,
        ]
    )

    if additional_args:
        args.extend(additional_args)

    return args


def get_kt_env():
    """Get environment variables for KT tests"""
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def run_inference(
    base_url: str,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> List[str]:
    """
    Run inference on prompts and return generated texts

    Args:
        base_url: Server base URL
        prompts: List of input prompts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)

    Returns:
        List of generated texts
    """
    outputs = []

    for prompt in prompts:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )

        if response.status_code == 200:
            result = response.json()
            outputs.append(result["choices"][0]["text"])
        else:
            raise RuntimeError(
                f"Inference failed: {response.status_code} {response.text}"
            )

    return outputs


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts (simple token overlap)

    Returns:
        Similarity score in [0, 1]
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union)


def get_cpu_memory_usage() -> float:
    """Get current CPU memory usage in GB"""
    return psutil.virtual_memory().used / (1024**3)


def get_gpu_memory_usage() -> List[float]:
    """Get GPU memory usage for all visible GPUs in GB"""
    import torch

    if not torch.cuda.is_available():
        return []

    usage = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        usage.append(allocated)

    return usage


def load_test_config() -> Dict:
    """
    Load test configuration from environment variables

    Returns:
        Dict with test configuration (model paths, etc.)
    """
    config = {
        "model_dir": os.getenv("SGLANG_CI_MODEL_DIR", "/data/sglang-ci/models"),
        "kt_weight_dir": os.getenv(
            "SGLANG_CI_KT_WEIGHT_DIR", "/data/sglang-ci/kt-weights"
        ),
        "hf_cache": os.getenv("HF_HOME", "/data/sglang-ci/hf-cache"),
        "cuda_devices": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
    }

    # DeepSeek-V3 model paths
    config["deepseek_v3_gpu"] = os.path.join(config["model_dir"], "DeepSeek-V3-FP8")
    config["deepseek_v3_kt"] = os.path.join(config["kt_weight_dir"], "DeepSeek-V3-INT4")

    return config


def verify_kt_installation() -> Tuple[bool, str]:
    """
    Verify KT-kernel installation

    Returns:
        (success, message) tuple
    """
    try:
        # Check kt_kernel import
        # Check AMX support
        import torch
        from kt_kernel import KTMoEWrapper

        if not torch._C._cpu._is_amx_tile_supported():
            return False, "AMX instructions not supported on this CPU"

        return True, "KT-kernel installed and AMX supported"

    except ImportError as e:
        return False, f"Failed to import kt_kernel: {e}"
    except Exception as e:
        return False, f"KT verification failed: {e}"


# Common test prompts
TEST_PROMPTS = [
    "What is the capital of France?",
    "Write a Python function to calculate fibonacci numbers.",
    "Explain quantum computing in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "How does a neural network learn?",
]

# Performance test prompts (longer inputs)
PERF_TEST_PROMPTS = [
    "Write a detailed explanation of machine learning, including supervised learning, unsupervised learning, and reinforcement learning. Provide examples for each type.",
    "Explain the theory of relativity and its implications for modern physics. Include both special and general relativity.",
    "Describe the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
]
