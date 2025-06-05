# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

from sglang.srt.layers.moe.ep_moe.kernels import grouped_gemm_triton
from sglang.srt.utils import is_hip

_is_hip_ = is_hip()

# ===== Logging Configuration =====


def setup_logging(log_file=None, log_level=logging.INFO):
    """设置日志配置，使日志能够正确打印到控制台或文件"""
    # 创建日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 如果指定了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


# 在文件开头调用setup_logging
logger = setup_logging()

# ===== Type Definitions =====


class BenchmarkConfig(TypedDict):
    """Configuration for the grouped GEMM kernel benchmark."""

    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int


# Define a set of TOKENS_PER_EXPERT values to tune for
# TOKENS_PER_EXPERT_VALUES = [1, 2, 4, 7, 8, 16, 32, 64 ,128 ,160, 256]
TOKENS_PER_EXPERT_VALUES = [160]

# ===== Configuration Utilities =====


def get_config_dtype_str(
    dtype: torch.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
) -> str:
    """Convert dtype and quantization settings to a string representation."""
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    else:
        return str(dtype).split(".")[-1]


def get_config_file_name(
    hidden_size: int,
    dtype_str: str,
    block_shape: Optional[List[int]] = None,
) -> str:
    """Generate a filename for saving the configuration."""
    block_str = ""
    if block_shape is not None:
        block_str = f"_block{block_shape[0]}x{block_shape[1]}"
    return f"grouped_gemm_config_{hidden_size}_{dtype_str}{block_str}.json"


def get_default_config(
    batch_size: int,
    hidden_size: int,
    dtype_str: str,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    """Get default configuration based on batch size and hidden size."""
    if batch_size <= 32:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
        }
    elif batch_size <= 128:
        return {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
        }
    else:
        return {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
        }


def get_rocm_configs_compute_bound() -> List[Dict[str, int]]:
    """Generate configurations for ROCm (AMD) GPUs."""
    configs: List[BenchmarkConfig] = []
    for block_m in [32, 64, 128, 256]:
        for block_k in [32, 64, 128, 256]:
            for block_n in [16, 32, 64, 128, 256]:
                configs.append(
                    {
                        "BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "BLOCK_SIZE_K": block_k,
                    }
                )
    return configs


def get_configs_compute_bound() -> List[Dict[str, int]]:
    """Generate configurations for compute-bound workloads."""
    # Reduced search space for faster tuning
    configs: List[BenchmarkConfig] = []
    if _is_hip_:
        configs = get_rocm_configs_compute_bound()
    else:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128, 256]:
                for block_n in [32, 64, 128, 256]:
                    configs.append(
                        {
                            "BLOCK_SIZE_M": block_m,
                            "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_K": block_k,
                        }
                    )
    return configs


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    """Sort configuration parameters for consistent representation."""
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
    }


# ===== Benchmarking Functions =====


def prepare_tensors(
    total_tokens: int,
    hidden_size: int,
    num_experts_per_partition: int,
    intermediate_size: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    block_shape: Optional[List[int]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Prepare tensors for benchmarking."""
    # 确定数据类型，与实际代码保持一致
    if use_fp8_w8a8 and block_shape is None:
        # 如果使用FP8量化且不是块量化，使用fp8_dtype
        fp8_dtype = torch.float8_e4m3fn
        a_dtype = fp8_dtype
    else:
        # 否则使用指定的dtype
        a_dtype = dtype

    # 创建输入张量a，形状为[total_tokens, hidden_size]
    gateup_input = torch.rand((total_tokens, hidden_size), device="cuda", dtype=a_dtype)
    logger.info(
        f"Created input tensor a with shape {gateup_input.shape} and dtype {gateup_input.dtype}"
    )

    # 创建权重张量b，形状为[num_experts_per_partition, 2 * intermediate_size, hidden_size]
    w13_weight = torch.nn.Parameter(
        torch.empty(
            num_experts_per_partition,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        requires_grad=False,
    )
    logger.info(
        f"Created w13_weight tensor a with shape {w13_weight.shape} and dtype {w13_weight.dtype}"
    )

    gateup_output = torch.empty(
        gateup_input.shape[0],
        w13_weight.shape[1],
        device="cuda",
        dtype=dtype,
    )
    logger.info(
        f"Created gateup_output tensor a with shape {gateup_output.shape} and dtype {gateup_output.dtype}"
    )

    # 创建分段指针张量，形状为[batch_size+1]
    tokens_per_expert_segment = total_tokens // num_experts_per_partition
    seg_indptr = torch.zeros(
        num_experts_per_partition + 1, dtype=torch.int64, device="cuda"
    )
    for i in range(1, num_experts_per_partition + 1):
        seg_indptr[i] = i * tokens_per_expert_segment

    logger.info(
        f"Created seg_indptr tensor a with shape {seg_indptr.shape} and dtype {seg_indptr.dtype}"
    )

    # 创建权重索引张量，形状为[batch_size]
    weight_indices = torch.zeros(
        num_experts_per_partition, dtype=torch.int64, device="cuda"
    )
    logger.info(
        f"Created weight_indices tensor a with shape {weight_indices.shape} and dtype {weight_indices.dtype}"
    )

    # 创建缩放张量（如果需要）
    scale_a = None
    scale_b = None

    if use_fp8_w8a8 and block_shape is None:
        # 如果使用FP8量化且不是块量化，创建缩放张量
        # scale_a 模拟 w13_input_scale，基于输入的最大值计算
        max_value = (
            torch.max(gateup_input).repeat(num_experts_per_partition).to(torch.float32)
        )
        scale_a = max_value / torch.finfo(torch.float8_e4m3fn).max

        # scale_b 模拟 w13_weight_scale，每个专家一个缩放因子
        scale_b = torch.ones(
            num_experts_per_partition, dtype=torch.float32, device="cuda"
        )
    elif block_shape is not None:
        # 如果使用块量化，创建块缩放张量
        block_n, block_k = block_shape[0], block_shape[1]
        n_tiles = (2 * intermediate_size + block_n - 1) // block_n
        k_tiles = (hidden_size + block_k - 1) // block_k
        # scale_a = torch.ones(num_experts_per_partition, dtype=torch.float32, device="cuda")
        scale_a = None
        scale_b = torch.ones(
            num_experts_per_partition,
            n_tiles,
            k_tiles,
            dtype=torch.float32,
            device="cuda",
        )

    logger.info(
        f"Created scale_b tensor a with shape {scale_b.shape} and dtype {scale_b.dtype}"
    )

    return (
        gateup_input,
        w13_weight,
        gateup_output,
        seg_indptr,
        weight_indices,
        scale_a,
        scale_b,
    )


def prepare_tensors_down(
    total_tokens: int,
    hidden_size: int,
    num_experts_per_partition: int,
    intermediate_size: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    block_shape: Optional[List[int]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Prepare tensors for benchmarking."""
    # 确定数据类型，与实际代码保持一致
    if use_fp8_w8a8 and block_shape is None:
        # 如果使用FP8量化且不是块量化，使用fp8_dtype
        fp8_dtype = torch.float8_e4m3fn
        a_dtype = fp8_dtype
    else:
        # 否则使用指定的dtype
        a_dtype = dtype

    # 创建输入张量a，形状为[total_tokens, hidden_size]
    gateup_input = torch.rand(
        (total_tokens, intermediate_size), device="cuda", dtype=a_dtype
    )
    logger.info(
        f"Created input tensor a with shape {gateup_input.shape} and dtype {gateup_input.dtype}"
    )

    # 创建权重张量b，形状为[num_experts_per_partition, 2 * intermediate_size, hidden_size]
    w13_weight = torch.nn.Parameter(
        torch.empty(
            num_experts_per_partition,
            hidden_size,
            intermediate_size,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        requires_grad=False,
    )
    logger.info(
        f"Created w13_weight tensor a with shape {w13_weight.shape} and dtype {w13_weight.dtype}"
    )

    gateup_output = torch.empty(
        gateup_input.shape[0],
        w13_weight.shape[1],
        device="cuda",
        dtype=dtype,
    )
    logger.info(
        f"Created gateup_output tensor a with shape {gateup_output.shape} and dtype {gateup_output.dtype}"
    )

    # 创建分段指针张量，形状为[batch_size+1]
    tokens_per_expert_segment = total_tokens // num_experts_per_partition
    seg_indptr = torch.zeros(
        num_experts_per_partition + 1, dtype=torch.int64, device="cuda"
    )
    for i in range(1, num_experts_per_partition + 1):
        seg_indptr[i] = i * tokens_per_expert_segment

    logger.info(
        f"Created seg_indptr tensor a with shape {seg_indptr.shape} and dtype {seg_indptr.dtype}"
    )

    # 创建权重索引张量，形状为[batch_size]
    weight_indices = torch.zeros(
        num_experts_per_partition, dtype=torch.int64, device="cuda"
    )
    logger.info(
        f"Created weight_indices tensor a with shape {weight_indices.shape} and dtype {weight_indices.dtype}"
    )

    # 创建缩放张量（如果需要）
    scale_a = None
    scale_b = None

    if use_fp8_w8a8 and block_shape is None:
        # 如果使用FP8量化且不是块量化，创建缩放张量
        # scale_a 模拟 w13_input_scale，基于输入的最大值计算
        max_value = (
            torch.max(gateup_input).repeat(num_experts_per_partition).to(torch.float32)
        )
        scale_a = max_value / torch.finfo(torch.float8_e4m3fn).max

        # scale_b 模拟 w13_weight_scale，每个专家一个缩放因子
        scale_b = torch.ones(
            num_experts_per_partition, dtype=torch.float32, device="cuda"
        )
    elif block_shape is not None:
        # 如果使用块量化，创建块缩放张量
        block_n, block_k = block_shape[0], block_shape[1]
        n_tiles = (2 * intermediate_size + block_n - 1) // block_n
        k_tiles = (hidden_size + block_k - 1) // block_k
        # scale_a = torch.ones(num_experts_per_partition, dtype=torch.float32, device="cuda")
        scale_a = None
        scale_b = torch.ones(
            num_experts_per_partition,
            n_tiles,
            k_tiles,
            dtype=torch.float32,
            device="cuda",
        )

    logger.info(
        f"Created scale_b tensor a with shape {scale_b.shape} and dtype {scale_b.dtype}"
    )

    return (
        gateup_input,
        w13_weight,
        gateup_output,
        seg_indptr,
        weight_indices,
        scale_a,
        scale_b,
    )


def benchmark_config(
    config: BenchmarkConfig,
    num_experts_per_partition: int,
    hidden_size: int,
    dtype: torch.dtype,
    tokens_per_expert: int,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    block_shape: Optional[List[int]] = None,
    model_config: Optional[AutoConfig] = None,
) -> float:
    """Benchmark a specific configuration for the grouped GEMM kernel."""
    logger.info(f"Benchmarking config: {config}")
    logger.info(
        f"Parameters: num_experts_per_partition={num_experts_per_partition}, hidden_size={hidden_size}, dtype={dtype}"
    )
    logger.info(
        f"Quantization: use_fp8_w8a8={use_fp8_w8a8}, use_int8_w8a8={use_int8_w8a8}, use_int8_w8a16={use_int8_w8a16}"
    )
    if block_shape:
        logger.info(f"Block shape: {block_shape}")

    # 从模型配置中获取top_k
    top_k = model_config.num_experts_per_tok

    # 使用输入的tokens_per_expert
    total_tokens = int(tokens_per_expert * top_k)

    # 获取intermediate_size
    intermediate_size = getattr(model_config, "moe_intermediate_size", hidden_size // 2)

    # 准备张量
    (
        gateup_input,
        w13_weight,
        gateup_output,
        seg_indptr,
        weight_indices,
        scale_a,
        scale_b,
    ) = prepare_tensors_down(
        total_tokens=total_tokens,
        hidden_size=hidden_size,
        num_experts_per_partition=num_experts_per_partition,
        intermediate_size=intermediate_size,
        dtype=dtype,
        use_fp8_w8a8=use_fp8_w8a8,
        block_shape=block_shape,
    )

    # 运行一次以预热
    from sglang.srt.layers.moe.ep_moe import override_config

    with override_config(config):
        grouped_gemm_triton(
            gateup_input,
            w13_weight,
            gateup_output,
            num_experts_per_partition,
            True,
            seg_indptr,
            weight_indices,
            use_fp8_w8a8,
            scale_a,
            scale_b,
            block_shape,
        )
    torch.cuda.synchronize()

    # 使用CUDA图捕获10次调用
    logger.info("Capturing CUDA graph")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            with override_config(config):
                grouped_gemm_triton(
                    gateup_input,
                    w13_weight,
                    gateup_output,
                    num_experts_per_partition,
                    True,
                    seg_indptr,
                    weight_indices,
                    use_fp8_w8a8,
                    scale_a,
                    scale_b,
                    block_shape,
                )
    torch.cuda.synchronize()

    # 预热
    logger.info("Warming up CUDA graph")
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    # 测量延迟
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    iters = 100
    logger.info(f"Running benchmark for {iters} iters")
    for _ in range(iters):
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    avg = sum(latencies) / iters * 1000  # us
    logger.info(f"Average latency: {avg:.2f} us")
    graph.reset()
    return avg


# ===== Ray Distributed Tuning =====


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    """Ray worker for distributed benchmarking."""

    def __init__(self, seed: int) -> None:
        """Initialize the worker with a specific seed."""
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(seed)
        self.seed = seed

    def tune(
        self,
        num_experts_per_partition: int,
        hidden_size: int,
        dtype: torch.dtype,
        tokens_per_expert: int,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        block_shape: Optional[List[int]],
        model_config: Optional[AutoConfig],
        search_space: List[Dict[str, int]],
    ) -> Tuple[Dict[str, int], float]:
        """Tune a subset of the search space and return the best config with its latency."""
        best_config = None
        best_time = float("inf")

        for config in tqdm(search_space):
            try:
                kernel_time = benchmark_config(
                    config=config,
                    num_experts_per_partition=num_experts_per_partition,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    tokens_per_expert=tokens_per_expert,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    block_shape=block_shape,
                    model_config=model_config,
                )

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
                    logger.info(
                        f"New best config: {best_config} with kernel_time={best_time:.2f} us"
                    )
            except Exception as e:
                logger.warning(f"Failed to benchmark config {config}: {e}")
                continue

        now = datetime.now()
        logger.info(
            f"{now.ctime()} Completed tuning for num_experts_per_partition={num_experts_per_partition}, hidden_size={hidden_size}"
        )
        # assert best_config is not None
        return best_config, best_time


def distribute_search_space(
    search_space: List[Dict[str, int]], num_workers: int
) -> List[List[Dict[str, int]]]:
    """Distribute the search space among workers."""
    # 确保每个worker至少获得一个配置
    if len(search_space) < num_workers:
        # 如果搜索空间小于worker数量，复制配置以填满worker数量
        search_space = search_space * (num_workers // len(search_space) + 1)
        search_space = search_space[:num_workers]

    # 计算每个worker应该处理的配置数量
    chunk_size = max(1, len(search_space) // num_workers)

    # 分配配置给workers
    search_chunks = []
    for i in range(0, len(search_space), chunk_size):
        chunk = search_space[i : i + chunk_size]
        if chunk:  # 只添加非空chunk
            search_chunks.append(chunk)

    # 确保chunk数量等于worker数量
    while len(search_chunks) < num_workers:
        # 如果chunk数量不足，从最大的chunk中分割
        largest_chunk_idx = max(
            range(len(search_chunks)), key=lambda i: len(search_chunks[i])
        )
        largest_chunk = search_chunks[largest_chunk_idx]
        mid = len(largest_chunk) // 2
        search_chunks[largest_chunk_idx] = largest_chunk[:mid]
        search_chunks.append(largest_chunk[mid:])

    return search_chunks


# ===== Configuration Saving =====
def save_config(
    config: Dict[str, int],
    hidden_size: int,
    dtype: torch.dtype,
    batch_size: int,
    block_shape: Optional[List[int]] = None,
) -> None:
    """Save the best configuration to a JSON file."""
    # 确定文件名
    dtype_str = get_config_dtype_str(dtype)
    block_shape_str = f"_block{block_shape[0]}x{block_shape[1]}" if block_shape else ""
    filename = f"grouped_gemm_config_{hidden_size}_{dtype_str}_b{batch_size}{block_shape_str}.json"

    # 确保目录存在
    directory = os.path.dirname(filename)
    if directory:  # 只有当目录路径不为空时才创建目录
        os.makedirs(directory, exist_ok=True)

    # 保存配置
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved best configuration to {filename}")


# ===== Model Configuration Loading =====


def load_model_config(
    model_path: str, args: argparse.Namespace
) -> Tuple[
    AutoConfig, int, torch.dtype, int, int, Optional[List[int]], bool, bool, bool
]:
    """Load model configuration and extract relevant parameters."""
    logger.info(f"Loading model config from {model_path}")
    try:
        # 尝试加载模型配置
        model_config = AutoConfig.from_pretrained(model_path)
        logger.info(f"Loaded model config: {model_config}")

        # 从模型配置中获取hidden_size
        hidden_size = getattr(model_config, "hidden_size", 4096)  # 默认值4096
        logger.info(f"Using hidden_size from model config: {hidden_size}")

        # 从模型配置中获取数据类型
        dtype = torch.bfloat16  # 默认值
        if hasattr(model_config, "torch_dtype"):
            dtype_str = model_config.torch_dtype
            if isinstance(dtype_str, str):
                if dtype_str == "bfloat16":
                    dtype = torch.bfloat16
                elif dtype_str == "float16":
                    dtype = torch.float16
                elif dtype_str == "float32":
                    dtype = torch.float32
            logger.info(f"Using dtype from model config: {dtype}")

        # 从模型配置中获取top_k (num_experts_per_tok)
        top_k = getattr(model_config, "num_experts_per_tok", 2)  # 默认值2
        logger.info(f"Using top_k (num_experts_per_tok) from model config: {top_k}")

        # 使用用户输入的ep_size，而不是从模型配置中获取
        ep_size = args.ep_size
        logger.info(f"Using ep_size from user input: {ep_size}")

        # 从模型配置中获取batch_size (n_routed_experts)
        n_routed_experts = getattr(model_config, "n_routed_experts", 8)  # 默认值8
        num_experts_per_partition = n_routed_experts // ep_size
        logger.info(
            f"Using num_experts_per_partition (n_routed_experts/ep_size) from model config: {num_experts_per_partition} = {n_routed_experts}/{ep_size}"
        )

        # 从模型配置中获取block_shape
        block_shape = None
        if hasattr(model_config, "quantization_config"):
            quant_config = model_config.quantization_config
            if isinstance(quant_config, dict) and "weight_block_size" in quant_config:
                block_shape = quant_config["weight_block_size"]
                logger.info(f"Using block_shape from model config: {block_shape}")

        # 确定量化设置
        use_fp8_w8a8 = False
        use_int8_w8a8 = False
        use_int8_w8a16 = False

        if hasattr(model_config, "quantization_config"):
            quant_config = model_config.quantization_config
            if isinstance(quant_config, dict):
                quant_method = quant_config.get("quant_method", "")
                if quant_method == "fp8":
                    use_fp8_w8a8 = True
                    logger.info("Using FP8 quantization")
                elif quant_method == "int8":
                    use_int8_w8a8 = True
                    logger.info("Using INT8 quantization")

        # 将模型配置传递给benchmark_config函数
        model_config.top_k = top_k

        return (
            model_config,
            hidden_size,
            dtype,
            top_k,
            num_experts_per_partition,
            block_shape,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
        )

    except Exception as e:
        logger.error(f"Failed to load model config: {e}")
        import traceback

        logger.error(traceback.format_exc())
        logger.error("Cannot proceed without model configuration")
        raise


# ===== Main Function =====


def main():
    """Main function to run the tuning process."""
    parser = argparse.ArgumentParser(description="Tune grouped GEMM Triton kernel")
    parser.add_argument(
        "--model",
        type=str,
        default="/data00/models/DeepSeek-R1/",
        help="Path to model config or directory",
    )
    parser.add_argument(
        "--log-file", type=str, default="tuning_grouped_gemm.log", help="Log file path"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument(
        "--ep-size", type=int, default=16, help="Expert parallelism size"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_file, getattr(logging, args.log_level))

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Set random seed to {args.seed}")

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        return

    logger.info(
        f"CUDA is available. Found {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )

    # 加载模型配置
    try:
        (
            model_config,
            hidden_size,
            dtype,
            top_k,
            num_experts_per_partition,
            block_shape,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
        ) = load_model_config(args.model, args)
    except Exception as e:
        logger.error(f"Failed to load model config: {e}")
        return

    logger.info(
        f"Tuning grouped GEMM Triton kernel with hidden_size={hidden_size}, dtype={dtype}, num_experts_per_partition={num_experts_per_partition}"
    )
    if block_shape:
        logger.info(f"Using block shape: {block_shape}")

    # Dictionary to store best configurations for each tokens_per_expert value
    best_configs = {}

    # Initialize Ray once for all tuning
    ray.init(local_mode=True, num_gpus=torch.cuda.device_count())
    num_gpus = int(ray.available_resources()["GPU"])
    logger.info(f"Using {num_gpus} GPUs for distributed tuning")

    # Create workers once
    workers = [BenchmarkWorker.remote(i) for i in range(num_gpus)]

    try:
        # Tune for each tokens_per_expert value
        for tokens_per_expert in TOKENS_PER_EXPERT_VALUES:
            logger.info(f"\nTuning for tokens_per_expert = {tokens_per_expert}")

            # 生成搜索空间
            search_space = get_configs_compute_bound()

            logger.info(
                f"Start tuning over {len(search_space)} configurations for tokens_per_expert={tokens_per_expert}..."
            )

            # 将搜索空间分配给不同的worker
            search_chunks = distribute_search_space(search_space, num_gpus)

            # 分配任务给workers
            futures = []
            for i, chunk in enumerate(search_chunks):
                worker = workers[i % num_gpus]
                future = worker.tune.remote(
                    num_experts_per_partition=num_experts_per_partition,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    tokens_per_expert=tokens_per_expert,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    block_shape=block_shape,
                    model_config=model_config,
                    search_space=chunk,
                )
                futures.append(future)

            # 收集结果
            start = time.time()
            results = ray.get(futures)
            end = time.time()

            # 找出最佳配置
            best_config = None
            best_time = float("inf")

            for config, kernel_time in results:
                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config

            logger.info(
                f"New best config for tokens_per_expert={tokens_per_expert}: {best_config} with kernel_time={best_time:.2f} us"
            )
            logger.info(
                f"Tuning for tokens_per_expert={tokens_per_expert} took {end - start:.2f} seconds"
            )

            # Store the best configuration for this tokens_per_expert value
            best_configs[str(tokens_per_expert)] = best_config

            # 使用最佳配置运行基准测试
            try:
                latency = benchmark_config(
                    config=best_config,
                    num_experts_per_partition=num_experts_per_partition,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    tokens_per_expert=tokens_per_expert,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    block_shape=block_shape,
                    model_config=model_config,
                )

                logger.info(
                    f"Verified latency for tokens_per_expert={tokens_per_expert}: {latency:.2f} us"
                )
            except Exception as e:
                logger.error(
                    f"Failed to benchmark best config for tokens_per_expert={tokens_per_expert}: {e}"
                )
                import traceback

                logger.error(traceback.format_exc())
    finally:
        # 关闭Ray
        ray.shutdown()

    # Save all best configurations to a single file
    try:
        # 确定文件名
        dtype_str = get_config_dtype_str(dtype)
        block_shape_str = (
            f"_block{block_shape[0]}x{block_shape[1]}" if block_shape else ""
        )
        filename = f"grouped_gemm_configs_{hidden_size}_{dtype_str}_b{num_experts_per_partition}{block_shape_str}.json"

        # 确保目录存在
        directory = os.path.dirname(filename)
        if directory:  # 只有当目录路径不为空时才创建目录
            os.makedirs(directory, exist_ok=True)

        # 保存配置
        with open(filename, "w") as f:
            json.dump(best_configs, f, indent=2)

        logger.info(f"Saved all best configurations to {filename}")
    except Exception as e:
        logger.error(f"Failed to save configurations: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return


if __name__ == "__main__":
    main()
