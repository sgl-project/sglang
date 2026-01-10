# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
# Adapted again from sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py

import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch
import triton
from ray.experimental.tqdm_ray import tqdm

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_hip
from sglang.tune.fused_moe_utils import (
    BenchmarkConfig,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    sort_config,
)


@dataclass
class TuningCheckpoint:
    """Checkpoint for resumable auto-tuning."""

    model: str
    tp_size: int
    ep_size: int
    dtype: str
    per_channel_quant: bool
    completed_batch_sizes: Dict[int, BenchmarkConfig] = field(default_factory=dict)
    pending_batch_sizes: List[int] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        # Convert string keys back to int when loading from JSON
        if self.completed_batch_sizes:
            self.completed_batch_sizes = {
                int(k): v for k, v in self.completed_batch_sizes.items()
            }


def get_checkpoint_path(config_path: str) -> str:
    """Return checkpoint path for a config file."""
    return config_path.replace(".json", ".checkpoint.json")


def save_checkpoint(checkpoint_path: str, checkpoint: TuningCheckpoint) -> None:
    """Atomically save checkpoint (write temp -> rename)."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(asdict(checkpoint), f, indent=2)
    os.rename(temp_path, checkpoint_path)
    print(f"Checkpoint saved: {len(checkpoint.completed_batch_sizes)} batch sizes completed")


def load_checkpoint(checkpoint_path: str) -> Optional[TuningCheckpoint]:
    """Load checkpoint if exists, return None otherwise."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        checkpoint = TuningCheckpoint(**data)
        print(f"Loaded checkpoint: {len(checkpoint.completed_batch_sizes)} batch sizes already completed")
        return checkpoint
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Warning: Could not load checkpoint ({e}), starting fresh")
        return None


def cleanup_checkpoint(checkpoint_path: str) -> None:
    """Remove checkpoint file after successful completion."""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Cleaned up checkpoint file: {checkpoint_path}")

_is_hip = is_hip()


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    per_channel_quant: bool,
    block_shape: List[int] = None,
    num_iters: int = 100,
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        input_gating = gating_output[i]
        new_topk_output = select_experts(x, input_gating, topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
        )

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                moe_runner_config=moe_runner_config,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    # Flush L2 cache with 256 MB data
    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache_flush.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        prepare(i)
        start_events[i].record()
        graph.replay()
        end_events[i].record()
    torch.cuda.synchronize()

    latencies: List[float] = []
    for i in range(num_iters):
        latencies.append(start_events[i].elapsed_time(end_events[i]))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU.
        self.device_id = int(ray.get_gpu_ids()[0])

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
        num_iters: int,
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")
        update_interval = 10
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            pbar = tqdm(total=len(search_space))
            for i, config in enumerate(search_space):
                try:
                    kernel_time = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=num_iters,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError):
                    # Some configurations may be invalid and fail to compile.
                    if (i + 1) % update_interval == 0:
                        pbar.update(update_interval)
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config

                if (i + 1) % update_interval == 0:
                    pbar.update(update_interval)
            # Update any remaining progress
            remaining = len(search_space) % update_interval
            if remaining > 0:
                pbar.update(remaining)
            pbar.close()
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def tune_fused_moe_triton(
    *,
    model: str,
    tp_size: int,
    ep_size: int,
    dtype: str,
    per_channel_quant: bool,
    batch_sizes: Optional[List[int]],
    seed: int,
    disable_shared_experts_fusion: bool,
    num_iters: int,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> Dict[int, BenchmarkConfig]:
    """Run fused MoE Triton tuning programmatically.

    Args:
        checkpoint_path: Path to save/load checkpoint file. If None, checkpointing is disabled.
        resume: If True and checkpoint exists, resume from checkpoint.

    Returns:
        A mapping of batch size to the tuned kernel config.
    """

    model_config = get_model_config(
        model, tp_size, ep_size, disable_shared_experts_fusion
    )

    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype_torch = model_config["dtype"]
    block_shape = model_config["block_shape"]

    use_fp8_w8a8 = dtype == "fp8_w8a8"
    use_int8_w8a8 = dtype == "int8_w8a8"
    use_int8_w8a16 = dtype == "int8_w8a16"

    batch_sizes = batch_sizes if batch_sizes else get_default_batch_sizes()

    # Load checkpoint if resuming
    checkpoint: Optional[TuningCheckpoint] = None
    if resume and checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            # Validate checkpoint matches current tuning parameters
            if (
                checkpoint.model != model
                or checkpoint.tp_size != tp_size
                or checkpoint.ep_size != ep_size
                or checkpoint.dtype != dtype
                or checkpoint.per_channel_quant != per_channel_quant
            ):
                print("Warning: Checkpoint parameters don't match, starting fresh")
                checkpoint = None

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(seed) for _ in range(num_gpus)]
    print(f"Using {num_gpus} GPUs for tuning")

    search_space = get_configs_compute_bound()
    if block_shape is not None:
        block_n, block_k = block_shape[0], block_shape[1]
        search_space = [
            config for config in search_space if block_k % config["BLOCK_SIZE_K"] == 0
        ]

    # Determine which batch sizes to tune
    if checkpoint:
        remaining_batch_sizes = [
            bs for bs in batch_sizes if bs not in checkpoint.completed_batch_sizes
        ]
        completed_configs = dict(checkpoint.completed_batch_sizes)
        print(f"Resuming: {len(completed_configs)} completed, {len(remaining_batch_sizes)} remaining")
    else:
        remaining_batch_sizes = batch_sizes
        completed_configs = {}

    if not remaining_batch_sizes:
        print("All batch sizes already completed!")
        return {M: sort_config(config) for M, config in completed_configs.items()}

    print(f"Start tuning over {len(search_space)} configurations for {len(remaining_batch_sizes)} batch sizes...")

    start = time.perf_counter()

    # Process batch sizes one at a time to enable checkpointing after each
    for batch_size in remaining_batch_sizes:
        worker_idx = remaining_batch_sizes.index(batch_size) % num_gpus
        worker = workers[worker_idx]

        print(f"Tuning batch_size={batch_size}...")
        config = ray.get(
            worker.tune.remote(
                batch_size,
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_torch,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                per_channel_quant,
                block_shape,
                search_space,
                num_iters,
            )
        )
        completed_configs[batch_size] = sort_config(config)

        # Save checkpoint after each batch size completes
        if checkpoint_path:
            remaining = [bs for bs in batch_sizes if bs not in completed_configs]
            checkpoint = TuningCheckpoint(
                model=model,
                tp_size=tp_size,
                ep_size=ep_size,
                dtype=dtype,
                per_channel_quant=per_channel_quant,
                completed_batch_sizes=completed_configs,
                pending_batch_sizes=remaining,
                timestamp=datetime.now().isoformat(),
            )
            save_checkpoint(checkpoint_path, checkpoint)

    end = time.perf_counter()
    print(f"Tuning took {end - start:.2f} seconds")

    # Cleanup checkpoint on successful completion
    if checkpoint_path:
        cleanup_checkpoint(checkpoint_path)

    return completed_configs
