import time
from typing import Any, Dict, List, Optional

import torch
import triton

from sglang.auto_tune.utils import (
    get_configs_compute_bound,
    get_config_filename,
    get_default_batch_sizes,
    save_configs,
    sort_config,
)
from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import get_device, is_hip


def benchmark_config(
    config: Dict[str, int],
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: List[int] = None,
    num_iters: int = 100,
) -> float:
    torch.set_default_device(get_device())
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(-127, 127, (num_experts, shard_intermediate_size, hidden_size), dtype=torch.int8)
        w2 = torch.randint(-127, 127, (num_experts, hidden_size, shard_intermediate_size // 2), dtype=torch.int8)
    elif use_int4_w4a16:
        w1 = torch.randint(0, 255, (num_experts, shard_intermediate_size, hidden_size // 2), dtype=torch.uint8)
        w2 = torch.randint(0, 255, (num_experts, hidden_size, shard_intermediate_size // 4), dtype=torch.uint8)
    else:
        w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype)
        w2 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype)
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = w2_scale = a1_scale = a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size), dtype=torch.float32)
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_int4_w4a16 and block_shape:
        block_n = 1 if block_shape[0] == 0 else block_shape[0]
        block_k = block_shape[1]
        n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
        n_tiles_w2 = (hidden_size + block_n - 1) // block_n
        k_tiles_w1 = (hidden_size + block_k - 1) // block_k
        k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
        w1_scale = torch.randn((num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.bfloat16)
        w2_scale = torch.randn((num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.bfloat16)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32)
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
            w1_scale = torch.rand((num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
            w2_scale = torch.rand((num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32)

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if is_hip() else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if is_hip() else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(top_k=topk, renormalize=True)
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        new_topk_output = select_experts(x, gating_output[i], topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        with override_config(config):
            fused_moe(
                x, w1, w2, topk_output,
                moe_runner_config=MoeRunnerConfig(inplace=True),
                use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16, use_int4_w4a16=use_int4_w4a16,
                w1_scale=w1_scale, w2_scale=w2_scale,
                a1_scale=a1_scale, a2_scale=a2_scale,
                per_channel_quant=per_channel_quant, block_shape=block_shape,
            )

    run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

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

    latencies = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
    avg = sum(latencies) / (num_iters * 10) * 1000
    graph.reset()
    return avg


def tune_moe_kernel(
    num_experts: int, shard_intermediate_size: int, hidden_size: int, topk: int,
    dtype: torch.dtype, use_fp8_w8a8=False, use_int8_w8a8=False,
    use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
    block_shape=None, batch_sizes=None, search_space=None, num_iters=10, verbose=True,
) -> Dict[int, Dict]:
    if batch_sizes is None:
        batch_sizes = get_default_batch_sizes()
    if search_space is None:
        search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_k = block_shape[1]
            search_space = [c for c in search_space if block_k % c["BLOCK_SIZE_K"] == 0]

    best_configs = {}
    total_start = time.perf_counter()

    for batch_idx, num_tokens in enumerate(batch_sizes):
        if verbose:
            print(f"[{batch_idx + 1}/{len(batch_sizes)}] Tuning batch_size={num_tokens} ({len(search_space)} configs)...")

        best_config = None
        best_time = float("inf")

        for config in search_space:
            try:
                kernel_time = benchmark_config(
                    config, num_tokens, num_experts, shard_intermediate_size,
                    hidden_size, topk, dtype, use_fp8_w8a8, use_int8_w8a8,
                    use_int8_w8a16, use_int4_w4a16, per_channel_quant,
                    block_shape, num_iters=num_iters,
                )
            except (triton.runtime.autotuner.OutOfResources, RuntimeError, Exception):
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config

        if best_config is not None:
            best_configs[num_tokens] = sort_config(best_config)
            if verbose:
                print(f"  Best: {sort_config(best_config)} ({best_time:.2f} us)")
        elif verbose:
            print(f"  WARNING: No valid config for batch_size={num_tokens}")

    if verbose:
        print(f"\nTuning completed in {time.perf_counter() - total_start:.2f}s")
        print(f"Batch sizes tuned: {len(best_configs)}/{len(batch_sizes)}")
    return best_configs


def run_moe_tuning(model_config, tp_size, ep_size=1, batch_sizes=None, output_dir=None, verbose=True):
    if not model_config.get("is_moe", False):
        if verbose:
            print("  Model is not MoE, skipping MoE tuning.")
        return {}

    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]

    if E == 0:
        if verbose:
            print("  No experts found, skipping MoE tuning.")
        return {}

    if verbose:
        print(f"  Model: {model_config['architecture']}")
        print(f"  Experts: {E}, TopK: {topk}")
        print(f"  Hidden: {hidden_size}, Shard Intermediate: {shard_intermediate_size}")
        print(f"  Dtype: {dtype}, Block Shape: {block_shape}")

    best_configs = tune_moe_kernel(
        E, shard_intermediate_size, hidden_size, topk, dtype,
        False, False, False, False, False, block_shape,
        batch_sizes, verbose=verbose,
    )

    if not best_configs:
        if verbose:
            print("No valid configs found. Skipping config save.")
        return {}

    filename = get_config_filename(
        E, shard_intermediate_size, hidden_size, topk, dtype,
        False, False, False, False, False, block_shape,
    )
    saved_path = save_configs(best_configs, filename, output_dir)
    if verbose:
        print(f"Saved config to: {saved_path}")
    return best_configs