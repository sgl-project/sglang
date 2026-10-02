# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import copy
import json
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import ray
import torch
import triton
from common_utils import (
    BenchmarkConfig,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    save_configs,
    sort_config,
)
from ray.experimental.tqdm_ray import tqdm

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device, is_hip, is_xpu
from sglang.srt.utils.hf_transformers_utils import get_config

_is_hip = is_hip()
_is_xpu = is_xpu()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_heuristics_configs(use_valu: bool = False):
    if use_valu:
        block_m_range = [1, 2]
        block_n_range = [8, 16, 32, 64, 128, 256]
        block_k_range = [16, 32, 64, 128, 256]
        group_m_range = [1, 4]
    else:
        block_m_range = [16, 32, 64]
        block_n_range = [16, 32, 64, 128, 256, 512]
        block_k_range = [16, 32, 64, 128, 256, 512]
        group_m_range = [1, 16]
    num_warps_range = [2, 4, 8]
    num_stage_range = [2, 3, 4, 8]

    independent_param_ranges = {
        "BLOCK_SIZE_N": block_n_range,
        "BLOCK_SIZE_K": block_k_range,
        "GROUP_SIZE_M": group_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stage_range,
    }
    keys, values = zip(*independent_param_ranges.items())
    independent_configs = list()
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        independent_configs.append(config)
    configs = {m: independent_configs for m in block_m_range}
    return configs


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
    use_int4_w4a16: bool,
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
    elif use_int4_w4a16:
        w1 = torch.randint(
            0,
            255,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size // 2,
            ),
            dtype=torch.uint8,
        )
        w2 = torch.randint(
            0,
            255,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 4,
            ),
            dtype=torch.uint8,
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
    if use_int4_w4a16:
        block_n = 1 if (block_shape[0] == 0) else block_shape[0]
        block_k = block_shape[1]
        n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
        n_tiles_w2 = (hidden_size + block_n - 1) // block_n
        k_tiles_w1 = (hidden_size + block_k - 1) // block_k
        k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
        w1_scale = torch.randn(
            (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.bfloat16
        )
        w2_scale = torch.randn(
            (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.bfloat16
        )
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

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        model_config = get_config(args.model, trust_remote_code=True)
        architecture = model_config.architectures[0]
        is_dsv4 = architecture == "DeepseekV4ForCausalLM"
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
            swiglu_limit=10.0 if is_dsv4 else None,
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
                use_int4_w4a16=use_int4_w4a16,
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
    def __init__(self, seed: int, server_args: ServerArgs) -> None:
        torch.set_default_device(get_device())
        torch.get_device_module().manual_seed_all(0)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. Ray isolates each worker to a single visible
        # GPU via CUDA_VISIBLE_DEVICES, so the local ordinal is always 0. On
        # ROCm using the global ray gpu id here raises "invalid device ordinal".
        self.device_id = 0 if is_hip() else int(ray.get_gpu_ids()[0])
        set_global_server_args_for_scheduler(server_args)

    def benchmark(
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
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(0)
        dtype_str = get_config_dtype_str(
            dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int4_w4a16=use_int4_w4a16,
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        N = shard_intermediate_size // 2
        op_config = get_moe_configs(
            num_experts,
            N,
            dtype_str,
            block_n,
            block_k,
            per_channel_quant,
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                False,
                block_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
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
                use_int4_w4a16,
                per_channel_quant,
                block_shape,
            )
        return config, kernel_time

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
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")
        with (
            torch.get_device_module().device(self.device_id)
            if _is_xpu or _is_hip
            else nullcontext()
        ):
            for config in tqdm(search_space):
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
                        use_int4_w4a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=10,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError):
                    # Some configurations may be invalid and fail to compile.
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config

    def heuristics_tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        heuristics_space: Dict[int, List[Dict]],
        valu_thresh: int,
    ):
        discard = 1.3

        dummy_config = {
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        }
        log_file_name = (
            os.environ.get("HEURISTICS_TUNE_LOG", "heuristics_tune.log")
            + f".pid{os.getpid()}"
        )

        # tune for dispatch config
        with open(log_file_name, "w") as f:
            print(
                f"E={num_experts}, M={num_tokens}, N={shard_intermediate_size}, hidden_size={hidden_size}, topk={topk}, dtype={dtype}",
                file=f,
            )
            best_config = None
            perfect_time = float("inf")

            for use_valu in [True, False]:
                if use_valu and num_tokens >= valu_thresh:
                    continue
                heuristics_space = get_heuristics_configs(use_valu)
                for block_m, other_configs in heuristics_space.items():
                    # keep 'down' config
                    dummy_config["BLOCK_SIZE_M"] = block_m
                    best_time = float("inf")
                    best_up_config = None
                    for i, up_cfg in tqdm(
                        enumerate(other_configs),
                        desc=f"UP bs={num_tokens},bm={block_m},valu={use_valu}",  # codespell:ignore
                        total=len(other_configs),
                    ):
                        try:
                            up_cfg["BLOCK_SIZE_M"] = block_m
                            config = {
                                "BLOCK_SIZE_M": block_m,
                                "USE_VALU": use_valu,
                                "UP": up_cfg,
                                "DOWN": dummy_config,
                            }
                            seed_everything(self.seed)
                            torch.cuda.manual_seed(self.seed)
                            moe_time = benchmark_config(
                                config,
                                num_tokens,
                                num_experts,
                                shard_intermediate_size,
                                hidden_size,
                                topk,
                                dtype,
                                False,
                                False,
                                False,
                                None,
                                10,
                                best_time * discard,
                            )
                            if moe_time < best_time:
                                best_time = moe_time
                                best_up_config = copy.deepcopy(up_cfg)
                            print(
                                f"[{(i + 1) / len(other_configs) * 100:.2f}% UP][{i + 1}/{len(other_configs)}] moe_time={moe_time:.2f} best_time={best_time:.2f}, "
                                f"block_m={block_m}, up_config={up_cfg}",
                                file=f,
                            )
                        except triton.runtime.autotuner.OutOfResources:
                            print(
                                f"[{(i + 1) / len(other_configs) * 100:.2f}% UP][{i + 1}/{len(other_configs)}] BREAK DOWN!!! bad_config={up_cfg}",
                                file=f,
                            )
                            continue

                    # keep 'up' best config
                    best_time = float("inf")
                    best_dn_config = None
                    for i, dn_cfg in tqdm(
                        enumerate(other_configs),
                        desc=f"DN bs={num_tokens},bm={block_m},valu={use_valu}",  # codespell:ignore
                        total=len(other_configs),
                    ):
                        try:
                            dn_cfg["BLOCK_SIZE_M"] = block_m
                            config = {
                                "BLOCK_SIZE_M": block_m,
                                "USE_VALU": use_valu,
                                "UP": best_up_config,
                                "DOWN": dn_cfg,
                            }
                            seed_everything(self.seed)
                            torch.cuda.manual_seed(self.seed)
                            moe_time = benchmark_config(
                                config,
                                num_tokens,
                                num_experts,
                                shard_intermediate_size,
                                hidden_size,
                                topk,
                                dtype,
                                False,
                                False,
                                False,
                                None,
                                10,
                                best_time * discard,
                            )
                            if moe_time < best_time:
                                best_time = moe_time
                                best_dn_config = copy.deepcopy(dn_cfg)

                            print(
                                f"[{(i + 1) / len(other_configs) * 100:.2f}% DN][{i + 1}/{len(other_configs)}] moe_time={moe_time:.2f} best_time={best_time:.2f}, "
                                f"block_m={block_m}, dn_config={dn_cfg}",
                                file=f,
                            )
                        except triton.runtime.autotuner.OutOfResources:
                            print(
                                f"[{(i + 1) / len(other_configs) * 100:.2f}% DN][{i + 1}/{len(other_configs)}] BREAK DOWN!!! bad_config={dn_cfg}",
                                file=f,
                            )
                            continue

                    if perfect_time > best_time:
                        print(f"Alternate at M={block_m}, use_valu={use_valu}", file=f)
                        perfect_time = best_time
                        best_config = {
                            "BLOCK_SIZE_M": block_m,
                            "USE_VALU": use_valu,
                            "UP": copy.deepcopy(best_up_config),
                            "DOWN": copy.deepcopy(best_dn_config),
                        }
                    print(
                        f"TILE_M={block_m}, perfect_time={perfect_time}, canadiate config: {best_config}",
                        file=f,
                    )
            print(
                f"Tune Over: M={num_tokens}, perfect_time={perfect_time}, best_config={best_config}",
                file=f,
                flush=True,
            )
        # assert best_config is not None
        return best_config


def main(args: argparse.Namespace):
    server_args = ServerArgs(
        model_path=args.model, tp_size=args.tp_size, ep_size=args.ep_size
    )

    model_config = get_model_config(
        args.model, args.tp_size, args.ep_size, args.disable_shared_experts_fusion
    )

    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]

    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int4_w4a16 = args.dtype == "int4_w4a16"
    per_channel_quant = args.per_channel_quant

    if args.batch_sizes is not None:
        batch_sizes = args.batch_sizes
        batch_sizes.reverse()
    elif args.batch_size is not None:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = get_default_batch_sizes()
        batch_sizes.reverse()

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed, server_args) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        if args.search_space_file:
            with open(args.search_space_file) as f:
                search_space = json.load(f)
            if not isinstance(search_space, list) or not all(
                isinstance(config, dict) for config in search_space
            ):
                raise ValueError(
                    "--search-space-file must contain a JSON list of configs"
                )
        else:
            search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_k = block_shape[1]
            search_space = [
                config
                for config in search_space
                if block_k % config["BLOCK_SIZE_K"] == 0
            ]

        filename = get_config_filename(
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
        )
        print(
            f"Start tuning over {len(search_space)} configurations to create {filename}..."
        )

        start = time.time()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                    search_space,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            filename,
        )
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    elif args.heuristics_tune:
        assert not (use_fp8_w8a8 or use_int8_w8a16 or use_int8_w8a8)
        heuristics_space = get_heuristics_configs()
        start = time.time()
        configs = _distribute(
            "heuristics_tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    heuristics_space,
                    args.valu_thresh,
                )
                for batch_size in batch_sizes
            ],
        )

        best_configs = {M: config for M, config in zip(batch_sizes, configs)}
        save_configs(
            best_configs,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            block_shape,
        )
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16"],
        default="auto",
    )
    parser.add_argument(
        "--per-channel-quant",
        action="store_true",
    )
    parser.add_argument("--heuristics-tune", action="store_true")
    parser.add_argument("--valu-thresh", type=int, default=0)
    parser.add_argument(
        "--n-share-experts-fusion",
        type=int,
        default=0,
        help="The number of shared_experts need to be replica to fuse with normal experts in deepseek v3/r1",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Tune or benchmark an explicit set of token counts in parallel.",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--search-space-file",
        type=str,
        help="JSON file containing an explicit list of Triton configs to evaluate with --tune.",
    )
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    os.environ["USE_ACEXT_CUDA"] = "0"
    main(args)
