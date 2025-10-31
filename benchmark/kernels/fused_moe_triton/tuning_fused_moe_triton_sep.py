# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict

import ray
import torch
import triton
import triton.language as tl
from ray.experimental.tqdm_ray import tqdm
from sgl_kernel import silu_and_mul
from transformers import AutoConfig

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    get_config_dtype_str,
    invoke_fused_moe_kernel,
    moe_align_block_size,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_file_name,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_hip

_is_hip = is_hip()


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


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
    topk_ids_dir: str,
    block_shape: List[int] = None,
    num_iters: int = 100,
) -> float:
    ncu_enable = os.getenv("NCU_ENABLE", "0") == "1"
    if ncu_enable:
        num_iters = 1
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)
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
    topk_output = select_experts(hidden_states, input_gating, topk_config)

    def prepare(i: int):
        input_gating = gating_output[i]
        topk_ids = torch.load(f"{topk_ids_dir}/topk_ids_layer{i%58+3}_idx{i//58}.pt")
        new_topk_output = select_experts(hidden_states, input_gating, topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        tokens, _topk = topk_output.topk_ids.shape
        topk_output.topk_ids.copy_(topk_ids[:tokens, :_topk])
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    moe_use_tma = False

    def run():
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
        )
        topk_weights, topk_ids, _ = topk_output

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], num_experts
        )
        M = hidden_states.shape[0]
        E, N, _ = w1.shape

        topk = topk_ids.shape[1]
        padded_tokens = (
            min(M * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1) if moe_use_tma else 0
        )
        total_tokens = M * topk + padded_tokens
        cache = torch.empty(
            total_tokens * max(N, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache1 = cache[: total_tokens * N].view(
            (total_tokens, N),
        )
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache3 = cache[: M * topk * w2.shape[1]].view(
            (M, topk, w2.shape[1]),
        )

        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )
        apply_router_weight_on_input = moe_runner_config.apply_router_weight_on_input

        with override_config(config):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(10 if not ncu_enable else 1):
                invoke_fused_moe_kernel(
                    hidden_states,
                    w1,
                    None,
                    intermediate_cache1,
                    None,
                    w1_scale,
                    None,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    apply_router_weight_on_input,
                    topk_ids.shape[1],
                    config,
                    compute_type=compute_type,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=block_shape,
                    b_use_tma=moe_use_tma,
                    c_sorted=moe_use_tma,
                    filter_expert=False,
                )
            end_event.record()
            end_event.synchronize()
            time_cost0 = start_event.elapsed_time(end_event)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

            silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            for _ in range(10 if not ncu_enable else 1):
                invoke_fused_moe_kernel(
                    intermediate_cache2,
                    w2,
                    None,
                    intermediate_cache3,
                    a2_scale,
                    w2_scale,
                    None,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    not apply_router_weight_on_input,
                    1,
                    config,
                    compute_type=compute_type,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=block_shape,
                    a_use_tma=moe_use_tma,
                    b_use_tma=moe_use_tma,
                    filter_expert=False,
                )
            end_event.record()
            end_event.synchronize()
            time_cost1 = start_event.elapsed_time(end_event)
        return time_cost0, time_cost1

    # JIT compilation & warmup
    if not ncu_enable:
        moe_use_tma = False
        run()
        moe_use_tma = True
        run()
    latencies: List[float] = []
    latencies1: List[float] = []
    latencies_tma: List[float] = []
    latencies1_tma: List[float] = []

    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()
        moe_use_tma = False
        t0, t1 = run()
        torch.cuda.synchronize()
        latencies.append(t0)
        latencies1.append(t1)

        moe_use_tma = True
        t0, t1 = run()
        torch.cuda.synchronize()
        latencies_tma.append(t0)
        latencies1_tma.append(t1)

    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    avg_tma = sum(latencies_tma) / (num_iters * 10) * 1000  # us
    avg1 = sum(latencies1) / (num_iters * 10) * 1000  # us
    avg1_tma = sum(latencies1_tma) / (num_iters * 10) * 1000  # us

    return avg, avg_tma, avg1, avg1_tma


def get_rocm_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []
    waves_per_eu_range = 0
    for block_m in [32, 64, 128, 256]:
        for block_k in [32, 64, 128, 256]:
            for block_n in [16, 32, 64, 128, 256]:
                for num_stages in [2]:
                    for num_warps in [1, 2, 4, 8]:
                        for group_size in [1, 4, 8, 16, 32]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu_range,
                                }
                            )
    return configs


def get_configs_compute_bound() -> List[Dict[str, int]]:
    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    configs: List[BenchmarkConfig] = []
    if _is_hip:
        configs = get_rocm_configs_compute_bound()
    else:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for block_n in [32, 64, 128, 256]:
                    for num_stages in [2, 3, 4, 5]:
                        for num_warps in [4, 8]:
                            for group_size in [1, 16, 32, 64]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                    }
                                )
    return configs


class BestConfigTrace:
    def __init__(self, name):
        self.name = name
        self.config = None
        self.time_cost = float("inf")
        self.time_cost_all = None  # kernel0 without tma,, kernel0 with tma, kernel1 without tma, kernel1 with tma

    def update(self, config, time_cost, time_cost_all):
        if time_cost < self.time_cost:
            print(
                f"New best config for {self.name}: {config}, {time_cost=}, {time_cost_all=}, org: {self.config}, {self.time_cost_all}",
                flush=True,
            )
            self.config = config
            self.time_cost = time_cost
            self.time_cost_all = time_cost_all

    @property
    def total_time(self):
        return self.time_cost_all[0] + min(self.time_cost_all[2], self.time_cost_all[3])

    def config_dict(self, down_moe=False):
        if not down_moe:
            return self.config
        else:
            return {
                **self.config,
                "USE_TMA": self.time_cost_all[2] > self.time_cost_all[3],
            }


class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU.
        self.device_id = 0  # int(ray.get_gpu_ids()[0])

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
        block_shape: List[int],
        cfg: Dict[str, int],
        topk_ids_dir: str,
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(0)
        dtype_str = get_config_dtype_str(
            dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            kernel_time = benchmark_config(
                cfg,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                topk_ids_dir,
                block_shape,
            )
        return cfg, kernel_time

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
        block_shape: List[int],
        search_space: List[Dict[str, int]],
        topk_ids_dir: str,
    ) -> Dict[str, int]:
        trace0 = BestConfigTrace("kernel0")
        trace1 = BestConfigTrace("kernel1")
        trace2 = BestConfigTrace("kernel all")

        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            for config in tqdm(search_space):
                try:
                    kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma = benchmark_config(
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
                        topk_ids_dir,
                        block_shape,
                        num_iters=10,
                    )
                except triton.runtime.autotuner.OutOfResources:
                    # Some configurations may be invalid and fail to compile.
                    continue
                kt0 = kt0_no_tma
                kt1 = min(kt1_no_tma, kt1_tma)
                trace0.update(
                    config,
                    kt0,
                    (kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma),
                )
                trace1.update(
                    config,
                    kt1,
                    (kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma),
                )
                trace2.update(
                    config,
                    kt0 + kt1,
                    (kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma),
                )

        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert trace0.config is not None
        assert trace1.config is not None
        print(
            f"{num_tokens=}, {trace0.config=}, {trace0.time_cost_all=}, {trace1.config=}, {trace1.time_cost_all=}"
        )
        if trace0.config["BLOCK_SIZE_M"] != trace1.config["BLOCK_SIZE_M"]:
            best_trace = trace0 if trace0.total_time < trace1.total_time else trace1
            best_trace = (
                best_trace if best_trace.total_time < trace2.total_time else trace2
            )
            return (
                best_trace.config_dict(),
                best_trace.config_dict(True),
                best_trace.time_cost_all,
                best_trace.time_cost_all,
            )
        return (
            trace0.config_dict(),
            trace1.config_dict(True),
            trace0.time_cost_all,
            trace1.time_cost_all,
        )


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
        ),
        **({"USE_TMA": config["USE_TMA"]} if "USE_TMA" in config else {}),
    }


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
    down_moe: bool = False,
) -> None:
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(
        num_experts,
        shard_intermediate_size // 2,
        dtype_str,
        block_shape,
        down_moe=down_moe,
    )

    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"]:
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        E = (
            config.n_routed_experts + (0 if args.disable_shared_experts_fusion else 1)
            if config.architectures[0] in ["DeepseekV3ForCausalLM"]
            else config.n_routed_experts
        )
        topk = (
            config.num_experts_per_tok
            + (0 if args.disable_shared_experts_fusion else 1)
            if config.architectures[0] in ["DeepseekV3ForCausalLM"]
            else config.num_experts_per_tok
        )
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "Llama4ForConditionalGeneration":
        E = config.text_config.num_local_experts + (
            0 if args.disable_shared_experts_fusion else 1
        )
        topk = config.text_config.num_experts_per_tok
        intermediate_size = config.text_config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in [
        "Grok1ForCausalLM",
        "Grok1ImgGen",
        "Grok1AForCausalLM",
    ]:
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["Glm4MoeForCausalLM"]:
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size

    hidden_size = getattr(config, "hidden_size", None) or config.text_config.hidden_size
    dtype = config.torch_dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2

    topk_ids_dir = args.topk_ids_dir
    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            8192,
        ]
        batch_sizes.reverse()
    else:
        batch_sizes = [args.batch_size]
    if len(batch_sizes) == 1:
        worker = BenchmarkWorker(args.seed)
        if args.tune:
            search_space = get_configs_compute_bound()
            worker.tune(
                batch_sizes[0],
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                block_shape,
                search_space,
                topk_ids_dir,
            )
        else:
            cfg = {
                "BLOCK_SIZE_M": args.configs[0],
                "BLOCK_SIZE_N": args.configs[1],
                "BLOCK_SIZE_K": args.configs[2],
                "GROUP_SIZE_M": args.configs[3],
                "num_warps": args.configs[4],
                "num_stages": args.configs[5],
            }

            _, (t0, t0_tma, t1, t1_tma) = worker.benchmark(
                args.batch_size,
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                block_shape,
                cfg,
                topk_ids_dir,
            )
            print(f"{t0=}, {t0_tma=}, {t1=}, {t1_tma=}")
        return

    assert args.tune

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [
        ray.remote(num_gpus=1)(BenchmarkWorker).remote(args.seed)
        for _ in range(num_gpus)
    ]

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

    search_space = get_configs_compute_bound()
    if block_shape is not None:
        block_n, block_k = block_shape[0], block_shape[1]
        search_space = [
            config for config in search_space if block_k % config["BLOCK_SIZE_K"] == 0
        ]
    print(f"Start tuning over {len(search_space)} configurations...")

    start = time.perf_counter()
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
                block_shape,
                search_space,
                topk_ids_dir,
            )
            for batch_size in batch_sizes
        ],
    )
    print(f"{configs=}", flush=True)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(f"tuning_result_{cur_time}.txt", "w") as f:
        print(configs, file=f)
    batch_sizes.reverse()
    configs0 = [config[0] for config in configs]
    configs1 = [config[1] for config in configs]
    configs0.reverse()
    configs1.reverse()
    best_configs0 = {M: sort_config(config) for M, config in zip(batch_sizes, configs0)}
    save_configs(
        best_configs0,
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

    best_configs1 = {M: sort_config(config) for M, config in zip(batch_sizes, configs1)}
    save_configs(
        best_configs1,
        E,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        block_shape,
        down_moe=True,
    )
    end = time.perf_counter()
    print(f"Tuning took {end - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    parser.add_argument("--configs", type=int, nargs="+", required=False)
    parser.add_argument("--topk-ids-dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
