# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import dataclasses
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Tuple

import ray
import torch
import triton
import triton.language as tl
from common_utils import (
    BenchmarkConfig,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    sort_config,
)

try:
    from ray.experimental.tqdm_ray import tqdm
except Exception:  # ray's tqdm needs a ray context; fall back to plain tqdm.
    from tqdm import tqdm

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    get_config_dtype_str,
    invoke_fused_moe_kernel,
    moe_align_block_size,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_file_name,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device_count, is_hip
from sglang.srt.utils.hf_transformers_utils import get_config

_is_hip = is_hip()

# Layout of the saved topk_ids dataset, resolved per-model in BenchmarkWorker so it
# is correct in both the serial (XPU) path and each ray worker process (CUDA).
# Defaults preserve the original DeepSeek-V3 behavior (61 layers, 3 dense).
_NUM_LAYERS = 61
_DENSE_LAYERS = 3
# Snapshots saved per MoE layer (the capture snippet stores idx 0 and 1).
_NUM_SNAPSHOTS = 2


def _resolve_layer_layout(model_path: str) -> None:
    """Set the global topk_ids layer layout from the model's HF config."""
    global _NUM_LAYERS, _DENSE_LAYERS
    cfg = get_config(model_path, trust_remote_code=True)
    if hasattr(cfg, "text_config"):
        cfg = cfg.get_text_config()
    _NUM_LAYERS = getattr(cfg, "num_hidden_layers", _NUM_LAYERS)
    # DeepSeek-family models expose the count of leading dense layers; other MoE
    # models (e.g. Mixtral) have none.
    _DENSE_LAYERS = getattr(cfg, "first_k_dense_replace", 0)


def _get_accel_device() -> str:
    """Pick the active accelerator: cuda, else xpu, else raise."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    raise RuntimeError("No CUDA or XPU accelerator available for tuning.")


_ACCEL = _get_accel_device()
# torch.cuda / torch.xpu expose the same synchronize / Event / manual_seed_all API.
_accel_mod = torch.cuda if _ACCEL == "cuda" else torch.xpu
# Graph capture is available on CUDA (CUDAGraph) and on XPU builds that expose
# torch.xpu.XPUGraph / torch.xpu.graph; fall back to eager otherwise.
_supports_graph = _ACCEL == "cuda" or (
    _ACCEL == "xpu" and hasattr(torch.xpu, "XPUGraph")
)


def _accel_synchronize() -> None:
    _accel_mod.synchronize()


def _accel_event():
    return _accel_mod.Event(enable_timing=True)


def _accel_manual_seed_all(seed: int) -> None:
    _accel_mod.manual_seed_all(seed)


def _accel_new_graph():
    """Return a new capture graph object for the active accelerator."""
    return torch.cuda.CUDAGraph() if _ACCEL == "cuda" else torch.xpu.XPUGraph()


def _accel_graph_ctx(graph):
    """Return the graph-capture context manager for the active accelerator."""
    return torch.cuda.graph(graph) if _ACCEL == "cuda" else torch.xpu.graph(graph)


class _TemplateTopKOutput:
    """Minimal stand-in for the object returned by select_experts."""

    def __init__(self, topk_weights, topk_ids):
        self.topk_weights = topk_weights
        self.topk_ids = topk_ids
        self.router_logits = None


def _build_template_topk(hidden_states, input_gating, topk_config):
    """Build the throwaway topk template used only to size buffers / supply weights.

    The real routing comes from the saved dataset via prepare(); this template's
    topk_ids are overwritten. Fall back to a native softmax-topk when the fused
    kernel has no implementation for the given dtype/device (e.g. float32 on XPU).
    """
    try:
        return select_experts(hidden_states, input_gating, topk_config)
    except NotImplementedError:
        probs = torch.softmax(input_gating.float(), dim=-1)
        weights, ids = torch.topk(probs, topk_config.top_k, dim=-1)
        if getattr(topk_config, "renormalize", False):
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return _TemplateTopKOutput(weights.to(torch.float32), ids.to(torch.int32))


@dataclasses.dataclass
class MoeInputs:
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor


class KernelWrapper:
    def __init__(self, moe_inputs, use_cuda_graph=True, inner_iter=10, **kwargs):
        self.func = invoke_fused_moe_kernel
        # Graph capture is used on CUDA and on XPU builds that support it;
        # disabled otherwise (falls back to an eager replay loop).
        self.use_cuda_graph = use_cuda_graph and _supports_graph
        self.moe_inputs = moe_inputs
        self.inner_iter = inner_iter
        self.kwargs = kwargs
        if self.use_cuda_graph:
            self.graph = self.cuda_graph_wrapper()
        else:
            self.graph = None

    def cuda_graph_wrapper(self):
        moe_input = self.moe_inputs[0]
        self.func(
            **self.kwargs,
            topk_ids=moe_input.topk_ids,
            sorted_token_ids=moe_input.sorted_token_ids,
            expert_ids=moe_input.expert_ids,
            num_tokens_post_padded=moe_input.num_tokens_post_padded,
        )
        _accel_synchronize()

        # Capture inner_iter invocations into a single replayable graph.
        graph = _accel_new_graph()
        with _accel_graph_ctx(graph):
            for k in range(self.inner_iter):
                moe_input = self.moe_inputs[k]
                self.func(
                    **self.kwargs,
                    topk_ids=moe_input.topk_ids,
                    sorted_token_ids=moe_input.sorted_token_ids,
                    expert_ids=moe_input.expert_ids,
                    num_tokens_post_padded=moe_input.num_tokens_post_padded,
                )
        _accel_synchronize()

        # Warmup
        for _ in range(5):
            graph.replay()
        _accel_synchronize()
        return graph

    def forward_cost(self, try_cnt=2):
        time_cost = float("inf")
        for _ in range(try_cnt):
            start_event = _accel_event()
            end_event = _accel_event()
            start_event.record()
            if self.use_cuda_graph:
                self.graph.replay()
            else:
                for k in range(self.inner_iter):
                    moe_input = self.moe_inputs[k]
                    self.func(
                        **self.kwargs,
                        topk_ids=moe_input.topk_ids,
                        sorted_token_ids=moe_input.sorted_token_ids,
                        expert_ids=moe_input.expert_ids,
                        num_tokens_post_padded=moe_input.num_tokens_post_padded,
                    )
            end_event.record()
            _accel_synchronize()
            time_cost = min(time_cost, start_event.elapsed_time(end_event))
        return time_cost


def load_topk_ids(topk_ids_dir, i: int):
    moe_layers = _NUM_LAYERS - _DENSE_LAYERS
    # Cycle over the available (layer, snapshot) files so any requested sample
    # index maps onto a captured file, regardless of model depth.
    num_samples = moe_layers * _NUM_SNAPSHOTS
    j = i % num_samples
    layer = j % moe_layers + _DENSE_LAYERS
    idx = j // moe_layers
    return torch.load(f"{topk_ids_dir}/topk_ids_layer{layer}_idx{idx}.pt")


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
    topk_ids_list,
    block_shape: List[int] = None,
    ep_size: int = 1,
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

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output_ = _build_template_topk(hidden_states, input_gating, topk_config)
    sorted_token_ids_, expert_ids_, num_tokens_post_padded_ = moe_align_block_size(
        topk_output_.topk_ids, config["BLOCK_SIZE_M"], num_experts
    )
    inner_iter = 10 if not ncu_enable else 1
    moe_inputs = [
        MoeInputs(
            topk_output_.topk_ids.clone(),
            sorted_token_ids_.clone(),
            expert_ids_.clone(),
            num_tokens_post_padded_.clone(),
        )
        for _ in range(inner_iter)
    ]
    M = hidden_states.shape[0]
    E, N, _ = w1.shape

    padded_tokens = min(M * topk, E + 1) * (
        config["BLOCK_SIZE_M"] - 1
    )  # if moe_use_tma else 0
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

    def prepare(i: int, inner_iter):  # update inputs according to topk_ids
        for k in range(inner_iter):
            topk_ids = topk_ids_list[i * inner_iter + k]
            # With EP, saved topk_ids are global expert indices; remap to local.
            if ep_size > 1:
                topk_ids = (topk_ids // ep_size).to(
                    device=moe_inputs[k].topk_ids.device,
                    dtype=moe_inputs[k].topk_ids.dtype,
                )
            tokens, _topk = moe_inputs[k].topk_ids.shape
            moe_inputs[k].topk_ids.copy_(topk_ids[:tokens, :_topk])
            sorted_token_ids_, expert_ids_, num_tokens_post_padded_ = (
                moe_align_block_size(
                    moe_inputs[k].topk_ids, config["BLOCK_SIZE_M"], num_experts
                )
            )
            moe_inputs[k].sorted_token_ids.copy_(sorted_token_ids_)
            moe_inputs[k].expert_ids.copy_(expert_ids_)
            moe_inputs[k].num_tokens_post_padded.copy_(num_tokens_post_padded_)

    def get_kernel_wrapper(moe_use_tma, inner_iter, use_cuda_graph):
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
        )
        apply_router_weight_on_input = moe_runner_config.apply_router_weight_on_input
        kernel0 = KernelWrapper(
            A=hidden_states,
            B=w1,
            bias=None,
            C=intermediate_cache1,
            A_scale=a1_scale,
            B_scale=w1_scale,
            B_zp=None,
            topk_weights=topk_output_.topk_weights,
            moe_inputs=moe_inputs,
            mul_routed_weight=apply_router_weight_on_input,
            top_k=topk,
            config=config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=False,
            block_shape=block_shape,
            b_use_tma=moe_use_tma,
            c_sorted=moe_use_tma,
            filter_expert=False,
            use_cuda_graph=use_cuda_graph,
            inner_iter=inner_iter,
        )
        kernel1 = KernelWrapper(
            A=intermediate_cache2,
            B=w2,
            bias=None,
            C=intermediate_cache3,
            A_scale=a2_scale,
            B_scale=w2_scale,
            B_zp=None,
            topk_weights=topk_output_.topk_weights,
            moe_inputs=moe_inputs,
            mul_routed_weight=not apply_router_weight_on_input,
            top_k=1,
            config=config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=False,
            block_shape=block_shape,
            a_use_tma=moe_use_tma,
            b_use_tma=moe_use_tma,
            filter_expert=False,
            use_cuda_graph=use_cuda_graph,
            inner_iter=inner_iter,
        )
        return kernel0, kernel1

    use_cuda_graph = True if not ncu_enable else False
    kernel0, kernel1 = get_kernel_wrapper(False, inner_iter, use_cuda_graph)
    kernel_tma0, kernel_tma1 = get_kernel_wrapper(True, inner_iter, use_cuda_graph)

    # JIT compilation & warmup
    if not ncu_enable:
        kernel0.forward_cost()
        kernel1.forward_cost()
        kernel_tma0.forward_cost()
        kernel_tma1.forward_cost()

    ts0 = []
    ts1 = []
    ts_tma0 = []
    ts_tma1 = []

    for i in range(num_iters // inner_iter):
        prepare(i, inner_iter)
        ts0.append(kernel0.forward_cost())
        ts1.append(kernel1.forward_cost())
        ts_tma0.append(kernel_tma0.forward_cost())
        ts_tma1.append(kernel_tma1.forward_cost())
    _accel_synchronize()
    avg = sum(ts0) / (num_iters) * 1000  # us
    avg1 = sum(ts1) / (num_iters) * 1000  # us
    avg_tma = sum(ts_tma0) / (num_iters) * 1000  # us
    avg1_tma = sum(ts_tma1) / (num_iters) * 1000  # us

    return avg, avg_tma, avg1, avg1_tma


class BestConfigTrace:
    def __init__(self, name, down_moe=False):
        self.name = name
        self.down_moe = down_moe
        self.best_costs_m = {}  # block_m: best_cost

    def update(self, config, time_cost_all):
        block_m = config["BLOCK_SIZE_M"]
        if not self.down_moe:
            time_cost = time_cost_all[0]
        else:
            time_cost = min(time_cost_all[2], time_cost_all[3])
        if (
            block_m not in self.best_costs_m
            or time_cost < self.best_costs_m[block_m][1]
        ):
            self.best_costs_m[block_m] = config, time_cost, time_cost_all

    def time_cost(self, block_m):
        if block_m not in self.best_costs_m:
            return float("inf")
        time_cost = self.best_costs_m[block_m][1]
        return time_cost

    def config_dict(self, block_m):
        if block_m not in self.best_costs_m:
            return {}
        config, _, time_cost_all = self.best_costs_m[block_m]
        if not self.down_moe:
            return config
        else:
            return {
                **config,
                "USE_TMA": time_cost_all[2] > time_cost_all[3],
            }


class BenchmarkWorker:

    def __init__(self, seed: int, server_args: ServerArgs) -> None:
        torch.set_default_device(_ACCEL)
        _accel_manual_seed_all(0)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU.
        self.device_id = 0  # int(ray.get_gpu_ids()[0])
        set_global_server_args_for_scheduler(server_args)
        # Resolve the saved-dataset layer layout for this model. Done here so it is
        # also set inside each ray worker process (CUDA), not just the serial path.
        _resolve_layer_layout(server_args.model_path)

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
        block_shape: List[int],
        cfg: Dict[str, int],
        topk_ids_dir: str,
        ep_size: int = 1,
    ) -> Tuple[Dict[str, int], float]:
        _accel_manual_seed_all(0)
        topk_ids_list = [load_topk_ids(topk_ids_dir, i) for i in range(100)]
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
                use_int4_w4a16,
                topk_ids_list,
                block_shape,
                ep_size=ep_size,
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
        use_int4_w4a16: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
        topk_ids_dir: str,
        ep_size: int = 1,
    ) -> Dict[str, int]:
        trace0 = BestConfigTrace("kernel0", down_moe=False)
        trace1 = BestConfigTrace("kernel1", down_moe=True)
        topk_ids_list = [load_topk_ids(topk_ids_dir, i) for i in range(100)]

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
                        use_int4_w4a16,
                        topk_ids_list,
                        block_shape,
                        ep_size=ep_size,
                        num_iters=100,
                    )
                except triton.runtime.autotuner.OutOfResources:
                    # Some configurations may be invalid and fail to compile.
                    continue
                trace0.update(
                    config,
                    (kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma),
                )
                trace1.update(
                    config,
                    (kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma),
                )

        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        best_block_m = 16
        for block_m in (32, 64, 128, 256):
            if trace0.time_cost(block_m) + trace1.time_cost(block_m) < trace0.time_cost(
                best_block_m
            ) + trace1.time_cost(best_block_m):
                best_block_m = block_m

        return (
            trace0.config_dict(best_block_m),
            trace1.config_dict(best_block_m),
            trace0.time_cost(best_block_m),
            trace1.time_cost(best_block_m),
        )

    def cmp_configs(
        self,
        num_tokens: List[int],
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        block_shape: List[int],
        cmp_config_files: List[str],
        topk_ids_dir: str,
        ep_size: int = 1,
    ):
        # compare performance of different configs
        cmp_configs = []
        for file in cmp_config_files:
            with open(file) as f:
                cmp_configs.append({int(key): val for key, val in json.load(f).items()})
        for i, file in enumerate(cmp_config_files):
            print(f"config {i}: {file}")

        topk_ids_list = [load_topk_ids(topk_ids_dir, i) for i in range(100)]
        _accel_manual_seed_all(0)
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            for bs in num_tokens:
                kernel_times = []
                cfgs = []
                for configs in cmp_configs:
                    cfg_org = configs[min(configs.keys(), key=lambda x: abs(x - bs))]
                    cfgs.append(cfg_org)
                    cfg = cfg_org.copy()
                    cfg.pop("USE_TMA", None)
                    kernel_time = benchmark_config(
                        cfg,
                        bs,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        topk_ids_list,
                        block_shape,
                        ep_size=ep_size,
                    )
                    kernel_times.append(kernel_time)
                print(f"batch_size={bs=}:")
                for i, cfg in enumerate(cfgs):
                    print(f"  config {i} {cfg}: {kernel_times[i]}")


def save_configs_sep(
    configs: Dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: List[int],
    down_moe: bool = False,
) -> None:
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int4_w4a16=use_int4_w4a16,
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

    server_args = ServerArgs(
        model_path=args.model, tp_size=args.tp_size, ep_size=args.ep_size
    )

    model_config = get_model_config(
        args.model,
        args.tp_size,
        args.ep_size,
        args.disable_shared_experts_fusion,
        args.topk_ids_dir,
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

    topk_ids_dir = args.topk_ids_dir
    if args.batch_size is None:
        batch_sizes = get_default_batch_sizes()
        batch_sizes.reverse()
    else:
        batch_sizes = [args.batch_size]

    if args.cmp_configs is not None:
        worker = BenchmarkWorker(args.seed, server_args)
        worker.cmp_configs(
            batch_sizes,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
            args.cmp_configs,
            topk_ids_dir,
            args.ep_size,
        )
        return

    if len(batch_sizes) == 1:
        worker = BenchmarkWorker(args.seed, server_args)
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
                use_int4_w4a16,
                block_shape,
                search_space,
                topk_ids_dir,
                args.ep_size,
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
                use_int4_w4a16,
                block_shape,
                cfg,
                topk_ids_dir,
                args.ep_size,
            )
            print(f"{t0=}, {t0_tma=}, {t1=}, {t1_tma=}")
        return

    assert args.tune

    if _ACCEL == "cuda":
        ray.init()
        num_gpus = int(ray.available_resources()["GPU"])
        use_ray = True
    else:
        # XPU: Ray's IntelGPUAcceleratorManager (requires dpctl) advertises XPUs
        # as the "GPU" resource and isolates each worker via ONEAPI_DEVICE_SELECTOR,
        # so the assigned device appears as index 0 inside the worker (same as
        # CUDA). When Ray cannot see multiple devices, fall back to a single
        # in-process worker.
        ray.init(num_gpus=get_device_count())
        num_gpus = int(ray.available_resources().get("GPU", 0))
        use_ray = num_gpus > 1

    if use_ray:
        workers = [
            ray.remote(num_gpus=1)(BenchmarkWorker).remote(args.seed, server_args)
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

    else:
        serial_worker = BenchmarkWorker(args.seed, server_args)

        def _distribute(method: str, inputs: List[Any]) -> List[Any]:
            worker_method = getattr(serial_worker, method)
            return [worker_method(*input_args) for input_args in inputs]

    search_space = get_configs_compute_bound()
    if block_shape is not None:
        block_n, block_k = block_shape[0], block_shape[1]
        search_space = [
            config for config in search_space if block_k % config["BLOCK_SIZE_K"] == 0
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
        False,
        block_shape,
    )
    print(
        f"Start tuning over {len(search_space)} configurations to create {filename}..."
    )

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
                use_int4_w4a16,
                block_shape,
                search_space,
                topk_ids_dir,
                args.ep_size,
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
    save_configs_sep(
        best_configs0,
        E,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        block_shape,
    )

    best_configs1 = {M: sort_config(config) for M, config in zip(batch_sizes, configs1)}
    save_configs_sep(
        best_configs1,
        E,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
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
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int8_w4a16"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    parser.add_argument("--configs", type=int, nargs="+", required=False)
    parser.add_argument("--topk-ids-dir", type=str, required=True)
    parser.add_argument("--cmp-configs", type=str, nargs="+", required=False)
    args = parser.parse_args()

    main(args)
