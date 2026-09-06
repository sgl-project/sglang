"""Benchmark GLM-5.2's W4A-FP8 CUTLASS MoE dispatch on H200/SM90.

Times ``cutlass_w4a8_moe`` (the full MoE-layer call: preprocess+reorder+quant,
GEMM1, silu+requant, GEMM2, post-reorder) across GLM-5.2's two production
sharding layouts, at the shapes ``dispatch_w4a8_moe_mm_sm90``
(``python/sglang/kernels/aot/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu:88``)
is tuned for -- hidden=6144, moe_intermediate_size=2048, E=256, topk=8:

- **TP-8** (``mode="tp"``): all 256 experts present locally, the
  intermediate size partitioned by tp_size=8 -- gemm1 (n=512, k=6144) and
  gemm2 (n=6144, k=256).
- **EP-8 without DeepEP** (``mode="ep"``): 256/8=32 experts present locally
  at the full intermediate size -- gemm1 (n=4096, k=6144) and gemm2
  (n=6144, k=2048).

Each timed call is preceded by a 256MB L2 flush so small-M numbers aren't
optimistic from a warm cache the way production never sees. Shapes with
m<=2048 run captured under a CUDA graph (matching how decode-sized batches
run in production); larger m runs eager, since prefill of that size is not
graphed in production either.

Run on H200:

    cd sglang && python test/manual/layers/moe/bench_w4a8_sm90_dispatch.py
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts

GROUP_SIZE = 128
# Decode-sized batches run under a captured graph in production.
CUDA_GRAPH_MAX_TOKENS = 2048
L2_FLUSH_BYTES = 256 * 1024 * 1024


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    input_tensor_int8 = int4_values_interleaved.to(torch.int8)
    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]
    return ((high_nibbles << 4) | (low_nibbles & 0x0F)).to(torch.int8)


def pack_interleave(
    num_experts: int,
    ref_weight: torch.Tensor,
    ref_scale: torch.Tensor,
    alignment: int = 4,
):
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8).contiguous()

    scale_interleaved = ref_scale.reshape(
        ref_scale.shape[0],
        ref_scale.shape[1],
        ref_scale.shape[2] // alignment,
        alignment,
    )
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)
    w_scale = scale_interleaved.reshape(
        ref_scale.shape[0],
        ref_scale.shape[2] // alignment,
        ref_scale.shape[1] * alignment,
    ).contiguous()
    return w_q, w_scale


class Shape(msgspec.Struct, frozen=True):
    tokens: int
    hidden: int
    inter: int  # moe_intermediate_size, full (pre-partition)
    num_experts: int
    top_k: int
    parallel_size: int
    mode: str  # "tp" (tensor-parallel) or "ep" (expert-parallel, no DeepEP)

    @property
    def local_experts(self) -> int:
        if self.mode == "ep":
            return self.num_experts // self.parallel_size
        return self.num_experts

    @property
    def local_inter(self) -> int:
        """Per-local-expert intermediate size (N in gemm1/gemm2)."""
        if self.mode == "ep":
            return self.inter
        return self.inter // self.parallel_size

    @property
    def gemm1_nk(self) -> Tuple[int, int]:
        return (2 * self.local_inter, self.hidden)

    @property
    def gemm2_nk(self) -> Tuple[int, int]:
        return (self.hidden, self.local_inter)

    def label(self) -> str:
        n1, k1 = self.gemm1_nk
        n2, k2 = self.gemm2_nk
        return (
            f"{self.mode}{self.parallel_size} m={self.tokens:>5} h={self.hidden} "
            f"E={self.local_experts:>3} k={self.top_k} "
            f"gemm1(n={n1},k={k1}) gemm2(n={n2},k={k2})"
        )


_BODY = dict(hidden=6144, inter=2048, num_experts=256, top_k=8, parallel_size=8)
_TP_TOKENS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_EP_TOKENS = (8, 16, 32, 64, 128, 256)
DEFAULT_SHAPES: List[Shape] = [
    Shape(tokens=m, mode="tp", **_BODY) for m in _TP_TOKENS
] + [Shape(tokens=m, mode="ep", **_BODY) for m in _EP_TOKENS]


def _make_weights(shape: Shape, seed: int = 0) -> dict:
    """Builds the mode-keyed (not M-keyed) expert weights and stride tables."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    e = shape.local_experts
    n = shape.local_inter
    k = shape.hidden

    ref_w1 = torch.randint(
        -8, 8, (e, 2 * n, k), dtype=torch.int8, device=device, generator=g
    )
    ref_w2 = torch.randint(
        -8, 8, (e, k, n), dtype=torch.int8, device=device, generator=g
    )
    affine_coeff = 0.005
    scale_1 = (
        torch.randn(e, 2 * n, k // GROUP_SIZE, dtype=dtype, device=device, generator=g)
        * affine_coeff
    )
    scale_2 = (
        torch.randn(e, k, n // GROUP_SIZE, dtype=dtype, device=device, generator=g)
        * affine_coeff
    )

    # w1's alignment is fixed: k=6144 % 512 == 0 in both modes. w2's alignment
    # depends on the per-expert intermediate size: 256 (TP) % 512 != 0 -> 1,
    # 2048 (EP) % 512 == 0 -> 4.
    w1_q, w1_scale = pack_interleave(e, ref_w1, scale_1)
    w2_alignment = 4 if n % 512 == 0 else 1
    w2_q, w2_scale = pack_interleave(e, ref_w2, scale_2, alignment=w2_alignment)

    a_strides1 = torch.full((e, 3), k, device=device, dtype=torch.int64)
    c_strides1 = torch.full((e, 3), 2 * n, device=device, dtype=torch.int64)
    a_strides2 = torch.full((e, 3), n, device=device, dtype=torch.int64)
    c_strides2 = torch.full((e, 3), k, device=device, dtype=torch.int64)

    expert_offsets = torch.empty((e + 1,), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    return dict(
        w1_q=w1_q,
        w2_q=w2_q,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a_strides1=a_strides1,
        b_strides1=a_strides1,
        c_strides1=c_strides1,
        a_strides2=a_strides2,
        b_strides2=a_strides2,
        c_strides2=c_strides2,
        s_strides13=c_strides1,
        s_strides2=c_strides2,
        expert_offsets=expert_offsets,
        problem_sizes1=problem_sizes1,
        problem_sizes2=problem_sizes2,
    )


def _make_activations(shape: Shape, weights: dict, seed: int = 0) -> dict:
    """Builds the per-M activations and routing, then merges in the shared weights."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    k = shape.hidden

    a = torch.randn(shape.tokens, k, dtype=dtype, device=device, generator=g)
    a1_scale = torch.randn(1, dtype=torch.float32, device=device, generator=g)
    a2_scale = torch.randn(1, dtype=torch.float32, device=device, generator=g)

    score = torch.randn(
        (shape.tokens, shape.num_experts), dtype=dtype, device=device, generator=g
    )
    topk_weights, topk_ids, _ = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=shape.top_k, renormalize=False),
    )

    if shape.mode == "ep":
        # A real EP-8 rank sees only its 32 local experts; cutlass_w4a8_moe
        # only remaps non-local ids to the num_local_experts sentinel itself
        # when get_parallel().moe_ep_size > 1 (cutlass_w4a8_moe.py), and this
        # bench runs single-rank, so pre-apply the same global-id -> local-id
        # (or sentinel) mapping here.
        local_experts = shape.local_experts
        expert_map = torch.full(
            (shape.num_experts,), local_experts, dtype=torch.int32, device=device
        )
        expert_map[:local_experts] = torch.arange(
            local_experts, dtype=torch.int32, device=device
        )
        topk_ids = expert_map[topk_ids]

    return dict(
        a=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        **weights,
    )


def make_runner(inputs: dict) -> Callable[[], torch.Tensor]:
    def _call():
        return cutlass_w4a8_moe(
            inputs["a"],
            inputs["w1_q"],
            inputs["w2_q"],
            inputs["w1_scale"],
            inputs["w2_scale"],
            inputs["topk_weights"],
            inputs["topk_ids"],
            inputs["a_strides1"],
            inputs["b_strides1"],
            inputs["c_strides1"],
            inputs["a_strides2"],
            inputs["b_strides2"],
            inputs["c_strides2"],
            inputs["s_strides13"],
            inputs["s_strides2"],
            inputs["expert_offsets"],
            inputs["problem_sizes1"],
            inputs["problem_sizes2"],
            inputs["a1_scale"],
            inputs["a2_scale"],
            False,
        )

    return _call


_l2_flush_buffer: Optional[torch.Tensor] = None


def _l2_flush() -> None:
    """Zeroes a 256MB buffer so the next timed call can't hit warm L2 state."""
    global _l2_flush_buffer
    if _l2_flush_buffer is None:
        _l2_flush_buffer = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    _l2_flush_buffer.zero_()


def time_call(fn: Callable, warmup: int = 5, iters: int = 30) -> Tuple[float, float]:
    """Returns (median_us, min_us) across ``iters`` L2-flushed calls after ``warmup``."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        _l2_flush()
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))  # ms -> us
    return times[len(times) // 2], times[0]


def _capture_graph(call_fn: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call_fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call_fn()
    return g


def run_one_shape(shape: Shape, weights: dict):
    inputs = _make_activations(shape, weights, seed=0)
    call = make_runner(inputs)

    if shape.tokens <= CUDA_GRAPH_MAX_TOKENS:
        graph = _capture_graph(call)
        median_us, min_us = time_call(graph.replay)
        del graph
        torch.cuda.empty_cache()
        run_mode = "graph"
    else:
        median_us, min_us = time_call(call)
        run_mode = "eager"

    print(
        f"{shape.label()}  [{run_mode:>5}]  median={median_us:9.2f} us  min={min_us:9.2f} us"
    )


def _init_single_rank_runtime():
    """cutlass_w4a8_moe reads get_parallel().moe_ep_size -- needs a TP/EP
    group even at world_size=1."""
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    cap = torch.cuda.get_device_capability()
    if cap[0] != 9:
        print(f"WARNING: device cap {cap} is not SM90; cutlass_w4a8_moe_mm may fail.")

    _init_single_rank_runtime()

    print(f"Device: {torch.cuda.get_device_name()} (cap {cap[0]}.{cap[1]})")

    weights_by_mode: Dict[str, dict] = {}
    for shape in DEFAULT_SHAPES:
        if shape.mode not in weights_by_mode:
            weights_by_mode[shape.mode] = _make_weights(shape)
        run_one_shape(shape, weights_by_mode[shape.mode])


if __name__ == "__main__":
    main()
