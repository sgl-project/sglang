"""Benchmark: MoE LoRA-A shrink JIT (moe_lora_shrink) vs the Triton split-K kernel.

Mirrors the qwen3.5-35b local-EP gate_up shrink shape (num_experts=64, hidden=2048,
rank=16, top_k=8) and sweeps batch size.

Run:
    python python/sglang/jit_kernel/benchmark/bench_moe_lora_shrink.py
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.moe_lora_shrink import moe_lora_shrink
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-benchmark-1-gpu-large")

try:
    import triton

    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.lora.triton_ops.virtual_experts import (
        _fused_virtual_topk_ids,
        _invoke_moe_lora_shrink_splitk,
        fused_sanitize_expert_ids,
    )

    HAS_TRITON_REF = True
except Exception:
    HAS_TRITON_REF = False

NUM_EXPERTS = 64
TOP_K = 8
HIDDEN = 2048
RANK = 16
BLOCK_M = 16
DTYPE = torch.bfloat16

_TRITON_CONFIG = {
    "BLOCK_SIZE_M": BLOCK_M,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 4,
}


def _build_routing(topk_ids, tlm):
    virtual_topk_ids, _, virtual_num_experts = _fused_virtual_topk_ids(
        topk_ids, tlm, NUM_EXPERTS, shared_outer=False, max_loras=1
    )
    sorted_token_ids, expert_ids, npp = moe_align_block_size(
        virtual_topk_ids, BLOCK_M, virtual_num_experts
    )
    num_tokens = topk_ids.numel()
    max_nonempty = min(num_tokens, virtual_num_experts)
    tight = triton.cdiv(num_tokens + max_nonempty * (BLOCK_M - 1), BLOCK_M) * BLOCK_M
    return (
        sorted_token_ids[:tight].contiguous(),
        fused_sanitize_expert_ids(expert_ids[: tight // BLOCK_M], virtual_num_experts),
        npp,
    )


def _jit_call(output, hidden_states, lora_a, sti, eid, npp):
    moe_lora_shrink(output, hidden_states, lora_a, sti, eid, npp, TOP_K, BLOCK_M)


def _triton_call(intermediate, hidden_states, lora_a, topk_ids, sti, eid, npp):
    _invoke_moe_lora_shrink_splitk(
        hidden_states,
        lora_a,
        intermediate,
        topk_ids,
        sti,
        eid,
        npp,
        TOP_K,
        _TRITON_CONFIG,
    )


@marker.parametrize("bs", [1, 8, 16, 32, 64, 128], [16])
@marker.benchmark("impl", ["jit", "triton"])
def benchmark(bs: int, impl: str):
    if not HAS_TRITON_REF:
        raise RuntimeError("sglang triton ops required to build routing buffers")
    device = "cuda"
    torch.manual_seed(0)
    topk_ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, device=device)[:TOP_K] for _ in range(bs)]
    ).to(torch.int32)
    tlm = torch.zeros(bs, device=device, dtype=torch.int32)
    hidden_states = torch.randn(bs, HIDDEN, device=device, dtype=DTYPE) * 0.1
    lora_a = torch.randn(NUM_EXPERTS, RANK, HIDDEN, device=device, dtype=DTYPE) * 0.1
    sti, eid, npp = _build_routing(topk_ids, tlm)
    output = torch.empty(bs * TOP_K, RANK, device=device, dtype=DTYPE)

    if impl == "jit":
        return marker.do_bench(
            _jit_call,
            input_args=(output, hidden_states, lora_a, sti, eid, npp),
            graph_clone_args=(1, 2, 3, 4, 5),  # read tensors; output (0) is written
            memory_output=(output,),
        )
    else:
        return marker.do_bench(
            _triton_call,
            input_args=(output, hidden_states, lora_a, topk_ids, sti, eid, npp),
            graph_clone_args=(1, 2, 3, 4, 5, 6),  # output (0) is written
            memory_output=(output,),
        )


if __name__ == "__main__":
    benchmark.run()
