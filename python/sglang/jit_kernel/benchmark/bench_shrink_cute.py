"""Compare the CuTeDSL shrink vs the CUDA JIT pipe kernel vs Triton (rank16 K2048)."""

import torch
import triton

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.moe_lora_shrink import moe_lora_shrink as moe_lora_shrink_cuda
from sglang.jit_kernel.moe_lora_shrink_cute import moe_lora_shrink_cute
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import (
    _fused_virtual_topk_ids,
    _invoke_moe_lora_shrink_splitk,
    fused_sanitize_expert_ids,
)

E, TK, H, R, BM, DT = 64, 8, 2048, 16, 16, torch.bfloat16
TCFG = {
    "BLOCK_SIZE_M": BM,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 4,
}


def _mk(bs):
    torch.manual_seed(0)
    topk = torch.stack([torch.randperm(E, device="cuda")[:TK] for _ in range(bs)]).to(
        torch.int32
    )
    tlm = torch.zeros(bs, device="cuda", dtype=torch.int32)
    hs = torch.randn(bs, H, device="cuda", dtype=DT) * 0.1
    la = torch.randn(E, R, H, device="cuda", dtype=DT) * 0.1
    vt, _, vne = _fused_virtual_topk_ids(topk, tlm, E, False, 1)
    sti, eid, npp = moe_align_block_size(vt, BM, vne)
    nt = topk.numel()
    tight = triton.cdiv(nt + min(nt, vne) * (BM - 1), BM) * BM
    sti = sti[:tight].contiguous()
    eid = fused_sanitize_expert_ids(eid[: tight // BM], vne)
    return topk, hs, la, sti, eid, npp


@marker.parametrize("bs", [1, 8, 16, 32, 64, 128], [16])
@marker.benchmark("impl", ["cute", "cuda", "triton"])
def benchmark(bs: int, impl: str):
    topk, hs, la, sti, eid, npp = _mk(bs)
    out = torch.empty(bs * TK, R, device="cuda", dtype=DT)
    if impl == "cute":
        return marker.do_bench(
            lambda *a: moe_lora_shrink_cute(*a, npp, TK, BM),
            input_args=(out, hs, la, sti, eid),
            graph_clone_args=(1, 2, 3, 4),
            memory_output=(out,),
        )
    if impl == "cuda":
        return marker.do_bench(
            lambda *a: moe_lora_shrink_cuda(*a, npp, TK, BM),
            input_args=(out, hs, la, sti, eid),
            graph_clone_args=(1, 2, 3, 4),
            memory_output=(out,),
        )
    return marker.do_bench(
        lambda *a: _invoke_moe_lora_shrink_splitk(*a, npp, TK, TCFG),
        input_args=(hs, la, out, topk, sti, eid),
        graph_clone_args=(0, 1, 3, 4, 5),
        memory_output=(out,),
    )


if __name__ == "__main__":
    benchmark.run()
