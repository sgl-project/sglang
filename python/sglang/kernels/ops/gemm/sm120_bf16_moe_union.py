"""Small-M BF16 MoE with original weights and all selected expert contributions."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        "sm120_bf16_moe_union",
        cuda_files=["gemm/sm120_bf16_moe_union.cuh"],
        cuda_wrappers=[("run", "sglang::Bf16MoeUnionRaw<2>::run")],
        extra_cuda_cflags=["-O3"],
    )


def bf16_moe_union(x, w1, w2, ids, weights, *, out=None):
    routes = x.shape[0] * 22
    route_map = torch.empty((routes, 5), dtype=torch.int32, device=x.device)
    z = torch.empty((routes, 2688), dtype=x.dtype, device=x.device)
    partial = torch.empty((routes, 1024), dtype=x.dtype, device=x.device)
    if out is None:
        out = torch.empty_like(x)
    _module().run(x, w1, w2, ids, weights, route_map, z, partial, out)
    return out
