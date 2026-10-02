"""Share expert weights across up to 16 tokens with production rounding.

Each token's top-k IDs must be distinct valid expert indices, as produced by
the serving router. All selected contributions retain their original precision.
"""

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)


@cache_once
def _module(slots):
    args = make_cpp_args(2688, 1024, 1024, 2688, 22, slots, False)
    return load_jit(
        "sm120_nvfp4_moe_mtp_union",
        *args,
        cuda_files=["gemm/sm120_nvfp4_moe_mtp_union.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Nvfp4MoeMtpUnion<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def nvfp4_moe_mtp_union(x, w1, s1, w2, s2, a1, a2, g1, g2, ids, routing, out=None):
    m = x.shape[0]
    module = _module(8 if m <= 8 else 16)
    if out is None:
        out = torch.empty((m, 1024), dtype=torch.bfloat16, device=x.device)
    z = torch.empty(m * 22 * 2688 // 8, dtype=torch.int32, device=x.device)
    zs = torch.empty(m * 22 * 2688 // 16, dtype=torch.uint8, device=x.device)
    partial = torch.empty(m * 22 * 1024, dtype=torch.bfloat16, device=x.device)
    xq = torch.empty(m * 1024 // 8, dtype=torch.int32, device=x.device)
    xs = torch.empty(m * 1024 // 16, dtype=torch.uint8, device=x.device)
    module.run(
        x,
        w1.view(torch.uint8).view(-1),
        s1.view(torch.uint8).view(-1),
        w2.view(torch.uint8).view(-1),
        s2.view(torch.uint8).view(-1),
        a1.view(-1),
        a2.view(-1),
        g1.view(-1),
        g2.view(-1),
        ids.to(torch.int32).view(-1),
        routing.float().view(-1),
        z,
        zs,
        partial,
        out,
        xq,
        xs,
    )
    return out
