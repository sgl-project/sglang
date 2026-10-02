"""Native SM120 NVFP4 MoE for 1 to 16 tokens with production rounding."""

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)


@cache_once
def _module():
    args = make_cpp_args(2688, 1024, 1024, 2688, 22, False)
    return load_jit(
        "sm120_nvfp4_moe_mtp_mma",
        *args,
        cuda_files=["gemm/sm120_nvfp4_moe_mtp_mma.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Nvfp4MoeMtpMma<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def nvfp4_moe_mtp_mma(x, w1, s1, w2, s2, a1, a2, g1, g2, ids, routing, out=None):
    m = x.shape[0]
    if m >= 8:
        from sglang.kernels.ops.gemm.sm120_nvfp4_moe_mtp_union import (
            nvfp4_moe_mtp_union,
        )

        # Reuse an expert's weights across token rows once the larger batch
        # amortizes routing lookup and input-quantization launch overhead.
        return nvfp4_moe_mtp_union(
            x, w1, s1, w2, s2, a1, a2, g1, g2, ids, routing, out=out
        )
    module = _module()
    if out is None:
        out = torch.empty((m, 1024), dtype=torch.bfloat16, device=x.device)
    z = torch.empty(m * 22 * 2688 // 8, dtype=torch.int32, device=x.device)
    zs = torch.empty(m * 22 * 2688 // 16, dtype=torch.uint8, device=x.device)
    partial = torch.empty(m * 22 * 1024, dtype=torch.bfloat16, device=x.device)
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
    )
    return out
