"""GDN verify-only native TGV K-batch split, followed by FP32 reduction."""
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import _run_tgv


@triton.jit
def _sum_three(partial, output, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    # Preserve PDL chaining: the next HC consumer may start its independent
    # prologue, but neither this sum nor that consumer reads before its wait.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < ELEMENTS
    x0 = tl.load(partial + index, valid, other=0.0)
    x1 = tl.load(partial + ELEMENTS + index, valid, other=0.0)
    x2 = tl.load(partial + 2 * ELEMENTS + index, valid, other=0.0)
    total = (x0 + x1) + x2
    tl.store(output + index, total.to(tl.bfloat16), valid)


def gdn_outproj_batchsplit(x, weight):
    # Caller guards exactly M4 and BF16 [N2560,K6144], no bias/TP slicing.
    # Zero-copy views: each batch spans a distinct contiguous K segment
    # inside every original row, not a copied/padded weight matrix.
    x = x.detach()
    weight = weight.detach()
    xs = x.as_strided((3, 4, 2048), (2048, x.stride(0), 1))
    ws = weight.as_strided((3, 2560, 2048), (2048, weight.stride(0), 1))
    partial = torch.empty((3, 4, 2560), device=x.device, dtype=torch.float32)
    output = torch.empty((4, 2560), device=x.device, dtype=torch.bfloat16)
    # Same mature CTA64x16x128 / 2CTA / stage12 tactic18. L=3 supplies
    # 3 independent K slices:40CTAs each,120 total; no DSMEM splitK sum.
    _run_tgv(xs, ws.transpose(-2, -1), None, partial, True, 18)
    _sum_three[(triton.cdiv(4 * 2560, 256),)](
        partial, output, 4 * 2560, BLOCK=256,
        num_warps=4, enable_fp_fusion=False, launch_pdl=True,
    )
    return output
