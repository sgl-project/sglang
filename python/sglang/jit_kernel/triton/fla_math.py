"""Value-only Triton math shared by standalone FLA and fused kernels.

Callers own pointer arithmetic, masks, tile shapes, and stores. Keeping the
numerical formulas here prevents fused kernels from becoming a second source
of truth for numerically load-bearing operations.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def l2norm_row_values(b_x, eps):
    """Normalize one row-block after the caller's fp32 load conversion."""
    b_var = tl.sum(b_x * b_x, axis=1)
    return b_x / tl.sqrt(b_var + eps)[:, None]


@triton.jit
def gdn_gating_values(
    blk_A_log,
    blk_a,
    blk_b,
    blk_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    EXP_GATE: tl.constexpr,
):
    """Compute GDN gate and dtype-rounded beta from already-loaded values."""
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    if EXP_GATE:
        # libdevice.exp is bit-identical to torch.exp (expf); tl.exp's fast
        # exp2 path is not, and this value feeds the SSM decay directly.
        blk_g = libdevice.exp(blk_g)

    # Preserve fused_gdn_gating's unusual bf16-before-fp32-store behavior.
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32)).to(blk_b.dtype)
    return blk_g, blk_beta_output
